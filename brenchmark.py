"""
brenchmark.py — Publication-grade benchmark framework

Design: Checkpoint mode + Multi-run statistics
  - Each run executes ONE full simulation (e.g. 300 tasks)
  - Metrics are recorded at checkpoints (every N completed tasks)
  - The same simulation is repeated M times with different random seeds
  - Statistics (mean, std, 95% CI) are computed across runs at each checkpoint
  - Mann-Whitney U tests compare schedulers at each checkpoint

Output files:
  figs/benchmark_raw.csv        — every checkpoint of every run
  figs/benchmark_summary.csv    — mean / std / 95% CI per checkpoint
  figs/statistical_tests.csv    — pairwise Mann-Whitney U test results
"""

import csv
import os
import sys
import time
import random
import argparse
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from scipy import stats as sp_stats

from scheduler.RLscheduler import RLScheduler
from scheduler.GAscheduler import GAScheduler
from scheduler.Heftscheduler import HEFTScheduler
from scheduler.PSOscheduler import PSOScheduler
from scheduler.RRscheduler import RoundRobinScheduler
from environment.simulation import Simulation
from environment.task import Task, TaskStatus


# ====================================================================
#  BenchmarkTester
# ====================================================================

class BenchmarkTester:

    def __init__(self, num_runs: int = 20, base_seed: int = 42):
        self.num_runs = num_runs
        self.base_seed = base_seed

        self.schedulers = {
            "RoundRobin": RoundRobinScheduler,
            "HEFT":       HEFTScheduler,
            "GA":         GAScheduler,
            "PSO":        PSOScheduler,
            "RL":         RLScheduler,
        }

    # ----------------------------------------------------------------
    #  Environment setup
    # ----------------------------------------------------------------

    def create_simulation(self, task_count: int, num_edge_servers: int,
                          seed: int) -> Simulation:
        random.seed(seed)
        np.random.seed(seed)

        sim = Simulation(num_servers=num_edge_servers)

        count1 = task_count // 3
        count2 = task_count // 3
        count3 = task_count - count1 - count2

        all_tasks = []
        all_tasks.extend(Task.generate_single_dag(0, count1))
        all_tasks.extend(Task.generate_linear_dag(count1, count2))
        all_tasks.extend(Task.generate_fork_join_dag(count1 + count2, count3))

        random.shuffle(all_tasks)
        sim.add_tasks(all_tasks)
        return sim

    # ----------------------------------------------------------------
    #  Single run with checkpoints
    # ----------------------------------------------------------------

    def run_single_with_checkpoints(self, scheduler_class,
                                    total_tasks: int, num_edge_servers: int,
                                    seed: int, checkpoint_interval: int = 20,
                                    max_time: float = 10000) -> list[dict]:
        """
        Run one full simulation, return a list of metric dicts
        (one per checkpoint).
        """
        random.seed(seed)
        np.random.seed(seed)
        if HAS_TORCH:
            torch.manual_seed(seed)

        sim = self.create_simulation(total_tasks, num_edge_servers, seed)

        if scheduler_class is RLScheduler:
            scheduler = RLScheduler(sim, pretrain_episodes=5)
        else:
            scheduler = scheduler_class(sim)

        # Build milestone list: 20, 40, ..., total_tasks
        milestones = list(range(checkpoint_interval,
                                total_tasks + 1,
                                checkpoint_interval))
        if not milestones or milestones[-1] != total_tasks:
            milestones.append(total_tasks)

        checkpoints = []
        milestone_idx = 0
        current_time = 0.0

        while (len(sim.completed_tasks) < len(sim.tasks)
               and current_time < max_time):

            sim.step(scheduler, current_time)

            # Check if we crossed one or more milestones this step
            while (milestone_idx < len(milestones)
                   and len(sim.completed_tasks) >= milestones[milestone_idx]):
                m = self._collect_metrics(sim, current_time)
                m["completed_tasks"] = milestones[milestone_idx]
                checkpoints.append(m)
                milestone_idx += 1

            current_time += 0.1

        # If simulation timed out, fill remaining milestones
        while milestone_idx < len(milestones):
            m = self._collect_metrics(sim, current_time)
            m["completed_tasks"] = len(sim.completed_tasks)
            checkpoints.append(m)
            milestone_idx += 1

        return checkpoints

    # ----------------------------------------------------------------
    #  Metric collection (fixed utilization)
    # ----------------------------------------------------------------

    def _collect_metrics(self, sim: Simulation, current_time: float) -> dict:
        completed = [t for t in sim.tasks.values()
                     if t.task_id in sim.completed_tasks]

        # 1) Makespan
        makespan = round(current_time, 2)

        # 2) Avg end-to-end latency (READY -> COMPLETED)
        e2e = [t.end_time - t.ready_time
               for t in completed
               if t.ready_time is not None and t.end_time is not None]
        avg_e2e = float(np.mean(e2e)) if e2e else 0.0

        # 3) Resource utilization (weighted, bounded [0, 1])
        #    util_s = sum(compute_demand * duration) / (capacity * makespan)
        utils = []
        for server in sim.servers.values():
            resource_seconds = sum(
                t.compute_demand * (t.end_time - t.start_time)
                for t in server.task_history
                if (t.task_id in sim.completed_tasks
                    and t.start_time is not None
                    and t.end_time is not None)
            )
            denom = server.total_compute * current_time
            util = resource_seconds / denom if denom > 0 else 0.0
            utils.append(util)

        avg_util = float(np.mean(utils)) if utils else 0.0
        load_std = float(np.std(utils)) if utils else 0.0

        # 4) Completion ratio
        completed_ratio = len(sim.completed_tasks) / max(len(sim.tasks), 1)

        return {
            "makespan":         makespan,
            "avg_e2e_latency":  round(avg_e2e, 4),
            "avg_utilization":  round(avg_util, 4),
            "load_balance_std": round(load_std, 4),
            "completed_ratio":  round(completed_ratio, 4),
        }

    # ================================================================
    #  Main benchmark entry
    # ================================================================

    def run_benchmark(self, edge_counts: list, total_tasks: int = 300,
                      checkpoint_interval: int = 20,
                      output_dir: str = "figs"):
        os.makedirs(output_dir, exist_ok=True)

        raw_results: list[dict] = []
        total_runs = len(edge_counts) * len(self.schedulers) * self.num_runs
        progress = 0
        wall_start = time.time()

        for edge_count in edge_counts:
            print(f"\n{'='*60}")
            print(f"  Edge servers: {edge_count}  |  "
                  f"Tasks: {total_tasks}  |  "
                  f"Checkpoint every {checkpoint_interval}")
            print(f"{'='*60}")

            for sched_name, sched_class in self.schedulers.items():
                final_makespans = []

                for run_idx in range(self.num_runs):
                    seed = self.base_seed + run_idx

                    t0 = time.time()
                    checkpoints = self.run_single_with_checkpoints(
                        sched_class, total_tasks, edge_count, seed,
                        checkpoint_interval)
                    elapsed = time.time() - t0

                    # Store every checkpoint as a raw row
                    for cp in checkpoints:
                        raw_results.append({
                            "edge_servers":    edge_count,
                            "completed_tasks": cp["completed_tasks"],
                            "scheduler":       sched_name,
                            "run":             run_idx,
                            "seed":            seed,
                            **{k: v for k, v in cp.items()
                               if k != "completed_tasks"},
                        })

                    if checkpoints:
                        final_makespans.append(checkpoints[-1]["makespan"])

                    progress += 1
                    pct = progress / total_runs * 100
                    eta = ((time.time() - wall_start) / progress
                           * (total_runs - progress))
                    sys.stdout.write(
                        f"\r    [{progress}/{total_runs}] {pct:5.1f}%  "
                        f"{sched_name:<12s} run {run_idx+1:>2d}/{self.num_runs}  "
                        f"({elapsed:5.1f}s)  ETA {eta/60:5.1f}min   ")
                    sys.stdout.flush()

                m = np.mean(final_makespans) if final_makespans else 0
                s = (np.std(final_makespans, ddof=1)
                     if len(final_makespans) > 1 else 0)
                print(f"\n    {sched_name:<12s}: "
                      f"makespan = {m:7.1f} +/- {s:5.1f}")

        wall_total = time.time() - wall_start
        print(f"\n\n{'='*60}")
        print(f"  Wall time: {wall_total/60:.1f} min  |  "
              f"Total runs: {total_runs}")
        print(f"{'='*60}")

        # ---------- Export ----------
        raw_file = os.path.join(output_dir, "benchmark_raw.csv")
        summary_file = os.path.join(output_dir, "benchmark_summary.csv")
        stat_file = os.path.join(output_dir, "statistical_tests.csv")

        self._export_raw(raw_results, raw_file)
        summary = self._compute_summary(raw_results)
        self._export_summary(summary, summary_file)
        self._run_statistical_tests(raw_results, stat_file)

        return raw_results, summary

    # ================================================================
    #  Summary statistics
    # ================================================================

    def _compute_summary(self, raw_results: list) -> list:
        """Group by (edge, completed_tasks, scheduler), compute stats."""
        import pandas as pd
        df = pd.DataFrame(raw_results)

        metrics = ["makespan", "avg_e2e_latency",
                    "avg_utilization", "load_balance_std"]
        summary = []

        groups = df.groupby(["edge_servers", "completed_tasks", "scheduler"])
        for (edge, tasks, sched), grp in groups:
            for metric in metrics:
                vals = grp[metric].values
                n = len(vals)
                mean = float(np.mean(vals))
                std  = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                se   = std / np.sqrt(n) if n > 1 else 0.0
                ci   = 1.96 * se

                summary.append({
                    "edge_servers":    edge,
                    "completed_tasks": tasks,
                    "scheduler":       sched,
                    "metric":          metric,
                    "mean":            round(mean, 4),
                    "std":             round(std, 4),
                    "ci_lower":        round(mean - ci, 4),
                    "ci_upper":        round(mean + ci, 4),
                    "min":             round(float(np.min(vals)), 4),
                    "max":             round(float(np.max(vals)), 4),
                    "median":          round(float(np.median(vals)), 4),
                    "n":               n,
                })

        return summary

    # ================================================================
    #  Statistical significance tests
    # ================================================================

    def _run_statistical_tests(self, raw_results: list,
                               output_file: str):
        """Pairwise Mann-Whitney U tests at each checkpoint."""
        import pandas as pd
        df = pd.DataFrame(raw_results)

        metrics = ["makespan", "avg_e2e_latency",
                    "avg_utilization", "load_balance_std"]
        scheds = sorted(df["scheduler"].unique())
        results = []

        for (edge, tasks), cfg in df.groupby(
                ["edge_servers", "completed_tasks"]):
            for metric in metrics:
                for i, sa in enumerate(scheds):
                    for sb in scheds[i + 1:]:
                        va = cfg.loc[cfg["scheduler"] == sa, metric].values
                        vb = cfg.loc[cfg["scheduler"] == sb, metric].values
                        if len(va) < 2 or len(vb) < 2:
                            continue
                        try:
                            stat, p = sp_stats.mannwhitneyu(
                                va, vb, alternative="two-sided")
                        except ValueError:
                            stat, p = 0.0, 1.0

                        sig = ""
                        if p < 0.001:
                            sig = "***"
                        elif p < 0.01:
                            sig = "**"
                        elif p < 0.05:
                            sig = "*"

                        results.append({
                            "edge_servers":    edge,
                            "completed_tasks": tasks,
                            "metric":          metric,
                            "scheduler_A":     sa,
                            "scheduler_B":     sb,
                            "mean_A":          round(float(np.mean(va)), 4),
                            "mean_B":          round(float(np.mean(vb)), 4),
                            "U_statistic":     round(float(stat), 2),
                            "p_value":         round(float(p), 6),
                            "significant":     sig,
                        })

        if results:
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"  Stat tests  -> {output_file} ({len(results)} rows)")
        else:
            print("  No statistical test results")

    # ================================================================
    #  CSV export
    # ================================================================

    def _export_raw(self, results: list, output_file: str):
        if not results:
            return
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"  Raw data    -> {output_file} ({len(results)} rows)")

    def _export_summary(self, summary: list, output_file: str):
        if not summary:
            return
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
        print(f"  Summary     -> {output_file} ({len(summary)} rows)")


# ====================================================================
#  CLI entry
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cloud-Edge Scheduling Benchmark")
    parser.add_argument(
        "--runs", type=int, default=20,
        help="Number of runs per configuration (default: 20)")
    parser.add_argument(
        "--tasks", type=int, default=300,
        help="Total task count per run (default: 300)")
    parser.add_argument(
        "--interval", type=int, default=20,
        help="Checkpoint interval (default: every 20 tasks)")
    parser.add_argument(
        "--edge", type=int, nargs="+", default=None,
        help="Edge server counts, e.g. --edge 3 5 7")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 5 runs, edge=[3,7]")
    args = parser.parse_args()

    if args.quick:
        num_runs    = 5
        edge_counts = [3, 7]
    else:
        num_runs    = args.runs
        edge_counts = args.edge if args.edge else [3, 5, 7]

    total_tasks = args.tasks
    interval    = args.interval
    n_checkpoints = total_tasks // interval
    total = len(edge_counts) * 5 * num_runs

    print("=" * 70)
    print("  Cloud-Edge Scheduling Benchmark (Publication-grade)")
    print("=" * 70)
    print(f"  Runs / config  : {num_runs}")
    print(f"  Edge servers   : {edge_counts}")
    print(f"  Total tasks    : {total_tasks}")
    print(f"  Checkpoint     : every {interval} tasks "
          f"({n_checkpoints} data points / run)")
    print(f"  Schedulers     : RoundRobin, HEFT, GA, PSO, RL")
    print(f"  Total sim runs : {total}")
    print("=" * 70)

    tester = BenchmarkTester(num_runs=num_runs)
    raw, summary = tester.run_benchmark(
        edge_counts, total_tasks=total_tasks,
        checkpoint_interval=interval)

    print("\n" + "=" * 70)
    print("  Done! Output files in figs/")
    print("=" * 70)
