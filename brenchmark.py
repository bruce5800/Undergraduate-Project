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
import json
import os
import subprocess
import sys
import time
import random
import argparse
from datetime import datetime
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
from scheduler.LeastLoadedScheduler import LeastLoadedScheduler
from scheduler.ShortestQueueScheduler import ShortestQueueScheduler
from scheduler.A3CR2NScheduler import A3CR2NScheduler
from environment.simulation import Simulation
from environment.task import Task, TaskStatus, TaskKind
from environment.model_catalog import assign_models_zipf


# ====================================================================
#  指标定义
# ====================================================================
# 两类 workload 共用的基础指标
COMMON_METRICS = ["makespan", "avg_e2e_latency",
                  "avg_utilization", "load_balance_std"]

# inference workload 额外追加的 AIGC 标准指标
AIGC_METRICS = [
    "ttft_p50", "ttft_p95", "ttft_p99",
    "tpot_p50", "tpot_p95",
    "goodput_tps",
    "slo_attainment",
    "req_e2e_p50", "req_e2e_p95",
]


# ====================================================================
#  BenchmarkTester
# ====================================================================

class BenchmarkTester:

    def __init__(self, num_runs: int = 20, base_seed: int = 42,
                 aigc_mode: bool = False, aigc_zipf_alpha: float = 1.2,
                 workload: str = "dag",
                 ablation: str = "none",
                 ttft_slo: float = 2.0,
                 tpot_slo: float = 0.1,
                 trace_preset: str = "uniform",
                 arrival_rate: float = None):
        self.num_runs = num_runs
        self.base_seed = base_seed
        # M1: AIGC 模式 —— 给任务按 Zipf 分配模型 ID，触发冷加载与权重驻留物理
        self.aigc_mode = aigc_mode
        self.aigc_zipf_alpha = aigc_zipf_alpha
        # M2: workload 类型 —— "dag" (通用三种 DAG) 或 "inference" (prefill/decode 推理请求)
        self.workload = workload
        # M3 step3: 消融名称
        self.ablation = ablation
        # M4 step1: AIGC SLO 阈值 —— 用于 slo_attainment 指标
        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        # M4 step2: trace 配置 —— prompt/output 分布 + Poisson 到达率
        self.trace_preset = trace_preset
        self.arrival_rate = arrival_rate

        self.schedulers = {
            "RoundRobin":    RoundRobinScheduler,
            "LeastLoaded":   LeastLoadedScheduler,
            "ShortestQueue": ShortestQueueScheduler,
            "HEFT":          HEFTScheduler,
            "GA":            GAScheduler,
            "PSO":           PSOScheduler,
            "A3C_R2N2":      A3CR2NScheduler,
            "RL":            RLScheduler,
        }

    # ----------------------------------------------------------------
    #  Ablation 配置：哪个 ablation 关闭哪个开关
    # ----------------------------------------------------------------
    ABLATION_KWARGS = {
        "none":               {},
        # 物理层（Simulation/Server）
        "no_batching":        {"sim": {"enable_batching": False}},
        # RL scheduler 层 —— 单项
        "no_warm_reward":     {"rl": {"enable_warm_reward": False}},
        "no_batch_reward":    {"rl": {"enable_batch_reward": False}},
        "no_affinity_reward": {"rl": {"enable_affinity_reward": False}},
        "no_aigc_state":      {"rl": {"enable_aigc_state": False}},
        "no_action_mask":     {"rl": {"enable_action_mask": False}},
        "no_gae":             {"rl": {"enable_gae": False}},
        "no_pretrain":        {"rl": {"enable_pretrain": False}},
        "no_entropy":         {"rl": {"enable_entropy": False}},
        # 组合：所有 AIGC 行为奖励一起关
        "no_all_aigc_rewards": {"rl": {
            "enable_warm_reward":     False,
            "enable_batch_reward":    False,
            "enable_affinity_reward": False,
        }},
        # 组合：所有 AIGC 设计（state + reward）一起关 —— 接近 A3C-R2N2
        "no_aigc_full":       {"rl": {
            "enable_warm_reward":     False,
            "enable_batch_reward":    False,
            "enable_affinity_reward": False,
            "enable_aigc_state":      False,
        }},
    }

    def _sim_kwargs(self) -> dict:
        return self.ABLATION_KWARGS.get(self.ablation, {}).get("sim", {})

    def _rl_kwargs(self) -> dict:
        return self.ABLATION_KWARGS.get(self.ablation, {}).get("rl", {})

    def _metrics_list(self) -> list:
        """该 workload 下需要汇总/做显著性检验的指标列。"""
        if self.workload == "inference":
            return COMMON_METRICS + AIGC_METRICS
        return list(COMMON_METRICS)

    # ----------------------------------------------------------------
    #  Environment setup
    # ----------------------------------------------------------------

    def create_simulation(self, task_count: int, num_edge_servers: int,
                          seed: int) -> Simulation:
        random.seed(seed)
        np.random.seed(seed)

        sim = Simulation(num_servers=num_edge_servers, **self._sim_kwargs())

        if self.workload == "inference":
            # M2: LLM 推理负载 —— task_count 个 task 单位 = task_count/2 个推理请求
            num_requests = max(task_count // 2, 1)
            rng = random.Random(seed)
            all_tasks = Task.generate_inference_workload(
                num_requests=num_requests,
                task_id_offset=0,
                rng=rng,
                # M4 step2: 选择 prompt/output 分布与 Poisson 到达率
                dist=self.trace_preset,
                arrival_rate=self.arrival_rate,
            )
            # 不 shuffle —— 依赖关系由 task.dependencies 保证，保留请求级顺序
            # 便于调试观察。
        else:
            # 默认：3 种通用 DAG 混合
            count1 = task_count // 3
            count2 = task_count // 3
            count3 = task_count - count1 - count2

            all_tasks = []
            all_tasks.extend(Task.generate_single_dag(0, count1))
            all_tasks.extend(Task.generate_linear_dag(count1, count2))
            all_tasks.extend(Task.generate_fork_join_dag(count1 + count2, count3))

            random.shuffle(all_tasks)

            # M1: AIGC 模式下给所有任务分配模型 ID
            if self.aigc_mode:
                rng = random.Random(seed)  # 独立 RNG，不污染上面的全局 random
                assign_models_zipf(all_tasks, rng, alpha=self.aigc_zipf_alpha)

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
            scheduler = RLScheduler(sim, pretrain_episodes=5,
                                     **self._rl_kwargs())
        elif scheduler_class is A3CR2NScheduler:
            scheduler = A3CR2NScheduler(sim, pretrain_episodes=5)
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

        out = {
            "makespan":         makespan,
            "avg_e2e_latency":  round(avg_e2e, 4),
            "avg_utilization":  round(avg_util, 4),
            "load_balance_std": round(load_std, 4),
            "completed_ratio":  round(completed_ratio, 4),
        }

        # M4 step1: inference workload 追加 AIGC 标准指标
        if self.workload == "inference":
            out.update(self._collect_aigc_request_metrics(sim, current_time))

        return out

    # ----------------------------------------------------------------
    #  AIGC 请求级指标（M4 step1）
    # ----------------------------------------------------------------

    def _collect_aigc_request_metrics(self, sim: Simulation,
                                       current_time: float) -> dict:
        """按 req_id 聚合 prefill/decode 计算 LLM 推理标准指标。

        TTFT  : prefill.end_time - prefill.ready_time
        TPOT  : (decode.end_time - decode.start_time) / output_tokens
        E2E   : decode.end_time - prefill.ready_time
        Good- : Σ output_tokens / current_time
        SLO   : 同时满足 TTFT <= slo_ttft AND TPOT <= slo_tpot 的请求占比
        """
        # 按 req_id 聚合 completed 的 prefill / decode
        prefills_by_req = {}
        decodes_by_req = {}
        total_output_tokens = 0

        for tid in sim.completed_tasks:
            t = sim.tasks.get(tid)
            if t is None or t.req_id is None:
                continue
            if t.kind == TaskKind.PREFILL:
                prefills_by_req[t.req_id] = t
            elif t.kind == TaskKind.DECODE:
                decodes_by_req[t.req_id] = t
                total_output_tokens += getattr(t, "output_tokens", 0)

        # 只统计 prefill+decode 都完成的请求
        full_reqs = set(prefills_by_req.keys()) & set(decodes_by_req.keys())

        ttfts, tpots, e2es = [], [], []
        slo_hits = 0
        for rid in full_reqs:
            p = prefills_by_req[rid]
            d = decodes_by_req[rid]
            if p.ready_time is None or p.end_time is None:
                continue
            if d.start_time is None or d.end_time is None:
                continue

            ttft = p.end_time - p.ready_time
            e2e = d.end_time - p.ready_time
            out_tok = max(getattr(d, "output_tokens", 0), 1)
            tpot = (d.end_time - d.start_time) / out_tok

            ttfts.append(ttft)
            tpots.append(tpot)
            e2es.append(e2e)
            if ttft <= self.ttft_slo and tpot <= self.tpot_slo:
                slo_hits += 1

        def _p(arr, q):
            return float(np.percentile(arr, q)) if arr else 0.0

        n = len(full_reqs)
        return {
            "ttft_p50":      round(_p(ttfts, 50), 4),
            "ttft_p95":      round(_p(ttfts, 95), 4),
            "ttft_p99":      round(_p(ttfts, 99), 4),
            "tpot_p50":      round(_p(tpots, 50), 4),
            "tpot_p95":      round(_p(tpots, 95), 4),
            "req_e2e_p50":   round(_p(e2es, 50), 4),
            "req_e2e_p95":   round(_p(e2es, 95), 4),
            "goodput_tps":   round(total_output_tokens / max(current_time, 1e-6), 2),
            "slo_attainment": round(slo_hits / max(n, 1), 4),
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

        # M4 step1: 指标列表随 workload 扩展
        metrics = self._metrics_list()
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

        # M4 step1: 指标列表随 workload 扩展
        metrics = self._metrics_list()
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
    parser.add_argument(
        "--workload", type=str, choices=["dag", "inference"], default="dag",
        help="Workload type: 'dag' (3 generic DAG patterns, default) or "
             "'inference' (M2: LLM prefill/decode requests). "
             "In inference mode --tasks N produces N/2 requests (= N tasks).")
    parser.add_argument(
        "--aigc", action="store_true",
        help="AIGC mode (dag workload only): assign models to tasks via Zipf. "
             "Ignored in inference workload (always model-assigned).")
    parser.add_argument(
        "--aigc-alpha", type=float, default=1.2,
        help="Zipf alpha for model popularity (default 1.2)")
    parser.add_argument(
        "--ablation", type=str, default="none",
        choices=list(BenchmarkTester.ABLATION_KWARGS.keys()),
        help="M3 step3 ablation: disable one component "
             "(no_batching / no_warm_reward / no_batch_reward / "
             "no_affinity_reward / no_aigc_state / no_action_mask / "
             "no_gae / no_pretrain / no_entropy). 'none' = full model.")
    parser.add_argument(
        "--rl-only", action="store_true",
        help="Run only the RL scheduler (skip other 7 baselines). "
             "Useful for fast ablation studies focused on RL component analysis.")
    parser.add_argument(
        "--ttft-slo", type=float, default=2.0,
        help="TTFT SLO threshold in seconds (inference workload only). Default 2.0s.")
    parser.add_argument(
        "--tpot-slo", type=float, default=0.1,
        help="TPOT SLO threshold in seconds/token (inference workload only). "
             "Default 0.1s (=10 tok/s).")
    parser.add_argument(
        "--trace-preset", type=str, default="uniform",
        choices=["uniform", "lognormal"],
        help="Prompt/output length distribution for inference workload. "
             "'uniform' = uniform random over fixed ranges (default); "
             "'lognormal' = Azure-LLM-trace-like long-tailed distribution.")
    parser.add_argument(
        "--arrival-rate", type=float, default=None,
        help="Poisson arrival rate (req/s) for inference workload. "
             "None = all requests arrive at t=0 (burst); "
             "set to e.g. 5.0 for one request per 200ms on average.")
    parser.add_argument(
        "--out", type=str, default="figs",
        help="Output directory for CSVs (default: figs). "
             "Use a labelled subdir like 'figs/m1_baseline' to keep runs separate.")
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing CSVs in --out (default: refuse)")
    args = parser.parse_args()

    # ---- Output directory safety: refuse to silently overwrite ----
    output_dir = args.out
    expected_files = ["benchmark_raw.csv", "benchmark_summary.csv",
                       "statistical_tests.csv"]
    existing = [f for f in expected_files
                if os.path.exists(os.path.join(output_dir, f))]
    if existing and not args.force:
        print(f"ERROR: {output_dir}/ already contains benchmark output:")
        for f in existing:
            print(f"  - {f}")
        print()
        print("To avoid overwriting historical data, do ONE of:")
        print(f"  1) Write to a new subdir:  --out {output_dir}/run_{datetime.now().strftime('%Y%m%d_%H%M')}")
        print(f"  2) Overwrite anyway:        --force")
        sys.exit(1)

    if args.quick:
        num_runs    = 5
        edge_counts = [3, 7]
    else:
        num_runs    = args.runs
        edge_counts = args.edge if args.edge else [3, 5, 7]

    total_tasks = args.tasks
    interval    = args.interval
    n_checkpoints = total_tasks // interval
    n_schedulers = 1 if args.rl_only else 8
    total = len(edge_counts) * n_schedulers * num_runs

    # ---- Print run header ----
    if args.workload == "inference":
        workload_desc = (f"inference (LLM prefill/decode, "
                         f"~{total_tasks // 2} requests = {total_tasks} tasks)")
        arrival_desc = (f"Poisson λ={args.arrival_rate} req/s"
                        if args.arrival_rate else "burst (t=0)")
        aigc_line = ("  AIGC physics   : implicit (every task has model_id)\n"
                     f"  Trace preset   : {args.trace_preset}, arrival = {arrival_desc}\n"
                     f"  SLO thresholds : TTFT≤{args.ttft_slo}s, "
                     f"TPOT≤{args.tpot_slo}s/tok")
    else:
        workload_desc = "dag (3 generic patterns)"
        aigc_line = (f"  AIGC mode      : {args.aigc} "
                     f"(zipf alpha={args.aigc_alpha})" if args.aigc else
                     "  AIGC mode      : off")

    print("=" * 70)
    print("  Cloud-Edge Scheduling Benchmark (Publication-grade)")
    print("=" * 70)
    print(f"  Runs / config  : {num_runs}")
    print(f"  Edge servers   : {edge_counts}")
    print(f"  Workload       : {workload_desc}")
    print(f"  Total tasks    : {total_tasks}")
    print(f"  Checkpoint     : every {interval} tasks "
          f"({n_checkpoints} data points / run)")
    print(f"  Schedulers     : RoundRobin, LeastLoaded, ShortestQueue, "
          f"HEFT, GA, PSO, A3C_R2N2, RL")
    print(aigc_line)
    print(f"  Ablation       : {args.ablation}")
    print(f"  Output dir     : {output_dir}")
    print(f"  Total sim runs : {total}")
    print("=" * 70)

    # ---- Write run manifest (config snapshot for later cross-run comparison) ----
    os.makedirs(output_dir, exist_ok=True)
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_sha = "unknown"
    manifest = {
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "git_commit":  git_sha,
        "num_runs":    num_runs,
        "edge_counts": edge_counts,
        "workload":    args.workload,
        "total_tasks": total_tasks,
        "interval":    interval,
        "aigc":        args.aigc and args.workload == "dag",
        "aigc_alpha":  (args.aigc_alpha
                        if args.aigc and args.workload == "dag" else None),
        "ablation":    args.ablation,
        "ttft_slo":    args.ttft_slo if args.workload == "inference" else None,
        "tpot_slo":    args.tpot_slo if args.workload == "inference" else None,
        "trace_preset": args.trace_preset if args.workload == "inference" else None,
        "arrival_rate": args.arrival_rate if args.workload == "inference" else None,
        "quick":       args.quick,
        "schedulers":  ["RoundRobin", "LeastLoaded", "ShortestQueue",
                         "HEFT", "GA", "PSO", "A3C_R2N2", "RL"],
        "cli_argv":    sys.argv,
    }
    with open(os.path.join(output_dir, "run_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    tester = BenchmarkTester(num_runs=num_runs,
                              aigc_mode=args.aigc,
                              aigc_zipf_alpha=args.aigc_alpha,
                              workload=args.workload,
                              ablation=args.ablation,
                              ttft_slo=args.ttft_slo,
                              tpot_slo=args.tpot_slo,
                              trace_preset=args.trace_preset,
                              arrival_rate=args.arrival_rate)

    if args.rl_only:
        tester.schedulers = {"RL": tester.schedulers["RL"]}
        print(f"  --rl-only: 只跑 RL 调度器（其他 baseline 跳过）")
    raw, summary = tester.run_benchmark(
        edge_counts, total_tasks=total_tasks,
        checkpoint_interval=interval,
        output_dir=output_dir)

    print("\n" + "=" * 70)
    print(f"  Done! Output files in {output_dir}/")
    print("=" * 70)
