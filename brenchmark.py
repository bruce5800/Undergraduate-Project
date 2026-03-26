"""
brenchmark.py — 论文级基准测试框架

P0 改进：
  1. 修正资源利用率：加权计算 util = Σ(compute_demand × duration) / (capacity × makespan)
     结果严格 ∈ [0, 1]，不再出现 > 1 的异常值
  2. 多轮运行 + 统计检验：每配置 N 次（默认 20），报告均值 ± 标准差，
     Mann-Whitney U 检验各调度器之间是否有显著差异
  3. 任务规模梯度：100 / 200 / 300 / 500，分析可扩展性

输出文件：
  figs/benchmark_raw.csv        — 每一轮原始数据
  figs/benchmark_summary.csv    — 均值 / 标准差 / 95%CI / 中位数 / 最小最大
  figs/statistical_tests.csv    — 成对 Mann-Whitney U 检验结果
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

        # 全部 5 种调度器
        self.schedulers = {
            "RoundRobin": RoundRobinScheduler,
            "HEFT":       HEFTScheduler,
            "GA":         GAScheduler,
            "PSO":        PSOScheduler,
            "RL":         RLScheduler,
        }

    # ----------------------------------------------------------------
    #  环境创建
    # ----------------------------------------------------------------

    def create_simulation(self, task_count: int, num_edge_servers: int,
                          seed: int) -> Simulation:
        """创建仿真环境，使用指定 seed 确保可复现"""
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
    #  单轮运行
    # ----------------------------------------------------------------

    def run_single(self, scheduler_name: str, scheduler_class,
                   task_count: int, num_edge_servers: int, seed: int,
                   max_time: float = 10000) -> dict:
        """运行一次完整仿真，返回最终指标 dict"""
        # 设置所有随机种子
        random.seed(seed)
        np.random.seed(seed)
        if HAS_TORCH:
            torch.manual_seed(seed)

        sim = self.create_simulation(task_count, num_edge_servers, seed)

        # RL 使用较少的预训练轮次以控制总耗时
        if scheduler_class is RLScheduler:
            scheduler = RLScheduler(sim, pretrain_episodes=5)
        else:
            scheduler = scheduler_class(sim)

        current_time = 0.0
        while (len(sim.completed_tasks) < len(sim.tasks)
               and current_time < max_time):
            sim.step(scheduler, current_time)
            current_time += 0.1

        return self._collect_metrics(sim, current_time)

    # ----------------------------------------------------------------
    #  指标收集（修正利用率）
    # ----------------------------------------------------------------

    def _collect_metrics(self, sim: Simulation, current_time: float) -> dict:
        completed = [t for t in sim.tasks.values()
                     if t.task_id in sim.completed_tasks]

        # 1) Makespan
        makespan = round(current_time, 2)

        # 2) 平均端到端延迟 (READY → COMPLETED)
        e2e = [t.end_time - t.ready_time
               for t in completed
               if t.ready_time is not None and t.end_time is not None]
        avg_e2e = float(np.mean(e2e)) if e2e else 0.0

        # 3) 修正的资源利用率 (P0-1)
        #    util_s = Σ_task(compute_demand × duration) / (capacity × makespan)
        #    结果严格 ∈ [0, 1]
        utils = []
        for server in sim.servers.values():
            resource_seconds = sum(
                t.compute_demand * (t.end_time - t.start_time)
                for t in server.task_history
                if (t.task_id in sim.completed_tasks
                    and t.start_time is not None
                    and t.end_time is not None)
            )
            denominator = server.total_compute * current_time
            util = resource_seconds / denominator if denominator > 0 else 0.0
            utils.append(util)

        avg_util = float(np.mean(utils)) if utils else 0.0
        load_std = float(np.std(utils)) if utils else 0.0

        # 4) 完成率
        completed_ratio = len(sim.completed_tasks) / max(len(sim.tasks), 1)

        return {
            "makespan":          makespan,
            "avg_e2e_latency":   round(avg_e2e, 4),
            "avg_utilization":   round(avg_util, 4),
            "load_balance_std":  round(load_std, 4),
            "completed_ratio":   round(completed_ratio, 4),
        }

    # ================================================================
    #  主测试入口
    # ================================================================

    def run_benchmark(self, edge_counts: list, task_counts: list,
                      output_dir: str = "figs"):
        os.makedirs(output_dir, exist_ok=True)

        raw_results: list[dict] = []
        total_runs = (len(edge_counts) * len(task_counts)
                      * len(self.schedulers) * self.num_runs)
        progress = 0
        wall_start = time.time()

        for edge_count in edge_counts:
            print(f"\n{'='*60}")
            print(f"  边缘服务器数量: {edge_count}")
            print(f"{'='*60}")

            for task_count in task_counts:
                print(f"\n  任务规模: {task_count}")
                print(f"  {'-'*50}")

                for sched_name, sched_class in self.schedulers.items():
                    run_makespans = []

                    for run_idx in range(self.num_runs):
                        seed = self.base_seed + run_idx

                        t0 = time.time()
                        metrics = self.run_single(
                            sched_name, sched_class,
                            task_count, edge_count, seed)
                        elapsed = time.time() - t0

                        raw_results.append({
                            "edge_servers": edge_count,
                            "task_count":   task_count,
                            "scheduler":    sched_name,
                            "run":          run_idx,
                            "seed":         seed,
                            **metrics,
                        })

                        run_makespans.append(metrics["makespan"])
                        progress += 1

                        # 进度条
                        pct = progress / total_runs * 100
                        eta = ((time.time() - wall_start) / progress
                               * (total_runs - progress))
                        sys.stdout.write(
                            f"\r    [{progress}/{total_runs}] {pct:5.1f}%  "
                            f"{sched_name:<12s} run {run_idx+1:>2d}/{self.num_runs}  "
                            f"({elapsed:5.1f}s)  ETA {eta/60:5.1f}min   ")
                        sys.stdout.flush()

                    # 该调度器在本配置的汇总
                    m = np.mean(run_makespans)
                    s = np.std(run_makespans, ddof=1) if len(run_makespans) > 1 else 0
                    print(f"\n    {sched_name:<12s}: "
                          f"makespan = {m:7.1f} ± {s:5.1f}")

        wall_total = time.time() - wall_start
        print(f"\n\n{'='*60}")
        print(f"  总耗时: {wall_total/60:.1f} 分钟  |  总运行数: {total_runs}")
        print(f"{'='*60}")

        # ---------- 导出 ----------
        raw_file = os.path.join(output_dir, "benchmark_raw.csv")
        summary_file = os.path.join(output_dir, "benchmark_summary.csv")
        stat_file = os.path.join(output_dir, "statistical_tests.csv")

        self._export_raw(raw_results, raw_file)
        summary = self._compute_summary(raw_results)
        self._export_summary(summary, summary_file)
        self._run_statistical_tests(raw_results, stat_file)

        return raw_results, summary

    # ================================================================
    #  统计汇总
    # ================================================================

    def _compute_summary(self, raw_results: list) -> list:
        """按 (edge, task_count, scheduler) 分组，计算每个指标的描述统计量"""
        import pandas as pd
        df = pd.DataFrame(raw_results)

        metrics = ["makespan", "avg_e2e_latency",
                    "avg_utilization", "load_balance_std"]
        summary = []

        groups = df.groupby(["edge_servers", "task_count", "scheduler"])
        for (edge, tasks, sched), grp in groups:
            for metric in metrics:
                vals = grp[metric].values
                n = len(vals)
                mean = float(np.mean(vals))
                std  = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                se   = std / np.sqrt(n) if n > 1 else 0.0
                ci   = 1.96 * se  # 95 % CI

                summary.append({
                    "edge_servers": edge,
                    "task_count":   tasks,
                    "scheduler":    sched,
                    "metric":       metric,
                    "mean":         round(mean, 4),
                    "std":          round(std, 4),
                    "ci_lower":     round(mean - ci, 4),
                    "ci_upper":     round(mean + ci, 4),
                    "min":          round(float(np.min(vals)), 4),
                    "max":          round(float(np.max(vals)), 4),
                    "median":       round(float(np.median(vals)), 4),
                    "n":            n,
                })

        return summary

    # ================================================================
    #  统计显著性检验
    # ================================================================

    def _run_statistical_tests(self, raw_results: list,
                               output_file: str):
        """成对 Mann-Whitney U 检验（非参数，不假设正态分布）"""
        import pandas as pd
        df = pd.DataFrame(raw_results)

        metrics = ["makespan", "avg_e2e_latency",
                    "avg_utilization", "load_balance_std"]
        scheds = sorted(df["scheduler"].unique())
        results = []

        for (edge, tasks), cfg in df.groupby(["edge_servers", "task_count"]):
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
                            "edge_servers":  edge,
                            "task_count":    tasks,
                            "metric":        metric,
                            "scheduler_A":   sa,
                            "scheduler_B":   sb,
                            "mean_A":        round(float(np.mean(va)), 4),
                            "mean_B":        round(float(np.mean(vb)), 4),
                            "U_statistic":   round(float(stat), 2),
                            "p_value":       round(float(p), 6),
                            "significant":   sig,
                        })

        if results:
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"  统计检验结果 → {output_file} ({len(results)} 条)")
        else:
            print("  无统计检验结果")

    # ================================================================
    #  CSV 导出
    # ================================================================

    def _export_raw(self, results: list, output_file: str):
        if not results:
            return
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"  原始数据    → {output_file} ({len(results)} 条)")

    def _export_summary(self, summary: list, output_file: str):
        if not summary:
            return
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
        print(f"  汇总统计    → {output_file} ({len(summary)} 条)")


# ====================================================================
#  命令行入口
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="边缘计算调度算法基准测试（论文级）")
    parser.add_argument(
        "--runs", type=int, default=20,
        help="每配置运行次数 (默认 20)")
    parser.add_argument(
        "--quick", action="store_true",
        help="快速模式：5 轮、少量配置，用于开发测试")
    parser.add_argument(
        "--edge", type=int, nargs="+", default=None,
        help="自定义边缘服务器数量列表，如 --edge 3 5 7")
    parser.add_argument(
        "--tasks", type=int, nargs="+", default=None,
        help="自定义任务规模列表，如 --tasks 100 300 500")
    args = parser.parse_args()

    if args.quick:
        num_runs    = 5
        edge_counts = [3, 7]
        task_counts = [100, 300]
    else:
        num_runs    = args.runs
        edge_counts = args.edge  if args.edge  else [3, 5, 7]
        task_counts = args.tasks if args.tasks else [100, 200, 300, 500]

    total = len(edge_counts) * len(task_counts) * 5 * num_runs

    print("=" * 70)
    print("  边缘计算调度算法基准测试（论文级）")
    print("=" * 70)
    print(f"  运行次数/配置 : {num_runs}")
    print(f"  边缘服务器    : {edge_counts}")
    print(f"  任务规模      : {task_counts}")
    print(f"  调度器        : RoundRobin, HEFT, GA, PSO, RL")
    print(f"  总运行数      : {total}")
    print("=" * 70)

    tester = BenchmarkTester(num_runs=num_runs)
    raw, summary = tester.run_benchmark(edge_counts, task_counts)

    print("\n" + "=" * 70)
    print("  测试完成！输出文件在 figs/ 目录下")
    print("=" * 70)
