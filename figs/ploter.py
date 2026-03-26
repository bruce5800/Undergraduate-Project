"""
ploter.py — 论文级可视化

读取 benchmark_raw.csv / benchmark_summary.csv / statistical_tests.csv，
生成以下图表：
  1. 折线图 + 95% CI 阴影带（核心对比图）
  2. 箱线图 / 小提琴图（分布展示）
  3. 综合四指标面板
  4. 可扩展性分析图（不同任务规模下的表现）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from itertools import combinations

# ---- 全局样式 ----
matplotlib.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei",
                        "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# 配色与样式
COLORS = {
    "RoundRobin": "#1f77b4",
    "HEFT":       "#9467bd",
    "GA":         "#2ca02c",
    "PSO":        "#d62728",
    "RL":         "#ff7f0e",
}
MARKERS = {
    "RoundRobin": "o",
    "HEFT":       "D",
    "GA":         "^",
    "PSO":        "v",
    "RL":         "s",
}
METRIC_LABELS = {
    "makespan":         "总运行时间 Makespan (s)",
    "avg_e2e_latency":  "平均端到端延迟 (s)",
    "avg_utilization":  "平均资源利用率",
    "load_balance_std": "负载均衡标准差",
}
# 标记哪些指标"越小越好"（用于高亮最优值）
LOWER_IS_BETTER = {"makespan", "avg_e2e_latency", "load_balance_std"}


class BenchmarkPlotter:
    """论文级基准测试可视化"""

    def __init__(self, data_dir: str = "figs"):
        self.data_dir = data_dir
        self.raw: pd.DataFrame | None = None
        self.summary: pd.DataFrame | None = None
        self.tests: pd.DataFrame | None = None

    # ----------------------------------------------------------------
    #  数据加载
    # ----------------------------------------------------------------

    def load(self):
        raw_path = os.path.join(self.data_dir, "benchmark_raw.csv")
        sum_path = os.path.join(self.data_dir, "benchmark_summary.csv")
        tst_path = os.path.join(self.data_dir, "statistical_tests.csv")

        if os.path.exists(raw_path):
            self.raw = pd.read_csv(raw_path)
            print(f"  原始数据: {len(self.raw)} 条")
        if os.path.exists(sum_path):
            self.summary = pd.read_csv(sum_path)
            print(f"  汇总统计: {len(self.summary)} 条")
        if os.path.exists(tst_path):
            self.tests = pd.read_csv(tst_path)
            print(f"  统计检验: {len(self.tests)} 条")

        if self.raw is None:
            raise FileNotFoundError(
                f"找不到 {raw_path}，请先运行 brenchmark.py")

    def _ensure_data(self):
        if self.raw is None:
            self.load()

    # ----------------------------------------------------------------
    #  1. 折线图 + 95% CI 阴影（核心对比图）
    # ----------------------------------------------------------------

    def plot_lines_with_ci(self, metric: str = "makespan",
                           output_file: str | None = None):
        """
        横轴 = 任务规模，纵轴 = 指标均值
        每条线 = 一个调度器，阴影 = 95% CI
        每个边缘服务器数量一个子图
        """
        self._ensure_data()
        df = self.summary[self.summary["metric"] == metric]

        edge_counts = sorted(df["edge_servers"].unique())
        n_cols = len(edge_counts)

        fig, axes = plt.subplots(1, n_cols,
                                  figsize=(6 * n_cols, 5), squeeze=False)

        for idx, edge in enumerate(edge_counts):
            ax = axes[0, idx]
            sub = df[df["edge_servers"] == edge]

            for sched in sub["scheduler"].unique():
                sd = sub[sub["scheduler"] == sched].sort_values("task_count")
                x = sd["task_count"].values
                y = sd["mean"].values
                lo = sd["ci_lower"].values
                hi = sd["ci_upper"].values

                color = COLORS.get(sched, "#333")
                marker = MARKERS.get(sched, "o")

                ax.plot(x, y, color=color, marker=marker,
                        linewidth=2, markersize=7, label=sched)
                ax.fill_between(x, lo, hi, color=color, alpha=0.15)

            ax.set_title(f"边缘服务器 = {edge}", fontsize=13, fontweight="bold")
            ax.set_xlabel("任务数量", fontsize=11)
            if idx == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(fontsize=9, loc="best")

        fig.suptitle(METRIC_LABELS.get(metric, metric),
                     fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.savefig(output_file)
            print(f"  保存: {output_file}")
        plt.close(fig)

    # ----------------------------------------------------------------
    #  2. 箱线图（分布展示）
    # ----------------------------------------------------------------

    def plot_box(self, task_count: int = 300,
                 output_file: str | None = None):
        """
        在指定任务规模下，展示各调度器 4 个指标的分布（箱线图）
        """
        self._ensure_data()
        df = self.raw[self.raw["task_count"] == task_count]
        if df.empty:
            print(f"  无 task_count={task_count} 的数据")
            return

        edge_counts = sorted(df["edge_servers"].unique())
        metrics = list(METRIC_LABELS.keys())

        for edge in edge_counts:
            sub = df[df["edge_servers"] == edge]
            scheds = sorted(sub["scheduler"].unique())

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            for ax_idx, metric in enumerate(metrics):
                ax = axes[ax_idx // 2, ax_idx % 2]

                data_for_box = [
                    sub.loc[sub["scheduler"] == s, metric].values
                    for s in scheds
                ]
                bp = ax.boxplot(data_for_box, labels=scheds, patch_artist=True,
                                widths=0.5, showmeans=True,
                                meanprops=dict(marker="D", markerfacecolor="white",
                                               markeredgecolor="black", markersize=6))

                for patch, sched in zip(bp["boxes"], scheds):
                    patch.set_facecolor(COLORS.get(sched, "#ccc"))
                    patch.set_alpha(0.7)

                # 显著性标注
                self._annotate_significance(
                    ax, scheds, data_for_box, metric, edge, task_count)

                ax.set_title(METRIC_LABELS.get(metric, metric),
                             fontsize=12, fontweight="bold")
                ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)
                ax.grid(True, alpha=0.3, axis="y", linestyle="--")

            fig.suptitle(
                f"调度器性能分布对比  |  边缘={edge}  任务={task_count}",
                fontsize=14, fontweight="bold")
            plt.tight_layout()

            if output_file:
                base, ext = os.path.splitext(output_file)
                path = f"{base}_edge{edge}{ext}"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fig.savefig(path)
                print(f"  保存: {path}")
            plt.close(fig)

    def _annotate_significance(self, ax, scheds, data_list,
                               metric, edge, task_count):
        """在箱线图上方标注显著性 * / ** / ***"""
        if self.tests is None:
            return
        mask = ((self.tests["edge_servers"] == edge) &
                (self.tests["task_count"] == task_count) &
                (self.tests["metric"] == metric))
        sig_df = self.tests[mask]
        if sig_df.empty:
            return

        y_max = max(max(d) for d in data_list if len(d) > 0)
        y_step = y_max * 0.06
        level = 0

        for _, row in sig_df.iterrows():
            sig = row["significant"]
            if not sig:
                continue
            try:
                i = scheds.index(row["scheduler_A"])
                j = scheds.index(row["scheduler_B"])
            except ValueError:
                continue

            y = y_max + y_step * (level + 1)
            ax.plot([i + 1, j + 1], [y, y], "k-", linewidth=1)
            ax.text((i + j + 2) / 2, y, sig,
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
            level += 1

        if level > 0:
            ax.set_ylim(top=y_max + y_step * (level + 2))

    # ----------------------------------------------------------------
    #  3. 四指标综合面板（某个边缘配置）
    # ----------------------------------------------------------------

    def plot_panel(self, edge_count: int = 5,
                   output_file: str | None = None):
        """2×2 面板：同一边缘配置下 4 个指标的折线 + CI"""
        self._ensure_data()
        metrics = list(METRIC_LABELS.keys())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for ax_idx, metric in enumerate(metrics):
            ax = axes[ax_idx // 2, ax_idx % 2]
            df = self.summary[(self.summary["metric"] == metric) &
                              (self.summary["edge_servers"] == edge_count)]

            for sched in sorted(df["scheduler"].unique()):
                sd = df[df["scheduler"] == sched].sort_values("task_count")
                x = sd["task_count"].values
                y = sd["mean"].values
                lo = sd["ci_lower"].values
                hi = sd["ci_upper"].values

                color = COLORS.get(sched, "#333")
                marker = MARKERS.get(sched, "o")
                ax.plot(x, y, color=color, marker=marker,
                        linewidth=2, markersize=6, label=sched)
                ax.fill_between(x, lo, hi, color=color, alpha=0.12)

            ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
            ax.set_xlabel("任务数量", fontsize=10)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(fontsize=8, loc="best")

        fig.suptitle(f"综合性能面板  |  边缘服务器 = {edge_count}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.savefig(output_file)
            print(f"  保存: {output_file}")
        plt.close(fig)

    # ----------------------------------------------------------------
    #  4. 可扩展性柱状图（固定边缘，不同任务规模）
    # ----------------------------------------------------------------

    def plot_scalability_bars(self, metric: str = "makespan",
                              output_file: str | None = None):
        """分组柱状图：横轴 = 任务规模分组，每组内各调度器一根柱子"""
        self._ensure_data()
        df = self.summary[self.summary["metric"] == metric]
        edge_counts = sorted(df["edge_servers"].unique())

        fig, axes = plt.subplots(1, len(edge_counts),
                                  figsize=(6 * len(edge_counts), 5),
                                  squeeze=False)

        for idx, edge in enumerate(edge_counts):
            ax = axes[0, idx]
            sub = df[df["edge_servers"] == edge]
            scheds = sorted(sub["scheduler"].unique())
            tasks = sorted(sub["task_count"].unique())

            n_sched = len(scheds)
            bar_width = 0.8 / n_sched
            x_base = np.arange(len(tasks))

            for si, sched in enumerate(scheds):
                sd = sub[sub["scheduler"] == sched].sort_values("task_count")
                means = sd["mean"].values
                stds = sd["std"].values
                x = x_base + si * bar_width

                ax.bar(x, means, bar_width, yerr=stds, capsize=3,
                       color=COLORS.get(sched, "#999"), alpha=0.85,
                       label=sched, edgecolor="white", linewidth=0.5)

            ax.set_title(f"边缘服务器 = {edge}", fontsize=13, fontweight="bold")
            ax.set_xlabel("任务数量", fontsize=11)
            if idx == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
            ax.set_xticks(x_base + bar_width * (n_sched - 1) / 2)
            ax.set_xticklabels(tasks)
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            ax.legend(fontsize=8, loc="best")

        fig.suptitle(f"可扩展性分析 — {METRIC_LABELS.get(metric, metric)}",
                     fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.savefig(output_file)
            print(f"  保存: {output_file}")
        plt.close(fig)

    # ----------------------------------------------------------------
    #  5. 统计摘要表格（控制台输出）
    # ----------------------------------------------------------------

    def print_summary_table(self, task_count: int = 300):
        """打印论文可用的均值±标准差表格"""
        self._ensure_data()
        df = self.summary[self.summary["task_count"] == task_count]
        if df.empty:
            print(f"  无 task_count={task_count} 的数据")
            return

        metrics = list(METRIC_LABELS.keys())

        for edge in sorted(df["edge_servers"].unique()):
            print(f"\n  边缘服务器 = {edge}, 任务 = {task_count}")
            print(f"  {'调度器':<12s}", end="")
            for m in metrics:
                print(f" | {METRIC_LABELS[m][:12]:>14s}", end="")
            print()
            print("  " + "-" * 76)

            sub = df[df["edge_servers"] == edge]
            for sched in sorted(sub["scheduler"].unique()):
                ss = sub[sub["scheduler"] == sched]
                print(f"  {sched:<12s}", end="")
                for m in metrics:
                    row = ss[ss["metric"] == m]
                    if row.empty:
                        print(f" | {'N/A':>14s}", end="")
                    else:
                        mean = row["mean"].values[0]
                        std = row["std"].values[0]
                        print(f" | {mean:7.2f}±{std:<5.2f}", end="")
                print()

    # ================================================================
    #  一键生成全部报告
    # ================================================================

    def generate_report(self, output_dir: str = "figs/report"):
        """生成论文所需的全部图表"""
        self._ensure_data()
        os.makedirs(output_dir, exist_ok=True)

        print("\n  生成论文图表...")

        # 1) 各指标折线 + CI
        for metric in METRIC_LABELS:
            self.plot_lines_with_ci(
                metric,
                os.path.join(output_dir, f"lines_{metric}.png"))

        # 2) 箱线图（取最大任务规模）
        max_tasks = int(self.raw["task_count"].max())
        self.plot_box(max_tasks,
                      os.path.join(output_dir, "boxplot.png"))

        # 3) 综合面板
        for edge in sorted(self.raw["edge_servers"].unique()):
            self.plot_panel(
                edge,
                os.path.join(output_dir, f"panel_edge{edge}.png"))

        # 4) 可扩展性柱状图
        for metric in ["makespan", "avg_e2e_latency"]:
            self.plot_scalability_bars(
                metric,
                os.path.join(output_dir, f"scalability_{metric}.png"))

        # 5) 摘要表格
        for tc in sorted(self.raw["task_count"].unique()):
            self.print_summary_table(tc)

        print(f"\n  全部图表已保存到 {output_dir}/")


# ================================================================
#  命令行入口
# ================================================================

if __name__ == "__main__":
    plotter = BenchmarkPlotter()
    plotter.generate_report()
