"""
ploter.py — Publication-quality benchmark visualization

Reads benchmark_raw.csv / benchmark_summary.csv / statistical_tests.csv
and generates:
  1. Line plots with 95% CI shading (main comparison)
  2. Box plots with significance annotations (distribution)
  3. Four-metric composite panels
  4. Scalability grouped bar charts
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from itertools import combinations

# ---- Global style ----
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color palette & markers
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
    "makespan":         "Makespan (s)",
    "avg_e2e_latency":  "Avg. End-to-End Latency (s)",
    "avg_utilization":  "Avg. Resource Utilization",
    "load_balance_std": "Load Balance Std. Dev.",
}
# Metrics where lower is better (for highlighting best values)
LOWER_IS_BETTER = {"makespan", "avg_e2e_latency", "load_balance_std"}


class BenchmarkPlotter:
    """Publication-quality benchmark visualization."""

    def __init__(self, data_dir: str = "figs"):
        self.data_dir = data_dir
        self.raw: pd.DataFrame | None = None
        self.summary: pd.DataFrame | None = None
        self.tests: pd.DataFrame | None = None

    # ----------------------------------------------------------------
    #  Data loading
    # ----------------------------------------------------------------

    def load(self):
        raw_path = os.path.join(self.data_dir, "benchmark_raw.csv")
        sum_path = os.path.join(self.data_dir, "benchmark_summary.csv")
        tst_path = os.path.join(self.data_dir, "statistical_tests.csv")

        if os.path.exists(raw_path):
            self.raw = pd.read_csv(raw_path)
            print(f"  Raw data : {len(self.raw)} rows")
        if os.path.exists(sum_path):
            self.summary = pd.read_csv(sum_path)
            print(f"  Summary  : {len(self.summary)} rows")
        if os.path.exists(tst_path):
            self.tests = pd.read_csv(tst_path)
            print(f"  Stat tests: {len(self.tests)} rows")

        if self.raw is None:
            raise FileNotFoundError(
                f"Cannot find {raw_path}. Run brenchmark.py first.")

    def _ensure_data(self):
        if self.raw is None:
            self.load()

    # ----------------------------------------------------------------
    #  1. Line plots with 95% CI shading
    # ----------------------------------------------------------------

    def plot_lines_with_ci(self, metric: str = "makespan",
                           output_file: str | None = None):
        """
        X-axis = task count, Y-axis = metric mean.
        One line per scheduler, shaded area = 95% CI.
        One subplot per edge server count.
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
                sd = sub[sub["scheduler"] == sched].sort_values("completed_tasks")
                x = sd["completed_tasks"].values
                y = sd["mean"].values
                lo = sd["ci_lower"].values
                hi = sd["ci_upper"].values

                color = COLORS.get(sched, "#333")
                marker = MARKERS.get(sched, "o")

                ax.plot(x, y, color=color, marker=marker,
                        linewidth=2, markersize=7, label=sched)
                ax.fill_between(x, lo, hi, color=color, alpha=0.15)

            ax.set_title(f"Edge Servers = {edge}",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("Completed Tasks", fontsize=11)
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
            print(f"  Saved: {output_file}")
        plt.close(fig)

    # ----------------------------------------------------------------
    #  2. Box plots with significance annotations
    # ----------------------------------------------------------------

    def plot_box(self, completed_tasks: int = 300,
                 output_file: str | None = None):
        """
        Box plots showing the distribution of 4 metrics across schedulers
        at a given completed_tasks checkpoint.
        """
        self._ensure_data()
        df = self.raw[self.raw["completed_tasks"] == completed_tasks]
        if df.empty:
            print(f"  No data for completed_tasks={completed_tasks}")
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
                bp = ax.boxplot(
                    data_for_box, labels=scheds, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6))

                for patch, sched in zip(bp["boxes"], scheds):
                    patch.set_facecolor(COLORS.get(sched, "#ccc"))
                    patch.set_alpha(0.7)

                # Significance annotations
                self._annotate_significance(
                    ax, scheds, data_for_box, metric, edge, completed_tasks)

                ax.set_title(METRIC_LABELS.get(metric, metric),
                             fontsize=12, fontweight="bold")
                ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)
                ax.grid(True, alpha=0.3, axis="y", linestyle="--")

            fig.suptitle(
                f"Scheduler Performance Distribution  |  "
                f"Edge={edge}  Tasks={completed_tasks}",
                fontsize=14, fontweight="bold")
            plt.tight_layout()

            if output_file:
                base, ext = os.path.splitext(output_file)
                path = f"{base}_edge{edge}{ext}"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fig.savefig(path)
                print(f"  Saved: {path}")
            plt.close(fig)

    def _annotate_significance(self, ax, scheds, data_list,
                               metric, edge, completed_tasks):
        """Draw significance brackets (* / ** / ***) above box plots."""
        if self.tests is None:
            return
        mask = ((self.tests["edge_servers"] == edge) &
                (self.tests["completed_tasks"] == completed_tasks) &
                (self.tests["metric"] == metric))
        sig_df = self.tests[mask]
        if sig_df.empty:
            return

        y_max = max(max(d) for d in data_list if len(d) > 0)
        y_step = y_max * 0.06
        level = 0

        for _, row in sig_df.iterrows():
            sig = row["significant"]
            if not sig or (isinstance(sig, float) and np.isnan(sig)):
                continue
            try:
                i = scheds.index(row["scheduler_A"])
                j = scheds.index(row["scheduler_B"])
            except ValueError:
                continue

            y = y_max + y_step * (level + 1)
            ax.plot([i + 1, j + 1], [y, y], "k-", linewidth=1)
            ax.text((i + j + 2) / 2, y, str(sig),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
            level += 1

        if level > 0:
            ax.set_ylim(top=y_max + y_step * (level + 2))

    # ----------------------------------------------------------------
    #  3. Four-metric composite panel
    # ----------------------------------------------------------------

    def plot_panel(self, edge_count: int = 5,
                   output_file: str | None = None):
        """2x2 panel: 4 metrics for one edge configuration, lines + CI."""
        self._ensure_data()
        metrics = list(METRIC_LABELS.keys())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for ax_idx, metric in enumerate(metrics):
            ax = axes[ax_idx // 2, ax_idx % 2]
            df = self.summary[(self.summary["metric"] == metric) &
                              (self.summary["edge_servers"] == edge_count)]

            for sched in sorted(df["scheduler"].unique()):
                sd = df[df["scheduler"] == sched].sort_values("completed_tasks")
                x = sd["completed_tasks"].values
                y = sd["mean"].values
                lo = sd["ci_lower"].values
                hi = sd["ci_upper"].values

                color = COLORS.get(sched, "#333")
                marker = MARKERS.get(sched, "o")
                ax.plot(x, y, color=color, marker=marker,
                        linewidth=2, markersize=6, label=sched)
                ax.fill_between(x, lo, hi, color=color, alpha=0.12)

            ax.set_title(METRIC_LABELS[metric],
                         fontsize=12, fontweight="bold")
            ax.set_xlabel("Completed Tasks", fontsize=10)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(fontsize=8, loc="best")

        fig.suptitle(f"Composite Performance Panel  |  Edge Servers = {edge_count}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.savefig(output_file)
            print(f"  Saved: {output_file}")
        plt.close(fig)

    # ----------------------------------------------------------------
    #  4. Scalability grouped bar chart
    # ----------------------------------------------------------------

    def plot_scalability_bars(self, metric: str = "makespan",
                              output_file: str | None = None):
        """Grouped bar chart: X = task count groups, bars = schedulers."""
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
            tasks = sorted(sub["completed_tasks"].unique())

            n_sched = len(scheds)
            bar_width = 0.8 / n_sched
            x_base = np.arange(len(tasks))

            for si, sched in enumerate(scheds):
                sd = sub[sub["scheduler"] == sched].sort_values("completed_tasks")
                means = sd["mean"].values
                stds = sd["std"].values
                x = x_base + si * bar_width

                ax.bar(x, means, bar_width, yerr=stds, capsize=3,
                       color=COLORS.get(sched, "#999"), alpha=0.85,
                       label=sched, edgecolor="white", linewidth=0.5)

            ax.set_title(f"Edge Servers = {edge}",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("Completed Tasks", fontsize=11)
            if idx == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
            ax.set_xticks(x_base + bar_width * (n_sched - 1) / 2)
            ax.set_xticklabels(tasks)
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            ax.legend(fontsize=8, loc="best")

        fig.suptitle(
            f"Scalability Analysis — {METRIC_LABELS.get(metric, metric)}",
            fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.savefig(output_file)
            print(f"  Saved: {output_file}")
        plt.close(fig)

    # ----------------------------------------------------------------
    #  5. Summary table (console output)
    # ----------------------------------------------------------------

    def print_summary_table(self, completed_tasks: int = 300):
        """Print a mean +/- std table suitable for papers."""
        self._ensure_data()
        df = self.summary[self.summary["completed_tasks"] == completed_tasks]
        if df.empty:
            print(f"  No data for completed_tasks={completed_tasks}")
            return

        metrics = list(METRIC_LABELS.keys())
        short_labels = {
            "makespan":         "Makespan",
            "avg_e2e_latency":  "E2E Latency",
            "avg_utilization":  "Utilization",
            "load_balance_std": "Load Std",
        }

        for edge in sorted(df["edge_servers"].unique()):
            print(f"\n  Edge Servers = {edge}, Tasks = {completed_tasks}")
            header = f"  {'Scheduler':<12s}"
            for m in metrics:
                header += f" | {short_labels[m]:>14s}"
            print(header)
            print("  " + "-" * (14 + len(metrics) * 17))

            sub = df[df["edge_servers"] == edge]
            for sched in sorted(sub["scheduler"].unique()):
                ss = sub[sub["scheduler"] == sched]
                line = f"  {sched:<12s}"
                for m in metrics:
                    row = ss[ss["metric"] == m]
                    if row.empty:
                        line += f" | {'N/A':>14s}"
                    else:
                        mean = row["mean"].values[0]
                        std = row["std"].values[0]
                        line += f" | {mean:7.2f}±{std:<5.2f}"
                print(line)

    # ================================================================
    #  Generate full report
    # ================================================================

    def generate_report(self, output_dir: str = "figs/report"):
        """Generate all publication figures."""
        self._ensure_data()
        os.makedirs(output_dir, exist_ok=True)

        print("\n  Generating publication figures...")

        # 1) Line plots with CI for each metric
        for metric in METRIC_LABELS:
            self.plot_lines_with_ci(
                metric,
                os.path.join(output_dir, f"lines_{metric}.png"))

        # 2) Box plots at the largest checkpoint
        max_tasks = int(self.raw["completed_tasks"].max())
        self.plot_box(max_tasks,
                      os.path.join(output_dir, "boxplot.png"))

        # 3) Composite panels for each edge count
        for edge in sorted(self.raw["edge_servers"].unique()):
            self.plot_panel(
                edge,
                os.path.join(output_dir, f"panel_edge{edge}.png"))

        # 4) Scalability bar charts
        for metric in ["makespan", "avg_e2e_latency"]:
            self.plot_scalability_bars(
                metric,
                os.path.join(output_dir, f"scalability_{metric}.png"))

        # 5) Summary tables (only at key checkpoints to avoid flooding)
        checkpoints = sorted(self.raw["completed_tasks"].unique())
        # Show summary at first, middle, and last checkpoint
        key_cps = [checkpoints[0], checkpoints[len(checkpoints)//2],
                   checkpoints[-1]]
        for tc in dict.fromkeys(key_cps):  # deduplicate while preserving order
            self.print_summary_table(tc)

        print(f"\n  All figures saved to {output_dir}/")


# ================================================================
#  CLI entry
# ================================================================

if __name__ == "__main__":
    plotter = BenchmarkPlotter()
    plotter.generate_report()
