"""
plot_fig5_load.py — Paper Figure 5: Load (λ) sensitivity

Two side-by-side panels:
  (a) SLO Attainment vs arrival rate λ
  (b) Energy per Token vs arrival rate λ

Shows 5 representative schedulers (out of 9):
  RL (ours), PSO (best heuristic), ShortestQueue (best LB),
  A3C-R2N2 (controlled RL baseline), GNN (alternative AIGC encoder).

Data: figs/exp2_lambda{0.5,1.0,4.0,8.0}/ + figs/energy_scan2/ (λ=2)
Output: figs/report/fig5_load.{png,pdf}
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

N_RUNS = 30
EDGE = "5"
TASKS = "100"
OUTPUT_DIR = "figs/report"

# λ → data directory
LAMBDA_DIRS = {
    0.5: "figs/exp2_lambda0.5",
    1.0: "figs/exp2_lambda1.0",
    2.0: "figs/energy_scan2",
    4.0: "figs/exp2_lambda4.0",
    8.0: "figs/exp2_lambda8.0",
}

# Representative schedulers
SCHEDULER_STYLE = [
    # name           color       marker  linestyle  size  label
    ("ShortestQueue","#2ca02c",   "^",   "--",       7,   "ShortestQueue (LB)"),
    ("PSO",          "#e377c2",   "h",   "--",       8,   "PSO (heuristic)"),
    ("GNN",          "#bcbd22",   "P",   "-.",       8,   "GNN (RL baseline)"),
    ("A3C_R2N2",     "#17becf",   "D",   "-.",       7,   "A3C-R2N2 (RL baseline)"),
    ("RL",           "#d62728",   "*",   "-",       14,   "RL (ours)"),
]


def load_metric(d, sched, metric):
    """Return (mean, std) at given edge/completed."""
    path = os.path.join(d, "benchmark_summary.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            if (row["edge_servers"] == EDGE
                    and row["completed_tasks"] == TASKS
                    and row["scheduler"] == sched
                    and row["metric"] == metric):
                return float(row["mean"]), float(row["std"])
    raise ValueError(f"Not found: {d} {sched} {metric}")


def ci95(std, n=N_RUNS):
    return 1.96 * std / np.sqrt(n)


def gather():
    """Return dict: sched -> {'slo': [(λ, mean, ci), ...], 'e': [...]}"""
    lambdas = sorted(LAMBDA_DIRS.keys())
    out = {}
    for sched, color, marker, ls, size, label in SCHEDULER_STYLE:
        slo_pts, e_pts = [], []
        for lam in lambdas:
            slo_m, slo_s = load_metric(LAMBDA_DIRS[lam], sched, "slo_attainment")
            e_m, e_s = load_metric(LAMBDA_DIRS[lam], sched, "energy_per_token")
            slo_pts.append((lam, slo_m, ci95(slo_s)))
            e_pts.append((lam, e_m, ci95(e_s)))
        out[sched] = {"slo": slo_pts, "e": e_pts, "style": (color, marker, ls, size, label)}
    return out


def plot_one(ax, data, metric_key, ylabel, title, ylim=None, legend=False,
              annotate_regimes=False):
    for sched, _, _, _, _, label in SCHEDULER_STYLE:
        d = data[sched]
        color, marker, ls, size, _ = d["style"]
        pts = d[metric_key]
        lams = [p[0] for p in pts]
        means = [p[1] for p in pts]
        cis = [p[2] for p in pts]
        is_rl = sched == "RL"
        ax.errorbar(
            lams, means, yerr=cis,
            marker=marker, color=color,
            markersize=size,
            markeredgecolor="black" if is_rl else color,
            markeredgewidth=1.3 if is_rl else 0.5,
            linewidth=2.2 if is_rl else 1.3,
            linestyle=ls,
            capsize=3,
            label=label,
            zorder=10 if is_rl else 5,
        )

    # Optional regime shading on SLO panel
    if annotate_regimes:
        # Sweet spot λ ∈ [1, 2]
        ax.axvspan(1.0, 2.0, alpha=0.10, color="#d62728", zorder=1)
        # Saturated λ ≥ 4
        ax.axvspan(4.0, 8.0, alpha=0.08, color="#888888", zorder=1)
        # Labels at top
        ax.text(1.4, 0.62, "Sweet\nspot",
                ha="center", va="top", fontsize=9, color="#d62728",
                fontweight="bold", alpha=0.85)
        ax.text(6.0, 0.62, "Saturated",
                ha="center", va="top", fontsize=9, color="#666",
                fontweight="bold", alpha=0.85)

    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.0, 2.0, 4.0, 8.0])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax.set_xlabel("Arrival rate λ (req/s)  →  higher load",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=11, pad=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(loc="best", fontsize=9, framealpha=0.95)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = gather()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    plot_one(ax1, data, "slo", "SLO Attainment  ↑  better",
             "(a) SLO Attainment vs Load",
             ylim=(0.10, 0.68), legend=True, annotate_regimes=True)

    plot_one(ax2, data, "e", "Energy per Token (J/tok)  ↓  better",
             "(b) Energy per Token vs Load",
             ylim=(2.4, 3.6), legend=False)

    # Annotate: RL maintains energy advantage across full load range
    ax2.annotate(
        "RL maintains energy advantage\nacross entire load range",
        xy=(2, 2.61), xytext=(3.5, 3.30),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4),
        fontsize=10, color="#d62728", fontweight="bold",
    )

    fig.suptitle(
        "Figure 5. Load sensitivity"
        "  ·  edge=5, uniform model mix, N=30 runs, error bars = 95% CI",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "fig5_load.png")
    out_pdf = os.path.join(OUTPUT_DIR, "fig5_load.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}\nSaved: {out_pdf}")

    # Print summary table for paper
    print("\n=== Fig 5 data ===")
    print(f"{'Scheduler':<22} | {'λ=0.5':>14} | {'λ=1':>14} | {'λ=2':>14} | {'λ=4':>14} | {'λ=8':>14}")
    for sched, *_, label in SCHEDULER_STYLE:
        slo_row = " | ".join(
            f"{m:>5.3f} ± {c:>5.3f}" for _, m, c in data[sched]["slo"]
        )
        print(f"{label:<22} | {slo_row}  (SLO)")


if __name__ == "__main__":
    main()
