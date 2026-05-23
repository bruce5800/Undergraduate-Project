"""
plot_fig4_pareto.py — Paper Figure 4: SLO vs Energy Pareto comparison

Reads figs/energy_scan2/benchmark_summary.csv and produces a publication-quality
matplotlib figure showing all 9 schedulers in (SLO, Energy/token) space.

Layout: two side-by-side panels —
  (a) Full view: all 9 schedulers including RoundRobin outlier
  (b) Zoom: 8 strong schedulers (RoundRobin excluded)

Ideal corner is upper-left in both panels.

Outputs:
    figs/report/fig4_pareto.png
    figs/report/fig4_pareto.pdf
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, ConnectionPatch

DATA_DIR = "figs/energy_scan2"
OUTPUT_DIR = "figs/report"
N_RUNS = 30

# Per-scheduler style: (name, marker, color, marker_size, display_label)
SCHEDULER_STYLE = [
    ("RoundRobin",   "X",  "#888888",   10,  "RoundRobin"),
    ("LeastLoaded",  "v",  "#1f77b4",    9,  "LeastLoaded"),
    ("ShortestQueue","^",  "#2ca02c",    9,  "ShortestQueue"),
    ("HEFT",         "s",  "#9467bd",    9,  "HEFT"),
    ("GA",           "p",  "#8c564b",   10,  "GA"),
    ("PSO",          "h",  "#e377c2",   10,  "PSO"),
    ("A3C_R2N2",     "D",  "#17becf",    9,  "A3C-R2N2"),
    ("GNN",          "P",  "#bcbd22",   10,  "GNN"),
    ("RL",           "*",  "#d62728",   22,  "RL (ours)"),
]

# Per-scheduler label offset for the ZOOM panel (manual tuning for readability)
ZOOM_LABEL_OFFSET = {
    "LeastLoaded":   (0.005,  0.005),
    "ShortestQueue": (0.005, -0.012),
    "HEFT":          (0.005,  0.000),
    "GA":            (-0.075, -0.010),
    "PSO":           (-0.060,  0.008),
    "A3C_R2N2":      (0.005, -0.005),
    "GNN":           (-0.045, -0.012),
    "RL":            (-0.090,  0.010),
}


def load_metric(sched, metric):
    path = os.path.join(DATA_DIR, "benchmark_summary.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            if (row["edge_servers"] == "5"
                    and row["completed_tasks"] == "100"
                    and row["scheduler"] == sched
                    and row["metric"] == metric):
                return float(row["mean"]), float(row["std"])
    raise ValueError(f"row not found: {sched} {metric}")


def ci95(std, n=N_RUNS):
    return 1.96 * std / np.sqrt(n)


def load_all():
    data = []
    for name, marker, color, size, label in SCHEDULER_STYLE:
        slo_m, slo_s = load_metric(name, "slo_attainment")
        e_m, e_s = load_metric(name, "energy_per_token")
        data.append({
            "name":  name, "label": label,
            "marker": marker, "color": color, "size": size,
            "slo": slo_m, "slo_ci": ci95(slo_s),
            "e": e_m, "e_ci": ci95(e_s),
        })
    return data


def draw_points(ax, data, label_offsets=None, fontsize=9):
    """Draw markers with error bars and per-point labels."""
    for d in data:
        is_rl = d["name"] == "RL"
        ax.errorbar(
            d["e"], d["slo"],
            xerr=d["e_ci"], yerr=d["slo_ci"],
            fmt=d["marker"],
            color=d["color"],
            markersize=d["size"],
            markeredgecolor="black" if is_rl else d["color"],
            markeredgewidth=1.4 if is_rl else 0.5,
            ecolor=d["color"],
            elinewidth=1.0,
            capsize=2.5,
            zorder=10 if is_rl else 5,
        )
        if label_offsets is not None:
            off = label_offsets.get(d["name"], (0.005, 0.005))
            ax.annotate(
                d["label"],
                (d["e"] + off[0], d["slo"] + off[1]),
                fontsize=fontsize,
                fontweight="bold" if is_rl else "normal",
                color=d["color"] if is_rl else "#222",
            )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_all()

    # ============== Figure: 2 panels side by side ===============
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={"width_ratios": [1, 1.15]})

    # ============== Panel A: full view ===============
    full_offsets = {
        "RoundRobin":   (0.15,  0.005),
        "LeastLoaded":  (0.10,  0.005),
        "ShortestQueue":(0.10, -0.012),
        "HEFT":         (0.10,  0.005),
        "GA":           (0.10, -0.012),
        "PSO":          (0.10,  0.005),
        "A3C_R2N2":     (0.10, -0.012),
        "GNN":          (0.10,  0.005),
        "RL":           (0.15,  0.015),
    }
    draw_points(axA, data, label_offsets=full_offsets, fontsize=9)

    # Show the "Pareto-dominant region" of RL: upper-left quadrant relative to RL
    rl = next(d for d in data if d["name"] == "RL")
    rect = Rectangle(
        (1.5, rl["slo"]), rl["e"] - 1.5, 0.4 - rl["slo"],
        linewidth=0, facecolor="#d62728", alpha=0.08, zorder=1,
    )
    axA.add_patch(rect)

    # "Better →" arrow pointing upper-left
    axA.annotate(
        "", xy=(2.3, 0.37), xytext=(4.5, 0.31),
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.8),
    )
    axA.text(4.6, 0.31, "Better\n(higher SLO,\nlower energy)",
             fontsize=10, color="darkred", fontweight="bold")

    # Annotate the dense-cluster zoom region
    zoom_rect = Rectangle(
        (2.55, 0.18), 3.05 - 2.55, 0.32 - 0.18,
        linewidth=1.2, edgecolor="gray", facecolor="none",
        linestyle="--", zorder=2,
    )
    axA.add_patch(zoom_rect)
    axA.text(2.55, 0.17, "zoom →", fontsize=8, color="gray", style="italic")

    axA.set_xlabel("Energy per Token (J / tok)  →  worse",
                   fontsize=11, fontweight="bold")
    axA.set_ylabel("SLO Attainment  ↑  better",
                   fontsize=11, fontweight="bold")
    axA.set_title("(a) Full view: all 9 schedulers",
                  fontsize=11, pad=10)
    axA.set_xlim(1.8, 8.5)
    axA.set_ylim(0.10, 0.40)
    axA.grid(True, alpha=0.3, linestyle="--")
    axA.set_axisbelow(True)

    # ============== Panel B: zoom into 8 strong schedulers ===============
    data_dense = [d for d in data if d["name"] != "RoundRobin"]
    draw_points(axB, data_dense, label_offsets=ZOOM_LABEL_OFFSET, fontsize=10)

    # Shade Pareto-dominant region in zoom panel too
    rect_z = Rectangle(
        (2.55, rl["slo"]), rl["e"] - 2.55, 0.34 - rl["slo"],
        linewidth=0, facecolor="#d62728", alpha=0.08, zorder=1,
    )
    axB.add_patch(rect_z)

    # Annotate RL's Pareto dominance
    axB.text(2.58, 0.327, "Only RL falls in this region",
             fontsize=9, color="#d62728", fontweight="bold", style="italic")

    axB.set_xlabel("Energy per Token (J / tok)  →  worse",
                   fontsize=11, fontweight="bold")
    axB.set_title("(b) Zoom: 8 strong schedulers (RoundRobin excluded)",
                  fontsize=11, pad=10)
    axB.set_xlim(2.45, 3.10)
    axB.set_ylim(0.18, 0.34)
    axB.grid(True, alpha=0.3, linestyle="--")
    axB.set_axisbelow(True)

    # ============== Super title and save ===============
    fig.suptitle(
        "Figure 4. Pareto comparison of 9 schedulers"
        "  ·  edge=5, λ=2 req/s, N=30 runs, error bars = 95% CI",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "fig4_pareto.png")
    out_pdf = os.path.join(OUTPUT_DIR, "fig4_pareto.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    print("\n=== Numbers (edge=5, λ=2, N=30) ===")
    print(f"{'Scheduler':<15} {'SLO ± CI':>16} {'E/tok ± CI':>16}")
    for d in data:
        print(f"{d['label']:<15} "
              f"{d['slo']:>7.3f} ± {d['slo_ci']:>5.3f}    "
              f"{d['e']:>7.3f} ± {d['e_ci']:>5.3f}")


if __name__ == "__main__":
    main()
