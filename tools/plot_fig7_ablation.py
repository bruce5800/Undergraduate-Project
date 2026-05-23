"""
plot_fig7_ablation.py — Paper Figure 7: Component ablation bar chart

Two side-by-side horizontal-bar panels:
  (a) ΔSLO% from removing each component
  (b) ΔEnergy/token% from removing each component

12 ablation configs sorted by ΔSLO% ascending (largest negative first).

Visual encoding:
  • Red bars: p < 0.05 against Full RL (statistically significant)
  • Gray bars: within noise (p >= 0.05)
  • Light gray ±10% noise band shaded
  • Significance asterisks at bar tips

Data: figs/abl_paper_*/ vs figs/abl_paper_none/  (N=30 each)
Output: figs/report/fig7_ablation.{png,pdf}
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

N_RUNS = 30
EDGE = "5"
TASKS = "100"
OUTPUT_DIR = "figs/report"

# (config_name, display_label, category)
ABLATIONS = [
    ("no_batching",          "− Continuous batching",     "Physical"),
    ("no_pretrain",          "− Pretraining (20→0 ep)",   "PPO"),
    ("no_action_mask",       "− Action mask",             "PPO"),
    ("no_gae",               "− GAE (use MC return)",     "PPO"),
    ("no_entropy",           "− Entropy regularization",  "PPO"),
    ("no_warm_reward",       "− Warm bonus",              "Reward"),
    ("no_batch_reward",      "− Batch bonus",             "Reward"),
    ("no_affinity_reward",   "− Affinity bonus",          "Reward"),
    ("no_cloud_overuse",     "− Cloud-overuse penalty",   "Reward"),
    ("no_all_aigc_rewards",  "− All AIGC rewards",        "Reward (joint)"),
    ("no_aigc_state",        "− AIGC state features",     "State"),
    ("no_aigc_full",         "− All AIGC design",         "Joint"),
]

CATEGORY_COLORS = {
    "Physical":       "#d62728",   # red
    "PPO":            "#1f77b4",   # blue
    "Reward":         "#2ca02c",   # green
    "Reward (joint)": "#7f7f7f",   # gray (joint)
    "State":          "#ff7f0e",   # orange
    "Joint":          "#7f7f7f",   # gray
}


def load_raw(d):
    """Load list of slo_attainment and energy_per_token raw values."""
    slo, e = [], []
    with open(os.path.join(d, "benchmark_raw.csv")) as f:
        for row in csv.DictReader(f):
            if (row["edge_servers"] == EDGE
                    and row["completed_tasks"] == TASKS
                    and row["scheduler"] == "RL"):
                slo.append(float(row["slo_attainment"]))
                e.append(float(row["energy_per_token"]))
    return slo, e


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Full RL baseline
    full_slo, full_e = load_raw("figs/abl_paper_none")
    base_slo = np.mean(full_slo)
    base_e = np.mean(full_e)

    # Compute Δ% and p for each ablation
    rows = []
    for cfg, label, category in ABLATIONS:
        abl_slo, abl_e = load_raw(f"figs/abl_paper_{cfg}")
        abl_slo_mean, abl_e_mean = np.mean(abl_slo), np.mean(abl_e)
        d_slo = (abl_slo_mean - base_slo) / base_slo * 100
        d_e = (abl_e_mean - base_e) / base_e * 100
        _, p_slo = stats.mannwhitneyu(full_slo, abl_slo, alternative="two-sided")
        _, p_e = stats.mannwhitneyu(full_e, abl_e, alternative="two-sided")
        rows.append({
            "cfg": cfg, "label": label, "category": category,
            "d_slo": d_slo, "p_slo": p_slo,
            "d_e": d_e, "p_e": p_e,
        })

    # Sort by ΔSLO% ascending (most negative first)
    rows.sort(key=lambda r: r["d_slo"])

    # ============== Plot ==============
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.5))
    ypos = np.arange(len(rows))

    # ---- Panel (a): ΔSLO% ----
    for i, r in enumerate(rows):
        color = CATEGORY_COLORS[r["category"]] if r["p_slo"] < 0.05 \
            else "#bbbbbb"   # gray if within noise
        ax1.barh(
            i, r["d_slo"], color=color,
            edgecolor="black", linewidth=0.5,
        )
        # Significance + value label at bar tip
        sig = ("***" if r["p_slo"] < 0.001
               else "**" if r["p_slo"] < 0.01
               else "*" if r["p_slo"] < 0.05
               else "")
        offset = 1.5 if r["d_slo"] >= 0 else -1.5
        ha = "left" if r["d_slo"] >= 0 else "right"
        ax1.text(
            r["d_slo"] + offset, i,
            f"{r['d_slo']:+.1f}%{('  ' + sig) if sig else ''}",
            va="center", ha=ha,
            fontsize=9, fontweight="bold" if sig else "normal",
        )

    # Noise band ±10%
    ax1.axvspan(-10, 10, color="lightgray", alpha=0.25, zorder=0,
                label="Within-noise band (±10%)")
    ax1.axvline(0, color="black", linewidth=1.0, zorder=2)

    ax1.set_yticks(ypos)
    ax1.set_yticklabels([r["label"] for r in rows], fontsize=10)
    ax1.invert_yaxis()  # top = most negative
    ax1.set_xlabel("ΔSLO%  vs Full RL  (negative = component was important)",
                   fontsize=11, fontweight="bold")
    ax1.set_title("(a) Impact on SLO Attainment", fontsize=11, pad=10)
    ax1.set_xlim(-105, 18)
    ax1.grid(True, alpha=0.3, axis="x", linestyle="--")
    ax1.set_axisbelow(True)
    ax1.legend(loc="lower left", fontsize=9, framealpha=0.95)

    # ---- Panel (b): ΔEnergy/token% ----
    for i, r in enumerate(rows):
        color = CATEGORY_COLORS[r["category"]] if r["p_e"] < 0.05 \
            else "#bbbbbb"
        ax2.barh(
            i, r["d_e"], color=color,
            edgecolor="black", linewidth=0.5,
        )
        sig = ("***" if r["p_e"] < 0.001
               else "**" if r["p_e"] < 0.01
               else "*" if r["p_e"] < 0.05
               else "")
        offset = 2.0 if r["d_e"] >= 0 else -2.0
        ha = "left" if r["d_e"] >= 0 else "right"
        ax2.text(
            r["d_e"] + offset, i,
            f"{r['d_e']:+.1f}%{('  ' + sig) if sig else ''}",
            va="center", ha=ha,
            fontsize=9, fontweight="bold" if sig else "normal",
        )

    ax2.axvspan(-10, 10, color="lightgray", alpha=0.25, zorder=0)
    ax2.axvline(0, color="black", linewidth=1.0, zorder=2)

    ax2.set_yticks(ypos)
    ax2.set_yticklabels([])   # share with left panel
    ax2.invert_yaxis()
    ax2.set_xlabel("ΔEnergy/token%  vs Full RL  (positive = component was important)",
                   fontsize=11, fontweight="bold")
    ax2.set_title("(b) Impact on Energy per Token", fontsize=11, pad=10)
    ax2.set_xlim(-15, 145)
    ax2.grid(True, alpha=0.3, axis="x", linestyle="--")
    ax2.set_axisbelow(True)

    # Add a manual legend for category colors on the left panel
    handles = []
    for cat, c in CATEGORY_COLORS.items():
        if cat in {"Reward (joint)", "Joint"}:
            continue   # avoid duplicate gray
        handles.append(plt.Rectangle((0, 0), 1, 1, color=c, label=cat))
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#bbbbbb",
                                  label="Within noise (p ≥ 0.05)"))
    ax1.legend(handles=handles + [
        plt.Rectangle((0, 0), 1, 1, color="lightgray", alpha=0.3,
                       label="±10% noise band")
    ], loc="lower left", fontsize=8.5, framealpha=0.95)

    fig.suptitle(
        "Figure 7. Component ablation: only continuous batching and pretraining matter "
        "(N=30, edge=5, λ=2, *** p<0.001)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "fig7_ablation.png")
    out_pdf = os.path.join(OUTPUT_DIR, "fig7_ablation.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}\nSaved: {out_pdf}")

    # Print summary
    print(f"\n=== Fig 7 sorted ablation (Full RL: SLO={base_slo:.3f}, "
          f"E/tok={base_e:.3f}) ===")
    print(f"{'Config':<25} {'Cat':<14} {'ΔSLO%':>9} {'p_SLO':>9} "
          f"{'ΔE/tok%':>9} {'p_E/tok':>9}")
    for r in rows:
        print(f"{r['cfg']:<25} {r['category']:<14} "
              f"{r['d_slo']:>+9.2f} {r['p_slo']:>9.4f} "
              f"{r['d_e']:>+9.2f} {r['p_e']:>9.4f}")


if __name__ == "__main__":
    main()
