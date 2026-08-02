#!/usr/bin/env python3
"""R1 修订: 奖励权重敏感性分析。

汇总 figs/wsens_<preset>/ 五组扫描结果（edge=5, λ=2, N=30, 完成 100 任务），
输出:
  1. 每组 SLO / E-per-token 的 mean±std
  2. 各组 vs canonical 的 Mann-Whitney U 双侧检验
  3. 各组 vs 论文最强基线 (PSO, abl_paper 时代主表数据) 的方向性对比
  4. LaTeX 表格代码（可直接粘进论文）

用法: python3 tools/analyze_weight_sensitivity.py
"""
import csv
import os
import sys

from scipy.stats import mannwhitneyu

ROOT = os.path.join(os.path.dirname(__file__), "..")
PRESETS = ["canonical", "uniform", "time_heavy", "aigc_heavy", "base_heavy"]
WEIGHT_STRS = {
    "canonical":  ".25/.10/.10/.15/.20/.10/.10",
    "uniform":    "1/7 each",
    "time_heavy": ".55/.05/.05/.10/.10/.05/.10",
    "aigc_heavy": ".10/.05/.05/.25/.30/.15/.10",
    "base_heavy": ".40/.20/.20/.05/.05/.05/.05",
}
# 论文 Table III (主对比表) 的最强基线参考点
PSO_SLO, PSO_ETOK = 0.235, 2.81


def load_runs(preset):
    """返回 (slo_list, etok_list)，取 completed_tasks==100 的行。"""
    path = os.path.join(ROOT, "figs", f"wsens_{preset}", "benchmark_raw.csv")
    slo, etok = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["completed_tasks"] == "100" and row["scheduler"] == "RL":
                slo.append(float(row["slo_attainment"]))
                etok.append(float(row["energy_per_token"]))
    return slo, etok


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    data = {p: load_runs(p) for p in PRESETS}
    can_slo, can_etok = data["canonical"]

    print(f"{'preset':<12} {'weights':<28} {'SLO':>14} {'E/tok':>14} "
          f"{'p(SLO)':>8} {'p(E/tok)':>9}  vs PSO")
    print("-" * 100)
    rows = []
    for p in PRESETS:
        slo, etok = data[p]
        if p == "canonical":
            p_slo = p_etok = float("nan")
        else:
            p_slo = mannwhitneyu(slo, can_slo, alternative="two-sided").pvalue
            p_etok = mannwhitneyu(etok, can_etok,
                                  alternative="two-sided").pvalue
        beats_pso = ("Pareto-dom" if mean(slo) > PSO_SLO
                     and mean(etok) < PSO_ETOK
                     else ("SLO only" if mean(slo) > PSO_SLO
                           else ("energy only" if mean(etok) < PSO_ETOK
                                 else "neither")))
        print(f"{p:<12} {WEIGHT_STRS[p]:<28} "
              f"{mean(slo):.3f}±{std(slo):.3f} "
              f"{mean(etok):>7.2f}±{std(etok):.2f} "
              f"{p_slo:>8.3f} {p_etok:>9.3f}  {beats_pso}")
        rows.append((p, mean(slo), std(slo), mean(etok), std(etok),
                     p_slo, p_etok))

    print(f"\n(reference: strongest baseline PSO  SLO={PSO_SLO}, "
          f"E/tok={PSO_ETOK}; N=30 per preset)")

    # ---- LaTeX ----
    print("\n---- LaTeX rows (preset & weights & SLO & E/tok & p vs canonical) ----")
    label = {"canonical": "canonical (paper)", "uniform": "uniform",
             "time_heavy": "time-heavy", "aigc_heavy": "AIGC-heavy",
             "base_heavy": "base-heavy"}
    for p, ms, ss, me, se, pv_s, pv_e in rows:
        pcell = ("---" if p == "canonical"
                 else f"{pv_s:.2f} / {pv_e:.2f}")
        print(f"    {label[p]} & {WEIGHT_STRS[p]} & "
              f"${ms:.3f} \\pm {ss:.3f}$ & ${me:.2f} \\pm {se:.2f}$ & "
              f"{pcell} \\\\")


if __name__ == "__main__":
    main()
