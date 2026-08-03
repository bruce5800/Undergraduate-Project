# Experiment Data Index (Reproducibility Manifest)

Every directory below contains `benchmark_raw.csv` (per-run rows),
`benchmark_summary.csv` (mean/std/CI aggregates), and
`run_manifest.json` recording the **exact CLI invocation, git commit,
and seeds** that produced it. All experiments use fixed base seed 42;
re-running the recorded CLI reproduces the numbers exactly.

**Canonical configuration** (unless a directory name says otherwise):
`--workload inference --trace-preset lognormal --arrival-rate 2.0
--runs 30 --tasks 100 --edge 5` (1 cloud + 5 heterogeneous edge
servers, Poisson λ=2 req/s, N=30 seeds).

## Paper table → data directory

| Paper artifact | Data directories |
|---|---|
| Table III (main comparison, 11 schedulers) | `energy_scan2` (RR/LL/SQ/HEFT/GA/PSO/A3C/GNN/RL) + `csucb_edge5` + `nsga_edge5` |
| Table IV (Mann-Whitney significance) | same three directories, tests on `benchmark_raw.csv` |
| Table V (topology sensitivity) | `exp2_edge3`, `exp2_edge7` (+ edge5 from Table III dirs); `csucb_edge{3,7}`, `nsga_edge{3,7}` |
| Tables VI–VII (load sweep, SLO and E/tok) | `exp2_lambda{0.5,1.0,4.0,8.0}` (+ λ=2 from Table III dirs); `csucb_lambda*`, `nsga_lambda*` |
| Table VIII (workload mix) | `exp2_largeheavy`, `csucb_largeheavy`, `nsga_largeheavy` (+ uniform from Table III dirs) |
| Table IX (12-component ablation) | `abl_paper_none` (Full RL) + `abl_paper_<component>` ×12 |
| Table X (reward-weight sensitivity) | `wsens_{canonical,uniform,time_heavy,aigc_heavy,base_heavy}` |
| §V-B RL-vs-NSGA-II sweet-spot p-values | `rl_lambda0.5`, `rl_lambda1.0` vs `nsga_lambda{0.5,1.0}` |

Note: `abl_paper_none` is the canonical Full-RL run quoted throughout
the paper (SLO 0.302, E/tok 2.61); `wsens_canonical` is its exact
reproduction under the parameterized-reward code path.

## Paper figure → script → data

Script filenames predate the paper's final figure numbering (the
paper's Figs. 1–4 are drawio diagrams; results figures are offset).

| Paper figure | Script (in `tools/`) | Reads | Writes |
|---|---|---|---|
| Fig. 5 (Pareto plane) | `plot_fig4_pareto.py` | `energy_scan2`, `csucb_edge5`, `nsga_edge5` | `report/fig4_pareto.{pdf,png}` |
| Fig. 6 (topology) | `plot_fig5_topology.py` | `exp2_edge{3,7}`, `energy_scan2` | `report/fig5_topology.{pdf,png}` |
| Fig. 7 (load) | `plot_fig6_load.py` | `exp2_lambda*`, `energy_scan2` | `report/fig6_load.{pdf,png}` |
| Fig. 8 (ablation) | `plot_fig7_ablation.py` | `abl_paper_*` | `report/fig7_ablation.{pdf,png}` |
| Table X aggregation | `analyze_weight_sensitivity.py` | `wsens_*` | stdout (incl. LaTeX rows) |

Generated figures are copied into `paper/figures/` for the LaTeX
build.

## Baseline-specific directories

- `csucb_*` — CS-UCB (PerLLM-style constrained bandit), δ=0.5, λ=1.0
  (grid-tuned; see `run_manifest.json` → `csucb_kwargs`).
- `nsga_*` — NSGA-II multi-objective baseline (population 30, 30
  generations per window, knee-point execution).

## `_archive/`

Superseded or diagnostic runs kept for provenance, **not** used by
any paper number: early ablation eras (`abl_*`, `ablN30_*`),
F-patch diagnostics (`F12_*`, `diagB*`), earlier main-experiment
iterations (`exp1_main_v1..v3`, `energy_scan`), pre-energy-metric
load sweeps (`sens_lambda*`), milestone sanity checks (`m1_aigc`,
`m2_sanity`), and GNN development runs (`gnn_compare_*`). Their
manifests record which code era produced them.
