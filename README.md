# AIGC Cloud-Edge Inference Scheduling Simulator

Simulation platform for **cloud-edge LLM inference scheduling**,
accompanying the paper *"Pareto-Dominant Reinforcement Learning for
Cloud-Edge LLM Inference Scheduling"*. It models the five physical
characteristics that distinguish LLM/AIGC inference from generic DAG
workloads, and benchmarks 11 schedulers — including an AIGC-aware PPO
scheduler — on joint SLO-attainment / energy-per-token objectives.

## Modeled AIGC physics (all implemented)

| Physics | Milestone | Notes |
|---|---|---|
| Model-weight residency + cold load + LRU eviction | M1 | 5–25 s cold loads, calibrated to NVMe read |
| Prefill/decode two-phase lifecycle | M2 | each request = 2-node DAG with shared KV state |
| Cross-server KV-cache migration cost | M2 | KV size charged through the transfer-time model |
| KV cache VRAM accounting | M2 | per-token KV occupies memory during execution |
| Continuous batching (admission-time) | M3 | β\_prefill=0.30, β\_decode=0.05, matched to vLLM |
| Memory-bandwidth decode floor | M4 | 20–80 ms/token by model class |
| Poisson arrivals + log-normal lengths | M4 | calibrated to Azure LLM Inference Trace 2023 |
| Energy model (idle/peak per tier) | E1–E2 | A100/T4/Jetson profiles; J/token as first-class metric |

## Repository layout

```
.
├── benchmark.py            # Benchmark CLI entry (multi-seed + statistics)
├── brenchmark.py           # Compatibility shim for the historical entry name
│                           #   (run_manifest.json files record it; both work)
├── bench/                  # Benchmark framework package
│   ├── metrics.py          #   metric definitions + LLM mix presets
│   └── tester.py           #   BenchmarkTester (runs, checkpoints, stats)
├── environment/            # Simulator core
│   ├── task.py             #   PREFILL/DECODE/GENERIC tasks, KV cache
│   ├── server.py           #   residency, batching, cold load, admission
│   ├── network.py          #   latency matrix + bandwidth
│   ├── energy.py           #   power profiles + energy integration
│   ├── simulation.py       #   0.1 s discrete-time main loop
│   └── model_catalog.py    #   llama-7b/13b/70b-int8, sdxl specs
├── scheduler/              # 11 schedulers, one file each
│   ├── RRscheduler / LeastLoaded / ShortestQueue   (greedy)
│   ├── Heftscheduler                               (list scheduling)
│   ├── GAscheduler / PSOscheduler                  (metaheuristics)
│   ├── NSGAIIScheduler                             (multi-objective GA, knee point)
│   ├── PerLLMScheduler                             (CS-UCB constrained bandit)
│   ├── A3CR2NScheduler / GNNScheduler              (DRL baselines)
│   └── RLscheduler                                 (ours: AIGC-aware PPO)
├── figs/                   # Experiment data — see figs/README.md for the
│                           #   paper-table ↔ data-directory reproducibility map
├── tools/                  # Paper figure generation + analysis scripts
├── tests/                  # Per-milestone smoke tests (11 files)
├── demos/                  # Interactive demos (animation, AIGC physics)
├── paper/  paper-zh/       # LaTeX sources (English / Chinese)
├── submission/             # Cover letter, highlights, venue research
└── docs/                   # Design notes
```

## Schedulers

| Scheduler | Family | AIGC-aware |
|---|---|---|
| RoundRobin / LeastLoaded / ShortestQueue | greedy | ✗ |
| HEFT | list scheduling | ✗ |
| GA / PSO | metaheuristic | ✗ |
| NSGA-II | multi-objective metaheuristic (knee-point execution) | ✗ |
| CS-UCB | PerLLM-style constrained bandit | ✗ |
| A3C-R2N2 | DRL (GRU + residual) | ✗ |
| GNN | DRL (graph attention, AIGC edge features) | partial |
| **RL (ours)** | PPO + action masking + GAE + pretrain | ✓ state & reward |

RL extras: 12 ablation switches (`--ablation`, incl. `no_batching`,
`no_pretrain`, `no_all_aigc_rewards`, `no_aigc_full`, …) and 5
reward-weight presets (`--reward-weights canonical / uniform /
time_heavy / aigc_heavy / base_heavy`).

## Quick start

```bash
pip install torch numpy scipy matplotlib

# Canonical paper configuration (Table III conditions, RL only, ~10 s)
python3 benchmark.py --workload inference --trace-preset lognormal \
    --arrival-rate 2.0 --runs 30 --tasks 100 --edge 5 --rl-only \
    --out figs/my_run

# Any scheduler subset
python3 benchmark.py --workload inference --trace-preset lognormal \
    --arrival-rate 2.0 --runs 30 --tasks 100 --edge 5 \
    --schedulers PSO,NSGA_II,CS_UCB,RL --out figs/my_comparison

# Smoke tests
for t in tests/test_*.py; do python3 "$t"; done
```

Outputs per `--out` directory: `benchmark_raw.csv` (every checkpoint of
every run), `benchmark_summary.csv` (mean/std/95% CI),
`statistical_tests.csv` (pairwise Mann-Whitney U), and
`run_manifest.json` (exact CLI + git commit + seeds). Existing outputs
are never silently overwritten (`--force` to override). All runs use
fixed base seed 42 and are byte-for-byte reproducible.

## Reproducing the paper

Every table and figure maps to a data directory and a generating
command — see **[figs/README.md](figs/README.md)**. Paper figures are
regenerated by `tools/plot_fig*.py`; the reward-weight sensitivity
table by `tools/analyze_weight_sensitivity.py`.

## Demos

```bash
python3 demos/aigc_demo.py         # generic vs AIGC physics comparison
python3 demos/animation_demo.py    # live matplotlib animation of a run
```
