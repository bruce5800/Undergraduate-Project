"""
benchmark.py — Cloud-Edge Scheduling Benchmark CLI 入口

（原 brenchmark.py，R1 修订后的仓库整理中更名并拆分：
 指标定义在 bench/metrics.py，核心测试类在 bench/tester.py。
 brenchmark.py 保留为兼容垫片，历史 run_manifest.json 中记录的
 命令行仍可原样复现。）

Design: Checkpoint mode + Multi-run statistics
  - Each run executes ONE full simulation (e.g. 300 tasks)
  - Metrics are recorded at checkpoints (every N completed tasks)
  - The same simulation is repeated M times with different random seeds
  - Statistics (mean, std, 95% CI) are computed across runs
  - Mann-Whitney U tests compare schedulers at each checkpoint

Output files (under --out DIR):
  benchmark_raw.csv       — every checkpoint of every run
  benchmark_summary.csv   — mean / std / 95% CI per checkpoint
  statistical_tests.csv   — pairwise Mann-Whitney U test results
  run_manifest.json       — exact CLI, git commit, seeds
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

from bench.metrics import LLM_MIX_LISTS
from bench.tester import BenchmarkTester


# ====================================================================
#  CLI entry
# ====================================================================

def main():
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
        "--reward-weights", type=str, default="canonical",
        choices=list(BenchmarkTester.REWARD_WEIGHT_PRESETS.keys()),
        help="R1 revision: reward-weight preset for the RL scheduler "
             "(canonical / uniform / time_heavy / aigc_heavy / "
             "base_heavy). Only affects RLScheduler.")
    parser.add_argument(
        "--rl-only", action="store_true",
        help="Run only the RL scheduler (skip other 7 baselines). "
             "Useful for fast ablation studies focused on RL component analysis.")
    parser.add_argument(
        "--schedulers", type=str, default=None,
        help="Comma-separated scheduler subset to run, e.g. "
             "--schedulers CS_UCB or --schedulers PSO,RL. "
             "Default: the standard 8-scheduler set (RL included).")
    parser.add_argument(
        "--csucb-delta", type=float, default=1.0,
        help="CS-UCB exploration coefficient delta (default 1.0)")
    parser.add_argument(
        "--csucb-lam", type=float, default=1.0,
        help="CS-UCB constraint-satisfaction weight lambda (default 1.0)")
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
        "--llm-mix", type=str, default="uniform",
        choices=list(LLM_MIX_LISTS.keys()),
        help="LLM mix preset for inference workload: "
             "'uniform' (default) / 'small-heavy' (mostly 7b) / "
             "'large-heavy' (mostly 70b, stress AIGC physics).")
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
    n_schedulers = 1 if args.rl_only else 9   # +GNN
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
        "reward_weights_preset": args.reward_weights,
        "reward_weights": BenchmarkTester.REWARD_WEIGHT_PRESETS.get(
            args.reward_weights),
        "ttft_slo":    args.ttft_slo if args.workload == "inference" else None,
        "tpot_slo":    args.tpot_slo if args.workload == "inference" else None,
        "trace_preset": args.trace_preset if args.workload == "inference" else None,
        "arrival_rate": args.arrival_rate if args.workload == "inference" else None,
        "quick":       args.quick,
        "schedulers":  (["RL"] if args.rl_only else
                        ([s.strip() for s in args.schedulers.split(",")]
                         if args.schedulers else
                         ["RoundRobin", "LeastLoaded", "ShortestQueue",
                          "HEFT", "GA", "PSO", "A3C_R2N2", "RL"])),
        "csucb_kwargs": {"delta": args.csucb_delta, "lam": args.csucb_lam},
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
                              arrival_rate=args.arrival_rate,
                              llm_mix=args.llm_mix,
                              reward_weights_preset=args.reward_weights)

    tester.csucb_kwargs = {"delta": args.csucb_delta,
                           "lam": args.csucb_lam}

    if args.rl_only:
        tester.schedulers = {"RL": tester.schedulers["RL"]}
        print(f"  --rl-only: 只跑 RL 调度器（其他 baseline 跳过）")
    elif args.schedulers:
        wanted = [s.strip() for s in args.schedulers.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in tester.schedulers]
        if unknown:
            print(f"ERROR: unknown scheduler(s): {unknown}; "
                  f"valid: {list(tester.schedulers.keys())}")
            sys.exit(1)
        tester.schedulers = {k: tester.schedulers[k] for k in wanted}
        print(f"  --schedulers: {wanted}")
    raw, summary = tester.run_benchmark(
        edge_counts, total_tasks=total_tasks,
        checkpoint_interval=interval,
        output_dir=output_dir)

    print("\n" + "=" * 70)
    print(f"  Done! Output files in {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
