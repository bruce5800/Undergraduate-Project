"""
bench/metrics.py — 指标定义与模型混合预设

从 brenchmark.py 拆分（R1 修订后的仓库整理）。
"""

# ====================================================================
#  指标定义
# ====================================================================
# 两类 workload 共用的基础指标
COMMON_METRICS = ["makespan", "avg_e2e_latency",
                  "avg_utilization", "load_balance_std",
                  # E2: 能耗指标
                  "total_energy_J", "avg_power_W"]

# inference workload 额外追加的 AIGC 标准指标
AIGC_METRICS = [
    "ttft_p50", "ttft_p95", "ttft_p99",
    "tpot_p50", "tpot_p95",
    "goodput_tps",
    "slo_attainment",
    "req_e2e_p50", "req_e2e_p95",
    # E2: 能效指标
    "energy_per_token", "energy_per_request",
]

# inference workload 的模型混合预设（控制工作负载的 AIGC 物理压力）
LLM_MIX_LISTS = {
    "uniform":     ["llama-7b", "llama-13b", "llama-70b"],
    "small-heavy": ["llama-7b"] * 8 + ["llama-13b"] * 2 + ["llama-70b"] * 1,
    "large-heavy": ["llama-7b"] * 1 + ["llama-13b"] * 2 + ["llama-70b"] * 7,
}


