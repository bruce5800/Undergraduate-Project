"""
environment/energy.py — 服务器能耗模型

每台服务器对应一个 power_profile：(idle_W, max_W)。
瞬时功率 = idle_W + (max_W − idle_W) × compute_util
能耗  E = ∫ 瞬时功率 dt

数值参考 NVIDIA 公开 TDP，idle 按典型比例（10–20%）估算：

| Profile      | Idle (W) | TDP (W) | Real-world GPU |
|--------------|---------:|--------:|----------------|
| cloud_A100   |    50    |   400   | A100 80GB SXM4 |
| edge_A100    |    30    |   250   | A100 40GB PCIe |
| edge_T4      |    20    |    70   | T4 PCIe        |
| edge_Jetson  |    10    |    30   | Jetson AGX Orin|

设计目标：让 power 与 compute_capacity **非单调相关**（T4 是 70W / 20 TFLOPS
的"能效甜点"），从而给 RL 提供独立于基础特征的优化维度，破解 cloud collapse。
"""

POWER_PROFILES = {
    "cloud_A100":  {"idle_W": 50,  "max_W": 400},
    "edge_A100":   {"idle_W": 30,  "max_W": 250},
    "edge_T4":     {"idle_W": 20,  "max_W":  70},
    "edge_Jetson": {"idle_W": 10,  "max_W":  30},
}


def instantaneous_power(server) -> float:
    """瞬时功率 (W) = idle + (max − idle) × compute_util。

    未配置 power_profile 时返回 0（向后兼容）。
    """
    profile = getattr(server, "power_profile", None)
    if profile is None or profile not in POWER_PROFILES:
        return 0.0
    p = POWER_PROFILES[profile]
    util = server.used_compute / max(server.total_compute, 1e-6)
    util = min(max(util, 0.0), 1.0)
    return p["idle_W"] + (p["max_W"] - p["idle_W"]) * util


def step_energy(server, dt: float) -> float:
    """累加 dt 秒的能耗到 server.accumulated_energy_J，返回这次累加的焦耳数。

    J = W × s。Server 未配置 power_profile 时 no-op。
    """
    if getattr(server, "power_profile", None) is None:
        return 0.0
    energy = instantaneous_power(server) * dt
    server.accumulated_energy_J = getattr(server, "accumulated_energy_J", 0.0) + energy
    return energy


# ================================================================
# 按 (compute, memory) 自动推断 profile —— 跟 simulation.py 的默认配置对应
# ================================================================

def infer_power_profile(server_type, compute, memory) -> str:
    """根据 server 规格自动推断 profile 名。"""
    # 云服务器：profile cloud_A100
    if str(server_type).endswith("CLOUD"):
        return "cloud_A100"
    # 边缘：按 (compute, memory) 区分
    if compute >= 40 and memory >= 64:
        return "edge_A100"   # 强边缘
    if compute >= 15 and memory >= 24:
        return "edge_T4"     # 中等边缘
    return "edge_Jetson"     # 弱边缘
