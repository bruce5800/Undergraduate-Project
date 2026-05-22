"""
test_energy_smoke.py — E1: 能耗模型烟雾测试

验证：
  T1. POWER_PROFILES 4 档都正确定义
  T2. instantaneous_power 公式正确（idle + (max-idle) × util）
  T3. step_energy 累加正确（J = W × s）
  T4. infer_power_profile 按 (compute, memory) 正确分档
  T5. Simulation 默认配置后每台 server 有正确 power_profile
  T6. 仿真步运行后能耗实际累加
  T7. 端到端：跑完一个 workload 后 total_energy_J > 0
  T8. 通用 GENERIC 任务（无 model_id）的仿真也能正常累加能耗
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.simulation import Simulation
from environment.server import Server, ServerType
from environment.task import Task, TaskStatus
from environment.energy import (
    POWER_PROFILES, instantaneous_power, step_energy, infer_power_profile,
)


def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b}")


# ====================================================================
# T1. profiles
# ====================================================================

def test_power_profiles_defined():
    for name in ["cloud_A100", "edge_A100", "edge_T4", "edge_Jetson"]:
        assert name in POWER_PROFILES
        p = POWER_PROFILES[name]
        assert "idle_W" in p and "max_W" in p
        assert 0 < p["idle_W"] < p["max_W"]
    # 数值排序：cloud > edge_A100 > edge_T4 > edge_Jetson
    assert POWER_PROFILES["cloud_A100"]["max_W"] > POWER_PROFILES["edge_A100"]["max_W"]
    assert POWER_PROFILES["edge_A100"]["max_W"] > POWER_PROFILES["edge_T4"]["max_W"]
    assert POWER_PROFILES["edge_T4"]["max_W"] > POWER_PROFILES["edge_Jetson"]["max_W"]
    print(f"  [PASS] T1 4 power profiles defined and well-ordered")


# ====================================================================
# T2. instantaneous_power 公式
# ====================================================================

def test_instantaneous_power_formula():
    s = Server(0, ServerType.CLOUD, compute_capacity=200, memory=128,
               storage=500, bandwidth=1000, power_profile="cloud_A100")
    # util = 0
    s.used_compute = 0
    assert_close(instantaneous_power(s), 50.0, msg="idle = 50W")
    # util = 1
    s.used_compute = 200
    assert_close(instantaneous_power(s), 400.0, msg="max = 400W")
    # util = 0.5
    s.used_compute = 100
    assert_close(instantaneous_power(s), 50 + (400 - 50) * 0.5,
                 msg="50% util")
    # 没配 profile 的 server
    s2 = Server(1, ServerType.EDGE, compute_capacity=50, memory=64,
                storage=256, bandwidth=500)   # power_profile=None
    assert_close(instantaneous_power(s2), 0.0, msg="未配 profile → 0")
    print(f"  [PASS] T2 power formula correct")


# ====================================================================
# T3. step_energy 累加 J = W × s
# ====================================================================

def test_step_energy_accumulation():
    s = Server(0, ServerType.CLOUD, compute_capacity=200, memory=128,
               storage=500, bandwidth=1000, power_profile="cloud_A100")
    s.used_compute = 100  # 50% util → power = 225W
    e = step_energy(s, 0.1)
    expected = 225.0 * 0.1
    assert_close(s.accumulated_energy_J, expected, tol=1e-3)
    assert_close(e, expected, tol=1e-3)
    # 再加一步
    step_energy(s, 0.1)
    assert_close(s.accumulated_energy_J, expected * 2, tol=1e-3)
    print(f"  [PASS] T3 step_energy accumulates (J=W·s)")


# ====================================================================
# T4. infer_power_profile 分档
# ====================================================================

def test_infer_power_profile():
    assert infer_power_profile(ServerType.CLOUD, 200, 128) == "cloud_A100"
    assert infer_power_profile(ServerType.EDGE, 50, 64) == "edge_A100"
    assert infer_power_profile(ServerType.EDGE, 20, 32) == "edge_T4"
    assert infer_power_profile(ServerType.EDGE, 10, 16) == "edge_Jetson"
    print(f"  [PASS] T4 profile inference correct")


# ====================================================================
# T5. Simulation 默认配置
# ====================================================================

def test_simulation_default_profiles():
    sim = Simulation(num_servers=8)
    # 期望：server 0 = cloud_A100, 强边 (50T/64GB) = edge_A100, ...
    expected = {
        0: "cloud_A100",   # cloud
        1: "edge_A100",    # 50T 64GB
        2: "edge_T4",      # 20T 32GB
        3: "edge_Jetson",  # 10T 16GB
        4: "edge_A100",
        5: "edge_Jetson",
        6: "edge_T4",
        7: "edge_Jetson",
    }
    for sid, want in expected.items():
        got = sim.servers[sid].power_profile
        assert got == want, f"server {sid}: expected {want}, got {got}"
    print(f"  [PASS] T5 simulation default profiles correct")


# ====================================================================
# T6. 仿真步运行后能耗累加
# ====================================================================

def test_simulation_step_accumulates_energy():
    sim = Simulation(num_servers=5)

    class _NoopSched:
        def schedule(self): pass

    # 空仿真步：所有 server idle，但 idle power > 0，能耗也应累加
    for _ in range(10):
        sim.step(_NoopSched(), current_time=0.0, dt=0.1)

    cloud = sim.servers[0]
    # 10 步 × 0.1s × 50W idle = 50 J
    assert_close(cloud.accumulated_energy_J, 50.0, tol=1e-3,
                 msg="idle cloud 10 步应累 50 J")
    print(f"  [PASS] T6 sim step accumulates idle energy correctly")


# ====================================================================
# T7. 端到端能耗 > 0
# ====================================================================

def test_e2e_energy_positive():
    random.seed(0)
    sim = Simulation(num_servers=5)
    tasks = Task.generate_inference_workload(
        num_requests=5, task_id_offset=0, rng=random.Random(0))
    sim.add_tasks(tasks)

    # 用最简单调度器跑完
    from scheduler.LeastLoadedScheduler import LeastLoadedScheduler
    sched = LeastLoadedScheduler(sim)
    t = 0.0
    while len(sim.completed_tasks) < len(sim.tasks) and t < 500:
        sim.step(sched, t, dt=0.1)
        t += 0.1

    total = sum(s.accumulated_energy_J for s in sim.servers.values())
    assert total > 0, "端到端跑完应累计能耗 > 0"
    print(f"  [PASS] T7 e2e total_energy_J = {total:.1f} J after {t:.1f}s")


# ====================================================================
# T8. 通用 DAG 也能跑（向后兼容）
# ====================================================================

def test_generic_dag_energy():
    random.seed(0)
    sim = Simulation(num_servers=3)
    tasks = Task.generate_single_dag(0, 10)
    sim.add_tasks(tasks)

    from scheduler.LeastLoadedScheduler import LeastLoadedScheduler
    sched = LeastLoadedScheduler(sim)
    t = 0.0
    while len(sim.completed_tasks) < len(sim.tasks) and t < 200:
        sim.step(sched, t, dt=0.1)
        t += 0.1

    total = sum(s.accumulated_energy_J for s in sim.servers.values())
    assert total > 0
    print(f"  [PASS] T8 generic DAG energy = {total:.1f} J (backward compat OK)")


# ====================================================================
# main
# ====================================================================

if __name__ == "__main__":
    tests = [
        test_power_profiles_defined,
        test_instantaneous_power_formula,
        test_step_energy_accumulation,
        test_infer_power_profile,
        test_simulation_default_profiles,
        test_simulation_step_accumulates_energy,
        test_e2e_energy_positive,
        test_generic_dag_energy,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print()
    if failed == 0:
        print(f"All {len(tests)} tests passed.")
        sys.exit(0)
    else:
        print(f"{failed}/{len(tests)} tests failed.")
        sys.exit(1)
