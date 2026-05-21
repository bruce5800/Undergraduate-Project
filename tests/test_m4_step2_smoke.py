"""
test_m4_step2_smoke.py — M4 step 2: Poisson arrival + lognormal trace + memory floor

验证：
  T1. arrival_time 默认为 0 → 向后兼容
  T2. arrival_time > current_time 时 task 保持 WAITING；时间到了才 READY
  T3. 工厂支持 arrival_rate（Poisson）：到达时间单调递增、均值 ≈ N/rate
  T4. 工厂支持 dist="lognormal"：prompt/output 分布长尾、范围合理
  T5. ModelSpec 各 LLM 模型有正的 decode/prefill floor
  T6. Server.process_tasks 应用 floor：低 workload 时 exec_time = floor × tokens
  T7. floor 不影响 generic 任务（无 model_id 走原逻辑）

直接 python tests/test_m4_step2_smoke.py 运行。
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from environment.simulation import Simulation
from environment.server import Server, ServerType
from environment.task import Task, TaskKind, TaskStatus
from environment.model_catalog import CATALOG


def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b}")


# ====================================================================
# T1. 默认 arrival_time = 0 → 向后兼容
# ====================================================================

def test_arrival_time_default_zero():
    t = Task(task_id=0, compute_demand=5, workload=200, input_size=2,
             output_size=0.1, dependencies=[])
    assert hasattr(t, "arrival_time"), "Task 应该有 arrival_time 字段"
    assert t.arrival_time == 0.0, f"默认应为 0.0，got {t.arrival_time}"
    print("  [PASS] T1 default arrival_time = 0.0")


# ====================================================================
# T2. arrival_time gate
# ====================================================================

def test_arrival_time_gates_ready():
    sim = Simulation(num_servers=2)
    t = Task(task_id=0, compute_demand=5, workload=200, input_size=2,
             output_size=0.1, dependencies=[],
             arrival_time=5.0)  # 直到 t=5 才到达
    sim.add_tasks([t])

    # t=0：还没到，应保持 WAITING
    class _NoopSched:
        def schedule(self): pass
    sim.step(_NoopSched(), current_time=0.0)
    assert t.status == TaskStatus.WAITING, \
        f"t<arrival_time 时应保持 WAITING，got {t.status}"

    # t=3：还是没到
    sim.step(_NoopSched(), current_time=3.0)
    assert t.status == TaskStatus.WAITING

    # t=5.0：到了，应转为 READY
    sim.step(_NoopSched(), current_time=5.0)
    assert t.status == TaskStatus.READY, \
        f"current_time >= arrival_time 时应 READY，got {t.status}"
    assert_close(t.ready_time, 5.0, msg="ready_time 记录在到达时刻")
    print("  [PASS] T2 arrival_time gates READY transition")


# ====================================================================
# T3. Poisson 到达序列
# ====================================================================

def test_poisson_arrival_sequence():
    rng = random.Random(0)
    N = 200
    rate = 10.0  # 10 req/s
    tasks = Task.generate_inference_workload(
        num_requests=N, task_id_offset=0, rng=rng,
        arrival_rate=rate)
    # 取所有 prefill 的 arrival_time（每请求有 2 个 task 共享 arrival）
    prefill_arrivals = [t.arrival_time for t in tasks
                        if t.kind == TaskKind.PREFILL]
    assert len(prefill_arrivals) == N

    # 单调递增
    for a, b in zip(prefill_arrivals, prefill_arrivals[1:]):
        assert b >= a, f"arrival_time 应单调递增"

    # 大数定律：均值 ≈ N / (2 × rate)（因为 N 个累积 → 总时长 ≈ N/rate, 均值 = 总/2）
    mean_arrival = np.mean(prefill_arrivals)
    expected = N / rate / 2  # 累积均匀样本的均值近似
    relative = abs(mean_arrival - expected) / expected
    assert relative < 0.3, \
        f"均值 {mean_arrival:.2f} 偏离预期 {expected:.2f} 太多 (rel={relative:.2%})"
    print(f"  [PASS] T3 Poisson arrival: N={N}, rate={rate}, "
          f"mean_arrival={mean_arrival:.2f}s (expected ~{expected:.1f})")


# ====================================================================
# T4. lognormal 分布
# ====================================================================

def test_lognormal_dist():
    rng = random.Random(1)
    N = 500
    tasks = Task.generate_inference_workload(
        num_requests=N, task_id_offset=0, rng=rng,
        dist="lognormal")
    prefills = [t for t in tasks if t.kind == TaskKind.PREFILL]
    prompts = [t.prompt_tokens for t in prefills]
    decodes = [t for t in tasks if t.kind == TaskKind.DECODE]
    outputs = [t.output_tokens for t in decodes]

    # 中位数应该在 lognormal 设定的范围内：
    #   prompt μ=5.5 → median ≈ exp(5.5) ≈ 245
    #   output μ=4.5 → median ≈ exp(4.5) ≈ 90
    p_med = np.median(prompts)
    o_med = np.median(outputs)
    # 允许 ±50% 偏离（采样波动）
    assert 100 < p_med < 600, f"prompt median {p_med} 不在 [100, 600]"
    assert 30 < o_med < 200, f"output median {o_med} 不在 [30, 200]"

    # 长尾：P99 应明显高于 P50
    p_p99 = np.percentile(prompts, 99)
    assert p_p99 > p_med * 3, \
        f"P99 prompt {p_p99} 应 > 3× median {p_med}（验证长尾）"

    # 上下界约束生效
    assert min(prompts) >= 16
    assert max(prompts) <= 4096
    assert min(outputs) >= 10
    assert max(outputs) <= 2000
    print(f"  [PASS] T4 lognormal dist: "
          f"prompt median={p_med:.0f} P99={p_p99:.0f}, output median={o_med:.0f}")


# ====================================================================
# T5. ModelSpec floor 存在
# ====================================================================

def test_model_floor_calibrated():
    for mid in ["llama-7b", "llama-13b", "llama-70b"]:
        spec = CATALOG[mid]
        assert spec.decode_floor_sec_per_token > 0, \
            f"{mid} decode_floor 应 > 0"
        assert spec.prefill_floor_sec_per_token > 0, \
            f"{mid} prefill_floor 应 > 0"
        # 大模型 floor 应更大
        if mid == "llama-7b":
            assert spec.decode_floor_sec_per_token < 0.025
        if mid == "llama-70b":
            assert spec.decode_floor_sec_per_token > 0.05
    print("  [PASS] T5 model floors calibrated")


# ====================================================================
# T6. Server 应用 floor
# ====================================================================

def test_server_applies_decode_floor():
    """在算力极大的云上跑 decode，exec_time 应被 floor 拉到至少 floor × output_tokens。"""
    s = Server(server_id=0, server_type=ServerType.CLOUD,
               compute_capacity=10000.0,  # 故意夸张大，确保算力不是瓶颈
               memory=128.0, storage=500.0, bandwidth=1000.0)
    s.loaded_models["llama-7b"] = 0.0
    s.weight_vram_used = CATALOG["llama-7b"].weights_GB

    pair = Task.generate_inference_request(0, 0, "llama-7b",
                                            prompt_tokens=200,
                                            output_tokens=100)
    pair[0].status = TaskStatus.COMPLETED  # 略过 prefill
    decode = pair[1]
    decode.status = TaskStatus.READY

    s.add_task(decode, priority=1.0)
    s.process_tasks(current_time=0.0)
    assert decode.status == TaskStatus.RUNNING

    actual_dur = decode.end_time - decode.start_time
    expected_floor = CATALOG["llama-7b"].decode_floor_sec_per_token * 100  # 0.02 × 100 = 2.0s
    # 因 batch=1, floor 主导
    assert_close(actual_dur, expected_floor, tol=1e-3,
                 msg=f"decode 应被 floor 拉到 {expected_floor}s")
    print(f"  [PASS] T6 decode floor applied: "
          f"exec={actual_dur:.3f}s (floor={expected_floor:.3f}s, "
          f"compute-based would be {decode.workload/10000.0:.5f}s)")


# ====================================================================
# T7. floor 不影响 generic 任务
# ====================================================================

def test_floor_skips_generic():
    s = Server(server_id=0, server_type=ServerType.CLOUD,
               compute_capacity=100.0, memory=128.0,
               storage=500.0, bandwidth=1000.0)
    t = Task(task_id=0, compute_demand=5, workload=200, input_size=2,
             output_size=0.1, dependencies=[])
    t.status = TaskStatus.READY
    s.add_task(t, priority=1.0)
    s.process_tasks(current_time=0.0)

    expected = 200.0 / 100.0  # workload / compute
    actual = t.end_time - t.start_time
    assert_close(actual, expected, tol=1e-6,
                 msg="generic 任务应严格走 workload/compute")
    print(f"  [PASS] T7 generic skips floor: exec={actual}s")


# ====================================================================
# main
# ====================================================================

if __name__ == "__main__":
    tests = [
        test_arrival_time_default_zero,
        test_arrival_time_gates_ready,
        test_poisson_arrival_sequence,
        test_lognormal_dist,
        test_model_floor_calibrated,
        test_server_applies_decode_floor,
        test_floor_skips_generic,
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
