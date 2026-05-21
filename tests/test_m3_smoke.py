"""
test_m3_smoke.py — M3 step 1: Static continuous batching

验证：
  T1. 通用任务无 batching：单任务 batch=1, exec_time 与 M2 完全一致
  T2. 同模型同阶段（decode）：第二个任务获得 batch_size=2，exec 略增
  T3. 不同模型隔离：A 模型 decode 与 B 模型 decode 不混批
  T4. 不同阶段隔离：同模型的 PREFILL 与 DECODE 不混批
  T5. Max batch saturation：超过 max_batch_size 的请求 can_allocate=False
  T6. 吞吐速比：8 个同模 decode 同时启动，总 wall time ≈ T×1.35（5.9× speedup）
  T7. batch_size_at_admit 字段正确写入

直接 python test_m3_smoke.py 运行。
"""

import os
import sys

# 让从子目录运行也能找到 environment / scheduler 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.server import (
    Server, ServerType,
    PREFILL_BATCH_OVERHEAD, DECODE_BATCH_OVERHEAD,
)
from environment.task import Task, TaskKind, TaskStatus
from environment.model_catalog import CATALOG


def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b}")


def fresh_server(memory=128.0, compute=100.0) -> Server:
    return Server(server_id=0, server_type=ServerType.CLOUD,
                  compute_capacity=compute, memory=memory,
                  storage=500.0, bandwidth=1000.0)


# ====================================================================
# T1. Generic 任务无 batching（M3 零回归）
# ====================================================================

def test_generic_no_batching():
    s = fresh_server()
    t = Task(task_id=0, compute_demand=5.0, workload=200.0,
             input_size=1.0, output_size=0.1, dependencies=[])
    s.add_task(t, priority=1.0)
    s.process_tasks(current_time=0.0)
    assert t.batch_size_at_admit == 1, "generic 任务 batch=1"
    # exec = workload/compute = 200/100 = 2.0，无 batching 修正
    assert_close(t.end_time - t.start_time, 2.0,
                 msg="generic 任务执行时长未受 batching 干扰")
    print("  [PASS] T1 generic no batching")


# ====================================================================
# T2. 同模同阶段 → batch
# ====================================================================

def test_same_model_same_kind_batches():
    s = fresh_server()
    pair_a = Task.generate_inference_request(0, 0, "llama-7b", 256, 100)
    pair_b = Task.generate_inference_request(1, 2, "llama-7b", 256, 100)
    # 直接拿 decode（也可以是 prefill，效果对称，只是 overhead 不同）
    decode_a = pair_a[1]
    decode_b = pair_b[1]
    # 先把 prefill 标记为 COMPLETED（模拟已完成），让 decode 可以单独跑
    for prefill in [pair_a[0], pair_b[0]]:
        prefill.status = TaskStatus.COMPLETED
    decode_a.status = TaskStatus.READY
    decode_b.status = TaskStatus.READY

    s.add_task(decode_a, priority=1.0)
    s.process_tasks(current_time=0.0)
    assert decode_a.batch_size_at_admit == 1
    solo_dur = decode_a.end_time - decode_a.start_time

    s.add_task(decode_b, priority=1.0)
    s.process_tasks(current_time=0.0)
    assert decode_b.batch_size_at_admit == 2, \
        f"第二个同模 decode batch 应为 2，got {decode_b.batch_size_at_admit}"
    batched_dur = decode_b.end_time - decode_b.start_time
    expected = solo_dur * (1.0 + 1 * DECODE_BATCH_OVERHEAD)
    assert_close(batched_dur, expected,
                 msg="batch=2 时 decode 时长应为 solo×(1+overhead)")
    print(f"  [PASS] T2 same-model decode batches "
          f"(solo={solo_dur:.4f}s → batch2={batched_dur:.4f}s)")


# ====================================================================
# T3. 不同模型不混批
# ====================================================================

def test_different_models_isolated():
    s = fresh_server()
    decode_7b = Task.generate_inference_request(0, 0, "llama-7b", 256, 100)[1]
    decode_13b = Task.generate_inference_request(1, 2, "llama-13b", 256, 100)[1]
    decode_7b.status = TaskStatus.READY
    decode_13b.status = TaskStatus.READY
    # 先把对应的 prefill 标 COMPLETED 以满足依赖
    Task.generate_inference_request  # noqa: just to remind

    s.add_task(decode_7b, priority=1.0)
    s.process_tasks(current_time=0.0)
    s.add_task(decode_13b, priority=1.0)
    s.process_tasks(current_time=0.0)
    assert decode_7b.batch_size_at_admit == 1
    assert decode_13b.batch_size_at_admit == 1, \
        "不同模型应该各自 batch=1"
    print("  [PASS] T3 different models isolated")


# ====================================================================
# T4. 不同阶段不混批
# ====================================================================

def test_different_kinds_isolated():
    s = fresh_server()
    pair_a = Task.generate_inference_request(0, 0, "llama-7b", 256, 100)
    pair_b = Task.generate_inference_request(1, 2, "llama-7b", 256, 100)
    # A 的 prefill 仍在跑（不标 COMPLETED）；B 的 decode 通过依赖伪造为 ready
    pair_b[0].status = TaskStatus.COMPLETED
    pair_b[1].status = TaskStatus.READY

    s.add_task(pair_a[0], priority=1.0)  # A 的 prefill
    s.process_tasks(current_time=0.0)
    s.add_task(pair_b[1], priority=1.0)  # B 的 decode
    s.process_tasks(current_time=0.0)
    # 同模型，但一 PREFILL 一 DECODE → 都该是 batch=1
    assert pair_a[0].batch_size_at_admit == 1
    assert pair_b[1].batch_size_at_admit == 1, \
        "prefill 和 decode 不能混批"
    print("  [PASS] T4 different kinds isolated")


# ====================================================================
# T5. Max batch saturation
# ====================================================================

def test_max_batch_size_blocks_admission():
    s = fresh_server()
    spec = CATALOG["llama-70b"]
    max_b = spec.max_batch_size  # 8
    # 先把 70b 加载好（避免冷加载干扰）—— 简化：直接放进 loaded_models
    s.loaded_models["llama-70b"] = 0.0
    s.weight_vram_used = spec.weights_GB

    decodes = []
    for i in range(max_b + 2):  # 比上限多 2 个
        pair = Task.generate_inference_request(
            i, 2 * i, "llama-70b", 256, 100)
        pair[0].status = TaskStatus.COMPLETED
        d = pair[1]
        d.status = TaskStatus.READY
        decodes.append(d)

    for d in decodes:
        s.add_task(d, priority=1.0)
    s.process_tasks(current_time=0.0)

    running = [d for d in decodes if d.status == TaskStatus.RUNNING]
    blocked = [d for d in decodes if d.status == TaskStatus.QUEUED]
    assert len(running) == max_b, \
        f"应 running {max_b} 个（达 max_batch），实际 {len(running)}"
    assert len(blocked) == 2, \
        f"剩 2 个应被 batch slot 阻塞，实际 {len(blocked)}"
    print(f"  [PASS] T5 max_batch={max_b} blocks excess admissions")


# ====================================================================
# T6. 吞吐加速比
# ====================================================================

def test_throughput_speedup():
    """8 个同模 decode 同时启动，总 wall time ≈ T_solo × (1+7×0.05) = 1.35T。
    串行执行时 wall time = 8T，吞吐速比 = 8 / 1.35 ≈ 5.9×。
    """
    s = fresh_server()
    s.loaded_models["llama-7b"] = 0.0
    s.weight_vram_used = CATALOG["llama-7b"].weights_GB

    N = 8
    decodes = []
    for i in range(N):
        pair = Task.generate_inference_request(i, 2 * i, "llama-7b", 256, 100)
        pair[0].status = TaskStatus.COMPLETED
        d = pair[1]
        d.status = TaskStatus.READY
        decodes.append(d)

    for d in decodes:
        s.add_task(d, priority=1.0)
    s.process_tasks(current_time=0.0)

    # 所有应都已 running
    for d in decodes:
        assert d.status == TaskStatus.RUNNING, \
            f"task {d.task_id} 未启动 (batch_size={d.batch_size_at_admit})"

    durations = [d.end_time - d.start_time for d in decodes]
    # 第一个 batch=1（solo），后面递增到 batch=N
    # 真实模型里 batch 是 contemporaneous 的，所以最后一个的 duration 最具代表性
    final = decodes[-1]
    # M4 step2 后 solo_exec 含 memory floor，需从实测时长读取
    solo_dur = decodes[0].end_time - decodes[0].start_time
    expected_final = solo_dur * (1.0 + (N - 1) * DECODE_BATCH_OVERHEAD)
    assert_close(final.end_time - final.start_time, expected_final,
                 msg=f"batch={N} decode 时长")

    # 吞吐速比 = N × T_solo / wall_time, where wall_time = 最晚的 end_time
    wall_time = max(d.end_time for d in decodes) - 0.0
    speedup = (N * solo_dur) / wall_time
    print(f"  [PASS] T6 throughput speedup: "
          f"N=8 decodes, solo={solo_dur:.4f}s, wall={wall_time:.4f}s, "
          f"speedup={speedup:.2f}×")
    assert speedup > 5.0, f"8-decode 吞吐速比应 > 5×，实际 {speedup:.2f}"


# ====================================================================
# T7. batch_size_at_admit 字段
# ====================================================================

def test_batch_size_field_written():
    s = fresh_server()
    s.loaded_models["llama-7b"] = 0.0
    s.weight_vram_used = CATALOG["llama-7b"].weights_GB

    decodes = []
    for i in range(3):
        pair = Task.generate_inference_request(i, 2 * i, "llama-7b", 256, 100)
        pair[0].status = TaskStatus.COMPLETED
        d = pair[1]
        d.status = TaskStatus.READY
        decodes.append(d)
        s.add_task(d, priority=1.0)
        s.process_tasks(current_time=0.0)

    sizes = [d.batch_size_at_admit for d in decodes]
    assert sizes == [1, 2, 3], f"batch_size_at_admit 应递增，got {sizes}"
    print(f"  [PASS] T7 batch_size_at_admit recorded: {sizes}")


# ====================================================================
# main
# ====================================================================

if __name__ == "__main__":
    tests = [
        test_generic_no_batching,
        test_same_model_same_kind_batches,
        test_different_models_isolated,
        test_different_kinds_isolated,
        test_max_batch_size_blocks_admission,
        test_throughput_speedup,
        test_batch_size_field_written,
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
