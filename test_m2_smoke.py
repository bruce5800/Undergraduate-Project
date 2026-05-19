"""
test_m2_smoke.py — M2 第一步：Task 拆 prefill/decode 两阶段

验证：
  T1. 向后兼容：旧的 generate_single_dag 等生成器产出的 task.kind == GENERIC
  T2. 工厂返回 [prefill, decode] 两个 Task，结构、字段正确
  T3. workload 公式正确：prefill ∝ prompt_tokens；decode ∝ output_tokens
  T4. prefill.output_size = KV cache 大小（GB），用于 KV 迁移代价
  T5. decode 依赖 prefill，共享 req_id
  T6. 输入校验：非 LLM 模型 / 非法 token 数应抛错
  T7. 批量生成器：N 请求 → 2N 任务，task_id / req_id 单调递增

直接 python test_m2_smoke.py 运行。
"""

import random
import sys

from environment.task import Task, TaskKind, TaskStatus
from environment.model_catalog import CATALOG


def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b}")


# ====================================================================
# T1. 向后兼容
# ====================================================================

def test_backward_compat_generic():
    random.seed(0)
    tasks = Task.generate_single_dag(0, 5)
    assert len(tasks) == 5
    for t in tasks:
        assert t.kind == TaskKind.GENERIC, \
            f"existing generators must produce GENERIC, got {t.kind}"
        assert t.prompt_tokens == 0
        assert t.output_tokens == 0
        assert t.req_id is None
        assert t.model_id is None
    print("  [PASS] T1 backward compat: generic generators unchanged")


# ====================================================================
# T2. 工厂结构
# ====================================================================

def test_factory_structure():
    pair = Task.generate_inference_request(
        req_id=42, task_id_offset=100,
        model_id="llama-7b",
        prompt_tokens=512, output_tokens=128)

    assert len(pair) == 2, f"expected 2 tasks, got {len(pair)}"
    prefill, decode = pair
    assert prefill.kind == TaskKind.PREFILL
    assert decode.kind == TaskKind.DECODE
    assert prefill.task_id == 100
    assert decode.task_id == 101
    assert prefill.model_id == "llama-7b"
    assert decode.model_id == "llama-7b"
    assert prefill.req_id == 42
    assert decode.req_id == 42
    print("  [PASS] T2 factory structure")


# ====================================================================
# T3. workload 公式
# ====================================================================

def test_workload_formula():
    spec = CATALOG["llama-13b"]
    prompt = 1000
    output = 200
    pair = Task.generate_inference_request(
        req_id=0, task_id_offset=0,
        model_id="llama-13b",
        prompt_tokens=prompt, output_tokens=output)
    prefill, decode = pair

    expected_prefill_workload = prompt / 1000.0 * spec.prefill_tflops_per_ktoken
    expected_decode_workload = output / 1000.0 * spec.decode_tflops_per_ktoken
    assert_close(prefill.workload, expected_prefill_workload,
                 msg="prefill workload")
    assert_close(decode.workload, expected_decode_workload,
                 msg="decode workload")
    print("  [PASS] T3 workload formula")


# ====================================================================
# T4. KV cache → output_size
# ====================================================================

def test_kv_cache_encoded_in_output_size():
    spec = CATALOG["llama-7b"]
    prompt = 1024
    pair = Task.generate_inference_request(
        req_id=0, task_id_offset=0,
        model_id="llama-7b",
        prompt_tokens=prompt, output_tokens=100)
    prefill, decode = pair

    expected_kv_GB = prompt * spec.kv_cache_MB_per_token / 1024.0
    assert_close(prefill.output_size, expected_kv_GB,
                 msg="prefill.output_size = KV cache GB")
    # decode 的输出是文本，极小
    assert decode.output_size < 0.01, \
        f"decode.output_size should be tiny (text), got {decode.output_size}"
    print(f"  [PASS] T4 KV cache encoded "
          f"(llama-7b @ 1024 tok = {expected_kv_GB:.3f} GB)")


# ====================================================================
# T5. 依赖关系 + req_id
# ====================================================================

def test_dependency_and_req_id():
    pair = Task.generate_inference_request(
        req_id=7, task_id_offset=50,
        model_id="llama-7b",
        prompt_tokens=100, output_tokens=50)
    prefill, decode = pair

    assert prefill.dependencies == [], "prefill 无依赖"
    assert decode.dependencies == [prefill.task_id], \
        f"decode 必须依赖 prefill，got {decode.dependencies}"
    assert prefill.req_id == decode.req_id == 7
    print("  [PASS] T5 dependency + req_id")


# ====================================================================
# T6. 输入校验
# ====================================================================

def test_input_validation():
    # 未知模型
    try:
        Task.generate_inference_request(0, 0, "nonexistent-model", 100, 100)
        raise AssertionError("应当抛 ValueError")
    except ValueError:
        pass

    # 非 LLM 模型
    try:
        Task.generate_inference_request(0, 0, "sdxl", 100, 100)
        raise AssertionError("sdxl 是 Diffusion，应当抛 ValueError")
    except ValueError:
        pass

    # 非法 token 数
    try:
        Task.generate_inference_request(0, 0, "llama-7b", 0, 100)
        raise AssertionError("prompt_tokens=0 应当抛 ValueError")
    except ValueError:
        pass

    print("  [PASS] T6 input validation")


# ====================================================================
# T7. 批量生成器
# ====================================================================

def test_batch_workload_generator():
    rng = random.Random(123)
    N = 10
    tasks = Task.generate_inference_workload(
        num_requests=N, task_id_offset=200, rng=rng)

    assert len(tasks) == 2 * N, f"N={N} → 2N tasks, got {len(tasks)}"

    # task_id 单调递增
    ids = [t.task_id for t in tasks]
    assert ids == list(range(200, 200 + 2 * N)), \
        f"task_id 应单调递增，got {ids}"

    # req_id 取 0..N-1，每个出现两次（一次 prefill，一次 decode）
    req_counts = {}
    for t in tasks:
        req_counts[t.req_id] = req_counts.get(t.req_id, 0) + 1
    assert set(req_counts.keys()) == set(range(N))
    assert all(c == 2 for c in req_counts.values())

    # 每个 req_id 都有正好一个 prefill 和一个 decode
    for req_id in range(N):
        pair = [t for t in tasks if t.req_id == req_id]
        kinds = sorted([t.kind for t in pair], key=lambda k: k.value)
        assert kinds == [TaskKind.PREFILL, TaskKind.DECODE], \
            f"req {req_id} pair kinds {kinds}"

    # 所有任务都是 LLM family（不应有 sdxl）
    for t in tasks:
        assert CATALOG[t.model_id].family == "LLM"

    print(f"  [PASS] T7 batch workload: {N} reqs → {len(tasks)} tasks")


# ====================================================================
# main
# ====================================================================

if __name__ == "__main__":
    tests = [
        test_backward_compat_generic,
        test_factory_structure,
        test_workload_formula,
        test_kv_cache_encoded_in_output_size,
        test_dependency_and_req_id,
        test_input_validation,
        test_batch_workload_generator,
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
