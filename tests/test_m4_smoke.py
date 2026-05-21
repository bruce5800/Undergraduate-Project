"""
test_m4_smoke.py — M4 step 1: AIGC 评估指标 (TTFT / TPOT / goodput / SLO)

验证：
  T1. workload="dag" 时 _metrics_list 不含 AIGC 指标
  T2. workload="inference" 时 _metrics_list 包含 9 个 AIGC 指标
  T3. 空 completed_tasks 时 AIGC 指标全为 0（不崩）
  T4. 手工构造 3 个请求的精确场景，验证 TTFT/TPOT/E2E/Goodput/SLO 公式
  T5. 仅完成 prefill、decode 未完成的请求不计入统计
  T6. SLO 阈值生效：调严 ttft_slo 后达成率下降

直接 python tests/test_m4_smoke.py 运行。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.simulation import Simulation
from environment.task import Task, TaskKind, TaskStatus
from brenchmark import BenchmarkTester, COMMON_METRICS, AIGC_METRICS


def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b}")


def _build_request_in_sim(sim, req_id, ttft, tpot, output_tokens):
    """在 sim 里塞一个已经"完成"的 (prefill, decode) 请求，精确控制时序。

    TTFT = ttft（prefill 完成于 t=ttft，ready_time=0）
    TPOT = tpot（decode 耗时 = tpot * output_tokens）
    """
    pair = Task.generate_inference_request(
        req_id=req_id, task_id_offset=2 * req_id,
        model_id="llama-7b", prompt_tokens=100, output_tokens=output_tokens)
    prefill, decode = pair

    prefill.ready_time = 0.0
    prefill.start_time = 0.0
    prefill.end_time = ttft
    decode.ready_time = ttft
    decode.start_time = ttft       # 忽略传输延迟，使测试可手算
    decode.end_time = ttft + tpot * output_tokens

    sim.tasks[prefill.task_id] = prefill
    sim.tasks[decode.task_id] = decode
    sim.completed_tasks.add(prefill.task_id)
    sim.completed_tasks.add(decode.task_id)
    return pair


# ====================================================================
# T1/T2. _metrics_list 行为
# ====================================================================

def test_metrics_list_for_dag():
    t = BenchmarkTester(workload="dag")
    cols = t._metrics_list()
    assert cols == COMMON_METRICS, f"dag 模式只应有 COMMON_METRICS，got {cols}"
    print("  [PASS] T1 dag _metrics_list = COMMON only")


def test_metrics_list_for_inference():
    t = BenchmarkTester(workload="inference")
    cols = t._metrics_list()
    assert cols == COMMON_METRICS + AIGC_METRICS
    # 9 个 AIGC 指标都得在
    for key in ["ttft_p50", "ttft_p95", "ttft_p99",
                "tpot_p50", "tpot_p95",
                "goodput_tps", "slo_attainment",
                "req_e2e_p50", "req_e2e_p95"]:
        assert key in cols, f"missing AIGC metric: {key}"
    print(f"  [PASS] T2 inference _metrics_list has all 9 AIGC metrics")


# ====================================================================
# T3. 空 completed
# ====================================================================

def test_empty_completed():
    sim = Simulation(num_servers=2)
    t = BenchmarkTester(workload="inference")
    m = t._collect_aigc_request_metrics(sim, current_time=1.0)
    for k in AIGC_METRICS:
        assert m[k] == 0 or m[k] == 0.0, \
            f"空场景 {k} 应为 0，got {m[k]}"
    print("  [PASS] T3 empty completed → all AIGC metrics = 0")


# ====================================================================
# T4. 手工场景：3 个请求精确计算
# ====================================================================

def test_aigc_metrics_manual_scenario():
    sim = Simulation(num_servers=2)
    # 3 个请求：
    #   #0: TTFT=0.5, TPOT=0.05, out=20 → 满足 SLO (TTFT≤2.0 且 TPOT≤0.1)
    #   #1: TTFT=1.5, TPOT=0.05, out=20 → 满足 SLO
    #   #2: TTFT=3.0, TPOT=0.05, out=20 → TTFT 超 SLO 阈值
    _build_request_in_sim(sim, 0, ttft=0.5, tpot=0.05, output_tokens=20)
    _build_request_in_sim(sim, 1, ttft=1.5, tpot=0.05, output_tokens=20)
    _build_request_in_sim(sim, 2, ttft=3.0, tpot=0.05, output_tokens=20)

    t = BenchmarkTester(workload="inference", ttft_slo=2.0, tpot_slo=0.1)
    current_time = 4.0
    m = t._collect_aigc_request_metrics(sim, current_time)

    # TTFT P50 of [0.5, 1.5, 3.0] = 1.5
    assert_close(m["ttft_p50"], 1.5, msg="TTFT P50")
    # TPOT P50 of [0.05, 0.05, 0.05] = 0.05
    assert_close(m["tpot_p50"], 0.05, msg="TPOT P50")
    # E2E P50 = e2e of req#1 = ttft+tpot*out = 1.5 + 1.0 = 2.5
    assert_close(m["req_e2e_p50"], 2.5, msg="E2E P50")
    # Goodput = 60 tokens / 4.0 s = 15.0
    assert_close(m["goodput_tps"], 15.0, msg="Goodput")
    # SLO 达成率: 2 / 3 ≈ 0.6667
    assert_close(m["slo_attainment"], 2.0 / 3.0, tol=1e-3, msg="SLO attainment")
    print(f"  [PASS] T4 manual 3-req scenario: "
          f"ttft_p50={m['ttft_p50']}, goodput={m['goodput_tps']}, "
          f"slo={m['slo_attainment']:.3f}")


# ====================================================================
# T5. 只完成 prefill 的请求不计入
# ====================================================================

def test_partial_request_not_counted():
    sim = Simulation(num_servers=2)
    # Request 0: 完整 prefill + decode
    _build_request_in_sim(sim, 0, ttft=1.0, tpot=0.05, output_tokens=10)
    # Request 1: 只完成 prefill
    pair = Task.generate_inference_request(
        req_id=1, task_id_offset=2, model_id="llama-7b",
        prompt_tokens=100, output_tokens=20)
    prefill, decode = pair
    prefill.ready_time = 0.0
    prefill.end_time = 0.8
    sim.tasks[prefill.task_id] = prefill
    sim.tasks[decode.task_id] = decode
    sim.completed_tasks.add(prefill.task_id)
    # decode 没加进 completed_tasks → 不应计入

    t = BenchmarkTester(workload="inference")
    m = t._collect_aigc_request_metrics(sim, current_time=2.0)

    # 只有请求 0 的 TTFT 应被计算
    assert_close(m["ttft_p50"], 1.0,
                 msg="只 req#0 完成；P50 应等于其 TTFT=1.0")
    # SLO 计算仅基于完整请求：req#0 满足 → 1/1 = 1.0
    assert_close(m["slo_attainment"], 1.0,
                 msg="只 req#0 完成且满足 SLO")
    print("  [PASS] T5 partial request excluded from stats")


# ====================================================================
# T6. SLO 阈值生效
# ====================================================================

def test_slo_threshold_effect():
    sim = Simulation(num_servers=2)
    _build_request_in_sim(sim, 0, ttft=0.5, tpot=0.05, output_tokens=10)
    _build_request_in_sim(sim, 1, ttft=1.5, tpot=0.05, output_tokens=10)
    _build_request_in_sim(sim, 2, ttft=2.5, tpot=0.05, output_tokens=10)

    # 宽松 SLO (TTFT≤3.0): 3/3 全部满足
    loose = BenchmarkTester(workload="inference", ttft_slo=3.0, tpot_slo=0.1)
    m_loose = loose._collect_aigc_request_metrics(sim, current_time=3.0)
    assert_close(m_loose["slo_attainment"], 1.0,
                 msg="ttft_slo=3 → 3/3 都满足")

    # 严格 SLO (TTFT≤1.0): 仅 req#0 满足
    tight = BenchmarkTester(workload="inference", ttft_slo=1.0, tpot_slo=0.1)
    m_tight = tight._collect_aigc_request_metrics(sim, current_time=3.0)
    assert_close(m_tight["slo_attainment"], 1.0 / 3.0,
                 tol=1e-3, msg="ttft_slo=1 → 1/3 满足")
    print(f"  [PASS] T6 SLO threshold: loose={m_loose['slo_attainment']}, "
          f"tight={m_tight['slo_attainment']:.3f}")


# ====================================================================
# main
# ====================================================================

if __name__ == "__main__":
    tests = [
        test_metrics_list_for_dag,
        test_metrics_list_for_inference,
        test_empty_completed,
        test_aigc_metrics_manual_scenario,
        test_partial_request_not_counted,
        test_slo_threshold_effect,
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
