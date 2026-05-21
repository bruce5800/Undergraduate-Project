"""
test_m1_smoke.py — M1 阶段烟雾测试

验证 4 个核心不变量：
  T1. 向后兼容：model_id=None 的任务行为完全不变（资源会计与原版一致）
  T2. 冷加载触发：首次使用模型的任务 start_time 包含 cold_load_sec
  T3. 暖启动：第二次使用同模型的任务无冷加载延迟
  T4. LRU 驱逐：当显存装不下新模型时，最旧的 unpinned 模型被驱逐

直接 python test_m1_smoke.py 运行，全部通过返回 0。
"""

import os
import sys

# 让从子目录运行也能找到 environment / scheduler 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.server import Server, ServerType
from environment.task import Task, TaskStatus
from environment.model_catalog import CATALOG


def fresh_server(memory_GB: float = 60.0) -> Server:
    """返回一个干净的边缘服务器。"""
    return Server(server_id=1, server_type=ServerType.EDGE,
                  compute_capacity=50.0, memory=memory_GB,
                  storage=256.0, bandwidth=500.0)


def fresh_task(tid: int, model_id=None, input_size=2.0) -> Task:
    return Task(task_id=tid, compute_demand=5.0, workload=100.0,
                input_size=input_size, output_size=0.1,
                dependencies=[], model_id=model_id)


def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b}")


# ====================================================================
# T1. 向后兼容
# ====================================================================

def test_backward_compat():
    """model_id=None 的任务流程必须与改造前完全一致。"""
    s = fresh_server()
    t = fresh_task(0, model_id=None, input_size=8.0)

    assert s.can_allocate(t), "干净服务器应该能容纳普通任务"
    s.add_task(t, priority=1.0)
    s.process_tasks(current_time=0.0)

    assert t.status == TaskStatus.RUNNING
    assert_close(t.cold_load_delay, 0.0, msg="普通任务不应有冷加载延迟")
    assert_close(t.start_time, 0.0, msg="普通任务 start_time = current_time")
    # exec_time = 100 / 50 = 2.0
    assert_close(t.end_time, 2.0, msg="exec_time = workload/compute")
    assert_close(s.weight_vram_used, 0.0, msg="普通任务不应占权重显存")
    assert s.loaded_models == {}, "普通任务不应加载任何模型"
    print("  [PASS] T1 backward compat")


# ====================================================================
# T2. 冷加载触发
# ====================================================================

def test_cold_load_on_first_use():
    s = fresh_server(memory_GB=60.0)
    t = fresh_task(0, model_id="llama-7b", input_size=2.0)  # llama-7b=14GB

    # 调度器查询冷加载代价应该等于 ModelSpec.cold_load_sec
    assert_close(s.cold_load_cost("llama-7b"),
                 CATALOG["llama-7b"].cold_load_sec,
                 msg="冷查询返回 spec 中的 cold_load_sec")

    s.add_task(t, priority=1.0)
    s.process_tasks(current_time=10.0)

    assert t.status == TaskStatus.RUNNING
    expected_cold = CATALOG["llama-7b"].cold_load_sec
    assert_close(t.cold_load_delay, expected_cold,
                 msg="首次使用模型应产生冷加载延迟")
    # start_time = current_time + transfer_delay + cold = 10 + 0 + 5 = 15
    assert_close(t.start_time, 10.0 + expected_cold,
                 msg="start_time 应包含冷加载")
    assert_close(s.weight_vram_used, 14.0, msg="llama-7b 权重 14GB 应被记账")
    assert "llama-7b" in s.loaded_models
    assert s.model_refs.get("llama-7b") == 1, "应有 1 个 running 任务持有引用"
    print("  [PASS] T2 cold load on first use")


# ====================================================================
# T3. 暖启动：第二次无冷加载
# ====================================================================

def test_warm_start_on_second_use():
    s = fresh_server(memory_GB=60.0)
    t1 = fresh_task(0, model_id="llama-7b", input_size=2.0)
    t2 = fresh_task(1, model_id="llama-7b", input_size=2.0)

    # 第一个任务触发加载并完成
    s.add_task(t1, priority=1.0)
    s.process_tasks(current_time=0.0)
    # 手动释放：模拟 simulation.step 在 t1.end_time 之后调用 update_resource(allocate=False)
    s.update_resource(t1, allocate=False)
    s.running_tasks.remove(t1)

    # 此刻模型仍然驻留（unpinned 状态），第二个任务应该是暖启动
    assert "llama-7b" in s.loaded_models, "模型应在 unpinned 状态下保持驻留"
    assert s.model_refs.get("llama-7b", 0) == 0, "引用已释放"
    assert_close(s.cold_load_cost("llama-7b"), 0.0,
                 msg="已加载模型查询代价应为 0")

    s.add_task(t2, priority=1.0)
    s.process_tasks(current_time=100.0)
    assert_close(t2.cold_load_delay, 0.0, msg="第二次使用应无冷加载")
    assert_close(t2.start_time, 100.0, msg="暖启动 start_time = current_time")
    print("  [PASS] T3 warm start")


# ====================================================================
# T4. LRU 驱逐
# ====================================================================

def test_lru_eviction():
    """显存装不下两个大模型，强制 LRU 驱逐。

    场景：服务器 50 GB；先加载 llama-7b (14)+ sdxl (12)= 26 GB；
    然后请求 llama-13b (26 GB)；总和 52 GB 装不下，
    必须驱逐 unpinned 的 LRU 项。
    """
    s = fresh_server(memory_GB=50.0)

    # T_a 用 llama-7b（先来）
    ta = fresh_task(0, model_id="llama-7b", input_size=2.0)
    s.add_task(ta, priority=1.0)
    s.process_tasks(current_time=0.0)
    s.update_resource(ta, allocate=False)
    s.running_tasks.remove(ta)
    # llama-7b 现在 unpinned，last_use_time=0.0

    # T_b 用 sdxl（后来）
    tb = fresh_task(1, model_id="sdxl", input_size=2.0)
    s.add_task(tb, priority=1.0)
    s.process_tasks(current_time=5.0)
    s.update_resource(tb, allocate=False)
    s.running_tasks.remove(tb)
    # sdxl unpinned，last_use_time=5.0；llama-7b 仍是 0.0（最旧）

    assert s.loaded_models.keys() == {"llama-7b", "sdxl"}
    assert_close(s.weight_vram_used, 14.0 + 12.0)

    # T_c 用 llama-13b（26 GB），加上 input=2 共 28 GB
    # 当前 used_memory=0, pinned=0, weight=26 → free=24
    # 装 13b 需要 26+2=28，free 24 不够，需要驱逐
    # LRU 优先驱逐 llama-7b（last_use=0.0 比 sdxl 的 5.0 旧）
    tc = fresh_task(2, model_id="llama-13b", input_size=2.0)
    assert s.can_allocate(tc), "驱逐 llama-7b 后应能容纳 llama-13b"

    s.add_task(tc, priority=1.0)
    s.process_tasks(current_time=10.0)
    assert tc.status == TaskStatus.RUNNING
    assert "llama-7b" not in s.loaded_models, "LRU 应驱逐 llama-7b"
    assert "sdxl" in s.loaded_models, "sdxl 是更近使用，应保留"
    assert "llama-13b" in s.loaded_models, "新模型应已加载"
    # 验证显存会计正确
    assert_close(s.weight_vram_used, 12.0 + 26.0,
                 msg="weight_vram = sdxl + llama-13b")
    print("  [PASS] T4 LRU eviction")


# ====================================================================
# T5. Pinned 模型不可驱逐
# ====================================================================

def test_pinned_not_evicted():
    """正在被 running 任务使用的模型不可驱逐 —— 资源不足应导致调度失败。"""
    s = fresh_server(memory_GB=20.0)  # 故意小，只够装一个

    ta = fresh_task(0, model_id="llama-7b", input_size=2.0)
    s.add_task(ta, priority=1.0)
    s.process_tasks(current_time=0.0)
    # ta 正在 running，llama-7b pinned (refs=1)

    # 试图调度 sdxl(12GB)：14 pinned + 12 = 26 > 20, 不能驱逐 llama-7b
    tb = fresh_task(1, model_id="sdxl", input_size=2.0)
    assert not s.can_allocate(tb), "pinned 模型挡道时应拒绝分配"
    print("  [PASS] T5 pinned not evicted")


# ====================================================================
# main
# ====================================================================

if __name__ == "__main__":
    tests = [
        test_backward_compat,
        test_cold_load_on_first_use,
        test_warm_start_on_second_use,
        test_lru_eviction,
        test_pinned_not_evicted,
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
