"""
test_m3_step2_smoke.py — M3 step 2: AIGC-aware RL scheduler

验证：
  T1. state_dim 公式正确（10 + |M| + 2 + 10 × num_servers）
  T2. Generic 任务 state 中 AIGC 部分全 0
  T3. AIGC 任务（PREFILL/DECODE）kind/model one-hot 正确写入
  T4. 服务器特征：is_my_model_loaded 反映 loaded_models
  T5. is_sibling_server 信号：DECODE 任务在 prefill 已分配后能看到
  T6. Warm bonus：模型已加载的服务器 reward 高于未加载
  T7. Batch bonus：加入已有 batch 的 reward 高于 solo
  T8. Affinity bonus：DECODE 落在 prefill 服务器 reward 显著高

直接 python test_m3_step2_smoke.py 运行。
"""

import sys
import torch

from environment.simulation import Simulation
from environment.task import Task, TaskKind, TaskStatus
from environment.model_catalog import CATALOG
from scheduler.RLscheduler import StateEncoder, RLScheduler


def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b}")


def fresh_sim(num_servers=4):
    return Simulation(num_servers=num_servers)


# ====================================================================
# T1. state_dim 公式
# ====================================================================

def test_state_dim_formula():
    sim = fresh_sim(num_servers=4)
    encoder = StateEncoder(sim)
    n_models = len(CATALOG)
    n_servers = 4
    expected = (10 + n_models) + 2 + 10 * n_servers
    assert encoder.state_dim == expected, \
        f"state_dim={encoder.state_dim}, expected {expected}"

    # 实际编码一个任务的 vector 长度也应一致
    t = Task(task_id=0, compute_demand=5, workload=200, input_size=2,
             output_size=0.1, dependencies=[])
    v = encoder.encode(t)
    assert v.shape[0] == expected, f"encode 返回长度 {v.shape[0]} != {expected}"
    print(f"  [PASS] T1 state_dim={expected} (|M|={n_models}, ns={n_servers})")


# ====================================================================
# T2. Generic 任务 AIGC 部分全 0
# ====================================================================

def test_generic_aigc_block_zero():
    sim = fresh_sim()
    encoder = StateEncoder(sim)
    t = Task(task_id=0, compute_demand=5, workload=200, input_size=2,
             output_size=0.1, dependencies=[])
    v = encoder.encode(t)

    # Task block 布局: [基础 6, kind one-hot 3, kv 1, model one-hot |M|]
    # kind one-hot 应该 GENERIC=1, PREFILL=0, DECODE=0
    assert_close(v[6].item(), 1.0, msg="generic 应在 GENERIC 位 = 1")
    assert_close(v[7].item(), 0.0, msg="generic PREFILL 位应为 0")
    assert_close(v[8].item(), 0.0, msg="generic DECODE 位应为 0")
    # kv_cache_GB / 5.0
    assert_close(v[9].item(), 0.0, msg="generic kv_cache 应为 0")
    # model one-hot 全 0
    n_models = len(CATALOG)
    model_block = v[10:10 + n_models].tolist()
    assert all(x == 0.0 for x in model_block), \
        f"generic model one-hot 应全 0，got {model_block}"
    print("  [PASS] T2 generic AIGC block all zeros")


# ====================================================================
# T3. AIGC 任务 kind/model 编码
# ====================================================================

def test_aigc_task_encoding():
    sim = fresh_sim()
    encoder = StateEncoder(sim)
    pair = Task.generate_inference_request(0, 0, "llama-13b", 512, 100)
    prefill, decode = pair

    # PREFILL
    v_p = encoder.encode(prefill)
    assert_close(v_p[6].item(), 0.0, msg="PREFILL: GENERIC=0")
    assert_close(v_p[7].item(), 1.0, msg="PREFILL: PREFILL=1")
    assert_close(v_p[8].item(), 0.0, msg="PREFILL: DECODE=0")
    # model one-hot at index of "llama-13b"
    n_models = len(CATALOG)
    model_block_p = v_p[10:10 + n_models].tolist()
    idx_13b = encoder._MODEL_INDEX["llama-13b"]
    assert model_block_p[idx_13b] == 1.0
    assert sum(model_block_p) == 1.0, "model one-hot 应只有 1 个 1"

    # DECODE
    v_d = encoder.encode(decode)
    assert_close(v_d[7].item(), 0.0, msg="DECODE: PREFILL=0")
    assert_close(v_d[8].item(), 1.0, msg="DECODE: DECODE=1")
    print("  [PASS] T3 AIGC task kind+model one-hot")


# ====================================================================
# T4. is_my_model_loaded
# ====================================================================

def test_is_my_model_loaded_feature():
    sim = fresh_sim()
    encoder = StateEncoder(sim)
    # 找一台边缘 server，手动把 llama-7b 标为已加载
    edge = sim.servers[1]
    edge.loaded_models["llama-7b"] = 0.0
    edge.weight_vram_used = CATALOG["llama-7b"].weights_GB

    pair = Task.generate_inference_request(0, 0, "llama-7b", 200, 50)
    prefill = pair[0]
    v = encoder.encode(prefill)

    # Per-server block 从 task_block_end + global(2) 之后开始
    n_models = len(CATALOG)
    task_block_size = 10 + n_models
    server_block_start = task_block_size + 2

    # 每台 server 10 维，is_my_model_loaded 在每个 server block 的第 6 位（index 5）
    for i, sid in enumerate(encoder.server_ids):
        offset = server_block_start + i * 10
        is_loaded_feat = v[offset + 5].item()
        if sid == 1:
            assert_close(is_loaded_feat, 1.0,
                         msg=f"server {sid} 已加载 llama-7b，is_loaded=1")
        else:
            assert_close(is_loaded_feat, 0.0,
                         msg=f"server {sid} 未加载，is_loaded=0")
    print("  [PASS] T4 is_my_model_loaded reflects loaded_models")


# ====================================================================
# T5. is_sibling_server (decode 看 prefill 在哪)
# ====================================================================

def test_sibling_server_feature():
    sim = fresh_sim()
    encoder = StateEncoder(sim)
    pair = Task.generate_inference_request(0, 0, "llama-7b", 200, 50)
    prefill, decode = pair

    # 假装 prefill 已被分配到 server 2
    prefill.assigned_server = 2
    sim.tasks[prefill.task_id] = prefill
    sim.tasks[decode.task_id] = decode

    v = encoder.encode(decode)
    n_models = len(CATALOG)
    server_block_start = (10 + n_models) + 2

    # is_sibling_server 是每个 server block 的第 10 位（index 9）
    for i, sid in enumerate(encoder.server_ids):
        offset = server_block_start + i * 10
        sib_feat = v[offset + 9].item()
        expected = 1.0 if sid == 2 else 0.0
        assert_close(sib_feat, expected,
                     msg=f"server {sid} sibling 应 = {expected}")
    print("  [PASS] T5 is_sibling_server signals prefill location")


# ====================================================================
# T6. Warm bonus
# ====================================================================

def test_warm_bonus_in_reward():
    sim = fresh_sim()
    pair = Task.generate_inference_request(0, 0, "llama-7b", 200, 50)
    sim.add_tasks(pair)

    scheduler = RLScheduler(sim, pretrain_episodes=0)

    prefill = pair[0]
    warm_server = sim.servers[1]
    cold_server = sim.servers[2]
    warm_server.loaded_models["llama-7b"] = 0.0
    warm_server.weight_vram_used = CATALOG["llama-7b"].weights_GB

    r_warm = scheduler.calculate_reward(prefill, warm_server, transfer_time=1.0)
    r_cold = scheduler.calculate_reward(prefill, cold_server, transfer_time=1.0)
    assert r_warm > r_cold, \
        f"warm reward ({r_warm:.3f}) 应高于 cold ({r_cold:.3f})"
    print(f"  [PASS] T6 warm bonus: warm={r_warm:.3f} > cold={r_cold:.3f}")


# ====================================================================
# T7. Batch bonus
# ====================================================================

def test_batch_bonus_in_reward():
    sim = fresh_sim()
    pair = Task.generate_inference_request(0, 0, "llama-7b", 200, 50)
    sim.add_tasks(pair)
    scheduler = RLScheduler(sim, pretrain_episodes=0)

    decode = pair[1]
    decode.kind = TaskKind.DECODE  # 确保字段正确

    server_with_batch = sim.servers[1]
    server_with_batch.loaded_models["llama-7b"] = 0.0
    server_with_batch.weight_vram_used = CATALOG["llama-7b"].weights_GB

    # 模拟 3 个同模同 kind 任务正在 running
    for i in range(3):
        fake = Task.generate_inference_request(
            100 + i, 200 + 2 * i, "llama-7b", 200, 50)[1]
        fake.status = TaskStatus.RUNNING
        fake.kind = TaskKind.DECODE
        server_with_batch.running_tasks.append(fake)

    server_empty = sim.servers[2]
    server_empty.loaded_models["llama-7b"] = 0.0
    server_empty.weight_vram_used = CATALOG["llama-7b"].weights_GB

    r_batch = scheduler.calculate_reward(decode, server_with_batch, 1.0)
    r_solo  = scheduler.calculate_reward(decode, server_empty, 1.0)
    assert r_batch > r_solo, \
        f"batch reward ({r_batch:.3f}) 应高于 solo ({r_solo:.3f})"
    print(f"  [PASS] T7 batch bonus: batch={r_batch:.3f} > solo={r_solo:.3f}")


# ====================================================================
# T8. Affinity bonus
# ====================================================================

def test_affinity_bonus_in_reward():
    sim = fresh_sim()
    pair = Task.generate_inference_request(0, 0, "llama-7b", 1024, 100)
    sim.add_tasks(pair)
    scheduler = RLScheduler(sim, pretrain_episodes=0)

    prefill, decode = pair
    prefill.assigned_server = 1   # prefill 落在 server 1
    sim.tasks[prefill.task_id] = prefill   # 确保 sim 能找到

    server_local = sim.servers[1]
    server_remote = sim.servers[2]
    # 让两边都已加载，排除 warm 干扰
    for s in [server_local, server_remote]:
        s.loaded_models["llama-7b"] = 0.0
        s.weight_vram_used = CATALOG["llama-7b"].weights_GB

    r_local = scheduler.calculate_reward(decode, server_local, transfer_time=1.0)
    r_remote = scheduler.calculate_reward(decode, server_remote, transfer_time=1.0)
    diff = r_local - r_remote
    # 期望 affinity_bonus 主导：+1.0 affinity × 0.10 权重 ≈ +0.10
    # 实际还会被异构 server 的 compute 差异微调 (~0.003)，所以用 0.05 做下界
    assert diff > 0.05, \
        f"affinity bonus 应明显 (local={r_local:.3f} vs remote={r_remote:.3f}, diff={diff:.3f})"
    print(f"  [PASS] T8 affinity bonus: "
          f"local={r_local:.3f} > remote={r_remote:.3f} (diff={diff:.3f})")


# ====================================================================
# main
# ====================================================================

if __name__ == "__main__":
    tests = [
        test_state_dim_formula,
        test_generic_aigc_block_zero,
        test_aigc_task_encoding,
        test_is_my_model_loaded_feature,
        test_sibling_server_feature,
        test_warm_bonus_in_reward,
        test_batch_bonus_in_reward,
        test_affinity_bonus_in_reward,
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
