"""
test_m3_step3_smoke.py — M3 step 3: Ablation switches

每个 ablation 都对应一组开关；本测试确认开关确实改变了对应行为：

  T1. no_batching:        Server.enable_batching=False → batch_size 永远=1
  T2. no_warm_reward:     warm_bonus 项被压为 0
  T3. no_batch_reward:    batch_bonus 项被压为 0
  T4. no_affinity_reward: affinity_bonus 项被压为 0
  T5. no_aigc_state:      StateEncoder.state_dim 回到 M0 维度
  T6. no_action_mask:     非法动作可被选中（policy 不再被 mask）
  T7. no_gae:             update_policy 走 MC 分支
  T8. no_pretrain:        构造时跳过 _pretrain
  T9. no_entropy:         entropy_coeff = 0

直接 python test_m3_step3_smoke.py 运行。
"""

import os
import sys

# 让从子目录运行也能找到 environment / scheduler 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.simulation import Simulation
from environment.server import Server, ServerType
from environment.task import Task, TaskKind, TaskStatus
from environment.model_catalog import CATALOG
from scheduler.RLscheduler import RLScheduler


def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b}")


# ====================================================================
# T1. no_batching
# ====================================================================

def test_no_batching():
    """Simulation enable_batching=False 时 process_tasks 走 solo 分支。"""
    sim = Simulation(num_servers=4, enable_batching=False)
    s = sim.servers[0]
    assert s.enable_batching is False

    # 加载 llama-7b 并跑 5 个同模 decode
    s.loaded_models["llama-7b"] = 0.0
    s.weight_vram_used = CATALOG["llama-7b"].weights_GB

    decodes = []
    for i in range(5):
        pair = Task.generate_inference_request(i, 2 * i, "llama-7b", 200, 50)
        pair[0].status = TaskStatus.COMPLETED
        d = pair[1]
        d.status = TaskStatus.READY
        decodes.append(d)
        s.add_task(d, priority=1.0)
        s.process_tasks(current_time=0.0)

    # 关掉 batching 后，所有任务 batch_size_at_admit 应该都是 1
    sizes = [d.batch_size_at_admit for d in decodes]
    assert all(b == 1 for b in sizes), \
        f"no_batching 模式所有 batch=1，实际 {sizes}"
    print(f"  [PASS] T1 no_batching: all batch_size_at_admit={sizes}")


# ====================================================================
# T2/T3/T4: 各项 reward bonus 被消融
# ====================================================================

def _make_rl(sim, **kwargs):
    """构造一个不预训练的 RL，便于测 reward。"""
    kwargs.setdefault("enable_pretrain", False)
    return RLScheduler(sim, pretrain_episodes=0, **kwargs)


def test_no_warm_reward():
    sim = Simulation(num_servers=4)
    full = _make_rl(sim)
    abl  = _make_rl(sim, enable_warm_reward=False)

    pair = Task.generate_inference_request(0, 0, "llama-7b", 200, 50)
    prefill = pair[0]
    server_warm = sim.servers[1]
    server_cold = sim.servers[2]
    server_warm.loaded_models["llama-7b"] = 0.0
    server_warm.weight_vram_used = CATALOG["llama-7b"].weights_GB

    # full 模式：warm > cold
    diff_full = (full.calculate_reward(prefill, server_warm, 1.0)
                 - full.calculate_reward(prefill, server_cold, 1.0))
    # ablation 模式：warm ≈ cold (差异由其他项的轻微 server 差异引起)
    diff_abl  = (abl.calculate_reward(prefill, server_warm, 1.0)
                 - abl.calculate_reward(prefill, server_cold, 1.0))
    assert diff_full > diff_abl + 0.05, \
        f"关掉 warm 后 warm-cold 差应明显缩小: full={diff_full:.3f}, abl={diff_abl:.3f}"
    print(f"  [PASS] T2 no_warm_reward: diff_full={diff_full:.3f} → abl={diff_abl:.3f}")


def test_no_batch_reward():
    sim = Simulation(num_servers=4)
    full = _make_rl(sim)
    abl  = _make_rl(sim, enable_batch_reward=False)

    pair = Task.generate_inference_request(0, 0, "llama-7b", 200, 50)
    decode = pair[1]

    # 构造一台"已有 batch"的 server
    server_batch = sim.servers[1]
    server_empty = sim.servers[2]
    for s in [server_batch, server_empty]:
        s.loaded_models["llama-7b"] = 0.0
        s.weight_vram_used = CATALOG["llama-7b"].weights_GB
    for i in range(3):
        f = Task.generate_inference_request(100 + i, 200 + 2*i, "llama-7b", 200, 50)[1]
        f.status = TaskStatus.RUNNING
        f.kind = TaskKind.DECODE
        server_batch.running_tasks.append(f)

    diff_full = (full.calculate_reward(decode, server_batch, 1.0)
                 - full.calculate_reward(decode, server_empty, 1.0))
    diff_abl  = (abl.calculate_reward(decode, server_batch, 1.0)
                 - abl.calculate_reward(decode, server_empty, 1.0))
    assert diff_full > diff_abl + 0.03, \
        f"关掉 batch 后 batch-solo 差应缩小: full={diff_full:.3f}, abl={diff_abl:.3f}"
    print(f"  [PASS] T3 no_batch_reward: diff_full={diff_full:.3f} → abl={diff_abl:.3f}")


def test_no_affinity_reward():
    sim = Simulation(num_servers=4)
    full = _make_rl(sim)
    abl  = _make_rl(sim, enable_affinity_reward=False)

    pair = Task.generate_inference_request(0, 0, "llama-7b", 1024, 100)
    prefill, decode = pair
    prefill.assigned_server = 1
    sim.tasks[prefill.task_id] = prefill
    sim.tasks[decode.task_id] = decode

    server_local = sim.servers[1]
    server_remote = sim.servers[2]
    for s in [server_local, server_remote]:
        s.loaded_models["llama-7b"] = 0.0
        s.weight_vram_used = CATALOG["llama-7b"].weights_GB

    diff_full = (full.calculate_reward(decode, server_local, 1.0)
                 - full.calculate_reward(decode, server_remote, 1.0))
    diff_abl  = (abl.calculate_reward(decode, server_local, 1.0)
                 - abl.calculate_reward(decode, server_remote, 1.0))
    assert diff_full > diff_abl + 0.03, \
        f"关掉 affinity 后 local-remote 差应缩小: full={diff_full:.3f}, abl={diff_abl:.3f}"
    print(f"  [PASS] T4 no_affinity_reward: diff_full={diff_full:.3f} → abl={diff_abl:.3f}")


# ====================================================================
# T5. no_aigc_state: state_dim 缩回 M0
# ====================================================================

def test_no_aigc_state():
    sim = Simulation(num_servers=4)
    rl_full = _make_rl(sim)
    rl_abl  = _make_rl(sim, enable_aigc_state=False)

    full_dim = rl_full.state_encoder.state_dim
    abl_dim  = rl_abl.state_encoder.state_dim
    expected_abl = 8 + 5 * 4   # 8 + 5 × num_servers
    assert abl_dim == expected_abl, \
        f"no_aigc_state state_dim 应回到 {expected_abl}，实际 {abl_dim}"
    assert full_dim > abl_dim, \
        f"完整 state {full_dim} 应大于消融态 {abl_dim}"

    # 实际编码长度也得对
    t = Task(0, 5, 200, 2, 0.1, dependencies=[])
    v = rl_abl.state_encoder.encode(t)
    assert v.shape[0] == expected_abl
    print(f"  [PASS] T5 no_aigc_state: dim {full_dim} → {abl_dim}")


# ====================================================================
# T6. no_action_mask
# ====================================================================

def test_no_action_mask():
    """关掉 mask 后 policy.forward 接收 mask=None，logits 不被屏蔽。"""
    sim = Simulation(num_servers=4)
    rl_abl = _make_rl(sim, enable_action_mask=False)
    assert rl_abl.enable_action_mask is False

    # 直接调 ActorCritic：mask=None 时 logits 全部有效，非法动作仍能采样
    import torch
    state = torch.zeros(rl_abl.state_encoder.state_dim).unsqueeze(0)
    # mask 全为 False 模拟"无服务器可用"，但传 None 时不该报错
    probs, _, logits = rl_abl.policy(state, None)
    # 验证不被 mask 影响：所有 logit 都是有限值（不是 -inf）
    assert not torch.isinf(logits).any(), \
        "no_action_mask 模式 logits 不应被屏蔽为 -inf"
    print("  [PASS] T6 no_action_mask: logits 不被屏蔽")


# ====================================================================
# T7. no_gae
# ====================================================================

def test_no_gae():
    sim = Simulation(num_servers=4)
    rl_abl = _make_rl(sim, enable_gae=False)
    assert rl_abl.enable_gae is False
    # 模拟一段轨迹并触发 update_policy，验证不抛错（走的是 MC 分支）
    import torch
    n = max(rl_abl.update_interval, 5)
    state_dim = rl_abl.state_encoder.state_dim
    for i in range(n):
        s = torch.zeros(state_dim)
        log_p = torch.tensor(-0.5)
        v = torch.tensor([0.0])
        rl_abl.store_transition(s, 0, reward=0.5,
                                 log_prob=log_p, value=v, done=False)
    rl_abl.update_policy()  # 不抛错即视为通过
    print("  [PASS] T7 no_gae: MC return 分支可正常 update")


# ====================================================================
# T8. no_pretrain
# ====================================================================

def test_no_pretrain():
    sim = Simulation(num_servers=4)
    # 给 sim 加点 task 让 pretrain 有材料可跑（如果会跑的话）
    pair = Task.generate_inference_request(0, 0, "llama-7b", 200, 50)
    sim.add_tasks(pair)

    rl_abl = RLScheduler(sim, pretrain_episodes=5, enable_pretrain=False)
    # 看 _decision_count：若进过 _pretrain 末尾会 = _warmup_steps；
    # 若被跳过则保持 0
    assert rl_abl._decision_count == 0, \
        f"no_pretrain 应跳过预训练，decision_count={rl_abl._decision_count}"
    # 任务也不应被改状态
    assert all(t.status == TaskStatus.WAITING for t in sim.tasks.values()), \
        "no_pretrain 不应改动任务状态"
    print("  [PASS] T8 no_pretrain: skipped pretrain")


# ====================================================================
# T9. no_entropy
# ====================================================================

def test_no_entropy():
    sim = Simulation(num_servers=4)
    rl_full = _make_rl(sim)
    rl_abl  = _make_rl(sim, enable_entropy=False)
    assert rl_full.entropy_coeff > 0
    assert_close(rl_abl.entropy_coeff, 0.0,
                 msg="no_entropy → entropy_coeff = 0")
    print(f"  [PASS] T9 no_entropy: "
          f"coeff {rl_full.entropy_coeff} → {rl_abl.entropy_coeff}")


# ====================================================================
# T10. M4 patch: no_batching 让 GPU 串行（修复旧 batching 模型）
# ====================================================================

def test_no_batching_means_serial_gpu():
    """M4 patch: no_batching 应该让同台 inference 任务串行排队（GPU 串行模型）。

    旧模型问题：multiple inference 任务能并发 → 关 batching 反而更快（错误）
    新模型：no_batching 时只允许 1 个 inference 同台 → 必须排队，反映 GPU 串行真相
    """
    sim_off = Simulation(num_servers=4, enable_batching=False)
    sim_on  = Simulation(num_servers=4, enable_batching=True)

    for sim in (sim_off, sim_on):
        s = sim.servers[0]
        s.loaded_models["llama-7b"] = 0.0
        s.weight_vram_used = CATALOG["llama-7b"].weights_GB

    def _admit_n(sim, n):
        s = sim.servers[0]
        admitted = 0
        for i in range(n):
            pair = Task.generate_inference_request(i, 2*i, "llama-7b", 200, 50)
            pair[0].status = TaskStatus.COMPLETED
            d = pair[1]
            d.status = TaskStatus.READY
            s.add_task(d, priority=1.0)
            s.process_tasks(current_time=0.0)
            if d.status == TaskStatus.RUNNING:
                admitted += 1
        return admitted

    n_off = _admit_n(sim_off, 5)
    n_on  = _admit_n(sim_on, 5)

    assert n_off == 1, \
        f"no_batching 应该只准 1 个 inference 同台，实际 {n_off}"
    assert n_on  >= 4, \
        f"batching on 应该准多个 inference 同台，实际 {n_on}"
    print(f"  [PASS] T10 no_batching serializes GPU: off→{n_off} admitted, on→{n_on}")


# ====================================================================
# main
# ====================================================================

if __name__ == "__main__":
    tests = [
        test_no_batching,
        test_no_warm_reward,
        test_no_batch_reward,
        test_no_affinity_reward,
        test_no_aigc_state,
        test_no_action_mask,
        test_no_gae,
        test_no_pretrain,
        test_no_entropy,
        test_no_batching_means_serial_gpu,
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
