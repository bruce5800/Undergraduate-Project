"""
test_a3cr2n_smoke.py — A3C-R2N2 baseline 烟雾测试

验证：
  T1. state_dim 不包含 AIGC 特征（应是 8 + 5×num_servers）
  T2. R2N2 网络：GRU + residual + actor/critic 头能正向传播
  T3. reward 不含 AIGC bonus（warm/batch/affinity 关掉都不影响 reward 值）
  T4. 完整跑一个小推理 workload 能完成
  T5. 与我们升级版 RL 的 state_dim 不同（A3C 应该更小）
  T6. 预训练能正常运行且不污染 sim 状态

直接 python tests/test_a3cr2n_smoke.py 运行。
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from environment.simulation import Simulation
from environment.task import Task, TaskKind, TaskStatus
from environment.model_catalog import CATALOG
from scheduler.A3CR2NScheduler import (
    A3CR2NScheduler, GenericStateEncoder, R2N2ActorCritic
)
from scheduler.RLscheduler import RLScheduler


def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b}")


# ====================================================================
# T1. State dim 是 generic 大小
# ====================================================================

def test_generic_state_dim():
    sim = Simulation(num_servers=4)
    encoder = GenericStateEncoder(sim)
    expected = 8 + 5 * 4
    assert encoder.state_dim == expected, \
        f"A3C generic state_dim = {expected}，got {encoder.state_dim}"
    # 实际编码一个任务也应一致
    t = Task(0, 5, 200, 2, 0.1, dependencies=[])
    v = encoder.encode(t)
    assert v.shape[0] == expected
    print(f"  [PASS] T1 generic state_dim = {expected} (no AIGC features)")


# ====================================================================
# T2. R2N2 网络 forward 通畅
# ====================================================================

def test_r2n2_forward():
    state_dim, action_dim = 28, 4
    net = R2N2ActorCritic(state_dim, action_dim, hidden_dim=64)
    x = torch.randn(1, state_dim)
    probs, value, hidden = net(x)
    assert probs.shape == (1, action_dim)
    assert value.shape == (1, 1)
    assert hidden.shape == (1, 64), f"GRU hidden 应 (1, 64)，got {hidden.shape}"
    # 概率应归一
    assert_close(probs.sum().item(), 1.0, tol=1e-5, msg="概率和应为 1")

    # 二次 forward：把 hidden 喂回去，应能继续
    probs2, value2, hidden2 = net(x, hidden)
    assert hidden2.shape == (1, 64)
    # GRU 状态确实在变化
    assert not torch.allclose(hidden, hidden2), "GRU 隐状态应该会被更新"

    # 带 mask
    mask = torch.tensor([[True, False, True, True]])
    probs3, _, _ = net(x, action_mask=mask)
    assert_close(probs3[0, 1].item(), 0.0, tol=1e-6,
                 msg="mask 掉的动作概率应为 0")
    print("  [PASS] T2 R2N2 forward OK (GRU + residual + mask)")


# ====================================================================
# T3. Reward 不含 AIGC bonus
# ====================================================================

def test_reward_no_aigc_bonus():
    """A3C-R2N2 的 reward 只看 time + balance；同 server 上 model 是否已加载
    不应影响 reward 值（因为 generic reward 不看 loaded_models）。"""
    sim = Simulation(num_servers=4)
    pair = Task.generate_inference_request(0, 0, "llama-7b", 200, 50)
    sim.add_tasks(pair)
    sched = A3CR2NScheduler(sim, pretrain_episodes=0)

    prefill = pair[0]
    server_warm = sim.servers[1]
    server_cold = sim.servers[2]
    # warm server: 把模型标为已加载
    server_warm.loaded_models["llama-7b"] = 0.0
    server_warm.weight_vram_used = CATALOG["llama-7b"].weights_GB

    r_warm = sched._calculate_reward(prefill, server_warm, transfer_time=1.0)
    r_cold = sched._calculate_reward(prefill, server_cold, transfer_time=1.0)
    # generic reward 只看 time + util，server 1 vs 2 的 compute 不同会有微差异，
    # 但 "warm" 状态本身不应进入 reward
    # 验证：如果我们清空 warm server 的 loaded_models，reward 不应变化
    r_warm_v2 = sched._calculate_reward(prefill, server_warm, transfer_time=1.0)
    assert_close(r_warm, r_warm_v2, tol=1e-9,
                 msg="同一调用 reward 必须确定性相同")

    # 关键：把 server_warm 的 loaded_models 清掉，reward 应当一字不差
    server_warm.loaded_models.clear()
    server_warm.weight_vram_used = 0.0
    r_warm_no_model = sched._calculate_reward(prefill, server_warm, transfer_time=1.0)
    assert_close(r_warm, r_warm_no_model, tol=1e-9,
                 msg="A3C reward 应忽略 loaded_models 状态")
    print(f"  [PASS] T3 A3C reward ignores AIGC state "
          f"(r_warm={r_warm:.3f}, r_cold={r_cold:.3f})")


# ====================================================================
# T4. 完整推理 workload 跑通
# ====================================================================

def test_a3c_full_run():
    random.seed(0)
    sim = Simulation(num_servers=5)
    rng = random.Random(0)
    tasks = Task.generate_inference_workload(
        num_requests=10, task_id_offset=0, rng=rng, dist="uniform")
    sim.add_tasks(tasks)
    sched = A3CR2NScheduler(sim, pretrain_episodes=2)

    t = 0.0
    while len(sim.completed_tasks) < len(sim.tasks) and t < 500:
        sim.step(sched, t)
        t += 0.1
    done = len(sim.completed_tasks)
    assert done == len(sim.tasks), \
        f"未全部完成 ({done}/{len(sim.tasks)})"
    print(f"  [PASS] T4 A3C-R2N2 full inference run: "
          f"makespan={t:.2f}s, done={done}/{len(sim.tasks)}")


# ====================================================================
# T5. A3C state_dim 应比 RL 小（因为没 AIGC 块）
# ====================================================================

def test_a3c_state_smaller_than_rl():
    sim = Simulation(num_servers=4)
    a3c_enc = GenericStateEncoder(sim)
    rl = RLScheduler(sim, pretrain_episodes=0)
    rl_dim = rl.state_encoder.state_dim
    a3c_dim = a3c_enc.state_dim
    assert a3c_dim < rl_dim, \
        f"A3C state ({a3c_dim}) 应 < RL state ({rl_dim}); 差额即 AIGC 信号"
    diff = rl_dim - a3c_dim
    n_models = len(CATALOG)
    # 期望差额：任务 AIGC 4+|M| 维 + 每 server AIGC 5 维
    expected_diff = (4 + n_models) + 5 * 4   # 4 servers
    assert diff == expected_diff, \
        f"AIGC 信号差额 {diff} != 期望 {expected_diff}"
    print(f"  [PASS] T5 A3C state ({a3c_dim}) < RL state ({rl_dim}); "
          f"diff = {diff} (4 task AIGC + |M|={n_models} model one-hot + 5 server AIGC × ns)")


# ====================================================================
# T6. 预训练后状态干净（任务都是 WAITING、completed_tasks 空）
# ====================================================================

def test_pretrain_resets_environment():
    sim = Simulation(num_servers=4)
    rng = random.Random(0)
    tasks = Task.generate_inference_workload(
        num_requests=5, task_id_offset=0, rng=rng, dist="uniform")
    sim.add_tasks(tasks)
    sched = A3CR2NScheduler(sim, pretrain_episodes=2)

    # 预训练完后 sim 应该重置
    for t in sim.tasks.values():
        assert t.status == TaskStatus.WAITING, \
            f"预训练后任务应回到 WAITING，got {t.status}"
        assert t.assigned_server is None
    assert len(sim.completed_tasks) == 0
    for s in sim.servers.values():
        assert s.weight_vram_used == 0.0
        assert len(s.loaded_models) == 0
    print("  [PASS] T6 pretrain leaves env clean")


# ====================================================================
# main
# ====================================================================

if __name__ == "__main__":
    tests = [
        test_generic_state_dim,
        test_r2n2_forward,
        test_reward_no_aigc_bonus,
        test_a3c_full_run,
        test_a3c_state_smaller_than_rl,
        test_pretrain_resets_environment,
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
