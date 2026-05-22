"""
test_gnn_smoke.py — GNN scheduler 烟雾测试

验证：
  T1. State encoder 产出形状正确（task / server / edge 三段独立）
  T2. GNNActorCritic forward 通畅，输出概率归一
  T3. Action mask 生效（被 mask 掉的 server 概率为 0）
  T4. AIGC 信号在边特征里（cold_load / sibling / batch 维度有变化）
  T5. no_aigc_state ablation 把 AIGC 边维度清零
  T6. 完整跑一个小推理 workload 能完成
  T7. Pretrain 后环境状态干净

直接 python tests/test_gnn_smoke.py 运行。
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from environment.simulation import Simulation
from environment.task import Task, TaskKind, TaskStatus
from environment.model_catalog import CATALOG
from scheduler.GNNScheduler import (
    GNNScheduler, GNNStateEncoder, GNNActorCritic,
)


def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b}")


# ====================================================================
# T1. State encoder shapes
# ====================================================================

def test_encoder_shapes():
    sim = Simulation(num_servers=5)
    enc = GNNStateEncoder(sim)
    pair = Task.generate_inference_request(0, 0, "llama-7b", 200, 50)
    task = pair[0]

    task_feat, server_feats, edge_feats = enc.encode(task)
    n_servers = 5

    assert task_feat.shape == (enc.task_dim,), \
        f"task_feat: expected ({enc.task_dim},), got {task_feat.shape}"
    assert server_feats.shape == (n_servers, enc.server_dim), \
        f"server_feats: expected ({n_servers}, {enc.server_dim}), got {server_feats.shape}"
    assert edge_feats.shape == (n_servers, enc.edge_dim), \
        f"edge_feats: expected ({n_servers}, {enc.edge_dim}), got {edge_feats.shape}"
    print(f"  [PASS] T1 shapes: task={task_feat.shape}, "
          f"server={server_feats.shape}, edge={edge_feats.shape}")


# ====================================================================
# T2. Network forward
# ====================================================================

def test_network_forward():
    net = GNNActorCritic(task_dim=16, server_dim=7, edge_dim=6, hidden_dim=32)
    B, S = 1, 5
    task_feat = torch.randn(B, 16)
    server_feats = torch.randn(B, S, 7)
    edge_feats = torch.randn(B, S, 6)

    probs, value, logits = net(task_feat, server_feats, edge_feats)
    assert probs.shape == (B, S)
    assert value.shape == (B, 1)
    assert logits.shape == (B, S)
    assert_close(probs.sum().item(), 1.0, tol=1e-5,
                 msg="概率和应为 1")
    print(f"  [PASS] T2 GNN forward: probs={probs.shape}, value={value.shape}")


# ====================================================================
# T3. Action mask
# ====================================================================

def test_action_mask():
    net = GNNActorCritic(task_dim=16, server_dim=7, edge_dim=6, hidden_dim=32)
    task_feat = torch.randn(1, 16)
    server_feats = torch.randn(1, 5, 7)
    edge_feats = torch.randn(1, 5, 6)
    mask = torch.tensor([[True, False, True, True, False]])

    probs, _, _ = net(task_feat, server_feats, edge_feats, action_mask=mask)
    assert_close(probs[0, 1].item(), 0.0, tol=1e-6,
                 msg="被 mask 的位置概率应为 0")
    assert_close(probs[0, 4].item(), 0.0, tol=1e-6)
    # 有效位置概率和应为 1
    valid_sum = probs[0, [0, 2, 3]].sum().item()
    assert_close(valid_sum, 1.0, tol=1e-5)
    print("  [PASS] T3 action mask")


# ====================================================================
# T4. AIGC 信号在边特征中
# ====================================================================

def test_aigc_signal_in_edges():
    """改变 server 的 AIGC 状态（loaded_models / 已有 batch）→ 边特征对应维度变化。"""
    sim = Simulation(num_servers=5)
    enc = GNNStateEncoder(sim)
    pair = Task.generate_inference_request(0, 0, "llama-7b", 200, 50)
    task = pair[0]

    # 基础 edge_feats
    _, _, edge_feats_0 = enc.encode(task)
    # 把 server 1 的 llama-7b 标为已加载
    sim.servers[1].loaded_models["llama-7b"] = 0.0
    sim.servers[1].weight_vram_used = CATALOG["llama-7b"].weights_GB
    _, _, edge_feats_1 = enc.encode(task)

    # Edge layout: [transfer, cold_load, is_loaded, sibling, batch, can_alloc]
    # Server 1 的 is_loaded 应从 0 → 1
    assert_close(edge_feats_0[1, 2].item(), 0.0, msg="未加载时 is_loaded=0")
    assert_close(edge_feats_1[1, 2].item(), 1.0, msg="加载后 is_loaded=1")
    # cold_load 应从 0.25/30 → 0
    assert edge_feats_0[1, 1].item() > 0, "未加载时 cold_load_cost > 0"
    assert_close(edge_feats_1[1, 1].item(), 0.0, msg="加载后 cold_load=0")
    print(f"  [PASS] T4 AIGC signals in edges: "
          f"is_loaded 0→{edge_feats_1[1, 2].item():.0f}, "
          f"cold_load {edge_feats_0[1, 1].item():.3f}→0")


# ====================================================================
# T5. no_aigc_state 清零边的 AIGC 维度
# ====================================================================

def test_no_aigc_state_zeroes_edges():
    sim = Simulation(num_servers=5)
    # 给 server 1 加载模型，让 AIGC 维度有值
    sim.servers[1].loaded_models["llama-7b"] = 0.0
    sim.servers[1].weight_vram_used = CATALOG["llama-7b"].weights_GB

    pair = Task.generate_inference_request(0, 0, "llama-7b", 200, 50)
    task = pair[0]
    sim.add_tasks(pair)

    gnn_full = GNNScheduler(sim, pretrain_episodes=0, enable_aigc_state=True)
    gnn_abl = GNNScheduler(sim, pretrain_episodes=0, enable_aigc_state=False)

    _, _, edge_feats = gnn_full.encoder.encode(task)
    # 调用 _maybe_zero_aigc_edges
    full_edges = gnn_full._maybe_zero_aigc_edges(edge_feats)
    abl_edges = gnn_abl._maybe_zero_aigc_edges(edge_feats)

    # Full 模式：维度 1-4 应保持
    assert torch.allclose(full_edges, edge_feats), \
        "enable_aigc_state=True 时 edges 不应改变"
    # Ablation 模式：维度 1-4 应为 0，维度 0 (transfer) 和 5 (can_alloc) 保留
    assert torch.allclose(abl_edges[:, 1:5], torch.zeros_like(abl_edges[:, 1:5])), \
        "no_aigc_state 应把 AIGC 维度清零"
    assert torch.allclose(abl_edges[:, 0], edge_feats[:, 0]), \
        "transfer_time 应保留"
    assert torch.allclose(abl_edges[:, 5], edge_feats[:, 5]), \
        "can_allocate 应保留"
    print("  [PASS] T5 no_aigc_state zeroes AIGC edge dims")


# ====================================================================
# T6. 完整跑一个小 inference workload
# ====================================================================

def test_full_run():
    random.seed(0)
    sim = Simulation(num_servers=5)
    rng = random.Random(0)
    tasks = Task.generate_inference_workload(
        num_requests=10, task_id_offset=0, rng=rng, dist="uniform")
    sim.add_tasks(tasks)
    sched = GNNScheduler(sim, pretrain_episodes=2)

    t = 0.0
    while len(sim.completed_tasks) < len(sim.tasks) and t < 500:
        sim.step(sched, t)
        t += 0.1
    done = len(sim.completed_tasks)
    assert done == len(sim.tasks), \
        f"未全部完成 ({done}/{len(sim.tasks)})"
    print(f"  [PASS] T6 GNN full run: makespan={t:.2f}s, done={done}/{len(sim.tasks)}")


# ====================================================================
# T7. Pretrain 后状态干净
# ====================================================================

def test_pretrain_clean():
    sim = Simulation(num_servers=4)
    rng = random.Random(0)
    tasks = Task.generate_inference_workload(
        num_requests=5, task_id_offset=0, rng=rng, dist="uniform")
    sim.add_tasks(tasks)
    sched = GNNScheduler(sim, pretrain_episodes=2)

    for t in sim.tasks.values():
        assert t.status == TaskStatus.WAITING, \
            f"预训练后任务应回到 WAITING，got {t.status}"
        assert t.assigned_server is None
    assert len(sim.completed_tasks) == 0
    print("  [PASS] T7 pretrain leaves env clean")


# ====================================================================
# main
# ====================================================================

if __name__ == "__main__":
    tests = [
        test_encoder_shapes,
        test_network_forward,
        test_action_mask,
        test_aigc_signal_in_edges,
        test_no_aigc_state_zeroes_edges,
        test_full_run,
        test_pretrain_clean,
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
