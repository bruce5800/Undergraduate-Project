"""
diagnose_rl_reward.py — 诊断 RL reward 信号为啥没影响策略

排查 4 个假设：
  H1. AIGC reward 项数值范围太小 → 被 advantage normalization 抹平
  H2. AIGC reward 项与 base reward 高度相关 → 不带新信息
  H3. PPO 预训练后 policy 过度确定 → 后续 reward 改不动它
  H4. State 中 AIGC 特征不影响 policy 输出 → 网络忽略了 AIGC 维度

直接 python tools/diagnose_rl_reward.py 运行；输出表格 + 解读建议。
"""

import os
import sys
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.simulation import Simulation
from environment.task import Task, TaskKind, TaskStatus
from environment.model_catalog import CATALOG
from scheduler.RLscheduler import RLScheduler


def _set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ====================================================================
# 准备：训练好一个 RL，并保留若干典型 (task, server) 决策场景
# ====================================================================

def build_trained_rl(seed=0):
    """构造一个 sim + workload 并预训练 RL（与 brenchmark 一致）。"""
    _set_seed(seed)
    sim = Simulation(num_servers=5)
    rng = random.Random(seed)
    tasks = Task.generate_inference_workload(
        num_requests=50, task_id_offset=0, rng=rng,
        dist="lognormal", arrival_rate=2.0)
    sim.add_tasks(tasks)
    rl = RLScheduler(sim, pretrain_episodes=5)
    return sim, rl


def build_scenarios(sim):
    """构造若干典型决策场景：(task, candidate_server) 组合。

    每个场景模拟一个 'ready' 时刻：1 个待调度任务 + 5 个候选 server。
    我们关心：policy 对这 5 个 server 的概率分布如何变化（受 AIGC 信号影响吗？）。
    """
    rng = random.Random(42)
    scenarios = []
    # 5 个不同类型的请求
    for spec in [
        ("llama-7b",  500, 100, "PREFILL"),   # 短 prompt 7B prefill
        ("llama-13b", 1500, 50, "PREFILL"),   # 长 prompt 13B prefill
        ("llama-70b", 200, 200, "DECODE"),    # 70B 长 decode
        ("llama-7b",  300, 50, "DECODE"),     # 普通 7B decode
        ("llama-13b", 100, 300, "DECODE"),    # 13B 长 decode
    ]:
        model, prompt, output, kind_name = spec
        pair = Task.generate_inference_request(
            req_id=len(scenarios), task_id_offset=10000 + 2 * len(scenarios),
            model_id=model, prompt_tokens=prompt, output_tokens=output)
        task = pair[0] if kind_name == "PREFILL" else pair[1]
        if kind_name == "DECODE":
            # decode 需要 prefill 先完成；给它一个虚构的 prefill server
            pair[0].assigned_server = 1  # 假设在 server 1
        scenarios.append((spec, task))
    return scenarios


# ====================================================================
# H1: Reward 项数值范围
# ====================================================================

def diagnose_reward_magnitudes(rl, sim, scenarios, n=200):
    """对每个场景采样若干 (task, server) 决策，记录 6 项 reward 分量。

    输出每项的 mean/std 和占总 reward 的相对贡献。
    """
    print("\n" + "=" * 72)
    print("H1: Reward magnitudes")
    print("=" * 72)

    components = {
        "time": [], "balance": [], "match": [],
        "warm": [], "batch": [], "affinity": [],
    }
    weights = {"time": 0.30, "balance": 0.15, "match": 0.10,
               "warm": 0.15, "batch": 0.20, "affinity": 0.10}

    # 把 calculate_reward 拆开，复刻其内部逻辑获取各分量
    rng = random.Random(0)
    for _ in range(n):
        _, task = rng.choice(scenarios)
        server = rng.choice(list(sim.servers.values()))
        transfer = rng.uniform(0.5, 5.0)

        exec_time = task.workload / max(server.total_compute, 1e-6)
        total_time = exec_time + transfer
        worst = task.workload / 10.0 + 5.0
        components["time"].append(max(1.0 - total_time / worst, -1.0))

        cu = server.used_compute / max(server.total_compute, 1e-6)
        components["balance"].append(1.0 - cu)
        components["match"].append(
            min(task.compute_demand / max(server.total_compute, 1e-6), 1.0))

        # warm
        if task.model_id is None or task.model_id not in CATALOG:
            warm = 0.0
        elif task.model_id in server.loaded_models:
            warm = 1.0
        else:
            cold_sec = CATALOG[task.model_id].cold_load_sec
            warm = max(-1.0, -cold_sec / 20.0)
        components["warm"].append(warm)

        # batch
        kind = getattr(task, "kind", TaskKind.GENERIC)
        if kind in (TaskKind.PREFILL, TaskKind.DECODE) and task.model_id in CATALOG:
            existing = server._current_batch_size(task.model_id, kind)
            max_b = CATALOG[task.model_id].max_batch_size
            if existing == 0:
                batch = 0.0
            elif existing >= max_b:
                batch = -0.5
            else:
                w = 1.0 if kind == TaskKind.DECODE else 0.4
                batch = min(1.0, w * existing / max(max_b, 1) * 4.0)
        else:
            batch = 0.0
        components["batch"].append(batch)

        # affinity
        affinity = 0.0
        if kind == TaskKind.DECODE:
            for dep_id in task.dependencies:
                dep = sim.tasks.get(dep_id)
                if dep is not None and dep.assigned_server == server.server_id:
                    affinity = 1.0
                    break
            else:
                if task.output_size > 0.1:
                    affinity = -0.5
        components["affinity"].append(affinity)

    # 输出
    print(f"\n{'Component':<10}  {'raw_mean':>10}  {'raw_std':>10}  "
          f"{'weight':>8}  {'wgt_mean':>10}  {'wgt_std':>10}  "
          f"{'frac_var':>10}")
    print("-" * 80)
    total_var = 0.0
    weighted_vars = {}
    for k in ["time", "balance", "match", "warm", "batch", "affinity"]:
        arr = np.array(components[k])
        wgt = weights[k]
        wgt_arr = arr * wgt
        weighted_vars[k] = float(np.var(wgt_arr))
        total_var += weighted_vars[k]
    for k in ["time", "balance", "match", "warm", "batch", "affinity"]:
        arr = np.array(components[k])
        wgt = weights[k]
        wgt_arr = arr * wgt
        frac = weighted_vars[k] / max(total_var, 1e-9)
        print(f"{k:<10}  {arr.mean():>10.4f}  {arr.std():>10.4f}  "
              f"{wgt:>8.2f}  {wgt_arr.mean():>10.4f}  {wgt_arr.std():>10.4f}  "
              f"{frac:>10.2%}")

    print("\n解读：")
    print("  - 'wgt_std' 越大，该项对最终 reward 的影响力越大")
    print("  - 'frac_var' 是该项加权后对总 reward 方差的贡献占比")
    print("  - 如果 AIGC 三项 (warm/batch/affinity) 的 frac_var 总和 < 30%，"
          "说明信号被 base reward 淹没")


# ====================================================================
# H2: Reward 项的两两相关性
# ====================================================================

def diagnose_reward_correlations(rl, sim, scenarios, n=500):
    print("\n" + "=" * 72)
    print("H2: Reward component correlations")
    print("=" * 72)

    components = {k: [] for k in ["time", "balance", "match", "warm", "batch", "affinity"]}
    rng = random.Random(1)
    weights = {"time": 0.30, "balance": 0.15, "match": 0.10,
               "warm": 0.15, "batch": 0.20, "affinity": 0.10}

    for _ in range(n):
        _, task = rng.choice(scenarios)
        server = rng.choice(list(sim.servers.values()))
        transfer = rng.uniform(0.5, 5.0)
        # 让一些 server 上有模型加载
        if rng.random() < 0.3 and task.model_id:
            server.loaded_models[task.model_id] = 0.0
        else:
            server.loaded_models.pop(task.model_id, None)

        # 复用 H1 的计算
        exec_time = task.workload / max(server.total_compute, 1e-6)
        total_time = exec_time + transfer
        worst = task.workload / 10.0 + 5.0
        components["time"].append(max(1.0 - total_time / worst, -1.0))
        cu = server.used_compute / max(server.total_compute, 1e-6)
        components["balance"].append(1.0 - cu)
        components["match"].append(
            min(task.compute_demand / max(server.total_compute, 1e-6), 1.0))
        if task.model_id in server.loaded_models:
            warm = 1.0
        elif task.model_id in CATALOG:
            warm = max(-1.0, -CATALOG[task.model_id].cold_load_sec / 20.0)
        else:
            warm = 0.0
        components["warm"].append(warm)
        components["batch"].append(0.0)
        components["affinity"].append(0.0)

    arrs = {k: np.array(v) for k, v in components.items()}
    print(f"\n{'':<10}", end="")
    for k in components:
        print(f"{k:>10}", end="")
    print()
    for k1 in components:
        print(f"{k1:<10}", end="")
        for k2 in components:
            if arrs[k1].std() < 1e-9 or arrs[k2].std() < 1e-9:
                c = float("nan")
            else:
                c = float(np.corrcoef(arrs[k1], arrs[k2])[0, 1])
            print(f"{c:>10.2f}", end="")
        print()
    print("\n解读：")
    print("  - 若 warm 与 time 高度正相关，说明 warm_bonus 没带新信息")
    print("    （选 warm server 已经隐含 time 优势）")


# ====================================================================
# H3: 预训练后的策略熵 (policy entropy)
# ====================================================================

def diagnose_policy_entropy(rl, sim, scenarios):
    print("\n" + "=" * 72)
    print("H3: Policy entropy (after pretrain)")
    print("=" * 72)
    print("\nPolicy 在每个场景下对 5 个 server 的概率分布：")
    print("(如果分布过尖锐 → policy 已经'选定'答案，后续 reward 改不动)")
    print()

    for spec, task in scenarios:
        model, prompt, output, kind_name = spec
        state = rl.state_encoder.encode(task, ready_task_count=5)
        with torch.no_grad():
            probs, _, _ = rl.policy(state.unsqueeze(0))
        probs = probs.squeeze(0).numpy()
        # 计算熵
        entropy = -float((probs * np.log(probs + 1e-10)).sum())
        max_entropy = float(np.log(len(probs)))  # uniform 时的最大熵
        print(f"  {model:<10} {kind_name:<8} prompt={prompt:>4} output={output:>4}: "
              f"probs=[{', '.join(f'{p:.3f}' for p in probs)}]  "
              f"entropy={entropy:.3f} / max={max_entropy:.3f} "
              f"({100*entropy/max_entropy:.0f}%)")

    print("\n解读：")
    print("  - 100% = uniform random（policy 不区分服务器）")
    print("  - <30% = policy 已经过度确定（很难再被 reward 推动）")
    print("  - 30-80% = healthy exploration")


# ====================================================================
# H4: State 中 AIGC 信号是否影响 policy 输出
# ====================================================================

def diagnose_aigc_signal_impact(rl, sim, scenarios):
    print("\n" + "=" * 72)
    print("H4: 修改 state 的 AIGC 信号，看 policy 输出是否变化")
    print("=" * 72)

    # 取一个场景，构造两个 state：
    # 1) 正常 state
    # 2) 把所有"server-AIGC"特征清零（模拟"没有 AIGC 信号"）
    spec, task = scenarios[2]   # llama-70b decode（AIGC 物理最强）
    model, prompt, output, kind_name = spec
    state_normal = rl.state_encoder.encode(task, ready_task_count=5)

    # 找出 AIGC 部分的索引并清零
    n_models = len(CATALOG)
    task_block_size = 10 + n_models   # 6 基础 + 3 kind one-hot + 1 kv + |M| model one-hot
    server_block_start = task_block_size + 2  # +2 全局
    n_servers = len(rl.state_encoder.server_ids)

    state_no_aigc = state_normal.clone()
    # 每个 server 的最后 5 维是 AIGC 特征
    for i in range(n_servers):
        start = server_block_start + i * 10 + 5
        state_no_aigc[start:start + 5] = 0.0

    with torch.no_grad():
        probs_normal, _, _ = rl.policy(state_normal.unsqueeze(0))
        probs_no_aigc, _, _ = rl.policy(state_no_aigc.unsqueeze(0))
    probs_normal = probs_normal.squeeze(0).numpy()
    probs_no_aigc = probs_no_aigc.squeeze(0).numpy()

    # KL divergence
    kl = float(np.sum(probs_normal * (np.log(probs_normal + 1e-10) - np.log(probs_no_aigc + 1e-10))))
    l1 = float(np.sum(np.abs(probs_normal - probs_no_aigc)))

    print(f"\nScenario: {model} {kind_name} prompt={prompt} output={output}")
    print(f"  normal probs    : [{', '.join(f'{p:.3f}' for p in probs_normal)}]")
    print(f"  no-AIGC probs   : [{', '.join(f'{p:.3f}' for p in probs_no_aigc)}]")
    print(f"  KL divergence   : {kl:.6f}")
    print(f"  L1 distance     : {l1:.6f}")

    print("\n解读：")
    print("  - L1 / KL 接近 0：policy 输出几乎不变 → 网络没在用 AIGC 信号")
    print("  - L1 > 0.2：policy 实质性使用了 AIGC 信号")


# ====================================================================
# main
# ====================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  RL Reward Signal Diagnostic")
    print("=" * 72)

    sim, rl = build_trained_rl(seed=0)
    scenarios = build_scenarios(sim)

    diagnose_reward_magnitudes(rl, sim, scenarios)
    diagnose_reward_correlations(rl, sim, scenarios)
    diagnose_policy_entropy(rl, sim, scenarios)
    diagnose_aigc_signal_impact(rl, sim, scenarios)

    print("\n" + "=" * 72)
    print("  Diagnostic complete.")
    print("=" * 72)
