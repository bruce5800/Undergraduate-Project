"""
PerLLMScheduler — CS-UCB (Constraint-Satisfaction UCB) bandit 基线

R1 修订新增：复现 PerLLM (Yang et al., arXiv:2405.14636) 的调度算法骨架，
适配到本仿真器的逐任务派发粒度。PerLLM 是与本文最接近的先验工作
（edge-cloud + 学习式 + 能耗一等目标 + 逐请求），差异在于它用约束满足
组合 bandit 而非深度 RL，且不向策略暴露 serving 物理。

与原文的对应关系：
  - 组合多臂 bandit（原文 §3.2 CMAB）：每个 (model, kind, server) 组合是
    一个 arm——按服务类型独立统计对应 PerLLM 的 "personalized" 语义
  - 约束满足机制（原文式 (3)）：can_allocate 做硬过滤（对应 C2 算力 /
    C3 带宽显存 / C4 单服务器约束）；处理时限作为软约束进 f(y)（对应 C1）
  - CS-UCB 选择（原文式 (6)）：可行 arm 中取
    UCB(a,t) = R_bar(a) + delta * sqrt(ln t / L(a)) 最大者
  - 奖励（原文式 (2)(4)）：r = -E_norm + lam * f(y)，其中 E_norm 是
    该决策的归一化能耗估计（传输 + 推理 + 冷加载时段），f(y) 是三项
    归一化裕度的最小值

与 RL (ours) 的公平性约定：CS-UCB 与其他 baseline 一样，只使用通用可观测
接口（算力/显存/带宽/队列），不读取 loaded_models / batch 占用 / sibling
等 AIGC 状态特征——这正是"物理暴露与否"这一对比轴的定义。

超参 delta（探索系数）与 lam（约束项权重）经 {0.5,1,2}x{0.5,1} 网格调优。
"""

import math
import logging

from environment.task import TaskStatus
from environment.server import ServerType
from environment.energy import POWER_PROFILES
from scheduler.base import BaseScheduler


class CSUCBScheduler(BaseScheduler):
    logger = logging.getLogger(__name__)

    def __init__(self, sim_env, delta: float = 1.0, lam: float = 1.0):
        super().__init__(sim_env)
        self.delta = delta
        self.lam = lam

        self.server_ids = sorted(sim_env.servers.keys())
        self._cloud_id = next(
            (s.server_id for s in sim_env.servers.values()
             if s.type == ServerType.CLOUD),
            self.server_ids[0]
        )

        # bandit 统计: arm key -> (选择次数 L, 平均奖励 R_bar)
        self._counts: dict = {}
        self._means: dict = {}
        self._t = 0  # 全局决策计数

    # ----------------------------------------------------------------
    #  成本估计（只用通用接口，和 HEFT/GA/PSO 同一信息量）
    # ----------------------------------------------------------------

    def _power_est(self, server, util_after: float) -> float:
        """放置后瞬时功率估计 (W)；无 profile 时退化为 0。"""
        p = POWER_PROFILES.get(getattr(server, "power_profile", None))
        if p is None:
            return 0.0
        u = min(max(util_after, 0.0), 1.0)
        return p["idle_W"] + (p["max_W"] - p["idle_W"]) * u

    def _estimate(self, task, server):
        """返回 (total_time_est, energy_est, f_y)。"""
        src = task.assigned_server if task.assigned_server is not None \
            else self._cloud_id
        transfer = self.sim.network.estimate_transfer_time(
            src, server.server_id, task.output_size)
        cold = server.cold_load_cost(task.model_id) \
            if task.model_id is not None else 0.0
        exec_time = task.workload / max(server.total_compute, 1e-6)
        total_time = transfer + cold + exec_time

        util_after = ((server.used_compute + task.compute_demand)
                      / max(server.total_compute, 1e-6))
        energy = total_time * self._power_est(server, util_after)

        # f(y) = min(时间裕度, 算力裕度, 显存裕度)，对应原文式 (3)
        worst_time = task.workload / 10.0 + 5.0   # 与 RL time_reward 同一归一化
        slack_time = max(-1.0, min(1.0, 1.0 - total_time / worst_time))
        slack_compute = max(0.0, min(1.0, (
            server.total_compute - server.used_compute - task.compute_demand)
            / max(server.total_compute, 1e-6)))
        free_mem = (server.total_memory
                    - getattr(server, "weight_vram_used", 0.0)
                    - server.used_memory)
        need_mem = getattr(task, "kv_cache_GB", 0.0)
        slack_mem = max(0.0, min(1.0, (free_mem - need_mem)
                                 / max(server.total_memory, 1e-6)))
        f_y = min(slack_time, slack_compute, slack_mem)
        return total_time, energy, f_y

    # ----------------------------------------------------------------
    #  CS-UCB 决策
    # ----------------------------------------------------------------

    def _arm_key(self, task, server_id):
        return (task.model_id, getattr(task, "kind", None), server_id)

    def _select(self, task):
        """约束过滤 -> 可行 arm 中 UCB 最大者。返回 (server, est) 或 None。"""
        feasible = [s for s in self.sim.servers.values()
                    if s.can_allocate(task)]
        if not feasible:
            return None

        self._t += 1
        ests = {s.server_id: self._estimate(task, s) for s in feasible}
        e_worst = max(e for (_, e, _) in ests.values()) or 1.0

        best, best_ucb = None, -math.inf
        for s in feasible:
            key = self._arm_key(task, s.server_id)
            n = self._counts.get(key, 0)
            if n == 0:
                ucb = math.inf   # 未探索的 arm 强制探索
            else:
                ucb = (self._means[key]
                       + self.delta * math.sqrt(math.log(self._t) / n))
            if ucb > best_ucb:
                best, best_ucb = s, ucb

        # 观测奖励并更新该 arm 统计（原文式 (4)：负能耗 + lam * f(y)）
        total_time, energy, f_y = ests[best.server_id]
        reward = -(energy / max(e_worst, 1e-6)) + self.lam * f_y
        key = self._arm_key(task, best.server_id)
        n = self._counts.get(key, 0)
        self._means[key] = (self._means.get(key, 0.0) * n + reward) / (n + 1)
        self._counts[key] = n + 1
        return best, total_time

    def schedule(self):
        ready_tasks = [t for t in self.sim.tasks.values()
                       if t.status == TaskStatus.READY]
        if not ready_tasks:
            return

        for task in ready_tasks:
            picked = self._select(task)
            if picked is None:
                continue     # 本轮无可行服务器，下轮再试
            target, _ = picked

            src = task.assigned_server if task.assigned_server is not None \
                else self._cloud_id
            transfer_time = self.sim.network.estimate_transfer_time(
                src, target.server_id, task.output_size)

            task.assigned_server = target.server_id
            task.transfer_delay = transfer_time
            effective_priority = 1.0 / max(transfer_time, 1e-6)
            target.add_task(task, priority=effective_priority)
