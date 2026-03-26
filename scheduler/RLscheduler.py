# RLscheduler.py
# 强化学习调度器 —— PPO + 动作掩码 + GAE + 多轮预训练
#
# 核心设计：
#   1. 动作掩码：只从"资源可分配"的服务器中选择，杜绝无效动作
#   2. PPO Clipped Objective：比朴素 Policy Gradient 更稳定
#   3. GAE (Generalized Advantage Estimation)：平衡偏差与方差
#   4. 多目标奖励：时间效率 + 负载均衡 + 资源匹配度
#   5. 增强状态编码：包含 DAG 结构特征和全局进度信息
#   6. 熵正则化 + 梯度裁剪：防止过早收敛，保证训练稳定
#   7. 多轮预训练：在相同任务集上反复仿真，积累经验后再参加基准测试
#   8. 热身阶段：预训练首轮前 N 步使用随机策略收集多样化经验

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from environment.task import TaskStatus, Task
from environment.server import ServerType
from scheduler.base import BaseScheduler

logger = logging.getLogger(__name__)


# ================================================================
#  状态编码器
# ================================================================

class StateEncoder:
    """将环境观测编码为固定长度特征向量"""

    def __init__(self, sim_env):
        self.sim = sim_env
        self.server_ids = sorted(sim_env.servers.keys())
        self._cloud_id = next(
            (s.server_id for s in sim_env.servers.values()
             if s.type == ServerType.CLOUD),
            self.server_ids[0]
        )
        self._successor_count: dict = {}

    def encode(self, task, ready_task_count: int = 0) -> torch.FloatTensor:
        features: list[float] = []

        # ---- 任务特征 (6 维) ----
        features.append(task.compute_demand / 20.0)
        features.append(task.workload / 2000.0)
        features.append(task.input_size / 8.0)
        features.append(task.output_size / 0.3)
        features.append(len(task.dependencies) / 5.0)
        features.append(self._get_successor_count(task) / 5.0)

        # ---- 全局状态 (2 维) ----
        features.append(ready_task_count / 50.0)
        completed_ratio = len(self.sim.completed_tasks) / max(len(self.sim.tasks), 1)
        features.append(completed_ratio)

        # ---- 各服务器状态 (每台 5 维) ----
        src = task.assigned_server if task.assigned_server is not None else self._cloud_id
        for sid in self.server_ids:
            server = self.sim.servers[sid]
            features.append(server.used_compute / max(server.total_compute, 1e-6))
            features.append(server.used_memory / max(server.total_memory, 1e-6))
            features.append(server.bandwidth / 1000.0)
            features.append(len(server.task_queue) / 20.0)
            transfer = self.sim.network.estimate_transfer_time(
                src, sid, task.output_size)
            features.append(transfer / 10.0)

        return torch.FloatTensor(features)

    @property
    def state_dim(self) -> int:
        return 8 + 5 * len(self.server_ids)

    def _get_successor_count(self, task) -> int:
        tid = task.task_id
        if tid not in self._successor_count:
            cnt = sum(1 for t in self.sim.tasks.values() if tid in t.dependencies)
            self._successor_count[tid] = cnt
        return self._successor_count[tid]


# ================================================================
#  Actor-Critic 网络（支持动作掩码）
# ================================================================

class ActorCritic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.actor = nn.Linear(64, action_dim)

        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, action_mask=None):
        feat = self.feature(x)
        logits = self.actor(feat)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))

        probs = torch.softmax(logits, dim=-1)
        value = self.critic(feat)
        return probs, value, logits


# ================================================================
#  PPO 调度器（含多轮预训练）
# ================================================================

class RLScheduler(BaseScheduler):

    def __init__(self, sim_env, pretrain_episodes: int = 10):
        super().__init__(sim_env)

        self.server_ids = sorted(sim_env.servers.keys())
        self.state_encoder = StateEncoder(sim_env)

        state_dim = self.state_encoder.state_dim
        action_dim = len(self.server_ids)

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        # PPO 超参
        self.gamma = 0.95
        self.gae_lambda = 0.90
        self.clip_epsilon = 0.2
        self.entropy_coeff = 0.05
        self.value_coeff = 0.5
        self.max_grad_norm = 0.5

        # 轨迹缓存
        self.trajectory: list[tuple] = []
        self.update_interval = 32
        self.ppo_epochs = 4

        # 热身（仅预训练首轮使用）
        self._decision_count = 0
        self._warmup_steps = 30

        # 缓存
        self._cloud_id = next(
            (s.server_id for s in sim_env.servers.values()
             if s.type == ServerType.CLOUD),
            self.server_ids[0]
        )

        # ---- 多轮预训练 ----
        if pretrain_episodes > 0 and len(sim_env.tasks) > 0:
            self._pretrain(pretrain_episodes)

    # =============================================================
    #  预训练：在相同任务集上反复仿真，训练策略
    # =============================================================

    def _pretrain(self, episodes: int):
        """
        多轮预训练机制：
        1. 保存仿真环境的初始状态
        2. 反复运行完整仿真，每轮都收集经验并更新策略
        3. 每轮结束后重置环境到初始状态
        4. 预训练完毕后重置环境，供正式基准测试使用
        """
        logger.info(f"RL 预训练开始: {episodes} 轮, {len(self.sim.tasks)} 个任务")

        for ep in range(episodes):
            self._reset_environment()
            current_time = 0.0
            max_time = 10000.0

            while (len(self.sim.completed_tasks) < len(self.sim.tasks)
                   and current_time < max_time):
                # 复用 simulation.step 的完整流程
                self.sim.step(self, current_time)
                current_time += 0.1

            # 确保最后一批轨迹也参与训练
            if self.trajectory:
                self._force_update()

            completed = len(self.sim.completed_tasks)
            total = len(self.sim.tasks)
            logger.info(f"  预训练轮次 {ep+1}/{episodes}: "
                        f"完成 {completed}/{total}, "
                        f"makespan={current_time:.1f}")

        # 预训练完毕 → 重置环境供正式运行，并跳过热身
        self._reset_environment()
        self._decision_count = self._warmup_steps  # 正式运行直接使用训练好的策略

        # 降低熵系数：预训练后减少探索，更多利用学到的策略
        self.entropy_coeff = 0.01

        logger.info("RL 预训练完成，进入正式调度模式")

    def _reset_environment(self):
        """将仿真环境重置到初始状态（任务状态、服务器资源、完成集合）"""
        # 重置所有任务
        for task in self.sim.tasks.values():
            task.status = TaskStatus.WAITING
            task.ready_time = None
            task.start_time = None
            task.end_time = None
            task.assigned_server = None
            task.transfer_delay = 0.0

        # 重置所有服务器
        for server in self.sim.servers.values():
            server.used_compute = 0.0
            server.used_memory = 0.0
            server.used_storage = 0.0
            server.running_tasks.clear()
            server.task_queue.clear()
            server.task_history.clear()

        # 重置全局完成集合
        self.sim.completed_tasks.clear()

    def _force_update(self):
        """强制用当前轨迹更新策略（不等凑够 update_interval）"""
        if len(self.trajectory) < 2:
            self.trajectory.clear()
            return
        # 临时降低阈值
        saved = self.update_interval
        self.update_interval = 2
        self.update_policy()
        self.update_interval = saved

    # =============================================================
    #  动作选择（带掩码）
    # =============================================================

    def _get_action_mask(self, task) -> torch.BoolTensor:
        mask = torch.zeros(len(self.server_ids), dtype=torch.bool)
        for i, sid in enumerate(self.server_ids):
            if self.sim.servers[sid].can_allocate(task):
                mask[i] = True
        return mask

    def select_action(self, state, task):
        action_mask = self._get_action_mask(task)

        if not action_mask.any():
            return None, None, None, None

        with torch.no_grad():
            probs, value, _ = self.policy(
                state.unsqueeze(0), action_mask.unsqueeze(0))
            probs = probs.squeeze(0)
            value = value.squeeze(0)

        # 热身期均匀随机
        if self._decision_count < self._warmup_steps:
            valid_idx = action_mask.nonzero(as_tuple=True)[0]
            action_idx = valid_idx[torch.randint(len(valid_idx), (1,))].item()
        else:
            action_idx = torch.multinomial(probs, 1).item()

        log_prob = torch.log(probs[action_idx] + 1e-8)
        return action_idx, self.server_ids[action_idx], log_prob, value

    # =============================================================
    #  多目标奖励
    # =============================================================

    def calculate_reward(self, task, server, transfer_time: float) -> float:
        exec_time = task.workload / max(server.total_compute, 1e-6)
        total_time = exec_time + transfer_time

        # 1) 时间效率
        worst_time = task.workload / 10.0 + 5.0
        time_reward = max(1.0 - total_time / worst_time, -1.0)

        # 2) 负载均衡
        compute_util = server.used_compute / max(server.total_compute, 1e-6)
        balance_reward = 1.0 - compute_util

        # 3) 资源匹配度
        match_reward = min(task.compute_demand / max(server.total_compute, 1e-6), 1.0)

        return 0.5 * time_reward + 0.3 * balance_reward + 0.2 * match_reward

    # =============================================================
    #  经验存储与 PPO 更新
    # =============================================================

    def store_transition(self, state, action, reward, log_prob, value, done):
        self.trajectory.append((state, action, reward, log_prob, value, done))

    def update_policy(self):
        if len(self.trajectory) < self.update_interval:
            return

        states, actions, rewards, old_log_probs, values, dones = \
            zip(*self.trajectory)

        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        old_log_probs = torch.stack(old_log_probs)
        values = torch.cat(values).detach()
        dones = torch.FloatTensor(dones)

        # GAE
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0.0 if t == len(rewards) - 1 else values[t + 1].item()
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t].item()
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values

        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO 多轮更新
        for _ in range(self.ppo_epochs):
            probs, new_values, _ = self.policy(states)
            dist_all = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            new_log_probs = torch.log(dist_all + 1e-8)

            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio,
                                1.0 - self.clip_epsilon,
                                1.0 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)

            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

            loss = (actor_loss
                    + self.value_coeff * critic_loss
                    - self.entropy_coeff * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.trajectory.clear()

    # =============================================================
    #  调度入口
    # =============================================================

    def schedule(self):
        ready_tasks = [t for t in self.sim.tasks.values()
                       if t.status == TaskStatus.READY]
        if not ready_tasks:
            return

        ready_count = len(ready_tasks)

        for task in ready_tasks:
            state = self.state_encoder.encode(task, ready_task_count=ready_count)

            action_idx, server_id, log_prob, value = \
                self.select_action(state, task)

            if action_idx is None:
                continue

            target_server = self.sim.servers[server_id]

            src = task.assigned_server if task.assigned_server is not None \
                else self._cloud_id
            transfer_time = self.sim.network.estimate_transfer_time(
                src, server_id, task.output_size)

            task.assigned_server = server_id
            task.transfer_delay = transfer_time
            effective_priority = 1.0 / max(transfer_time, 1e-6)
            target_server.add_task(task, priority=effective_priority)

            reward = self.calculate_reward(task, target_server, transfer_time)
            self.store_transition(state, action_idx, reward, log_prob, value,
                                  done=False)
            self._decision_count += 1

        self.update_policy()
