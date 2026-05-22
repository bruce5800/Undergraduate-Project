"""
A3CR2NScheduler — Tuli et al. TPDS'20 "A3C + R2N2" 边缘-云调度算法的复现

参考文献：
  Tuli, S., Ilager, S., Ramamohanarao, K., & Buyya, R. (2022).
  "Dynamic Scheduling for Stochastic Edge-Cloud Computing Environments
   Using A3C Learning and Residual Recurrent Neural Networks."
  IEEE Transactions on Parallel and Distributed Systems, 33(4), 940–954.

本调度器是论文里 "AIGC-aware RL (ours) vs generic edge-cloud DRL" 这条对比的
**文献级实物对照**。与我们的 RLScheduler 相比有 3 个关键差异：

  1. 状态不含 AIGC 物理（无 model_id one-hot、kind、KV、batch_count、affinity）
     —— 验证"AIGC-aware state 的价值"
  2. 奖励不含 AIGC 行为奖励（无 warm / batch / affinity bonus）
     —— 验证"AIGC-aware reward 的价值"
  3. 网络换为 GRU + residual skip（即 R2N2 的精髓），更新为 A3C 风格
     （单 epoch PG + value MSE + entropy，**无 PPO clipping**）
     —— 论文原版方法学，保留 recurrent 时序建模能力

实现简化：
  - 论文中是异步多 actor + 全局参数同步；我们的仿真器是单进程，本实现用
    "单 actor + 同一损失" 等价简化（与单机 A2C 等价，是 A3C 的退化情形）
  - 论文中 GRU 隐状态在 episode 内连续传递；本实现在 schedule() 调用间保留隐状态，
    update_policy 时重置 hidden（避免 BPTT 复杂度，与作者公开实现保持一致风格）
  - 论文中 reward 含 energy / cost / migration 项；我们的仿真器只有 latency + util，
    用其前两项的相对权重近似（α=0.7 时间，β=0.3 负载均衡）
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim

from environment.task import TaskStatus
from environment.server import ServerType
from scheduler.base import BaseScheduler


logger = logging.getLogger(__name__)


# ================================================================
#  Generic 状态编码器（**不含** AIGC 特征）
# ================================================================

class GenericStateEncoder:
    """通用边缘-云状态编码：8 + 5 × num_servers 维。

    与我们升级版的 RL StateEncoder 形成对照：这个不暴露 AIGC 物理信号
    (model_id / kind / KV / batch_count / sibling_server)。
    """

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
        features = []
        # ---- 任务特征 6 维 ----
        features.append(task.compute_demand / 20.0)
        features.append(task.workload / 2000.0)
        features.append(task.input_size / 8.0)
        features.append(task.output_size / 0.3)
        features.append(len(task.dependencies) / 5.0)
        features.append(self._get_successor_count(task) / 5.0)

        # ---- 全局 2 维 ----
        features.append(ready_task_count / 50.0)
        features.append(len(self.sim.completed_tasks) / max(len(self.sim.tasks), 1))

        # ---- 每服务器 5 维 ----
        src = task.assigned_server if task.assigned_server is not None \
            else self._cloud_id
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
#  R2N2 网络：GRUCell + residual skip + Actor-Critic 头
# ================================================================

class R2N2ActorCritic(nn.Module):
    """Residual Recurrent NN：输入 → 投影 → GRUCell + residual skip → Actor / Critic。

    Recurrent layer 用 GRUCell 而非 LSTM——论文里也是 GRU。
    Skip connection 加 residual 让梯度反传更稳。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 投影
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        # Recurrent layer (R2N2 精髓)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        # Residual skip 投影
        self.skip = nn.Linear(hidden_dim, hidden_dim)
        # Actor
        self.actor = nn.Linear(hidden_dim, action_dim)
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, hidden=None, action_mask=None):
        h = torch.relu(self.norm(self.input_proj(x)))   # (B, H)
        if hidden is None or hidden.shape[0] != h.shape[0]:
            hidden = torch.zeros(h.shape[0], self.hidden_dim, device=h.device)
        new_hidden = self.gru(h, hidden)
        # Residual skip
        out = torch.relu(new_hidden + self.skip(h))
        logits = self.actor(out)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        probs = torch.softmax(logits, dim=-1)
        value = self.critic(out)
        return probs, value, new_hidden


# ================================================================
#  A3C 调度器
# ================================================================

class A3CR2NScheduler(BaseScheduler):

    def __init__(self, sim_env, pretrain_episodes: int = 5):
        super().__init__(sim_env)
        self.server_ids = sorted(sim_env.servers.keys())
        self.encoder = GenericStateEncoder(sim_env)

        state_dim = self.encoder.state_dim
        action_dim = len(self.server_ids)
        self.policy = R2N2ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        # A3C 超参（注意：无 PPO clipping）
        self.gamma = 0.95
        self.entropy_coeff = 0.05
        self.value_coeff = 0.5
        self.max_grad_norm = 0.5

        # 轨迹缓存
        self.trajectory = []
        self.update_interval = 32

        # GRU 隐状态（episode 内连续传递）
        self.hidden = None

        # 缓存云服务器
        self._cloud_id = next(
            (s.server_id for s in sim_env.servers.values()
             if s.type == ServerType.CLOUD),
            self.server_ids[0]
        )

        # 多轮预训练
        if pretrain_episodes > 0 and len(sim_env.tasks) > 0:
            self._pretrain(pretrain_episodes)

    # ----------------------------------------------------------------
    #  动作选择
    # ----------------------------------------------------------------

    def _get_action_mask(self, task) -> torch.BoolTensor:
        mask = torch.zeros(len(self.server_ids), dtype=torch.bool)
        for i, sid in enumerate(self.server_ids):
            if self.sim.servers[sid].can_allocate(task):
                mask[i] = True
        return mask

    def _select_action(self, state, task):
        mask = self._get_action_mask(task)
        if not mask.any():
            return None, None, None, None

        with torch.no_grad():
            probs, value, self.hidden = self.policy(
                state.unsqueeze(0), self.hidden, mask.unsqueeze(0))
            probs = probs.squeeze(0)
            value = value.squeeze(0)

        action_idx = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action_idx] + 1e-8)
        return action_idx, self.server_ids[action_idx], log_prob, value

    # ----------------------------------------------------------------
    #  奖励（generic：只有时间 + 负载均衡，**无** AIGC bonus）
    # ----------------------------------------------------------------

    def _calculate_reward(self, task, server, transfer_time):
        exec_time = task.workload / max(server.total_compute, 1e-6)
        total_time = exec_time + transfer_time
        worst_time = task.workload / 10.0 + 5.0
        time_reward = max(1.0 - total_time / worst_time, -1.0)

        compute_util = server.used_compute / max(server.total_compute, 1e-6)
        balance_reward = 1.0 - compute_util

        return 0.7 * time_reward + 0.3 * balance_reward

    # ----------------------------------------------------------------
    #  PPO 之外：A3C-style 单 epoch 更新（无 clipping）
    # ----------------------------------------------------------------

    def _update_policy(self):
        if len(self.trajectory) < self.update_interval:
            return

        states, actions, rewards, _, values, dones = zip(*self.trajectory)
        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        values = torch.cat(values).detach()
        dones = torch.FloatTensor(dones)

        # n-step return（A3C 风格，无 GAE）
        returns = torch.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running * (1 - dones[t].item())
            returns[t] = running
        advantages = returns - values
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 单 epoch 更新（不同于 PPO 的多 epoch + clipping）
        # 注意：批更新时不传 hidden，避免 BPTT 复杂度；这是一种简化
        probs_all, value_pred, _ = self.policy(states, hidden=None)
        chosen_probs = probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        new_log_probs = torch.log(chosen_probs + 1e-8)

        actor_loss = -(new_log_probs * advantages.detach()).mean()
        critic_loss = nn.MSELoss()(value_pred.squeeze(), returns)
        entropy = -(probs_all * torch.log(probs_all + 1e-8)).sum(dim=-1).mean()

        loss = (actor_loss
                + self.value_coeff * critic_loss
                - self.entropy_coeff * entropy)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.trajectory.clear()
        self.hidden = None   # episode-step 间隔不重要时重置

    # ----------------------------------------------------------------
    #  预训练（与 RLScheduler 同结构，便于公平对比）
    # ----------------------------------------------------------------

    def _pretrain(self, episodes: int):
        logger.info(f"A3C-R2N2 预训练: {episodes} 轮, {len(self.sim.tasks)} 个任务")
        for ep in range(episodes):
            self._reset_environment()
            current_time = 0.0
            max_time = 10000.0
            while (len(self.sim.completed_tasks) < len(self.sim.tasks)
                   and current_time < max_time):
                self.sim.step(self, current_time)
                current_time += 0.1
            if self.trajectory:
                self._force_update()
            logger.info(f"  预训练轮次 {ep+1}/{episodes}: "
                        f"完成 {len(self.sim.completed_tasks)}/{len(self.sim.tasks)}, "
                        f"makespan={current_time:.1f}")
        self._reset_environment()
        # 预训练完后降低探索
        self.entropy_coeff = 0.01

    def _reset_environment(self):
        for task in self.sim.tasks.values():
            task.status = TaskStatus.WAITING
            task.ready_time = None
            task.start_time = None
            task.end_time = None
            task.assigned_server = None
            task.transfer_delay = 0.0
        for server in self.sim.servers.values():
            server.used_compute = 0.0
            server.used_memory = 0.0
            server.used_storage = 0.0
            server.running_tasks.clear()
            server.task_queue.clear()
            server.task_history.clear()
            # 重置 M1 模型状态，避免预训练污染正式跑
            server.loaded_models.clear()
            server.model_refs.clear()
            server.weight_vram_used = 0.0
            # E1 修复：能耗也要重置
            server.accumulated_energy_J = 0.0
        self.sim.completed_tasks.clear()
        self.hidden = None

    def _force_update(self):
        if len(self.trajectory) < 2:
            self.trajectory.clear()
            return
        saved = self.update_interval
        self.update_interval = 2
        self._update_policy()
        self.update_interval = saved

    # ----------------------------------------------------------------
    #  调度入口
    # ----------------------------------------------------------------

    def schedule(self):
        ready = [t for t in self.sim.tasks.values()
                 if t.status == TaskStatus.READY]
        if not ready:
            return
        ready_count = len(ready)

        for task in ready:
            state = self.encoder.encode(task, ready_task_count=ready_count)
            action_idx, server_id, log_prob, value = \
                self._select_action(state, task)
            if action_idx is None:
                continue

            target_server = self.sim.servers[server_id]
            # 防御：action mask 应已挡掉非法动作；万一漏了，给个负奖励
            if not target_server.can_allocate(task):
                self.trajectory.append(
                    (state, action_idx, -1.0, log_prob, value, False))
                continue

            src = task.assigned_server if task.assigned_server is not None \
                else self._cloud_id
            transfer_time = self.sim.network.estimate_transfer_time(
                src, server_id, task.output_size)

            task.assigned_server = server_id
            task.transfer_delay = transfer_time
            effective_priority = 1.0 / max(transfer_time, 1e-6)
            target_server.add_task(task, priority=effective_priority)

            reward = self._calculate_reward(task, target_server, transfer_time)
            self.trajectory.append(
                (state, action_idx, reward, log_prob, value, False))

        self._update_policy()
