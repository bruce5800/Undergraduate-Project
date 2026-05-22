"""
GNNScheduler — 基于图神经网络的 AIGC-aware 调度器

核心想法（解决 MLP RL 的"cloud collapse"问题）：
  把"调度决策"建模为一张小图——
    nodes : 1 个 task 节点 + N 个 server 节点
    edges : task → 每个 server 一条，**边特征编码 AIGC 物理**：
            transfer_time, cold_load_cost, is_my_model_loaded,
            is_sibling_server, batch_count_norm, can_allocate

  GNN 强迫网络做"关系性"比较，而非数值大小比较。
  AIGC 信号进入边特征 → 进入消息传递 → 网络结构上不可绕过。

  这是 Decima (SIGCOMM'19) 的方法学在 AIGC 推理场景的应用。

参考：
  Mao, H. et al. "Learning Scheduling Algorithms for Data Processing
  Clusters." SIGCOMM 2019.

架构：
  GNNStateEncoder  → 产 (task_feat, server_feats, edge_feats)
  GNNActorCritic   → 1-hop GAT-style message passing + attention 输出
  GNNScheduler     → PPO 训练循环（复用 RLscheduler 的 pretrain/ablation/update 结构）

实现亮点：纯 PyTorch 实现，**不依赖 PyG / DGL**——graph 小（typical N=5-8），
矩阵操作完全可写出来。
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from environment.task import TaskStatus, Task, TaskKind
from environment.server import ServerType
from environment.model_catalog import CATALOG
from scheduler.base import BaseScheduler


logger = logging.getLogger(__name__)


# ================================================================
#  GNN 状态编码器：结构化输入
# ================================================================

class GNNStateEncoder:
    """把 sim 状态编码为 (task_feat, server_feats, edge_feats) 三元组。

    与 MLP 版的 flat state 不同：
      - 任务特征单独成向量
      - 每台服务器一个特征向量（堆成 2D tensor）
      - 每条 task→server 边一个特征向量（堆成 2D tensor）

    Edge features 是 GNN 与 MLP 最大的差异之处 —— AIGC 物理（cold_load,
    sibling, batch_count）进入边而非节点，网络必须经过消息传递才能用上。
    """

    _MODEL_IDS = sorted(CATALOG.keys())
    _MODEL_INDEX = {mid: i for i, mid in enumerate(_MODEL_IDS)}

    def __init__(self, sim_env):
        self.sim = sim_env
        self.server_ids = sorted(sim_env.servers.keys())
        self._cloud_id = next(
            (s.server_id for s in sim_env.servers.values()
             if s.type == ServerType.CLOUD),
            self.server_ids[0]
        )
        self._successor_count: dict = {}

    @property
    def task_dim(self) -> int:
        # 6 基础 + 3 kind one-hot + 1 kv + |M| model one-hot + 2 (prompt/output)
        return 6 + 3 + 1 + len(self._MODEL_IDS) + 2

    @property
    def server_dim(self) -> int:
        # compute_util, mem_util, vram_free, queue_len, bandwidth,
        # is_cloud, loaded_models_count
        return 7

    @property
    def edge_dim(self) -> int:
        # transfer_time, cold_load, is_my_model_loaded, is_sibling,
        # batch_count_norm, can_allocate
        return 6

    @property
    def num_servers(self) -> int:
        return len(self.server_ids)

    def encode(self, task, ready_task_count: int = 0):
        """返回 (task_feat, server_feats, edge_feats)。"""
        task_feat = self._task_features(task)
        server_feats = torch.stack([
            self._server_features(sid) for sid in self.server_ids])
        edge_feats = torch.stack([
            self._edge_features(task, sid) for sid in self.server_ids])
        return task_feat, server_feats, edge_feats

    # ---- 内部 ----

    def _task_features(self, task) -> torch.FloatTensor:
        f = [
            task.compute_demand / 20.0,
            task.workload / 2000.0,
            task.input_size / 8.0,
            task.output_size / 0.3,
            len(task.dependencies) / 5.0,
            self._succ_count(task) / 5.0,
        ]
        kind = getattr(task, "kind", TaskKind.GENERIC)
        f.append(1.0 if kind == TaskKind.GENERIC else 0.0)
        f.append(1.0 if kind == TaskKind.PREFILL else 0.0)
        f.append(1.0 if kind == TaskKind.DECODE else 0.0)
        f.append(getattr(task, "kv_cache_GB", 0.0) / 5.0)
        # Model one-hot
        model_onehot = [0.0] * len(self._MODEL_IDS)
        if task.model_id in self._MODEL_INDEX:
            model_onehot[self._MODEL_INDEX[task.model_id]] = 1.0
        f.extend(model_onehot)
        # prompt/output token counts (normalized)
        f.append(getattr(task, "prompt_tokens", 0) / 4096.0)
        f.append(getattr(task, "output_tokens", 0) / 2000.0)
        return torch.FloatTensor(f)

    def _server_features(self, sid) -> torch.FloatTensor:
        s = self.sim.servers[sid]
        vram_free = max(s.total_memory - s.used_memory - s.weight_vram_used, 0)
        return torch.FloatTensor([
            s.used_compute / max(s.total_compute, 1e-6),
            s.used_memory / max(s.total_memory, 1e-6),
            vram_free / max(s.total_memory, 1e-6),
            len(s.task_queue) / 20.0,
            s.bandwidth / 1000.0,
            1.0 if s.type == ServerType.CLOUD else 0.0,
            len(s.loaded_models) / max(len(self._MODEL_IDS), 1),
        ])

    def _edge_features(self, task, sid) -> torch.FloatTensor:
        s = self.sim.servers[sid]
        src = task.assigned_server if task.assigned_server is not None \
            else self._cloud_id
        transfer = self.sim.network.estimate_transfer_time(
            src, sid, task.output_size)
        cold_load = s.cold_load_cost(task.model_id) if task.model_id else 0.0
        is_loaded = (1.0 if (task.model_id
                              and task.model_id in s.loaded_models) else 0.0)
        # Sibling server (for DECODE: prefill's server)
        sib = 0.0
        if getattr(task, "kind", None) == TaskKind.DECODE:
            for dep_id in task.dependencies:
                dep = self.sim.tasks.get(dep_id)
                if dep is not None and dep.assigned_server == sid:
                    sib = 1.0
                    break
        # batch count for same model+kind
        kind = getattr(task, "kind", TaskKind.GENERIC)
        batch_count = s._current_batch_size(task.model_id, kind)
        max_b = (CATALOG[task.model_id].max_batch_size
                 if task.model_id in CATALOG else 1)
        batch_norm = batch_count / max(max_b, 1)
        # can_allocate
        can_alloc = 1.0 if s.can_allocate(task) else 0.0
        return torch.FloatTensor([
            transfer / 10.0,
            cold_load / 30.0,
            is_loaded,
            sib,
            batch_norm,
            can_alloc,
        ])

    def _succ_count(self, task) -> int:
        tid = task.task_id
        if tid not in self._successor_count:
            cnt = sum(1 for t in self.sim.tasks.values()
                      if tid in t.dependencies)
            self._successor_count[tid] = cnt
        return self._successor_count[tid]


# ================================================================
#  GNN Actor-Critic 网络
# ================================================================

class GNNActorCritic(nn.Module):
    """1-hop GAT-style message passing 上跑 actor + critic。

    Forward pipeline:
      task    ─► task_enc     ─► task_emb     (B, H)
      server  ─► server_enc   ─► server_emb   (B, S, H)
      edge    ─► edge_enc     ─► edge_emb     (B, S, H)

      msg = MLP([task_broadcast, server_emb, edge_emb])  (B, S, H)
      server_new = MLP([server_emb, msg])                (B, S, H)

      # Actor: task ↔ server attention
      logits = MLP([task_broadcast, server_new]) → (B, S)
      probs  = softmax(masked_logits)

      # Critic: pool servers + task → value
      value  = MLP([task_emb, mean_pool(server_new)])
    """

    def __init__(self, task_dim: int, server_dim: int, edge_dim: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.task_enc = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.server_enc = nn.Sequential(
            nn.Linear(server_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
        )

        # Message function
        self.msg_fn = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Server update
        self.server_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Actor head: task-server attention
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Critic head: pool + value
        self.critic = nn.Sequential(
            nn.Linear(2 * hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, task_feat, server_feats, edge_feats, action_mask=None):
        """
        task_feat:    (B, task_dim)
        server_feats: (B, S, server_dim)
        edge_feats:   (B, S, edge_dim)
        action_mask:  (B, S) bool, True = valid action
        """
        S = server_feats.shape[1]

        # 1. Encode
        task_emb = self.task_enc(task_feat)         # (B, H)
        server_emb = self.server_enc(server_feats)  # (B, S, H)
        edge_emb = self.edge_enc(edge_feats)        # (B, S, H)

        # 2. Broadcast task to each edge
        task_broad = task_emb.unsqueeze(1).expand(-1, S, -1)  # (B, S, H)

        # 3. Message passing: task → server
        msg_input = torch.cat([task_broad, server_emb, edge_emb], dim=-1)
        msg = self.msg_fn(msg_input)                            # (B, S, H)

        # 4. Server node update
        server_new = self.server_update(
            torch.cat([server_emb, msg], dim=-1))               # (B, S, H)

        # 5. Actor: task ↔ updated server attention
        attn_in = torch.cat([task_broad, server_new], dim=-1)   # (B, S, 2H)
        logits = self.attention(attn_in).squeeze(-1)            # (B, S)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        probs = torch.softmax(logits, dim=-1)

        # 6. Critic: pool servers + task → value
        pooled = server_new.mean(dim=1)                         # (B, H)
        value_in = torch.cat([task_emb, pooled], dim=-1)        # (B, 2H)
        value = self.critic(value_in)                           # (B, 1)

        return probs, value, logits


# ================================================================
#  GNN 调度器（PPO + pretrain + ablation 框架）
# ================================================================

class GNNScheduler(BaseScheduler):
    """GNN-based PPO scheduler, AIGC-aware via edge features."""

    def __init__(self, sim_env, pretrain_episodes: int = 20,
                 # 复用 RL 的 ablation 接口，便于对照实验
                 enable_warm_reward: bool = True,
                 enable_batch_reward: bool = True,
                 enable_affinity_reward: bool = True,
                 enable_aigc_state: bool = True,  # 控制是否清零 edge 中的 AIGC 维度
                 enable_action_mask: bool = True,
                 enable_gae: bool = True,
                 enable_pretrain: bool = True,
                 enable_entropy: bool = True,
                 enable_cloud_overuse: bool = True):
        super().__init__(sim_env)
        self.enable_warm_reward = enable_warm_reward
        self.enable_batch_reward = enable_batch_reward
        self.enable_affinity_reward = enable_affinity_reward
        self.enable_aigc_state = enable_aigc_state
        self.enable_action_mask = enable_action_mask
        self.enable_gae = enable_gae
        self.enable_pretrain = enable_pretrain
        self.enable_entropy = enable_entropy
        self.enable_cloud_overuse = enable_cloud_overuse

        self.server_ids = sorted(sim_env.servers.keys())
        self.encoder = GNNStateEncoder(sim_env)

        self.policy = GNNActorCritic(
            task_dim=self.encoder.task_dim,
            server_dim=self.encoder.server_dim,
            edge_dim=self.encoder.edge_dim,
            hidden_dim=64,
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        # PPO 超参（与 RLScheduler 对齐以保证公平对比）
        self.gamma = 0.95
        self.gae_lambda = 0.90
        self.clip_epsilon = 0.2
        self.entropy_coeff = 0.05 if enable_entropy else 0.0
        self.value_coeff = 0.5
        self.max_grad_norm = 0.5

        # 轨迹缓存：每项是 (task_feat, server_feats, edge_feats, action, reward, log_prob, value, done)
        self.trajectory: list[tuple] = []
        self.update_interval = 32
        self.ppo_epochs = 4

        self._decision_count = 0
        self._warmup_steps = 30

        self._cloud_id = next(
            (s.server_id for s in sim_env.servers.values()
             if s.type == ServerType.CLOUD),
            self.server_ids[0]
        )

        if enable_pretrain and pretrain_episodes > 0 and len(sim_env.tasks) > 0:
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

    def _maybe_zero_aigc_edges(self, edge_feats):
        """no_aigc_state ablation：把 edge_feats 中 AIGC 物理维度清零。

        Edge layout: [transfer, cold_load, is_loaded, sibling, batch, can_alloc]
                                  ↑─── AIGC 信号 ───↑
        保留 transfer 与 can_alloc（基础物理与有效性），其余清零。
        """
        if self.enable_aigc_state:
            return edge_feats
        z = edge_feats.clone()
        # 清零索引 1,2,3,4（cold_load, is_loaded, sibling, batch_norm）
        z[..., 1:5] = 0.0
        return z

    def _select_action(self, task_feat, server_feats, edge_feats, task):
        mask = self._get_action_mask(task)
        if not mask.any():
            return None, None, None, None

        edge_feats_used = self._maybe_zero_aigc_edges(edge_feats)

        with torch.no_grad():
            mask_arg = mask.unsqueeze(0) if self.enable_action_mask else None
            probs, value, _ = self.policy(
                task_feat.unsqueeze(0),
                server_feats.unsqueeze(0),
                edge_feats_used.unsqueeze(0),
                mask_arg,
            )
            probs = probs.squeeze(0)
            value = value.squeeze(0)

        if self._decision_count < self._warmup_steps:
            valid_idx = mask.nonzero(as_tuple=True)[0]
            action_idx = valid_idx[torch.randint(len(valid_idx), (1,))].item()
        else:
            action_idx = torch.multinomial(probs, 1).item()

        log_prob = torch.log(probs[action_idx] + 1e-8)
        return action_idx, self.server_ids[action_idx], log_prob, value

    # ----------------------------------------------------------------
    #  奖励（与 RLScheduler 同步设计，便于对照）
    # ----------------------------------------------------------------

    def _calculate_reward(self, task, server, transfer_time):
        exec_time = task.workload / max(server.total_compute, 1e-6)
        total_time = exec_time + transfer_time
        worst_time = task.workload / 10.0 + 5.0
        time_reward = max(1.0 - total_time / worst_time, -1.0)

        compute_util = server.used_compute / max(server.total_compute, 1e-6)
        balance_reward = 1.0 - compute_util

        match_reward = min(task.compute_demand / max(server.total_compute, 1e-6), 1.0)

        # AIGC bonuses
        if not self.enable_warm_reward:
            warm_bonus = 0.0
        elif task.model_id is None or task.model_id not in CATALOG:
            warm_bonus = 0.0
        elif task.model_id in server.loaded_models:
            warm_bonus = 1.0
        else:
            cold_sec = CATALOG[task.model_id].cold_load_sec
            warm_bonus = max(-1.0, -cold_sec / 20.0)

        kind = getattr(task, "kind", TaskKind.GENERIC)
        if not self.enable_batch_reward:
            batch_bonus = 0.0
        elif kind in (TaskKind.PREFILL, TaskKind.DECODE) \
                and task.model_id in CATALOG:
            existing = server._current_batch_size(task.model_id, kind)
            max_b = CATALOG[task.model_id].max_batch_size
            if existing == 0:
                batch_bonus = 0.0
            elif existing >= max_b:
                batch_bonus = -0.5
            else:
                w = 1.0 if kind == TaskKind.DECODE else 0.4
                batch_bonus = min(1.0, w * existing / max(max_b, 1) * 4.0)
        else:
            batch_bonus = 0.0

        affinity_bonus = 0.0
        if self.enable_affinity_reward and kind == TaskKind.DECODE:
            for dep_id in task.dependencies:
                dep = self.sim.tasks.get(dep_id)
                if dep is not None and dep.assigned_server == server.server_id:
                    affinity_bonus = 1.0
                    break
            else:
                if task.output_size > 0.1:
                    affinity_bonus = -0.5

        # Cloud overuse penalty
        cloud_overuse = 0.0
        if self.enable_cloud_overuse and server.type == ServerType.CLOUD:
            edge_can_serve = any(
                s.type == ServerType.EDGE and s.can_allocate(task)
                for s in self.sim.servers.values()
            )
            if edge_can_serve:
                cu = server.used_compute / max(server.total_compute, 1e-6)
                cloud_overuse = -cu

        return (0.25 * time_reward
                + 0.10 * balance_reward
                + 0.10 * match_reward
                + 0.15 * warm_bonus
                + 0.20 * batch_bonus
                + 0.10 * affinity_bonus
                + 0.10 * cloud_overuse)

    # ----------------------------------------------------------------
    #  PPO update
    # ----------------------------------------------------------------

    def _store(self, state, action, reward, log_prob, value, done):
        self.trajectory.append((state, action, reward, log_prob, value, done))

    def _update_policy(self):
        if len(self.trajectory) < self.update_interval:
            return

        # Unpack
        states = [t[0] for t in self.trajectory]
        actions = torch.LongTensor([t[1] for t in self.trajectory])
        rewards = torch.FloatTensor([t[2] for t in self.trajectory])
        old_log_probs = torch.stack([t[3] for t in self.trajectory])
        values = torch.cat([t[4] for t in self.trajectory]).detach()
        dones = torch.FloatTensor([t[5] for t in self.trajectory])

        # Stack the 3 tensors of each state separately
        task_feats = torch.stack([s[0] for s in states])      # (B, task_dim)
        server_feats = torch.stack([s[1] for s in states])    # (B, S, server_dim)
        edge_feats = torch.stack([s[2] for s in states])      # (B, S, edge_dim)

        # Returns / Advantages
        if self.enable_gae:
            advantages = torch.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                next_val = 0.0 if t == len(rewards) - 1 else values[t + 1].item()
                delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t].item()
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards)
            running = 0.0
            for t in reversed(range(len(rewards))):
                running = rewards[t] + self.gamma * running * (1 - dones[t].item())
                returns[t] = running
            advantages = returns - values

        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO multi-epoch updates
        for _ in range(self.ppo_epochs):
            probs, new_values, _ = self.policy(task_feats, server_feats, edge_feats)
            chosen = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            new_log_probs = torch.log(chosen + 1e-8)

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

    # ----------------------------------------------------------------
    #  Pretrain（与 RLScheduler 同步）
    # ----------------------------------------------------------------

    def _pretrain(self, episodes: int):
        logger.info(f"GNN 预训练: {episodes} 轮, {len(self.sim.tasks)} 个任务")
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
        self._reset_environment()
        self._decision_count = self._warmup_steps
        self.entropy_coeff = 0.01 if self.enable_entropy else 0.0

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
            server.loaded_models.clear()
            server.model_refs.clear()
            server.weight_vram_used = 0.0
            # E1 修复：能耗也要重置
            server.accumulated_energy_J = 0.0
        self.sim.completed_tasks.clear()

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
            task_feat, server_feats, edge_feats = self.encoder.encode(
                task, ready_task_count=ready_count)

            action_idx, server_id, log_prob, value = self._select_action(
                task_feat, server_feats, edge_feats, task)
            if action_idx is None:
                continue

            target_server = self.sim.servers[server_id]
            if not target_server.can_allocate(task):
                # 防御：被 mask 漏掉的非法动作
                self._store((task_feat, server_feats, edge_feats),
                            action_idx, -1.0, log_prob, value, False)
                self._decision_count += 1
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
            self._store((task_feat, server_feats, edge_feats),
                        action_idx, reward, log_prob, value, False)
            self._decision_count += 1

        self._update_policy()
