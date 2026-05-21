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
from environment.task import TaskStatus, Task, TaskKind
from environment.server import ServerType
from environment.model_catalog import CATALOG
from scheduler.base import BaseScheduler

logger = logging.getLogger(__name__)


# ================================================================
#  状态编码器
# ================================================================

class StateEncoder:
    """将环境观测编码为固定长度特征向量。

    M3 step 2 升级：state 暴露 AIGC 物理可观测量
      - 任务侧：kind (PREFILL/DECODE/GENERIC)、KV cache 大小、model one-hot
      - 服务器侧：是否加载本任务的模型、当前同模同阶段并发数、显存余量、
                  冷加载代价、是否是同请求 prefill 所在服务器（亲和性信号）
    """

    # 固定排序的 model_id 列表，用于稳定的 one-hot 编码索引
    _MODEL_IDS = sorted(CATALOG.keys())
    _MODEL_INDEX = {mid: i for i, mid in enumerate(_MODEL_IDS)}

    def __init__(self, sim_env, enable_aigc_state: bool = True):
        self.sim = sim_env
        self.server_ids = sorted(sim_env.servers.keys())
        self._cloud_id = next(
            (s.server_id for s in sim_env.servers.values()
             if s.type == ServerType.CLOUD),
            self.server_ids[0]
        )
        self._successor_count: dict = {}
        # M3 step3: enable_aigc_state=False 时回退到 M0 的 8 + 5*ns 维状态
        self.enable_aigc_state = enable_aigc_state

    # ----------------------------------------------------------------
    #  特征编码主入口
    # ----------------------------------------------------------------

    def encode(self, task, ready_task_count: int = 0) -> torch.FloatTensor:
        features: list[float] = []

        # ============= 任务侧特征 =============
        # ---- 基础 (6 维) ----
        features.append(task.compute_demand / 20.0)
        features.append(task.workload / 2000.0)
        features.append(task.input_size / 8.0)
        features.append(task.output_size / 0.3)
        features.append(len(task.dependencies) / 5.0)
        features.append(self._get_successor_count(task) / 5.0)

        # ---- M3: AIGC 任务特征 (4 + |M| 维) ----
        # M3 step3: enable_aigc_state=False 时整个 AIGC 任务块被跳过
        kind = getattr(task, "kind", TaskKind.GENERIC)
        if self.enable_aigc_state:
            features.append(1.0 if kind == TaskKind.GENERIC else 0.0)
            features.append(1.0 if kind == TaskKind.PREFILL else 0.0)
            features.append(1.0 if kind == TaskKind.DECODE else 0.0)
            features.append(getattr(task, "kv_cache_GB", 0.0) / 5.0)
            # model one-hot
            model_onehot = [0.0] * len(self._MODEL_IDS)
            if task.model_id in self._MODEL_INDEX:
                model_onehot[self._MODEL_INDEX[task.model_id]] = 1.0
            features.extend(model_onehot)

        # ============= 全局状态 (2 维) =============
        features.append(ready_task_count / 50.0)
        completed_ratio = len(self.sim.completed_tasks) / max(len(self.sim.tasks), 1)
        features.append(completed_ratio)

        # ============= 服务器侧特征（每台 10 维）=============
        src = task.assigned_server if task.assigned_server is not None \
            else self._cloud_id
        sibling_server_id = self._get_sibling_server_id(task)

        for sid in self.server_ids:
            server = self.sim.servers[sid]
            # ---- 基础 5 维（原有）----
            features.append(server.used_compute / max(server.total_compute, 1e-6))
            features.append(server.used_memory / max(server.total_memory, 1e-6))
            features.append(server.bandwidth / 1000.0)
            features.append(len(server.task_queue) / 20.0)
            transfer = self.sim.network.estimate_transfer_time(
                src, sid, task.output_size)
            features.append(transfer / 10.0)

            # ---- M3: AIGC 服务器特征 5 维 ----
            # M3 step3: enable_aigc_state=False 时整个服务器 AIGC 块被跳过
            if self.enable_aigc_state:
                # 1) is_my_model_loaded：当前模型是否在该服务器（避免冷加载）
                is_loaded = (task.model_id is not None
                             and task.model_id in server.loaded_models)
                features.append(1.0 if is_loaded else 0.0)
                # 2) batch_count：同模同阶段并发数（归一化到 max_batch_size）
                batch_count = server._current_batch_size(task.model_id, kind)
                max_b = (CATALOG[task.model_id].max_batch_size
                         if task.model_id in CATALOG else 1)
                features.append(batch_count / max(max_b, 1))
                # 3) vram_free / total
                vram_free = (server.total_memory
                             - server.used_memory
                             - server.weight_vram_used)
                features.append(max(vram_free, 0.0) / max(server.total_memory, 1e-6))
                # 4) cold_load_cost：归一化到 30s
                cold = server.cold_load_cost(task.model_id) \
                    if task.model_id is not None else 0.0
                features.append(cold / 30.0)
                # 5) is_sibling_server：本服务器是否是同请求 prefill 所在地
                features.append(1.0 if sid == sibling_server_id else 0.0)

        return torch.FloatTensor(features)

    @property
    def state_dim(self) -> int:
        # M3 step3: 消融关掉 AIGC state 时，回到 M0 维度
        if not self.enable_aigc_state:
            return 8 + 5 * len(self.server_ids)
        # 任务 (10+|M|) + 全局 (2) + 每台服务器 10
        return (10 + len(self._MODEL_IDS) + 2
                + 10 * len(self.server_ids))

    # ----------------------------------------------------------------
    #  内部辅助
    # ----------------------------------------------------------------

    def _get_successor_count(self, task) -> int:
        tid = task.task_id
        if tid not in self._successor_count:
            cnt = sum(1 for t in self.sim.tasks.values() if tid in t.dependencies)
            self._successor_count[tid] = cnt
        return self._successor_count[tid]

    def _get_sibling_server_id(self, task):
        """对于 DECODE 任务，找到它依赖的 prefill 已被分配到哪台服务器。

        若 task 不是 DECODE 或 prefill 还未分配，返回 None。
        用于"序列亲和性"信号：让 RL 看到"我的 KV cache 在哪里"。
        """
        if getattr(task, "kind", None) != TaskKind.DECODE:
            return None
        for dep_id in task.dependencies:
            dep = self.sim.tasks.get(dep_id)
            if dep is not None and dep.assigned_server is not None:
                return dep.assigned_server
        return None


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

    def __init__(self, sim_env, pretrain_episodes: int = 10,
                 # ---- M3 step3: 7 个消融开关 ----
                 enable_warm_reward: bool = True,
                 enable_batch_reward: bool = True,
                 enable_affinity_reward: bool = True,
                 enable_aigc_state: bool = True,
                 enable_action_mask: bool = True,
                 enable_gae: bool = True,
                 enable_pretrain: bool = True,
                 enable_entropy: bool = True):
        super().__init__(sim_env)

        # 保存消融开关
        self.enable_warm_reward = enable_warm_reward
        self.enable_batch_reward = enable_batch_reward
        self.enable_affinity_reward = enable_affinity_reward
        self.enable_aigc_state = enable_aigc_state
        self.enable_action_mask = enable_action_mask
        self.enable_gae = enable_gae
        self.enable_pretrain = enable_pretrain
        self.enable_entropy = enable_entropy

        self.server_ids = sorted(sim_env.servers.keys())
        self.state_encoder = StateEncoder(sim_env,
                                          enable_aigc_state=enable_aigc_state)

        state_dim = self.state_encoder.state_dim
        action_dim = len(self.server_ids)

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        # PPO 超参
        self.gamma = 0.95
        self.gae_lambda = 0.90
        self.clip_epsilon = 0.2
        # M3 step3: enable_entropy=False 时熵正则化系数归零
        self.entropy_coeff = 0.05 if enable_entropy else 0.0
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

        # ---- 多轮预训练（M3 step3 可关）----
        if enable_pretrain and pretrain_episodes > 0 and len(sim_env.tasks) > 0:
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
            # M3 step3: 消融时不向 policy 提供掩码 —— 让网络通过 reward 学习避开非法动作
            mask_arg = action_mask.unsqueeze(0) if self.enable_action_mask else None
            probs, value, _ = self.policy(state.unsqueeze(0), mask_arg)
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
        """M3 step 2 升级版多目标奖励。

        老 3 项（time / balance / match）保持，加入 3 项 AIGC 行为奖励：
          - warm_bonus: 模型已在该服务器上 → 避免冷加载（论文核心 affinity）
          - batch_bonus: 加入已存在的同模 batch → 拿到吞吐红利
          - affinity_bonus: DECODE 落在它 prefill 所在的服务器 → 0 KV 迁移
        """
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

        # ---- M3: AIGC 奖励（各自可关掉做消融）----
        # 4) Warm bonus: 模型已加载 → +1；否则按冷加载秒数衰减到 [-1, 0]
        if not self.enable_warm_reward:
            warm_bonus = 0.0
        elif task.model_id is None or task.model_id not in CATALOG:
            warm_bonus = 0.0
        elif task.model_id in server.loaded_models:
            warm_bonus = 1.0
        else:
            cold_sec = CATALOG[task.model_id].cold_load_sec
            warm_bonus = max(-1.0, -cold_sec / 20.0)

        # 5) Batch bonus: 加入已存在的同模同阶段 batch（不含自己）
        kind = getattr(task, "kind", TaskKind.GENERIC)
        if not self.enable_batch_reward:
            batch_bonus = 0.0
        elif kind in (TaskKind.PREFILL, TaskKind.DECODE) and task.model_id in CATALOG:
            existing_batch = server._current_batch_size(task.model_id, kind)
            max_b = CATALOG[task.model_id].max_batch_size
            if existing_batch == 0:
                batch_bonus = 0.0  # 首个不算 batch
            elif existing_batch >= max_b:
                batch_bonus = -0.5  # batch 已满，应该被 admission control 拒
            else:
                # decode batch 收益高（overhead 低），prefill 收益小（overhead 高）
                weight = 1.0 if kind == TaskKind.DECODE else 0.4
                batch_bonus = min(1.0, weight * existing_batch / max(max_b, 1) * 4.0)
        else:
            batch_bonus = 0.0

        # 6) Affinity bonus: DECODE 落在 prefill 服务器
        affinity_bonus = 0.0
        if self.enable_affinity_reward and kind == TaskKind.DECODE:
            for dep_id in task.dependencies:
                dep = self.sim.tasks.get(dep_id)
                if dep is not None and dep.assigned_server == server.server_id:
                    affinity_bonus = 1.0
                    break
            else:
                # 同请求 prefill 在别处 → 付 KV 迁移代价（output_size 已经计了），
                # 这里再给个明确信号
                if task.output_size > 0.1:  # KV 较大才惩罚明显
                    affinity_bonus = -0.5

        return (0.30 * time_reward
                + 0.15 * balance_reward
                + 0.10 * match_reward
                + 0.15 * warm_bonus
                + 0.20 * batch_bonus
                + 0.10 * affinity_bonus)

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

        # M3 step3: GAE 可关 —— 消融时用蒙特卡洛 return + value baseline
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
            # 蒙特卡洛 returns（无 lambda 平滑）
            returns = torch.zeros_like(rewards)
            running = 0.0
            for t in reversed(range(len(rewards))):
                running = rewards[t] + self.gamma * running * (1 - dones[t].item())
                returns[t] = running
            advantages = returns - values

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

            # M3 step3: 关掉 action_mask 后，policy 可能选到非法服务器
            # —— 给强负奖励，让 PPO 学会规避；不入队，下轮再调度此 task
            if not target_server.can_allocate(task):
                self.store_transition(state, action_idx, reward=-1.0,
                                       log_prob=log_prob, value=value,
                                       done=False)
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

            reward = self.calculate_reward(task, target_server, transfer_time)
            self.store_transition(state, action_idx, reward, log_prob, value,
                                  done=False)
            self._decision_count += 1

        self.update_policy()
