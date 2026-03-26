import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from environment.task import TaskStatus, Task
from environment.server import ServerType
from scheduler.base import BaseScheduler

class StateEncoder:
    """将环境状态编码为特征向量"""
    def __init__(self, sim_env):
        self.sim = sim_env
    
    def encode(self, task):
        # 特征维度设计
        features = []
        
        # 任务特征 (归一化)
        features.append(task.compute_demand / 100)  # 假设最大需求100TFLOPS
        features.append(task.input_size / 50)       # 假设最大输入50GB
        features.append(task.output_size / 20)      # 假设最大输出20GB
        
        # 服务器状态特征
        for server in self.sim.servers.values():
            features.append(server.used_compute / server.total_compute)
            features.append(server.used_memory / server.total_memory)
            features.append(server.bandwidth / 1000)  # 假设最大带宽1Gbps
            
            # 网络特征（当前任务所在位置到该服务器的传输时间）
            if task.assigned_server is not None:
                src = task.assigned_server
            else:  # 初始输入数据位置假设在云端
                src = next(s for s in self.sim.servers.values() if s.type==ServerType.CLOUD).server_id
            transfer_time = self.sim.network.estimate_transfer_time(
                src, server.server_id, task.output_size)
            features.append(transfer_time / 10)  # 假设最大传输时间10秒
        
        return torch.FloatTensor(features)

class ActorCritic(nn.Module):
    """策略网络与价值网络共享底层特征"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Linear(64, 1)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        probs = self.actor(features)
        value = self.critic(features)
        return probs, value

class RLScheduler(BaseScheduler):
    def __init__(self, sim_env):
        super().__init__(sim_env)
        self.state_encoder = StateEncoder(sim_env)
        
        # 状态维度 = 任务特征数 + 服务器数*(服务器特征数 + 网络特征数)
        sample_task = Task(0, 10, 5, 2, [])
        state_dim = len(self.state_encoder.encode(sample_task))
        action_dim = len(sim_env.servers)  # 每个服务器是一个动作
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.gamma = 0.99  # 折扣因子
        
        # 经验回放缓存
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
    def select_action(self, state, task):
        """根据策略选择服务器，返回 (action_index, server_id)"""
        with torch.no_grad():
            probs, _ = self.policy(state)

        # 转换为服务器ID
        server_ids = list(self.sim.servers.keys())
        action_idx = np.random.choice(len(server_ids), p=probs.numpy())
        return action_idx, server_ids[action_idx]
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append( (state, action, reward, next_state, done) )
    
    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones)
        
        # 计算TD误差
        _, next_values = self.policy(next_states)
        target_values = rewards + (1 - dones) * self.gamma * next_values.squeeze()
        
        # 计算Critic损失
        _, current_values = self.policy(states)
        critic_loss = nn.MSELoss()(current_values.squeeze(), target_values.detach())
        
        # 计算Actor损失
        probs, _ = self.policy(states)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)))
        advantages = target_values - current_values.squeeze().detach()
        actor_loss = -(log_probs * advantages).mean()
        
        # 总损失
        total_loss = critic_loss + actor_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
    def calculate_reward(self, task, server_id):
        """定义奖励函数"""
        server = self.sim.servers[server_id]
        
        # 基础奖励：执行时间的倒数
        exec_time = task.compute_demand / server.total_compute
        reward = 1.0 / (exec_time + 1e-6)
        
        # 惩罚项
        if not server.can_allocate(task):
            reward -= 10.0  # 资源不足惩罚
        
        # 边缘优先奖励
        if server.type == ServerType.EDGE:
            reward += 0.5
        
        return reward
    
    def schedule(self):
        """强化学习调度入口"""
        ready_tasks = [t for t in self.sim.tasks.values() if t.status == TaskStatus.READY]
        
        for task in ready_tasks:
            # 编码当前状态
            state = self.state_encoder.encode(task)

            # 选择动作（服务器）
            action_idx, server_id = self.select_action(state, task)
            target_server = self.sim.servers[server_id]

            # 执行调度
            if target_server.can_allocate(task):
                # 计算数据传输时间
                if task.assigned_server is not None:
                    src = task.assigned_server
                else:
                    src = next(s for s in self.sim.servers.values() if s.type==ServerType.CLOUD).server_id
                transfer_time = self.sim.network.estimate_transfer_time(
                    src, target_server.server_id, task.output_size)

                # 记录调度结果
                task.assigned_server = target_server.server_id

                min_transfer_time = 1e-6  # 最小时间阈值（0.001毫秒）
                effective_priority = 1 / max(transfer_time, min_transfer_time)
                target_server.add_task(task, priority=effective_priority)

                # 计算奖励
                reward = self.calculate_reward(task, server_id)

                # 获取下一状态（调度后的状态）
                next_state = self.state_encoder.encode(task)

                # 存储经验（使用 action_idx 而非 server_id，与策略网络输出对齐）
                self.store_transition(state, action_idx, reward, next_state, done=False)

                # 更新策略
                self.update_policy()