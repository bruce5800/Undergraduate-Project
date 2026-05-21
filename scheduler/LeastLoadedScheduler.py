"""
LeastLoadedScheduler — 简单贪心 baseline

策略：把每个 READY 任务派到 **当前计算利用率最低** 的服务器（在 can_allocate 通过的前提下）。

设计动机：
  - 这是审稿人的"低门槛对照"：如果你的复杂 RL 调度器赢不了这个简单策略，论文就站不住
  - 比 RR 多一点点智能（看 server 当前负载），比 HEFT 简单得多（不算 EFT）
  - 与"shortest-queue"互为补充——这个看当前 running 占用，那个看排队长度

实现说明：
  - 利用率 = used_compute / total_compute（归一化，跨异构服务器可比）
  - 平局时取 server_id 更小的（确定性）
  - 找不到能容纳的服务器时跳过本轮（任务下轮再试）
"""

import logging
from environment.task import TaskStatus
from environment.server import ServerType
from scheduler.base import BaseScheduler


class LeastLoadedScheduler(BaseScheduler):
    logger = logging.getLogger(__name__)

    def __init__(self, sim_env):
        super().__init__(sim_env)
        self.server_ids = sorted(sim_env.servers.keys())
        # 缓存云服务器 ID 用作未分配任务的传输源
        self._cloud_id = next(
            (s.server_id for s in sim_env.servers.values()
             if s.type == ServerType.CLOUD),
            self.server_ids[0]
        )

    def _least_loaded(self, task):
        """按 (compute_util, server_id) 升序找第一个 can_allocate 的服务器。"""
        candidates = sorted(
            self.sim.servers.values(),
            key=lambda s: (
                s.used_compute / max(s.total_compute, 1e-6),
                s.server_id,
            )
        )
        for s in candidates:
            if s.can_allocate(task):
                return s
        return None

    def schedule(self):
        ready_tasks = [t for t in self.sim.tasks.values()
                       if t.status == TaskStatus.READY]
        if not ready_tasks:
            return

        for task in ready_tasks:
            target = self._least_loaded(task)
            if target is None:
                # 当前所有服务器都装不下；下轮再调度
                continue

            src = task.assigned_server if task.assigned_server is not None \
                else self._cloud_id
            transfer_time = self.sim.network.estimate_transfer_time(
                src, target.server_id, task.output_size)

            task.assigned_server = target.server_id
            task.transfer_delay = transfer_time
            effective_priority = 1.0 / max(transfer_time, 1e-6)
            target.add_task(task, priority=effective_priority)
