"""
ShortestQueueScheduler — 简单贪心 baseline

策略：把每个 READY 任务派到 **当前任务队列最短** 的服务器（在 can_allocate 通过的前提下）。

设计动机：
  - 跟 LeastLoaded 互补：一个看正在 running 的，一个看排队中的
  - 这是工业界负载均衡器的"默认行为"（nginx least_conn、AWS LB、k8s）
  - 简单、强、可解释——审稿人会问"你跟 LB 比怎么样"，这个 baseline 一句话能答

实现说明：
  - 主键：len(task_queue)
  - 平局：用计算利用率次低作为 tie-breaker（避免把队都集中在一台空闲机器）
  - 找不到能容纳的服务器时跳过本轮
"""

import logging
from environment.task import TaskStatus
from environment.server import ServerType
from scheduler.base import BaseScheduler


class ShortestQueueScheduler(BaseScheduler):
    logger = logging.getLogger(__name__)

    def __init__(self, sim_env):
        super().__init__(sim_env)
        self.server_ids = sorted(sim_env.servers.keys())
        self._cloud_id = next(
            (s.server_id for s in sim_env.servers.values()
             if s.type == ServerType.CLOUD),
            self.server_ids[0]
        )

    def _shortest_queue(self, task):
        """按 (queue_len, compute_util, server_id) 升序找第一个 can_allocate 的服务器。"""
        candidates = sorted(
            self.sim.servers.values(),
            key=lambda s: (
                len(s.task_queue),
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
            target = self._shortest_queue(task)
            if target is None:
                continue

            src = task.assigned_server if task.assigned_server is not None \
                else self._cloud_id
            transfer_time = self.sim.network.estimate_transfer_time(
                src, target.server_id, task.output_size)

            task.assigned_server = target.server_id
            task.transfer_delay = transfer_time
            effective_priority = 1.0 / max(transfer_time, 1e-6)
            target.add_task(task, priority=effective_priority)
