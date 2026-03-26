# HEFTscheduler.py
# HEFT (Heterogeneous Earliest Finish Time) 贪心调度器
#
# 算法核心思路：
#   1. 计算每个任务的"向上排名"(upward rank)：
#      rank(t) = 平均执行时间 + max(传输时间 + rank(后继任务))
#      rank 值越大说明任务越靠近关键路径起点，优先调度。
#   2. 按 rank 降序排列所有 READY 任务。
#   3. 对每个任务，遍历所有服务器，选择能使"最早完成时间 (EFT)"最小的服务器。
#      EFT = max(依赖就绪时间 + 传输时间, 服务器空闲时间) + 执行时间
#   4. 将任务分配到 EFT 最小的服务器。

import logging
import numpy as np
from environment.task import TaskStatus
from environment.server import ServerType
from scheduler.base import BaseScheduler

class HEFTScheduler(BaseScheduler):
    logger = logging.getLogger(__name__)

    def __init__(self, sim_env):
        super().__init__(sim_env)
        self.server_ids = sorted(sim_env.servers.keys())

        # 缓存云服务器 ID，作为未分配任务的传输源
        cloud_server = next(
            (s for s in sim_env.servers.values() if s.type == ServerType.CLOUD), None
        )
        self.cloud_server_id = cloud_server.server_id if cloud_server else self.server_ids[0]

        # 每台服务器的预计下次空闲时间（用于 EFT 计算，仅在调度阶段使用）
        self._server_available_time = {sid: 0.0 for sid in self.server_ids}

        # 预计算：每单位 output_size 的平均传输时间基准（网络拓扑固定后不变）
        self._avg_transfer_base = self._precompute_avg_transfer_base()
        # 预计算：各服务器计算能力的倒数均值（用于平均执行时间）
        self._avg_compute_inverse = float(np.mean([
            1.0 / max(self.sim.servers[sid].total_compute, 1e-6)
            for sid in self.server_ids
        ]))

    # ------------------------------------------------------------------ #
    #  向上排名计算                                                         #
    # ------------------------------------------------------------------ #

    def _precompute_avg_transfer_base(self) -> float:
        """预计算所有服务器对之间单位数据的平均传输时间（网络拓扑不变时只需算一次）"""
        times = []
        for src in self.server_ids:
            for dst in self.server_ids:
                if src != dst:
                    t = self.sim.network.estimate_transfer_time(src, dst, 1.0)
                    times.append(t)
        return float(np.mean(times)) if times else 0.0

    def _avg_exec_time(self, task) -> float:
        """任务在所有服务器上的平均执行时间"""
        return task.workload * self._avg_compute_inverse

    def _avg_transfer_time(self, task) -> float:
        """任务输出数据的平均传输时间（基于预计算基准线性缩放）"""
        return self._avg_transfer_base * task.output_size

    def _compute_upward_ranks(self, tasks: list) -> dict:
        """
        用动态规划（拓扑逆序）计算所有任务的向上排名。
        rank(t) = w(t) + max_{s ∈ successors(t)} [c(t,s) + rank(s)]
        其中 w(t) 为平均执行时间，c(t,s) 为平均传输时间。
        """
        task_map = {t.task_id: t for t in tasks}
        all_task_map = {t.task_id: t for t in self.sim.tasks.values()}

        # 构建后继关系（在全部任务范围内）
        successors = {t.task_id: [] for t in tasks}
        for t in self.sim.tasks.values():
            for dep_id in t.dependencies:
                if dep_id in successors:
                    successors[dep_id].append(t.task_id)

        ranks = {}
        avg_comm = {t.task_id: self._avg_transfer_time(t) for t in tasks}
        avg_exec = {t.task_id: self._avg_exec_time(t) for t in tasks}

        def rank(tid):
            if tid in ranks:
                return ranks[tid]
            task = task_map.get(tid)
            if task is None:
                return 0.0
            succ_ids = successors.get(tid, [])
            # 只考虑本批次内的后继；批次外的后继忽略（不在 task_map 中）
            batch_succs = [s for s in succ_ids if s in task_map]
            if not batch_succs:
                ranks[tid] = avg_exec[tid]
            else:
                ranks[tid] = avg_exec[tid] + max(
                    avg_comm[tid] + rank(s) for s in batch_succs
                )
            return ranks[tid]

        for t in tasks:
            rank(t.task_id)

        return ranks

    # ------------------------------------------------------------------ #
    #  EFT 估算                                                            #
    # ------------------------------------------------------------------ #

    def _estimate_eft(self, task, server_id: int,
                      task_finish_times: dict) -> float:
        """
        估算将 task 分配到 server_id 的最早完成时间。

        EFT = max(dep_ready + transfer, server_available) + exec_time

        task_finish_times: 本轮已分配任务的预计完成时间 {task_id: finish_time}
        """
        server = self.sim.servers[server_id]

        # 依赖就绪时间：取所有前驱任务完成时间的最大值
        dep_ready = 0.0
        for dep_id in task.dependencies:
            dep_time = task_finish_times.get(dep_id, 0.0)
            # 加上数据传输时间
            src_id = self.sim.tasks[dep_id].assigned_server \
                if dep_id in self.sim.tasks and self.sim.tasks[dep_id].assigned_server is not None \
                else self.cloud_server_id
            transfer = self.sim.network.estimate_transfer_time(src_id, server_id, task.output_size)
            dep_ready = max(dep_ready, dep_time + transfer)

        # 如果没有依赖，考虑当前任务的传输时间（从云端下发）
        if not task.dependencies:
            src_id = task.assigned_server if task.assigned_server is not None else self.cloud_server_id
            transfer = self.sim.network.estimate_transfer_time(src_id, server_id, task.output_size)
            dep_ready = max(dep_ready, transfer)

        exec_time = task.workload / max(server.total_compute, 1e-6)
        start_time = max(dep_ready, self._server_available_time[server_id])
        return start_time + exec_time

    # ------------------------------------------------------------------ #
    #  调度入口                                                             #
    # ------------------------------------------------------------------ #

    def schedule(self):
        ready_tasks = [t for t in self.sim.tasks.values() if t.status == TaskStatus.READY]
        if not ready_tasks:
            return

        # 1. 计算向上排名并按降序排列（关键路径任务优先）
        ranks = self._compute_upward_ranks(ready_tasks)
        sorted_tasks = sorted(ready_tasks, key=lambda t: ranks.get(t.task_id, 0.0), reverse=True)

        # 2. 本轮分配过程中记录各任务的预计完成时间，用于后续任务的 EFT 计算
        task_finish_times = {
            t.task_id: t.end_time
            for t in self.sim.tasks.values()
            if t.status == TaskStatus.COMPLETED and t.end_time is not None
        }

        # 3. 按 rank 顺序逐个分配到 EFT 最小的服务器
        for task in sorted_tasks:
            best_server_id = None
            best_eft = float('inf')

            for sid in self.server_ids:
                server = self.sim.servers[sid]
                if not server.can_allocate(task):
                    continue
                eft = self._estimate_eft(task, sid, task_finish_times)
                if eft < best_eft:
                    best_eft = eft
                    best_server_id = sid

            if best_server_id is None:
                # 所有服务器资源不足，等待下轮（资源释放后重试）
                # self.logger.warning(f"HEFT: 无可用服务器承载任务{task.task_id}，等待下轮")
                continue

            # 执行分配
            target_server = self.sim.servers[best_server_id]
            src_id = task.assigned_server if task.assigned_server is not None else self.cloud_server_id
            transfer_time = self.sim.network.estimate_transfer_time(
                src_id, best_server_id, task.output_size
            )

            task.assigned_server = best_server_id
            task.transfer_delay = transfer_time
            effective_priority = 1 / max(transfer_time, 1e-6)
            target_server.add_task(task, priority=effective_priority)

            # 更新本轮追踪表，供后续任务的 EFT 计算使用
            task_finish_times[task.task_id] = best_eft
            self._server_available_time[best_server_id] = best_eft

            self.logger.debug(f"HEFT: 任务{task.task_id}(rank={ranks[task.task_id]:.2f}) "
                              f"-> 服务器{best_server_id} (EFT={best_eft:.4f})")