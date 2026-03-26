import random
import logging
import numpy as np
from environment.task import TaskStatus
from environment.server import ServerType
from scheduler.base import BaseScheduler

class PSOScheduler(BaseScheduler):
    logger = logging.getLogger(__name__)

    def __init__(self, sim_env, num_particles=30, generations=30, w=0.7, c1=1.5, c2=1.5):
        super().__init__(sim_env)
        self.num_particles = num_particles
        self.generations = generations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        self.server_ids = sorted(sim_env.servers.keys())
        self.num_servers = len(self.server_ids)
        self.valid_server_indices = list(range(self.num_servers))

        # 缓存云服务器 ID，供传输源兜底使用
        cloud_servers = [s for s in sim_env.servers.values() if s.type == ServerType.CLOUD]
        self.cloud_server_id = cloud_servers[0].server_id if cloud_servers else self.server_ids[0]

    # ------------------------------------------------------------------ #
    #  粒子群初始化                                                         #
    # ------------------------------------------------------------------ #

    def initialize_particles(self, tasks):
        """每轮调用都完全重置粒子群和全局最优"""
        # Bug 修复2：每次 update_policy 都必须重置，否则旧轮的 gbest 会污染新轮
        self.particles = []
        self.gbest_position = None
        self.gbest_fitness = float('inf')

        num_tasks = len(tasks)

        for _ in range(self.num_particles):
            position = [random.choice(self.valid_server_indices) for _ in range(num_tasks)]
            velocity = [random.uniform(-0.5, 0.5) for _ in range(num_tasks)]

            server_assignment = [self.server_ids[idx] for idx in position]
            fitness = self.evaluate_fitness(server_assignment, tasks)

            self.particles.append({
                'position': position,
                'velocity': velocity,
                'pbest_position': position.copy(),
                'pbest_fitness': fitness
            })

            if fitness < self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_position = position.copy()

    # ------------------------------------------------------------------ #
    #  适应度评估                                                           #
    # ------------------------------------------------------------------ #

    def evaluate_fitness(self, individual, tasks):
        for server_id in individual:
            if server_id not in self.sim.servers:
                return float('inf')

        makespan, energy, cost, balance = self.simulate_schedule(individual, tasks)

        balance_penalty = 1 - max(0.0, min(1.0, balance))
        return makespan * 0.4 + energy * 0.2 + cost * 0.2 + balance_penalty * 0.2

    def simulate_schedule(self, individual, tasks):
        """individual 是 server_id 列表（非 index）"""
        tasks_list = list(tasks)
        task_completion_times = {}
        server_free_at = {sid: 0.0 for sid in self.sim.servers.keys()}
        server_loads   = {sid: 0.0 for sid in self.sim.servers.keys()}
        total_energy = 0.0
        total_cost   = 0.0

        for task_idx, server_id in enumerate(individual):
            task      = tasks_list[task_idx]
            server    = self.sim.servers[server_id]

            # 依赖就绪时间
            dep_ready_time = max(
                (task_completion_times.get(dep, 0.0) for dep in task.dependencies),
                default=0.0
            )

            # Bug 修复3：传输源必须是合法的 int server_id
            src_id = task.assigned_server if task.assigned_server is not None else self.cloud_server_id
            transfer_time = self.sim.network.estimate_transfer_time(src_id, server_id, task.output_size)

            actual_start_time = max(dep_ready_time + transfer_time, server_free_at[server_id])
            execution_time    = task.workload / max(server.total_compute, 1e-6)
            finish_time       = actual_start_time + execution_time

            task_completion_times[task.task_id] = finish_time
            server_free_at[server_id]           = finish_time

            server_loads[server_id] += task.compute_demand
            total_energy += task.compute_demand * 0.1
            total_cost   += task.compute_demand * 0.05

        makespan    = max(task_completion_times.values()) if task_completion_times else 0.0
        load_values = list(server_loads.values())
        balance     = self._calculate_balance(load_values)

        return makespan, total_energy, total_cost, balance

    # ------------------------------------------------------------------ #
    #  粒子更新                                                             #
    # ------------------------------------------------------------------ #

    def update_particles(self, tasks):
        for particle in self.particles:
            new_velocity = []
            new_position = []

            for v, x, pbest, gbest in zip(particle['velocity'],
                                          particle['position'],
                                          particle['pbest_position'],
                                          self.gbest_position):
                r1, r2 = random.random(), random.random()
                new_v = self.w * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (gbest - x)
                new_v = max(-1.0, min(1.0, new_v))
                new_velocity.append(new_v)

                new_x = int(round(x + new_v))
                new_x = max(0, min(new_x, self.num_servers - 1))
                new_position.append(new_x)

            particle['velocity'] = new_velocity
            particle['position'] = new_position

            server_assignment = [self.server_ids[idx] for idx in new_position]
            fitness = self.evaluate_fitness(server_assignment, tasks)

            if fitness < particle['pbest_fitness']:
                particle['pbest_position'] = new_position.copy()
                particle['pbest_fitness']  = fitness

            if fitness < self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_position = new_position.copy()

    def update_policy(self, tasks):
        self.initialize_particles(tasks)  # 内部已重置 gbest

        no_improvement_count = 0
        previous_best = float('inf')

        for _ in range(self.generations):
            self.update_particles(tasks)

            if abs(self.gbest_fitness - previous_best) < 1e-6:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
                previous_best = self.gbest_fitness

            if no_improvement_count >= 5:
                break

        return [self.server_ids[idx] for idx in self.gbest_position]

    # ------------------------------------------------------------------ #
    #  调度入口                                                             #
    # ------------------------------------------------------------------ #

    def schedule(self):
        ready_tasks = [t for t in self.sim.tasks.values() if t.status == TaskStatus.READY]

        if not ready_tasks:
            return

        best_solution = self.update_policy(ready_tasks)

        for task_idx, server_id in enumerate(best_solution):
            if task_idx >= len(ready_tasks):
                break

            task = ready_tasks[task_idx]

            # Bug 修复1：PSO 建议的服务器资源不足时，回退到最空闲服务器，
            #           保证每个 READY 任务都能被分配，不会永久卡住。
            target_server = self._pick_server(task, preferred_server_id=server_id)
            if target_server is None:
                self.logger.debug(
                    f"PSO: 所有服务器均无法承载任务{task.task_id}，跳过（将在下轮重试）"
                )
                continue

            src_id = task.assigned_server if task.assigned_server is not None else self.cloud_server_id
            transfer_time = self.sim.network.estimate_transfer_time(
                src_id, target_server.server_id, task.output_size
            )

            task.assigned_server = target_server.server_id
            task.transfer_delay = transfer_time
            effective_priority   = 1 / max(transfer_time, 1e-6)
            target_server.add_task(task, priority=effective_priority)

            self.logger.debug(f"PSO分配: 任务{task.task_id} -> 服务器{target_server.server_id}")

    def _pick_server(self, task, preferred_server_id):
        """
        优先选择 PSO 推荐的服务器；若资源不足，
        按剩余计算容量降序选择第一个可用服务器。
        返回 Server 对象，全部不可用时返回 None。
        """
        # 1. 优先尝试 PSO 推荐
        if preferred_server_id in self.sim.servers:
            preferred = self.sim.servers[preferred_server_id]
            if preferred.can_allocate(task):
                return preferred

        # 2. 回退：按剩余算力排序，选第一个能容纳的服务器
        candidates = sorted(
            self.sim.servers.values(),
            key=lambda s: s.total_compute - s.used_compute,
            reverse=True
        )
        for server in candidates:
            if server.can_allocate(task):
                return server

        return None  # 真正无资源可用

    # ------------------------------------------------------------------ #
    #  负载均衡度计算                                                        #
    # ------------------------------------------------------------------ #

    def _calculate_balance(self, load_values):
        if not load_values or sum(load_values) == 0:
            return 1.0

        loads     = np.array(load_values)
        mean_load = np.mean(loads)
        if mean_load == 0:
            return 1.0

        std_load = np.std(loads)
        return 1.0 / (1.0 + (std_load / (mean_load + 1e-6)))