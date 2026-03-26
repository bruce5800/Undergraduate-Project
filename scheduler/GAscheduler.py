# GAscheduler.py (修复版)
import logging
import numpy as np
import random
from typing import List, Tuple, Dict
from environment.server import ServerType
from environment.task import TaskStatus

class GAScheduler:
    """优化的遗传算法调度器"""

    logger = logging.getLogger(__name__)

    def __init__(self, sim_env, population_size=30, generations=50,
                 crossover_rate=0.8, mutation_rate=0.1, elitism_rate=0.1):
        self.sim = sim_env
        self.population_size = min(population_size, 50)
        self.generations = min(generations, 100)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate

        self.server_ids = list(self.sim.servers.keys())
        self.server_capacities = {
            sid: self.sim.servers[sid].total_compute
            for sid in self.server_ids
        }

        self.best_solution_cache = None
        self.cached_task_count = 0

        self.task_count_thresholds = {
            'small': 50,
            'medium': 200,
            'large': 500
        }

        # 缓存云服务器 ID，供调度兜底使用
        cloud_server = next(
            (s for s in sim_env.servers.values() if s.type == ServerType.CLOUD), None
        )
        self.cloud_server_id = cloud_server.server_id if cloud_server else self.server_ids[0]

    # ------------------------------------------------------------------ #
    #  自适应参数                                                           #
    # ------------------------------------------------------------------ #

    def _get_adaptive_parameters(self, task_count: int) -> Tuple[int, int]:
        if task_count <= self.task_count_thresholds['small']:
            return 30, 50
        elif task_count <= self.task_count_thresholds['medium']:
            return 20, 30
        else:
            return 15, 20

    # ------------------------------------------------------------------ #
    #  种群初始化                                                           #
    # ------------------------------------------------------------------ #

    def initialize_population(self, tasks: List) -> List:
        """智能初始化种群"""
        population = []
        task_count = len(tasks)
        # 预计算均值，避免 O(n²) 重复计算
        mean_demand = np.mean([t.compute_demand for t in tasks])

        # 策略1: 完全随机
        for _ in range(self.population_size // 3):
            population.append([random.choice(self.server_ids) for _ in range(task_count)])

        # 策略2: 启发式（大任务倾向高算力服务器）
        weights = [self.server_capacities[sid] for sid in self.server_ids]
        for _ in range(self.population_size // 3):
            individual = []
            for task in tasks:
                if task.compute_demand > mean_demand:
                    server_idx = random.choices(range(len(self.server_ids)), weights=weights)[0]
                else:
                    server_idx = random.randint(0, len(self.server_ids) - 1)
                individual.append(self.server_ids[server_idx])
            population.append(individual)

        # 策略3: 基于缓存解变异 / 补充随机
        # Bug 修复1：缓存长度与当前 task_count 不一致时禁止使用，
        #           否则 population[0] = self.best_solution_cache 会写入长度错误的染色体
        remaining = self.population_size - len(population)
        if (self.best_solution_cache
                and self.cached_task_count == task_count
                and len(self.best_solution_cache) == task_count):
            for _ in range(remaining):
                individual = self.best_solution_cache.copy()
                for _ in range(max(1, task_count // 20)):
                    idx = random.randint(0, task_count - 1)
                    individual[idx] = random.choice(self.server_ids)
                population.append(individual)
        else:
            for _ in range(remaining):
                population.append([random.choice(self.server_ids) for _ in range(task_count)])

        return population

    # ------------------------------------------------------------------ #
    #  适应度评估                                                           #
    # ------------------------------------------------------------------ #

    def fast_evaluate_fitness(self, individual: List, tasks: List) -> float:
        """快速适应度评估"""
        task_count = len(tasks)

        # Bug 修复2：依赖惩罚用 task_id -> list_index 映射，
        #           原代码用 dep_id < task_count 做下标越界保护，但 dep_id 是 task_id
        #           （全局唯一整数），不是列表下标，会访问到错误任务或越界。
        task_id_to_idx = {task.task_id: i for i, task in enumerate(tasks)}

        server_loads = {sid: 0.0 for sid in self.server_ids}
        for i, server_id in enumerate(individual):
            task = tasks[i]
            server_loads[server_id] += task.compute_demand / self.server_capacities[server_id]

        max_load = max(server_loads.values())
        avg_load = np.mean(list(server_loads.values()))
        load_balance = np.std(list(server_loads.values())) / avg_load if avg_load > 0 else 0

        dependency_penalty = 0.0
        for i, task in enumerate(tasks):
            for dep_id in task.dependencies:
                if dep_id in task_id_to_idx:          # 正确：用映射查下标
                    dep_idx = task_id_to_idx[dep_id]
                    if individual[i] != individual[dep_idx]:
                        dependency_penalty += 0.01

        return max_load + 0.5 * load_balance + dependency_penalty

    # ------------------------------------------------------------------ #
    #  选择 / 交叉 / 变异                                                   #
    # ------------------------------------------------------------------ #

    def tournament_selection(self, population: List, tasks: List, tournament_size: int = 3) -> List:
        selected = []
        for _ in range(len(population)):
            contestants = random.sample(population, min(tournament_size, len(population)))
            best = min(contestants, key=lambda ind: self.fast_evaluate_fitness(ind, tasks))
            selected.append(best)
        return selected

    def two_point_crossover(self, parent1: List, parent2: List) -> Tuple[List, List]:
        if random.random() < self.crossover_rate and len(parent1) > 2:
            point1 = random.randint(1, len(parent1) // 2)
            point2 = random.randint(point1 + 1, len(parent1) - 1)
            child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
            return child1, child2
        return parent1.copy(), parent2.copy()

    def adaptive_mutate(self, individual: List, tasks: List) -> List:
        if random.random() < self.mutation_rate:
            individual = individual.copy()
            task_count = len(tasks)
            mutation_count = max(1, task_count // 50) if task_count > 100 else max(1, task_count // 20)
            # 预计算均值，避免循环内重复 O(n) 计算
            mean_demand = np.mean([t.compute_demand for t in tasks])

            for _ in range(mutation_count):
                task_idx = random.randint(0, task_count - 1)
                task = tasks[task_idx]

                if task.compute_demand > mean_demand:
                    current_server = individual[task_idx]
                    other_servers = [sid for sid in self.server_ids if sid != current_server]
                    if other_servers:
                        capacities = [self.server_capacities[sid] for sid in other_servers]
                        total_cap = sum(capacities)
                        weights = [c / total_cap for c in capacities]
                        new_server = random.choices(other_servers, weights=weights)[0]
                    else:
                        new_server = current_server
                else:
                    new_server = random.choice(self.server_ids)

                individual[task_idx] = new_server
        return individual

    # ------------------------------------------------------------------ #
    #  主优化循环                                                           #
    # ------------------------------------------------------------------ #

    def update_policy(self, tasks: List) -> List:
        task_count = len(tasks)
        adaptive_pop_size, adaptive_gens = self._get_adaptive_parameters(task_count)

        population = self.initialize_population(tasks)
        # Bug 修复1（续）：写入缓存解前同样做长度校验
        if (self.best_solution_cache
                and self.cached_task_count == task_count
                and len(self.best_solution_cache) == task_count):
            population[0] = self.best_solution_cache

        elitism_count = max(1, int(adaptive_pop_size * self.elitism_rate))
        best_solution = None
        best_fitness = float('inf')

        # Bug 修复3：早停逻辑使用单独计数器跟踪"连续无改进代数"，
        #           原代码判断条件 abs(fitness_scores[0][0] - best_fitness) < 0.001
        #           实际上在 best_fitness 已更新后恒为 0，导致第 11 代后立即退出。
        no_improve_count = 0

        for gen in range(adaptive_gens):
            fitness_scores = [
                (self.fast_evaluate_fitness(ind, tasks), ind)
                for ind in population
            ]
            fitness_scores.sort(key=lambda x: x[0])
            elites = [ind for _, ind in fitness_scores[:elitism_count]]

            current_best_fitness = fitness_scores[0][0]
            if current_best_fitness < best_fitness - 1e-6:
                best_fitness = current_best_fitness
                best_solution = fitness_scores[0][1].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if gen > 10 and no_improve_count >= 5:
                break

            selected = self.tournament_selection(population, tasks, tournament_size=3)

            new_population = elites.copy()
            while len(new_population) < adaptive_pop_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.two_point_crossover(parent1, parent2)
                child1 = self.adaptive_mutate(child1, tasks)
                child2 = self.adaptive_mutate(child2, tasks)
                new_population.extend([child1, child2])

            population = new_population[:adaptive_pop_size]

        self.best_solution_cache = best_solution.copy() if best_solution else None
        self.cached_task_count = task_count
        return best_solution

    # ------------------------------------------------------------------ #
    #  批量调度                                                             #
    # ------------------------------------------------------------------ #

    def schedule_batch(self, tasks: List) -> Dict:
        if not tasks:
            return {}

        task_count = len(tasks)

        if task_count > 200:
            sorted_tasks = sorted(tasks, key=lambda t: t.compute_demand)
            group_size = min(100, task_count // 5)
            groups = [sorted_tasks[i:i + group_size] for i in range(0, task_count, group_size)]

            schedule_result = {}
            for group in groups:
                best_solution = self.update_policy(group)
                for i, server_id in enumerate(best_solution):
                    schedule_result[group[i].task_id] = server_id
            return schedule_result
        else:
            best_solution = self.update_policy(tasks)
            return {tasks[i].task_id: best_solution[i] for i in range(len(tasks))}

    # ------------------------------------------------------------------ #
    #  调度入口                                                             #
    # ------------------------------------------------------------------ #

    def schedule(self):
        """优化的调度入口"""
        ready_tasks = [t for t in self.sim.tasks.values() if t.status == TaskStatus.READY]

        if not ready_tasks:
            return

        schedule_plan = self.schedule_batch(ready_tasks)

        for task in ready_tasks:
            if task.task_id not in schedule_plan:
                continue

            server_id = schedule_plan[task.task_id]

            # Bug 修复4：与 PSO 一致，资源不足时回退到最空闲服务器，
            #           而非直接跳过导致任务永久卡在 READY 状态。
            target_server = self._pick_server(task, preferred_server_id=server_id)
            if target_server is None:
                self.logger.warning(
                    f"GA: 所有服务器均无法承载任务{task.task_id}，将在下轮重试"
                )
                continue

            src = task.assigned_server if task.assigned_server is not None else self.cloud_server_id
            transfer_time = self.sim.network.estimate_transfer_time(
                src, target_server.server_id, task.output_size
            )

            task.assigned_server = target_server.server_id
            effective_priority = 1 / max(transfer_time, 1e-6)
            target_server.add_task(task, priority=effective_priority)

    def _pick_server(self, task, preferred_server_id):
        """
        优先使用 GA 推荐的服务器；资源不足时按剩余算力降序兜底。
        全部不可用返回 None。
        """
        if preferred_server_id in self.sim.servers:
            preferred = self.sim.servers[preferred_server_id]
            if preferred.can_allocate(task):
                return preferred

        candidates = sorted(
            self.sim.servers.values(),
            key=lambda s: s.total_compute - s.used_compute,
            reverse=True
        )
        for server in candidates:
            if server.can_allocate(task):
                return server
        return None