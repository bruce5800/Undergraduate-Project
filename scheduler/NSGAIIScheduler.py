"""
NSGAIIScheduler — 多目标遗传算法 (NSGA-II) 基线

R1 修订新增：补齐基线家族中"多目标元启发式搜索"这一空档，对应
Yu et al. (arXiv:2507.15553) 用 NSGA-II 做云边 LLM 实例间多目标
逐请求路由的设定。

与 GAScheduler 的关系：继承并复用染色体编码（READY 任务 -> server_id
列表）、种群初始化、两点交叉、自适应变异与全部派发管线，仅替换主循环：

  - 双目标适应度 (f1, f2)，均为最小化：
      f1 = 完成时间代理：所有服务器中最大的"现有负载 + 新指派执行时间
           + 传输 + 冷加载"估计（makespan 代理）
      f2 = 能耗代理：各任务 估计时长 x 放置后服务器功率估计 之和
    两个目标与论文的 (SLO, E/tok) 轴一一对应，估计方式与 CS-UCB
    基线一致（只用通用可观测接口，不读 AIGC 状态）
  - 快速非支配排序 + 拥挤距离（Deb et al., 2002）
  - 二元锦标赛选择，按 (rank, -crowding) 排序
  - 从末代第一前沿取 knee point（到理想点归一化欧氏距离最小的解）
    作为执行方案——多目标前沿到单一执行策略的标准折中
"""

import math
import random
from typing import List

from environment.energy import POWER_PROFILES
from scheduler.GAscheduler import GAScheduler


class NSGAIIScheduler(GAScheduler):
    """NSGA-II 多目标调度器（继承 GA 的编码/算子/派发管线）"""

    def __init__(self, sim_env, population_size=30, generations=30,
                 crossover_rate=0.8, mutation_rate=0.1):
        # elitism_rate 由非支配排序取代，传 0 占位
        super().__init__(sim_env, population_size=population_size,
                         generations=generations,
                         crossover_rate=crossover_rate,
                         mutation_rate=mutation_rate,
                         elitism_rate=0.0)

    # ------------------------------------------------------------------ #
    #  双目标适应度                                                         #
    # ------------------------------------------------------------------ #

    def _power_est(self, server, util_after: float) -> float:
        p = POWER_PROFILES.get(getattr(server, "power_profile", None))
        if p is None:
            return 0.0
        u = min(max(util_after, 0.0), 1.0)
        return p["idle_W"] + (p["max_W"] - p["idle_W"]) * u

    def evaluate_objectives(self, individual: List, tasks: List):
        """返回 (f1 完成时间代理, f2 能耗代理)，均为最小化。"""
        servers = self.sim.servers
        # 各服务器现有相对负载作为基线
        finish = {sid: servers[sid].used_compute
                  / max(servers[sid].total_compute, 1e-6)
                  for sid in self.server_ids}
        energy = 0.0
        for i, sid in enumerate(individual):
            task = tasks[i]
            server = servers[sid]
            src = task.assigned_server if task.assigned_server is not None \
                else self.cloud_server_id
            transfer = self.sim.network.estimate_transfer_time(
                src, sid, task.output_size)
            cold = server.cold_load_cost(task.model_id) \
                if task.model_id is not None else 0.0
            exec_time = task.workload / max(server.total_compute, 1e-6)
            t = transfer + cold + exec_time
            finish[sid] += t

            util_after = ((server.used_compute + task.compute_demand)
                          / max(server.total_compute, 1e-6))
            energy += t * self._power_est(server, util_after)
        return max(finish.values()), energy

    # ------------------------------------------------------------------ #
    #  NSGA-II 核心：非支配排序 + 拥挤距离                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dominates(a, b):
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def _fast_nondominated_sort(self, objs):
        n = len(objs)
        S = [[] for _ in range(n)]
        dom_count = [0] * n
        fronts = [[]]
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._dominates(objs[p], objs[q]):
                    S[p].append(q)
                elif self._dominates(objs[q], objs[p]):
                    dom_count[p] += 1
            if dom_count[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in S[p]:
                    dom_count[q] -= 1
                    if dom_count[q] == 0:
                        nxt.append(q)
            i += 1
            fronts.append(nxt)
        return fronts[:-1]

    @staticmethod
    def _crowding_distance(front, objs):
        dist = {i: 0.0 for i in front}
        if len(front) <= 2:
            for i in front:
                dist[i] = math.inf
            return dist
        for m in range(2):
            ordered = sorted(front, key=lambda i: objs[i][m])
            dist[ordered[0]] = dist[ordered[-1]] = math.inf
            span = objs[ordered[-1]][m] - objs[ordered[0]][m]
            if span <= 0:
                continue
            for k in range(1, len(ordered) - 1):
                dist[ordered[k]] += ((objs[ordered[k + 1]][m]
                                      - objs[ordered[k - 1]][m]) / span)
        return dist

    # ------------------------------------------------------------------ #
    #  主循环（覆盖 GA 的 update_policy；派发管线复用父类）                    #
    # ------------------------------------------------------------------ #

    def update_policy(self, tasks: List) -> List:
        pop = self.initialize_population(tasks)
        pop = pop[:self.population_size]

        for _ in range(self.generations):
            # 变异产生子代（二元锦标赛按 rank/crowding 选父本）
            objs = [self.evaluate_objectives(ind, tasks) for ind in pop]
            fronts = self._fast_nondominated_sort(objs)
            rank = {}
            crowd = {}
            for r, front in enumerate(fronts):
                cd = self._crowding_distance(front, objs)
                for i in front:
                    rank[i] = r
                    crowd[i] = cd[i]

            def tournament():
                a, b = random.randrange(len(pop)), random.randrange(len(pop))
                if (rank[a], -crowd[a]) <= (rank[b], -crowd[b]):
                    return pop[a]
                return pop[b]

            offspring = []
            while len(offspring) < self.population_size:
                c1, c2 = self.two_point_crossover(tournament(), tournament())
                offspring.append(self.adaptive_mutate(c1, tasks))
                if len(offspring) < self.population_size:
                    offspring.append(self.adaptive_mutate(c2, tasks))

            # 环境选择：父代 + 子代合并后按 (rank, crowding) 截断
            union = pop + offspring
            u_objs = [self.evaluate_objectives(ind, tasks) for ind in union]
            u_fronts = self._fast_nondominated_sort(u_objs)
            new_pop = []
            for front in u_fronts:
                if len(new_pop) + len(front) <= self.population_size:
                    new_pop.extend(union[i] for i in front)
                else:
                    cd = self._crowding_distance(front, u_objs)
                    rest = sorted(front, key=lambda i: -cd[i])
                    new_pop.extend(
                        union[i] for i in
                        rest[:self.population_size - len(new_pop)])
                    break
            pop = new_pop

        # knee point：第一前沿中到理想点归一化距离最小的解
        objs = [self.evaluate_objectives(ind, tasks) for ind in pop]
        first = self._fast_nondominated_sort(objs)[0]
        f1s = [objs[i][0] for i in first]
        f2s = [objs[i][1] for i in first]
        f1_min, f1_max = min(f1s), max(f1s)
        f2_min, f2_max = min(f2s), max(f2s)

        def knee_dist(i):
            d1 = ((objs[i][0] - f1_min) / (f1_max - f1_min)
                  if f1_max > f1_min else 0.0)
            d2 = ((objs[i][1] - f2_min) / (f2_max - f2_min)
                  if f2_max > f2_min else 0.0)
            return d1 * d1 + d2 * d2

        best = min(first, key=knee_dist)
        return pop[best].copy()
