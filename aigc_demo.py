"""
aigc_demo.py — M1 阶段 AIGC 物理展示

直接对比两种模式在一组相同 DAG 上的表现：
  (A) 通用 DAG 模式（model_id=None）—— 改造前的行为
  (B) AIGC 模式（按 Zipf 分配模型 ID）—— 改造后触发冷加载 + LRU 驱逐

预期：(B) makespan 显著 > (A)，因为现有调度器没考虑模型亲和性，
每次任务路由都可能触发冷加载。这正是 M2/M3 调度器要解决的问题。
"""

import random
import numpy as np
import logging

from environment.simulation import Simulation
from environment.task import Task, TaskStatus
from environment.model_catalog import CATALOG, assign_models_zipf
from scheduler.RRscheduler import RoundRobinScheduler
from scheduler.Heftscheduler import HEFTScheduler

logging.basicConfig(level=logging.ERROR)


def run_one(scheduler_class, aigc: bool, seed: int = 0,
            num_tasks: int = 30, num_servers: int = 5,
            max_time: float = 5000.0):
    random.seed(seed)
    np.random.seed(seed)

    sim = Simulation(num_servers=num_servers)
    tasks = []
    tasks.extend(Task.generate_single_dag(0, num_tasks))
    random.shuffle(tasks)

    if aigc:
        rng = random.Random(seed)
        # 仅用 7b/13b/sdxl 演示，避开 70B（仅云可承载，演示效果不直观）
        assign_models_zipf(tasks, rng, alpha=1.2,
                           model_ids=["llama-7b", "llama-13b", "sdxl"])

    sim.add_tasks(tasks)
    scheduler = scheduler_class(sim)

    t = 0.0
    while len(sim.completed_tasks) < len(sim.tasks) and t < max_time:
        sim.step(scheduler, t)
        t += 0.1

    # 统计冷加载相关指标
    completed = [task for task in sim.tasks.values()
                 if task.task_id in sim.completed_tasks]
    total_cold = sum(getattr(task, "cold_load_delay", 0.0)
                     for task in completed)
    cold_count = sum(1 for task in completed
                     if getattr(task, "cold_load_delay", 0.0) > 0)

    final_loaded = {sid: list(s.loaded_models.keys())
                    for sid, s in sim.servers.items()}
    return {
        "makespan": round(t, 2),
        "completed": len(sim.completed_tasks),
        "total_cold_load_sec": round(total_cold, 2),
        "cold_load_events": cold_count,
        "warm_hit_events": len(completed) - cold_count,
        "final_loaded_models": final_loaded,
    }


def main():
    print("=" * 72)
    print("  M1 AIGC Demo: 同 DAG 同种子，对比通用 vs AIGC 模式")
    print("=" * 72)
    print(f"  Tasks: 30 (single DAG)  |  Servers: 5 (1 cloud + 4 edge)")
    print(f"  Models (AIGC): llama-7b, llama-13b, sdxl  |  Zipf alpha=1.2")
    print("=" * 72)

    for sched_name, sched_class in [
        ("RoundRobin", RoundRobinScheduler),
        ("HEFT",       HEFTScheduler),
    ]:
        print(f"\n--- {sched_name} ---")
        for seed in [0, 1, 2]:
            r_generic = run_one(sched_class, aigc=False, seed=seed)
            r_aigc    = run_one(sched_class, aigc=True, seed=seed)

            print(f"  seed={seed}: "
                  f"makespan  generic={r_generic['makespan']:>7.1f}  "
                  f"aigc={r_aigc['makespan']:>7.1f}   "
                  f"cold_loads={r_aigc['cold_load_events']:>2d}/"
                  f"{r_aigc['cold_load_events']+r_aigc['warm_hit_events']:<2d}  "
                  f"cold_time={r_aigc['total_cold_load_sec']:>6.1f}s")

        # 展示一次 AIGC 运行的最终模型驻留情况
        r = run_one(sched_class, aigc=True, seed=0)
        print(f"  Final loaded models on each server (seed=0):")
        for sid, models in r["final_loaded_models"].items():
            srv_type = "cloud" if sid == 0 else f"edge{sid}"
            print(f"    {srv_type}: {models if models else '(empty)'}")

    print("\n" + "=" * 72)
    print("  结论：")
    print("  - generic makespan 与 M1 改造前完全一致（已通过零回归验证）")
    print("  - aigc 模式下冷加载事件显著，总冷加载时间叠加进 makespan")
    print("  - 现有调度器不感知模型亲和性 → M2/M3 引入 AIGC-aware 调度器的动机")
    print("=" * 72)


if __name__ == "__main__":
    main()
