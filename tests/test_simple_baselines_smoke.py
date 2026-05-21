"""
test_simple_baselines_smoke.py — 验证 LeastLoaded 与 ShortestQueue 两个简单 baseline

验证：
  T1. LeastLoaded 把第一个任务放到 used_compute 最低的服务器
  T2. ShortestQueue 把第一个任务放到 queue 最短的服务器
  T3. 两个 scheduler 都能完整跑完一个小推理负载
  T4. can_allocate 全失败时不崩，任务保留为 READY

直接 python tests/test_simple_baselines_smoke.py 运行。
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.simulation import Simulation
from environment.task import Task, TaskStatus
from scheduler.LeastLoadedScheduler import LeastLoadedScheduler
from scheduler.ShortestQueueScheduler import ShortestQueueScheduler


def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b}")


# ====================================================================
# T1. LeastLoaded 选 used_compute 最低的服务器
# ====================================================================

def test_least_loaded_picks_lowest_util():
    sim = Simulation(num_servers=4)
    # 让 server 0/1 已经有占用，server 2/3 空闲
    sim.servers[0].used_compute = 100.0  # cloud 200, util 0.50
    sim.servers[1].used_compute = 25.0   # edge 50,  util 0.50
    sim.servers[2].used_compute = 0.0    # edge 20,  util 0
    sim.servers[3].used_compute = 0.0    # edge 10,  util 0

    sched = LeastLoadedScheduler(sim)

    # 给一个能装到任何 server 的轻任务
    t = Task(task_id=0, compute_demand=1.0, workload=10.0,
             input_size=1.0, output_size=0.1, dependencies=[])
    t.status = TaskStatus.READY
    sim.add_tasks([t])
    sched.schedule()

    # 最低 util 是 server 2 或 3（都 0），按 server_id tie-break 应选 2
    assert t.assigned_server == 2, \
        f"least-loaded 应选 server 2 (util=0, smaller id), got {t.assigned_server}"
    print(f"  [PASS] T1 least loaded picks server {t.assigned_server}")


# ====================================================================
# T2. ShortestQueue 选 queue 最短
# ====================================================================

def test_shortest_queue_picks_shortest():
    sim = Simulation(num_servers=4)
    # 给 server 0/1 队列填一些假任务
    for sid, n in [(0, 5), (1, 3), (2, 0), (3, 2)]:
        for i in range(n):
            fake = Task(task_id=10000 + sid * 100 + i, compute_demand=1.0,
                         workload=10.0, input_size=0.1, output_size=0.01,
                         dependencies=[])
            fake.status = TaskStatus.QUEUED
            import heapq
            heapq.heappush(sim.servers[sid].task_queue, (1.0, fake))

    sched = ShortestQueueScheduler(sim)
    t = Task(task_id=0, compute_demand=1.0, workload=10.0,
             input_size=1.0, output_size=0.1, dependencies=[])
    t.status = TaskStatus.READY
    sim.add_tasks([t])
    sched.schedule()

    # server 2 队列最短（0），应被选中
    assert t.assigned_server == 2, \
        f"shortest-queue 应选 server 2 (queue=0), got {t.assigned_server}"
    print(f"  [PASS] T2 shortest queue picks server {t.assigned_server}")


# ====================================================================
# T3. 两个 scheduler 能完整跑完小推理负载
# ====================================================================

def _run_until_done(sim, scheduler, max_time=500.0):
    t = 0.0
    while len(sim.completed_tasks) < len(sim.tasks) and t < max_time:
        sim.step(scheduler, t)
        t += 0.1
    return t, len(sim.completed_tasks)


def test_least_loaded_full_run():
    random.seed(0)
    sim = Simulation(num_servers=5)
    rng = random.Random(0)
    tasks = Task.generate_inference_workload(
        num_requests=10, task_id_offset=0, rng=rng,
        dist="uniform", arrival_rate=None)
    sim.add_tasks(tasks)
    sched = LeastLoadedScheduler(sim)
    t, done = _run_until_done(sim, sched)
    assert done == len(sim.tasks), \
        f"未全部完成：{done}/{len(sim.tasks)}（max_time 命中？）"
    print(f"  [PASS] T3a LeastLoaded full run: makespan={t:.2f}s, done={done}/{len(sim.tasks)}")


def test_shortest_queue_full_run():
    random.seed(1)
    sim = Simulation(num_servers=5)
    rng = random.Random(1)
    tasks = Task.generate_inference_workload(
        num_requests=10, task_id_offset=0, rng=rng,
        dist="uniform", arrival_rate=None)
    sim.add_tasks(tasks)
    sched = ShortestQueueScheduler(sim)
    t, done = _run_until_done(sim, sched)
    assert done == len(sim.tasks), \
        f"未全部完成：{done}/{len(sim.tasks)}"
    print(f"  [PASS] T3b ShortestQueue full run: makespan={t:.2f}s, done={done}/{len(sim.tasks)}")


# ====================================================================
# T4. can_allocate 全失败时不崩
# ====================================================================

def test_all_servers_full_skips_task():
    sim = Simulation(num_servers=2)
    # 把每台服务器的 compute 全占满
    for s in sim.servers.values():
        s.used_compute = s.total_compute
    sched = LeastLoadedScheduler(sim)

    t = Task(task_id=0, compute_demand=5.0, workload=10.0,
             input_size=1.0, output_size=0.1, dependencies=[])
    t.status = TaskStatus.READY
    sim.add_tasks([t])
    sched.schedule()
    # 没法分配，task 应保留为 READY
    assert t.status == TaskStatus.READY, \
        f"无可用服务器时应保留 READY，got {t.status}"
    print("  [PASS] T4 all-full → task stays READY")


# ====================================================================
# main
# ====================================================================

if __name__ == "__main__":
    tests = [
        test_least_loaded_picks_lowest_util,
        test_shortest_queue_picks_shortest,
        test_least_loaded_full_run,
        test_shortest_queue_full_run,
        test_all_servers_full_skips_task,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print()
    if failed == 0:
        print(f"All {len(tests)} tests passed.")
        sys.exit(0)
    else:
        print(f"{failed}/{len(tests)} tests failed.")
        sys.exit(1)
