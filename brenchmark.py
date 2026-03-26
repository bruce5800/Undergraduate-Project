# benchmark.py (修复版)
from scheduler.RLscheduler import RLScheduler
from scheduler.GAscheduler import GAScheduler
from scheduler.Heftscheduler import HEFTScheduler
from scheduler.PSOscheduler import PSOScheduler
from scheduler.RRscheduler import RoundRobinScheduler
from environment.simulation import Simulation
from environment.task import Task, TaskStatus

import csv
import random
import numpy as np

class BenchmarkTester:
    """基准测试类（支持分阶段记录）"""
    
    def __init__(self, seed=42):
        self.seed = seed
        
        self.schedulers = {
            "RoundRobin": RoundRobinScheduler,
            "RL": RLScheduler,
            "GA": GAScheduler,
            "HEFT": HEFTScheduler
        }

    # ---------------- 初始化环境 ----------------

    def create_simulation(self, task_count, num_edge_servers, seed=42):
        # 修复4：每次创建仿真环境前重置种子，确保所有调度器面对完全相同的任务参数
        random.seed(seed)
        np.random.seed(seed)

        sim = Simulation(num_servers=num_edge_servers)

        all_tasks = []

        count1 = task_count // 3
        count2 = task_count // 3
        count3 = task_count - count1 - count2

        tasks1 = Task.generate_single_dag(task_id_offset=0, num_tasks=count1)
        all_tasks.extend(tasks1)

        tasks2 = Task.generate_linear_dag(task_id_offset=count1, num_tasks=count2)
        all_tasks.extend(tasks2)

        tasks3 = Task.generate_fork_join_dag(task_id_offset=count1 + count2, num_tasks=count3)
        all_tasks.extend(tasks3)

        # 打乱任务加入顺序（task_id 和内部依赖关系保持不变）
        random.shuffle(all_tasks)

        sim.add_tasks(all_tasks)
        return sim

    # ---------------- 执行调度并分阶段记录 ----------------

    def run_scheduler_with_checkpoints(self, scheduler_class, total_task_count, num_edge_servers,
                                       checkpoint_interval=20, max_time=5000):
        # 修复4：每个调度器运行前重置种子，保证公平对比
        random.seed(self.seed)
        np.random.seed(self.seed)

        sim = self.create_simulation(total_task_count, num_edge_servers, seed=self.seed)
        scheduler = scheduler_class(sim)
        current_time = 0

        checkpoints = []

        milestones = list(range(checkpoint_interval, total_task_count + 1, checkpoint_interval))
        if not milestones or milestones[-1] != total_task_count:
            milestones.append(total_task_count)

        current_milestone_idx = 0

        print(f"  总任务数: {total_task_count}, 检查点: {milestones}")

        while len(sim.completed_tasks) < len(sim.tasks) and current_time < max_time:

            # 1. 检查任务完成
            for task in sim.tasks.values():
                if task.status == TaskStatus.RUNNING and task.end_time is not None:
                    if current_time >= task.end_time:
                        task.status = TaskStatus.COMPLETED
                        sim.completed_tasks.add(task.task_id)
                        server = sim.servers[task.assigned_server]
                        server.update_resource(task, allocate=False)
                        if task in server.running_tasks:
                            server.running_tasks.remove(task)

            # 2. 修复3：用 while 循环捕获同一时间步内跨越多个里程碑的情况
            while (current_milestone_idx < len(milestones) and
                   len(sim.completed_tasks) >= milestones[current_milestone_idx]):
                milestone = milestones[current_milestone_idx]
                metrics = self.collect_metrics_at_checkpoint(sim, current_time, milestone)
                checkpoints.append({
                    'completed_tasks': milestone,
                    'metrics': metrics,
                    'time': current_time
                })
                current_milestone_idx += 1

            # 3. 更新任务依赖状态
            for task in sim.tasks.values():
                if task.status == TaskStatus.WAITING and task.check_dependencies(sim.completed_tasks):
                    task.status = TaskStatus.READY
                    # 修复1：在任务变为 READY 时记录就绪时刻，用于计算端到端延迟
                    if task.ready_time is None:
                        task.ready_time = current_time

            # 4. 执行调度
            scheduler.schedule()

            # 5. 处理队列中的任务
            for server in sim.servers.values():
                server.process_tasks(current_time)

            # 6. 增加时间
            current_time += 1

        # 仿真结束后，若仍有未触发的里程碑（max_time 超时），补录最后状态
        while current_milestone_idx < len(milestones):
            metrics = self.collect_metrics_at_checkpoint(
                sim, current_time, len(sim.completed_tasks)
            )
            checkpoints.append({
                'completed_tasks': len(sim.completed_tasks),
                'metrics': metrics,
                'time': current_time
            })
            current_milestone_idx += 1

        return checkpoints

    def collect_metrics_at_checkpoint(self, sim, current_time, target_completed_tasks):
        """在检查点收集性能指标"""

        completed_tasks_list = [
            t for t in sim.tasks.values() if t.task_id in sim.completed_tasks
        ]

        # 修复1：avg_completion_time 改为端到端延迟（READY 时刻 -> 执行完成）
        e2e_times = []
        for task in completed_tasks_list:
            if task.ready_time is not None and task.end_time is not None:
                e2e_times.append(task.end_time - task.ready_time)
        avg_completion_time = sum(e2e_times) / len(e2e_times) if e2e_times else 0

        # 修复2：资源利用率 = 服务器有效忙碌时长 / 当前总时长（结果在 0~1 之间）
        utilization = []
        for server in sim.servers.values():
            busy_time = sum(
                t.end_time - t.start_time
                for t in server.task_history
                if t.task_id in sim.completed_tasks
                   and t.start_time is not None
                   and t.end_time is not None
            )
            util = busy_time / current_time if current_time > 0 else 0
            utilization.append(util)

        avg_utilization = sum(utilization) / len(utilization) if utilization else 0
        load_std = float(np.std(utilization)) if utilization else 0

        return {
            'makespan': current_time,
            'avg_completion_time': avg_completion_time,
            'avg_utilization': avg_utilization,
            'load_balance_std': load_std,
            'completed_tasks': target_completed_tasks,
            'total_tasks': len(sim.tasks)
        }

    # ---------------- 运行基准测试 ----------------

    def run_benchmark_with_checkpoints(self, edge_server_counts, max_task_count=500,
                                       checkpoint_interval=20,
                                       output_file="figs/benchmark_results.csv"):
        results = []

        for edge_count in edge_server_counts:
            print(f"\n{'='*60}")
            print(f"测试边缘服务器数量: {edge_count}")
            print('='*60)

            for scheduler_name, scheduler_class in self.schedulers.items():
                print(f"\n调度器: {scheduler_name}")
                print("-" * 40)

                checkpoints = self.run_scheduler_with_checkpoints(
                    scheduler_class,
                    total_task_count=max_task_count,
                    num_edge_servers=edge_count,
                    checkpoint_interval=checkpoint_interval
                )

                for checkpoint in checkpoints:
                    results.append({
                        '边缘服务器数量': edge_count,
                        '调度器': scheduler_name,
                        '任务数量': checkpoint['completed_tasks'],
                        '总运行时间(makespan)': checkpoint['metrics']['makespan'],
                        '平均端到端延迟': checkpoint['metrics']['avg_completion_time'],
                        '平均利用率': checkpoint['metrics']['avg_utilization'],
                        '负载均衡标准差': checkpoint['metrics']['load_balance_std'],
                        '已完成任务数': checkpoint['completed_tasks'],
                        '总任务数': max_task_count
                    })

                if checkpoints:
                    last = checkpoints[-1]
                    print(f"  最终完成: {last['completed_tasks']}/{max_task_count} 任务")
                    print(f"  Makespan: {last['metrics']['makespan']:.1f}")
                    print(f"  平均端到端延迟: {last['metrics']['avg_completion_time']:.4f}")
                    print(f"  平均利用率: {last['metrics']['avg_utilization']:.4f}")

        self.export_detailed_results(results, output_file)
        return results

    def export_detailed_results(self, results, output_file):
        """导出详细结果到CSV文件"""
        if not results:
            print("没有结果可导出")
            return

        fieldnames = [
            '边缘服务器数量', '调度器', '任务数量',
            '总运行时间(makespan)', '平均端到端延迟', '平均利用率', '负载均衡标准差',
            '已完成任务数', '总任务数'
        ]

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"\n详细结果已导出到: {output_file}")


# ---------------- 主程序 ----------------

if __name__ == "__main__":
    print("=" * 70)
    print("边缘计算调度算法基准测试")
    print("=" * 70)

    tester = BenchmarkTester(seed=42)

    edge_server_counts = [3, 5, 7]
    max_task_count = 300
    checkpoint_interval = 20

    print(f"\n开始检查点基准测试...")
    print(f"边缘服务器数量: {edge_server_counts}")
    print(f"最大任务数量: {max_task_count}")
    print(f"检查点间隔: 每{checkpoint_interval}个任务记录一次")
    print(f"预计数据点: {max_task_count // checkpoint_interval} 个/调度器")

    results = tester.run_benchmark_with_checkpoints(
        edge_server_counts,
        max_task_count=max_task_count,
        checkpoint_interval=checkpoint_interval,
        output_file="figs/benchmark_results.csv"
    )

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)