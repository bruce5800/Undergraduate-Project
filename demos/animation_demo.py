import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from environment.simulation import Simulation
from environment.task import Task, TaskStatus
from visualizer import TaskVisualizer
from scheduler.RLscheduler import RLScheduler
from scheduler.GAscheduler import GAScheduler
from scheduler.PSOscheduler import PSOScheduler
from scheduler.RRscheduler import RoundRobinScheduler

def main():
    # 设置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True,
        handlers=[
            logging.StreamHandler(),  # 控制台
            logging.FileHandler('logs/RLscheduler.log', encoding='utf-8')  # 文件
        ]
    )

    # 初始化仿真环境
    sim = Simulation(num_servers=4)

    # 创建任务
    all_tasks = []

    # 任务组1: 复杂DAG结构 (15个任务)
    tasks1 = Task.generate_single_dag(task_id_offset=0, num_tasks=15)
    all_tasks.extend(tasks1)
    logging.info(f"生成任务组1: {len(tasks1)}个任务")

    # 任务组2: 链式结构 (10个任务)
    tasks2 = Task.generate_linear_dag(task_id_offset=len(all_tasks), num_tasks=10)
    all_tasks.extend(tasks2)
    logging.info(f"生成任务组2: {len(tasks2)}个任务")

    # 任务组3: Fork-Join结构 (9个任务)
    tasks3 = Task.generate_fork_join_dag(task_id_offset=len(all_tasks), num_tasks=9)
    all_tasks.extend(tasks3)
    logging.info(f"生成任务组3: {len(tasks3)}个任务")

    # 将任务添加到仿真环境
    sim.add_tasks(all_tasks)

    # 初始化强化学习调度器
    scheduler = RLScheduler(sim)

    # 初始化可视化
    visualizer = TaskVisualizer(sim)

    # 仿真循环
    current_time = 0
    time_step = 0.1  # 时间步长（秒）
    total_tasks = len(all_tasks)


    def update(frame):
        nonlocal current_time

        # 执行一个仿真时间步
        sim.step(scheduler, current_time)

        # 增加时间
        current_time += time_step

        # 更新可视化
        visualizer.update_plots(current_time)

        # 检查终止条件
        if len(sim.completed_tasks) == total_tasks:
            ani.event_source.stop()
            logging.info(f"所有任务完成! 总耗时: {current_time:.1f}s")
            print(f"总任务数: {total_tasks}")
            print(f"总耗时: {current_time:.1f}秒")

            # 打印统计信息
            for s_id, server in sim.servers.items():
                completed_count = len([t for t in server.task_history
                                      if t.status == TaskStatus.COMPLETED])
                print(f"服务器 {s_id}: 完成 {completed_count} 个任务")

    ani = FuncAnimation(visualizer.fig, update, interval=200, save_count=1000)
    plt.show()

if __name__ == "__main__":
    main()
