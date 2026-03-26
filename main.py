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
    time_step = 1  # 时间步长（秒）
    total_tasks = len(all_tasks)
    
    
    def update(frame):
        nonlocal current_time

        # 1. 先检查任务完成（处理上一个时间步的任务）
        for task in sim.tasks.values():
            if task.status == TaskStatus.RUNNING and task.end_time is not None:
                if current_time >= task.end_time:
                    task.status = TaskStatus.COMPLETED
                    sim.completed_tasks.add(task.task_id)
                    server = sim.servers[task.assigned_server]
                    server.update_resource(task, allocate=False)

                    # 从服务器的运行列表中移除
                    if task in server.running_tasks:
                        server.running_tasks.remove(task)

                    logging.info(f"任务 {task.task_id} 完成于服务器 {task.assigned_server} "
                                f"(耗时: {task.end_time - task.start_time:.1f}s)")

        # 2. 更新任务依赖状态
        for task in sim.tasks.values():
            if task.status == TaskStatus.WAITING and task.check_dependencies(sim.completed_tasks):
                task.status = TaskStatus.READY
                if task.ready_time is None:
                    task.ready_time = current_time
                logging.info(f"任务 {task.task_id} 依赖满足，进入就绪状态")

        # 3. 执行调度（把READY任务放入队列）
        scheduler.schedule()

        # 4. 处理队列中的任务（开始执行，设置end_time）
        for s_id, server in sim.servers.items():
            prev_running = len(server.running_tasks)
            server.process_tasks(current_time)  # 这会设置end_time
            if len(server.running_tasks) > prev_running:
                task = server.running_tasks[-1]
                logging.info(f"服务器 {s_id} 开始执行任务 {task.task_id} "
                            f"(预计完成: {task.end_time:.1f}s)")

        # 5. 增加时间
        current_time += time_step
        
        # 6. 更新可视化
        visualizer.update_plots(current_time)

        # 7. 检查终止条件
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
