import random
from enum import Enum
from typing import List, Set

class TaskStatus(Enum):
    WAITING = 1   # 依赖未满足
    READY = 2     # 依赖已满足，等待调度
    QUEUED = 3    # 已加入服务器队列，等待资源
    RUNNING = 4   # 正在执行
    COMPLETED = 5 # 已完成

class Task:
    def __init__(self, task_id: int,
                 compute_demand: float,  # 并发GPU算力占用 (TFLOPS)，用于资源分配检查
                 workload: float,        # 总计算工作量 (TFLOPS)，用于计算执行时间
                 input_size: float,      # 输入数据/模型内存占用 (GB)
                 output_size: float,     # 输出数据大小 (GB)，同时决定传输时间
                 dependencies: List[int],# 依赖任务ID列表
                 priority=1
                ):
        self.task_id = task_id
        self.compute_demand = compute_demand
        self.workload = workload
        self.input_size = input_size
        self.output_size = output_size
        self.dependencies = dependencies
        self.priority = priority

        self.status = TaskStatus.WAITING
        self.ready_time = None      # 依赖满足、进入READY状态的时刻
        self.start_time = None
        self.end_time = None
        self.assigned_server = None  # 分配的服务器
        self.transfer_delay = 0.0   # 数据传输延迟（秒），由调度器设置
    
    def check_dependencies(self, completed_tasks: Set[int]) -> bool:
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def __lt__(self, other):
        return self.task_id < other.task_id

    # ---------- 复杂DAG结构：15任务模板，并联重复 ----------
    @staticmethod
    def generate_single_dag(task_id_offset: int = 0, num_tasks: int = 15) -> List['Task']:
        """
        生成复杂DAG结构，支持任意任务数量
        通过重复15任务模板并联实现，每个模板是独立的DAG子图
        """
        tasks = []
        current_id = task_id_offset
        remaining = num_tasks
        block_size = 15
        
        while remaining > 0:
            current_block_size = min(block_size, remaining)
            block_tasks = Task._generate_single_dag_block(current_id, current_block_size)
            tasks.extend(block_tasks)
            current_id += current_block_size
            remaining -= current_block_size
        
        return tasks

    @staticmethod
    def _generate_single_dag_block(start_id: int, num_tasks: int) -> List['Task']:
        """生成单个15任务DAG块（独立子图，内部有依赖，不连接外部）"""
        dep_map = {
            0: [],
            1: [0], 2: [0], 3: [0],
            4: [1], 5: [1], 6: [2], 7: [2], 8: [3], 9: [3],
            10: [4, 6], 11: [5, 8], 12: [7, 9],
            13: [10, 11], 14: [12, 13]
        }

        tasks = []
        for local_id in range(num_tasks):
            global_id = start_id + local_id
            deps = [start_id + d for d in dep_map.get(local_id, []) if d < num_tasks]
            task = Task(
                task_id=global_id,
                compute_demand=random.uniform(3, 20),
                workload=random.uniform(100, 2000),
                input_size=random.uniform(2, 8),
                output_size=random.uniform(0.05, 0.3),
                dependencies=deps
            )
            tasks.append(task)
        return tasks

    # ---------- 链式结构：10任务一组，并联重复 ----------
    @staticmethod
    def generate_linear_dag(task_id_offset: int = 0, num_tasks: int = 10) -> List['Task']:
        """
        生成链式结构，支持任意任务数量
        每10个任务组成一个独立的链式DAG子图，子图之间无依赖
        """
        tasks = []
        current_id = task_id_offset
        remaining = num_tasks
        block_size = 10
        
        while remaining > 0:
            current_block_size = min(block_size, remaining)
            
            for i in range(current_block_size):
                global_id = current_id + i
                deps = [global_id - 1] if i > 0 else []
                task = Task(
                    task_id=global_id,
                    compute_demand=random.uniform(3, 20),
                    workload=random.uniform(100, 2000),
                    input_size=random.uniform(2, 8),
                    output_size=random.uniform(0.05, 0.3),
                    dependencies=deps
                )
                tasks.append(task)
            
            current_id += current_block_size
            remaining -= current_block_size
        
        return tasks

    # ---------- Fork-Join结构：9任务模板，并联重复 ----------
    @staticmethod
    def generate_fork_join_dag(task_id_offset: int = 0, num_tasks: int = 9) -> List['Task']:
        """
        生成Fork-Join结构，支持任意任务数量
        通过重复9任务模板并联实现，每个模板是独立的DAG子图
        """
        tasks = []
        current_id = task_id_offset
        remaining = num_tasks
        block_size = 9
        
        while remaining > 0:
            current_block_size = min(block_size, remaining)
            block_tasks = Task._generate_fork_join_block(current_id, current_block_size)
            tasks.extend(block_tasks)
            current_id += current_block_size
            remaining -= current_block_size
        
        return tasks

    @staticmethod
    def _generate_fork_join_block(start_id: int, num_tasks: int) -> List['Task']:
        """生成单个9任务Fork-Join块（独立子图）"""
        dep_map = {
            0: [],
            1: [0], 2: [0], 3: [0],
            4: [1, 2],
            5: [3, 4],
            6: [5], 7: [5],
            8: [6, 7]
        }

        tasks = []
        for local_id in range(num_tasks):
            global_id = start_id + local_id
            deps = [start_id + d for d in dep_map.get(local_id, []) if d < num_tasks]
            task = Task(
                task_id=global_id,
                compute_demand=random.uniform(3, 20),
                workload=random.uniform(100, 2000),
                input_size=random.uniform(2, 8),
                output_size=random.uniform(0.05, 0.3),
                dependencies=deps
            )
            tasks.append(task)
        return tasks