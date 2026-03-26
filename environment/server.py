from enum import Enum
import heapq
from environment.task import Task, TaskStatus

class ServerType(Enum):
    CLOUD = 1
    EDGE = 2

class Server:
    def __init__(self, 
                 server_id: int,
                 server_type: ServerType,
                 compute_capacity: float, # 计算能力 (TFLOPS)
                 memory: float,           # 内存容量 (GB)
                 storage: float,          # 存储容量 (GB)
                 bandwidth: float,        # 上行带宽 (Mbps)
                 location: str = None     # 位置 (可选，针对边缘服务器)
                ):
        self.server_id = server_id
        self.type = server_type
        self.total_compute = compute_capacity
        self.total_memory = memory
        self.total_storage = storage
        self.bandwidth = bandwidth
        self.location = location  # 位置（针对边缘服务器）
        
        self.used_compute = 0.0
        self.used_memory = 0.0
        self.used_storage = 0.0
        
        self.running_tasks = []  # 当前运行任务
        self.task_queue = []     # 任务等待队列 (优先级队列)
        self.task_history = []   # 已完成任务记录

    def can_allocate(self, task: Task) -> bool:
        """检查是否满足资源需求"""
        return (self.used_compute + task.compute_demand <= self.total_compute and
                self.used_memory + task.input_size <= self.total_memory and
                self.used_storage + task.output_size <= self.total_storage)

    def add_task(self, task: Task, priority: float):
        """添加任务到等待队列"""
        heapq.heappush(self.task_queue, (priority, task))
        
    def update_resource(self, task: Task, allocate: bool):
        """更新资源分配"""
        if allocate:
            self.used_compute += task.compute_demand
            self.used_memory += task.input_size
            self.used_storage += task.output_size
        else:
            self.used_compute -= task.compute_demand
            self.used_memory -= task.input_size
            self.used_storage -= task.output_size

    # 任务执行逻辑
    def process_tasks(self, current_time):
        """处理任务队列，尽可能多地启动任务"""
        while self.task_queue:
            priority, task = heapq.heappop(self.task_queue)
            if task.status == TaskStatus.COMPLETED:  # 防止重复处理
                continue

            if self.can_allocate(task):
                # 更新任务状态为运行中
                task.status = TaskStatus.RUNNING
                task.assigned_server = self.server_id

                # 计算执行时间
                exec_time = task.compute_demand / self.total_compute
                task.start_time = current_time
                task.end_time = current_time + exec_time

                self.running_tasks.append(task)
                self.task_history.append(task)
                self.update_resource(task, allocate=True)
            else:
                # 资源不足，放回队列，停止本轮处理
                heapq.heappush(self.task_queue, (priority, task))
                break