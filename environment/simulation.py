import logging
from environment.server import Server, ServerType
from environment.network import Network
from environment.task import TaskStatus

logger = logging.getLogger(__name__)


class Simulation:
    def __init__(self, num_servers=4):
        self.tasks = {}
        self.network = Network()
        self.completed_tasks = set()
        self.setup_servers(num_servers)
        self.setup_network()

    def add_tasks(self, tasks):
        """添加任务到仿真环境"""
        for task in tasks:
            self.tasks[task.task_id] = task

    def step(self, scheduler, current_time):
        """执行一个仿真时间步：完成检查 -> 依赖更新 -> 调度 -> 处理队列"""
        # 1. 检查任务完成
        for task in self.tasks.values():
            if task.status == TaskStatus.RUNNING and task.end_time is not None:
                if current_time >= task.end_time:
                    task.status = TaskStatus.COMPLETED
                    self.completed_tasks.add(task.task_id)
                    server = self.servers[task.assigned_server]
                    server.update_resource(task, allocate=False)
                    if task in server.running_tasks:
                        server.running_tasks.remove(task)

        # 2. 更新任务依赖状态
        for task in self.tasks.values():
            if task.status == TaskStatus.WAITING and task.check_dependencies(self.completed_tasks):
                task.status = TaskStatus.READY
                if task.ready_time is None:
                    task.ready_time = current_time

        # 3. 执行调度
        scheduler.schedule()

        # 4. 处理队列中的任务
        for server in self.servers.values():
            server.process_tasks(current_time)

    def setup_servers(self, num_servers):
        """设置服务器节点"""
        # 创建云服务器
        cloud = Server(0, ServerType.CLOUD, 
                      compute_capacity=500, memory=1000, 
                      storage=2000, bandwidth=1000)
        
        # 创建边缘服务器
        edge_configs = [
            (1, 150, 256, 512, 500),   # 服务器1
            (2, 200, 512, 1024, 800),  # 服务器2
            (3, 100, 128, 256, 300),   # 服务器3
            (4, 200, 512, 1024, 800),  # 服务器4
            (5, 100, 128, 256, 300),   # 服务器5
            (6, 200, 512, 1024, 800),  # 服务器6
            (7, 100, 128, 256, 300)    # 服务器7
        ]
        
        edge_servers = []
        for i, config in enumerate(edge_configs[:num_servers-1]):
            server_id, compute, memory, storage, bandwidth = config
            edge_servers.append(
                Server(server_id, ServerType.EDGE, compute, memory, storage, bandwidth)
            )
        
        self.servers = {s.server_id: s for s in [cloud] + edge_servers}

    def setup_network(self):
        """配置网络拓扑"""
        # 云服务器到边缘服务器的连接
        for i in range(1, len(self.servers)):
            latency = 40 + i * 5  # 不同延迟
            bandwidth = 500 - i * 30  # 不同带宽
            self.network.add_link(0, i, latency=latency, bandwidth=bandwidth)
        
        # 边缘服务器之间的连接（add_link 内部已处理双向）
        for i in range(1, len(self.servers)):
            for j in range(i + 1, len(self.servers)):
                latency = 20 + abs(i - j) * 5
                bandwidth = 600 - abs(i - j) * 50
                self.network.add_link(i, j, latency=latency, bandwidth=bandwidth)