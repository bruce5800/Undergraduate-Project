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
        """设置服务器节点（模拟AIGC云边推理场景）"""
        # 云服务器：高算力GPU集群
        cloud = Server(0, ServerType.CLOUD,
                      compute_capacity=200, memory=128,
                      storage=500, bandwidth=1000)

        # 边缘服务器：异构GPU节点（强/中/弱三档）
        #   (id, compute_TFLOPS, memory_GB, storage_GB, bandwidth_Mbps)
        edge_configs = [
            (1, 50, 64, 256, 500),    # 强 - 如 A100 推理卡
            (2, 20, 32, 128, 300),    # 中 - 如 T4
            (3, 10, 16,  64, 200),    # 弱 - 如 Jetson AGX
            (4, 50, 64, 256, 500),    # 强
            (5, 10, 16,  64, 200),    # 弱
            (6, 20, 32, 128, 300),    # 中
            (7, 10, 16,  64, 200)     # 弱
        ]
        
        edge_servers = []
        for i, config in enumerate(edge_configs[:num_servers-1]):
            server_id, compute, memory, storage, bandwidth = config
            edge_servers.append(
                Server(server_id, ServerType.EDGE, compute, memory, storage, bandwidth)
            )
        
        self.servers = {s.server_id: s for s in [cloud] + edge_servers}

    def setup_network(self):
        """配置网络拓扑（云边 + 边边链路）"""
        # 云-边缘链路：高延迟、中等带宽（典型广域网）
        for i in range(1, len(self.servers)):
            latency = 30 + i * 5           # 35~65 ms
            bandwidth = 400 - i * 30       # 370~190 Mbps
            self.network.add_link(0, i, latency=latency, bandwidth=bandwidth)

        # 边缘-边缘链路：低延迟、高带宽（同园区/同城网络）
        for i in range(1, len(self.servers)):
            for j in range(i + 1, len(self.servers)):
                latency = 5 + abs(i - j) * 3       # 8~23 ms
                bandwidth = 800 - abs(i - j) * 50   # 750~500 Mbps
                self.network.add_link(i, j, latency=latency, bandwidth=bandwidth)