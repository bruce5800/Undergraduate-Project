import logging
from environment.task import TaskStatus
from environment.server import ServerType

class RoundRobinScheduler:

    logger = logging.getLogger(__name__)

    def __init__(self, sim_env):
        """
        轮询调度器
        """
        self.sim = sim_env
        self.server_ids = sorted(sim_env.servers.keys())  # 确保固定顺序
        self.current_idx = 0  # 当前分配的服务器索引
        self.fallback_server = next(  # 设置云服务器为回退选项
            s.server_id for s in self.sim.servers.values() 
            if s.type == ServerType.CLOUD
        )

    def get_next_server(self):
        """ 获取下一个可用服务器（带故障转移机制） """
        start_idx = self.current_idx
        while True:
            server_id = self.server_ids[self.current_idx]
            self.current_idx = (self.current_idx + 1) % len(self.server_ids)
            
            # 检查服务器是否可用
            if self.sim.servers[server_id]:
                return server_id
            
            # 如果循环一周都没有可用服务器，使用云服务器
            if self.current_idx == start_idx:
                return self.fallback_server

    def schedule(self):
        """ 执行轮询调度 """
        ready_tasks = [t for t in self.sim.tasks.values() 
                      if t.status == TaskStatus.READY]
        
        if not ready_tasks:
            return

        # self.logger.info(f"RoundRobin scheduling {len(ready_tasks)} tasks...")
        
        for task in ready_tasks:
            target_server_id = self.get_next_server()
            target_server = self.sim.servers[target_server_id]
            
            # 处理任务转移逻辑
            if task.assigned_server is not None:
                src = task.assigned_server
            else:
                src = self.fallback_server
            
            # 计算传输时间
            transfer_time = self.sim.network.estimate_transfer_time(
                src, target_server_id, task.output_size
            )
            
            # 分配任务
            task.assigned_server = target_server_id
            effective_priority = 1 / max(transfer_time, 1e-6)
            
            # 尝试添加任务（即使服务器不可用也记录）
            if target_server.can_allocate(task):
                target_server.add_task(task, priority=effective_priority)
                status = "Success"
            else:
                status = "Failed"
            
            # self.logger.info(f"Task {task.task_id} assigned to Server {target_server_id} "
            #       f"({status}) - Transfer Time: {transfer_time:.2f}s)")