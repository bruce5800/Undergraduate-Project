from abc import ABC, abstractmethod


class BaseScheduler(ABC):
    """调度器抽象基类，所有调度器必须实现 schedule() 方法"""

    def __init__(self, sim_env):
        self.sim = sim_env

    @abstractmethod
    def schedule(self):
        """执行一轮调度，将 READY 任务分配到服务器队列"""
        pass
