from typing import Dict

class Network:
    def __init__(self):
        # 节点间传输延迟 (ms)
        self.latency_matrix: Dict[(int, int), float] = {}  
        # 节点间可用带宽 (Mbps)
        self.bandwidth_matrix: Dict[(int, int), float] = {}
        # 正在传输中的任务数据
        self.active_transfers = []  
        
    def add_link(self, node1: int, node2: int, 
                 latency: float, bandwidth: float):
        """添加网络连接"""
        self.latency_matrix[(node1, node2)] = latency
        self.latency_matrix[(node2, node1)] = latency
        self.bandwidth_matrix[(node1, node2)] = bandwidth
        self.bandwidth_matrix[(node2, node1)] = bandwidth
        
    def estimate_transfer_time(self, src: int, dst: int, data_size: float) -> float:
        """估算数据传输时间"""
        if src == dst:
            return 0.0
        latency = self.latency_matrix.get((src, dst), 100)  # 默认100ms
        bandwidth = self.bandwidth_matrix.get((src, dst), 10)  # 默认10Mbps
        transfer_time = data_size * 1024 / (bandwidth / 8)  # 转换为MB，计算秒数
        return latency / 1000 + transfer_time  
