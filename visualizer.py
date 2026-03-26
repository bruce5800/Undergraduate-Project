import matplotlib.pyplot as plt

class TaskVisualizer:
    """任务可视化类"""
    
    def __init__(self, sim):
        self.sim = sim
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plt.style.use('ggplot')
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        
        # 记录数据用于绘图
        self.time_steps = []
        self.utilization = {s_id: [] for s_id in sim.servers}
        self.colors = plt.cm.tab10  # 颜色映射
        
    def update_plots(self, current_time):
        """更新绘图"""
        self.time_steps.append(current_time)
        
        # 更新资源利用率数据
        for s_id, server in self.sim.servers.items():
            util = server.used_compute / server.total_compute if server.total_compute > 0 else 0
            self.utilization[s_id].append(util)
        
        # 清空画布
        self.ax1.clear()
        self.ax2.clear()
        
        # 绘制资源利用率
        self.ax1.set_title(f'服务器资源利用率 (时间: {current_time}s)')
        for s_id, util in self.utilization.items():
            self.ax1.plot(self.time_steps, util, label=f'服务器 {s_id}', linewidth=2)
        self.ax1.legend(loc='upper right')
        self.ax1.set_ylim(0, 1.1)
        self.ax1.set_ylabel('利用率')
        self.ax1.grid(True, alpha=0.3)
        
        # 绘制任务调度甘特图
        self.ax2.set_title('任务调度甘特图')
        yticks = []
        ylabels = []
        
        # 为每个服务器绘制任务
        for i, (s_id, server) in enumerate(self.sim.servers.items()):
            yticks.append(i)
            ylabels.append(f'服务器 {s_id}')
            
            # 绘制该服务器上的所有任务（历史和当前）
            all_tasks = server.task_history + server.running_tasks
            for task in all_tasks:
                if task.start_time is not None and task.end_time is not None:
                    start = task.start_time
                    duration = task.end_time - task.start_time
                    if duration > 0:
                        # 使用任务ID确定颜色
                        color = self.colors(task.task_id % 10)
                        self.ax2.barh(i, duration, left=start, height=0.6, 
                                     color=color, edgecolor='black', alpha=0.8)
                        
                        # 在任务条上显示任务ID
                        text_x = start + duration / 2
                        self.ax2.text(text_x, i, f'T{task.task_id}', 
                                     ha='center', va='center', fontsize=8,
                                     color='white' if task.task_id % 2 == 0 else 'black')
        
        self.ax2.set_yticks(yticks)
        self.ax2.set_yticklabels(ylabels)
        self.ax2.set_xlabel('时间 (秒)')
        self.ax2.set_ylabel('服务器')
        self.ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()