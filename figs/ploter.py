# visualization/plotter.py
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional

class BenchmarkPlotter:
    """基准测试结果可视化类"""
    
    def __init__(self, csv_file: str = "figs/benchmark_results.csv"):
        """
        初始化可视化器
        
        Args:
            csv_file: CSV结果文件路径
        """
        self.csv_file = csv_file
        self.data = None
        
        # 颜色和标记样式
        self.colors = {
            "RoundRobin": "#1f77b4",
            "RL": "#ff7f0e", 
            "GA": "#2ca02c",
            "PSO": "#d62728",
            "HEFT": "#9467bd"
        }
        
        self.markers = {
            "RoundRobin": "o",
            "RL": "s",
            "GA": "^",
            "PSO": "D"
        }
        
        self.line_styles = {
            "RoundRobin": "-",
            "RL": "--",
            "GA": "-.",
            "PSO": ":"
        }
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_data(self) -> pd.DataFrame:
        """从CSV文件加载数据"""
        try:
            self.data = pd.read_csv(self.csv_file)
            print(f"成功加载数据，共 {len(self.data)} 行")
            return self.data
        except FileNotFoundError:
            print(f"错误: 找不到文件 {self.csv_file}")
            return None
    
    def plot_multi_metrics_comparison(self, metric: str = 'avg_completion_time', 
                                     output_file: str = "figs/multi_metrics_comparison.png"):
        """
        绘制指定指标的对比图
        
        Args:
            metric: 指标名称 ('总运行时间', '平均端到端延迟', '平均利用率', '负载均衡标准差')
            output_file: 输出文件路径
        """
        if self.data is None:
            self.load_data()
            if self.data is None:
                return
        
        # 指标名称映射
        metric_labels = {
            '总运行时间(makespan)': '总运行时间 (秒)',
            '平均端到端延迟': '平均端到端延迟 (秒)',
            '平均利用率': '资源利用率 (%)',
            '负载均衡标准差': '负载均衡标准差'
        }
        
        y_label = metric_labels.get(metric, metric)
        
        # 获取唯一的边缘服务器数量
        edge_counts = sorted(self.data['边缘服务器数量'].unique())
        schedulers = sorted(self.data['调度器'].unique())
        
        # 创建图形
        fig, axes = plt.subplots(1, len(edge_counts), figsize=(15, 5))
        
        if len(edge_counts) == 1:
            axes = [axes]
        
        # 对每种边缘服务器数量绘制一张图
        for idx, edge_count in enumerate(edge_counts):
            ax = axes[idx]
            
            # 为每个调度器收集数据
            for scheduler in schedulers:
                # 筛选数据
                mask = (self.data['边缘服务器数量'] == edge_count) & (self.data['调度器'] == scheduler)
                scheduler_data = self.data[mask].sort_values('任务数量')
                
                if len(scheduler_data) == 0:
                    continue
                
                # 获取数据点
                task_sizes = scheduler_data['任务数量'].values
                metric_values = scheduler_data[metric].values
                
                # 绘制折线
                ax.plot(
                    task_sizes, metric_values,
                    color=self.colors.get(scheduler, '#000000'),
                    marker=self.markers.get(scheduler, 'o'),
                    linestyle=self.line_styles.get(scheduler, '-'),
                    linewidth=2,
                    markersize=8,
                    label=scheduler
                )
            
            # 设置图表属性
            ax.set_title(f'边缘服务器数量: {edge_count}', fontsize=14, fontweight='bold')
            ax.set_xlabel('任务数量', fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='upper left')
            
            # 添加网格
            ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle(f'不同边缘服务器数量下的调度性能比较（{metric}）', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n图表已保存到: {output_file}")
    
    def plot_bar_comparison(self, edge_count: int = 5, task_size: int = 45,
                           output_file: str = "figs/bar_comparison.png"):
        """
        绘制柱状图比较不同调度器在特定配置下的表现
        
        Args:
            edge_count: 边缘服务器数量
            task_size: 任务数量
            output_file: 输出文件路径
        """
        if self.data is None:
            self.load_data()
            if self.data is None:
                return
        
        # 筛选特定配置的数据
        mask = (self.data['边缘服务器数量'] == edge_count) & (self.data['任务数量'] == task_size)
        subset_data = self.data[mask]
        
        if len(subset_data) == 0:
            print(f"没有找到边缘服务器={edge_count}, 任务数量={task_size}的数据")
            return
        
        schedulers = subset_data['调度器'].values
        metrics = ['总运行时间(makespan)', '平均端到端延迟', '平均利用率', '负载均衡标准差']
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # 为每个指标绘制柱状图
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # 获取每个调度器的指标值
            values = []
            for scheduler in schedulers:
                mask_scheduler = subset_data['调度器'] == scheduler
                value = subset_data[mask_scheduler][metric].values[0]
                values.append(value)
            
            # 创建柱状图
            x_pos = np.arange(len(schedulers))
            bars = ax.bar(x_pos, values, color=[self.colors.get(s, '#666666') for s in schedulers])
            
            # 设置图表属性
            ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
            ax.set_xlabel('调度器', fontsize=12)
            ax.set_ylabel(self._get_metric_unit(metric), fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(schedulers, rotation=45)
            
            # 在柱子上添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=10)
            
            # 添加网格
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.suptitle(f'调度器性能比较 (边缘服务器={edge_count}, 任务数={task_size})', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n图表已保存到: {output_file}")
    
    def _get_metric_unit(self, metric: str) -> str:
        """获取指标的单位"""
        units = {
            '总运行时间': '秒',
            '平均端到端延迟': '秒',
            '平均利用率': '%',
            '负载均衡标准差': '标准差'
        }
        return units.get(metric, '')
    
    def plot_heatmap(self, scheduler: str = "RL", metric: str = "总运行时间(makespan)",
                    output_file: str = "figs/heatmap_comparison.png"):
        """
        绘制热力图展示不同配置下的性能
        
        Args:
            scheduler: 调度器名称
            metric: 指标名称
            output_file: 输出文件路径
        """
        if self.data is None:
            self.load_data()
            if self.data is None:
                return
        
        # 筛选特定调度器的数据
        mask = self.data['调度器'] == scheduler
        scheduler_data = self.data[mask]
        
        if len(scheduler_data) == 0:
            print(f"没有找到调度器 '{scheduler}' 的数据")
            return
        
        # 获取唯一的边缘服务器数量和任务数量
        edge_counts = sorted(scheduler_data['边缘服务器数量'].unique())
        task_sizes = sorted(scheduler_data['任务数量'].unique())
        
        # 创建数据矩阵
        data_matrix = np.zeros((len(edge_counts), len(task_sizes)))
        
        # 填充数据矩阵
        for i, edge_count in enumerate(edge_counts):
            for j, task_size in enumerate(task_sizes):
                mask_cell = (scheduler_data['边缘服务器数量'] == edge_count) & \
                           (scheduler_data['任务数量'] == task_size)
                
                if scheduler_data[mask_cell][metric].any():
                    data_matrix[i, j] = scheduler_data[mask_cell][metric].values[0]
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 使用imshow绘制热力图
        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
        
        # 设置坐标轴标签
        ax.set_xticks(np.arange(len(task_sizes)))
        ax.set_yticks(np.arange(len(edge_counts)))
        ax.set_xticklabels(task_sizes)
        ax.set_yticklabels(edge_counts)
        
        # 在每个格子中显示数值
        for i in range(len(edge_counts)):
            for j in range(len(task_sizes)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.1f}',
                              ha="center", va="center", 
                              color="black" if data_matrix[i, j] < np.max(data_matrix) * 0.7 else "white")
        
        # 设置标题和标签
        ax.set_title(f'{scheduler}调度器 - {metric} 热力图', fontsize=16, fontweight='bold')
        ax.set_xlabel('任务数量', fontsize=12)
        ax.set_ylabel('边缘服务器数量', fontsize=12)
        
        # 添加颜色条
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(self._get_metric_unit(metric), rotation=-90, va="bottom")
        
        plt.tight_layout()
        
        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n热力图已保存到: {output_file}")
    
    def plot_all_metrics_for_scheduler(self, scheduler: str, edge_count: int = 5,
                                      output_file: str = "figs/scheduler_comprehensive.png"):
        """
        为特定调度器绘制所有指标的综合性图表
        
        Args:
            scheduler: 调度器名称
            edge_count: 边缘服务器数量
            output_file: 输出文件路径
        """
        if self.data is None:
            self.load_data()
            if self.data is None:
                return
        
        # 筛选特定调度器和边缘服务器数量的数据
        mask = (self.data['调度器'] == scheduler) & (self.data['边缘服务器数量'] == edge_count)
        scheduler_data = self.data[mask].sort_values('任务数量')
        
        if len(scheduler_data) == 0:
            print(f"没有找到调度器 '{scheduler}' 在边缘服务器数量 {edge_count} 下的数据")
            return
        
        # 获取任务数量列表
        task_sizes = scheduler_data['任务数量'].values
        
        # 创建图形（2x2的子图）
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # 定义要绘制的指标
        metrics = [
            ('总运行时间(makespan)', '总运行时间 (秒)'),
            ('平均端到端延迟', '平均端到端延迟 (秒)'),
            ('平均利用率', '资源利用率 (%)'),
            ('负载均衡标准差', '负载均衡标准差')
        ]
        
        # 为每个指标绘制折线图
        for idx, (metric_col, metric_label) in enumerate(metrics):
            ax = axes[idx]
            
            # 获取指标值
            metric_values = scheduler_data[metric_col].values
            
            # 绘制折线图
            ax.plot(task_sizes, metric_values, 
                   color=self.colors.get(scheduler, '#000000'),
                   marker=self.markers.get(scheduler, 'o'),
                   linewidth=2, markersize=8)
            
            # 设置图表属性
            ax.set_title(f'{metric_label}', fontsize=14, fontweight='bold')
            ax.set_xlabel('任务数量', fontsize=12)
            ax.set_ylabel(metric_label, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 填充区域
            ax.fill_between(task_sizes, 0, metric_values, alpha=0.2, 
                           color=self.colors.get(scheduler, '#000000'))
        
        plt.suptitle(f'{scheduler}调度器性能分析 (边缘服务器数量={edge_count})', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n综合性图表已保存到: {output_file}")
    
    def generate_report(self, output_dir: str = "figs/report"):
        """
        生成完整的可视化报告
        
        Args:
            output_dir: 输出目录
        """
        if self.data is None:
            self.load_data()
            if self.data is None:
                return
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("开始生成可视化报告...")

        # 1. 各指标对比图
        metrics = ['总运行时间(makespan)', '平均端到端延迟', '平均利用率', '负载均衡标准差']
        for metric in metrics:
            filename = metric.replace('/', '_').replace('\\', '_')
            self.plot_multi_metrics_comparison(
                metric, 
                os.path.join(output_dir, f"{filename}_comparison.png")
            )
        
        # 2. 为每个调度器生成综合性图表
        edge_counts = sorted(self.data['边缘服务器数量'].unique())
        for scheduler in self.data['调度器'].unique():
            for edge_count in edge_counts[:2]:  # 只生成前两个边缘服务器数量的图表
                self.plot_all_metrics_for_scheduler(
                    scheduler, edge_count,
                    os.path.join(output_dir, f"{scheduler}_edge{edge_count}_comprehensive.png")
                )
        
        # 3. 热力图
        for scheduler in self.data['调度器'].unique():
            self.plot_heatmap(
                scheduler, '总运行时间(makespan)',
                os.path.join(output_dir, f"{scheduler}_heatmap.png")
            )
        
        print(f"\n所有图表已保存到: {output_dir}")

if __name__ == "__main__":
    plotter = BenchmarkPlotter()
    plotter.generate_report()