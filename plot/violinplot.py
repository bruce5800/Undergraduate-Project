import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 假设你有四种算法的数据
# 每种算法在 Task Scale = 90 下的 CPU 负载分布数据，数据以numpy数组形式呈现
# 这里用随机数模拟数据，你应该替换成你实验中的实际数据
np.random.seed(42)
task_scale = 90
num_servers = 5

# 模拟数据：每个算法在 5 个服务器上的负载分布
round_robin_load = np.random.uniform(0.05, 0.15, size=(num_servers, task_scale))
ga_load = np.random.uniform(0.05, 0.20, size=(num_servers, task_scale))
pso_load = np.random.uniform(0.05, 0.25, size=(num_servers, task_scale))
rl_load = np.random.uniform(0.10, 0.35, size=(num_servers, task_scale))

# 将这些数据合并成一个字典，并指明算法标签
data = {
    'Round-Robin': round_robin_load.flatten(),
    'GA': ga_load.flatten(),
    'PSO': pso_load.flatten(),
    'RL': rl_load.flatten()
}

# 创建小提琴图
plt.figure(figsize=(10, 6))
sns.violinplot(data=data, inner="quart", palette="Set2")

# 设置图标题和标签
plt.title(f'Violin Plots of CPU Load Distribution (Task Scale = {task_scale})', fontsize=16)
plt.xlabel('Scheduling Algorithm', fontsize=14)
plt.ylabel('CPU Load', fontsize=14)

# 显示图像
plt.savefig(f"figs/violin_plot_task_scale_{task_scale}.png")
plt.show()