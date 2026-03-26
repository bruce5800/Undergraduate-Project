# AIGC 云边协同推理调度优化仿真平台

面向 AIGC（AI Generated Content）任务的云边协同推理场景，对多种调度算法进行仿真对比的实验平台。项目模拟了一个包含云服务器和多台异构边缘服务器的分布式计算环境，支持 DAG 任务依赖、网络传输延迟、资源约束等特性。

## 项目结构

```
├── environment/              # 仿真环境
│   ├── task.py               # 任务模型（DAG 依赖、计算需求、数据大小）
│   ├── server.py             # 服务器模型（云/边缘、资源管理、任务队列）
│   ├── network.py            # 网络模型（延迟矩阵、带宽、传输时间估算）
│   └── simulation.py         # 仿真引擎（环境初始化、时间步推进）
├── scheduler/                # 调度算法
│   ├── base.py               # 调度器抽象基类
│   ├── RRscheduler.py        # Round-Robin 轮询调度
│   ├── Heftscheduler.py      # HEFT 异构最早完成时间贪心调度
│   ├── GAscheduler.py        # 遗传算法（GA）调度
│   ├── PSOscheduler.py       # 粒子群优化（PSO）调度
│   └── RLscheduler.py        # 强化学习（Actor-Critic）调度
├── main.py                   # 单次仿真运行入口（带实时可视化）
├── brenchmark.py             # 多算法基准测试（批量对比 + CSV 导出）
├── visualizer.py             # 实时可视化（资源利用率 + 甘特图）
├── figs/                     # 测试结果与绘图
│   ├── benchmark_results.csv
│   └── ploter.py
└── plot/                     # 额外可视化脚本
```

## 调度算法

| 算法 | 类型 | 核心思路 |
|------|------|----------|
| **Round-Robin** | 基线 | 按固定顺序轮询分配，资源不足时回退到最空闲服务器 |
| **HEFT** | 贪心 | 计算向上排名（upward rank）确定优先级，选择使 EFT 最小的服务器 |
| **GA** | 元启发式 | 遗传算法搜索最优任务-服务器映射，支持自适应参数和精英保留 |
| **PSO** | 元启发式 | 粒子群优化搜索，多目标适应度（makespan + 能耗 + 成本 + 负载均衡） |
| **RL** | 学习型 | Actor-Critic 强化学习，在线学习调度策略 |

## 任务模型

支持三种 DAG 结构的任务生成：
- **复杂 DAG**：15 任务模板，多层并行 + 汇聚结构
- **链式 DAG**：10 任务串行依赖链
- **Fork-Join DAG**：9 任务 Fork-Join 并行结构

每种结构支持任意数量任务，通过并联独立子图实现扩展。

## 环境依赖

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib

```bash
pip install torch numpy matplotlib
```

## 快速开始

### 单次仿真（带实时可视化）

```bash
python main.py
```

运行后会弹出 Matplotlib 窗口，实时显示服务器资源利用率和任务调度甘特图。

### 基准测试

```bash
python brenchmark.py
```

默认配置：
- 边缘服务器数量：3 / 5 / 7
- 任务总数：300
- 检查点间隔：每 20 个任务记录一次

测试结果输出到 `figs/benchmark_results.csv`，包含以下指标：
- **Makespan**：总运行时间
- **平均端到端延迟**：任务从就绪到完成的平均耗时
- **平均利用率**：服务器计算资源平均利用率
- **负载均衡标准差**：各服务器利用率的标准差

## 自定义配置

### 修改服务器配置

编辑 `environment/simulation.py` 中的 `setup_servers()` 方法，调整云服务器和边缘服务器的计算能力、内存、存储和带宽参数。

### 添加新调度算法

1. 继承 `scheduler/base.py` 中的 `BaseScheduler`
2. 实现 `schedule()` 方法
3. 在 `brenchmark.py` 的 `self.schedulers` 字典中注册
