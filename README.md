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
│   └── RLscheduler.py        # 强化学习（PPO + 动作掩码 + GAE）调度
├── main.py                   # 单次仿真运行入口（带实时可视化）
├── brenchmark.py             # 基准测试（多轮运行 + 统计检验 + CSV 导出）
├── visualizer.py             # 实时可视化（资源利用率 + 甘特图）
├── figs/                     # 测试结果与绘图
│   ├── benchmark_raw.csv     # 每轮原始数据
│   ├── benchmark_summary.csv # 均值/标准差/95% CI
│   ├── statistical_tests.csv # Mann-Whitney U 检验
│   ├── ploter.py             # 论文级绘图（折线+CI、箱线图、复合面板）
│   └── report/               # 生成的图表
└── plot/                     # 额外可视化脚本
```

## 调度算法

| 算法 | 类型 | 核心思路 |
|------|------|----------|
| **Round-Robin** | 基线 | 按固定顺序轮询分配，资源不足时回退到最空闲服务器 |
| **HEFT** | 贪心 | 计算向上排名（upward rank）确定优先级，选择使 EFT 最小的服务器 |
| **GA** | 元启发式 | 遗传算法搜索最优任务-服务器映射，支持自适应参数和精英保留 |
| **PSO** | 元启发式 | 粒子群优化搜索，多目标适应度（makespan + 能耗 + 成本 + 负载均衡） |
| **RL (PPO)** | 学习型 | PPO + 动作掩码 + GAE + 多轮预训练，多目标奖励函数 |

## 任务模型

支持三种 DAG 结构的任务生成：
- **复杂 DAG**：15 任务模板，多层并行 + 汇聚结构
- **链式 DAG**：10 任务串行依赖链
- **Fork-Join DAG**：9 任务 Fork-Join 并行结构

每种结构支持任意数量任务，通过并联独立子图实现扩展。

## 环境依赖

- Python 3.10+
- PyTorch
- NumPy, SciPy, Pandas
- Matplotlib

```bash
pip install torch numpy scipy pandas matplotlib
```

## 快速开始

### 单次仿真（带实时可视化）

```bash
python main.py
```

运行后会弹出 Matplotlib 窗口，实时显示服务器资源利用率和任务调度甘特图。

### 基准测试

```bash
# 完整测试（20 轮 × 5 调度器 × 3 边缘配置，约 30 分钟）
python brenchmark.py

# 快速验证（5 轮、2 种边缘配置，约 3 分钟）
python brenchmark.py --quick

# 自定义参数
python brenchmark.py --runs 10 --tasks 500 --interval 50 --edge 3 5
```

默认配置：
- 边缘服务器数量：3 / 5 / 7
- 任务总数：300（单次运行，每 20 个任务记录一次检查点）
- 每配置运行 20 轮，不同随机种子

测试输出：
- `figs/benchmark_raw.csv` — 每轮每检查点的原始指标
- `figs/benchmark_summary.csv` — 均值 ± 标准差、95% CI、中位数
- `figs/statistical_tests.csv` — 成对 Mann-Whitney U 检验

### 生成图表

```bash
python figs/ploter.py
```

在 `figs/report/` 下生成论文级图表：折线图（带 95% CI 阴影）、箱线图（带显著性标注）、复合面板、可扩展性柱状图。

## 评估指标

| 指标 | 说明 |
|------|------|
| **Makespan** | 总运行时间（秒） |
| **Avg E2E Latency** | 任务从 READY 到 COMPLETED 的平均端到端延迟 |
| **Avg Utilization** | 加权计算资源利用率，Σ(demand × duration) / (capacity × makespan)，∈ [0, 1] |
| **Load Balance Std** | 各服务器利用率的标准差，越低越均衡 |

## 自定义配置

### 修改服务器配置

编辑 `environment/simulation.py` 中的 `setup_servers()` 方法，调整云服务器和边缘服务器的计算能力、内存、存储和带宽参数。

### 添加新调度算法

1. 继承 `scheduler/base.py` 中的 `BaseScheduler`
2. 实现 `schedule()` 方法
3. 在 `brenchmark.py` 的 `self.schedulers` 字典中注册

---

## TODO

### P1：强烈建议（大幅提升论文质量）

- [ ] **消融实验** — 对 RL 调度器各组件（动作掩码、GAE、预训练、熵正则化）逐一禁用，证明每个设计的贡献
- [ ] **收敛性分析** — 记录并绘制 RL 训练曲线（reward/loss vs episode）、GA/PSO 收敛过程（fitness vs generation）
- [ ] **调度时间开销对比** — 测量各算法单次 `schedule()` 调用的墙钟时间，证明在线可用性
- [ ] **在线任务到达模型** — 实现 Poisson 到达模型替代当前的静态任务集，更贴近真实推理服务场景

### P2：锦上添花

- [ ] **能耗模型** — 基于服务器类型和负载建模能耗，加入能效比（性能/瓦特）指标
- [ ] **更丰富的可视化** — CDF 分布曲线、多调度器甘特图对比、雷达图综合评分
- [ ] **大规模实验** — 1000+ 任务、15+ 服务器，验证算法在大规模场景下的表现和可扩展性
