# AIGC 云边协同推理调度仿真平台

面向 **AIGC（大模型推理）** 的云边协同调度仿真平台。在通用 DAG 调度基础上引入 AIGC 推理特有的物理约束 —— 模型权重驻留、KV cache、continuous batching、序列亲和性 —— 用于评估调度算法在真实 LLM 推理负载下的表现。

## AIGC 物理建模进度

| 物理特征 | 阶段 | 状态 |
|---|---|---|
| 模型权重常驻显存 + 冷加载 + LRU 驱逐 | M1 | ✅ |
| Prefill / Decode 两阶段任务 | M2 step 1 | ✅ |
| KV cache 跨机迁移代价（output_size → transfer_time） | M2 step 1 | ✅ |
| KV cache 占显存 | M2 step 3 | ✅ |
| Continuous batching（静态、admission-time） | M3 step 1 | ✅ |
| AIGC-aware RL 调度器（state + reward 升级） | M3 step 2 | ✅ |
| 9 项可关闭组件支持 ablation study | M3 step 3 | ✅ |
| TTFT / TPOT / goodput / SLO 等 AIGC 指标 | M4 | ⏳ |
| Azure LLM / BurstGPT 真实 trace | M4 | ⏳ |

## 项目结构

```
.
├── environment/                  # 仿真环境
│   ├── task.py                   # 任务模型（含 PREFILL/DECODE/GENERIC，KV cache）
│   ├── server.py                 # 服务器模型（含模型驻留、batching、冷加载）
│   ├── network.py                # 网络拓扑（延迟矩阵 + 带宽）
│   ├── simulation.py             # 仿真主循环
│   └── model_catalog.py          # AIGC 模型规格目录（llama-7b/13b/70b、sdxl）
├── scheduler/                    # 调度算法
│   ├── base.py
│   ├── RRscheduler.py            # Round-Robin（含 fallback）
│   ├── Heftscheduler.py          # HEFT（异构最早完成时间贪心）
│   ├── GAscheduler.py            # 遗传算法
│   ├── PSOscheduler.py           # 粒子群优化
│   └── RLscheduler.py            # PPO + 动作掩码 + GAE，AIGC-aware（M3 step 2）
├── tests/                        # 烟雾测试（每个 milestone 一组）
│   ├── test_m1_smoke.py
│   ├── test_m2_smoke.py
│   ├── test_m3_smoke.py
│   ├── test_m3_step2_smoke.py
│   └── test_m3_step3_smoke.py
├── demos/
│   └── aigc_demo.py              # generic vs AIGC 物理对比展示
├── figs/                         # benchmark 输出 + 论文图生成
│   ├── ploter.py                 # 论文级绘图脚本
│   └── report/                   # 已生成的图（PNG）
├── main.py                       # 单次仿真 + 实时可视化
├── brenchmark.py                 # 完整 benchmark 框架（多轮 + 统计检验）
└── visualizer.py                 # 实时可视化组件
```

## AIGC 模型目录

`environment/model_catalog.py::CATALOG` 内置 4 个真实模型，参数参考公开技术报告：

| model_id | 权重 (GB) | 冷加载 (s) | KV/token (MB) | max_batch | 可承载服务器 |
|---|---|---|---|---|---|
| llama-7b  | 14  | 5  | 0.5 | 32 | 边缘 64GB 及以上 |
| llama-13b | 26  | 9  | 0.8 | 16 | 边缘 64GB 及以上 |
| llama-70b (INT8) | 70 | 25 | 2.5 | 8 | **仅云**（128GB） |
| sdxl      | 12  | 4  | —   | —  | 边缘 64GB 及以上 |

## 调度算法

| 算法 | 类型 | AIGC 感知 |
|---|---|---|
| Round-Robin | 基线 | ❌ |
| HEFT | 贪心 | ❌ |
| GA | 元启发式 | ❌ |
| PSO | 元启发式 | ❌ |
| **RL (PPO)** | 学习型 | ✅ state 含 kind/model/loaded/batch_count/affinity；reward 含 warm/batch/affinity |

## 任务/负载类型

| Workload | 含义 | CLI |
|---|---|---|
| `dag` | 3 种通用 DAG（complex/linear/fork-join）混合，无模型概念 | `--workload dag`（默认） |
| `dag --aigc` | 上面 + Zipf 模型分配 → 触发 M1 物理 | `--workload dag --aigc` |
| `inference` | 真实 LLM 推理负载，每请求 = prefill→decode 二节点 DAG | `--workload inference` |

## 评估指标（M3 阶段）

| 指标 | 含义 |
|---|---|
| Makespan | 全部任务完成时长（秒） |
| Avg E2E Latency | 任务 READY → COMPLETED 的平均延迟 |
| Avg Utilization | 加权计算利用率 Σ(demand × duration) / (capacity × makespan) |
| Load Balance Std | 各服务器利用率的标准差 |

> M4 将替换为 AIGC 论文标准指标：TTFT P50/P95/P99、TPOT P50/P95、goodput (tokens/s)、SLO 达成率。

## 9 项消融开关

`--ablation NAME` 可单独关闭以下任一组件，用于 P1 消融实验：

| 名称 | 关闭后行为 | 类别 |
|---|---|---|
| `no_batching` | 退回 solo exec_time | 物理层 |
| `no_warm_reward` | RL 不奖励模型已加载 | RL reward |
| `no_batch_reward` | RL 不奖励加入 batch | RL reward |
| `no_affinity_reward` | RL 不奖励 decode 留在 prefill 那台 | RL reward |
| `no_aigc_state` | RL state 回到 M0 维度（仅基础特征） | RL state |
| `no_action_mask` | PPO policy 不掩码非法动作 | PPO 内部 |
| `no_gae` | 用 MC return 替代 GAE | PPO 内部 |
| `no_pretrain` | 跳过预训练 | PPO 内部 |
| `no_entropy` | entropy_coeff = 0 | PPO 内部 |

## 环境依赖

- Python 3.10+
- PyTorch
- NumPy, SciPy, Pandas
- Matplotlib

```bash
pip install torch numpy scipy pandas matplotlib
```

## 快速开始

### 单次仿真 + 实时可视化

```bash
python main.py
```

### Benchmark（多轮统计 + 显著性检验）

```bash
# 默认：通用 DAG，3/5/7 边缘服务器，20 轮 × 300 任务（约 30 分钟）
python brenchmark.py --out figs/baseline

# 快速验证（5 轮 × 2 边缘配置）
python brenchmark.py --quick --out figs/quick_check

# M1：通用 DAG + Zipf 模型分配
python brenchmark.py --aigc --out figs/m1_aigc

# M2：真·LLM 推理负载
python brenchmark.py --workload inference --out figs/m2_inference

# M3 消融实验（每行一组）
python brenchmark.py --workload inference --ablation none              --out figs/abl_full
python brenchmark.py --workload inference --ablation no_batching       --out figs/abl_no_batching
python brenchmark.py --workload inference --ablation no_warm_reward    --out figs/abl_no_warm
python brenchmark.py --workload inference --ablation no_batch_reward   --out figs/abl_no_batch
python brenchmark.py --workload inference --ablation no_affinity_reward --out figs/abl_no_affinity
python brenchmark.py --workload inference --ablation no_aigc_state     --out figs/abl_no_state
python brenchmark.py --workload inference --ablation no_action_mask    --out figs/abl_no_mask
python brenchmark.py --workload inference --ablation no_gae            --out figs/abl_no_gae
python brenchmark.py --workload inference --ablation no_pretrain       --out figs/abl_no_pretrain
python brenchmark.py --workload inference --ablation no_entropy        --out figs/abl_no_entropy
```

输出落到 `--out` 指定目录：

- `benchmark_raw.csv` — 每轮每检查点的原始指标
- `benchmark_summary.csv` — 均值/标准差/95% CI/中位数
- `statistical_tests.csv` — 成对 Mann-Whitney U 检验
- `run_manifest.json` — 时间戳、git commit、CLI 参数（可复现配置）

> 默认输出目录是 `figs/`，已存在 benchmark CSV 时**拒绝覆盖**，避免误删历史数据。要覆盖加 `--force`。

### 测试

```bash
# 单个
python tests/test_m1_smoke.py

# 全部一起
for t in tests/test_*.py; do python "$t"; done
```

每个测试覆盖一个 milestone：

| 文件 | 覆盖范围 | 用例数 |
|---|---|---|
| `test_m1_smoke.py` | 向后兼容、冷加载、暖启动、LRU 驱逐、pinned 不可驱逐 | 5 |
| `test_m2_smoke.py` | Task 拆分、KV cache 编码、显存占用、长 prompt 路由 | 11 |
| `test_m3_smoke.py` | Continuous batching、max_batch、吞吐速比 5.9× | 7 |
| `test_m3_step2_smoke.py` | RL state shape、kind/model 编码、3 项 AIGC 奖励 | 8 |
| `test_m3_step3_smoke.py` | 9 项 ablation 开关各自起效 | 9 |

### 演示脚本

```bash
python demos/aigc_demo.py
```

对比 generic vs AIGC 模式在同一 DAG 上的 makespan 与冷加载事件，用于论文 Figure 1。

## 实验复现（论文级）

```bash
python figs/ploter.py     # 在 figs/report/ 下生成论文图
```

生成：折线图（带 95% CI 阴影）、箱线图（带显著性标注）、复合面板、可扩展性柱状图。

## TODO（M4 路线）

### P0 — 仍待补的论文核心要素

- [ ] **AIGC 评估指标** — 用 TTFT P50/P95/P99 + TPOT P50/P95 + goodput + SLO 达成率替换/增补 makespan
- [ ] **真实 trace 接入** — 引入 Azure LLM Inference Trace 或 BurstGPT，替换 uniform random 的合成负载
- [ ] **完整 benchmark + ablation 跑一遍** — 跑出论文要用的所有数据

### P1 — 强烈建议

- [ ] **收敛性分析** — 记录并绘制 RL 训练曲线（reward/loss vs episode）、GA/PSO fitness vs generation
- [ ] **调度时间开销对比** — 测量各算法单次 `schedule()` 调用墙钟时间，证明在线可用性

### P2 — 锦上添花

- [ ] **能耗模型** — 基于服务器类型与负载估算能耗，引入能效比（性能/瓦特）
- [ ] **大规模实验** — 1000+ 任务、15+ 服务器，验证可扩展性

## 一句话向他人介绍这个项目

> 一个面向 AIGC 推理的云边协同调度仿真平台，建模了**模型权重常驻、冷加载、KV cache、continuous batching、序列亲和性**五类 AIGC 物理约束，配套 5 种调度算法（含 AIGC-aware PPO）与 9 项可关闭组件的消融实验框架，用于评估调度策略在 LLM 推理场景下的吞吐与延迟。
