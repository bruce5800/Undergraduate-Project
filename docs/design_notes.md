# 项目设计决策记录（Design Notes）

> 论文写作的内部参考。每条决策按「**做了什么 / 为什么 / 替代方案 / 代码位置**」组织。
> 数字均与源码一致（如不一致以源码为准）。

---

## 0. 项目一句话

面向 AIGC 云边协同推理的调度仿真平台：在通用 DAG 调度基础上引入 **模型驻留 / KV cache / continuous batching / 内存带宽墙 / 序列亲和性** 5 类物理约束，配套 5 种调度器（含 AIGC-aware PPO）与 9 项消融组件。

## 1. 整体设计哲学

### 1.1 为什么 AIGC 调度本质不同于通用 DAG 调度

| AIGC 物理 | 调度上的新约束 | 通用 DAG 仿真器是否能表达 |
|---|---|---|
| 模型权重常驻显存（GB 级），冷加载几秒到几十秒 | **模型亲和性**：去已加载该模型的服务器零代价 | ❌ |
| Prefill 计算密集 / Decode 显存密集，且 decode 共享 KV cache | **序列亲和性**：decode 必须在 prefill 服务器，否则迁移 KV | ❌ |
| Continuous batching 同模型请求拼 batch 吞吐 sub-linear 提升 | **请求合并**：同模同时段请求聚到一台机器更优 | ❌ |
| HBM 带宽墙：decode 即使算力无限也要 N ms/token | **memory-bound 决策**：增配大算力对 decode 无用 | ❌ |
| 用户感知是 TTFT/TPOT 长尾，不是平均 | **多 SLO 目标**：长尾延迟比平均更关键 | ❌ |

这 5 条就是论文 Introduction 里"为什么需要 AIGC-aware scheduling"那一节的核心论证。

### 1.2 论文 contribution 的 3 个支柱

1. **物理建模**：把上述 5 类物理建进开源仿真平台
2. **AIGC-aware RL 调度器**：state 暴露上述物理 + reward 对应激励
3. **9 项 ablation 验证**：每个组件单独可关，证明各自贡献

---

## 2. 各 milestone 关键决策

### M1：模型权重驻留物理

#### 2.1.1 引入 `ModelSpec` catalog

- **做了什么**：新建 `environment/model_catalog.py`，定义 4 个模型（llama-7b/13b/70b INT8/sdxl）
- **为什么**：通用 DAG 任务只有 `workload`，无法表达"这个任务需要 14GB 权重在 GPU 里"
- **替代方案**：直接给 task 加 `model_weight_GB` 字段——拒绝，因为多任务共享同模型时无法表达"共享"
- **代码位置**：`environment/model_catalog.py::ModelSpec`、`CATALOG`

#### 2.1.2 llama-70b 用 INT8 量化（70 GB）而非 FP16（140 GB）

- **做了什么**：catalog 里 llama-70b 权重 = 70 GB
- **为什么**：当前云服务器 128 GB 显存装不下 FP16 140GB；INT8 是主流生产部署，70 GB 还能体现"仅云端可承载"的物理约束（边缘最大 64GB）
- **替代方案**：把云显存调大到 256GB——拒绝，会破坏 baseline 数据
- **论文写作提示**：在 4.x 节"实验设置"里说明"我们采用 INT8 量化的 LLaMA-70B (≈70GB)，这是大多数生产部署的选择 [引用]"

#### 2.1.3 LRU 驱逐 + ref count pin

- **做了什么**：unpinned 模型按 last_use_time 排序，最旧的先驱逐；正在被 running 任务使用的模型 ref_count > 0，不可驱逐
- **为什么**：实际 GPU runtime（vLLM、TensorRT）都是 LRU；ref_count 保证不会驱逐"正在用的"模型导致仿真崩溃
- **替代方案**：LFU、FIFO——LRU 更标准，无理由换
- **代码位置**：`environment/server.py::_evict_lru_for`、`model_refs`

#### 2.1.4 三租户显存模型

```
total_memory = weight_vram_used + used_memory + free
weight_vram_used : 所有已加载模型的权重（含 pinned 与 unpinned）
used_memory      : running 任务的激活值 + KV cache
free             : 剩余可分配
```

- **做了什么**：Server 内部把显存分三个会计桶
- **为什么**：模型权重是"粘性"（驱逐才走），激活值是"瞬态"（任务结束就走），分桶才能正确做 LRU
- **代码位置**：`environment/server.py::Server.__init__`

---

### M2：两阶段任务模型 + KV cache

#### 2.2.1 TaskKind 三态

- **做了什么**：`enum TaskKind {GENERIC, PREFILL, DECODE}`，default GENERIC 保持向后兼容
- **为什么**：scheduler 在 state 编码、reward 计算时需要区分阶段（prefill 与 decode 物理特征完全不同）
- **代码位置**：`environment/task.py::TaskKind`

#### 2.2.2 关键设计：KV cache 编码到 `prefill.output_size`

- **做了什么**：`prefill.output_size = prompt_tokens × kv_cache_MB_per_token / 1024` GB
- **为什么**：仿真器原有 `transfer_time = output_size / bandwidth` 公式 **不用改一行代码**，自动把"decode 调到不同 server 要付 KV 迁移代价"算进去
- **替代方案**：新增 `KVCache` 类 + 显式迁移协议——拒绝，复杂度高 10 倍
- **论文写作提示**：这是技术报告里值得讲的 **engineering insight**，在"系统设计"小节用 1 段文字 + 公式即可
- **代码位置**：`environment/task.py::generate_inference_request`

#### 2.2.3 KV cache 同时占显存（M2 step 3）

- **做了什么**：Task 加 `kv_cache_GB` 字段，prefill 构建期间 + decode 使用期间均加到 `used_memory`
- **为什么**：之前只把 KV cache 算作传输 payload，但实际 GPU 显存也被它持续占着。不算就会导致"小边缘装下了不该装的请求"
- **代码位置**：`environment/server.py::_activation_footprint`

#### 2.2.4 prefill / decode workload 公式

```python
prefill_workload (TFLOPS) = prompt_tokens / 1000 × prefill_tflops_per_ktoken
decode_workload  (TFLOPS) = output_tokens / 1000 × decode_tflops_per_ktoken
```

- **做了什么**：workload 按 token 数线性计算
- **来源**：标准 LLM 推理量级估算 `2 × params × tokens`（FFN 是 2 × params 的 FLOPs/token，attention 在 prompt 大时占主导）
- **注意**：M2 step 1 单纯按这个算 → TPOT < 1ms 不合理 → 需要 M4 step 2 的 floor 修正

---

### M3：Continuous batching + AIGC-aware RL

#### 2.3.1 静态 batching（admission-time）而非动态 continuous batching

- **做了什么**：任务被接纳的瞬间，统计当前同模同阶段 running 数作为 batch_size，按 `T_batch = T_solo × (1 + (B-1) × overhead)` 算执行时长。已运行任务的 end_time 不变
- **为什么**：vLLM 真实做法是每个 iteration 重新计算 batch 组成、动态调整 end_time，仿真复杂度 10× 增加，对论文 contribution 影响小
- **审稿人会问**："这是 continuous batching 吗？" 答："是其简化模型；保留了 batch size 决定吞吐的关键非线性，但不模拟动态加入/退出"
- **代码位置**：`environment/server.py::process_tasks`、`_current_batch_size`

#### 2.3.2 Batching overhead 系数

| 阶段 | 每多 1 个并发请求执行时长增加 | 来源 |
|---|---|---|
| Prefill | +30% | 计算密集，并行收益低 |
| Decode  | +5%  | 内存密集，并行几乎免费 |

- **校准依据**：N=8 同模 decode 验证吞吐 5.93×，与 vLLM benchmark 5-6× 一致
- **代码位置**：`environment/server.py::PREFILL_BATCH_OVERHEAD`、`DECODE_BATCH_OVERHEAD`

#### 2.3.3 RL state 升级

| Block | 维度 | 字段 |
|---|---|---|
| 任务基础 | 6 | compute_demand, workload, input_size, output_size, deps, succs |
| 任务 AIGC（M3 加） | 4+\|M\| | kind one-hot (3), kv_cache_GB, model one-hot |
| 全局 | 2 | ready_count, completed_ratio |
| 服务器基础 | 5 | compute_util, mem_util, bandwidth, queue_len, transfer_time |
| 服务器 AIGC（M3 加） | 5 | is_my_model_loaded, batch_count/max, vram_free, cold_load_cost, **is_sibling_server** |

- **总维度**：`(10 + |M|) + 2 + 10 × num_servers`，\|M\|=4, ns=8 → 96 维
- **关键创新点**：`is_sibling_server` 让 RL 看到"我的 prefill 的 KV 在哪台"，明确激励序列亲和性
- **代码位置**：`scheduler/RLscheduler.py::StateEncoder.encode`

#### 2.3.4 Reward 权重设计

```
reward = 0.30 × time_reward     # 原有
       + 0.15 × balance_reward  # 原有
       + 0.10 × match_reward    # 原有
       + 0.15 × warm_bonus      # 新：避免冷加载
       + 0.20 × batch_bonus     # 新：参与同模 batch 拿吞吐红利 ← 论文核心
       + 0.10 × affinity_bonus  # 新：decode 留 prefill 那台
```

- **权重选择逻辑**：
  - batch_bonus 最大（20%）—— 这是论文最大卖点
  - warm + affinity 合计 25% —— 避免明显错误
  - 原有 3 项合计 55% —— 维持基础调度能力（不会牺牲负载均衡）
- **替代方案**：均匀权重 1/6 各项——拒绝，对应不到 contribution 优先级
- **代码位置**：`scheduler/RLscheduler.py::calculate_reward`

#### 2.3.5 Batch 满 admission control

- **做了什么**：同模同阶段 running 数已达 `max_batch_size` 时，新请求 `can_allocate=False`，挂回队列
- **为什么**：vLLM-style 真实行为；不挡的话 batch 会无限大，违反物理
- **代码位置**：`environment/server.py::_batch_slot_full`

---

### M3 step 3：9 项消融开关

| 名称 | 关闭后 | 类别 |
|---|---|---|
| `no_batching` | Server 退回 solo exec | 物理 |
| `no_warm_reward` | RL reward 去除 warm_bonus | RL reward |
| `no_batch_reward` | RL reward 去除 batch_bonus | RL reward |
| `no_affinity_reward` | RL reward 去除 affinity_bonus | RL reward |
| `no_aigc_state` | RL state 回到 M0 的 8+5×ns 维 | RL state |
| `no_action_mask` | PPO policy 不掩码非法动作 | PPO |
| `no_gae` | 用 MC return 替代 GAE | PPO |
| `no_pretrain` | 跳过预训练 | PPO |
| `no_entropy` | entropy_coeff=0 | PPO |

- **代码位置**：`brenchmark.py::BenchmarkTester.ABLATION_KWARGS`
- **论文用途**：Table X "Component Ablation"

---

### M4：AIGC 指标 + 真实 trace

#### 2.4.1 AIGC 标准指标定义（M4 step 1）

```
TTFT      = prefill.end_time - prefill.ready_time   # 请求级
TPOT      = (decode.end_time - decode.start_time) / output_tokens
E2E       = decode.end_time - prefill.ready_time
Goodput   = Σ output_tokens / current_time          # 系统级 tokens/s
SLO_hit   = (TTFT ≤ ttft_slo) AND (TPOT ≤ tpot_slo)
```

- **默认 SLO 阈值**：TTFT ≤ 2.0s，TPOT ≤ 0.1s/token（10 tok/s）—— vLLM、LMSYS 行业基准
- **百分位**：TTFT P50/P95/P99 + TPOT P50/P95（长尾比均值更关键）
- **代码位置**：`brenchmark.py::_collect_aigc_request_metrics`

#### 2.4.2 Poisson 到达（M4 step 2）

- **做了什么**：Task 加 `arrival_time` 字段，Simulation 在 WAITING→READY 时检查 `current_time >= arrival_time`
- **采样**：`arrival_time[i] = arrival_time[i-1] + Exp(arrival_rate)`
- **替代方案**：burst-Poisson 混合、固定间隔——保留为 future work
- **代码位置**：`environment/task.py::generate_inference_workload`

#### 2.4.3 lognormal prompt/output 分布

```
prompt ~ LogNormal(μ=5.5, σ=1.0)  # median ≈ 245 tokens
output ~ LogNormal(μ=4.5, σ=1.2)  # median ≈ 90 tokens
prompt ∈ [16, 4096]、output ∈ [10, 2000]（截断）
```

- **来源**：参考 Azure LLM Inference Trace 公开统计 + LMSYS-Chat-1M 论文
- **论文写作提示**：方法节里写"我们使用 Azure-trace-style log-normal 长尾分布生成 prompt/output 长度"，引用 Azure 那篇 paper

#### 2.4.4 内存带宽下限（HBM wall）

```python
solo_exec = max(workload / total_compute, floor × tokens)
```

| Model | decode_floor (s/token) | prefill_floor (s/token) | TPOT @ batch=1 |
|---|---|---|---|
| llama-7b  | 0.020 | 0.001  | ~20ms |
| llama-13b | 0.030 | 0.0015 | ~30ms |
| llama-70b | 0.080 | 0.005  | ~80ms |

- **校准依据**：vLLM 在 A100/A800 上的实测 throughput（50 tok/s for 7B, 33 for 13B, 12 for 70B）
- **为什么重要**：去除前 TPOT < 1ms 显然不真实；加上后 TPOT 进入 20-80ms 真实区间，SLO 检验有了实质意义
- **审稿人会问**：为什么不直接调小 compute_capacity？答：那样会同时拖慢 prefill（compute-bound），不符合 prefill 算力快+decode 内存慢的真实差异
- **代码位置**：`environment/model_catalog.py::ModelSpec.decode_floor_sec_per_token`、`environment/server.py::process_tasks`

---

## 3. 数值校准参考表

### 3.1 模型 catalog

| Model | weights_GB | cold_load_sec | kv_cache MB/tok | prefill TFLOPS/k-tok | decode TFLOPS/k-tok | max_batch | prefill_floor s/tok | decode_floor s/tok |
|---|---|---|---|---|---|---|---|---|
| llama-7b  | 14  | 5  | 0.5 | 14  | 0.5 | 32 | 0.001  | 0.020 |
| llama-13b | 26  | 9  | 0.8 | 26  | 0.9 | 16 | 0.0015 | 0.030 |
| llama-70b | 70  | 25 | 2.5 | 140 | 5.0 | 8  | 0.005  | 0.080 |
| sdxl      | 12  | 4  | — | (per-step 2.5) | — | — | — | — |

### 3.2 服务器集群

| ID | 类型 | compute (TFLOPS) | VRAM (GB) | storage (GB) | bandwidth (Mbps) | 真实对标 |
|---|---|---|---|---|---|---|
| 0 | cloud | 200 | 128 | 500  | 1000 | A100 集群 |
| 1 | edge  | 50  | 64  | 256  | 500  | A100 推理卡 |
| 2 | edge  | 20  | 32  | 128  | 300  | T4 |
| 3 | edge  | 10  | 16  | 64   | 200  | Jetson AGX |
| 4 | edge  | 50  | 64  | 256  | 500  | A100 |
| 5 | edge  | 10  | 16  | 64   | 200  | Jetson |
| 6 | edge  | 20  | 32  | 128  | 300  | T4 |
| 7 | edge  | 10  | 16  | 64   | 200  | Jetson |

### 3.3 网络拓扑

| 链路 | latency (ms) | bandwidth (Mbps) | 说明 |
|---|---|---|---|
| 云 ↔ 边_i | 30 + 5i (35–65) | 400 − 30i (370–190) | 广域网，云远 |
| 边_i ↔ 边_j | 5 + 3·\|i-j\| (8–23) | 800 − 50·\|i-j\| (750–500) | 同园区局域网 |

### 3.4 SLO 默认阈值

| 指标 | 阈值 | 来源 |
|---|---|---|
| TTFT | 2.0 s | 交互式聊天最高容忍 |
| TPOT | 0.1 s/token = 10 tok/s | 流式输出最低速 |

### 3.5 Workload 默认分布

| 参数 | uniform | lognormal |
|---|---|---|
| prompt | randint(64, 1024) | LogN(5.5, 1.0) clipped [16, 4096] |
| output | randint(50, 500)  | LogN(4.5, 1.2) clipped [10, 2000] |

### 3.6 Benchmark 默认运行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| num_runs | 20 | 每个配置 × 调度器跑 20 次取均值 |
| edge_counts | [3, 5, 7] | 三档边缘规模 |
| total_tasks | 300 | 单次仿真任务总数 |
| checkpoint_interval | 20 | 每 20 任务记录一次指标 |
| RL pretrain_episodes | 5 | 预训练轮数（quick mode） |
| Zipf alpha | 1.2 | 模型流行度 |

---

## 4. RL 调度器架构细节

### 4.1 PPO 超参

| 超参 | 值 | 说明 |
|---|---|---|
| 学习率 | 3e-4 | Adam |
| γ | 0.95 | discount |
| GAE λ | 0.90 | bias-variance trade-off |
| clip ε | 0.2 | PPO standard |
| entropy_coeff | 0.05 | 初始；预训练后降到 0.01 |
| value_coeff | 0.5 | critic loss 权重 |
| max_grad_norm | 0.5 | gradient clipping |
| update_interval | 32 | 凑够 32 步更新一次 |
| ppo_epochs | 4 | 每次更新做 4 epoch |
| warmup_steps | 30 | 首轮均匀随机收集多样化经验 |

### 4.2 网络结构

```
state (96 维) 
   → Linear(96 → 256) + LayerNorm + ReLU
   → Linear(256 → 128) + LayerNorm + ReLU
   → Linear(128 → 64)  + ReLU
   ├→ Actor head: Linear(64 → num_servers) + masked softmax
   └→ Critic head: Linear(64 → 32) + ReLU + Linear(32 → 1)
```

### 4.3 训练流程

1. **预训练**（构造时执行）：在相同任务集上跑 N 轮（默认 5），每轮结束 reset 环境，最后一批轨迹 force_update
2. **正式运行**：每个 schedule() 调用收集若干 transition，凑够 32 步触发 PPO 更新
3. **热身**：决策前 30 步用均匀随机策略（预训练后跳过）

---

## 5. 已知简化 & 审稿人可能提问

### 5.1 简化项（论文中需明确）

| 简化 | 影响 | 我们的态度 |
|---|---|---|
| Static batching at admission（非真 continuous） | 不模拟动态加入/退出 batch | 在 Methodology 里明确说"是 continuous batching 的简化模型" |
| Memory floor 是常数（非动态） | 不模拟 batch size 对 floor 的影响 | 影响小，因 floor 在 decode 占主导 |
| 模型 catalog 只有 4 个 | 不覆盖多模态 / MoE | "Future work: extend to MoE" |
| Trace 是 lognormal 合成（非真实下载） | 接近但不完全等于 Azure trace | "Trace distribution matches Azure published statistics" |
| 没建网络拥塞 | 假设网络带宽稳定 | 论文里不提；引用其他工作里说"orthogonal to this work" |

### 5.2 审稿人 FAQ 预备答案

| 问题 | 答案 |
|---|---|
| Q: 为啥不用真实 Azure trace？ | A: lognormal 参数直接取自 Azure 论文公开统计；引入 trace 文件涉及下载/解析无关变量，留作 reproducibility 附录 |
| Q: 为啥 batch overhead 用线性模型？ | A: vLLM benchmark 在 batch ≤ max_batch 区间线性是好近似；超过 max_batch 用 admission control 拦截，不需要拟合饱和段 |
| Q: 边缘服务器配置怎么选的？ | A: 真实异构云边场景：A100 推理卡 / T4 / Jetson AGX 三档；7 台是为了让 PP 和 DP 都有发挥空间 |
| Q: RL 收敛慢/方差大怎么办？ | A: 5 episode 预训练是 quick 配置；论文实验用 20+ episode，附录给收敛曲线 |
| Q: 为啥不考虑 prompt caching？ | A: 系统正交的优化；本文 focus on scheduling，不涉及 KV cache reuse across requests |

### 5.3 已知 "TODO 但还没做" 项目

- [ ] 完整 ablation 表（全部 9 项 × 各 SLO 配置）跑出来
- [ ] RL 训练曲线（reward / loss vs episode）的图
- [ ] 调度时间开销对比（each scheduler 的 wall-clock per call）
- [ ] 能耗 / 能效比指标
- [ ] 大规模可扩展性实验（1000+ 任务、15+ 服务器）

---

## 6. 实验设计矩阵（论文 4.x 节）

### 6.1 主对比（Main Experiments）

| 实验 | workload | configurations | 指标 | 论文 figure |
|---|---|---|---|---|
| Exp 1: 基础对比 | inference + lognormal + λ=2 | edge ∈ {3,5,7} × 5 schedulers | makespan, TTFT/TPOT/SLO | Fig 2 lines + CI |
| Exp 2: 负载敏感性 | inference + lognormal + λ ∈ {0.5, 1, 2, 4, 8} | edge=5 × 5 schedulers | SLO_attainment vs λ | Fig 3 |
| Exp 3: AIGC vs generic | dag / dag-aigc / inference | edge=5 × RL | makespan 各模式对比 | Fig 4 |

### 6.2 Ablation（必有）

| 配置 | 关闭项 | 期望效果 |
|---|---|---|
| Full | — | baseline (best) |
| − batching | no_batching | makespan / SLO 显著下降 |
| − warm | no_warm_reward | 冷加载多，TTFT 上升 |
| − batch | no_batch_reward | RL 不再聚集同模请求，goodput 下降 |
| − affinity | no_affinity_reward | KV 迁移多，TPOT 上升 |
| − aigc_state | no_aigc_state | RL 失去 AIGC 信息，退化到 baseline |
| − mask | no_action_mask | 非法动作多，训练不稳 |
| − GAE | no_gae | 训练方差变大 |
| − pretrain | no_pretrain | 早期 episode 表现差 |
| − entropy | no_entropy | 过早收敛到次优 |

→ 输出 paper Table 1: Component Contribution

### 6.3 调度器对比的统计检验

- 每配置 20 runs × 不同 seed
- 成对 **Mann-Whitney U** 检验（不假设正态）
- 输出 `figs/.../statistical_tests.csv`
- 显著性标记：`*` (p<0.05), `**` (p<0.01), `***` (p<0.001)

---

## 7. 论文章节 ↔ 设计映射

| Paper 章节 | 对应设计决策 | 代码引用 |
|---|---|---|
| 1. Introduction | §1.1 五条 AIGC 物理 | — |
| 2. Background | M0 + Distributed Transformer / Diffusion fig | `plot/distributed_*.drawio` |
| 3. Problem Formulation | inference task model（prefill→decode）+ SLO | `environment/task.py::TaskKind` |
| 4. System Design | M1+M2+M3 全部物理建模 + RL 设计 | `plot/trace_integration.drawio` |
| 4.1 Memory model | M1: weight resident + LRU | `environment/server.py` |
| 4.2 Two-phase task | M2: prefill/decode + KV cache encoded in output_size | `environment/task.py` |
| 4.3 Batching | M3: static admission-time batching | `environment/server.py::process_tasks` |
| 4.4 Memory wall | M4 step 2: decode_floor | `environment/model_catalog.py` |
| 4.5 RL design | state + reward + ablation hooks | `scheduler/RLscheduler.py` |
| 5. Experiments | §6 实验矩阵 | `brenchmark.py` |
| 5.1 Main results | Exp 1 | `figs/main/` |
| 5.2 Ablation | §6.2 | `figs/abl_*/` |
| 5.3 Sensitivity | Exp 2 | `figs/load_lambda*/` |
| 6. Discussion | §5.1 简化项 + §5.2 FAQ | — |
| 7. Conclusion + Future Work | §5.3 TODO 列表 | — |

---

## 8. 引文（论文中预计要引的关键文献）

> 写论文时按这个清单查 BibTeX；这里只列方向，具体论文等查的时候补全。

| 主题 | 代表文献方向 |
|---|---|
| LLM 推理系统 | vLLM (SOSP'23), TensorRT-LLM, FlashAttention |
| Continuous batching | Orca (OSDI'22), vLLM 那篇 |
| 模型并行 | Megatron-LM, ZeRO (DeepSpeed) |
| 扩散模型 | DDPM (NeurIPS'20), Stable Diffusion |
| LLM trace 公开数据 | Azure LLM Inference Trace 2023, BurstGPT 2024, LMSYS-Chat-1M |
| 云边推理 | InFog, EdgeServe, 一系列 2022-2024 边缘推理 paper |
| HEFT / 通用 DAG 调度 | Topcuoglu 2002 HEFT 原文 |
| RL for scheduling | DeepRM (HotNets'16), Decima (SIGCOMM'19), MARL 调度系列 |
| PPO | Schulman 2017 |

---

## 9. 项目时间线（自审用，不入论文）

- M1：模型权重物理（commit 1466274）
- M1.5：benchmark --out 支持（commit b7cb046）
- M2：两阶段任务 + KV cache
- M3 step 1：continuous batching
- M3 step 2：AIGC-aware RL（state + reward）
- M3 step 3：9 项 ablation
- 重构：tests/ + demos/ 子目录 + README
- M4 step 1：AIGC 指标 (TTFT/TPOT/goodput/SLO)
- M4 step 2：Poisson + lognormal + memory floor
- 文档：3 张 draw.io + 本设计记录

---

## 10. 维护提示

- **此文档与代码漂移检测**：跑全部测试通过 = 文档里的数值校准没失效
- **修改源码时**：同步更新此文档对应小节
- **写论文时**：先扫一遍本文档 §3 数值表，确保 paper 里所有数字都和代码一致
