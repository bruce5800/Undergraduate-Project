# 转投决策调研报告（2026-08-02）

背景：FGCS-D-26-02940 被拒（主因 PDF 编译事故污染了 2/3 审稿），FGCS 不接受重投。
本报告回答三个问题：Q1 新颖性威胁、Q2 贡献点存活评估、Q3 转投去向。
三路并行网络检索（共 30+ 次搜索、逐篇核实摘要页）汇总而成。

---

## Q1 新颖性威胁扫描（2024–2026）

结论：坑位**部分被占**。"RL 调云边 LLM 推理"的宽泛叙事已不新颖；可辩护的新颖性收窄为精确交集：
**深度 RL（PPO）× 异构云边服务器池 × serving 物理作为状态/奖励 × SLO+能耗 Pareto 占优 × 逐请求放置**。
该精确组合目前无人做到。重投必须把下列 HIGH/MEDIUM 论文纳入 related work，其中 PerLLM 必须作为对比基线。

### HIGH 威胁（必须引用 + 建议作为基线）

1. **PerLLM** — arXiv:2405.14636（2024-05）
   Edge-cloud 协同下多样 LLM 服务的逐请求调度，约束 UCB（组合 bandit），报告 1.6–2.2x 吞吐、>50% 能耗成本降低。
   四轴全中（云边/学习式/能耗一等/逐请求）。差异：bandit 而非深度 RL；无 Pareto 多目标框架；未暴露 serving 物理。
   https://arxiv.org/abs/2405.14636

2. **Compact LLM Deployment + World-Model PPO Offloading in MEC** — arXiv:2602.13628（2026-02）
   世界模型辅助 PPO 做设备侧逐请求推理卸载，能耗预算约束，能耗/查询最高降 50%。
   差异：设备侧"是否卸载"决策而非服务器池放置；QoS 特征是精度/幻觉而非 serving 物理；一半篇幅是模型压缩。
   https://arxiv.org/abs/2602.13628

### MEDIUM 威胁（必须引用、划清边界）

3. **Splitwise-DRL（Lyapunov 辅助云边分层 DRL）** — arXiv:2512.23310（2025-12）：模型分割粒度，非请求放置。
4. **MARLIN（多智能体博弈 RL，数据中心多目标 Pareto）** — arXiv:2605.13496（2026-05）：无边缘；已用 Pareto hypervolume 框架。
5. **Active Inference 云边 LLM 卸载** — IEEE TMC 2024（10.1109/TMC.2024.3415661）：明确以"胜过 DRL"定位，审稿人可能追问为何用 PPO。
6. **T2DRL（边缘模型缓存+卸载）** — arXiv:2501.14205（2025-01）：建模了模型驻留/缓存决策，成本非能耗。
7. **NSGA-II 云边 LLM 实例路由** — arXiv:2507.15553（2025-07）：同问题形状（多目标逐请求路由异构云边池），遗传算法；天然对比基线。
8. **Lodestar（在线学习 LLM 路由器）** — arXiv:2606.00946（2026-05）:显式建模 batching+KV 复用耦合，单集群、无能耗。
9. **Microsoft Intelligent Router** — arXiv:2408.13510（2024-08）：RL 逐请求路由、感知 prefill/decode 与 batch 混合;单集群。
10. **FREESH** — arXiv:2511.00807（2025-11）：能耗+SLO+异构池+逐请求，优化法非 RL、非边缘。
11. **Festina** — arXiv:2606.30391（2026-06）：serverless 共享 GPU 能耗优先放置,启发式。
12. **QoS-Aware Edge LLM 专家路由（DRL+异构图注意力）** — arXiv:2508.00234（2025-08）：无能耗、无云层。

### LOW 威胁（related work 引用即可）

- **VoltanaLLM** — arXiv:2509.04827：能耗+SLO+路由三要素，控制论方法、单集群。
- **SLICE** — arXiv:2510.18544：仅 2 篇引用（FlexiTensor TPDS'26、GoodServe arXiv:2605.16867），均未撞车。
- **MSAO** — arXiv:2604.02945：PerLLM 团队新作（多模态稀疏卸载）——说明该团队在此领域持续迭代，**时间窗口在收窄**。
- GreenLLM（arXiv:2508.16449）、TAPAS（ASPLOS'25）、BEAM（MLSys'26）：单集群能耗/功率方向。

---

## Q2 贡献点存活评估

| 贡献 | 判定 | 需要的动作 |
|---|---|---|
| C1 仿真平台 | **存活（措辞收窄）** | "first" → "first open simulator to **combine**（五物理+云边+SLO/能耗）"；新增 ~12 篇仿真器引用 |
| C2 RL 调度器 | **勉强存活（需重定位）** | 以"serving 物理感知的状态/奖励设计 + 云边池 Pareto 占优"为核心卖点；PerLLM 必须进基线，建议 NSGA-II 路由也进 |
| C3 消融零结果 | **完好存活，建议升为主卖点** | 无撞车。最接近的是 2026 制造调度 MARL 消融（arXiv:2606.31737,部分类似发现）；可引 Interpretable DRL Scheduling（arXiv:2403.16293）呼应 |
| C4 工况刻画 | **完好存活** | 引 Decima（SIGCOMM'19,高负载才有收益）、Microsoft Router（低负载排序无关紧要）、DAG-topology regimes（arXiv:2604.09202）作先例支撑 |

### C1 必引的仿真器清单（划边界用）

| 仿真器 | 出处 | 与本文的边界 |
|---|---|---|
| Vidur | MLSys 2024, arXiv:2405.05465 | 同构 DC、无能耗、无跨服务器 KV 本地性 |
| Splitwise / SplitwiseSim | ISCA 2024, arXiv:2311.18677 | 异构 DC SKU、KV 传输仅限 P/D 分离场景、功率仅 provisioning |
| LLMServingSim 1.0/2.0 | IISWC 2024 arXiv:2408.05499; CAL 2025 arXiv:2511.07229 | 2.0 有内存分层 KV 移动+功率，但 DC-only、无逐 token 能耗多目标 |
| Frontier | arXiv:2508.03148 / 2605.21312 | 原生跨服务器 KV 传输事件，DC 集群、无能耗 |
| Vidur+Vessim | PECS@Euro-Par'25, arXiv:2507.11417 | 能耗/碳进 LLM 仿真的关键先例，DC+电网、无边缘 |
| Helix | ASPLOS 2025, arXiv:2406.01566 | 地理分布异构池最接近云边，但模型跨节点分割、无能耗 |
| APEX / TokenSim / ReaLLM | arXiv:2411.17651 / 2503.08415 / ASAP 2025 | 并行方案/算子级/trace 驱动，均单集群 |
| AgentServeSim / Charon / KernelSight-LM | arXiv:2606.09613 / 2605.17164 / 2606.28565 | 2026 并发工作，引用即可 |

独有残余：①权重驻留/冷加载作为一等仿真原语（目前只在 ServerlessLLM OSDI'24 等**系统**论文中）；②云边池 + SLO/能耗联合框架。
佐证空白的综述：Mobile Edge Intelligence for LLMs（arXiv:2407.18921）、Network Edge Inference（arXiv:2604.22906,明确指出边缘 LLM 评估仍靠 gem5/ns-3/EdgeSimPy 级工具）。

---

## Q3 转投去向

注：CCF 2026-03 发布第 7 版目录，**TCC 从 C 升 B**。

| 期刊 | CCF | 中科院 | IF | 首审 | 费用 | 契合 | 风险 |
|---|---|---|---|---|---|---|---|
| **IEEE TCC**（首选） | B（新升） | 2区 | 5.0 | 6–10 周 | 订阅路免费 | 极好：已发 RL+LLM 推理资源管理仿真论文 | IEEE 12 页限制+超页费 |
| **JSA**（次选） | B | 2区 | 5.3 | ~10.9 周 | 订阅路免费 | 好：大量 DRL 云边卸载仿真论文 | 正式副标题"Embedded Software Design"，cover letter 需框定角度 |
| IEEE TSC（冲刺） | **A** | 2区 | 6.2 | 3–6 月 | 订阅路免费 | 很好 | CCF-A 门槛，单作者+纯仿真更难；预算 6–12 月 |
| JPDC | B | 3区 | 3.4 | ~24.7 周（慢） | 免费 | 好 | 最慢、低收益 |
| Computer Networks | B | 2区 | 4.7 | ~14.8 周 | 免费 | 中：可能被质疑"网络建模不足" | 错位风险 |
| ACM TAAS | B | 4区 | ~2.2 | ≤4 月 | ACM 全 OA;Bristol 在 Jisc-ACM Open 协议内可能免 APC | 概念契合 | IF/分区弱 |
| IEEE IoT-J | C | 1区TOP | 8.7 | ~6.9 周（快） | 超 8 页强制 $175/页 | 中：需 IoT 化改写 | CCF C+强制超页费 |

TCC 直接证据（2024–26 已发表）：Temporal-Aware GPU Resource Allocation for Distributed LLM Inference via RL (2025)；ReflexPilot: Startup-Aware DRL Edge-Cloud Scheduling (2025)；DRKC (2025)；CRTSE (2026)。

### 会议对冲

| 会议 | CCF | 截稿 | 状态 |
|---|---|---|---|
| **CCGrid 2027**（Dallas） | C | **2026-12-01 AoE** | 已核实（hpcclab.org/ccgrid27）；CFP 明含 cloud/edge + systems-for-AI，仿真友好，10 页 IEEE |
| ICDCS 2027 | B | 预计 2026-12~2027-01 | CFP 未出 |
| Euro-Par 2027 | B | 预计 2027-02~03 | 未出；仿真容忍度高 |
| ICPP 2027 | B | 预计 2027-04 | 未出 |
| Middleware 2027 | B | 预计两轮 | 未出；通常要求真实系统 |
| INFOCOM 2027 | A | 2026-07-31 | **已过** |

### 排序建议

1. **IEEE TCC**：范围完全对口且有直接先例（RL+LLM 推理 + 纯仿真都收），CCF B/中科院 2 区/免费/首审 6–10 周，IEEE 双栏稿几乎零改排。
2. **JSA**：速度与安全性最好的备选，单作者+纯仿真在该刊常见。
3. 对冲：CCGrid 2027（12-01 截稿）可与期刊路线并行规划（注意一稿多投规则——只能选一路先行）。

---

## 综合判定

- 方向**不需要换**；但论文需要**重定位**（"部分撞车"分支）：叙事重心从"RL 调度器赢了"移向"serving 物理感知设计 + 联合消融零结果 + 工况刻画"三件套。
- 修订范围 = R1 的 11 条 + related work 大改（新增 ~15–20 篇）+ 引言/贡献重写 + PerLLM（必须）与 NSGA-II（建议）基线 + 权重敏感性实验。
- 时间敏感：PerLLM 团队持续迭代（MSAO 2026-04），Lodestar/MARLIN 均为 2026 年上半年论文——**窗口在收窄，建议 4–6 周内完成修订并投出**。
