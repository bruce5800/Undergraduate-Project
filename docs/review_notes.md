# Paper Draft Review Notes

> 通读 `docs/paper_draft.md` 后整理。问题分 3 级：
> 🔴 必修（事实错误 / 残留 stub）
> 🟡 应修（数字 / 引用一致性）
> 🟢 建议（措辞 / 流畅度）

---

## 🔴 必修问题（10 项）

### R1. 元文档头需要删除（论文最终版不需要）

**位置**：第 1-17 行
**问题**：
```
# Paper Draft（草稿）
> 目标：完成论文 §5.1 主结果...
> 数据来源：figs/energy_scan2/...
> 状态：A sensitivity 实验后台跑中...

## 工作标题（待定）
**主选**：> Pareto-Dominant Reinforcement Learning...
**备选**：> AIGC-Aware Scheduling for...
```
**修法**：选定标题（推荐主选），删除整个"草稿/状态/备选标题"块，第一行直接是论文标题。

### R2. "Section Outline" 表是 meta-document，应删除

**位置**：第 382-401 行（夹在 §2 和 §3 之间）
**问题**：这是论文写作计划表（"✅/TODO"），现在所有节都完成了，留着会让 reviewer 觉得 "draft 状态未清理"。
**修法**：整段删除（含前后 `---` 分隔线）。

### R3. §5.5 末尾的 stub 残留

**位置**：第 1296-1304 行
**问题**：
```
> 复用现有 `figs/ablN30_*` 数据 + F12 数据
> 关键发现已确认：
> - `no_batching` → −84% SLO ...
```
这是写 §5.5 之前的占位 stub，跟上方已完成的 Table 6 + 4 Findings 重复。
**修法**：删除这 9 行（quote block）。

### R4. §1 一处 RL 算法事实错误

**位置**：第 149-150 行
**原文**：
> "Our AIGC-aware PPO scheduler, trained with **the same algorithmic backbone as A3C-R2N2** but with explicit AIGC state features and reward shaping..."

**问题**：PPO ≠ A3C！这俩是**不同 RL 算法**（PPO 有 clipping，A3C 是 async actor-critic 无 clipping）。
**修法**：改成 "...trained with **the same actor-critic family as A3C-R2N2 and equivalent pretraining budget**..."。这是诚实的对照定位。

### R5. §1 的 "GNN [Mao'19]-style" 引用错位

**位置**：第 144-145 行
**原文**：
> "...modern DRL baselines (A3C-R2N2 [Tuli'22], **GNN [Mao'19]-style**) all cluster..."

**问题**：我们的 GNN baseline 是自己实现的，不是 Mao 19 (Decima)。引用挂在错对象上。
**修法**：改成 "...modern DRL baselines (A3C-R2N2 [Tuli'22], a Decima-inspired GNN variant we implemented [Mao'19])..."

### R6. §6.1 Implication 2 措辞与 §5.5 不一致

**位置**：第 1346-1348 行
**原文**：
> "without pretrain, PPO collapses to an 'always-cloud' policy that is locally optimal under the gradient landscape but globally bad on SLO and energy."

**问题**：§5.5 的 no_pretrain SLO = 0.187（不是 0.05），不是"完全 cloud collapse"。"Cloud collapse" 是 F1+F2 之前的 OLD 现象，跟单纯关 pretrain 不是同义。
**修法**：改成 "without pretrain, the SLO drops 38% as the policy fails to escape locally optimal regions—analogous to the 'always-cloud' attractor we documented in the diagnostic process (§F1+F2)."

### R7. §5.0 "implementing four physics" — 应是 5

**位置**：第 825 行
**原文**：
> "All experiments use our AIGC inference simulator implementing **four physics**: (M1) ... (M2) ... (M3) ... (M4) ..."

**问题**：§1 和 Abstract 都强调**5 个 AIGC 物理**。这里写 4 是因为它对应 M1-M4 milestones，但读者会以为 contradiction。
**修法**：改成 "...implementing the five AIGC physics (organized as four engineering milestones M1–M4):" 然后保持原 4 个解释，最后加 "with the two-phase request lifecycle modeled through M2's task decomposition."

### R8. NVIDIA'24 in-text 引用但 BibTeX 没条目

**位置**：第 1492 行
**原文**：
> "TensorRT-LLM [NVIDIA'24] provides production-grade kernels..."

**问题**：NVIDIA'24 没在 `references.bib` 里。
**修法**：两选其一：
- (a) 删掉 in-text 引用："TensorRT-LLM provides production-grade kernels..."
- (b) 在 references.bib 加 entry：
```bib
@misc{nvidia2024tensorrtllm,
  author = {{NVIDIA Corporation}},
  title  = {{TensorRT-LLM}},
  howpublished = {\url{https://github.com/NVIDIA/TensorRT-LLM}},
  year   = {2024},
}
```
推荐 (a)，论文不依赖 TensorRT-LLM 任何具体 claim。

### R9. Abstract 和 §1 一行直接矛盾

**位置**：Abstract 第 41-43 行
**原文**：
> "+28% SLO and −7% energy per token at the canonical configuration (edge=5, λ=2 req/s), with statistical significance (p<0.05) on both axes against the four strongest comparators."

vs §1 第 153-154 行：
> "...both statistically significant (p<0.05 against all four strong baselines)."

vs §5.1 Statistical Significance 实际数据：
- SLO: p<0.05 vs all 4 ✓
- Energy/token: p<0.05 vs **3 of 4**（PSO p=0.11 不显著）

**问题**：Abstract 和 §1 都说"both axes against four"，但 energy 实际只 3/4 显著。
**修法**：Abstract 改成 "...significant at p<0.05 on SLO against all four comparators and on energy efficiency against three of four (PSO p=0.11, others p<0.05)."
§1 第 153-154 行类似改。

### R10. 双 `---` 分隔线

**位置**：第 59-61 行 + 第 1306-1308 行
**问题**：两处出现连续两个 `---`（多余空行）。
**修法**：每处删一个。

---

## 🟡 应修问题（5 项）

### Y1. Figure 编号叙述跟实际生成不对应

**位置**：§5.3 第 1092 行 `#### Fig 6 Description: SLO and Energy vs λ`
**实际**：load sensitivity 图是 `fig5_load.png`，topology 是 `fig6_topology.png`。
**修法**：把第 1092 行 "Fig 6" 改成 "Fig 5"；在 §5.2 加一段 "Fig 6 Description: SLO and Energy vs Edge Count"（topology）。
（或反过来：把生成的图改名让 §5.3 = Fig 6 = load。论文里 Fig 5 ≤ Fig 6 哪个先都行，**但要跟图编号自洽**。）

### Y2. §5.4 Workload section 引用 Table 5，但 §5.5 ablation 用 Table 6

**位置**：原 Section Outline 表（已建议删除 R2）说 §5.4 = Table 5，§5.5 = Table 5（错误重复）。Body 实际：§5.4 = Table 5，§5.5 = Table 6。**Body 是对的，Outline 表是错的**——R2 删除后此问题自动消除。

### Y3. §6.3 数字 "78% of RL's SLO"

**位置**：第 1443-1445 行
**原文**：
> "LeastLoaded and ShortestQueue achieve 78% of RL's SLO attainment with 0% of the training cost."

**核查**：§5.1 数据 LL=0.239, SQ=0.234, RL=0.302。78% = 0.239/0.302 ≈ 0.79 ≈ 78%。**数字正确** ✓

### Y4. §6.3 "$M-scale" 用了不太学术的金钱符号

**位置**：第 1454-1455 行
**原文**："a 5% energy reduction translates to meaningful annual savings ($M-scale at hyperscaler volumes)"
**修法**：改成 "millions of USD annually at hyperscaler volumes"，避免符号 `$M`（学术写作通常拼写）。

### Y5. §7.5 末尾"to our knowledge" 段对自己 contribution 的措辞略弱

**位置**：第 1590-1595 行
**原文**：
> "Our simulator (§4.1) is, to our knowledge, the first open-source platform that models all five AIGC physics listed in §1 *with* a focus on cross-server scheduling decisions and *with* multi-objective (SLO + energy) evaluation built in."

**建议**：拆成两句更有力：
> "To our knowledge, our simulator is the first open-source platform combining three properties: (i) modeling all five AIGC physics; (ii) focusing on cross-server scheduling rather than intra-server batching; (iii) supporting multi-objective (SLO + energy) evaluation. We release it with reproducibility manifests."

---

## 🟢 建议优化（4 项）

### G1. §1 末尾 paper roadmap 可省略

**位置**：第 209-214 行
> "The rest of this paper is organized as follows: §2 reviews..."

**理由**：很多顶会论文（OSDI/SOSP）不写 roadmap，认为读者会看 ToC。**保留** 也 OK，看 target venue 风格。

### G2. §4.1 / §4.2 篇幅过长（合计 ~2000 词，论文里可能 4-5 页）

**位置**：第 583-816 行
**建议**：投顶会前可能要压缩 30%（合并相近段落、把实现细节移到 appendix）；投中文核心 / EI 篇幅充裕没问题。

### G3. §6.2 "Single learning algorithm" limitation 写法可以更正面

**位置**：第 1417-1425 行
**原文**：
> "Our scheduler uses PPO with specific architectural choices ... Other RL families—offline RL, model-based RL ... might respond differently to AIGC reward shaping."

**建议**：把它包装成 "Future direction" 而非 "Limitation"，因为 "我们测了 PPO + A3C + GNN 3 个都一致" 本身是**强证据**而非 weakness。

### G4. §8 Conclusion 末段可以再升华一层

**位置**：第 1635-1640 行
**原文**：
> "We hope this work supports two longer-term directions: more **realistic AIGC scheduling benchmarks** built on the open platform we release, and a **research culture of aggressive joint ablation** for AIGC scheduling claims."

**建议**：加一句更广的影响力声明：
> "...As LLM workloads grow to dominate datacenter energy budgets, scheduler design will increasingly be judged on the multi-objective Pareto frontier rather than single-axis throughput; we hope this work contributes a methodology and an open platform for that broader research direction."

---

## 论文 review 总评

**强项**：
- ✅ 故事自洽（Abstract → §1 → §5 → §8 narrative 一脉相承）
- ✅ 反 prior-work 立论（§5.5 ablation null result + §6.1 implications）
- ✅ 诚实交代 limitations（§6.2）和 historical journey（§4.2 cloud overuse 段）
- ✅ 数据扎实（30 个配置 × N=30 × 9 调度器）
- ✅ 4 张图覆盖 main / topology / load / ablation 全 narrative

**弱项**：
- ⚠ Meta-content / stub 残留（R1, R2, R3）让 draft 看起来"未 finalize"
- ⚠ 几处事实/数字不一致（R4, R6, R7, R9）需要细心修
- ⚠ Figure / Table 编号需要 final-check

**修复工作量估计**：1-2 小时即可处理完所有 🔴 + 🟡 问题。🟢 建议看个人偏好可选。

**修完之后**：paper draft 就可以正式 LaTeX 化，论文实际可投状态。
