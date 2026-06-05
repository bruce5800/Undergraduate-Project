# FGCS 投稿材料清单

本目录是投 *Future Generation Computer Systems* (Elsevier) 用的提交材料。
论文正文在 `../paper/`（英文）/ `../paper-zh/`（中文备份）。

## 文件

| 文件 | 用途 |
|---|---|
| `highlights.txt` | 5 条 Highlights，每条 ≤85 字符（已校验）。投稿系统必填项。 |
| `cover_letter.md` | Cover letter 模板，带 `[...]` 待填字段。 |
| `README.md` | 本清单。 |

## Elsevier 提交前 checklist

投稿入口：<https://www.editorialmanager.com/fgcs/>（以官网为准）

- [ ] **正文 PDF**：首轮可直接用 `../paper/main.pdf`（IEEEtran 双栏，13 页，≤18 页限内）。Elsevier "Your Paper Your Way" 首轮不强制 elsarticle 格式。
- [ ] **Highlights**：粘贴 `highlights.txt` 的 5 条（去掉行尾字数标注）。
- [ ] **Cover letter**：填好 `cover_letter.md` 的 `[...]` 字段（日期、作者、单位、ORCID、3 位推荐审稿人）。
- [ ] **Abstract + Keywords**：从论文 `00_abstract.tex` 取。
- [ ] **Declaration of competing interests**：无利益冲突，系统内勾选/声明。
- [ ] **CRediT author statement**：单作者填 "[Name]: Conceptualization, Methodology, Software, Investigation, Writing."
- [ ] **Data/Code availability statement**：填仿真器开源地址（GitHub repo URL）。
- [ ] **Suggested reviewers**：3–5 位（cover letter 里已列位置，填真人）。
- [ ] **ORCID**：注册并绑定。
- [ ] **选 Subscription 渠道**（不勾 Gold OA）→ 零版面费。

## 投稿前论文侧已补三项（本轮已完成）

- [x] 补引竞品 SLICE（arXiv 2510.18544）+ §7 三维度差异化段落
- [x] §6 强化"为何用仿真 + 真机留作未来工作"的辩护
- [x] Highlights + Cover letter 起草

## 仍需你手动确认的事

1. ~~去年毕设是否进了知网/学校机构库~~ → ✅ **已确认未入库**，cover letter 毕设段已删，正文无 self-citation 负担。
2. ~~推荐审稿人 + 邮箱~~ → ✅ **已填 3 个干净候选**：Hao Zhang `haz094@ucsd.edu` · Xin Jin `xinjinpku@pku.edu.cn` · Rajkumar Buyya `rbuyya@unimelb.edu.au`（Bianchini 留作可选第 4）。
3. ~~作者单位 + ORCID~~ → ✅ **University of Bristol, Faculty of Science and Engineering**，邮箱 `nu25406@bristol.ac.uk`，ORCID `0009-0004-1487-3862`（已填入中英 main.tex + cover letter）。
4. ~~导出 4 张 drawio 图~~ → ✅ **全部已导出并提交**（Fig 1 生命周期 / Fig 2 扩散 / Fig 3 仿真器 / Fig 4 RL 架构）。

### 图导出对照表（已全部完成）

| drawio 源（plot/） | paper/figures/ | 论文编号 | 状态 |
|---|---|---|---|
| `llm_inference_lifecycle.drawio` | `llm_inference_lifecycle.{pdf,png}` | Fig 1 | ✅ |
| `distributed_diffusion.drawio` | `distributed_diffusion.{pdf,png}` | Fig 2 | ✅ |
| `trace_integration.drawio` | `trace_integration.{pdf,png}` | Fig 3 | ✅ |
| `rl_scheduler_arch.drawio` | `rl_scheduler_arch.{pdf,png}` | Fig 4 | ✅ |

Fig 5–8（Pareto/拓扑/负载/消融）是 matplotlib 直接生成，无需 drawio。
（旧 distributed_transformer 训练图已删，因与推理调度主题偏远。）

---

## 投稿当天还要做的（非文件，Elsevier 系统内操作）

- [ ] cover letter 顶部 `[Date]` 填投稿日期。
- [ ] Elsevier 账号用**永久邮箱**注册（防 Bristol 校园邮箱评审期内停用）；论文上印 Bristol 邮箱没问题，系统通信邮箱用永久的。
- [ ] 投稿系统里粘贴 **Highlights**（`highlights.txt` 5 条，去掉行尾字数标注）。
- [ ] 选 **Subscription 渠道**（不勾 Gold OA）→ 零版面费。
- [ ] 上传 `paper/main.pdf` 作为 manuscript，cover letter 贴进对应框。
