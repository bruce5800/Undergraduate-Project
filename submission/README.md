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

1. ~~去年毕设是否进了知网/学校机构库~~ → **已确认未入库**，cover letter 毕设段已删除，正文无需提毕设、无 self-citation 负担。
2. **推荐审稿人 3 位真名**（建议从你 references.bib 里引用的作者挑，避开同单位/同导师）。
3. **作者署名与单位**：`paper/main.tex` 第 41–46 行目前是匿名占位，投稿前替换为真实信息（FGCS 单盲，需作者实名）。
4. 🔴 **导出 RL 调度器架构图（Fig 4）**：`paper/figures/rl_scheduler_arch.pdf` 现在是**红框 PLACEHOLDER 占位**。投稿前务必在 draw.io 打开 `plot/rl_scheduler_arch.drawio` → Export as PDF（勾 Crop）→ 覆盖该文件（再导一份 .png）。否则论文里 Fig 4 会是占位图。中文版 symlink 同一文件，导一次即可。
