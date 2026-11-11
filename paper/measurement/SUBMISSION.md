# Measurement（Elsevier）投稿清单

> 本目录的稿件由 `论文\build_measurement_manuscript.py` 从已校对的 PMEA 版逐字转换而来：
> **所有数字、表格、图与 PMEA 版完全一致**，只改了期刊要求的格式（Highlights、非结构化摘要、
> 编号制参考文献、CRediT 声明、声明区顺序）。转换脚本可重复运行，不具备"手改"环节。

## 一、目标期刊

| 项 | 内容 |
|---|---|
| 期刊 | Measurement（Elsevier），ISSN 0263-2241 |
| 分区 | 中科院分区 2 区（工程技术）；JCR Q1（Instruments & Instrumentation） |
| 费用 | 订阅路线**免费**；仅当选 OA 才付 APC |
| 稿件类型 | Research paper（full length） |
| 投稿系统 | Editorial Manager（`https://www.editorialmanager.com/meas/`，以期刊官网 "Submit your paper" 链接为准） |
| 选它的理由 | 本文是"测量/计量学"问题：报告出来的可分性有多少来自采集条件。Measurement 收测量验证、不确定度、复现类工作，比传感类期刊更对口 |

> 分区与费用请在 LetPub 上以最新年份核对一次（我这边联网检索接口当前返回 401，无法在线复核）。

## 二、合规项（可直接核对）

| 要求 | 状态 | 依据 |
|---|---|---|
| Highlights：3–5 条、每条 ≤ 85 字符 | ✅ 5 条，最长 74 字符 | 稿件第 1 页；`Highlights.txt` 另附 |
| 摘要：非结构化、≤ 250 词 | ✅ 237 词 | 单段，无 Objective/Approach 小标题 |
| 关键词：1–7 个 | ✅ 6 个 | — |
| 参考文献：编号制、按正文首次出现排序 | ✅ 22 条，[1]–[22] 全部被引 | 转换脚本自动排序 |
| 参考文献格式 | ✅ Elsevier：`作者, 题名, 刊名 卷 (年) 页码. DOI`，刊名斜体 | — |
| 图表在正文首次引用处 | ✅ 7 图 7 表 | 与 PMEA 版同一套 |
| CRediT 作者贡献声明 | ✅ 新增 | References 之前 |
| Declaration of competing interest | ✅ | 原 "Conflict of interest" 改写 |
| Data availability statement | ✅ | 指向 MIT 仓库 |
| Ethics statement | ✅ | 公开去标识数据二次分析 |
| Funding | ✅ | 无资助 |
| 正文无中日韩字符 | ✅ 0 个 | 脚本检查 |
| 篇幅 | 21 页 / 7 936 词（Word 统计，含参考文献与表格） | Measurement 无硬性字数上限 |

## 三、上传文件

| 文件 | 用途 |
|---|---|
| `manuscript.pdf` | 主稿件（审稿用 PDF，含 Highlights 页与全部图表） |
| `manuscript.docx` | 主稿件源文件（系统要求 Word 版时用） |
| `cover_letter.pdf` | 投稿信（1 页，524 词） |
| `Highlights.txt` | Highlights 纯文本，供系统"Highlights"字段粘贴或单独上传 |
| `SUBMISSION.md` | 本清单 |

## 四、需要你本人在系统里做的

1. **登录 Editorial Manager**，Article Type 选 `Research paper`（若列表里有 `Full length article` 亦同）。
2. **填元数据**：标题与摘要直接从 `manuscript.pdf` 第 2 页复制；关键词 6 个；作者与单位英文名核对一次。
3. **上传顺序**：Cover Letter → Manuscript（PDF 优先）→ Highlights（可粘贴文本）；若系统要求，再补 DOCX。
4. **声明页**：所有作者确认无利益冲突、未一稿多投、数据可得性陈述属实。
5. **推荐审稿人**（可选，建议 3–5 位）：雷达生命体征测量、生理信号验证、机器学习可复现性三个方向；**不要填有合作关系的同行**。
6. **ORCID**：`0009-0002-3231-0444`（与稿件一致，若不对改 `论文\pmea_content.py` 后重跑两个 build 脚本）。
7. 系统生成 PDF 后**通读一遍**，重点看：公式与上下标、表 6 的 NC1 行、图 6 图例、参考文献 [1]–[22] 是否连续。

## 五、如果 Measurement 也拒

按顺序换，不要回投 J-BHI / TIM（同一编辑部已两次拒稿）：

| 顺序 | 期刊 | 分区/费用 | 需要改什么 |
|---|---|---|---|
| 1 | IEEE Sensors Journal | 2 区，订阅免费（超页可能收费） | IEEE 双栏模板、编号制（已满足）、cover letter 重写 |
| 2 | Biomedical Signal Processing and Control | 2 区，订阅免费 | Elsevier 模板（已接近），改 article type 与 cover letter |
| 3 | Physiological Measurement | 3 区，订阅免费 | 已有现成 `论文\投稿包_PMEA\`，可直接用 |

## 六、与 PMEA 版的关系（应对"是否一稿多投"）

- PMEA 版**从未上传**，只存在于本工作区；两版共用同一批数据、同一套代码、同一组数字。
- 若被问及，回答口径：先做 Physiological Measurement 的格式准备，因分区要求改为投 Measurement；两版内容相同、均未投稿，不存在重复投稿。
- 旧稿（J-BHI / TIM）在 `论文\_旧稿_已拒\`，cover letter 已如实披露曾投两个 IEEE 期刊且未获接受。
