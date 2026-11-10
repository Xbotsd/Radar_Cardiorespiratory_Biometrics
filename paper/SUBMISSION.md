# 投稿清单（Physiological Measurement）

> 目标期刊：**Physiological Measurement**（IOP Publishing，ISSN 0967-3334，single-anonymous 审稿）
> 投稿入口：<https://mc04.manuscriptcentral.com/pmea-ipem>
> 本文件对应稿件：`PMEA_manuscript.docx` / `PMEA_manuscript.pdf`

---

## 一、为什么换到这本刊

| 项 | J-BHI（已拒） | TIM（已拒） | **PMEA（本次）** |
|---|---|---|---|
| 定位 | 生物医学信息学 | 仪器与测量 | 生理功能的定量测量与**方法验证** |
| 对阴性/验证类结果 | 不友好 | 一般 | **明确接受**（"rigorous large-scale validation of existing methods"） |
| 费用 | 可选 OA | 可选 OA | 订阅路线**免费** |
| 首次决定 | 慢 | 中 | 平均 26 天，接受 <100 天 |

论文的核心是"用对照实验证明已有结论站不住"，属于**方法学验证**，正是 PMEA 的收稿范围，而不是 J-BHI/TIM 偏好的"新模型刷高指标"。这是换刊的主要理由。

---

## 二、已完成的合规项（可直接核对）

| 要求 | 状态 | 证据 |
|---|---|---|
| 结构化摘要：Objective / Approach / Main results / Significance | ✅ | 四个小标题齐全 |
| 摘要 ≤ 250 词 | ✅ 237 词 | 逐段计数（含四个小标题） |
| 正文 ≤ 8000 词 | ✅ 7 941 词（Word 统计，含参考文献与表格） | Word → 字数统计 |
| 字体 ≥ 12 pt、行距 1.5 | ✅ 12 pt / 1.5 | 稿件样式为 A4、2.5 cm 页边距 |
| 图表嵌入正文相应位置 | ✅ 7 图 7 表 | 图 1–7、表 1–7 均在首次引用处 |
| 参考文献 Harvard（字母序）+ 含文章标题 | ✅ 22 条，全部被正文引用 | 无编号制，按作者姓氏排序 |
| 单盲：正文含作者与单位 | ✅ | 首页作者块 |
| ORCID | ✅ 0009-0002-3231-0444 | 稿件 "ORCID" 小节 |
| Data availability | ✅ | 指向仓库与数据集 DOI |
| Code availability | ✅ | 指向 MIT 仓库，39 个测试 |
| Ethics statement | ✅ | 公开去标识数据的二次分析 |
| Funding / Conflict of interest | ✅ | 无资助、无利益冲突 |
| SAGER（性别/社会性别）声明 | ✅ 新增 2.6 节 | 60 女 / 50 男；未分层，说明理由 |
| 人类受试者 30+ 人规定 | ✅ 已说明 | 120 GHz 队列 24 人，仅作对照，所有主结果用 110 人队列 |
| 文件名仅含 `a-z A-Z 0-9 _` | ⚠️ 提交时改名 | 见第四节 |
| 非英文/非罗马字符 | ✅ 正文纯英文 | — |

---

## 三、提交前必须由你本人完成的 5 件事

1. **确认 ORCID**（1 分钟）。
   稿件填的是 `0009-0002-3231-0444`，来自你已有仓库的 Zenodo/CITATION 元数据。若不对，改 `论文/pmea_content.py` 中 `ORCID` 一节后重跑 `build_pmea_manuscript.py`。

2. **确认作者信息与单位英文名**（1 分钟）。
   现为 `School of Mechanical Engineering, Heilongjiang University of Science and Technology, Harbin 150022, China`。

3. **推荐审稿人（可选，建议填 3–5 位）**。
   建议方向：雷达生命体征（如 TU Delft / 中科院空天院 / 深圳先进院）、生理信号机器学习验证、开集识别。**不要填有利益冲突的合作者**；系统里填姓名 + 邮箱 + 单位即可。

4. **签署并上传 IOP 的作者声明/出版协议**（投稿系统里在线勾选即可）。

5. **选择出版路线**：订阅（免费）或金色 OA（USD 3 325）。默认选订阅即可，录用后再决定。

---

## 四、上传文件清单（IOP 只要求一个 PDF）

| 顺序 | 文件 | 说明 |
|---|---|---|
| 1 | `manuscript.pdf` | 由 `PMEA_manuscript.pdf` 改名；正文、图、表全含 |
| 2 | `cover_letter.pdf` | 由 `cover_letter.pdf` 改名 |
| 3 | （可选）`supplementary.pdf` | 如需附旧稿审稿意见与回复，见下 |

命名规则：只用 `a-z A-Z 0-9 _`，不要空格和中文。

**关于旧稿审稿意见**：IOP 明确鼓励作者附上其他期刊的审稿意见与逐条回复，可加速审稿。本项目工作区内**没有** J-BHI / TIM 的审稿意见原文；如你手上有决定信或审稿意见，建议合并成一个 `supplementary.pdf` 一并上传，并在 cover letter 中说明。没有则跳过此项，cover letter 已如实说明前两次投稿情况。

---

## 五、提交操作（约 10 分钟）

1. 打开 <https://mc04.manuscriptcentral.com/pmea-ipem>，用通讯作者邮箱注册/登录。
2. `Author` → `Start New Submission` → 选 **Research Paper**。
3. 填标题（与 PDF 完全一致）、摘要（复制四段，勿改数字）、关键词（6 个）。
4. 作者列表：Yudong Xu，单位、邮箱、ORCID；勾选 Corresponding Author。
5. 上传 `manuscript.pdf`（Article Type: Main Document）与 `cover_letter.pdf`。
6. 填 Cover Letter 文本框（可直接粘贴 `cover_letter.pdf` 正文）。
7. 回答问题：数据可得性（选 "available in repository"，填 GitHub 链接）、利益冲突（无）、资助（无）、伦理（公开数据二次分析）。
8. 推荐审稿人（可选）。
9. 生成 PDF 预览，**逐页检查图是否清晰、表是否错位**，确认后 Submit。

---

## 六、时间线与后续

| 阶段 | 预期 | 你要做什么 |
|---|---|---|
| 编辑初审 | 3–10 天 | 留意邮件；若被拒（desk reject），按第七节换刊 |
| 外审 | 3–8 周 | 收到意见后逐条回复；仓库里 `docs/NUMBERS.md` 可直接用于回应"数字从哪来" |
| 修回 | 1–2 轮 | 改动同步到仓库并重跑测试，保持论文与代码一致 |
| 录用 | <100 天（期刊中位数） | 校样、版权、OA 选择 |

---

## 七、若被拒：备选期刊阶梯

按"契合度 × 接收概率 × 成本"排序，换刊时只需改格式，结论与数字不用动：

| 顺位 | 期刊 | 为什么合适 | 需要改什么 |
|---|---|---|---|
| 2 | **IEEE Sensors Journal** | 测量与验证类论文的主场，接受"方法学再评估" | 双栏 IEEE 模板，压缩到 10–12 页；把"测量协议"提到更前 |
| 3 | **Biomedical Signal Processing and Control**（Elsevier） | 信号处理 + 验证，审稿较快 | Elsevier 模板；补一小节与最新雷达生物识别工作的对比表 |
| 4 | **Biomedical Physics & Engineering Express**（IOP） | IOP 内部转投通道，门槛低于 PMEA，可带审稿意见转 | 基本不用改，走 transfer 服务 |
| 5 | **Sensors**（MDPI, OA） | 对阴性/复现类结果最宽容，周期最短 | MDPI 模板；APC 约 CHF 2 600 |

> 不建议再投 J-BHI / TIM：同一编辑部已两次拒稿，且两刊的偏好与本文"阴性 + 协议审计"的写法冲突。

---

## 八、这次换刊相对旧稿改了什么（应对"是否一稿多投"的质疑）

| 维度 | J-BHI / TIM 旧稿 | PMEA 新稿 |
|---|---|---|
| 主张 | "我们能做跨物种识别，90.9%" | "跨物种可分性不超过同物种跨平台（采集条件）可分性" |
| 验证 | 窗口级随机交叉验证 | P0/P1/P2 三级泄漏阶梯 + 20 次随机划分 |
| 对照 | 无 | NC1 跨平台（110 人 60 GHz vs 24 人 120 GHz，同时换了受试者与协议）/ NC2 状态 / NC3 生理 |
| 采样率 | 未处理 | 50→10 Hz 抽取对照，逐特征漂移量化 |
| 迁移 | 无 | 双向迁移实验（96.3% vs 24.1%） |
| 年龄 | 作为正面结果 | 作为阴性结果如实报告（R² = −0.15） |
| 代码 | 部分、未测试 | 完整、39 个测试、结果表与论文一一对应 |

结论：**不是旧稿改格式，是换了问题、换了证据链**。cover letter 已写明这一点。
