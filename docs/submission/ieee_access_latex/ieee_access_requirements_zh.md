# IEEE Access 官方要求中文整理

整理日期：2026-09-08

## 基本定位

IEEE Access 是 IEEE 旗下的 fully open access、online-only、多学科期刊，覆盖 IEEE fields of interest，强调应用导向、跨学科、技术正确、表达清楚的原创研究。官方页面给出的特点包括：

- 投稿到在线发表通常为 4-6 周。
- 连续出版。
- 二元审稿决定：Accept 或 Reject。
- 平均录用率约 20%。
- 2025 JCR 影响因子为 4.2。
- 被 Ei Compendex、SCIE、Scopus、Inspec 等数据库收录。

## 稿件格式要求

正式投稿时必须满足：

- 使用 IEEE Access 官方模板。
- 双栏、单倍行距。
- 同时提交 LaTeX/Word 源文件和 PDF 文件。
- 源文件和 PDF 内容必须完全一致。
- 文件大小不超过 40 MB。
- 所有作者必须同时出现在源文件和 PDF 中。
- 每位作者都必须提供 short biography，放在参考文献之后。
- 至少选择 3 个关键词，最多 10 个关键词。
- IEEE Access 没有硬性页数上限，也没有超页费，但官方强烈建议控制在 20 页以内；超过 20 页可能拖慢审稿，特殊长文应先向主编预咨询。

## 摘要与关键词

摘要建议写成单段，避免编号公式和编号引用。摘要应覆盖：

- 研究背景和痛点。
- 本文提出的方法。
- 关键实验设置。
- 最重要的量化结果。
- 方法意义。

关键词建议从下面挑选并调整：

- Robot motion planning
- Reinforcement learning
- Soft actor-critic
- Robotic manipulators
- Obstacle avoidance
- Risk-aware control
- Collision avoidance
- Dynamic environments
- UR5 manipulator

## 内容质量要求

IEEE Access 官方列出的接收标准可理解为：

- 原创性：论文必须增强所在领域已有知识。
- 未重复发表：结果不能同时投给其他期刊；会议扩展版和预印本可以，但要披露。
- 技术质量：实验、统计和分析必须达到较高技术标准，并且描述足够详细。
- 结论支撑：结论必须由数据支持。
- 英文质量：英文语法必须标准；语法差可能直接拒稿。
- 参考文献：必须引用相关、准确、未撤稿的已有工作。
- 范围匹配：主题必须属于 IEEE Access 范围。

## 对本项目最合适的稿件类型

建议选择 `Research Article`。如果投稿系统要求更细，也可以考虑 `Applied Research`。

理由：

- 本项目有明确问题：UR5 机械臂在动态障碍物环境下的避障控制。
- 有方法：风险感知 SAC、连杆级碰撞风险、动态障碍物建模。
- 有实验：仿真训练、对比、消融、鲁棒性测试。
- 有结果：成功率、碰撞率、最小距离、回报、轨迹指标等。

## AI 使用披露

IEEE Access 要求披露 AI 生成内容。若使用 ChatGPT/Codex 辅助润色、翻译、代码解释或文本草拟，建议在 Acknowledgment 或单独声明中写清楚：

- 使用了什么 AI 工具。
- 用于哪些环节。
- 作者对最终内容、数据、引用和结论负责。

如果只是语法润色，官方说这类用法较常见，但仍建议披露。

可用英文占位句：

```tex
The authors used OpenAI's ChatGPT/Codex to assist with language polishing and code understanding during manuscript preparation. The authors reviewed and verified all AI-assisted content and take full responsibility for the final manuscript.
```

## 补充材料

IEEE Access 允许提交补充材料，例如：

- 代码。
- 数据。
- 训练配置。
- 附加实验结果。
- 视频。

视频最大 100 MB，必须在投稿时一起提交并参与同行评审。机械臂避障论文很适合准备一个短视频，展示：

- 随机动态障碍物场景。
- 普通 SAC 和风险感知 SAC 的对比。
- 成功避障轨迹。
- 失败案例或极端场景。

## 审稿流程

IEEE Access 官方流程包括：

1. 投稿要求检查：格式、语法、模板等。
2. 完整性检查：查重、禁止投稿名单、重复投稿等。
3. 范围与质量检查：编辑判断是否符合 IEEE Access 范围，是否有足够技术增量。
4. 分配 Associate Editor 和审稿人。
5. 决定通知。

审稿特点：

- 单盲审稿。
- 至少 2 名独立审稿人。
- 二元决定：Accept 或 Reject。
- 如果 Reject 但允许重投，通常只能重投一次，并且要提交逐条回复和修改说明。

## 录用后要求

录用后通常需要：

- 提交最终 LaTeX/Word 源文件。
- 提交最终 PDF，文件名要求为 `FINAL Article.pdf`。
- 提交图形摘要 Graphical Abstract，推荐尺寸 660 x 295，JPG，小于 45 KB。
- 如有视频，提交视频和封面图。
- 完成电子版权表。
- 支付 APC 或走学校/机构 OA 协议。

Early Access：

- 官方说明最终文件提交后，accepted version 通常 2-3 天内上线 IEEE Xplore Early Access。
- Early Access 上线后即视为发表，不能撤稿。

## APC 与学校协议

IEEE Access 当前官方 APC 为 2160 美元/篇，另加可能适用的本地税费，无页数上限、无超页费。

折扣：

- IEEE 普通会员通讯作者：5%。
- IEEE Society/Council 会员通讯作者：20%。
- 学生会员折扣通常不适用。
- 上海大学在 IEEE Open 的机构 OA 协议页面中有单独列出，投稿前应向上海大学图书馆确认是否有 APC 减免、免除名额或学校预付账户。

