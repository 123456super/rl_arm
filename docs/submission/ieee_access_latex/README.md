# IEEE Access 投稿与 LaTeX 写作资料包

这个文件夹用于准备本项目的 IEEE Access 期刊论文。目标是把“投稿要求、格式要求、写作结构、LaTeX 文件组织”集中放在一起，后续写论文时不用反复去官网翻页面。

## 推荐阅读顺序

1. `ieee_access_requirements_zh.md`
   - IEEE Access 官方投稿要求的中文整理。
   - 包括模板、页数、关键词、作者简介、AI 披露、补充材料、审稿方式、录用后文件等。

2. `paper_structure_for_ur5_rl_obstacle_avoidance.md`
   - 专门针对“UR5 机械臂 + 强化学习 + 动态障碍物避障 + 风险建模”的论文大纲。
   - 可以直接作为 Introduction、Method、Experiments、Results 的写作蓝图。

3. `submission_checklist_zh.md`
   - 投稿前逐项核对用。
   - 建议在正式提交前逐条打勾，特别是 LaTeX/PDF 一致性、作者信息、关键词、图表引用、语法检查。

4. `sources.md`
   - 本资料包使用的官方来源链接。
   - 以后如果老师问“这个要求从哪来的”，优先查这里。

5. `paper/main.tex`
   - LaTeX 主文件骨架。
   - 注意：正式投稿必须使用 IEEE Access 官方 LaTeX 模板。请从 IEEE Access 官网下载模板后，把官方模板中的 `ieeeaccess.cls` 和相关样式文件放到 `paper/` 目录，或直接把本项目内容迁移到官方模板。

6. `download_official_template_zh.md`
   - IEEE Access 官方 LaTeX 模板的下载、解压、放置和编译方法。

## 对你这篇论文的投稿定位

建议投稿类型选择：

- 首选：`Research Article`
- 备选：`Applied Research`

你的论文不适合写成 Survey 或 Tutorial。IEEE Access 更看重“原创、技术正确、实验充分、表达清楚”。因此这篇文章的重心应放在：

- 问题定义：动态障碍物环境下 UR5 机械臂避障控制。
- 方法贡献：风险感知奖励、连杆级距离/碰撞风险建模、SAC 训练策略、可选预测风险机制。
- 实验验证：成功率、碰撞率、最小安全距离、轨迹平滑性、到达误差、不同障碍物速度/密度下的鲁棒性。
- 对比实验：普通 SAC、无风险项 SAC、只用末端距离风险、人工势场或传统避障方法。
- 消融实验：去掉风险惩罚、去掉预测风险、改变安全距离阈值、改变风险权重。

## 时间提醒

你要求论文在 2027 年 5 月前完成 EI 检索。IEEE Access 官方强调投稿到在线发表通常为 4-6 周，但 EI/Scopus/Web of Science 等数据库收录还需要额外处理时间。保险做法是尽量在 2026 年底或 2027 年 1 月前完成投稿。
