# 项目文档入口

从本页开始阅读。文档按职责分为当前状态、研究设计、参考资料和历史归档；实验输出目录保持原位，不随文档重组移动。

## 当前文档

1. [当前研究状态](current/research_status.md)：研究方向、完成度、阻塞项和下一步。
2. [结果总表](current/results_summary.md)：哪些数据可以引用、哪些只能用于诊断。
3. [P3 冻结后状态](current/next_experiment_protocol.md)：修正单位与碰撞口径后的固定口径及冻结后约束。
4. [基础 reaching 恢复协议](current/reaching_recovery_protocol.md)：独立无障碍 reaching v2 基线、全量/条件口径和冻结范围。
5. [后续渐进式实验协议](current/successor_incremental_experiment_protocol.md)：以 reaching v2 为基线、逐级增加障碍物和安全机制的实验步骤与停止规则。
6. [指标和单位定义](reference/metrics_and_units.md)：速度、风险裕度和碰撞字段的统一定义。
7. [研究方向](design/research_direction.md)：新论文的研究问题、方法边界与长期实验矩阵。
8. [输出目录说明](../outputs/README.md)：磁盘结果的分层和可信度。
9. [后继安全协议](design/successor_safety_protocol.md)：P3 冻结后可行安全集方向的独立安全目标、评估协议和审查门槛；G0/G1 已完成，当前仍不授权安全方法 V2/G3/G4。

当前阅读顺序：先看[当前研究状态](current/research_status.md)了解进展与边界，再看[结果总表](current/results_summary.md)核对数字，随后看[基础 reaching 恢复协议](current/reaching_recovery_protocol.md)确认无障碍基线，最后看[P3 冻结后状态](current/next_experiment_protocol.md)确认安全实验边界。

## 文档职责

| 文件 | 职责 | 状态 |
| --- | --- | --- |
| `current/research_status.md` | 当前唯一状态页 | 现行 |
| `current/results_summary.md` | 当前唯一结果结论页 | 现行 |
| `current/reaching_recovery_protocol.md` | 独立无障碍 reaching 基线与冻结边界 | 现行 |
| `current/successor_incremental_experiment_protocol.md` | 以后续 reaching v2 为基线的渐进式实验步骤 | 现行 |
| `design/research_direction.md` | 新论文研究设计与长期实验矩阵 | 现行设计，不记录流水账 |
| `reference/metrics_and_units.md` | 指标、单位和碰撞字段定义 | 现行参考 |
| `reference/training_guide.md` | 历史命令与运行说明 | 参考；新实验以当前协议为准 |
| `archive/stage1/` | 阶段一论文大纲、结论、材料和部署预检 | 已冻结，只读归档 |
| `archive/p3_pre_20260803/` | 单位和碰撞口径修正前的 P3 时间线与诊断 | 已冻结，只读归档 |

## 维护规则

- 状态变化只修改 `current/research_status.md`，结果变化只修改 `current/results_summary.md`。
- 基础 reaching v1/v2 的结果和冻结边界以 `current/reaching_recovery_protocol.md` 为准；摘要页只引用其统一数字。
- 新实验严格按 `current/next_experiment_protocol.md` 执行；归档文件不再追加当前状态。
- 论文数字只从 `current/results_summary.md` 指向的 CSV 引用。
- 旧输出不移动、不覆盖；新输出写入新的日期化目录。
- `collision` 仅作为兼容字段使用；新报告必须写全 `collision_capsule_overlap`、`collision_pybullet_contact` 和 `termination_collision`。
