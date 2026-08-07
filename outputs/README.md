# 实验输出目录

旧输出保持只读，不移动、不覆盖，以保存配置中的相对路径和 checkpoint provenance。新实验必须使用新的日期化目录。

| 分层 | 目录 | 状态 |
| --- | --- | --- |
| 阶段一正式证据 | `rechecks/heldout_1004_1006/final_3methods/` | 已冻结，可引用 |
| 阶段一训练与历史结果 | `formal/`、`rechecks/link_fixed_penalty1/`、`sweeps/` | 历史复现/参数筛选 |
| P1/P2 开发 | `p2_dev/` | 开发验证 |
| P3 B1--B5 开发 | `p3_dev/`、`p3_dev_100k/` | 修正前开发数据 |
| P3 修正后开发 | `p3_postfix_dev/`、`p3_postfix_dev_100k/` | 已完成开发与冻结审计；不得与旧开发数据混合 |
| P3 失效诊断 | `p3_diagnostics/`、`p3_geometry_fix/` | 修正前根因证据 |
| 论文生成材料 | `paper/` | 派生文件，不是原始数据 |
| 部署预检 | `deployment_preflight/` | 阶段一离线预检 |
| 新统一实验 | `p3_unified_geometry/` | 未作为当前主结果目录；结果归档于 `p3_postfix_dev*` |
| 独立基础 reaching | `reaching_recovery_v1/`、`reaching_recovery_v2/` | v2 已冻结为无障碍基础基线；全量 582/600（97.0%），固定可达子集 582/582（100%） |

2026-08-03 前的 P3 输出采用旧连杆速度/碰撞口径，只能用于失效分析。2026-08-03 后的结果目录保存 `config.json`、checkpoint 选择 CSV、逐 episode metrics 和三类碰撞事件字段；冻结决策及其证据索引见 `p3_postfix_dev_100k/p3_freeze_decision.json`。

`reaching_recovery_v2/` 与 P3/VAPS 安全方法分开统计：它关闭动态障碍物和安全过滤器，只证明基础 reaching 执行链路；最终报告必须同时给出全量和固定可达条件口径，不得将条件 `100%` 写成全分布 `99%`。
