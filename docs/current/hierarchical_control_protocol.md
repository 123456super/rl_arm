# 分层控制通用实验协议 v1

> 更新时间：2026-08-20。状态：frozen baseline、revised v2 与 recovery-gated S1 nominal 均已有统计；recovery-gated Residual 训练暂缓。本文是 S1/S2/S3 的共同上位协议；阶段文档只能补充场景和实验矩阵，不能覆盖本文的动作、安全与评估语义。

## 1. 冻结控制链

```text
q_goal candidates = multi-start IK(goal)
q_ref path         = direct check or bidirectional RRT-Connect
qdot_nominal       = waypoint tracker or terminal DLS servo
qdot_candidate     = qdot_nominal
                   + residual_budget(h) * residual_scale * actor_action
qdot_cmd           = predictive safety filter(qdot_candidate)
```

当前有三个明确版本：`hierarchical_s1_nominal_final_v1`（旧 frozen baseline）、`hierarchical_s1_revised_v2`（只改 IK/planner/tracker）和 `hierarchical_s1_revised_recovery_gated_v1`（在 revised v2 上限制 recovery 只在 AVOID_HOLD 生效，SERVO 不被 recovery 覆盖）。recovery-gated 不改变任务目标分布、障碍几何、`d_safe`、`0.055 m` success tolerance、horizon、状态机或 actor residual 语义。

actor action 始终是有界 residual，而不是完整关节速度。静态和动态使用相同语义。旧 direct actor、legacy Cartesian residual actor 和旧 replay 与当前链路不兼容，必须由 signature 检查拒绝。

## 2. 规划与有效性检查

- reset 使用当前 `q`、目标 TCP 和当前障碍物快照规划。
- 多初值 IK 候选必须满足关节限位、目标误差阈值和状态有效性。
- 先检查起点到候选终点的直连；失败才运行双向 RRT-Connect。
- 节点和边统一检查 joint limit、TCP workspace、全连杆 capsule clearance、障碍物接触和 self-collision。
- PyBullet 检查必须保存并恢复真实 `q/qdot`，不能污染正在执行的 episode。
- `ik_not_found` 与 `plan_not_found` 只表示给定搜索预算内未找到，不得写成绝对不可达。
- revised v2 可使用结构化 rest poses、位置 DLS fallback、较宽的内部 IK 候选误差（最终 success 判定仍为 `0.055 m`），并在多个合法候选之间按路径最小 clearance 优先、长度次优选择。
- tracker 可使用更早 terminal servo、adaptive damping、关节限位 null-space 和 clearance-aware gain；所有命令仍经过同一 safety filter。

## 3. 共享状态机

| 状态 | 控制职责 | 进入/退出含义 |
| --- | --- | --- |
| `TRACK` | waypoint tracker 加有界 Residual | 正常路径执行 |
| `AVOID_HOLD` | 暂停任务轨迹，由预测 QP 主动增加危险连杆间隙 | 短时预测裕度低 |
| `REPLAN` | 从当前状态按最新障碍物快照重规划 | 冲突持续或旧路径失效 |
| `SERVO` | terminal DLS 控制 TCP 误差 | 到达最后 waypoint 的终端区域 |
| `PLAN_FAILED` | 零 nominal，按固定周期重新尝试 | IK 或规划在预算内未找到 |

`AVOID_HOLD` 表示暂停任务推进，不是关节被动零速。严格约束不可行时，恢复 QP 只放松冲突的预测连杆 barrier，仍保留 joint/workspace 硬约束；该状态必须记为 `RECOVERY_RELAXED`，不能称为严格安全命令。

在 `hierarchical_s1_revised_recovery_gated_v1` 中，主动 recovery 只允许在 `AVOID_HOLD` 生效；进入 `SERVO` 后不再用 recovery escape 覆盖 terminal DLS，SERVO nominal 仍经过同一严格 predictive safety filter。该 gating 不改变任务定义或安全约束。

若后续增加显式局部轨迹优化，只能作为 `AVOID_HOLD` 内的消融，不能改变 actor 动作语义、状态机接口或最终安全出口。

## 4. Residual 与安全出口

hierarchical 观测在历史基础观测之后追加：

- `qdot_nominal`：6 维；
- 下一 waypoint 关节误差：6 维；
- path progress：1 维；
- residual budget：1 维；
- 五状态 one-hot：5 维。

budget 随预测安全函数 `h` 从 `risk_start_m` 到 `risk_stop_m` 线性收紧。`AVOID_HOLD` 和 `PLAN_FAILED` 下 nominal 与 actor residual 均为零；主动 escape 只由最终预测 QP 产生。任意 nominal/Residual 组合后仍必须经过同一个 safety filter，RL 不得绕过安全出口。

## 5. 固定实验阶段

| 阶段 | canonical 配置 | 障碍运动 | Residual | 研究目的 |
| --- | --- | --- | --- | --- |
| S1-N | `hierarchical/s1_static.yaml --nominal-only` | 静态 | 零 | 验证规划、跟踪和 servo 基座 |
| S1-R | `hierarchical/s1_static.yaml` | 静态 | 训练 | 验证 Residual 静态增益 |
| S2-N | `hierarchical/s2_dynamic.yaml --nominal-only` | 动态 | 零 | 验证 hold/replan/filter 动态基座 |
| S2-R | `hierarchical/s2_dynamic.yaml` | 动态 | 训练 | 验证完整方法 |
| S3 | `hierarchical/s3_robust_dynamic.yaml` | 动态加误差/时延 | 训练 | 验证鲁棒性与方法边界 |

配置继承只能修改该阶段明确声明的场景字段。阶段间不得静默改变 success tolerance、episode horizon、capsule、manifest、观测排列或 actor action semantics。

## 6. 通用训练与评估规则

- 每个正式学习方法至少使用 3 个独立训练 seed。
- checkpoint 只按 validation manifest 选择；final manifest 不参与训练、curriculum 或选择。
- 正式结果同时报告各 seed、pooled 结果、完整 manifest 和 plan-conditioned 口径；所有比例保留分子/分母。
- 共同指标包括 IK/plan found、direct/RRT 比例、规划耗时、成功、timeout、final error、完成时间、路径进度、碰撞、最小预测 `h`、filter 事件、状态占比、replan 次数以及 nominal/residual/filter intervention 范数。
- 成功必须同时满足目标误差与安全间隙条件。smoke 或单 episode 不能替代正式统计。
- 每项消融一次只移除或替换一个机制；zero residual 是所有 Residual 结论的必要成对基线。

具体阶段的场景参数、失败分类、表格和命令由阶段协议规定。S1 见[S1 静态障碍物协议](s1_static_obstacle_protocol.md)。

## 7. 版本与止损规则

- actor/reward/cost signature 必须覆盖配置、观测/动作版本、风险表示、安全过滤、机器人关节和 capsule。
- S1 actor 必须从新 hierarchical actor 随机初始化；不得从 legacy actor/replay resume。
- planner/tracker 配置改变会产生新的 architecture/actor signature。旧 baseline actor、replay 和 revised v2 不得混用；旧输出目录必须冻结保留。
- S2 可以从同语义的 hierarchical S1 actor 迁移，但必须保留 signature 并在论文中声明迁移关系。
- plan-conditioned success 未达阶段门槛时，先修 IK、planner、tracker 或 servo，不训练 RL。
- nominal 存在大量物理接触时，先修 validity check 或 safety filter，不用 reward 掩盖。
- Residual 不优于 zero residual 时，保留为消融结论并检查观测、预算和训练协议，不替换总体架构。
- dynamic 性能不足时，只在同一状态机内调整预测窗口、hold/replan 条件、训练分布和 Residual 参数。
