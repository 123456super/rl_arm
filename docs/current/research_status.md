# 当前研究状态

> 更新时间：2026-08-20。本页是当前唯一状态入口，只记录决策、进度、下一步和结论边界。实现与实验规则见[分层控制实验协议](hierarchical_control_protocol.md)，S1 执行细节见[S1 静态障碍物协议](s1_static_obstacle_protocol.md)。

## 1. 当前决策

论文主架构的 frozen baseline 是 `hierarchical_s1_nominal_final_v1`。在该基线上已完成不改变任务定义的 IK/planner/tracker 改进分支 `hierarchical_s1_revised_v2`：结构化 IK 初值与 DLS fallback、多候选路径按 clearance/length 选择、起点边界容差、adaptive terminal DLS、关节中心性与 clearance-aware tracker。两者仍共用 `TRACK / AVOID_HOLD / REPLAN / SERVO / PLAN_FAILED` 状态机、最终 success 判定和唯一 safety filter 出口。

旧 direct-SAC 不再继续进行参数、训练步数或 hard-case curriculum 修补。历史证据支持改变职责划分，但不等于新方法已经取得更高性能。完整论证和结果来源见：

- [架构转向证据](../archive/legacy_direct_sac/evidence/architecture_transition_evidence.md)；
- [结果来源与校验](../archive/legacy_direct_sac/evidence/result_provenance.md)。

## 2. 当前完成度

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| hierarchical 规划、跟踪、servo 和状态机 | 已实现 | frozen baseline、revised v2 与 recovery-gated nominal 候选均已完成统计 |
| risk-conditioned residual budget | 已实现 | actor 只输出有界 residual |
| predictive safety filter 与主动 clearance recovery | 已实现 | 所有候选命令共用唯一安全出口 |
| actor/reward/cost signature | 已实现 | 拒绝 legacy actor/replay 混用 |
| canonical S1/S2/S3 配置 | 已实现 | 位于 `configs/experiments/hierarchical/` |
| 原 S1-N/S1-R final | 已完成并冻结 | pooled S1-R full success `529/600=88.17%`，plan-conditioned `529/555=95.32%`，collision `0/600` |
| revised v2 hard-case/nominal-only | 已完成 | `178/200=89.0%` full，`178/189=94.18%` plan-conditioned，未过门槛 |
| recovery-gated nominal-only | 已完成 | `187/200=93.5%` full，`187/189=98.94%` plan-conditioned，collision `0`，门槛通过；结果见 `outputs/hierarchical/s1_static_revised_recovery_gated/` |
| recovery-gated Residual 三训练 seed | 暂缓 | 用户当前决定先不训练；保留新 signature 与 nominal 结果，等待 paired comparison/下一步指示 |
| S2 dynamic 与 S3 robust final | 待运行 | S1 消融完成后执行 |

静态和动态各一次 nominal-only 单 episode smoke 已成功且无碰撞；它们只证明链路可运行，不是论文统计结果。

## 3. 冻结历史基线

- S0 无障碍 direct-SAC：固定 final manifest 上 `582/600=97.0%`，有限 IK 候选子集 `582/582=100%`。checkpoint 与适用边界见[S0 基础 reaching 历史基线](../archive/legacy_direct_sac/evidence/s0_reaching_baseline.md)。
- S1 静态 direct-SAC：历史最好记录为 `521/600=86.83%`，当前磁盘复跑为 `511/600=85.17%`；两者来源不同，不得合并。R3/R4 blind 为 `505/600=84.17%`，但 actor 身份和恢复阶段使其不能作为 curriculum 因果结论。

以上结果只作为架构调整前的外部历史对照，不属于 hierarchical 方法结果。

## 4. 当前进展与实验门槛

旧结果必须保留在 `outputs/hierarchical/s1_static/`，不得被 revised 输出覆盖。revised v2 输出使用 `outputs/hierarchical/s1_static_revised/`，recovery-gated 输出使用 `outputs/hierarchical/s1_static_revised_recovery_gated/`；两者均固定同一 `v1_final.json`，因此可以逐 reset paired 比较。planner/tracker/recovery-gating 配置变化会改变 actor signature；旧 actor、旧 replay 不得用于 gated 训练或评估。

## 5. 当前下一步

revised v2 nominal 已完成：IK found `189/200=94.5%`，plan found `189/200=94.5%`，full success `178/200=89.0%`，plan-conditioned `178/189=94.18%`，collision `0/200`。相对 frozen v1 的单 200-reset nominal，full 仅增加 1 个成功；plan-conditioned 从 `177/185=95.68%` 降至 `178/189=94.18%`，因此 nominal gate 未通过。

paired 结果为：old success→new success `175`，old success→new failure `2`，old failure→new success `3`，old failure→new failure `20`。revised v2 修复了 `9136`、`9137`、`9138`，但新增 `9122`、`9158` 两个回归。

失败结构已从 IK/plan 为主转为 terminal/filter：`IK_NOT_FOUND=11`、`SERVO_TIMEOUT=3`、`FILTER_STOP_TIMEOUT=8`，无 `PLAN_NOT_FOUND`。候选数均值从 `14.29` 增至 `59.61`，但初始规划时间从 `0.048s` 增至 `1.239s`，safety-filter intervention 从 `39.85%` 增至 `49.37%`。因此下一步只做 tracker/filter-aware 诊断，不训练 Residual。

当前已接入独立的 `hierarchical_s1_revised_tracker_v3` filter-aware terminal servo：利用上一周期 filter intervention ratio 和实际过滤命令，降低被截断时的 servo 增益、加入 TCP 速度阻尼和近目标 gain taper，并保留原 safety filter 作为唯一命令出口。v3 使用独立 signature/output，尚未正式评估，不能预先声称成功率提升。

v3 hard-case 已完成：`3/23=13.04%` success，与 v2 `3/23` 持平；`IK_NOT_FOUND=11` 未变，`FILTER_STOP_TIMEOUT=9`，`SERVO_TIMEOUT=0`，但没有新增成功。v3 只降低了平均干预和误差，未解决 terminal 收敛。

v4 hard-case 也已完成：`3/23=13.04%` success，`IK_NOT_FOUND=11`、`PLAN_NOT_FOUND=0` 未变；`FILTER_STOP_TIMEOUT=8`、`SERVO_TIMEOUT=1`。有限 terminal-stall replan 已触发，平均 replan 从 `0.96` 增至 `1.74`，平均总规划时间从 `2.43s` 增至 `3.02s`，但没有新增成功；平均最终误差仅从 `0.3579m` 降至 `0.3565m`。因此继续增加 replan 不具备收益证据，v4 不进入完整 200 episodes，也不训练 Residual。

最新 terminal task-space 诊断已完成（`outputs/hierarchical/s1_static_revised/terminal_diagnostics_v2.json`）：SERVO 中 nominal 朝目标比例为 `1.0`，nominal 朝目标速度均值 `0.430 m/s`；但 recovery command 占 SERVO 步数 `47.66%`，其接管时平均目标方向执行速度为负，整体执行速度降至 `0.055 m/s`，误差正向进展率仅 `64.16%`。这排除了 terminal DLS/Jacobian 方向错误，当前首要瓶颈是主动 recovery 过早接管与目标推进冲突。

两个 hard-case 对照完成：关闭 recovery 为 `8/23=34.78%`，相对 revised v2 修复 reset `2,16,17,19,20`，没有成功退化；仅延后 recovery 为 `5/23`，只修复 `19,20`。正式候选采用 recovery gating：`AVOID_HOLD` 保留 recovery，`SERVO` 禁止 recovery 覆盖 nominal DLS，命令仍经过同一严格 safety filter。该候选需先完成完整 200-reset nominal gate。

recovery-gated hard-case 已完成：`10/23=43.48%`，相对 revised v2 修复 reset `2,13,15,16,17,19,20`，相对 filter-only 再修复 `13,15`，无 success→failure、无碰撞、无 `SERVO_TIMEOUT`；剩余 `IK_NOT_FOUND=11`、`FILTER_STOP_TIMEOUT=2`。这两个 `FILTER_STOP_TIMEOUT` 是 AVOID_HOLD/时限主类，实际 safety-filter safe-stop 为 `0`。plan-conditioned hard-case 为 `10/12=83.33%`，说明 gating 显著改善 terminal 收敛。

recovery-gated 200-reset nominal 已完成并通过门槛：`187/200=93.5%` full success、`187/189=98.94%` plan-conditioned success、IK/plan found 均为 `189/200=94.5%`、timeout `2/200=1.0%`、collision `0/200`。相对 revised v2，full 增加 9 个成功，plan-conditioned 增加 9 个成功；相对 frozen v1，full 增加 10 个成功，但 plan/IK 分母不同，不能直接视为严格 paired 改善。当前不启动 Residual 训练，先冻结该 nominal 结果并保留旧输出。

下一步不训练 Residual。先完成 recovery-gated 与 frozen/revised nominal 的逐 reset paired comparison，并明确该版本是否作为新的 S1 nominal 基座；该改动不放宽 barrier、几何、success tolerance 或 horizon。

- plan-conditioned success `<95%`：只定位 IK、planner、tracker、servo 或 safety filter，不训练 Residual，也不更换总体架构。
- plan-conditioned success `>=95%`：满足 nominal gate，但是否立即训练由当前实验决策单独确定；本轮用户已明确暂缓 Residual 训练。
- S1 消融完成：使用相同控制层次进入 S2 dynamic，再进行 S3 速度、时延、观测误差和几何裕量实验。

## 6. revised v2 之后的固定流程

1. hard-case diagnostic；若 IK/plan/terminal timeout 没有改善，停止训练并回到模块归因。
2. revised nominal-only 200 final；已保存 CSV、summary 和 trace。
3. recovery-gated plan-conditioned 为 `98.94%`，collision 为 `0`，nominal gate 已通过；但当前明确暂缓 Residual 训练，不运行 4401/4402/4403。
4. 若后续获准训练，必须使用 recovery-gated 新 config/signature，从随机初始化开始，并与该 zero-residual nominal 成对比较。
5. 每个新 seed 按 validation manifest 选 checkpoint，再用同一 final manifest 评估；不得加载旧 actor/replay。
6. 报告 old/new nominal 与 old/new S1-R 的 pooled、逐 seed 及逐 reset paired 转移：old failure→new success、old success→new failure。

## 7. 当前不能声称

- recovery-gated nominal 已超过历史 direct-SAC `86.83%`，但这仍是 zero-residual nominal 统计，不等同于正式 S1-R final。
- 尚无 dynamic hierarchical final，不能声称动态性能已经良好。
- revised v2 未通过 gate；recovery-gated nominal 已通过 `98.94%` plan-conditioned gate，但 Residual 尚未训练，不能声称学习架构已提升。
- `RECOVERY_RELAXED` 不是严格 barrier 可行命令；当前没有整个混合状态机递归可行的形式化证明。
- 当前对象是单球障碍且使用仿真真值，不覆盖真实感知与真机闭环。
