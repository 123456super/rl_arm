# 当前研究状态

> 更新时间：2026-08-21。本页是当前唯一状态入口，只记录决策、进度、下一步和结论边界。实现与实验规则见[分层控制实验协议](hierarchical_control_protocol.md)，S1 执行细节见[S1 静态障碍物协议](s1_static_obstacle_protocol.md)。

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
| IK/predictive v1 hard-case | 已完成但未通过 | `9/23=39.13%` full，`9/12=75.00%` plan-conditioned，collision `0`；没有修复任何 IK failure |
| IK/predictive v2 hard-case | 已完成，达到最低回归门槛但无增益 | `10/23=43.48%` full，`10/12=83.33%` plan-conditioned，collision `0`；相对 recovery-gated 为 10→10，逐 reset 无修复也无回归 |
| IK target-shell v1 hard-case | 已完成但不采用 | `10/23=43.48%` full，`10/13=76.92%` plan-conditioned，collision `0`；IK found `13/23`，无成功增益，`9006` 仅由 IK failure 变为 filter timeout |
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

### 5.1 `hierarchical_s1_revised_ik_predictive_v1` hard-case 实验（2026-08-21）

本实验使用固定 manifest `v1_hierarchical_hard_cases.json` 的 23 个 reset，配置为 `configs/experiments/hierarchical/s1_static_revised_ik_predictive.yaml`，输出为 `outputs/hierarchical/s1_static_revised_ik_predictive/`。任务定义保持不变：`d_safe=0.12 m`、success tolerance `0.055 m`、20 Hz、12 s horizon、原状态机和唯一严格 safety-filter 出口。

过程设置：普通 IK 64 次失败后最多扩展至 256 次，使用 4 个 DLS fallback seeds；成功路径先按几何 clearance/length 取最多 8 条，再按 predictive safety-filter 的最小安全裕量和平均 intervention 排序。所有执行命令仍经过原 safety filter，不放宽几何或 barrier。

结果：full success `9/23=39.13%`，plan-conditioned success `9/12=75.00%`，IK found/plan found 均 `12/23=52.17%`，timeout `3/23=13.04%`，collision `0/23`。失败主类为 `IK_NOT_FOUND=11`、`FILTER_STOP_TIMEOUT=3`，`PLAN_NOT_FOUND=0`、`SERVO_TIMEOUT=0`。相对 recovery-gated hard-case 的 `10/23`、`10/12`，出现 seed `9060` 的 success→failure，且没有任何 IK failure→success。初始规划时间约 `1.075 s`、总规划时间约 `3.930 s`，高预算没有换来成功率收益。

结论：当前 IK 瓶颈不是简单的搜索次数不足；下一步必须统计 `joint_limit/workspace/obstacle_clearance/obstacle_contact/self_collision` 的拒绝原因。v1 predictive 实现还发现未实际完成 scoring 的候选曾参与排序，现已修复为未评分候选不参与 predictive 排序、全部评分失败时显式回退 geometric selector。v1 未通过 hard-case gate，不进入 200-reset nominal、不训练 Residual、不覆盖 recovery-gated 输出。

下一步不训练 Residual。`ik_predictive_v2` 已完成且只达到最低回归门槛，没有 paired 增益；因此不进入 200-reset nominal，转做独立的 `hierarchical_s1_revised_ik_feasibility_audit_v1`，区分 raw IK 目标不可达与障碍几何阻塞。

v2 已完成，达到上述最低 hard-case 回归门槛，但没有带来任何 paired 成功增益，因此不应直接进入 200-reset nominal。拒绝审计为：`goal_error=5760`、`obstacle_clearance=4624`、`obstacle_contact=1374`、`workspace=60`，而 `solver_error/short_solution/duplicate/joint_limit/self_collision=0`。逐 episode 审计显示 17 个 reset 至少有 raw IK 达到目标误差阈值，其中 5 个 reset 的所有目标候选最终被障碍 clearance/contact 拒绝；另外 6 个 reset 在当前 256+4 fallback 搜索中没有达到目标误差阈值。原汇总中的 `obstacle_free_goal_reachable_episodes=17` 只表示忽略障碍检查后的候选可达，不能解释为“没有障碍阻塞”；汇总脚本已改为同时报告 `raw_goal_unreachable`、`accepted_obstacle_valid_goal` 和 `obstacle_blocked_goal_reachable`。瓶颈已从搜索预算转为目标位姿可达性和障碍感知 IK。下一步不放宽 `d_safe`、success tolerance 或 horizon。

### 5.2 IK feasibility audit 与 target-shell 后续

feasibility audit 已确认：raw goal reachable `17/23`，raw goal unreachable `6/23`，最终接受 obstacle-valid IK 的 `12/23`，因此确有 `5/23` 属于“目标误差可达但所有候选被障碍几何拒绝”。这不是“无障碍也不可达”，而是目标容差球内的合法构型搜索不足。汇总脚本已修正字段含义，不再把 obstacle-free raw IK 误写成无障碍阻塞为零。

针对这 5 个 blocked reset，已完成 `hierarchical_s1_revised_ik_shell_v1`：仅当 exact-target IK 失败时，在原 success tolerance `0.055m` 内查询最多 32 个目标 shell 点，shell 半径 `0.04m`；候选仍以原始目标误差、原 `d_safe=0.12m`、原碰撞和原 success 判定检查。结果为 full `10/23`、plan-conditioned `10/13`、IK found `13/23`、timeout `3/23`、collision `0`；与 recovery-gated paired 为 success→success `10`、success→failure `0`、failure→success `0`、failure→failure `13`。seed `9006` 虽找到路径，但最终为 `FILTER_STOP_TIMEOUT`，因此 shell 不进入 nominal，也不再扩大 shell 搜索。

当前冻结回 recovery-gated：剩余 hard-case 失败为 `IK_NOT_FOUND=11` 和 `FILTER_STOP_TIMEOUT=2`；target-shell 不能解决终端推进问题。下一步只做 `9093/9143` 的 AVOID_HOLD/filter timeout 机制审计，不再继续 IK 变体、不训练 Residual。诊断只打开 `diagnostic_logging`，不改变控制参数、安全约束或成功判定。

### 5.3 严格 filter 目标推进可行性审计（已完成）

针对 `9093/9143` 的新诊断在每个 filter cycle 复用与线上严格 filter 完全相同的 joint velocity/acceleration/position box、workspace rows、predictive barrier rows、drift 和 barrier 条件，额外用线性规划只读求解：在这些约束下 TCP 当前目标方向的最大可实现速度。该最大值与实际 projected command 的目标方向速度同时写入 trace：

- `safety_filter_max_feasible_goal_velocity_mps`：严格约束下的最大目标方向 TCP 速度；
- `safety_filter_projected_goal_velocity_mps`：实际 filter 输出的目标方向 TCP 速度；
- `safety_filter_goal_velocity_feasibility_gap_mps`：二者差值；
- `safety_filter_goal_velocity_audit_status`：`optimal` 或明确的诊断失败状态。

这一步是只读诊断，不改变 safety filter 的 objective、`d_safe=0.12m`、success tolerance `0.055m`、horizon、recovery 或状态机。如果最大可行速度本身接近零，瓶颈是当前严格安全集合确实不允许足够的目标推进；如果最大可行速度明显为正而 projected 速度接近零，才有证据评估“可行集合内优先目标推进”的新 objective。新 objective 在此之前不进入 nominal，也不放宽任何 hard constraint。

此前 timeout trace（尚未包含本次新增 LP 字段）的结果为：`9093` 最终误差 `0.08961m`、SERVO `181` 步、recovery `17/240`、filtered `203/240`，平均执行目标方向速度 `0.01170m/s`、平均过滤损失 `0.30431m/s`；`9143` 最终误差 `0.06442m`、SERVO `211` 步、recovery `7/240`、filtered `235/240`，平均执行目标方向速度 `0.01240m/s`、平均过滤损失 `0.15200m/s`。两者 safe-stop 均为 `0`，OSQP 均为 `optimal`，因此失败不是求解不可行或 recovery 过早接管，而是持续投影后的目标推进不足。新增 LP 审计必须在两个 seed 完整重跑后才作为正式判断依据。

两个 seed 的完整 LP 审计已完成。`9093` 的最大可行目标方向速度均值/最小值为 `0.15518/0.01436 m/s`，实际投影均值为 `0.03266 m/s`，feasibility gap 为 `0.12252 m/s`，正最大可行速度比例为 `100%`；`9143` 对应为 `0.10021/-0.03521 m/s`、`0.01958 m/s`、`0.08064 m/s`、`98.75%`。两者 audit status 均为 `optimal`、safe-stop 均为 `0`。因此已确认：当前严格安全集合大多数时间仍允许目标推进，瓶颈是原始最小 joint-space intervention objective，而不是约束集合本身不可行。

基于该证据新增独立诊断配置 `s1_static_revised_recovery_gated_goal_velocity_priority_diag.yaml`。它只在严格同一约束集合内把 QP objective 改为“最大化 TCP 当前目标方向速度 + 极小的 requested-command 二次正则”，不改 `d_safe`、joint/workspace/predictive barrier rows、recovery、状态机、horizon 或 success rule；recovery-gated nominal 输出不覆盖。下一步只对 `9093/9143` 做 paired 对照，若无新增成功或出现碰撞/安全违规，立即停止该 objective 分支并冻结 recovery-gated。

首次 paired 对照（未做 QP 数值缩放）得到 `1/2`：`9093` 成功，`9143` 仍 timeout。`9143` 末段的最大可行目标速度约 `0.071m/s`，但 OSQP 多次返回 `maximum iterations reached`，随后回退到旧迭代投影，实际目标速度约 `0.0003m/s`；这不是新的控制结论，而是 objective 数值尺度问题。现已加入 `goal_velocity_objective_scale=10000`，同时按比例缩放线性项和二次项，保持数学 objective 不变，只改善 OSQP 数值条件。单 seed smoke（`9143`）已成功且无碰撞，末段 projected 目标速度约 `0.391m/s`；需重新完成两个 seed 的 paired 对照后，才决定是否保留该分支。

数值缩放后的完整重跑仍为 `1/2`：`9093` 成功，`9143` 在 `230/240` 步再次出现 `maximum iterations reached`，说明 OSQP 缩放不足以稳定解决近线性目标。现已改为严格约束下的直接 HiGHS LP：最大化 TCP 当前目标方向速度；LP 失败才退回 OSQP/原投影。该变更不放宽任何 hard constraint。`9143` 单 seed smoke 已达到成功、`31/31` 次 LP optimal、无碰撞和 safe-stop，平均执行目标方向速度约 `0.284m/s`。下一步重新跑两个 seed；该分支仍不进入 nominal final。

HiGHS LP 版本的两个 seed 完整 paired 结果为 `2/2=100%`：`9093` 最终误差 `0.05430m`、`9143` `0.05401m`，均成功、无 timeout、无碰撞、无 safe-stop。LP 实际目标方向速度与最大可行值几乎一致：`9093` 为 `0.28384/0.28384m/s`、gap `2.1e-9m/s`；`9143` 为 `0.13964/0.13964m/s`、gap `2.0e-9m/s`。但线上 filter 平均耗时分别约 `30ms/68ms`，最大峰值约 `0.47s/1.44s`，因此该实现暂列“成功率候选、实时性未达标”，不替换 recovery-gated nominal，也不运行 Residual 训练。下一步扩展至完整 23 个 hard cases，同时保留 filter latency、collision、safe-stop 和 success 作为联合 gate。

### 5.4 严格目标速度优先 hard-case 扩展（2026-08-21）

使用配置 `configs/experiments/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_hard_cases_v1.yaml`，固定 `v1_hierarchical_hard_cases.json` 的 23 个 reset。除 filter 可行集合内的目标函数外，任务定义完全继承 recovery-gated：`d_safe=0.12m`、success tolerance `0.055m`、预测时域、状态机、recovery 语义、碰撞规则和唯一 safety-filter 出口均未改变。结果为 full success `11/23=47.83%`、plan-conditioned `11/12=91.67%`、IK/plan found `12/23=52.17%`、timeout `1/23=4.35%`（seed `9027`）、collision `0/23`。相对 recovery-gated hard-case `10/23`，修复 `9093、9143`，但回归 `9027`，不能替换已通过 nominal gate 的 recovery-gated 基座。

实验过程设置：使用配置 `configs/experiments/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_hard_cases_v1.yaml`、固定 manifest `configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json`，4 个 manifest shard 并行运行，随后通过 `merge_hierarchical_evaluations.py` 合并。原始 shard、合并 CSV、summary 和 traces 分别保存在 `outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_hard_cases_v1/`；未覆盖 recovery-gated 旧输出。所有 23 个 reset 均为 nominal-only、zero residual。HiGHS LP 仅在既有 strict feasible set 内工作，LP 失败才回退既有 projection；没有修改 `d_safe`、success tolerance、prediction horizon、joint/workspace/predictive barrier、recovery、状态机或碰撞判定。

机制结果：direct path `11/12=91.67%`、RRT path `1/12=8.33%`，平均初始规划时间 `0.687s`、平均总规划时间 `2.158s`、平均 replan `0.391`；状态占比为 `TRACK 4.21%`、`AVOID_HOLD 1.01%`、`SERVO 13.59%`、`PLAN_FAILED 81.18%`。filter intervention step rate 为 `100%`，safe-stop 和 recovery-relaxed 均为 `0`，平均最小 predictive h 为 `0.05097m`。平均最终位置误差为 `0.08580m`，成功 episode 平均完成时间为 `1.6909s`。filter 平均 episode solve time `26.28ms`，episode mean 最大值 `47.59ms`，episode peak 最大值 `1.322s`，因此实时性不满足 nominal 采用条件。

失败主类为 `SUCCESS=11`、`IK_NOT_FOUND=11`、`FILTER_STOP_TIMEOUT=1`，`PLAN_NOT_FOUND=0`、`TRACK_TIMEOUT=0`、`SERVO_TIMEOUT=0`、`COLLISION_TERMINATION=0`。与 recovery-gated hard-case 逐 reset paired：old success→new success `9`、old success→new failure `1`（`9027`）、old failure→new success `2`（`9093、9143`）、old failure→new failure `11`。新增成功来自原先的 filter timeout；回归来自瞬时目标速度最大化导致的时序不一致，而非碰撞或 LP 不可行。`plan-conditioned 11/12=91.67%` 低于 `0.95` gate，`full 11/23=47.83%` 低于 `0.90` gate；该分支不进入 nominal。

`9027` 的 LP 每步均达到最优，并非 solver failure；末段由 `predictive_link_4` 与关节 acceleration/velocity 约束主导，当前 TCP 目标速度相邻周期正负切换，误差在约 `0.058–0.066m` 振荡，未进入 `0.055m` success 阈值。完整 23-case filter 平均耗时约 `26ms` 且存在约秒级峰值，实时性仍未达 nominal gate。

`9027` trace 细节：最终误差 `0.062095m`，240 步中 `SERVO=226`、`TRACK=12`、`AVOID_HOLD=2`；LP fallback stage 全部为 `goal_velocity_linprog`，末段最大可行与实际 projected 目标速度均值均约 `0.06305m/s`，范围约 `-0.06247～0.55612m/s`。因此 LP 没有明显可行性 gap，问题是逐步重算当前目标方向的时间一致性，而不是目标速度没有被执行。

已实现独立二阶段时间一致性候选 `configs/experiments/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_9027_v1.yaml`：第一阶段保持严格 LP 最大化目标速度；第二阶段仅在最优值 `0.02m/s` 近邻内，最小化相对 requested/previous command 的 L1 距离。新增 command alignment/sign-change 和 secondary LP 诊断字段；不放宽任何安全约束。单元测试现为 `35 passed`。截至本次文档更新，该候选尚未生成 smoke/paired 输出，因此不能把它写成已验证的控制结果。下一步只运行 `9027`，若消除振荡，再做 `9093/9143/9027` paired；在三例通过前不运行 23-case 重跑、200-reset nominal 或 Residual 训练。

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
