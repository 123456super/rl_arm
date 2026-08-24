# S1 静态障碍物实验谱系、失败方向与根因登记

> 更新时间：2026-08-24。本文是 S1 静态障碍物后续实验的详细登记册。它记录每次实验的父版本、固定条件、目标假设、结果、paired gate、失败原因和停止理由。开始任何新实验前，必须先查本登记册，避免重复已经否定的方向。

## 1. 结论先行

当前唯一可继续作为静态 nominal 候选的是：

```text
配置：configs/experiments/hierarchical/
      s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_t005_nominal_v1.yaml
结果：full success 189/200 = 94.5%
      plan-conditioned success 189/189 = 100%
      IK/plan found 189/200 = 94.5%
      collision_any 0/200，timeout 0
hard-case：12/23，plan-conditioned 12/12，collision 0
paired：相对 recovery-gated nominal 187 -> 189，success->failure = 0，failure->success = 2
```

goal-region 后续分支已经冻结，不得再直接扩展到 23-case 或 nominal：

```text
clearance terminal reuse       12/23，无 paired 增益
expanded goal-region 384       0/7，无 paired 增益
goal-error-first selector      0/2，无 paired 增益
goal-region terminal priority  0/2，无 paired 增益
priority + goal-error 组合     0/2，无 paired 增益
```

所有上述分支的 `collision_any=0`，但“无碰撞”不等于“成功率改进”。

## 2. 固定条件和实验身份

除非实验明确声明为诊断，不得修改以下条件：

| 条件 | 固定值 |
| --- | --- |
| 场景 | S1，单个静态球障碍，`scenario=random`，障碍速度 `[0,0]` |
| 控制频率 | 20 Hz |
| horizon | 12 s / 240 steps |
| success tolerance | `0.055 m` |
| safety margin | `d_safe=0.12 m` |
| 碰撞判定 | capsule overlap、PyBullet contact、termination collision 均保留 |
| 状态机 | `TRACK / AVOID_HOLD / REPLAN / SERVO / PLAN_FAILED` |
| safety filter | 唯一命令出口；所有 planner/tracker/recovery 命令必须经过同一严格出口 |
| 评估模式 | `--nominal-only`、zero residual；不加载 actor |
| 正式 hard-case manifest | `configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json`，23 seeds |
| 正式 nominal manifest | `configs/experiments/reaching_recovery/manifests/v1_final.json`，200 seeds |

每个新实验必须同时保存三件东西：配置 YAML、seed manifest、独立 output 目录。输出目录禁止覆盖父实验；paired 比较必须使用相同 reset seed，而不是只比较 episode index。

## 3. 正确的实验谱系

```text
frozen hierarchical baseline
        |
        v
revised v2: structured IK / DLS / clearance tracker
        |
        +--> tracker v3 --> tracker v4
        |
        +--> filter-only / late recovery --> recovery-gated
                                      |
                                      v
                         goal-velocity priority
                                      |
                                      v
                         temporal consistency 0.02
                                      |
                                      v
                         t005 (0.005 m/s)  [当前冻结候选]
                                      |
                                      +--> latency variant [拒绝]
                                      |
                                      +--> IK feasibility / goal-region branch
                                              |
                                              +--> predictive IK v1/v2
                                              +--> raw feasibility audit
                                              +--> obstacle-aware nullspace
                                              +--> target-shell
                                              +--> goal-region 96 samples
                                              +--> terminal reuse / clearance selector
                                              +--> expanded 384 samples
                                                     |
                                                     +--> goal-error selector
                                                     +--> goal-region terminal priority
                                                     +--> priority + goal-error
                                                     [全部冻结]
```

关键联系：t005 已经解决的是 **strict filter 的目标推进 objective 和时间一致性**；goal-region 分支解决的是 **IK/终端构型可行性**。不能把 t005 的 filter 结果直接解释为 goal-region 终端必然可完成，也不能把 goal-region 找到路径解释为 12 秒内可完成。

## 4. Gate 定义

### 4.1 诊断级 gate

适用于单 seed、2 seed 或机制诊断：

1. `collision_any == 0`；
2. 不出现 safety rule、`d_safe`、success tolerance、horizon 或状态机语义漂移；
3. 必须记录失败主类和 trace，不得只报告 success rate；
4. 若诊断 manifest 中没有旧方法成功 seed，则不能据此证明“无回归”；它只能证明该失败子集上是否出现新成功。

### 4.2 hard-case 候选 gate

固定 23 reset 后才使用以下联合条件：

```text
collision_any = 0
old_success_new_failure = 0
old_failure_new_success >= 1
plan-conditioned success >= 0.95（候选 nominal gate）
```

没有 `old_failure_new_success` 的版本只能标为“无回归审计”，不能进入 nominal。

### 4.3 nominal gate

只有 hard-case gate 通过后，才允许运行 200-reset nominal。nominal 还必须报告 pooled、逐 seed、逐 reset paired、timeout、最终误差、filter latency 和碰撞，不能只看 full success。

## 5. 已完成实验总表

### 5.1 架构、recovery 和 filter 目标推进线

| ID / 父版本 | 配置与输出 | 目的和只改变的内容 | 结果 | 决策与根因 |
| --- | --- | --- | --- | --- |
| revised-v2 | `s1_static_revised.yaml`；`outputs/hierarchical/s1_static_revised/` | structured IK、DLS、clearance tracker | nominal `178/200`，plan-conditioned `178/189=94.18%`；paired success→failure 2、failure→success 3 | 未过 nominal gate；失败转为 terminal/filter，不能训练 Residual |
| tracker-v3 | `s1_static_revised_tracker_v3.yaml`；`outputs/hierarchical/s1_static_revised_tracker_v3/` | filter-aware terminal servo、速度阻尼、gain taper | hard-case `3/23`，与 v2 持平；`FILTER_STOP_TIMEOUT=9` | 平均干预下降但无新增成功；不是方向错误，停止 |
| tracker-v4 | `s1_static_revised_tracker_v4.yaml`；`outputs/hierarchical/s1_static_revised_tracker_v4/` | bounded terminal-stall replan | hard-case `3/23`；replan、规划时间上升，无新增成功 | 增加 replan 不是有效根因修复，停止 |
| filter-only | `s1_static_revised_filter_only_diag.yaml`；`outputs/hierarchical/s1_static_revised_filter_only_diag/` | 关闭主动 recovery，严格 filter 保留 | hard-case `8/23`；相对 v2 修复 5，无回归、collision 0 | 证明 recovery 接管会损害 terminal 推进，但不是最终候选 |
| late recovery | `s1_static_revised_recovery_late_diag.yaml`；`outputs/hierarchical/s1_static_revised_recovery_late_diag/` | recovery 推迟到边界/短 TTC | hard-case `5/23` | 只修复部分样本，弱于 filter-only，停止 |
| recovery-gated | `s1_static_revised_recovery_gated.yaml`；`outputs/hierarchical/s1_static_revised_recovery_gated/` | AVOID_HOLD 可 recovery，SERVO 禁止 recovery 覆盖 DLS | hard-case `10/23`；nominal `187/200`、plan-conditioned `187/189=98.94%`、collision 0 | 通过 nominal gate，作为 t005 父版本；zero-residual，不训练 |
| filter timeout audit | `s1_static_revised_recovery_gated_filter_timeout_diag.yaml`；`outputs/hierarchical/s1_static_revised_recovery_gated_filter_timeout_diag/`；manifest `v1_hierarchical_filter_timeout_cases.json` | 复用严格约束，审计最大可行 TCP 目标速度 | 最大可行速度为正，而旧 projected 速度明显偏低；safe-stop 0 | 根因是 objective 不匹配，不是安全集合整体不可行 |
| priority 2-seed | `s1_static_revised_recovery_gated_goal_velocity_priority_diag.yaml`；`outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_diag/` | 同一 hard constraints 下 HiGHS LP 最大化目标方向速度 | `9093/9143=2/2`，collision/safe-stop 0；平均耗时约 30/68 ms，峰值约 0.47/1.44 s | 成功率有效但实时性未达标，不能直接替换 nominal |
| priority hard-case | `...goal_velocity_priority_hard_cases_v1.yaml`；`outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_hard_cases_v1/` | 扩展 23 reset | `11/23`，修复 9093/9143，但回归 9027；paired old→new failure 1 | 瞬时 LP 造成周期方向切换，停止 |
| temporal 0.02 | `...temporal_consistency_9027_v1.yaml` + triplet；`outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_9027_v1/` | 保留 LP 目标速度近邻，再按 requested/previous 连续性选择 | triplet 通过，修复 9027 的局部回归 | 继续收紧近邻带，形成 t005 |
| t005 | `...temporal_consistency_t005_hard_cases_v1.yaml`、`...t005_nominal_v1.yaml`；对应 `outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_t005_hard_cases_v1/` 与 `...t005_nominal_v1/` | 近邻阈值 `0.005 m/s` | hard-case `12/23`、plan-conditioned `12/12`、collision 0；nominal `189/200`、`189/189`、timeout 0 | 当前最佳静态候选，冻结 |
| t005 latency | `...t005_latency_hard_cases_v1.yaml`；`outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_t005_latency_hard_cases_v1/` | 仅跳过零请求 LP | success 仍 `12/23`；p95 24.57 ms、p99 267.78 ms、max 1.577 s | 未改善尾延迟，停止 |

### 5.2 IK / obstacle-aware feasibility 线

| ID / 父版本 | 配置与输出 | 目的 | 结果 | 根因与决策 |
| --- | --- | --- | --- | --- |
| IK predictive v1 | `s1_static_revised_ik_predictive.yaml`；`outputs/hierarchical/s1_static_revised_ik_predictive/` | IK 64→256、DLS seeds、predictive path selector | `9/23`，plan-conditioned `9/12`，timeout 3，collision 0；回归 9060 | 搜索预算增加没有修复 IK，且 predictive selector 回归；停止 |
| IK predictive v2 | `s1_static_revised_ik_predictive_v2.yaml` | 增加拒绝原因审计，修复未评分候选参与排序 | `10/23`，无 paired 增益、无回归 | 只是审计版本；拒绝总量以 `goal_error/obstacle_clearance/contact` 为主，不再盲增次数 |
| feasibility audit | `s1_static_revised_ik_feasibility_audit_v1.yaml` | 区分 raw goal error 不足与障碍阻塞 | raw goal reachable `17/23`；raw unreachable `6/23`；accepted obstacle-valid `12/23`；obstacle-blocked `5/23` | 根因从“搜索次数不足”转为目标可达性/障碍几何可行性 |
| nullspace v1/v2 | `s1_static_revised_ik_obstacle_aware_nullspace_9006_v1/v2.yaml` | 在 exact target 附近沿 clearance gradient 搜索 | 9006 仍 IK failure；2000 多分支最佳 clearance 约 `0.111m<0.12m` | 局部 nullspace 不能跨越障碍阻塞；不扩展 |
| target-shell | `s1_static_revised_ik_shell_v1.yaml`；`outputs/hierarchical/s1_static_revised_ik_shell_v1/` | 在原 `0.055m` success 球内搜索 shell target | `10/23`，IK found `13/23`，timeout 3，collision 0；无 failure→success；9006 变为 filter timeout | shell 改善入口 IK，但没有证明终端 12 秒可完成；停止 |

### 5.3 goal-region / terminal reuse 线

| ID / 父版本 | 配置与输出 | 只改变的内容 | 结果 | 决策 |
| --- | --- | --- | --- | --- |
| goal-region 基础 | `s1_static_revised_ik_goal_region_t005_hard_cases_v1.yaml` | 96 Fibonacci goal-region samples，半径 0.045m；exact IK 失败后启用 | `12/23`、plan-conditioned `12/13`、IK `13/23`、timeout 1、collision 0；paired 与 t005 `12→12`，无新增 | 无回归但无收益；不作为改进候选 |
| terminal reuse | `...retain_terminal_9006_v1.yaml` 及 hold/slow/track/waypoint smoke | AVOID_HOLD/replan 复用已验证 goal-region terminal | 9006 多次 smoke 均 `FILTER_STOP_TIMEOUT`，误差约 0.078–0.105m；无成功 | 复用终端不能解决 filter/时限问题 |
| clearance retain hard-case | `s1_static_revised_ik_goal_region_t005_clearance_retain_terminal_hard_cases_v1.yaml` | `candidate_selection=clearance_then_length`，保留 terminal | `12/23`、plan-conditioned `12/13`、collision 0；paired `old_failure→new_success=0` | 无回归审计，不扩展 nominal |
| predictive retain | `...predictive_retain_terminal_hard_cases_v1.yaml` | predictive clearance selector + terminal reuse | `11/23`、timeout 2；9143 success→failure | predictive selector 明确回归，永久拒绝 |
| expanded region | `...expanded_diagnostic_v1.yaml`；manifest `v1_hierarchical_goal_region_expanded_cases.json` | 384 samples，半径扩展到完整 `0.055m` | targeted `0/7`；IK 2/7；timeout 2；collision 0 | 9006/9110 找到终端仍 timeout；9098/9120 raw unreachable；9102/9108/9192 obstacle 拒绝；不再增加预算 |
| terminal priority smoke v1 | `...terminal_priority_smoke_v1.yaml` | 对 goal-region terminal 启用 t005 objective | `0/2`，且 9110 实际继承 96 samples、没有找到 terminal | 配置继承错误，不能作为 objective 结论；已由 v2 纠正 |
| terminal priority smoke v2 | `...expanded_terminal_priority_smoke_v2.yaml` | 正确继承 384 samples + t005 objective | `0/2`，两例 IK/plan found，均 timeout；目标速度优先实际生效 | 排除“goal-region 终端未使用目标速度 objective” |
| goal-error selector | `...expanded_goal_error_smoke_v1.yaml` | 合法终端按原始 goal error 优先 | 9006/9110 `0/2`；终端误差 `0.0212/0.0513m`，最终误差 `0.0796/0.1152m` | 终端更接近目标仍无法完成；selector 不是根因 |
| priority + goal-error | `...expanded_goal_error_priority_smoke_v1.yaml` | goal-error selector + t005 terminal priority | `0/2`；最终误差约 `0.0902/0.1146m`，collision 0 | 组合仍 timeout；goal-region 分支冻结 |

### 5.4 9006 定向 smoke 的完整记录

这些实验都继承 goal-region 终端复用父版本，只针对 `9006` 改变一个 tracker/selector 变量。它们不能替代 23-case paired，但必须保留，因为它们排除了“只要在 AVOID_HOLD/terminal 阶段换一个简单命令就能成功”的假设。

| 配置 | 改动 | 结果（success / final error / final state） | 结论 |
| --- | --- | --- | --- |
| `...retain_terminal_9006_v1.yaml` | 保留终端，goal-velocity priority nonzero request | `0 / 0.0870m / SERVO`（部分重复运行进入 AVOID_HOLD，约 `0.0840m`） | 复用终端本身无效 |
| `...hold_terminal_9006_v1.yaml` | AVOID_HOLD 零 waypoint command | `0 / 0.0782m / AVOID_HOLD` | 误差下降但仍超时 |
| `...clearance_terminal_9006_v1.yaml` | goal-region 内按 clearance/length 选终端 | `0 / 0.0832m / AVOID_HOLD` | 高 clearance 不等于可完成 |
| `...clearance_hold_terminal_9006_v1.yaml` | 高 clearance + hold gain 0 | `0 / 0.0851m / AVOID_HOLD` | 停住不能收敛 |
| `...clearance_slow_terminal_9006_v1.yaml` | 高 clearance + hold gain 0.25 | `0 / 0.0804m / AVOID_HOLD` | 低增益不能收敛 |
| `...clearance_track_terminal_9006_v1.yaml` | 高 clearance + hold gain 1.0 | `0 / 0.0802m / AVOID_HOLD` | 继续跟踪仍超时 |
| `...predictive_retain_terminal_9006_v1.yaml` | predictive clearance selector，16 条 shortlist | `0 / 0.0802m / AVOID_HOLD` | predictive 排序无收益，且 hard-case 9143 发生回归 |
| `...terminal_waypoint_9006_v1.yaml` | 改 servo trigger 到 0.055m | `0 / 0.0840m / PLAN_FAILED` | 过早切 terminal/重规划不能解决 |
| `...servo_damped_9006_v1.yaml` | filter-aware gain floor、近目标 gain scale | 未形成可采用的成功证据 | 只降低动作幅度，不解决可行推进 |

上述 smoke 的共同观测是：合法 path 可以存在，但 `FILTER_STOP_TIMEOUT` 仍发生；因此“找到 IK/path”与“在 12 秒内完成 terminal execution”必须分开统计。

### 5.5 本轮实际运行与未运行配置的边界

以下配置已经有正式输出或 targeted trace，结论可用于排除方向：

```text
s1_static_revised_ik_goal_region_t005_hard_cases_v1.yaml
s1_static_revised_ik_goal_region_t005_predictive_retain_terminal_hard_cases_v1.yaml
s1_static_revised_ik_goal_region_t005_clearance_retain_terminal_hard_cases_v1.yaml
s1_static_revised_ik_goal_region_t005_expanded_diagnostic_v1.yaml
s1_static_revised_ik_goal_region_t005_expanded_terminal_priority_smoke_v2.yaml
s1_static_revised_ik_goal_region_t005_expanded_goal_error_smoke_v1.yaml
s1_static_revised_ik_goal_region_t005_expanded_goal_error_priority_smoke_v1.yaml
```

以下配置虽然已创建，但没有通过前置 smoke，不能称为实验结果，也不允许直接运行来“补齐统计”：

```text
s1_static_revised_ik_goal_region_t005_terminal_priority_hard_cases_v1.yaml
s1_static_revised_ik_goal_region_t005_nominal_v1.yaml
s1_static_revised_ik_goal_region_t005_predictive_retain_terminal_nominal_v1.yaml
s1_static_revised_ik_goal_region_t005_retain_terminal_9006_v2.yaml
s1_static_revised_ik_goal_region_t005_servo_damped_9006_v1.yaml
```

原因分别是：terminal-priority smoke 没有 `old_failure→new_success`；goal-region nominal 没有 hard-case paired gain；predictive 分支已有 success→failure 回归。创建 YAML 不等于完成验证，必须以 output CSV、summary、paired JSON 和 trace 为证据。

本轮定向 smoke 的资产索引如下；`未形成输出`表示只创建了配置，不能据此声称该机制已验证或失败：

| 配置 | manifest | output / 证据 | 状态 |
| --- | --- | --- | --- |
| `s1_static_revised_ik_goal_region_t005_retain_terminal_9006_v1.yaml` | `v1_hierarchical_obstacle_aware_9006.json` | `outputs/hierarchical/s1_static_revised_ik_goal_region_t005_retain_terminal_9006_v1/`，9 次 smoke | 已运行，全部 timeout |
| `s1_static_revised_ik_goal_region_t005_clearance_terminal_9006_v1.yaml` | 同上 | `...clearance_terminal_9006_v1/smoke_9006.csv` | 已运行，timeout |
| `s1_static_revised_ik_goal_region_t005_clearance_hold_terminal_9006_v1.yaml` | 同上 | `...clearance_hold_terminal_9006_v1/smoke_9006*.csv` | 已运行，timeout |
| `s1_static_revised_ik_goal_region_t005_clearance_slow_terminal_9006_v1.yaml` | 同上 | `...clearance_slow_terminal_9006_v1/smoke_9006.csv` | 已运行，timeout |
| `s1_static_revised_ik_goal_region_t005_clearance_track_terminal_9006_v1.yaml` | 同上 | `...clearance_track_terminal_9006_v1/smoke_9006.csv` | 已运行，timeout |
| `s1_static_revised_ik_goal_region_t005_terminal_waypoint_9006_v1.yaml` | 同上 | `...terminal_waypoint_9006_v1/smoke_9006.csv` | 已运行，timeout/PLAN_FAILED |
| `s1_static_revised_ik_goal_region_t005_predictive_retain_terminal_9006_v1.yaml` | 同上 | `...predictive_retain_terminal_9006_v1/smoke_9006.csv` | 已运行，timeout；9143 paired 回归 |
| `s1_static_revised_ik_goal_region_t005_servo_damped_9006_v1.yaml` | 同上 | 无 output 目录 | 未运行，禁止自行补跑 |
| `s1_static_revised_ik_goal_region_t005_retain_terminal_9006_v2.yaml` | 同上 | 无 output 目录 | 未运行，禁止自行补跑 |
| `s1_static_revised_ik_goal_region_t005_9143_recheck_v1.yaml` | `v1_hierarchical_obstacle_aware_9143.json` | `...9143_recheck_v1/recheck_9143.csv` | 已运行，普通 t005 成功，仅作回归确认 |
| `s1_static_revised_ik_goal_region_t005_predictive_9143_recheck_v1.yaml` | 同上 | `...predictive_9143_recheck_v1/recheck_9143.csv` | 已运行，失败，确认 predictive 回归 |

### 5.6 全方向防重复索引

下面按“改变了什么机制”登记本轮所有方向。相同方向即使换文件名、换 seed 或换 output 目录，也不能视为新假设。

| 方向 | 本轮实际验证 | 结论 | 后续规则 |
| --- | --- | --- | --- |
| revised v2 的 IK/DLS、路径 clearance/length、tracker | revised-v2、tracker-v3、tracker-v4 | tracker 降低部分干预但没有新增成功；bounded replan 增加开销无收益 | 不再用更多 tracker gain/replan 掩盖 IK 缺陷 |
| recovery 接管时机 | filter-only、late-recovery、recovery-gated | SERVO 禁止 recovery 覆盖 DLS 是有效父版本 | 不再重复关闭/延迟 recovery，除非提出新的可证伪状态机假设 |
| filter 目标函数 | timeout audit、goal-velocity priority、temporal 0.02、t005、latency | 原 objective 目标推进不足；瞬时 LP 有 9027 回归；t005 通过 nominal gate；latency 无收益 | t005 冻结；不再只改 objective 权重或跳过零请求 LP |
| 普通 IK 增加预算 / predictive selector | predictive v1/v2、256 次 IK、4 DLS seeds、拒绝审计 | 无 IK failure→success；v1 还有 9060 回归，v2 只有无回归审计 | 不再盲增 IK 调用次数或只换 predictive 排序 |
| raw IK 可行性诊断 | feasibility audit、拒绝原因汇总 | 区分 raw goal-error 不足与 obstacle-blocked | 后续只能转向构型分支搜索/可行性证明 |
| obstacle-aware nullspace | nullspace v1/v2、9006 多分支 2000 扫描 | 最佳 clearance 约 0.111m，仍低于 0.12m | 不扩展同类局部梯度或 rest-pose 随机扫描 |
| target shell / goal region | shell v1、shell+t005、96 samples、384 samples | 可改善 IK 入口，但无 paired 新成功；9006 变成 terminal timeout | 不再扩大 shell 半径、sample 数或 success 球内随机点 |
| terminal 构型和命令复用 | retain、hold、slow、track、waypoint、servo damping smoke | 合法 path/terminal 仍不能在 12s 内完成 | 不再只换 terminal hold/gain/trigger/retain |
| terminal selector | clearance、predictive、goal-error、组合 priority | 无新增成功；predictive 导致 9143 回归 | 不再只改 selector 排序 |
| goal-region terminal priority | priority smoke v1（继承错误）、v2（继承正确）、priority+goal-error | 正确 v2 trace 已证明 objective 生效，但 9006/9110 仍 timeout | 终端问题必须升级到 time/dynamics reachability，不再重跑 23-case |
| nominal / Residual | recovery-gated nominal、t005 nominal；Residual 三 seed 未运行 | t005 是 zero-residual 最佳候选；Residual 尚无正式结果 | 未获授权前不训练、不把 nominal 写成 S1-R final |

单独的 9143 recheck 也已登记：普通 t005 recheck 成功，而 predictive terminal recheck 失败；这两项只是确认既有 paired 结论，不构成新改进方向。

## 6. 失败根因定位

### 6.0 t005 hard-case 的逐 seed 根因闭环

以下分类来自 t005 hard-case CSV 的 `hierarchical_ik_min_goal_error_m`、`hierarchical_ik_goal_reachable_count` 和 `hierarchical_ik_rejection_counts`，不是根据最终误差猜测：

| 根因层级 | reset seeds | 证据 | 可采取的研究层级 |
| --- | --- | --- | --- |
| raw goal error 不足 | `9021, 9065, 9095, 9098, 9120, 9142` | raw goal reachable count 为 0；最小 goal error 约 `0.0703–0.1413m`，均超过 `0.055m` | 目标/构型可达性分析；不能用障碍 selector 修复 |
| raw goal error 可达但障碍拒绝 | `9006, 9102, 9108, 9110, 9192` | raw reachable；候选全被 `obstacle_clearance`/`obstacle_contact` 拒绝；solver、joint-limit、self-collision 均无证据 | obstacle-aware 构型分支图、连续可行性或完备性分析 |
| 已找到 terminal/path 但时限内不收敛 | `9006, 9110`（expanded diagnostic） | 384 samples 下有 obstacle-valid terminal 和 path，但均 `FILTER_STOP_TIMEOUT`；最终误差仍约 `0.09–0.115m` | 带时间/动力学约束的 terminal reachability；不能继续换 selector |

因此“11 个 IK_NOT_FOUND”不是一个单一 bug：前 6 个是目标误差层，后 5 个是障碍几何层；shell 只把个别样本推进到第三层 terminal execution，不能算成功修复。

### 6.1 根因一：raw IK 目标误差不可达（6/23）

在 feasibility audit 和 expanded region 中，`9098/9120` 即使把 goal-region 采样扩展到 384 次，最小 goal error 仍约 `0.070–0.075m`，超过固定 success tolerance `0.055m`。这类 reset 不是 obstacle clearance 问题，也不是 solver、joint-limit 或 planner budget 问题；继续增加同类随机 IK 调用不能改变目标误差下界。

### 6.2 根因二：raw IK 可达但障碍几何拒绝（5/23）

`9006/9102/9108/9110/9192` 的 raw IK 误差已经进入容差球，但候选主要被 `obstacle_clearance` 或 `obstacle_contact` 拒绝。扩展到 384 samples 后，`9102`、`9108`、`9192` 仍无 obstacle-valid candidate；`9006/9110` 虽出现合法 terminal，却在执行阶段 timeout。这说明当前问题是障碍约束下的构型分支/可行域以及后续 terminal 可达性，而不是普通 IK 调用失败。`d_safe=0.12m` 未放宽，故不能把这些样本写成形式化不可达，只能写成“当前搜索证据下未找到”。

### 6.3 根因三：存在合法 goal-region 终端，但没有 12 秒内可完成的 terminal execution

`9006` 和 `9110` 在 expanded region 中找到 obstacle-valid terminal，RRT/direct path 也存在，但最终仍 `FILTER_STOP_TIMEOUT`。关键 trace 证据：

| seed | 终端/路径证据 | filter 证据 | 结果 |
| --- | --- | --- | --- |
| 9006 | goal-error-first terminal error `0.0212m`，path clearance `0.126m` | priority 组合中 projected goal speed 均值约 `0.081m/s`，末段约 `-0.023m/s`，干预均值约 `0.81`，max solve 约 `0.45s` | final error `0.0902m`，timeout |
| 9110 | terminal error `0.0513m`，path clearance `0.121m` | projected goal speed 均值约 `0.087m/s`，误差在末段反复增大/减小，干预均值约 `0.95`，max solve 约 `0.52s` | final error `0.1146m`，timeout |

目标速度 LP 在这两个例子中确实执行（primary/secondary 字段有效），但仍出现负向或振荡的末段实际速度；因此根因不是“goal-region 没有调用 priority”，而是 **终端附近的严格 predictive/barrier/acceleration 约束与当前终端构型、路径和 12 秒时限组合后，无法形成稳定单调的闭环收敛**。现有证据足以拒绝简单 selector、采样预算和 objective 开关，但不足以声称已经有形式化不可行证明。

### 6.4 已排除的伪根因

| 伪根因 | 排除证据 |
| --- | --- |
| IK solver 崩溃 | `solver_error=0`，短解、duplicate、joint-limit、self-collision 也为 0 |
| 只要增加 IK 次数就能解决 | predictive v1/v2、384 goal-region samples 均无 paired gain |
| 只要换 path selector 就能解决 | clearance、predictive、goal-error selector 均无新增成功；predictive 还造成 9143 回归 |
| 只要复用 terminal 就能解决 | retain/hold/slow/track/waypoint smoke 全部 timeout |
| filter 没有目标推进能力 | 9093/9143 LP audit 显示严格可行目标速度多数为正；t005 已修复该类问题 |
| goal-region terminal 没有使用 priority | v2 trace 显示 primary、secondary、projected goal velocity 均有效，但仍 timeout |
| 碰撞导致失败 | 所有本轮 goal-region 分支 `collision_any=0` |

## 7. 为什么前面的实验不能脱节

今后每次实验报告必须回答以下五个问题：

1. **它继承哪个父版本？** 明确 YAML `includes` 和父架构，不能只写新配置名。
2. **它针对父版本的哪个失败主类？** 例如 t005 针对 filter objective；goal-region 针对 IK/obstacle feasibility；不能把两者混为一个“成功率优化”。
3. **父版本在完全相同 seed 上是什么结果？** 必须保存 old/new CSV 和 seed-level transition。
4. **新机制是否实际生效？** 通过 trace 字段确认，例如 goal velocity priority 必须检查 primary/secondary/projected 字段，不能只看 YAML。
5. **失败后下一步是扩大样本、换模块，还是停止？** 只有当机制证据支持新假设时才能继续；同一模块同一假设不能重复运行。

特别注意本轮暴露的 gate 陷阱：`v1_hierarchical_goal_region_terminal_priority_cases.json` 只有 `9006/9110`，两者在旧 t005 中都是 failure。因此该 manifest 上 `old_failure_new_success>=1` 是必要的机制 gate，但它无法检查 success regression；要宣称候选，仍必须回到完整 23-case paired gate。第一次 terminal-priority smoke 还错误继承了 96 samples，导致 9110 未进入 goal-region；这类配置继承错误必须在运行前用 `load_config` 打印并断言。

## 8. 新实验登记卡片（必须填写）

新实验不得只提交一个 YAML 或一条运行命令；必须在登记册中先写完下面这张卡片，再开始 smoke。这样可以保证实验和父版本、失败主类、paired 结果以及停止理由一一对应：

```text
experiment_id:
parent_experiment_id:
parent_config:
child_config:
hypothesis:                 # 只写一个可证伪假设
target_failure_class:      # IK_NOT_FOUND / PLAN_NOT_FOUND / TRACK_TIMEOUT /
                            # SERVO_TIMEOUT / FILTER_STOP_TIMEOUT
changed_fields:             # 只列本实验实际改变的字段
fixed_invariants:           # d_safe、success tolerance、horizon、状态机、碰撞、filter 出口
diagnostic_manifest:
paired_old_csv:
child_output_dir:
mechanism_trace_fields:     # 证明新机制实际生效的字段
smoke_gate:
hard_case_gate:
nominal_gate:
result_summary:
paired_transitions:
failure_mode_counts:
root_cause_evidence:
decision:                   # adopt / audit-only / rejected / blocked
stop_reason:
next_allowed_hypothesis:
```

其中 `decision=adopt` 只有在完整 hard-case paired gate 和 nominal gate 都通过时才允许；`audit-only` 只能说明机制/回归情况；`rejected` 或 `blocked` 必须写明禁止的近邻方向。任何“只有 smoke、没有 old CSV、没有 trace”的结果都不能进入候选列表。

## 9. 后续重新开启研究时的唯一允许流程

1. 先读取本文第 5、6、5.6 节，确认新假设不在“已拒绝方向”中，并填写第 8 节登记卡片。
2. 选择一个明确父版本；保留固定任务定义和唯一 safety-filter 出口。
3. 建立最小机制 manifest，并同时准备父版本同 seed 的 old CSV。
4. 运行前断言：effective config 的关键字段、manifest seed 数、output 目录不存在覆盖、`nominal_only=true`。
5. 先做 targeted smoke：只验证机制是否实际生效、collision 是否为 0、是否出现至少一个新成功。
6. targeted smoke 通过后，使用完整 23-case manifest 做 paired hard-case；要求无回归且至少一个新增成功。
7. hard-case 通过后，才允许 200-reset nominal；否则不得训练 Residual。
8. 每次结果立即补充本登记册：配置、manifest、输出、summary、paired、trace、结果、根因、停止理由。

下一次若继续攻克 goal-region terminal，必须改变问题层级，例如做带时间/动力学约束的 terminal reachability 或可行性证明；不能再次尝试“更多 samples + clearance selector + priority 开关”的组合。
