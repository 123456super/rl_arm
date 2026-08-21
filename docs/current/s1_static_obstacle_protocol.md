# S1 静态障碍物实验协议

> 更新时间：2026-08-21。状态：原 S1 final 已冻结；recovery-gated nominal 已完成并通过 gate；`ik_predictive_v1` hard-case 未通过，当前转入 IK 拒绝原因审计；Residual 训练暂缓。本文只规定 S1 场景和执行细节；动作语义、状态机、安全出口、通用指标与版本规则继承[分层控制通用实验协议](hierarchical_control_protocol.md)。

## 1. 目标与固定条件

S1 验证最终论文架构的静态基座，并把失败归因到 IK、plan、tracking、servo、filter 或 Residual 中的具体模块。旧 direct-SAC 只作为外部历史对照，来源见[架构转向证据](../archive/legacy_direct_sac/evidence/architecture_transition_evidence.md)。

- UR5、20 Hz、12 s horizon、`0.055 m` success tolerance、固定 final manifest 和 capsule 定义不变。
- 单个静态球障碍：`scenario=random`、`speed_range=[0,0]`。
- canonical 配置：`configs/experiments/hierarchical/s1_static.yaml`。
- `env.hierarchical_control.enabled=true`，legacy `env.residual_control.enabled=false`。
- `risk.representation=robust_predictive`，静态障碍速度为零也不关闭预测链。
- actor/replay 必须具有 hierarchical signature；checkpoint selection 只使用 validation manifest。
- frozen S1-R pooled result 为 `529/600=88.17%` full success、`529/555=95.32%` plan-conditioned、`0/600` collision。该结果和 `outputs/hierarchical/s1_static/` 下所有 CSV/summary 永不覆盖。
- revised v2 配置为 `configs/experiments/hierarchical/s1_static_revised.yaml`，输出根目录为 `outputs/hierarchical/s1_static_revised/`。
- revised v2 nominal final 已完成：`178/200=89.0%` full、`178/189=94.18%` plan-conditioned、`0/200` collision。IK found 和 plan found 均为 `189/200=94.5%`，但 gate 未通过。
- 下一诊断版本为独立的 `configs/experiments/hierarchical/s1_static_revised_tracker_v3.yaml`，只在 revised v2 tracker 上增加 filter-aware terminal servo：TCP 速度阻尼、近目标增益 taper、上一周期过滤干预比例反馈；不改变 IK/planner、任务定义或 safety 约束。v3 输出不得写入 v2 目录。
- v3 hard-case 为 `3/23=13.04%`，v4（有限 terminal-stall replan）仍为 `3/23=13.04%`；v4 增加规划开销但没有新增成功，因此 v4 不进入 200-reset final。下一实验固定 IK/planner，转向记录 filter 前后 task-space 速度与误差单调性。
- terminal diagnostics v2 已完成：`nominal_toward_goal_rate=1.0`、`nominal_mean_toward_goal_mps=0.430`，但 `recovery_command_rate=0.4766`，整体 `executed_toward_goal=0.0554 m/s`，说明主要瓶颈是主动 recovery 接管而非 terminal DLS 方向。下一步只做两个 hard-case 归因对照：`s1_static_revised_filter_only_diag.yaml`（关闭主动 recovery，严格 filter 保留）与 `s1_static_revised_recovery_late_diag.yaml`（recovery 延后到边界/短 TTC）。
- hard-case 配对显示 filter-only 为 `8/23`，较 v2 `3/23` 新增 5 个成功且无回归；late-recovery 为 `5/23`。正式候选配置 `s1_static_revised_recovery_gated.yaml` 只允许 `AVOID_HOLD` 使用 recovery，`SERVO` 禁止 recovery 覆盖 nominal DLS；该配置必须先完成同一 `v1_final.json` 的 200-reset nominal gate，之后才能考虑 Residual 训练。
- recovery-gated hard-case 为 `10/23=43.48%`，plan-conditioned `10/12=83.33%`，相对 v2 新增 7 个成功且无回归、无碰撞；只剩 `IK_NOT_FOUND=11` 与 `FILTER_STOP_TIMEOUT=2`，且 safety-filter safe-stop 为 `0`。该候选随后已完成 200-reset nominal-only。
- recovery-gated 200-reset nominal 已完成：full success `187/200=93.5%`、plan-conditioned `187/189=98.94%`、IK/plan found `189/200=94.5%`、timeout `2/200=1.0%`、collision `0/200`；两个 gate 均通过。该结果是 zero-residual nominal，不是 Residual final。
- `ik_predictive_v1` hard-case 已完成但未通过：配置 `configs/experiments/hierarchical/s1_static_revised_ik_predictive.yaml`，输出 `outputs/hierarchical/s1_static_revised_ik_predictive/`；结果为 full `9/23=39.13%`、plan-conditioned `9/12=75.00%`、IK/plan found `12/23=52.17%`、timeout `3/23`、collision `0`。高预算 IK（64→256，4 个 DLS seeds）没有修复任何 IK failure，且 seed `9060` 相对 recovery-gated 发生 success→failure，因此不进入 nominal。
- `ik_predictive_v2` 只增加拒绝原因审计，配置为 `configs/experiments/hierarchical/s1_static_revised_ik_predictive_v2.yaml`。必须保持 `d_safe`、success tolerance、horizon、manifest、状态机和 filter 出口不变；predictive 排序只允许实际完成 scoring 的候选参与，全部 scoring 失败时回退 `clearance_then_length`。
- `ik_predictive_v2` hard-case 已完成：full `10/23=43.48%`、plan-conditioned `10/12=83.33%`、IK/plan found `12/23=52.17%`、timeout `2/23`、collision `0`。相对 recovery-gated hard-case 为 `10→10`，paired 为 old success→new success `10`、old success→new failure `0`、old failure→new success `0`、old failure→new failure `13`，因此只是无回归审计，不是性能提升。拒绝原因总计为 `goal_error=5760`、`obstacle_clearance=4624`、`obstacle_contact=1374`、`workspace=60`，无 solver/duplicate/joint-limit/self-collision；不继续增加 IK 次数。
- 下一步使用独立 `configs/experiments/hierarchical/s1_static_revised_ik_feasibility_audit_v1.yaml` 做 23-reset raw-IK/obstacle-free feasibility audit。注意 `obstacle_free_goal_reachable` 只代表忽略障碍后的候选可达；真正的障碍阻塞以“raw goal reachable 且最终 `hierarchical_ik_candidate_count=0`”判定。本次结果为 raw goal reachable `17`、raw goal unreachable `6`、obstacle-blocked goal reachable `5`。该审计只增加诊断字段，不改变控制行为，也不放宽 obstacle clearance。
- feasibility audit 已完成：raw goal reachable `17/23`、raw goal unreachable `6/23`、accepted obstacle-valid goal `12/23`、obstacle-blocked goal reachable `5/23`。因此下一步不是继续增加 IK budget，而是在原 `0.055m` success tolerance 内做 target-shell IK。
- 新增诊断候选 `configs/experiments/hierarchical/s1_static_revised_ik_shell_v1.yaml`：exact-target IK 失败后最多查询 32 个 shell target，半径 `0.04m`；候选最终仍按原始 goal error、`d_safe=0.12m`、碰撞和 success 判定筛选。shell 不是放宽目标或安全约束，而是利用已有 success tolerance 搜索障碍另一侧的合法目标构型。
- target-shell v1 hard-case 已完成：full `10/23=43.48%`、plan-conditioned `10/13=76.92%`、IK found `13/23`、timeout `3/23`、collision `0`。相对 recovery-gated 没有 failure→success，也没有 success→failure；`9006` 只是从 `IK_NOT_FOUND` 变为 `FILTER_STOP_TIMEOUT`。因此 shell 不进入 nominal，不再扩大 shell 搜索。
- 当前 nominal 基座冻结回 recovery-gated；下一步仅审计 `9093`、`9143` 两个 `FILTER_STOP_TIMEOUT`，不再增加 IK 变体、不训练 Residual。
- timeout 诊断配置为 `configs/experiments/hierarchical/s1_static_revised_recovery_gated_filter_timeout_diag.yaml`，只对 manifest `v1_hierarchical_filter_timeout_cases.json` 的 `9093/9143` 打开 `diagnostic_logging`，不改变 recovery、filter、geometry、horizon 或 success rule。
- timeout 诊断现额外记录严格 filter 约束下的 TCP 目标方向最大可行速度。该 LP 只读复用线上同一组 joint/workspace/predictive barrier constraints；只有当最大可行速度为正而实际 projected 速度接近零时，才允许考虑新的 filter objective。诊断期间不放宽 `d_safe`、success tolerance、horizon 或 recovery 语义。
- 两个 timeout seed 的 LP 审计已确认多数时刻存在正的目标方向可行速度，且实际 projected 速度明显偏低；因此新增独立 `s1_static_revised_recovery_gated_goal_velocity_priority_diag.yaml`，仅替换严格 QP objective，不改变任何 hard constraint 或状态机语义。该分支只做 `9093/9143` paired 对照，不进入 nominal final。
- 首次 goal-velocity-priority paired 为 `1/2`：`9093` 成功，`9143` 因 OSQP 数值尺度导致多次 `maximum iterations reached` 而回退旧投影。配置现使用 `goal_velocity_objective_scale=10000` 对 objective 整体缩放，保持 objective 比例和所有约束不变；`9143` smoke 已成功。必须重新跑完整两个 seed 后再决定是否采用。
- 数值缩放后的完整重跑仍使 `9143` 在 `230/240` 步触发 OSQP 最大迭代。现改为严格约束下直接 HiGHS LP 最大化 TCP 目标方向速度，LP 失败才回退 OSQP；`9143` smoke 已 `31/31` 次 LP optimal、成功且无碰撞。需重新完成两个 seed paired，对照结果不进入 nominal final。
- HiGHS LP 两个 seed paired 已为 `2/2` 成功、`0` timeout、`0` collision；但平均 filter 耗时约 `30ms/68ms`，最大峰值约 `0.47s/1.44s`，暂不替换 recovery-gated nominal。下一步扩展 23 个 hard cases，并将实时性纳入 gate。
- HiGHS LP 23-reset 扩展已完成：full `11/23=47.83%`、plan-conditioned `11/12=91.67%`、timeout `1/23`（`9027`）、collision `0`。它修复 `9093/9143`，但回归 `9027`，因此不替换 recovery-gated nominal。`9027` 的失败是 LP 最优解在 acceleration/barrier 约束下逐周期方向切换，误差在 `0.058–0.066m` 振荡；不是 solver failure。已新增只针对 `9027` 的二阶段 temporal-consistency 配置：保留目标速度最优值的 `0.02m/s` 近邻，再按 requested/previous command 的 L1 连续性择优。运行顺序为先 `9027`，再三例 paired，未通过前不做 23-case/nominal/Residual。
- HiGHS LP 23-reset 扩展的过程设置：4 个 manifest shard 并行、合并脚本统一汇总、nominal-only zero-residual、独立输出目录；严格安全集合、`d_safe=0.12m`、`0.055m` success tolerance、horizon、recovery、状态机、碰撞规则均未变。结果机制指标为 direct/RRT `11/12` 与 `1/12`，平均初始/总规划时间 `0.687s/2.158s`，平均 replan `0.391`，filter intervention `100%`，safe-stop/recovery-relaxed `0`，平均最小 predictive h `0.05097m`，平均 filter solve `26.28ms`，最大 episode peak `1.322s`。与 recovery-gated 的 paired 转移为 `9` 个 success→success、`1` 个 success→failure（`9027`）、`2` 个 failure→success（`9093/9143`）、`11` 个 failure→failure。
- temporal-consistency 目前只有实现和 `35 passed` 单元测试，尚未产生 smoke/paired 输出；文档不将其视为已验证性能结果。其后续验证必须先跑 `9027`，再跑三例 paired，且不进入 23-case、200-reset 或 Residual 流程，除非三例同时满足成功率、collision、safe-stop 和 latency 条件。

诊断 trace 汇总命令：

```bash
python scripts/analyze_terminal_diagnostics.py \
  --evaluation outputs/hierarchical/s1_static_revised/terminal_diag_hard_cases.csv \
  --trace-dir outputs/hierarchical/s1_static_revised/terminal_diag_traces \
  --output outputs/hierarchical/s1_static_revised/terminal_diagnostics.json
```

该诊断还区分纯 nominal servo 与 recovery escape：`nominal_toward_goal_rate`、`recovery_command_rate` 和 `nominal_mean_toward_goal_mps`。因此不能把 recovery 为了避障而产生的非目标方向速度误判为 tracker 方向错误。

## 2. 实验矩阵与顺序

| 编号 | 方法 | 主要目的 |
| --- | --- | --- |
| S1-N0 | planner + tracker，无 terminal servo | 量化 terminal servo 的必要性 |
| S1-N | planner + tracker + servo，zero residual | 验证确定性 nominal 基座 |
| S1-R | 完整 hierarchical Residual | 与 S1-N 成对验证学习增益 |
| S1-A1 | S1-R，固定 residual budget | 消融 risk-conditioned budget |
| S1-A2 | S1-R，current risk | 消融 robust predictive risk |
| Legacy | direct-SAC 历史最好 | 仅作架构调整前外部对照 |

执行顺序固定为：

1. 先对 revised v2 已知失败 reset 做 hard-case diagnostic，不训练 actor。
2. 完成 recovery-gated 固定 200-reset nominal-only final，保存 CSV、summary 和 traces。
3. recovery-gated plan-conditioned success 为 `98.94%`，已达到 `95%`；但本轮决定暂缓训练 4401/4402/4403，先完成 paired comparison 和版本冻结。
4. 若后续获准训练，每个 seed 只从 recovery-gated 新 signature 随机初始化，按 validation manifest 选点，再用同一 final manifest 评估。
5. 完成 old/revised/gated nominal 的 pooled、逐 seed 和逐 reset paired comparison 后，才决定是否采用 recovery-gated 作为新的 S1 nominal 基座。
6. 所有方法使用相同任务定义、manifest、horizon、capsule 和最终安全出口。

7. `ik_predictive_v2` 和 target-shell v1 均无 paired 增益，不运行 nominal final；保留 recovery-gated 作为 nominal 基座，下一步只分析 `9093/9143` 的 AVOID_HOLD/filter timeout。

S1-N 建议同时达到 full success `>=90%`。未过门时按失败类别定位确定性基座，不用 Residual 补偿实现缺陷。

## 3. S1 失败归因

每个 episode 只能先归到一个互斥主类，再附加安全事件：

1. `IK_NOT_FOUND`：搜索预算内无合法 IK 候选；
2. `PLAN_NOT_FOUND`：存在合法候选，但规划预算内无路径；
3. `TRACK_TIMEOUT`：已有路径，但未进入 servo；
4. `SERVO_TIMEOUT`：进入 servo 后仍未成功；
5. `FILTER_STOP_TIMEOUT`：safe-stop/hold 主导时限；
6. `COLLISION_TERMINATION`；
7. `SUCCESS`。

IK 候选拒绝必须额外记录互斥原因：`solver_error`、`short_solution`、`duplicate`、`goal_error`、`joint_limit`、`workspace`、`obstacle_clearance`、`obstacle_contact`、`self_collision`。`invalid_state` 不再作为最终汇总类别；RRT 的状态检查仍保留 bool 接口。

Feasibility audit 还需记录 `hierarchical_ik_min_goal_error_m`、`hierarchical_ik_goal_reachable_count` 和 `hierarchical_ik_obstacle_free_goal_reachable_count`，以区分 raw IK 目标误差不足与障碍几何阻塞。

`IK_NOT_FOUND` 和 `PLAN_NOT_FOUND` 不得写成“不可达”，除非另有完备性或高预算审计。

## 4. 报告要求

S1-N 首先报告：IK/plan found、direct/RRT path、full 与 plan-conditioned success、final error、timeout、completion time、planning time/iterations、waypoint/path progress、碰撞、filter 事件和状态占比。

主结果表：

| method | full success | plan-conditioned success | IK found | plan found | timeout | collision_any | mean final error | mean time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

机制表：

| method | direct path | RRT path | plan iterations | TRACK% | HOLD% | REPLAN/ep | SERVO% | filter intervention | residual norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

成功必须同时满足目标误差和 `d_min>d_safe`。pooled 结果之外保留各训练 seed，所有比例给出分子/分母。

## 5. 运行方式

以下命令省略 `conda run -n rl`，请在已激活 `rl` 环境的终端执行。hard-case 诊断需要先把失败 seed 列表作为 manifest；正式 nominal 支持按 manifest shard 并行。

hard-case diagnostic（不训练、不覆盖 final）：

```bash
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised.yaml \
  --nominal-only --episodes 23 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json \
  --output outputs/hierarchical/s1_static_revised/hard_cases.csv \
  --trace-output outputs/hierarchical/s1_static_revised/hard_case_traces
```

revised v2 nominal（四终端并行示例）：

```bash
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised.yaml \
  --nominal-only --episodes 200 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_final.json \
  --manifest-shard-index 0 --manifest-shard-count 4 \
  --output outputs/hierarchical/s1_static_revised/nominal_final_shard_0.csv \
  --trace-output outputs/hierarchical/s1_static_revised/nominal_traces/shard_0
```

tracker v3 hard-case diagnostic（先运行，不训练）：

```bash
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_tracker_v3.yaml \
  --nominal-only \
  --episodes 23 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json \
  --output outputs/hierarchical/s1_static_revised_tracker_v3/hard_cases.csv \
  --trace-output outputs/hierarchical/s1_static_revised_tracker_v3/hard_case_traces
```

IK/predictive v2 hard-case diagnostic（四终端并行，只做审计，不训练、不覆盖 v1 输出）：

终端 1：

```bash
export PYTHONPATH=src
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_ik_predictive_v2.yaml \
  --nominal-only --episodes 23 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json \
  --manifest-shard-index 0 --manifest-shard-count 4 \
  --output outputs/hierarchical/s1_static_revised_ik_predictive_v2/hard_cases_shard_0.csv \
  --trace-output outputs/hierarchical/s1_static_revised_ik_predictive_v2/traces/shard_0
```

终端 2/3/4：保持命令不变，分别将 `manifest-shard-index` 改为 `1`、`2`、`3`，并将 `hard_cases_shard_0.csv` 和 `traces/shard_0` 同步改为对应编号。

四个终端结束后，在任一终端执行：

```bash
export PYTHONPATH=src
python scripts/merge_hierarchical_evaluations.py \
  --input outputs/hierarchical/s1_static_revised_ik_predictive_v2/hard_cases_shard_0.csv \
  --input outputs/hierarchical/s1_static_revised_ik_predictive_v2/hard_cases_shard_1.csv \
  --input outputs/hierarchical/s1_static_revised_ik_predictive_v2/hard_cases_shard_2.csv \
  --input outputs/hierarchical/s1_static_revised_ik_predictive_v2/hard_cases_shard_3.csv \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json \
  --episodes 23 \
  --output outputs/hierarchical/s1_static_revised_ik_predictive_v2/hard_cases.csv \
  --summary-output outputs/hierarchical/s1_static_revised_ik_predictive_v2/hard_cases_summary.json \
  --architecture-version hierarchical_s1_revised_ik_predictive_v2

python scripts/summarize_hierarchical_ik_rejections.py \
  --input outputs/hierarchical/s1_static_revised_ik_predictive_v2/hard_cases.csv \
  --output outputs/hierarchical/s1_static_revised_ik_predictive_v2/ik_rejection_summary.json
```

Feasibility audit（四终端并行；只做诊断，不运行 nominal final）：将下列命令中的 shard index、CSV 和 trace 编号分别设为 `0/1/2/3`。

```bash
export PYTHONPATH=src
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_ik_feasibility_audit_v1.yaml \
  --nominal-only --episodes 23 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json \
  --manifest-shard-index 0 --manifest-shard-count 4 \
  --output outputs/hierarchical/s1_static_revised_ik_feasibility_audit_v1/hard_cases_shard_0.csv \
  --trace-output outputs/hierarchical/s1_static_revised_ik_feasibility_audit_v1/traces/shard_0
```

四个 shard 完成后合并并汇总：

```bash
export PYTHONPATH=src
python scripts/merge_hierarchical_evaluations.py \
  --input outputs/hierarchical/s1_static_revised_ik_feasibility_audit_v1/hard_cases_shard_0.csv \
  --input outputs/hierarchical/s1_static_revised_ik_feasibility_audit_v1/hard_cases_shard_1.csv \
  --input outputs/hierarchical/s1_static_revised_ik_feasibility_audit_v1/hard_cases_shard_2.csv \
  --input outputs/hierarchical/s1_static_revised_ik_feasibility_audit_v1/hard_cases_shard_3.csv \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json \
  --episodes 23 \
  --output outputs/hierarchical/s1_static_revised_ik_feasibility_audit_v1/hard_cases.csv \
  --summary-output outputs/hierarchical/s1_static_revised_ik_feasibility_audit_v1/hard_cases_summary.json \
  --architecture-version hierarchical_s1_revised_ik_feasibility_audit_v1

python scripts/summarize_hierarchical_ik_rejections.py \
  --input outputs/hierarchical/s1_static_revised_ik_feasibility_audit_v1/hard_cases.csv \
  --output outputs/hierarchical/s1_static_revised_ik_feasibility_audit_v1/ik_rejection_summary.json
```

Target-shell IK hard-case（四终端并行；不运行 nominal final）：

```bash
export PYTHONPATH=src
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_ik_shell_v1.yaml \
  --nominal-only --episodes 23 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json \
  --manifest-shard-index 0 --manifest-shard-count 4 \
  --output outputs/hierarchical/s1_static_revised_ik_shell_v1/hard_cases_shard_0.csv \
  --trace-output outputs/hierarchical/s1_static_revised_ik_shell_v1/traces/shard_0
```

其余三个终端将 shard index、CSV 和 trace 编号改为 `1`、`2`、`3`。完成后合并：

```bash
export PYTHONPATH=src
python scripts/merge_hierarchical_evaluations.py \
  --input outputs/hierarchical/s1_static_revised_ik_shell_v1/hard_cases_shard_0.csv \
  --input outputs/hierarchical/s1_static_revised_ik_shell_v1/hard_cases_shard_1.csv \
  --input outputs/hierarchical/s1_static_revised_ik_shell_v1/hard_cases_shard_2.csv \
  --input outputs/hierarchical/s1_static_revised_ik_shell_v1/hard_cases_shard_3.csv \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json \
  --episodes 23 \
  --output outputs/hierarchical/s1_static_revised_ik_shell_v1/hard_cases.csv \
  --summary-output outputs/hierarchical/s1_static_revised_ik_shell_v1/hard_cases_summary.json \
  --architecture-version hierarchical_s1_revised_ik_shell_v1

python scripts/summarize_hierarchical_ik_rejections.py \
  --input outputs/hierarchical/s1_static_revised_ik_shell_v1/hard_cases.csv \
  --output outputs/hierarchical/s1_static_revised_ik_shell_v1/ik_rejection_summary.json
```

Recovery-gated timeout 诊断（两个终端并行，固定 seeds `9093/9143`）。终端 A 使用 shard 0：

```bash
export PYTHONPATH=src
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_recovery_gated_filter_timeout_diag.yaml \
  --nominal-only --episodes 2 --manifest-shard-index 0 --manifest-shard-count 2 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_filter_timeout_cases.json \
  --output outputs/hierarchical/s1_static_revised_recovery_gated_filter_timeout_diag/hard_cases_shard_0.csv \
  --trace-output outputs/hierarchical/s1_static_revised_recovery_gated_filter_timeout_diag/traces/shard_0
```

终端 B 使用同一命令，将 shard index 改为 `1`，并将输出改为 `hard_cases_shard_1.csv`、`traces/shard_1`。

两个 shard 完成后先合并：

```bash
export PYTHONPATH=src
python scripts/merge_hierarchical_evaluations.py \
  --input outputs/hierarchical/s1_static_revised_recovery_gated_filter_timeout_diag/hard_cases_shard_0.csv \
  --input outputs/hierarchical/s1_static_revised_recovery_gated_filter_timeout_diag/hard_cases_shard_1.csv \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_filter_timeout_cases.json \
  --episodes 2 \
  --output outputs/hierarchical/s1_static_revised_recovery_gated_filter_timeout_diag/hard_cases.csv \
  --summary-output outputs/hierarchical/s1_static_revised_recovery_gated_filter_timeout_diag/hard_cases_summary.json \
  --architecture-version hierarchical_s1_revised_recovery_gated_filter_timeout_diag_v1
```

诊断完成后汇总：

```bash
export PYTHONPATH=src
python scripts/analyze_filter_timeout_cases.py \
  --evaluation outputs/hierarchical/s1_static_revised_recovery_gated_filter_timeout_diag/hard_cases.csv \
  --trace-dir outputs/hierarchical/s1_static_revised_recovery_gated_filter_timeout_diag/traces \
  --seeds 9093 9143 \
  --output outputs/hierarchical/s1_static_revised_recovery_gated_filter_timeout_diag/timeout_analysis.json
```

该汇总还会输出 `mean_max_feasible_goal_velocity_mps`、`min_max_feasible_goal_velocity_mps`、`mean_projected_goal_velocity_mps`、`mean_goal_velocity_feasibility_gap_mps` 和正最大可行速度比例，用于区分安全集合受限与投影目标不合适。

严格 goal-velocity-priority 对照（仅两个 timeout seed；两个终端并行，不能覆盖 recovery-gated 输出）：

```bash
export PYTHONPATH=src
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_diag.yaml \
  --nominal-only --episodes 2 \
  --manifest-shard-index 0 --manifest-shard-count 2 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_filter_timeout_cases.json \
  --output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_diag/hard_cases_shard_0.csv \
  --trace-output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_diag/traces/shard_0
```

完整 23-reset hard-case 扩展（四终端并行；仅作为 latency/safety/success 联合审计，不运行 nominal final）：

```bash
export PYTHONPATH=src
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_hard_cases_v1.yaml \
  --nominal-only --episodes 23 \
  --manifest-shard-index 0 --manifest-shard-count 4 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json \
  --output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_hard_cases_v1/hard_cases_shard_0.csv \
  --trace-output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_hard_cases_v1/traces/shard_0
```

其余终端将 shard index、CSV 和 trace 子目录改为 `1`、`2`、`3`。完成后合并：

9027 时间一致性 smoke（先运行，单终端；截至当前文档更新尚未执行）：

```bash
export PYTHONPATH=src
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_9027_v1.yaml \
  --nominal-only --episodes 1 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_temporal_consistency_9027.json \
  --output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_9027_v1/hard_cases.csv \
  --trace-output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_9027_v1/traces
```

若 `9027` 成功且 trace 中 `safety_filter_command_sign_change` 明显下降，再运行三例 paired（三个终端分别使用 shard `0/1/2`）：

```bash
export PYTHONPATH=src
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_timeout_triplet_v1.yaml \
  --nominal-only --episodes 3 \
  --manifest-shard-index 0 --manifest-shard-count 3 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_temporal_consistency_timeout_triplet.json \
  --output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_timeout_triplet_v1/hard_cases_shard_0.csv \
  --trace-output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_timeout_triplet_v1/traces/shard_0
```

三 shard 完成后合并并分析：

```bash
export PYTHONPATH=src
python scripts/merge_hierarchical_evaluations.py \
  --input outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_timeout_triplet_v1/hard_cases_shard_0.csv \
  --input outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_timeout_triplet_v1/hard_cases_shard_1.csv \
  --input outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_timeout_triplet_v1/hard_cases_shard_2.csv \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_temporal_consistency_timeout_triplet.json \
  --episodes 3 \
  --output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_timeout_triplet_v1/hard_cases.csv \
  --summary-output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_timeout_triplet_v1/hard_cases_summary.json \
  --architecture-version hierarchical_s1_revised_recovery_gated_goal_velocity_temporal_consistency_timeout_triplet_v1

python scripts/analyze_filter_timeout_cases.py \
  --evaluation outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_timeout_triplet_v1/hard_cases.csv \
  --trace-dir outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_timeout_triplet_v1/traces \
  --seeds 9093 9143 9027 \
  --output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_temporal_consistency_timeout_triplet_v1/timeout_analysis.json
```

```bash
export PYTHONPATH=src
python scripts/merge_hierarchical_evaluations.py \
  --input outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_hard_cases_v1/hard_cases_shard_0.csv \
  --input outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_hard_cases_v1/hard_cases_shard_1.csv \
  --input outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_hard_cases_v1/hard_cases_shard_2.csv \
  --input outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_hard_cases_v1/hard_cases_shard_3.csv \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json \
  --episodes 23 \
  --output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_hard_cases_v1/hard_cases.csv \
  --summary-output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_hard_cases_v1/hard_cases_summary.json \
  --architecture-version hierarchical_s1_revised_recovery_gated_goal_velocity_priority_hard_cases_v1
```

终端 2 将 shard、CSV 和 trace 编号改为 `1`。完成后合并并分析：

```bash
export PYTHONPATH=src
python scripts/merge_hierarchical_evaluations.py \
  --input outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_diag/hard_cases_shard_0.csv \
  --input outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_diag/hard_cases_shard_1.csv \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_filter_timeout_cases.json \
  --episodes 2 \
  --output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_diag/hard_cases.csv \
  --summary-output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_diag/hard_cases_summary.json \
  --architecture-version hierarchical_s1_revised_recovery_gated_goal_velocity_priority_diag_v1

python scripts/analyze_filter_timeout_cases.py \
  --evaluation outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_diag/hard_cases.csv \
  --trace-dir outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_diag/traces \
  --seeds 9093 9143 \
  --output outputs/hierarchical/s1_static_revised_recovery_gated_goal_velocity_priority_diag/timeout_analysis.json
```

terminal recovery 归因对照（两个终端并行，均为 23 个 hard-case reset）：

终端 A：关闭主动 recovery，只保留严格 predictive safety filter。

```bash
export PYTHONPATH=src
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_filter_only_diag.yaml \
  --nominal-only --episodes 23 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json \
  --output outputs/hierarchical/s1_static_revised_filter_only_diag/hard_cases.csv \
  --trace-output outputs/hierarchical/s1_static_revised_filter_only_diag/traces
```

终端 B：保留 recovery，但将触发推迟到 `h<=0` 或 TTC `<=0.05s`。

```bash
export PYTHONPATH=src
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_recovery_late_diag.yaml \
  --nominal-only --episodes 23 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json \
  --output outputs/hierarchical/s1_static_revised_recovery_late_diag/hard_cases.csv \
  --trace-output outputs/hierarchical/s1_static_revised_recovery_late_diag/traces
```

两个评估结束后，可在两个终端并行汇总 task-space 诊断：

```bash
export PYTHONPATH=src
python scripts/analyze_terminal_diagnostics.py \
  --evaluation outputs/hierarchical/s1_static_revised_filter_only_diag/hard_cases.csv \
  --trace-dir outputs/hierarchical/s1_static_revised_filter_only_diag/traces \
  --output outputs/hierarchical/s1_static_revised_filter_only_diag/terminal_diagnostics.json
```

```bash
export PYTHONPATH=src
python scripts/analyze_terminal_diagnostics.py \
  --evaluation outputs/hierarchical/s1_static_revised_recovery_late_diag/hard_cases.csv \
  --trace-dir outputs/hierarchical/s1_static_revised_recovery_late_diag/traces \
  --output outputs/hierarchical/s1_static_revised_recovery_late_diag/terminal_diagnostics.json
```

正式候选 recovery-gated 复核（先跑 hard-case）：

```bash
export PYTHONPATH=src
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_recovery_gated.yaml \
  --nominal-only --episodes 23 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json \
  --output outputs/hierarchical/s1_static_revised_recovery_gated/hard_cases.csv \
  --trace-output outputs/hierarchical/s1_static_revised_recovery_gated/traces
```

只有 hard-case 成功数不低于 filter-only 且 collision 仍为 0，才运行完整 200-reset nominal：

```bash
export PYTHONPATH=src
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_recovery_gated.yaml \
  --nominal-only --episodes 200 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_final.json \
  --output outputs/hierarchical/s1_static_revised_recovery_gated/nominal_final.csv \
  --trace-output outputs/hierarchical/s1_static_revised_recovery_gated/nominal_traces
```

建议将上述 200 episodes 拆成四个终端并行（每个终端使用不同 shard，输出文件名必须唯一）：

```bash
export PYTHONPATH=src
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_recovery_gated.yaml \
  --nominal-only --episodes 200 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_final.json \
  --manifest-shard-index 0 --manifest-shard-count 4 \
  --output outputs/hierarchical/s1_static_revised_recovery_gated/nominal_final_shard_0.csv \
  --trace-output outputs/hierarchical/s1_static_revised_recovery_gated/nominal_traces/shard_0
```

其余三个终端把 shard index 改为 `1`、`2`、`3`，并同步修改输出 CSV 与 trace 子目录。四个 shard 完成后合并：

```bash
export PYTHONPATH=src
python scripts/merge_hierarchical_evaluations.py \
  --input outputs/hierarchical/s1_static_revised_recovery_gated/nominal_final_shard_0.csv \
  --input outputs/hierarchical/s1_static_revised_recovery_gated/nominal_final_shard_1.csv \
  --input outputs/hierarchical/s1_static_revised_recovery_gated/nominal_final_shard_2.csv \
  --input outputs/hierarchical/s1_static_revised_recovery_gated/nominal_final_shard_3.csv \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_final.json \
  --output outputs/hierarchical/s1_static_revised_recovery_gated/nominal_final.csv \
  --summary-output outputs/hierarchical/s1_static_revised_recovery_gated/nominal_final_summary.json \
  --architecture-version hierarchical_s1_revised_recovery_gated_v1
```

如果 v3 仍在 SERVO 中长期无进展，使用 v4 的 bounded terminal-stall replan 诊断：

```bash
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_tracker_v4.yaml \
  --nominal-only \
  --episodes 23 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_hierarchical_hard_cases.json \
  --output outputs/hierarchical/s1_static_revised_tracker_v4/hard_cases.csv \
  --trace-output outputs/hierarchical/s1_static_revised_tracker_v4/hard_case_traces
```

其余终端将 shard index 改为 `1`、`2`、`3`，最后合并：

```bash
python scripts/merge_hierarchical_evaluations.py \
  --input outputs/hierarchical/s1_static_revised/nominal_final_shard_0.csv \
  --input outputs/hierarchical/s1_static_revised/nominal_final_shard_1.csv \
  --input outputs/hierarchical/s1_static_revised/nominal_final_shard_2.csv \
  --input outputs/hierarchical/s1_static_revised/nominal_final_shard_3.csv \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_final.json \
  --output outputs/hierarchical/s1_static_revised/nominal_final.csv \
  --summary-output outputs/hierarchical/s1_static_revised/nominal_final_summary.json \
  --architecture-version hierarchical_s1_revised_v2
```

训练 revised Residual（仅 nominal 门槛通过后；三个终端分别使用 seed-specific config）：

```bash
python scripts/train.py \
  --config configs/experiments/hierarchical/s1_static_revised_seed4401.yaml
```

评估训练 actor：

```bash
python scripts/evaluate.py \
  --config configs/experiments/hierarchical/s1_static_revised_seed4401.yaml \
  --checkpoint outputs/hierarchical/s1_static_revised/<run>/actor.pt \
  --episodes 200 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_final.json
```

旧/新 nominal 或旧/新 S1-R 完成后，逐 reset paired 对比：

```bash
python scripts/compare_hierarchical_evaluations.py \
  --old outputs/hierarchical/s1_static/nominal_final.csv \
  --new outputs/hierarchical/s1_static_revised/nominal_final.csv \
  --output outputs/hierarchical/s1_static_revised/nominal_paired_comparison.json
```

以上连续 seed 仅展示接口。正式运行必须使用冻结 manifest，不得以连续 seed 代替。

## 6. 进入 S2 的条件

- S1-N plan-conditioned success `>=95%`；
- full success `>=90%`，或未达到部分已有稳定的 IK/plan failure 归因；
- 物理接触保持低，safety filter 位于唯一命令出口；
- S1-R 已完成至少 3 个训练 seed，并与 S1-N 完成成对消融；
- planning、状态、nominal/residual/filter 指标能由评估 CSV 复现。

进入 S2 后不得修改观测排列、actor action semantics 或状态机名称；否则视为新架构版本，已有 checkpoint 和 replay 均不兼容。
