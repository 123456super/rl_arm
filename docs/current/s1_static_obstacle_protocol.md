# S1 静态障碍物实验协议

> 更新时间：2026-08-20。状态：原 S1 final 已冻结；recovery-gated nominal 已完成并通过 gate；Residual 训练按当前决策暂缓。本文只规定 S1 场景和执行细节；动作语义、状态机、安全出口、通用指标与版本规则继承[分层控制通用实验协议](hierarchical_control_protocol.md)。

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
