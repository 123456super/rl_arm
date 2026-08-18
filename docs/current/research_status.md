# 当前研究状态

> 更新时间：2026-08-18。本页是当前实验入口；更细的无障碍基线协议见
> [基础 reaching 恢复协议](reaching_recovery_protocol.md)，S1 执行链路见
> [S1 静态障碍物协议](s1_static_obstacle_protocol.md)。

## 当前结论

当前论文路线收敛为：

```text
S0 无障碍 reaching 基线
  -> S1 单个静态障碍物迁移
  -> S2 单个动态障碍物 residual 避障
```

S0 已完成并冻结。三个独立 v2 actor 在固定 final manifest
`9001--9200` 上均为全量 `194/200=97.0%`，pooled 为
`582/600=97.0%`。固定离线 IK 可达且无障碍候选路径找到的 194
reset 子集上，三个 actor 均为 `194/194=100%`，pooled 为
`582/582=100%`。

三个 actor 的失败 reset 完全一致：`9021, 9065, 9095, 9098, 9120,
9142`。这些 episode 均为 12 s timeout，`collision_any`、capsule
overlap 和 physical contact 均为零。由于这些 reset 在固定多初值
IK 搜索中未找到候选，S0 全量口径不能声明 99%。

S1 已完成基线重建、静态 fine-tune、candidate terminal refine，以及
R3/R4 blind final 闭环。当前最稳妥的结论是：

1. 在原始 final manifest `9001--9200` 上，`candidate terminal refine`
   仍是已完成方案里的最好结果，pooled 为 `521/600=86.83%`。
2. 在 disjoint blind manifest `9401--9600` 上，R3 candidate repair 与
   R4 hard-case repair 均为 `505/600=84.17%`，且三组 seed 的 blind
   final CSV 完全一致，因此当前没有证据表明 R4 hard-case curriculum
   带来了额外泛化收益。
3. 当前剩余失败仍全部是 timeout，不是碰撞；4302 仍是主要短板 seed。

## 冻结资产

| train seed | selected step | actor |
| ---: | ---: | --- |
| 4301 | 240000 | `outputs/reaching_recovery_v2/train/seed_4301/link_fixed_no_obstacle_speed100_seed4301_steps300000/actor_step_240000.pt` |
| 4302 | 280000 | `outputs/reaching_recovery_v2/train/seed_4302/link_fixed_no_obstacle_speed100_seed4302_steps300000/actor_step_280000.pt` |
| 4303 | 220000 | `outputs/reaching_recovery_v2/train/seed_4303/link_fixed_no_obstacle_speed100_seed4303_steps300000/actor_step_220000.pt` |

冻结清单保存于
`configs/experiments/reaching_incremental/frozen_reaching_v2_actors.json`。
后续 S1/S2 不得静默替换 actor、validation manifest、final manifest
或 success threshold。

## S1 当前进展

S1 静态障碍物链路已在同一批冻结 v2 actor 上重建，评估条件为
`scenario=random`、`speed_range=[0.0, 0.0]`、关闭 safety filter、
viability monitor、diagnostic logging、recovery、constraint
relaxation 和 maximin clearance 分支，final manifest 仍为完整
`9001--9200`。后续 R3/R4 的 blind final 使用冻结的 disjoint manifest
`9401--9600`，从而避免继续把 `9001--9200` 上的失败样本直接写成最终结论。

原始 final manifest `9001--9200` 上的完整结果如下：

| stage | pooled success | timeout | collision_any | mean final error | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| frozen S0 actor | `428/600=71.33%` | `156/600=26.0%` | `16/600=2.67%` | `0.0898 m` | 静态障碍下明显退化 |
| S1 fine-tune | `501/600=83.50%` | `99/600=16.5%` | `0/600=0%` | `0.0757 m` | 碰撞清零，但仍有大量 timeout |
| R1 extend +100k | `501/600=83.50%` | `99/600=16.5%` | `0/600=0%` | `0.0757 m` | 继续训练没有带来 full final 收益 |
| candidate terminal refine | `521/600=86.83%` | `79/600=13.17%` | `0/600=0%` | `0.0699 m` | focused terminal 修复有效，但仍未稳定收敛 |

blind final manifest `9401--9600` 上的 held-out 结果如下：

| stage | pooled success | timeout | collision_any | mean final error | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| R3 candidate repair | `505/600=84.17%` | `95/600=15.83%` | `0/600=0%` | `0.0750 m` | 选点为 530k / 500k / 550k |
| R4 hard-case repair | `505/600=84.17%` | `95/600=15.83%` | `0/600=0%` | `0.0750 m` | 选点为 540k / 510k / 560k；4301 因 OOM 改为 `batch_size=128` |

R3 与 R4 的 blind final 不只是 summary 一样，而是三个 seed 的 blind
episode 结果逐文件完全一致；因此这轮证据支持 “R3≈R4”，不支持
“R4 hard-case 单独修复带来了额外 blind gain”。

S1 fine-tune 选中的 checkpoint 为：

| train seed | selected step | final success |
| ---: | ---: | ---: |
| 4301 | 440000 | `175/200=87.5%` |
| 4302 | 480000 | `164/200=82.0%` |
| 4303 | 420000 | `162/200=81.0%` |

R1 extend +100k 后，validation 选择的是续训后的第一个 checkpoint
（4301 step 460000、4302 step 500000、4303 step 440000），full final
结果与 S1 fine-tune 完全持平，因此当前证据不支持继续盲目增加训练步数。

candidate terminal refine 只针对 `static_candidate_path_status=
candidate_path_found` 的 39 个 reset 做 focused curriculum 和 terminal
shaping，full final 提升到 `521/600=86.83%`。这说明一部分 timeout
确实是终端收敛问题，但 4302 没有改善，且仍有 79 个 timeout，说明 S1
还没有到稳定可交付的程度。

R3 candidate repair 从 candidate terminal refine 的 selected checkpoint
继续，只针对 `candidate_path_found` 残余失败做 repair；R4 hard-case
repair 则把 `not_found` / `not_checked_due_to_ik` 的 hard cases 单独建桶。
两条线的 validation selected checkpoint 确实不同，但 blind final 行为
表现等价，说明当前 hard-case curriculum 至少在这轮设置下没有转化为可见的
held-out 提升。

产物：

- frozen S1：`outputs/reaching_incremental/s1_static_obstacle/summary.json`
- S1 fine-tune：`outputs/reaching_incremental/s1_static_obstacle_finetune/summary.json`
- R1 extend +100k：`outputs/reaching_incremental/s1_static_obstacle_finetune_r1_extend100k/summary.json`
- candidate terminal refine：`outputs/reaching_incremental/s1_static_candidate_terminal_refine/summary.json`
- R3 blind final：`outputs/reaching_incremental/s1_r3_candidate_repair/eval_blind/summary.json`
- R4 blind final：`outputs/reaching_incremental/s1_r4_hard_case_repair/eval_blind/summary.json`

## 当前问题

S1 的主要剩余问题不是碰撞，而是终端收敛和 timeout。fine-tune 已将
`collision_any` 从 `16/600` 降到 `0/600`，但 timeout 仍为
`79/600=13.17%`。candidate terminal refine 进一步把 success 提到
`86.83%`，但剩余失败仍全部是 timeout，说明策略在静态障碍下更安全，
却对一部分 hard cases 仍无法在 12 s 内到达 `0.055 m` success
threshold。

最新诊断显示，剩余失败一半以上仍来自 `not_found` /
`not_checked_due_to_ik` 的 hard cases；candidate-path-found 样本上的
near-goal regression / nonconvergent timeout 已被显著压缩，但还没有清零。

blind held-out 上，R3 与 R4 都是 `505/600=84.17%`，且失败仍全部为
timeout。换句话说，这一轮已经说明：

1. hard-case 确实是一个独立桶，但把它从 candidate-path repair 里拆开训练，
   还没有带来额外的 blind 泛化收益。
2. 4302 仍然是主要短板，blind final 只有 `161/200=80.5%`，明显低于
   4301 的 `178/200=89.0%`。
3. 当前问题的核心仍是 terminal convergence / hard reset reachability，
   不是碰撞控制或安全约束。

这轮还暴露了两个执行层面的记录问题，需要在后续定版时显式保留：

1. R4-4301 在 `batch_size=256` 下首次 critic update 触发 CUDA OOM，
   因此 base config 被改成 `batch_size=128` 后重跑；但 4302/4303 已跑完，
   仍保留 `256`。所以现有 R4 三个 seed 不是严格同配置实验。
2. `scripts/run_s1_r3_r4_repairs.sh summarize-blind` 曾触发 `set -u` 下的
   `local ... out=...` 未绑定变量问题，现已修复，但这也意味着 blind
   汇总脚本曾经需要额外人工确认。

S0 结果不授权任何动态避障、安全、OOD、真机、viability monitor、
safety filter、recovery 或 relaxation 结论。S1 也必须继续分别报告
`success`、最终误差、`collision_any`、capsule overlap 和 physical
contact。

## 当前建议下一步

下一步不是 S2，也不是继续沿当前 R4 线盲目加训练，而是先把 S1 的剩余问题
收敛清楚：

1. 优先分析 4302 与 blind timeout 的失败簇，先回答为什么 R3/R4 的 selected
   checkpoint 已变化，但 blind 行为仍完全等价。
2. 继续保持 `candidate_path_found` 与 `not_found` /
   `not_checked_due_to_ik` 分桶，不要再混成单一 repair 目标。
3. 若要把 R4 作为正式 ablation/定版证据，先统一三组 seed 的配置，再重新做
   blind final；在当前 mixed-batch 配置下，不宜把 R4 写成“已正式验证更优”。
4. 维持 `collision_any=0` 作为硬约束式报告目标，不为提 success 牺牲安全口径。

只有 S1 静态障碍物在明确口径下稳定后，才允许讨论 S2 单动态障碍物
residual 避障。
