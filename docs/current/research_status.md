# 当前研究状态

> 更新时间：2026-08-14。本页是当前实验入口；更细的无障碍基线协议见
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
`9001--9200`。

| stage | pooled success | timeout | collision_any | mean final error | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| frozen S0 actor | `428/600=71.33%` | `156/600=26.0%` | `16/600=2.67%` | `0.0898 m` | 静态障碍下明显退化 |
| S1 fine-tune | `501/600=83.50%` | `99/600=16.5%` | `0/600=0%` | `0.0757 m` | 碰撞清零，但仍有大量 timeout |
| R1 extend +100k | `501/600=83.50%` | `99/600=16.5%` | `0/600=0%` | `0.0757 m` | 继续训练没有带来 full final 收益 |

S1 fine-tune 选中的 checkpoint 为：

| train seed | selected step | final success |
| ---: | ---: | ---: |
| 4301 | 440000 | `175/200=87.5%` |
| 4302 | 480000 | `164/200=82.0%` |
| 4303 | 420000 | `162/200=81.0%` |

R1 extend +100k 后，validation 选择的是续训后的第一个 checkpoint
（4301 step 460000、4302 step 500000、4303 step 440000），full final
结果与 S1 fine-tune 完全持平，因此当前证据不支持继续盲目增加训练步数。

产物：

- frozen S1：`outputs/reaching_incremental/s1_static_obstacle/summary.json`
- S1 fine-tune：`outputs/reaching_incremental/s1_static_obstacle_finetune/summary.json`
- R1 extend +100k：`outputs/reaching_incremental/s1_static_obstacle_finetune_r1_extend100k/summary.json`

## 当前问题

S1 的主要剩余问题不是碰撞，而是终端收敛和 timeout。fine-tune 已将
`collision_any` 从 `16/600` 降到 `0/600`，但 timeout 仍为
`99/600=16.5%`。这说明策略在静态障碍下更安全，但仍经常无法在 12 s
内到达 `0.055 m` success threshold。

S0 结果不授权任何动态避障、安全、OOD、真机、viability monitor、
safety filter、recovery 或 relaxation 结论。S1 也必须继续分别报告
`success`、最终误差、`collision_any`、capsule overlap 和 physical
contact。

## 唯一下一步

下一步不是 S2，也不是继续简单加训练步数，而是做 S1-R2 诊断和修复：

1. 建立独立 hard-case curriculum，不使用 `9001--9200` final seeds 训练。
2. 重点针对 timeout episode，提高 terminal convergence，而不是放宽
   collision 或 success threshold。
3. 维持 `collision_any=0` 作为硬约束式报告目标。
4. 若调参过程中继续参考 `9001--9200` final 失败样本，后续必须新增
   blind final manifest 才能形成最终结论。

只有 S1 静态障碍物在明确口径下稳定后，才允许讨论 S2 单动态障碍物
residual 避障。
