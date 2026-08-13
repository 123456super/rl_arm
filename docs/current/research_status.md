# 当前研究状态

> 更新时间：2026-08-13。本页是当前实验入口；更细的无障碍基线协议见
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

## 当前阻塞

S1 静态障碍物结果已被清理出当前 outputs，因此不能继续引用旧静态
障碍物数字作为当前证据。当前缺口不是无障碍 reaching，而是需要在
同一批冻结 v2 actor 上重建一个可复现的静态障碍物评估和迁移闭环。

S0 结果不授权任何动态避障、安全、OOD、真机、viability monitor、
safety filter、recovery 或 relaxation 结论。S1 也必须继续分别报告
`success`、最终误差、`collision_any`、capsule overlap 和 physical
contact。

## 唯一下一步

下一步是执行 S1 静态障碍物协议：

1. 用冻结 v2 actor 跑 `random` 零速度静态障碍物 final 评估。
2. 汇总三 seed 的 full final 指标，确认 S1 冻结 actor 的失败类型。
3. 如果 S1 冻结 actor 不通过，再从对应 v2 checkpoint 进行静态障碍物
   fine-tune。
4. 使用独立 `9301--9340` validation manifest 选择 checkpoint。
5. 使用完整 `9001--9200` final manifest 复核，不删除 reset、不放宽
   `0.055 m` success threshold。

只有 S1 静态障碍物在明确口径下稳定后，才允许讨论 S2 单动态障碍物
residual 避障。
