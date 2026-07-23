# 2026-07-23 最新训练进展报告

## 1. 当前进展

本轮重点完成了 `ldrc_adaptive` seed101 的 checkpoint 保存、选择和正式复评。

已完成事项：

- 修改训练保存机制，周期 checkpoint 不再覆盖。
- seed101 `ldrc_adaptive` 重跑到约 71.4k，中断前已保存 10k-70k checkpoint。
- 从 70k checkpoint 做近似续训，补出 80k、90k、100k checkpoint。
- 完成 40k-100k checkpoint 的 20 episodes validation。
- 对 40k 和 60k 做独立 `eval_seed=1001`、100 episodes 正式评估。

当前结论：seed101 `ldrc_adaptive` 的最终 100k checkpoint 不是最优，40k checkpoint 是当前正式结果的最佳选择。

## 2. 代码和链路变化

训练脚本已支持周期保存独立 checkpoint：

```text
actor_step_10000.pt
agent_state_step_10000.pt
...
actor_step_100000.pt
agent_state_step_100000.pt
```

同时继续保留兼容文件：

```text
actor.pt
agent_state.pt
```

另外增加了近似续训入口，可从指定 actor 和 agent_state 继续训练：

```text
--resume-actor
--resume-state
--start-step
--run-name
```

限制：当前续训不是严格无缝续训，因为 replay buffer、optimizer 和环境随机状态没有保存。

## 3. Checkpoint Validation 结果

validation 设置：

- 方法：`ldrc_adaptive`
- train seed：101
- validation seed：2001
- episodes：20
- 场景：`random_crossing`

结果目录：

```text
outputs/formal/eval/checkpoint_selection
```

| checkpoint | success_rate | collision_rate | non_end_link_collision_rate | mean_final_position_error | mean_min_distance | safety_violation_rate | mean_rms_jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40k | 40.0% | 10.0% | 10.0% | 0.170m | 0.141m | 2.74% | 72.39 |
| 50k | 35.0% | 20.0% | 15.0% | 0.172m | 0.120m | 7.93% | 83.64 |
| 60k | 45.0% | 5.0% | 5.0% | 0.170m | 0.110m | 9.17% | 88.71 |
| 70k | 40.0% | 25.0% | 20.0% | 0.240m | 0.127m | 9.57% | 96.04 |
| 80k | 10.0% | 30.0% | 25.0% | 0.422m | 0.097m | 9.55% | 71.62 |
| 90k | 20.0% | 25.0% | 25.0% | 0.458m | 0.097m | 9.57% | 82.18 |
| 100k | 10.0% | 20.0% | 20.0% | 0.448m | 0.099m | 8.83% | 79.09 |

validation 中 60k 排名第一，但 40k 的安全指标更好。

## 4. 正式 100 Episodes 复评

正式评估设置：

- eval seed：1001
- episodes：100
- 候选 checkpoint：40k、60k

结果文件：

```text
outputs/formal/eval/ldrc_adaptive_seed101_step40000_eval100_seed1001.csv
outputs/formal/eval/ldrc_adaptive_seed101_step60000_eval100_seed1001.csv
```

| checkpoint | success_rate | collision_rate | non_end_link_collision_rate | mean_final_position_error | mean_min_distance | safety_violation_rate | mean_rms_jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40k | 47.0% | 7.0% | 7.0% | 0.166m | 0.157m | 2.41% | 57.71 |
| 60k | 37.0% | 10.0% | 8.0% | 0.224m | 0.128m | 8.38% | 80.96 |

40k 在正式测试中全面优于 60k：

- 成功率更高：47.0% vs 37.0%
- 碰撞率更低：7.0% vs 10.0%
- 非末端碰撞率更低：7.0% vs 8.0%
- 最终位置误差更低：0.166m vs 0.224m
- 平均最小距离更高：0.157m vs 0.128m
- 安全违反率更低：2.41% vs 8.38%
- jerk 更低：57.71 vs 80.96

## 5. 当前判断

1. `ldrc_adaptive` seed101 的最优正式 checkpoint 应固定为 40k。
2. 80k 之后策略明显退化，说明后期训练不稳定问题真实存在。
3. 20 episodes validation 可以用于初筛，但结果仍有方差；正式结论必须以独立 seed 的 100 episodes 为准。
4. 当前 `ldrc_adaptive` seed101 的正式结果已从原最终 checkpoint 的 14.0% success_rate 提升到 47.0%。

推荐正式采用：

```text
outputs/formal/train/ldrc_adaptive/seed_101/ldrc_adaptive_seed101_steps100000/actor_step_40000.pt
```

对应正式评估结果：

```text
outputs/formal/eval/ldrc_adaptive_seed101_step40000_eval100_seed1001.csv
```

## 6. 下一步建议

当前不建议继续扩大 seed，也不建议立即大范围调参。

下一步优先做：

```text
把 checkpoint selection 流程应用到其它方法 -> 重新获得公平的四方法对比 -> 再决定是否调参
```

具体顺序：

1. 对 `ee_fixed`、`link_fixed`、`ldrc_fixed` 也使用周期 checkpoint 机制重跑或补跑。
2. 每个方法用同一 validation seed 做 checkpoint selection。
3. 每个方法选定 checkpoint 后，再用独立 eval seed 做 100 episodes 正式评估。
4. 汇总新的四方法结果，替换只看最终 checkpoint 的初筛结果。

如果后续仍出现明显后期退化，再进入小范围调参，优先关注 `lambda` 衰减、`c_safe`、cost 权重和 smoothing 参数。
