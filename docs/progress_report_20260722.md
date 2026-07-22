# 2026-07-22 最新训练进展报告

## 1. 当前进展

截至当前节点，项目已经跑通 UR5 仿真训练和评估链路，并完成两类关键实验：

- `ldrc_adaptive` 单方法 100k 训练，seed42。
- 四方法正式初筛，seed101，场景为 `random_crossing`，评估种子为 `1001`，每个方法 100 episodes。

前置链路，包括环境、URDF、训练脚本、checkpoint、评估脚本和结果落盘，已经可以正常使用。当前重点不再是基础可运行性，而是训练稳定性、checkpoint 选择和正式实验设计。

## 2. 当前最好单方法结果

目前表现最好的单方法结果来自：

```text
outputs/runs/20260722_114809_ur5_ldrc_adaptive_seed42_steps100000
```

100 episode 评估结果：

| 指标 | 结果 |
| --- | ---: |
| success_rate | 51.0% |
| collision_rate | 11.0% |
| non_end_link_collision_rate | 9.0% |
| mean_final_position_error | 0.170m |
| mean_min_distance | 0.138m |
| mean_safety_violation_rate | 5.83% |
| mean_rms_jerk | 67.91 |

该结果说明 `ldrc_adaptive` 已经能学到有效避障策略，可以作为阶段性 baseline。

但训练曲线显示，seed42 的 80k-90k 窗口可能比最终 100k 更稳；由于当前训练只保留最终 `actor.pt`，无法回退评估中间 checkpoint。

## 3. 四方法正式初筛结果

已完成 seed101 下四个方法的 100k 训练，并在 `random_crossing` 场景用 `eval_seed=1001` 做了 100 episode 评估。

评估结果：

| 方法 | success_rate | collision_rate | non_end_link_collision_rate | mean_min_distance | safety_violation_rate | mean_rms_jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ee_fixed` | 16.0% | 15.0% | 15.0% | 0.184m | 2.54% | 21.69 |
| `link_fixed` | 7.0% | 18.0% | 17.0% | 0.193m | 2.15% | 23.82 |
| `ldrc_fixed` | 4.0% | 53.0% | 49.0% | 0.063m | 32.20% | 38.52 |
| `ldrc_adaptive` | 14.0% | 24.0% | 17.0% | 0.117m | 6.04% | 57.61 |

当前判断：

- seed101 的最终 checkpoint 整体表现不好，不适合作为正式论文结论。
- `ldrc_fixed` 后期明显崩溃，碰撞率和非末端碰撞率过高。
- `ldrc_adaptive` 比 `ldrc_fixed` 安全得多，但成功率没有超过 `ee_fixed`。
- seed42 和 seed101 的 `ldrc_adaptive` 差异很大，说明当前训练对 seed 或最终 checkpoint 较敏感。

## 4. 当前核心问题

### 4.1 最终 checkpoint 不一定最优

当前训练脚本的周期保存会覆盖同一个 `actor.pt`。因此训练过程中 40k、60k、80k 等阶段的策略没有被保留下来。

这会带来一个问题：如果策略在中间阶段更好、后期退化，最终评估只能看到退化后的模型。

从已有结果看，这个问题已经出现：

- seed42 的 `ldrc_adaptive` 在 80k-90k 训练窗口表现更好。
- seed101 的 `ldrc_fixed` 后期崩溃。
- seed101 的 `ldrc_adaptive` 后期表现弱于预期。

### 4.2 训练稳定性不足

`ldrc_adaptive` 在 seed42 能达到 51% 成功率和 11% 碰撞率，但 seed101 的最终 checkpoint 只有 14% 成功率和 24% 碰撞率。

这说明当前还不能直接扩大多 seed 正式实验，否则结果会混入大量 checkpoint 选择和训练退化噪声。

### 4.3 约束强度可能退化

部分训练后期 `lambda` 降到接近 0。若此时安全违反和碰撞仍然存在，说明风险约束没有持续提供足够压力。

该问题后续需要结合 `episode_cost`、`c_safe`、`cost_ema`、`lambda_lr` 和安全违反率进一步分析。

## 5. 下一步建议

### 5.1 先改 checkpoint 保存机制

优先修改 `scripts/train.py`，在每个 `save_interval` 保存独立 checkpoint，例如：

```text
actor_step_20000.pt
agent_state_step_20000.pt
actor_step_40000.pt
agent_state_step_40000.pt
actor_step_60000.pt
agent_state_step_60000.pt
actor_step_80000.pt
agent_state_step_80000.pt
actor_step_100000.pt
agent_state_step_100000.pt
```

同时继续保留最终兼容文件：

```text
actor.pt
agent_state.pt
```

### 5.2 再重跑 seed101 的 `ldrc_adaptive`

修改保存机制后，先重跑：

```bash
conda run --no-capture-output -n rl python scripts/train.py \
  --config outputs/formal/train/configs/ldrc_adaptive_seed101_steps100000.yaml
```

然后分别评估 40k、60k、80k、100k checkpoint，确认是否存在中间 checkpoint 明显优于最终 checkpoint。

### 5.3 建立 checkpoint 选择规则

建议正式实验采用固定规则选择 checkpoint：

1. 每个周期 checkpoint 用 validation seed 做 20 episode 小评估。
2. 优先选择 success_rate 最高的 checkpoint。
3. 若 success_rate 相同，选择 collision_rate 更低的 checkpoint。
4. 若仍相同，选择 mean_min_distance 更高的 checkpoint。
5. 最终正式测试使用独立 eval seed，避免和 validation seed 混用。

### 5.4 暂缓大范围调参

当前不建议马上调整大量参数，也不建议直接补 seed202 和 seed303。

更合理的顺序是：

1. 先解决 checkpoint 保存和选择问题。
2. 再验证 seed101 的 `ldrc_adaptive` 是否只是最终 checkpoint 选错。
3. 如果不同 checkpoint 都不稳定，再小范围调参。

后续可优先考虑的调参方向：

- 降低 `risk.cost.c_safe`，加强风险约束触发。
- 提高 `risk.cost.k_violation` 或 `risk.cost.k_collision`。
- 调整 `sac.lambda_lr` 和 `sac.cost_ema_rho`，避免 `lambda` 过早衰减。
- 调整 `smoothing.beta_min` 和 `smoothing.beta_max`，平衡高风险响应和低风险平滑。

## 6. 当前推荐决策

当前阶段不要继续盲目扩大训练矩阵。

推荐下一步只做一件主线工作：

```text
加入周期 checkpoint 保存 -> 重跑 seed101 ldrc_adaptive -> 评估中间 checkpoint -> 确定是否需要调参
```

如果中间 checkpoint 明显更好，就把 checkpoint 选择机制纳入正式实验流程。  
如果中间 checkpoint 仍然不稳定，再进入小范围参数优化。
