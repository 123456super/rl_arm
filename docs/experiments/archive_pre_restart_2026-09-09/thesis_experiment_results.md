# 旧实验结果归档

> **归档状态：** 本文件于 2026-09-09 整体封存。下列数据只用于排查历史实现与实验口径，不得进入新论文结果、参数选择或方法门控。
> 当前方法与论文架构见 [论文大纲](../../thesis/thesis_outline.md)；历史任务状态见 [归档跟踪](thesis_experiment_tracker.md)；新实验数据只能写入 [新结果事实源](../restart_2026-09-09/experiment_results.md)。

> 方法路线已根据门控结果调整：最终 Proposed 使用冻结的 Instant-Link-SAC 名义 actor、候选动作条件预测、轨迹嵌入式多子步安全 QP，以及“全身模型内非线性验收—反例约束增广”的有限细化。第 9--11 节 Predictive-Link-SAC 结果作为负消融；第 7--8 节 endpoint/quintic 结果作为运动连续性过渡消融，均不得改写成最终 Proposed 已验证。

> 代码语义审计（2026-09-09）：历史配置名 `minimal_qp` 实际调用迭代半空间投影器，不是标准 QP 求解器；CSV 字段 `safety_qp_infeasible` 表示有限次投影后仍有约束残差，不能解释为数学上的 QP 不可行；当前 `active_constraints` 实现统计的是残差不超阈值的约束数量，也不能解释为真正活跃集；`physics_*_acceleration/jerk` 由发给 PyBullet 的 240 Hz 子步速度命令计算，不是关节反馈测量值。现有环境只在每个 20 Hz 控制周期结束后计算距离、碰撞和安全违反，因此第 2--11 节安全结果均属于控制周期采样结果。最终主实验必须先增加逐物理子步状态、距离和接触监测，并让全部变体使用同一新口径。

## 1. 统计口径、数据审计与使用状态

旧三方法主比较保存了 3 个训练随机种子、5 个场景、3 种方法、3 个评估随机种子，每个评估 seed 运行 100 episodes，共 13,500 行：

```text
3 train seeds x 5 scenarios x 3 methods x 3 eval seeds x 100 episodes
```

数据审计发现旧版 `scripts/evaluate.py` 使用 `episode_seed = eval_seed + episode`。当 eval seeds 为 `1004`、`1005`、`1006` 且各运行 100 episodes 时，三组实际 seed 范围几乎完全重叠。因此旧表的 13,500 行只有 4,590 个唯一 episode seed，不能表述为 13,500 个独立 episodes。第 2--4 节保留旧数值仅用于历史方向性分析；修复后的正式结果见第 5 节。

修复后的评估使用 Cantor pairing 将 `(eval_seed, episode_index)` 一一映射为 `episode_seed`，每个 CSV 显式记录该字段，汇总 manifest 同时检查预期行数和重复实验单元。正式统计仍采用：每个场景-方法-train seed 先合并三个 eval seed 的 300 个独立 episodes，再计算三个 train seed 均值之间的 `mean +/- sample std`（`n=3`）。

原始汇总文件：

```text
outputs/archive_pre_restart_2026-09-09/current_results/heldout_1004_1006/final_3methods/all_eval_episodes.csv
outputs/archive_pre_restart_2026-09-09/current_results/heldout_1004_1006/final_3methods/eval_summary_by_train_seed.csv
outputs/archive_pre_restart_2026-09-09/current_results/heldout_1004_1006/final_3methods/eval_summary_across_train_seeds.csv
outputs/archive_pre_restart_2026-09-09/current_results/heldout_1004_1006/final_3methods/eval_summary_macro_across_train_seeds.csv
```

主比较方法：

| 方法 | 含义 |
| --- | --- |
| `ee_fixed` | 末端风险 + 固定风险惩罚 SAC + 固定平滑 |
| `link_fixed_penalty1` | 连杆级动态风险 + 固定风险惩罚 SAC + 固定平滑，`w_R=1.0` |
| `ldrc_fixed` | 连杆级动态风险 + 约束 SAC + 固定平滑 |

`w_R=1.0` 由 train seeds `101`、`202` 的小规模筛选提出，随后在相同三个 train seeds 上重训。旧 eval seed 标签与训练/验证集合分离，但其 episode 实例因错误派生规则发生重叠；此外完整的超参数选择和训练随机性也并非完全独立。

## 2. 历史 Random Crossing 三方法对比（seed 重叠，仅作历史）

每种方法保存 900 行旧评估结果，但存在 episode seed 重叠，不作为最终正式统计。

| 方法 | success rate | collision rate | non-end-link collision | final position error | min distance | safety violation rate | RMS jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ee_fixed` | 17.0% +/- 2.0% | 13.44% +/- 5.19% | 13.11% +/- 5.17% | 0.369 +/- 0.017 m | 0.177 +/- 0.012 m | 2.38% +/- 1.44% | 25.0 +/- 2.0 |
| `link_fixed_penalty1` | 64.2% +/- 12.5% | 6.89% +/- 2.22% | 6.89% +/- 2.22% | 0.125 +/- 0.029 m | 0.193 +/- 0.020 m | 1.27% +/- 0.92% | 32.3 +/- 2.7 |
| `ldrc_fixed` | 56.3% +/- 18.0% | 5.56% +/- 2.91% | 5.56% +/- 2.91% | 0.172 +/- 0.054 m | 0.167 +/- 0.008 m | 3.12% +/- 1.27% | 32.7 +/- 0.2 |

旧数据中，`link_fixed_penalty1` 的成功率最高，最终位置误差、安全距离违反率和最小距离优于 `ldrc_fixed`；`ldrc_fixed` 的碰撞率略低。修复后的正式结果不再保持这一碰撞率排序，见第 5 节。

## 3. 历史非末端区域压力测试（seed 重叠，仅作历史）

每种“区域-方法”保存 900 行旧评估结果，但存在 episode seed 重叠。

| 区域 | 方法 | success rate | collision rate | final position error | RMS jerk |
| --- | --- | ---: | ---: | ---: | ---: |
| upper arm | `ee_fixed` | 13.3% +/- 0.6% | 3.00% +/- 3.61% | 0.414 +/- 0.034 m | 25.6 +/- 1.4 |
| upper arm | `link_fixed_penalty1` | 56.7% +/- 17.1% | 11.67% +/- 7.51% | 0.154 +/- 0.063 m | 34.8 +/- 3.9 |
| upper arm | `ldrc_fixed` | 54.2% +/- 19.3% | 9.33% +/- 0.58% | 0.180 +/- 0.083 m | 35.5 +/- 1.5 |
| elbow | `ee_fixed` | 16.7% +/- 1.2% | 5.67% +/- 3.21% | 0.381 +/- 0.008 m | 25.2 +/- 1.7 |
| elbow | `link_fixed_penalty1` | 63.8% +/- 16.4% | 8.67% +/- 5.03% | 0.135 +/- 0.049 m | 33.0 +/- 3.0 |
| elbow | `ldrc_fixed` | 56.6% +/- 24.1% | 7.33% +/- 2.31% | 0.169 +/- 0.079 m | 35.3 +/- 1.7 |
| forearm | `ee_fixed` | 19.4% +/- 5.5% | 5.33% +/- 3.21% | 0.390 +/- 0.029 m | 23.9 +/- 1.5 |
| forearm | `link_fixed_penalty1` | 72.8% +/- 16.2% | 3.00% +/- 2.65% | 0.109 +/- 0.037 m | 33.1 +/- 3.9 |
| forearm | `ldrc_fixed` | 67.0% +/- 18.2% | 4.33% +/- 1.15% | 0.131 +/- 0.052 m | 34.8 +/- 0.7 |
| wrist | `ee_fixed` | 15.8% +/- 3.2% | 11.00% +/- 3.46% | 0.429 +/- 0.038 m | 23.8 +/- 2.1 |
| wrist | `link_fixed_penalty1` | 76.1% +/- 14.7% | 2.33% +/- 2.52% | 0.099 +/- 0.031 m | 32.0 +/- 2.5 |
| wrist | `ldrc_fixed` | 62.3% +/- 25.4% | 3.00% +/- 1.00% | 0.147 +/- 0.071 m | 33.8 +/- 1.3 |

旧数据中，`link_fixed_penalty1` 在五个场景均获得最高平均成功率；在 upper arm 和 elbow 场景，`ldrc_fixed` 的碰撞率较低。该段只描述旧表；修复后的分场景结论见第 5.3 节。

## 4. 历史五场景宏平均（seed 重叠，仅作历史）

| 方法 | success rate | collision rate | final position error | min distance | safety violation rate | RMS jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ee_fixed` | 16.4% +/- 1.4% | 7.69% +/- 2.68% | 0.396 +/- 0.014 m | 0.189 +/- 0.013 m | 1.69% +/- 0.61% | 24.7 +/- 1.5 |
| `link_fixed_penalty1` | 66.7% +/- 15.3% | 6.51% +/- 3.63% | 0.125 +/- 0.041 m | 0.202 +/- 0.011 m | 1.25% +/- 0.63% | 33.0 +/- 3.1 |
| `ldrc_fixed` | 59.3% +/- 20.6% | 5.91% +/- 1.27% | 0.160 +/- 0.067 m | 0.169 +/- 0.012 m | 3.58% +/- 0.98% | 34.4 +/- 0.4 |

## 5. 三方法独立-seed 正式基线结果

### 5.1 设置与完整性

固定验证集选择出的既有 actor，仅重新生成测试 episode；执行器保持对应 checkpoint 的历史一阶 EMA 设置（`fixed_beta=0.35`），并关闭后来新增的 policy rate limiter，使本轮相对旧表只改变 episode seed 派生协议。实验包含 3 个 train seeds、5 个场景、3 种方法、3 个 eval seeds、每组 100 episodes：

```text
3 train seeds x 5 scenarios x 3 methods x 3 eval seeds x 100 episodes = 13,500 episodes
```

完整性检查：135/135 个文件、13,500/13,500 行、每个场景-方法-train seed 内 0 个重复 episode seed，所有 CSV 均记录 `episode_seed`。

原始汇总文件：

```text
outputs/archive_pre_restart_2026-09-09/current_results/baseline_3methods_independent_seeds/summary/manifest.json
outputs/archive_pre_restart_2026-09-09/current_results/baseline_3methods_independent_seeds/summary/all_eval_episodes.csv
outputs/archive_pre_restart_2026-09-09/current_results/baseline_3methods_independent_seeds/summary/eval_summary_by_train_seed.csv
outputs/archive_pre_restart_2026-09-09/current_results/baseline_3methods_independent_seeds/summary/eval_summary_across_train_seeds.csv
outputs/archive_pre_restart_2026-09-09/current_results/baseline_3methods_independent_seeds/summary/eval_summary_macro_across_train_seeds.csv
```

### 5.2 五场景宏平均

| 方法 | success rate | collision rate | non-end-link collision | final position error | min distance | safety violation rate | RMS jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ee_fixed` | 12.60% +/- 2.33% | 10.24% +/- 0.53% | 9.71% +/- 0.53% | 0.4158 +/- 0.0578 m | 0.1855 +/- 0.0270 m | 2.028% +/- 0.437% | 24.36 +/- 1.57 |
| `link_fixed_penalty1` | 61.33% +/- 13.90% | 6.80% +/- 3.98% | 6.64% +/- 3.87% | 0.1353 +/- 0.0299 m | 0.1962 +/- 0.0099 m | 1.464% +/- 0.750% | 32.73 +/- 2.56 |
| `ldrc_fixed` | 47.16% +/- 5.29% | 10.07% +/- 2.20% | 9.47% +/- 1.80% | 0.1873 +/- 0.0312 m | 0.1486 +/- 0.0057 m | 5.062% +/- 0.595% | 36.33 +/- 1.28 |

`link_fixed_penalty1` 在宏平均上取得最高成功率、最低碰撞率、最低非末端碰撞率、最低最终位置误差、最高最小距离和最低安全违反率。相对 `ldrc_fixed`，其碰撞率低 3.27 个百分点，三个 train seed 的差值方向一致；安全违反率低 3.60 个百分点，最小距离高 47.6 mm。相对 `ee_fixed`，成功率高 48.73 个百分点，但 RMS jerk 高 8.37。

train-seed 波动仍然明显：`link_fixed_penalty1` 的宏平均成功率分别为 63.53%、46.47%、74.00%，碰撞率分别为 7.13%、10.60%、2.67%。因此这些结果支持其作为当前最强基线，但 `n=3` 不足以宣称统计显著性或广泛泛化。

### 5.3 分场景结论

`link_fixed_penalty1` 在五个场景均获得最高成功率，在 random、elbow、forearm 和 wrist 四个场景获得最低碰撞率。upper-arm 场景中 `ee_fixed` 碰撞率略低（7.67% 对 8.78%），但其成功率仅 9.11%，明显低于 `link_fixed_penalty1` 的 57.56%，不能将该低碰撞率解释为更好的综合避障能力。`link_fixed_penalty1` 在五个场景的安全违反率均低于 `ldrc_fixed`。

## 6. Minimal QP 独立-seed 五场景反事实结果

### 6.1 设置与完整性

固定已有 `Instant-Link-SAC` actor，不进行重训；在相同 episode seed 下成对比较 `baseline_rtb` 与 `minimal_qp`。实验包含 3 个 train seeds、5 个场景、2 个执行变体、3 个 eval seeds、每组 100 episodes，共 9,000 次 episode 运行；每个变体包含 4,500 个无重复实验单元，两个变体共享 episode seed 以形成配对：

```text
3 train seeds x 5 scenarios x 2 variants x 3 eval seeds x 100 episodes
```

完整性检查：90/90 个文件、9,000/9,000 行、每个场景-变体-train seed 内无重复 episode seed；两个变体间相同 seed 的重复是预期的成对设计。

原始汇总文件：

```text
outputs/archive_pre_restart_2026-09-09/current_results/minimal_qp_heldout_counterfactual_independent_seeds/summary/manifest.json
outputs/archive_pre_restart_2026-09-09/current_results/minimal_qp_heldout_counterfactual_independent_seeds/summary/all_eval_episodes.csv
outputs/archive_pre_restart_2026-09-09/current_results/minimal_qp_heldout_counterfactual_independent_seeds/summary/eval_summary_by_train_seed.csv
outputs/archive_pre_restart_2026-09-09/current_results/minimal_qp_heldout_counterfactual_independent_seeds/summary/eval_summary_across_train_seeds.csv
outputs/archive_pre_restart_2026-09-09/current_results/minimal_qp_heldout_counterfactual_independent_seeds/summary/eval_summary_macro_across_train_seeds.csv
```

### 6.2 五场景宏平均

| 变体 | success rate | collision rate | final position error | min distance | safety violation rate | 子步命令 RMS jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_rtb` | 61.67% +/- 14.90% | 6.91% +/- 4.00% | 0.1345 +/- 0.0343 m | 0.1928 +/- 0.0101 m | 1.519% +/- 0.764% | 109.21 +/- 0.93 |
| `minimal_qp`（历史名称，实为投影器） | 63.71% +/- 13.41% | 1.49% +/- 0.19% | 0.1349 +/- 0.0318 m | 0.2034 +/- 0.0057 m | 0.238% +/- 0.019% | 108.76 +/- 0.88 |

按现有控制周期采样口径，反应式投影器将宏平均碰撞率相对降低 78.5%，安全违反率相对降低 84.4%，平均最小距离增加 10.6 mm；成功率提高 2.04 个百分点，子步命令 RMS jerk 未增加。全部三个 train seed 的碰撞率均下降：`7.13% -> 1.27%`、`10.80% -> 1.60%`、`2.80% -> 1.60%`。

### 6.3 分场景结果

| 场景 | baseline success | QP success | baseline collision | QP collision |
| --- | ---: | ---: | ---: | ---: |
| random crossing | 56.22% | 57.89% | 9.00% | 2.67% |
| upper arm | 58.33% | 60.22% | 8.44% | 0.89% |
| elbow | 60.56% | 63.56% | 8.11% | 1.67% |
| forearm | 65.33% | 68.33% | 6.44% | 1.56% |
| wrist | 67.89% | 68.56% | 2.56% | 0.67% |

五个场景的碰撞率均下降且成功率均未下降。成对 episode 中有 246 个 baseline 碰撞被 QP 避免，只有 2 个 episode 出现 baseline 不碰撞而 QP 碰撞。总体完成时间由 6.84 s 增至 7.16 s，主要来自避免碰撞后的提前终止；只比较两种变体均成功的 2,646 个配对 episode，平均完成时间仅增加约 0.026 s。

### 6.4 介入、实时性与失败尾部

- 平均介入率为 3.15% 控制周期；61.4% 的 episodes 完全没有介入，说明安全收益不是通过长期停止获得。
- 当前投影器平均计算时间为 0.012 ms，观测最大单次计算时间为 3.45 ms，低于 50 ms 控制周期；该数值不能外推为最终 OSQP 的求解时间。
- CSV 中的残差失败率为 0.248% 控制周期；7.4% 的 episodes 至少出现一次有限迭代后残差，20 个 episodes 的残差失败率超过 10%。这不等于已经证明约束集合不可行。
- 反应式投影后仍有 67 个控制周期采样碰撞 episodes，其中 16 个伴随投影残差失败，9 个没有触发安全层介入。

因此 P4.2 的工程门控已满足，但该结果只验证“当前连杆风险 actor + 反应式速度投影”的固定 actor 即时作用，不能替代逐子步监测、标准凸 QP、动作条件预测或完整 `Proposed` 方法实验，也不构成无碰撞保证。

## 7. endpoint/quintic 运动边界 validation（过渡消融）

本轮仅使用 eval seed 2001 选择约束参数：三个固定 actor、五个场景、每组 20 episodes。完整性检查为 75/75 个文件、1,500/1,500 行、0 个重复 episode 单元；该表是调参证据，不是最终 held-out 结论。

| 变体 | success | collision | safety violation | 投影残差失败 | 子步命令 RMS acceleration | 子步命令 RMS jerk | 子步命令最大 acceleration | 子步命令最大 jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minimal QP endpoint | 61.33% | 2.33% | 0.225% | 0.208% | 1.593 | 108.15 | 23.561 | 1413.449 |
| acceleration 8 | 61.00% | 2.67% | 0.177% | 1.184% | 1.589 | 107.94 | 8.000 | 479.931 |
| acceleration 8 + jerk 400 | 61.33% | 2.33% | 0.187% | 1.202% | 1.588 | 107.81 | 6.668 | 400.005 |
| acceleration 12 | 62.33% | 2.33% | 0.209% | 0.201% | 1.591 | 108.07 | 12.000 | 719.893 |
| acceleration 12 + jerk 500 | 62.67% | 2.33% | 0.216% | 1.196% | 1.591 | 108.07 | 8.335 | 500.005 |

相对 minimal QP endpoint，`12/500` 的 300 个配对 episode 中成功改善 6 个、恶化 2 个，碰撞引入/消除均为 0。jerk 命令边界将观测最大值降低 64.6%，但没有明显改变 RMS jerk。其投影残差失败率升高集中于 random crossing（5.150%），其他四场景均不超过 0.394%；该现象与运动边界压缩修正空间一致，但现有启发式投影器不足以证明 QP 可行域为空。仅 acceleration 12 的残差失败率为 0.201%，说明 jerk 边界与该尾部高度相关。由此冻结 `a_max=12 rad/s^2`、`j_max=500 rad/s^3` 进入独立 held-out，同时保留 acceleration-only 消融。

原始汇总文件：

```text
outputs/archive_pre_restart_2026-09-09/current_results/p5_motion_bounds_validation/summary/manifest.json
outputs/archive_pre_restart_2026-09-09/current_results/p5_motion_bounds_validation/summary/all_eval_episodes.csv
outputs/archive_pre_restart_2026-09-09/current_results/p5_motion_bounds_validation/summary/eval_summary_macro_across_train_seeds.csv
```

## 8. endpoint/quintic 运动边界 held-out（过渡消融）

参数仅由第 7 节的 seed 2001 validation 选择。本轮使用 eval seeds 1004/1005/1006、三个固定 actor、五个场景、每组 100 episodes；完整性检查为 135/135 个文件、13,500/13,500 行、0 个重复 episode 单元。

| 变体 | success | collision | final error | min distance | safety violation | 投影残差失败 | 子步命令 RMS jerk | 平均 episode 命令 peak jerk | 全局命令最大 acceleration | 全局命令最大 jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minimal QP endpoint | 63.49% | 1.56% | 0.1351 m | 0.2034 m | 0.215% | 0.225% | 109.06 | 292.41 | 51.536 | 3091.683 |
| acceleration 12 | 63.33% | 1.64% | 0.1361 m | 0.2033 m | 0.211% | 0.260% | 108.94 | 278.49 | 12.000 | 719.893 |
| acceleration 12 + jerk 500 | 63.18% | 1.67% | 0.1370 m | 0.2033 m | 0.203% | 0.274% | 108.83 | 265.25 | 8.335 | 500.005 |

`12/500` 将子步命令全局最大 jerk 降低 83.8%，平均 episode 命令 peak jerk 降低 9.3%，子步命令 RMS jerk 基本不变。代价是成功率下降 0.31 个百分点、碰撞率上升 0.11 个百分点、投影残差失败率上升 0.049 个百分点；安全违反率下降 0.012 个百分点。4,500 个配对 episode 中，成功改善/恶化为 40/54，碰撞引入/消除为 9/4，因此不能声称 jerk 约束改善碰撞，但绝对变化较小。

三个 train seed 的结论方向基本一致：成功率均未改善，碰撞率变化较小。观测最大投影计算时间为 3.85 ms，仍低于 50 ms 策略周期。该工程消融支持“endpoint/quintic 管线能够在数值容差内限制发出的子步速度命令峰值，并带来轻微任务/安全权衡”；它没有使用关节反馈计算实际加速度/jerk，安全约束也尚未施加到全部子步，不能据此声称轨迹嵌入式安全 QP 或最终 Proposed 已通过。

原始汇总文件：

```text
outputs/archive_pre_restart_2026-09-09/current_results/p5_motion_bounds_heldout/summary/manifest.json
outputs/archive_pre_restart_2026-09-09/current_results/p5_motion_bounds_heldout/summary/all_eval_episodes.csv
outputs/archive_pre_restart_2026-09-09/current_results/p5_motion_bounds_heldout/summary/eval_summary_macro_across_train_seeds.csv
```

## 9. P2.2 Predictive-Link-SAC checkpoint validation

本轮仅使用 validation seed 2001，在 random crossing 上比较 seed 101 的四种 predictive 配置及其 10k/20k/30k checkpoints，并以 Instant-Link 30k 为基准。完整性检查为 13/13 个文件、1,300/1,300 行、0 个重复 episode 单元。

| 候选 | success | collision | final error | min distance | safety violation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Instant-Link 30k | 16% | 17% | 0.3363 m | 0.1585 m | 4.21% |
| Predictive h=0.5 30k | 20% | 30% | 0.2620 m | 0.1125 m | 6.63% |
| Predictive compact 30k | 19% | 31% | 0.3717 m | 0.1284 m | 7.54% |
| Predictive h=1.0 30k | 14% | 23% | 0.4027 m | 0.1288 m | 6.76% |
| Predictive h=0.5, penalty=2, 20k | 7% | 17% | 0.5258 m | 0.1107 m | 5.92% |

`h=0.5/30k` 是唯一成功率高于 Instant-Link 30k 的主要候选，但碰撞率高 13 个百分点、最小距离低 45.9 mm。其配对 success 改善/恶化为 17/13，collision 引入/消除为 21/8。能保持碰撞率的 `h=1.0/10k` 与 `penalty=2/20k` 成功率分别只有 5% 和 7%。因此现有 30k checkpoints 没有候选同时保持任务与安全表现，不能据此启动 Predictive-Link-SAC 三 seed 正式训练。

由于 Instant-Link 自身在 30k 仍只有 16% 成功率，本结果也不足以证明预测状态在充分训练后必然失败。选择任务表现最好的 `h=0.5` full observation 做一次从头 100k、seed 101 门控；若 validation 仍显示明显安全退化，则回到预测状态或训练信号设计。

原始汇总文件：

```text
outputs/archive_pre_restart_2026-09-09/current_results/p2_predictive_checkpoint_validation/summary/manifest.json
outputs/archive_pre_restart_2026-09-09/current_results/p2_predictive_checkpoint_validation/summary/all_eval_episodes.csv
outputs/archive_pre_restart_2026-09-09/current_results/p2_predictive_checkpoint_validation/summary/eval_summary_by_train_seed.csv
```

## 10. P2.2 Predictive-Link-SAC 100k 门控

`h=0.5` full predictive observation 使用 seed 101 从头训练 100k steps。训练无 NaN/Inf，最后 10k 训练 bin 的 success/collision 为 55.4%/19.3%；但训练内结果只用于诊断。独立 validation 使用 seed 2001、random crossing、每个 checkpoint 100 episodes，完整性检查为 8/8 文件、800/800 行、0 个重复 episode 单元。

| 候选 | success | collision | final error | min distance | safety violation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Instant-Link 100k | 62% | 7% | 0.1413 m | 0.1847 m | 2.04% |
| Predictive h=0.5 90k | 47% | 23% | 0.1951 m | 0.1408 m | 4.26% |
| Predictive h=0.5 100k | 44% | 17% | 0.1918 m | 0.1523 m | 3.81% |

100k 是安全性最好的成熟 predictive checkpoint，但相对 Instant-Link 成功率低 18 个百分点、碰撞率高 10 个百分点、最小距离低 32.4 mm。配对 success 改善/恶化为 11/29，collision 引入/消除为 11/1；40k--90k checkpoints 也没有同时达到任务和安全门槛。因此 observation-only Predictive-Link-SAC 的 100k 门控失败，不进入 Predictive-Link-SAC 三 seed 正式训练。

代码审计显示预测量此前只进入 observation，固定风险惩罚仍完全由 current risk cost 构成。下一轮将预测 body risk 作为默认关闭的独立 dense reward 项做小权重消融；这是新的训练信号实验，不能与本节 observation-only 结果混为同一方法。

原始汇总文件：

```text
outputs/archive_pre_restart_2026-09-09/current_results/p2_predictive_100k_checkpoint_validation/summary/manifest.json
outputs/archive_pre_restart_2026-09-09/current_results/p2_predictive_100k_checkpoint_validation/summary/all_eval_episodes.csv
outputs/archive_pre_restart_2026-09-09/current_results/p2_predictive_100k_checkpoint_validation/summary/eval_summary_by_train_seed.csv
```

## 11. P2.2 raw predictive reward shaping validation

两个 shaping 权重各训练 seed 101、30k steps，并在 seed 2001 的相同 100 episodes 上比较。完整性检查为 8/8 文件、800/800 行、0 个重复 episode 单元。

| 候选 | success | collision | final error | min distance | safety violation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Instant-Link 30k | 16% | 17% | 0.3363 m | 0.1585 m | 4.21% |
| Predictive unshaped 30k | 20% | 30% | 0.2620 m | 0.1125 m | 6.63% |
| Predictive raw 0.1/20k | 10% | 16% | 0.5386 m | 0.1491 m | 3.80% |
| Predictive raw 0.1/30k | 13% | 16% | 0.3797 m | 0.1537 m | 4.12% |
| Predictive raw 0.25/30k | 9% | 23% | 0.3798 m | 0.1406 m | 4.43% |

raw 0.1/30k 相对 unshaped 将碰撞降低 14 个百分点，但成功率降低 7 个百分点；相对 Instant-Link 则成功率低 3 个百分点、碰撞率低 1 个百分点、最终误差高 43.4 mm。配对上，raw 0.1/30k 相对 Instant 的 success 改善/恶化为 8/11、collision 引入/消除为 10/11，未形成预测策略净收益。raw 0.25 更差。

该实验说明预测训练信号方向能影响安全行为，但直接惩罚完整 `risk_pred_body` 会与已有 current-risk cost 重叠并损害任务。结合第 9--10 节，Predictive-Link-SAC 未取得任务与安全共同收益，停止继续扩大训练。已实现但未运行的 excess shaping 不进入事实结论；后续预测信息改为作用于固定 actor 的动作条件安全约束。

```text
outputs/archive_pre_restart_2026-09-09/current_results/p2_predictive_reward_shaping_validation/summary/manifest.json
outputs/archive_pre_restart_2026-09-09/current_results/p2_predictive_reward_shaping_validation/summary/eval_summary_by_train_seed.csv
```

## 12. 结论边界

当前可以支持：

- 修复后的三方法正式基线包含 13,500 个无重复 episode 单元；`link_fixed_penalty1` 在本轮宏平均核心任务与安全指标上整体优于 `ee_fixed` 和 `ldrc_fixed`，可作为后续扩展起点。
- 对固定 `Instant-Link-SAC` actor，反应式投影器在三个 train seed 和五个场景中一致降低控制周期采样碰撞率与安全违反率，同时未降低宏平均成功率或增加子步命令 RMS jerk。
- 当前反应式投影器的平均介入和计算开销较低，但存在有限迭代残差尾部和残余碰撞；这不是标准 QP 的可行性或耗时结论。
- endpoint/quintic held-out 支持 `12/500` 在数值容差内限制发出的子步命令 acceleration/jerk 峰值，任务成功和安全指标仅小幅变化；它不证明实际关节反馈满足同一边界，不支持“jerk 约束降低碰撞”，也不等同于最终轨迹嵌入式 QP。
- Predictive-Link-SAC 的 observation-only、100k 和 raw reward-shaping 门控均未取得任务与安全共同收益；该负结果支持将预测模块与名义策略解耦。

不能支持：

- 约束 SAC 已整体优于固定惩罚 SAC；
- 旧 seed 重叠表代表 13,500 个独立 episodes，或旧表与修复后的正式表可以混用；
- 自适应平滑或风险自适应 jerk QP 已被多 seed、多场景验证；
- 动作条件预测、Top-k 多子步安全约束、轨迹嵌入式标准凸 QP，或验证驱动的全身非线性验收/反例约束增广已经完成；
- 三个 train seed 已证明统计显著性、广泛泛化或真实机器人安全保证。

原四方法矩阵、`ldrc_adaptive` 反事实、自适应平滑诊断和旧 `outputs/archive_pre_restart_2026-09-09/archive_old/formal_legacy/` 结果仅作历史/失败消融，不参与最终排序。第 2--4 节旧三方法结果存在 episode seed 重叠，不得与第 5 节正式基线混用；第 5 节使用历史 EMA 执行器，第 6 节使用 RTB 执行器，两者也不得直接比较 jerk 数值。
