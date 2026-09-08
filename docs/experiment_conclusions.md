# 仿真实验数据与结论整理（2026-07-27 更新）

> 执行器版本说明（2026-09-07）：本页全部正式结果由旧固定一阶 EMA 执行器（`fixed_beta=0.35`）产生。代码现已加入可切换的 Butterworth—五次多项式 RTB；在完成新执行器重训和 held-out 评估前，本页数值不得用于声称 RTB 有效，也不得与新执行器结果混合汇总。

本文档是当前正式仿真实验的唯一结论口径。论文、汇报和后续分析只能引用本文件列出的 eval-seed held-out 三方法汇总表；原四方法结果仅可作为历史记录、参数敏感性分析或自适应平滑失败消融，不得用于最终主方法排序。

## 1. 可信数据源与统计口径

最终主比较覆盖 3 个训练随机种子、5 个场景、3 种方法、3 个独立评估随机种子，每个评估 seed 运行 100 episodes：

```text
3 train seeds x 5 scenarios x 3 methods x 3 eval seeds x 100 episodes
= 13,500 episodes
```

最终主表的可信数据源：

```text
outputs/rechecks/heldout_1004_1006/final_3methods/all_eval_episodes.csv
outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_by_train_seed.csv
outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_across_train_seeds.csv
outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_macro_across_train_seeds.csv
```

每个“场景-方法-train seed”先合并 3 个 eval seed 的 300 episodes 求均值；表中的 `mean +/- std` 是 3 个 train seed 均值之间的均值和样本标准差（`n=3`），不是 episode 级标准差。checkpoint 由独立的 `validation_seed=2001`、每 checkpoint 20 episodes 的验证结果选择；正式评估使用 held-out eval seeds 1004、1005、1006。

以下结果仅保留作历史或诊断记录，不能用于正文主表和最终排序：

```text
outputs/formal/summary/
outputs/formal/diagnostics/counterfactual/
outputs/formal/paper_notes/
```

## 2. 最终实验设置

正式实验使用 PyBullet UR5 仿真环境，评估单个动态球形障碍物穿越机械臂工作空间时的整臂避障与静态目标到达能力。

| 项目 | 设置 |
| --- | --- |
| train seeds | 101、202、303 |
| validation seed | 2001 |
| held-out eval seeds | 1004、1005、1006 |
| 每个 train seed / 场景 / 方法 | 300 episodes |
| 每个场景 / 方法 | 900 episodes |
| 主场景 | `random_crossing` |
| 定向压力场景 | `upper_arm_crossing`、`elbow_crossing`、`forearm_crossing`、`wrist_crossing` |

最终主比较方法：

| 方法 | 含义 |
| --- | --- |
| `ee_fixed` | 末端风险 + 固定风险惩罚 SAC + 固定平滑 |
| `link_fixed_penalty1` | 连杆级动态风险 + 固定风险惩罚 SAC + 固定平滑，`w_R=1.0` |
| `ldrc_fixed` | 连杆级动态风险 + 约束 SAC + 固定平滑 |

`w_R=1.0` 由 train seeds 101、202 上的小规模筛选提出，随后在相同三个 train seeds 上按相同 100k 预算重训。最终比较使用未参与筛选的 eval seeds 1004--1006，因此评估随机性独立于筛选阶段，但完整的超参数选择和训练随机性并非完全独立。

## 3. 正文 Table 1：Random Crossing 主对比

数据源：`outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_across_train_seeds.csv`。每种方法共 900 个正式评估 episodes。

| 方法 | success rate | collision rate | non-end-link collision | final position error | min distance | safety violation rate | RMS jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ee_fixed` | 17.0% +/- 2.0% | 13.44% +/- 5.19% | 13.11% +/- 5.17% | 0.369 +/- 0.017 m | 0.177 +/- 0.012 m | 2.38% +/- 1.44% | 25.0 +/- 2.0 |
| `link_fixed_penalty1` | 64.2% +/- 12.5% | 6.89% +/- 2.22% | 6.89% +/- 2.22% | 0.125 +/- 0.029 m | 0.193 +/- 0.020 m | 1.27% +/- 0.92% | 32.3 +/- 2.7 |
| `ldrc_fixed` | 56.3% +/- 18.0% | 5.56% +/- 2.91% | 5.56% +/- 2.91% | 0.172 +/- 0.054 m | 0.167 +/- 0.008 m | 3.12% +/- 1.27% | 32.7 +/- 0.2 |

主场景结论：

1. `link_fixed_penalty1` 的成功率最高，最终位置误差、安全距离违反率和最小距离均优于 `ldrc_fixed`。
2. `ldrc_fixed` 的碰撞率略低于 `link_fixed_penalty1`（5.56% vs 6.89%），因此两者存在碰撞控制上的局部取舍。
3. `ee_fixed` 在任务完成和最终位置误差上明显落后，支持连杆级风险输入与经过标定的固定风险惩罚组合的有效性。

## 4. 正文 Table 2：非末端区域压力测试

数据源同 Table 1。每个单元格为 3 个 train seed 的 `mean +/- std`；每种方法在每个区域共 900 episodes。

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

压力测试结论：

1. `link_fixed_penalty1` 在五个场景均获得最高的三-seed 平均成功率，并在 forearm、wrist 场景同时取得较低碰撞率。
2. 在 upper arm 和 elbow 场景，`ldrc_fixed` 的碰撞率低于 `link_fixed_penalty1`；固定惩罚方法不能被表述为在所有安全指标上严格占优。
3. 结果不支持“约束 SAC 相较经过调参的固定惩罚 SAC 带来整体增益”的主张。约束方法的优势应限定为部分高风险区域中略低的碰撞率。

## 5. 宏平均与自适应平滑诊断

跨五个场景的宏平均如下：

| 方法 | success rate | collision rate | final position error | min distance | safety violation rate | RMS jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ee_fixed` | 16.4% +/- 1.4% | 7.69% +/- 2.68% | 0.396 +/- 0.014 m | 0.189 +/- 0.013 m | 1.69% +/- 0.61% | 24.7 +/- 1.5 |
| `link_fixed_penalty1` | 66.7% +/- 15.3% | 6.51% +/- 3.63% | 0.125 +/- 0.041 m | 0.202 +/- 0.011 m | 1.25% +/- 0.63% | 33.0 +/- 3.1 |
| `ldrc_fixed` | 59.3% +/- 20.6% | 5.91% +/- 1.27% | 0.160 +/- 0.067 m | 0.169 +/- 0.012 m | 3.58% +/- 0.98% | 34.4 +/- 0.4 |

原四方法矩阵中的 adaptive actor / execution 反事实诊断仍可作为附录失败案例：seed 303 的退化主要来自 adaptive actor；自适应执行器在 seed 202、303 上均显著提高 jerk，且没有稳定的任务或碰撞收益。因此 `ldrc_adaptive` 不纳入最终主比较、核心贡献或真实部署候选。

## 6. 最终论文结论口径

可以作为核心结论：

1. 在本文的单动态球形障碍物 UR5 仿真设定下，连杆级动态风险配合经过标定的固定风险惩罚 SAC（`link_fixed_penalty1`, `w_R=1.0`）相较末端风险 baseline 稳定提高了三-seed 平均目标到达能力，并明显降低最终位置误差。
2. `link_fixed_penalty1` 在 held-out 五场景比较中获得最高平均成功率，并在最终误差、最小距离、安全距离违反率、jerk 和平均风险代价上优于 `ldrc_fixed`；`ldrc_fixed` 的宏平均碰撞率略低。因此结果支持固定惩罚方法在任务与多数安全/平滑指标上的综合优势，但不支持其在全部安全指标上严格占优。
3. 当前证据不支持风险约束 SAC 相对于经过调参的固定惩罚 baseline 的整体增益，也不支持风险自适应平滑模块提升平滑性、安全性或任务能力。

必须保留的限定：

1. 结果基于 3 个 train seed，只报告均值和标准差（`n=3`），不作严格显著性或强泛化声明。
2. `w_R=1.0` 的筛选复用了 train seeds 101、202；最终比较仅对 eval seeds 独立。后续应使用新的 train seeds 和独立 held-out eval seeds 复核。
3. 环境仅含单个动态球形障碍物，结论不外推至多障碍、复杂形状、高速障碍或真实机器人安全保证。
4. 若开展真实低速验证，应优先部署 `link_fixed_penalty1`，仅报告可执行性、轨迹和风险响应，不进行高风险碰撞性基线对比。

不应使用的表述：

```text
ldrc_fixed 综合性能最优。
风险约束 SAC 相比经过调参的固定惩罚 SAC 已证明整体更优。
自适应平滑模块提升了平滑性或安全性。
link_fixed_penalty1 在全部安全指标上优于 baseline。
三-seed 结果已经证明统计显著性或广泛泛化能力。
```

## 7. 写作建议

正文保留 Table 1 和 Table 2；方法名称统一使用 `link_fixed_penalty1` 或“连杆级固定风险惩罚 SAC（`w_R=1.0`）”。附录可保留各 train seed 完整结果、固定惩罚权重筛选，以及 adaptive actor / execution 反事实诊断，并明确它们不参与最终主方法排序。

一句话总括：

> 当前最稳妥的论文主线是：在单动态球形障碍物 UR5 仿真中，连杆级动态风险配合经标定的固定风险惩罚 SAC（`w_R=1.0`）在 held-out 评估中取得最高目标到达表现，并在多数安全与平滑指标上优于约束 SAC；约束 SAC 仅在部分场景保留略低碰撞率，自适应平滑尚未获得稳定有效性证据。
