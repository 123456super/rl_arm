# 阶段一仿真实验数据与结论整理（正式结论截至 2026-07-27；状态更新 2026-07-31）

> **重构说明（2026-07-28）**：本文件保留原研究主题下的正式 held-out 仿真结论，作为新主题的阶段一基线、问题动机和可复现材料。新论文的研究设计以 [research_direction.md](research_direction.md) 为准；本文件的三方法排序不得直接写成新论文的最终主结论。

本文档是阶段一正式仿真的唯一结论口径。论文的阶段一基线、汇报和后续分析只能引用本文件列出的 eval-seed held-out 三方法汇总表；原四方法结果仅可作为历史记录、参数敏感性分析或自适应平滑失败消融。

## 0. 当前状态边界（2026-07-31）

- 阶段一正式结论：已冻结。本页的 held-out 主表、方法排序和限定条件保持不变；link_fixed_penalty1 仍只是仿真部署候选，不是真机安全认证。
- P3 研究分支：未冻结。2026-07-31 前完成的几何 frame 修正、mesh/三角形覆盖审计、strict safe-stop、TTC recovery 和固定 link Jacobian 修复均属于开发诊断，不能与阶段一数据合并。
- 当前阻塞：物理接触与胶囊模型仍有漏检/误报，动态漂移下 safe-stop 不能保持安全集，OSQP 存在超过 50 ms 的长尾，现有诊断样本不足以支持泛化。
- 后续口径：在固定时间预算、硬约束优先 deterministic escape controller 通过仿真回归前，不新增训练、不进入 P4/OOD、不做真机；其结果应追加到 experiment_progress.md，而不是改写本页正式主表。

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

## 6. 阶段一结论口径与新主题的承接

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

这些结果直接支撑新主题的研究动机：仅靠风险表示和训练目标（固定惩罚或约束 SAC）的替换，无法在碰撞控制、任务能力和执行行为上获得无条件优势。因此，新主题将把连杆级风险扩展为考虑感知不确定性与预测时域的安全裕度，并在策略输出后加入独立的安全过滤与失效安全机制。该推论是研究设计动机，不是对新方法有效性的提前结论。

## 7. 阶段一材料的写作使用方式

新论文可将 Table 1 和 Table 2 作为预研究或历史基线材料；方法名称统一使用 `link_fixed_penalty1` 或“连杆级固定风险惩罚 SAC（`w_R=1.0`）”。它们不能取代新方法的因子化消融、独立随机种子复核、OOD 和实机风险响应结果。附录可保留各 train seed 完整结果、固定惩罚权重筛选，以及 adaptive actor / execution 反事实诊断。

一句话总括：

> 阶段一最稳妥的结论是：在单动态球形障碍物 UR5 仿真中，连杆级动态风险配合经标定的固定风险惩罚 SAC（`w_R=1.0`）在 held-out 评估中取得最高目标到达表现；但训练目标替换无法保证全部安全指标的优势，自适应平滑也未获得稳定有效性证据。这构成引入预测风险、安全过滤和协同训练的起点。
