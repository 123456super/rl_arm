# 仿真实验数据与结论整理（2026-07-27 更新）

本文档是当前正式仿真实验的唯一结论口径，替代此前仅覆盖 train seed 101 的结论。论文、汇报和后续分析只能引用本文件列出的三-seed 统一汇总表，不再引用旧的单-seed 表格。

## 1. 可信数据源与口径

正式评估覆盖 3 个训练随机种子、5 个场景、4 种方法、3 个评估随机种子，每个评估 seed 运行 100 episodes：

```text
3 train seeds x 5 scenarios x 4 methods x 3 eval seeds x 100 episodes
= 18,000 episodes
```

当前可信数据源：

```text
outputs/formal/summary/all_eval_episodes.csv
outputs/formal/summary/eval_summary_by_train_seed.csv
outputs/formal/summary/eval_summary_across_train_seeds.csv
outputs/formal/summary/eval_summary_macro_across_train_seeds.csv
outputs/formal/diagnostics/counterfactual/
```

统计口径：每个“场景-方法-train seed”由 3 个 eval seed 共 300 episodes 取均值；表中的 `mean +/- std` 是 3 个 train seed 均值之间的均值和样本标准差（`n=3`），不是 900 个 episode 的标准差。

以下文件仅保留作历史记录，不能用于正式结论：

```text
outputs/formal/summary/table_by_region_stress.csv
outputs/formal/eval/by_region/summary_by_region_seed101_mean.csv
outputs/formal/eval/by_region/summary_by_region_seed101.csv
outputs/formal/paper_notes/table_random_crossing_main_corrected.csv
```

## 2. 实验设置

正式实验使用 PyBullet UR5 仿真环境，评估单个动态球形障碍物穿越机械臂工作空间时的整臂避障与静态目标到达能力。

| 项目 | 设置 |
| --- | --- |
| train seeds | 101、202、303 |
| eval seeds | 1001、1002、1003 |
| 每个 train seed / 场景 / 方法 | 300 episodes |
| 每个场景 / 方法 | 900 episodes |
| 主场景 | `random_crossing` |
| 定向压力场景 | `upper_arm_crossing`、`elbow_crossing`、`forearm_crossing`、`wrist_crossing` |

四种方法：

| 方法 | 含义 |
| --- | --- |
| `ee_fixed` | 仅末端风险固定惩罚 baseline |
| `link_fixed` | 连杆级动态风险固定惩罚 baseline |
| `ldrc_fixed` | 连杆级动态风险约束 SAC + 固定平滑 |
| `ldrc_adaptive` | `ldrc_fixed` + 风险自适应平滑执行 |

每个训练 seed 的 checkpoint 由 `validation_seed=2001`、20 episodes 的验证结果独立选择；正式评估不使用该 validation seed。

| 方法 | seed 101 | seed 202 | seed 303 |
| --- | ---: | ---: | ---: |
| `ee_fixed` | 100k | 70k | 70k |
| `link_fixed` | 100k | 80k | 40k |
| `ldrc_fixed` | 50k | 100k | 70k |
| `ldrc_adaptive` | 40k | 90k | 60k |

## 3. 主场景结果：Random Crossing

数据源：`outputs/formal/summary/eval_summary_across_train_seeds.csv`。表中为 3 个 train seed 的均值和标准差，每种方法共 900 个正式评估 episodes。

| 方法 | success rate | collision rate | non-end-link collision | final position error | min distance | safety violation rate | RMS jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ee_fixed` | 17.0% +/- 1.2% | 13.3% +/- 5.0% | 13.0% +/- 5.0% | 0.371 +/- 0.015 m | 0.178 +/- 0.012 m | 2.36% +/- 1.46% | 25.0 +/- 2.0 |
| `link_fixed` | 14.6% +/- 1.1% | 5.7% +/- 4.0% | 5.3% +/- 3.5% | 0.424 +/- 0.070 m | 0.215 +/- 0.012 m | 0.82% +/- 0.23% | 24.0 +/- 3.8 |
| `ldrc_fixed` | 57.2% +/- 18.0% | 5.3% +/- 3.1% | 5.3% +/- 3.1% | 0.169 +/- 0.052 m | 0.167 +/- 0.007 m | 3.10% +/- 1.25% | 32.8 +/- 0.3 |
| `ldrc_adaptive` | 52.2% +/- 24.9% | 14.0% +/- 10.4% | 13.0% +/- 8.7% | 0.170 +/- 0.079 m | 0.141 +/- 0.028 m | 6.43% +/- 3.44% | 69.2 +/- 9.5 |

结论：

1. `ldrc_fixed` 的成功率大幅高于两个 baseline，最终误差也明显更低，说明连杆级动态风险约束 SAC 能够提升目标到达能力。
2. `ldrc_fixed` 的碰撞率与保守的 `link_fixed` 接近，且远低于 `ee_fixed`；但它的安全距离违反率高于两个 baseline。因此，结果支持“任务完成与碰撞控制的较好折中”，不支持“所有安全指标均更优”。
3. `link_fixed` 的低碰撞率、较大最小距离和低违反率伴随极低成功率与较大最终误差，属于保守策略，不能作为综合最优方法。
4. `ldrc_adaptive` 的最终误差与 `ldrc_fixed` 基本相同，但碰撞率、违反率、jerk 和跨 seed 波动均更高，不能作为主方法或平滑性改进证据。

## 4. 非末端区域压力测试

数据源：`outputs/formal/summary/eval_summary_across_train_seeds.csv`。每个单元格为 3 个 train seed 的均值 +/- 标准差；每种方法在每个区域共 900 episodes。

| 区域 | 方法 | success rate | collision rate | final position error | RMS jerk |
| --- | --- | ---: | ---: | ---: | ---: |
| upper arm | `ee_fixed` | 13.0% +/- 0.0% | 3.0% +/- 3.6% | 0.412 +/- 0.034 m | 25.8 +/- 1.3 |
| upper arm | `link_fixed` | 12.7% +/- 3.1% | 3.7% +/- 2.3% | 0.489 +/- 0.072 m | 26.0 +/- 4.9 |
| upper arm | `ldrc_fixed` | 55.1% +/- 19.5% | 9.3% +/- 0.6% | 0.178 +/- 0.083 m | 35.6 +/- 1.6 |
| upper arm | `ldrc_adaptive` | 52.3% +/- 27.1% | 13.3% +/- 11.0% | 0.158 +/- 0.076 m | 85.8 +/- 9.7 |
| elbow | `ee_fixed` | 16.9% +/- 1.5% | 6.1% +/- 3.7% | 0.381 +/- 0.010 m | 25.4 +/- 1.6 |
| elbow | `link_fixed` | 16.2% +/- 4.7% | 6.8% +/- 3.9% | 0.463 +/- 0.022 m | 25.2 +/- 3.7 |
| elbow | `ldrc_fixed` | 57.7% +/- 24.0% | 7.6% +/- 2.4% | 0.166 +/- 0.078 m | 35.4 +/- 1.6 |
| elbow | `ldrc_adaptive` | 50.4% +/- 28.0% | 13.3% +/- 14.0% | 0.159 +/- 0.078 m | 77.5 +/- 17.4 |
| forearm | `ee_fixed` | 19.3% +/- 5.1% | 5.4% +/- 3.2% | 0.389 +/- 0.023 m | 24.0 +/- 1.5 |
| forearm | `link_fixed` | 18.4% +/- 5.4% | 5.3% +/- 5.1% | 0.432 +/- 0.013 m | 25.0 +/- 2.0 |
| forearm | `ldrc_fixed` | 67.1% +/- 18.7% | 4.4% +/- 1.3% | 0.130 +/- 0.052 m | 34.9 +/- 0.7 |
| forearm | `ldrc_adaptive` | 54.8% +/- 28.0% | 12.1% +/- 14.0% | 0.148 +/- 0.082 m | 67.5 +/- 23.6 |
| wrist | `ee_fixed` | 15.8% +/- 2.9% | 10.7% +/- 3.8% | 0.428 +/- 0.036 m | 23.7 +/- 2.1 |
| wrist | `link_fixed` | 19.3% +/- 2.1% | 5.0% +/- 6.2% | 0.423 +/- 0.024 m | 23.5 +/- 1.5 |
| wrist | `ldrc_fixed` | 62.7% +/- 24.3% | 3.2% +/- 0.8% | 0.146 +/- 0.069 m | 33.7 +/- 1.3 |
| wrist | `ldrc_adaptive` | 61.0% +/- 23.8% | 9.7% +/- 11.5% | 0.132 +/- 0.066 m | 51.7 +/- 18.9 |

结论：

1. `ldrc_fixed` 在四个定向压力场景中均获得最高的三-seed 平均成功率，且最终误差显著低于两个 baseline。
2. `ldrc_fixed` 在 forearm 和 wrist 场景同时取得较高成功率和较低碰撞率；upper arm 与 elbow 场景的碰撞率高于保守 baseline，仍需将安全距离违反率和碰撞率作为局限报告。
3. `ldrc_adaptive` 在所有压力场景中的平均碰撞率和 jerk 都高于 `ldrc_fixed`，不支持其带来稳定的安全或平滑收益。
4. 这些结果支持连杆级动态风险约束对非末端区域任务到达能力的作用；不支持将固定平滑 LDRC 解释为在全部安全指标上严格占优。

## 5. 自适应平滑反事实诊断

为区分“自适应执行层问题”和“adaptive actor 训练失败”，在 `random_crossing` 上对 seed 202、303 进行了 actor 与执行器交叉组合评估。每个组合使用 3 个 eval seed、共 300 episodes。

| train seed | actor | execution | success rate | collision rate | RMS jerk |
| --- | --- | --- | ---: | ---: | ---: |
| 202 | fixed | fixed | 77.7% | 2.0% | 32.6 |
| 202 | fixed | adaptive | 71.3% | 5.0% | 93.5 |
| 202 | adaptive | adaptive | 79.3% | 9.0% | 75.2 |
| 202 | adaptive | fixed | 79.7% | 9.0% | 27.8 |
| 303 | fixed | fixed | 50.3% | 6.0% | 32.7 |
| 303 | fixed | adaptive | 57.0% | 7.0% | 73.2 |
| 303 | adaptive | adaptive | 30.3% | 26.0% | 74.3 |
| 303 | adaptive | fixed | 26.3% | 26.0% | 31.4 |

诊断结论：

1. seed 303 的失败主要来自 `ldrc_adaptive` actor 学到的策略：改用固定执行器后，成功率仍低、碰撞率不变。
2. 自适应执行器在两颗 seed 上都大幅提高 jerk，且没有带来一致的任务或碰撞收益。
3. 因此，自适应平滑模块当前应作为未验证通过的消融项或失败案例，不应作为论文核心贡献、完整方法优势或真实部署候选。

典型 `Risk_global`、`d_min`、`beta` 曲线只能说明实现会随风险调整 `beta`，不能证明该机制提升安全性、响应性或平滑性。

## 6. 最终论文结论口径

可以作为核心结论：

1. 在本文的单动态球形障碍物 UR5 仿真设定下，连杆级动态风险约束 SAC 加固定平滑（`ldrc_fixed`）相较末端风险和连杆级固定惩罚 baseline，稳定地提高了三-seed 平均目标到达能力，并明显降低了最终位置误差。
2. `ldrc_fixed` 在主场景中将碰撞率控制在与保守 `link_fixed` 接近的水平；其代价是安全距离违反率与 jerk 高于保守 baseline。因此应将其定位为任务-碰撞折中改进，而不是严格安全保证或全指标安全最优。
3. `ldrc_adaptive` 没有显示稳定增益：其跨 seed 的成功率、碰撞率、违反率和 jerk 波动均更大。当前证据不支持风险自适应平滑模块的有效性主张。

必须保留的限定：

1. 结果基于 3 个 train seed，能够报告均值和标准差，但样本量不足以作严格显著性检验或强泛化声明。
2. checkpoint selection 会影响结果；应明确“validation seed 选择 checkpoint，独立 eval seeds 做正式评估”的流程。
3. 环境仅含单个动态球形障碍物，结论不外推到多障碍、复杂形状、高速障碍或真实机器人安全保证。
4. 真实实验若继续开展，应优先部署 `ldrc_fixed`，并只作为低速可执行性验证。

不应使用的表述：

```text
自适应平滑模块提升了平滑性或安全性。
完整方法 ldrc_adaptive 综合性能最优。
ldrc_fixed 在全部安全指标上优于 baseline。
三-seed 结果已经证明统计显著性或广泛泛化能力。
```

## 7. 写作建议

正文建议保留以下两张正式表：

```text
Table 1: Random Crossing 三-seed 主对比（mean +/- std, n=3 train seeds）
Table 2: 四个非末端区域压力测试（三-seed mean +/- std, n=3 train seeds）
```

可在附录中保留：

```text
Table A1: 各 train seed 的完整结果
Table A2: adaptive actor / execution 反事实诊断
Figure A1: 自适应 beta 的典型响应曲线，仅作机制实现说明
```

一句话总括：

> 当前最稳妥的论文主线是：连杆级动态风险约束 SAC 加固定平滑能够提升单动态障碍物场景下的整臂目标到达能力，并在碰撞控制与任务完成之间取得较好折中；风险自适应平滑模块尚未获得稳定有效性证据。
