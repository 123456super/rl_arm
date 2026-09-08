# 最新论文大纲下的实验进展说明

> 日期：2026-09-08  
> 对应大纲：`docs/thesis_outline.md`  
> 绑定计划：`docs/thesis_experiment_plan.md`  
> 说明：本文档按最新论文主线重新整理已有实验结果的可用范围。此前部分结果来自旧论文设计，不能直接支撑尚未实现的预测风险和安全 QP 结论。

## 0. 计划绑定

本文档记录实验进展，`docs/thesis_experiment_plan.md` 记录分步执行清单。后续每完成一个计划编号，应同步更新本文档中的状态、输出目录、关键指标和是否进入下一步。

| 计划编号 | 当前状态 | 绑定进展项 |
| --- | --- | --- |
| P0 | 部分完成 | 已有结果固化与使用边界已整理，仍需后续保持命名一致。 |
| P1 | 未完成 | 预测性连杆风险离线验证。 |
| P2 | 未完成 | `Predictive-Link-SAC` 状态扩展与小规模训练。 |
| P3 | 未完成 | 预测风险正式对比。 |
| P4 | 未完成 | 最小安全 QP 指令整形器。 |
| P5 | 未完成 | 加速度与 jerk 约束 QP。 |
| P6 | 未完成 | `Proposed` 完整方法主实验。 |
| P7 | 未完成 | 鲁棒性与泛化实验。 |
| P8 | 未完成 | 真实 UR5 低速验证。 |

## 1. 最新论文主线

最新论文拟围绕以下主线展开：

```text
当前连杆风险基线
-> 预测性连杆风险建模
-> SAC 名义关节速度控制
-> 风险自适应 jerk 约束安全 QP 指令整形
-> 子步监测与低速实机验证
```

根据 `docs/thesis_outline.md`，论文核心创新点是：

1. 面向机械臂全身的预测性连杆风险表征方法。
2. 风险自适应的 jerk 约束安全 QP 指令整形方法。
3. 面向预测与干预机制的分层评价体系。

因此，已有实验结果需要重新定位：旧结果可以证明“当前连杆级风险”是有效起点，但不能证明“预测风险”或“QP 指令整形器”已经有效。

## 2. 当前总体进展

截至 2026-09-08，实验进展可概括为：

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| UR5 PyBullet 动态障碍物环境 | 已完成 | 对应 P0；已支持固定基座 UR5、静态目标到达、单动态球形障碍物和多场景评估。 |
| 当前连杆级动态风险 | 已完成 | 对应 P0；已实现连杆胶囊体距离、接近速度、TTC 和当前风险聚合。 |
| 末端风险 baseline | 已完成 | 对应 P0；已作为 `ee_fixed` 完成 held-out 评估。 |
| 当前连杆风险固定惩罚 SAC | 已完成 | 对应 P0；`link_fixed_penalty1` 是目前最可靠的已有基线。 |
| 约束 SAC / LDRC baseline | 已完成但降级 | 对应 P0；可作为次要对比或旧方法讨论，不再作为核心创新。 |
| 自适应 beta / 历史平滑方案 | 已完成但失败 | 对应 P0；结果显示 jerk 增大且收益不稳定，仅作为失败消融或方法演进材料。 |
| 预测性连杆风险 | 未完成 | 对应 P1；尚未看到对应实现、训练结果或 Warning Lead Time 统计。 |
| `Predictive-Link-SAC` | 未完成 | 对应 P2/P3；尚无基于预测风险状态的新训练和 held-out 评估。 |
| 安全 QP 指令整形器 | 未完成 | 对应 P4/P5；尚无 QP 介入率、修正幅度、不可行率和求解耗时结果。 |
| `Proposed` 完整方法 | 未完成 | 对应 P6；尚不能写成已验证贡献。 |
| `Instant-Link-SAC+QP` 交叉消融 | 未完成 | 对应 P4/P6；需要用于区分预测风险收益和 QP 安全层收益。 |
| 真实 UR5 低速验证 | 未完成 | 对应 P8；当前只有离线部署预检材料，不能替代实机验证。 |

## 3. 已有结果中可以支撑新论文的部分

### 3.1 当前连杆风险基线有效性

最有用、最可信的数据源是：

```text
outputs/rechecks/heldout_1004_1006/final_3methods/
```

该目录包含：

```text
outputs/rechecks/heldout_1004_1006/final_3methods/all_eval_episodes.csv
outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_by_train_seed.csv
outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_across_train_seeds.csv
outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_macro_across_train_seeds.csv
```

实验设置：

| 项目 | 设置 |
| --- | --- |
| train seeds | 101、202、303 |
| held-out eval seeds | 1004、1005、1006 |
| 场景 | `random_crossing`、`upper_arm_crossing`、`elbow_crossing`、`forearm_crossing`、`wrist_crossing` |
| 方法 | `ee_fixed`、`link_fixed_penalty1`、`ldrc_fixed` |
| 总 episode 数 | 13,500 |

这些结果可以用于新论文中证明：

1. 只看末端风险不足以覆盖整臂动态避障问题。
2. 当前连杆级动态风险相较末端风险能够明显提高目标到达能力。
3. `link_fixed_penalty1` 是后续预测风险和安全 QP 方法的合理起点。
4. 约束 SAC 未显示相对固定惩罚 SAC 的整体优势，因此不适合作为新论文核心创新。

### 3.2 Random Crossing 主场景结果

可引用整理表：

```text
outputs/paper/final_materials/table_1_random_crossing.csv
```

主要结果如下：

| 方法 | success rate | collision rate | final position error | min distance | safety violation rate | RMS jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ee_fixed` | 17.00% +/- 2.00% | 13.44% +/- 5.19% | 0.369 +/- 0.017 m | 0.177 +/- 0.012 m | 2.38% +/- 1.44% | 25.0 +/- 2.0 |
| `link_fixed_penalty1` | 64.22% +/- 12.54% | 6.89% +/- 2.22% | 0.125 +/- 0.029 m | 0.193 +/- 0.020 m | 1.27% +/- 0.92% | 32.3 +/- 2.7 |
| `ldrc_fixed` | 56.33% +/- 18.02% | 5.56% +/- 2.91% | 0.172 +/- 0.054 m | 0.167 +/- 0.008 m | 3.12% +/- 1.27% | 32.7 +/- 0.2 |

可支撑的表述：

- `link_fixed_penalty1` 相比 `ee_fixed` 明显提高成功率，并降低碰撞率和最终位置误差。
- `ldrc_fixed` 的碰撞率略低于 `link_fixed_penalty1`，但成功率、最终误差、最小距离和安全距离违反率不占优。
- 当前结果支持连杆级风险作为新论文方法起点，但不支持约束 SAC 作为核心主方法。

不应使用的表述：

- `ldrc_fixed` 综合最优。
- 固定惩罚 SAC 在所有安全指标上严格优于约束 SAC。
- 这些结果证明预测风险或安全 QP 有效。

### 3.3 非末端连杆压力测试结果

可引用整理表：

```text
outputs/paper/final_materials/table_2_non_end_link_stress.csv
```

该结果可用于证明新论文问题设定的必要性：非末端连杆，包括 upper arm、elbow、forearm、wrist，确实会成为避障中的主要风险区域，不能只用末端风险表示。

可支撑的表述：

- 当前连杆风险方法在多个非末端压力场景下提升任务完成率。
- `link_fixed_penalty1` 在 forearm 和 wrist 场景中同时表现出较高成功率和较低碰撞率。
- upper arm 和 elbow 场景中 `ldrc_fixed` 的碰撞率略低，说明安全指标存在方法取舍。

建议用途：

- 新论文正文中作为“当前风险基线”和“全身连杆风险必要性”的实验依据。
- 更完整的分区域表可放入附录。

### 3.4 固定风险惩罚参数筛选

可用数据源：

```text
outputs/sweeps/summary/eval_summary_by_variant.csv
outputs/sweeps/summary/eval_by_train_seed.csv
outputs/paper/final_materials/figure_a1_fixed_penalty_sensitivity.png
```

该部分可以用于说明 `link_fixed_penalty1` 的来源，即 `fixed_risk_penalty=1.0` 是经过小规模筛选后选择的较强当前风险基线。

使用限制：

- 该筛选只有 2 个 train seeds，不能作为正式主比较。
- 只能说明参数选择过程，不能替代 held-out 主表。
- 新论文中应将其放在实验设置、基线选择或附录中。

### 3.5 LDRC 与 adaptive 历史结果

可用数据源：

```text
outputs/formal/summary/
outputs/formal/diagnostics/counterfactual/
docs/experiment_progress.md
docs/experiment_conclusions.md
```

这些结果可作为旧方法诊断，但在新论文中需要降级使用。

可支撑的表述：

- 约束 SAC 在部分场景可能有较低碰撞率，但没有形成整体优势。
- 历史自适应 beta / 滤波执行器没有稳定收益，并显著增加 jerk。
- 因此新论文转向“统一 QP 中处理安全和平滑”，而不是继续使用后级滤波或自适应 beta。

不应使用的表述：

- adaptive 平滑机制有效。
- 当前 adaptive 结果能够证明风险自适应 jerk QP 有效。
- LDRC 是新论文的核心主方法。

## 4. 已有结果不能支撑的最新大纲内容

最新大纲中的以下结论目前尚无实验支撑：

| 大纲内容 | 当前状态 | 所需补充 |
| --- | --- | --- |
| 预测风险比当前风险提前预警 | 未验证 | 需要 Warning Lead Time、未来最小距离误差、风险连杆识别率。 |
| `Predictive-Link-SAC` 优于 `Instant-Link-SAC` | 未验证 | 需要基于预测风险状态重新训练并做 held-out 评估。 |
| 安全 QP 能降低碰撞或安全违反 | 未验证 | 需要 `Predictive-Link-SAC` vs `Proposed` 对比。 |
| QP 介入克制且修正幅度有限 | 未验证 | 需要 Intervention Rate、Intervention Duration、CorrectionNorm。 |
| jerk 约束 QP 提升平滑性 | 未验证 | 需要无 QP、无 jerk QP、有 jerk QP、风险自适应 QP 的消融。 |
| QP 满足实时性 | 未验证 | 需要 Risk Prediction Time、QP Solve Time、Cycle Overrun Rate。 |
| 完整 `Proposed` 方法有效 | 未验证 | 需要完整方法的多 seed held-out 主比较。 |
| 真实 UR5 低速部署有效 | 未验证 | 需要真实平台 10 至 20 次低速验证。 |

因此，论文写作时必须避免把这些内容写成既成贡献。

## 5. 新论文建议使用的实验分层

### 5.1 已完成层：前期基础与基线

可写入当前论文：

```text
EE-SAC
Instant-Link-SAC / link_fixed_penalty1
LDRC-SAC / ldrc_fixed
```

作用：

- 证明末端风险不足；
- 证明当前连杆风险有价值；
- 说明 `link_fixed_penalty1` 是新方法扩展基线；
- 说明约束 SAC 不再作为核心创新。

### 5.2 待完成层：预测风险有效性

最低需要完成：

```text
Instant-Link-SAC
Predictive-Link-SAC
```

重点指标：

```text
Warning Lead Time
未来最小距离误差
风险连杆识别率
Success Rate
Collision Rate
Safety Violation Rate
Final Position Error
```

目标：

- 证明预测风险相较当前风险具有正预警提前量；
- 证明预测风险不会引入不可接受的误报和任务性能下降；
- 证明预测风险能够更准确识别未来高风险连杆。

### 5.3 待完成层：安全 QP 指令整形

最低需要完成：

```text
Predictive-Link-SAC
Proposed
Instant-Link-SAC+QP
```

重点指标：

```text
Collision Rate
Safety Violation Rate
Minimum Distance
Success Rate
Intervention Rate
Intervention Duration
Action Correction Norm
Constraint Active Rate
QP Infeasible Rate
QP Solve Time
RMS/Peak Acceleration
RMS/Peak Jerk
```

目标：

- `Predictive-Link-SAC` vs `Proposed`：验证安全 QP 的整体作用；
- `Instant-Link-SAC+QP` vs `Proposed`：区分预测风险和 QP 安全层的贡献；
- 无 jerk QP vs jerk QP：验证安全和平滑统一整形的作用。

## 6. 推荐论文结果使用方式

### 正文可用

建议正文保留以下旧结果：

1. Random Crossing 中 `ee_fixed`、`link_fixed_penalty1`、`ldrc_fixed` 的主对比。
2. upper arm、elbow、forearm、wrist 的非末端压力测试。
3. 简短说明 `link_fixed_penalty1` 的参数筛选来源。

这些内容应放在第五章实验中的“当前连杆风险基线与前期结果”小节，而不是作为最终 Proposed 方法结果。

### 附录可用

建议附录保留：

1. 各 train seed 的完整 held-out 结果。
2. 固定风险惩罚权重筛选曲线。
3. LDRC 训练诊断。
4. adaptive actor / execution 反事实诊断。

### 不建议继续作为正文核心

以下内容不建议作为正文核心结果：

1. `outputs/formal/summary/` 旧四方法主排序。
2. `ldrc_adaptive` 主比较。
3. adaptive beta 平滑曲线作为正面结果。
4. 旧 `link_fixed`，即非 `penalty1` 配置，作为固定惩罚方法代表。

## 7. 当前论文进度判断

按最新大纲衡量，目前论文处于：

```text
前期基线结果已完成；
新核心方法尚未完成；
正在从“当前连杆风险 SAC”过渡到“预测风险 + QP 指令整形”的阶段。
```

更具体地说：

1. 第二章和第三章中关于当前连杆几何风险基线的部分已有实现和实验支撑。
2. 第五章中“已有当前连杆风险基线”可以开始写。
3. 第三章的预测风险模型可以先写方法，但不能写实验结论。
4. 第四章的 QP 指令整形可以写设计和假设，但不能写有效性结论。
5. 第五章核心实验一、二、三，以及 QP 相关消融仍需新增实验。
6. 第六章真实 UR5 验证仍需现场实验后才能写结果。

## 8. 下一步优先级

建议按以下顺序推进：

1. 实现有限时域预测风险模块，并在已记录轨迹或固定策略轨迹上离线计算 `d_pred`、`T_enter` 和 Warning Lead Time。
2. 完成 `Instant-Link-SAC` 与 `Predictive-Link-SAC` 的同场景对比，确认预测风险是否有独立收益。
3. 实现 jerk 约束安全 QP 指令整形器，先用固定 actor 做反事实测试，验证约束方向、松弛、限位和不可行降级。
4. 加入 `Instant-Link-SAC+QP`，拆分“预测风险收益”和“QP 安全层收益”。
5. 完成风险自适应权重、预测时域、Top-k 活跃约束数量的消融。
6. 在仿真结果稳定后，再开展真实 UR5 低速验证。

## 9. 当前可写结论边界

截至 2026-09-08，可以写：

- 当前连杆级风险相较末端风险是更合理的整臂动态避障状态表示。
- `link_fixed_penalty1` 在已有 held-out 评估中显著提升任务成功率，并降低最终误差。
- 约束 SAC 当前没有显示整体优势，因此不作为最新论文核心创新。
- 历史自适应滤波方案没有稳定收益，促使论文转向统一 QP 指令整形。

截至 2026-09-08，不能写：

- 预测性连杆风险已经提升预警提前量。
- `Predictive-Link-SAC` 已经优于 `Instant-Link-SAC`。
- 风险自适应 jerk 约束安全 QP 已经降低碰撞。
- QP 指令整形器已经满足实时部署要求。
- 完整 `Proposed` 方法已经得到验证。
- 真实 UR5 低速实验已经证明方法有效。

## 10. 一句话总结

当前已有实验结果对最新论文仍然有价值，但它们主要支撑“为什么要从当前连杆风险基线继续做预测风险和安全 QP”。最新论文真正的核心实验，即 `Predictive-Link-SAC`、`Proposed` 和 `Instant-Link-SAC+QP`，截至 2026-09-08 仍需要补做。
