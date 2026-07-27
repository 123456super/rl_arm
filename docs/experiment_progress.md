# 仿真实验进展报告（2026-07-27）

## 1. 当前状态

已完成的仿真实验包括：四种初始方法（`ee_fixed`、`link_fixed`、`ldrc_fixed`、`ldrc_adaptive`）的 100k 三-seed训练、validation seed 2001 的 checkpoint selection、五场景统一评估与三-seed汇总；`link_fixed` 固定风险惩罚、`C_safe` 和事件代价的小规模参数敏感性筛选；`link_fixed_penalty1`（`fixed_risk_penalty=1.0`）的 100k 三-seed重训、checkpoint selection，以及与 `ee_fixed`、`ldrc_fixed` 的五场景 eval-seed held-out 评估和统一汇总；并已完成 `ldrc_adaptive` actor / execution 反事实诊断。

最终 eval-seed held-out 主对比表明，`link_fixed_penalty1` 获得最高任务完成表现，并在安全距离违反率、最小距离、最终误差和 jerk 上优于 `ldrc_fixed`；`ldrc_fixed` 仅保留略低的平均碰撞率。`ldrc_adaptive` 没有稳定增益，不作为核心方法。

### 当前进度标记

**当前已进行到：仿真主对比闭环完成，进入论文结果固化与真实低速部署准备阶段。**

主比较的模型训练、checkpoint selection、eval-seed held-out 评估和三-seed汇总均已完成；当前不需要继续训练或重跑主比较。下一项必做工作是将 held-out 三方法结果写入论文正文和结论文件，并在开始真实机器人前完成离线部署链路与低速安全检查。

## 2. 已完成工作

| 工作项 | 状态 | 结果 |
| --- | --- | --- |
| 原四种方法的 100k 训练与评估 | 已完成 | 提供初始消融、checkpoint selection 和 adaptive 诊断；旧主表不再作为最终方法排序依据 |
| 固定惩罚参数敏感性筛选 | 已完成 | 在两 seed、50 episodes 的筛选中，`fixed_risk_penalty=1.0` 是 `link_fixed` 的候选最优值 |
| `link_fixed_penalty1` 100k 重训 | 已完成 | train seeds 为 101、202、303；与原训练预算相同 |
| checkpoint selection | 已完成 | 每个 train seed 使用 validation seed 2001、20 episodes，独立选择 checkpoint |
| held-out 三方法统一评估 | 已完成 | eval seeds 为 1004、1005、1006；135 个 CSV、13,500 episodes |
| 三-seed 汇总 | 已完成 | 已生成按 train seed、跨 seed 和宏平均三类表格 |
| adaptive actor / execution 反事实诊断 | 已完成 | adaptive actor 训练退化，且自适应执行器显著增加 jerk |

## 3. 最终主对比口径

| 维度 | 设置 |
| --- | --- |
| 主比较方法 | `ee_fixed`、`link_fixed_penalty1`、`ldrc_fixed` |
| `link_fixed_penalty1` | 连杆级风险 + 固定风险惩罚 SAC + 固定平滑，`sac.fixed_risk_penalty=1.0` |
| train seeds | 101、202、303 |
| checkpoint validation | validation seed 2001，每 checkpoint 20 episodes |
| held-out eval seeds | 1004、1005、1006 |
| episodes | 每个 eval seed 100；每个场景-方法-train seed 合并 300 episodes |
| 场景 | `random_crossing`、`upper_arm_crossing`、`elbow_crossing`、`forearm_crossing`、`wrist_crossing` |
| 正式评估规模 | 3 methods x 3 train seeds x 3 eval seeds x 100 episodes x 5 scenarios = 13,500 episodes |

参数 `fixed_risk_penalty=1.0` 在早期 train seeds 101、202 的小规模筛选中提出；主结论使用未参与该筛选的 held-out eval seeds 1004--1006，但重训仍复用了 train seeds 101、202。因此，这一评估仅对 eval seeds 独立，尚未对完整的超参数选择和训练随机性完全独立。每个表格单元先对同一 train seed 的 300 episodes 求均值，再在 3 个 train seed 间计算 mean +/- sample std（`n=3`）。

最终主表数据源：

```text
outputs/rechecks/heldout_1004_1006/summary/all_eval_episodes_3methods.csv
outputs/rechecks/heldout_1004_1006/summary/eval_summary_by_train_seed_3methods.csv
outputs/rechecks/heldout_1004_1006/summary/eval_summary_across_train_seeds_3methods.csv
outputs/rechecks/heldout_1004_1006/summary/eval_summary_macro_across_train_seeds_3methods.csv
```

旧文件 `outputs/formal/summary/` 及原 `link_fixed`（惩罚 4.0）结果只保留作历史和敏感性分析，不能再用于主方法排序或正文主表。

## 4. 核心结果

跨五个场景的宏平均，统计量为 3 个 train seed 的 mean +/- std：

| 方法 | success rate | collision rate | final position error | min distance | safety violation rate | RMS jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ee_fixed` | 16.4% +/- 1.4% | 7.69% +/- 2.68% | 0.396 +/- 0.014 m | 0.189 +/- 0.013 m | 1.69% +/- 0.61% | 24.7 +/- 1.5 |
| `link_fixed_penalty1` | 66.7% +/- 15.3% | 6.51% +/- 3.63% | 0.125 +/- 0.041 m | 0.202 +/- 0.011 m | 1.25% +/- 0.63% | 33.0 +/- 3.1 |
| `ldrc_fixed` | 59.3% +/- 20.6% | 5.91% +/- 1.27% | 0.160 +/- 0.067 m | 0.169 +/- 0.012 m | 3.58% +/- 0.98% | 34.4 +/- 0.4 |

主场景 `random_crossing` 中，`link_fixed_penalty1` 达到 64.2% +/- 12.5% 成功率、6.89% +/- 2.22% 碰撞率和 0.125 +/- 0.029 m 最终误差；`ldrc_fixed` 为 56.3% +/- 18.0%、5.56% +/- 2.91% 和 0.172 +/- 0.054 m。

按场景观察，`link_fixed_penalty1` 在五个场景的平均成功率均高于 `ldrc_fixed`；尤其在 forearm 和 wrist 场景分别达到 72.8% 和 76.1%。在 upper arm 和 elbow 场景，`ldrc_fixed` 的碰撞率低于 `link_fixed_penalty1`，因此不能宣称固定惩罚方法在所有安全指标上严格占优。

## 5. Adaptive 诊断

以下诊断沿用原正式矩阵的 `random_crossing` 结果，不属于本次 held-out 三方法主表。在 seed 202、303 上，对 actor 与执行器交叉评估，每个组合 300 episodes：

| train seed | actor / execution | success rate | collision rate | RMS jerk |
| --- | --- | ---: | ---: | ---: |
| 202 | fixed / fixed | 77.7% | 2.0% | 32.6 |
| 202 | fixed / adaptive | 71.3% | 5.0% | 93.5 |
| 202 | adaptive / adaptive | 79.3% | 9.0% | 75.2 |
| 202 | adaptive / fixed | 79.7% | 9.0% | 27.8 |
| 303 | fixed / fixed | 50.3% | 6.0% | 32.7 |
| 303 | fixed / adaptive | 57.0% | 7.0% | 73.2 |
| 303 | adaptive / adaptive | 30.3% | 26.0% | 74.3 |
| 303 | adaptive / fixed | 26.3% | 26.0% | 31.4 |

结论：seed 303 的退化来自 adaptive actor 本身；固定执行器不能恢复其任务和碰撞表现。自适应执行器会显著提高 jerk，且没有稳定的性能收益。因此 `ldrc_adaptive` 保留为失败消融，不纳入当前主对比或真实部署候选。

## 6. 可支持的结论

1. 连杆级风险建模配合恰当标定的固定风险惩罚能够提高单动态障碍物场景下的目标到达能力。`link_fixed_penalty1` 在 eval-seed held-out 的五个场景中均获得最高的三-seed平均成功率，并降低最终位置误差。
2. `link_fixed_penalty1` 与 `ldrc_fixed` 存在碰撞控制折中：后者的宏平均碰撞率略低（5.91% vs 6.51%），但前者在成功率、最终误差、最小距离、安全距离违反率、jerk 和平均风险代价上更好。
3. 当前证据不支持“风险约束 SAC 相对经过调参的固定惩罚 baseline 带来整体增益”的主张。约束方法的优势应限定为部分高风险区域中的略低碰撞率，而非综合性能最优。
4. 固定惩罚权重是强敏感超参数。原 `link_fixed` 取值 4.0 导致过度保守，不能作为固定惩罚方法的代表性最终比较配置。
5. `ldrc_adaptive` 没有显示稳定增益；当前不支持自适应平滑提升任务能力、安全性或平滑性的主张。

## 7. 局限与风险

| 风险 | 影响 | 处理方式 |
| --- | --- | --- |
| 仅 3 个 train seed | 不能开展可靠显著性检验 | 报告 mean +/- std（n=3），不使用“统计显著”措辞 |
| `fixed_risk_penalty=1.0` 来自小规模筛选 | 可能存在配置选择偏差；筛选与重训复用了 train seeds 101、202 | 最终比较使用未参与筛选的 eval seeds 1004--1006；后续应增加未参与筛选的新 train seeds 复核 |
| 固定惩罚与约束方法各有安全指标取舍 | 不能宣称任一方法全安全指标最优 | 同时报告碰撞率、违反率、最小距离和任务指标 |
| adaptive actor seed 303 退化 | 不支持完整方法主张 | 将 adaptive 作为失败消融，不作真实部署候选 |
| 单球形障碍物仿真 | 外部泛化有限 | 限定为本文仿真设定，不作真实安全保证 |

## 8. 剩余工作

### 8.1 必做：论文结果固化

1. 使用 held-out 三方法汇总表制作正文 Table 1（`random_crossing`）和 Table 2（四个定向压力场景）；统计单位明确为 train seed，主方法名称写为 `link_fixed_penalty1` 或“连杆级固定风险惩罚 SAC（w_R=1.0）”。
2. 更新论文、汇报和结论文件中“`ldrc_fixed` 综合最优”以及“`link_fixed` 仅为保守 baseline”的表述；原四方法表格仅作为历史附录，不得混入新主表。
3. 在论文方法与实验设置中说明：`w_R=1.0` 由 train seeds 101、202 的小规模筛选选出，最终主结论使用 eval-seed held-out 的 1004--1006；该流程避免复用筛选阶段的 eval seeds，但仍需新 train seeds 验证以排除完整的配置选择偏差。

### 8.2 必做：真实低速部署前准备

1. 将 `link_fixed_penalty1` 的三个已选 checkpoint 固化为部署候选，并确认推理输入、关节速度限幅、急停和碰撞/距离监控链路。
2. 在不连接真实机械臂或使用严格限位的条件下，先完成离线回放与低速控制链路检查；随后使用轻质球体开展 10--20 次低速可执行性验证。
3. 真实实验仅报告可执行性、轨迹和风险响应，不进行高风险碰撞性基线对比，也不宣称形式化安全保证。

### 8.3 可选：后续研究，不阻塞论文主线

1. 若继续研究约束 SAC，应围绕 upper arm / elbow 的碰撞率优势定位改进目标，并使用新的 train seeds 和独立 held-out eval seeds 验证；不得以当前结果宣称约束机制的整体优势。
2. `ldrc_adaptive` 作为独立改进任务处理：先解决 actor 训练退化和 jerk 增大问题，再进行新的多-seed验证。
