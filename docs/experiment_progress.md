# 阶段一进展与新研究分支计划（2026-07-28）

## 1. 当前状态

已完成的仿真实验包括：四种初始方法（`ee_fixed`、`link_fixed`、`ldrc_fixed`、`ldrc_adaptive`）的 100k 三-seed训练、validation seed 2001 的 checkpoint selection、五场景统一评估与三-seed汇总；`link_fixed` 固定风险惩罚、`C_safe` 和事件代价的小规模参数敏感性筛选；`link_fixed_penalty1`（`fixed_risk_penalty=1.0`）的 100k 三-seed重训、checkpoint selection，以及与 `ee_fixed`、`ldrc_fixed` 的五场景 eval-seed held-out 评估和统一汇总；并已完成 `ldrc_adaptive` actor / execution 反事实诊断。

最终 eval-seed held-out 主对比表明，`link_fixed_penalty1` 获得最高任务完成表现，并在安全距离违反率、最小距离、最终误差和 jerk 上优于 `ldrc_fixed`；`ldrc_fixed` 仅保留略低的平均碰撞率。`ldrc_adaptive` 没有稳定增益，不作为核心方法。

### 当前进度标记

**阶段一已闭环；P1/P2 已完成开发验证；P3 的 B1--B5 单开发 seed 短训练与检查评估已完成，但设计尚未冻结。**

主比较的模型训练、checkpoint selection、eval-seed held-out 评估和三-seed汇总均已完成；它们被冻结为阶段一基线，当前不需要继续训练或重跑。新研究的实施基准见 [research_direction.md](research_direction.md)：B1--B5 已在一个开发 train seed 和一个开发 eval seed 上完成 10k step 链路检查，但没有稳定的任务--安全增益，不能进入 P4 或作为论文证据。

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
| P1 不确定性连杆预测风险 | 已完成（开发验证） | 已实现时间戳状态估计接口、短时最小预测距离、几何/感知/时延/跟踪裕度和 `h`；无效、未来或过期观测不会产生低风险结果 |
| P2 仿真安全过滤器 | 已完成（开发验证） | 策略命令可经连杆预测约束、关节位置/速度/加速度和 TCP 工作空间约束投影；不可用风险、无效输入或不可行约束均输出零速度 |
| P2 端到端失效注入与指标 | 已完成（开发验证） | 可注入有效、无效或过期障碍物估计；运行时记录预测状态、`h_min`、原始/过滤后命令、干预量、停止状态、违反量与求解耗时 |
| P2 解析点雅可比与仿真实时性 | 已完成（开发验证） | 以 PyBullet 点雅可比替代有限差分状态保存/恢复；20 episode、2,693 个控制步中滤波器均值 2.68 ms、P95 4.94 ms，无步超过 50 ms |
| P3 B1--B5 开发轮次 | 已完成（未冻结） | 共用 train seed 4101、10k step、eval seed 5101 的 20 episode 检查完成；结果未形成稳定综合优势 |

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
outputs/rechecks/heldout_1004_1006/final_3methods/all_eval_episodes.csv
outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_by_train_seed.csv
outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_across_train_seeds.csv
outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_macro_across_train_seeds.csv
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

## 8. 已固化事项与后续工作

### 8.1 已完成：阶段一结果固化

1. 已使用 held-out 三方法汇总表生成正文 Table 1（`random_crossing`）和 Table 2（四个定向压力场景），统计单位为 train seed，方法名称统一为 `link_fixed_penalty1` 或“连杆级固定风险惩罚 SAC（`w_R=1.0`）”。
2. 阶段一结论文件已统一为“`link_fixed_penalty1` 是历史综合候选；`ldrc_fixed` 仅在部分场景保留较低碰撞率”的口径；原四方法表格仅作为历史附录。
3. `docs/paper_materials.md` 已列出正文/附录图表、数据源与一键重建命令，并明确 `w_R=1.0` 筛选与 train seed 复用的局限。

### 8.2 已完成：阶段一策略离线预检；实机签核待现场完成

1. 三个 `link_fixed_penalty1` checkpoint 已固化为部署候选：train seeds 101、202、303 分别选择 step 100000、50000、100000。
2. `scripts/deployment_preflight.py` 已完成无硬件预检：确认 checkpoint 可加载、推理输入和命令均为有限值、归一化动作未越界，并在 PyBullet 中确认关节速度命令不超过 `0.7 rad/s` 仿真限幅。
3. 真实机器人控制接口、控制器限速/工作空间、急停与保护停、RGB-D 失效安全停止、相机外参和碰撞包络仍需在现场签核；完成后才可使用轻质球体开展 10--20 次低速可执行性验证。
4. 阶段一策略不可作为新论文的最终部署方法。新系统须先完成安全过滤器、感知失效分支和独立离线预检，才可进入现场签核；真实实验仍不进行高风险碰撞性基线对比。

### 8.3 已完成：P1/P2 仿真开发验证

1. [预测风险模块](../src/rl_risk_sac/utils/predictive_risk.py) 已实现连杆胶囊体短时最小表面距离、逐连杆几何裕度，以及感知、时延和跟踪误差裕度。输出包括 `d_pred`、`d_robust`、安全函数 `h`、状态年龄和可用性状态。
2. [安全过滤器](../src/rl_risk_sac/utils/safety_filter.py) 已以半空间投影实现 `J_h qdot + kappa h >= 0`，并同时处理关节位置、速度、加速度/命令连续性与外部工作空间约束。风险不可用、输入无效或约束不可行时，过滤器确定性输出零速度。
3. [UR5 仿真环境](../src/rl_risk_sac/envs/ur5_dynamic_obstacle_env.py) 在 `env.safety_filter.enabled=true` 时将过滤器作为策略命令的唯一出口。通过 PyBullet 点雅可比和固定最优投影参数构造连杆安全函数与 TCP 工作空间雅可比；默认配置保持关闭，不改变阶段一结果。
4. 已提供感知估计注入接口，可在仿真中复现无效和过期状态。端到端指标包括预测状态、`h_min`、原始与过滤后动作、干预范数、停止状态、活动约束数、最大违反量和求解时间。
5. 当前 P2/P3 聚焦回归结果为 `22 passed`，且 P2、B4、B5 的 PyBullet smoke test 已通过。有限差分已替换为 PyBullet 点雅可比；在 P2 的 20 episode、2,693 控制步测量中，过滤器均值求解时间为 2.68 ms、P95 为 4.94 ms，未出现超过 50 ms 控制周期的滤波器计算。它仅证明当前 PyBullet 进程内的过滤器计算可满足该周期，不覆盖感知、通信、控制器和真实硬件的端到端延迟，也不证明过滤器降低碰撞率或可直接用于真机。

### 8.4 当前状态：P3 因子化开发轮次已完成，设计未冻结

1. 已在 `configs/experiments/p3/` 下完成 B1--B5：共用 train seed `4101`、10k step 和 eval seed `5101` 的 20 episode 检查，输出隔离在 `outputs/p3_dev/b*/`。开发汇总如下，统计单位是 episode，仅用于发现问题：

| 方法 | success | collision | safety violation rate | 说明 |
| --- | ---: | ---: | ---: | --- |
| B1 末端当前距离 | 10% | 20% | 4.65% | 历史弱基线 |
| B2 连杆当前距离 | 15% | 20% | 4.48% | 任务误差改善，但无安全增益 |
| B3 连杆预测风险 | 5% | 15% | 6.58% | 碰撞略低但任务和距离违反变差 |
| B4 预测风险 + 非鲁棒过滤器 | 15% | 25% | 9.40% | 介入率 34.96%，不可行停止率 4.60%，未显示安全收益 |
| B5 鲁棒预测风险 + 鲁棒过滤器 | 0% | 20% | 4.55% | 介入率 47.96%，不可行停止率 8.23%，任务过于保守或训练不足 |

2. B4/B5 的过滤器在 PyBullet 内分别达到 P95 3.89 ms 和 4.88 ms，均无控制步超过 50 ms；实时性不再是当前开发瓶颈，但这不是端到端或实机实时性结论。
3. 当前结果不支持冻结 B1--B5 参数、不支持进入 P4/OOD，也不支持把 B4/B5 写成降低碰撞率的证据。下一轮应保持 B1--B5 的风险表示、裕度和过滤器参数不变，将共同开发预算从 10k 扩展到 100k，并以独立 checkpoint validation 后的同一开发评估口径复查。若完整开发预算后仍无可用任务--安全折中，应停止扩展到新 seeds，转而诊断奖励、风险代价和不可行约束来源。
4. 设计冻结后，才可用新的 train seeds 和 held-out eval seeds 完成独立复核与 OOD 扰动；当前 P1/P2 测试和单 seed P3 结果不得替代此证据。真机工作仍停留在现场签核前。
