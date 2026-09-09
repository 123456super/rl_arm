# 最新论文实验计划清单

> 初始计划日期：2026-09-08；方法路线修订：2026-09-09
> 绑定进展文档：`docs/current/thesis_experiment_progress.md`  
> 实验结果事实源：`docs/current/thesis_experiment_results.md`  
> 对应论文大纲：`docs/current/thesis_outline.md`  
> 原则：每一步只验证一个关键环节。若结果异常，先停在该步骤定位问题，不把训练、预测、QP 和实机验证混在同一轮实验里。

## 使用方式

本文档是后续实验的执行清单；`docs/current/thesis_experiment_progress.md` 是唯一当前状态记录；`docs/current/thesis_experiment_results.md` 是唯一结果事实源。本文档中的复选框仅表示计划项的执行记录，不替代当前状态表。每完成一个步骤，应同步更新进展文档状态；只有经过确认的结果才写入事实源。

建议记录规范：

```text
计划编号:
代码版本:
配置文件:
输入 checkpoint:
输出目录:
train seeds:
eval seeds:
episode seed 派生规则:
是否通过:
主要异常:
下一步处理:
```

## P0：冻结当前基线状态

- [x] **P0.1 固化已有可用结果**

目标：确认旧结果只作为当前连杆风险基线，不再承担新方法结论。

输入：

```text
outputs/current_results/heldout_1004_1006/final_3methods/
outputs/current_results/paper_final_materials/
outputs/archive_old/formal_legacy/
outputs/archive_old/heldout_two_methods_initial/
docs/current/thesis_experiment_progress.md
```

验证：

- `ee_fixed`、`link_fixed_penalty1`、`ldrc_fixed` 的主表能复现。
- 每个评估文件必须记录 `episode_seed`，不同 eval seed 之间不得出现重复 episode seed。
- 新论文中只把 `link_fixed_penalty1` 表述为 `Instant-Link-SAC` 或当前连杆风险基线。
- 不把 `ldrc_adaptive` 和旧 adaptive beta 写成有效贡献。

通过标准：

- 进展文档中明确写清已完成结果的使用边界。
- 后续所有新实验都以 `link_fixed_penalty1` 作为优先扩展起点。

失败定位：

- 如果表格数值和进展文档不一致，先检查 `outputs/current_results/paper_final_materials/` 是否由 `final_3methods/` 派生。
- 如果方法名混乱，先统一 `link_fixed_penalty1`、`Instant-Link-SAC`、`Predictive-Link-SAC` 的命名映射。
- 旧三方法结果使用了 `episode_seed = eval_seed + episode`，13,500 行中只有 4,590 个唯一实验单元；2026-09-09 已完成修复后的正式重跑，新结果包含 13,500 个无重复实验单元，P0.1 通过。

## P1：离线验证预测风险本身

- [x] **P1.1 实现预测风险计算模块**

目标：先只实现 `d_pred`、`T_enter`、`Risk_i^pred`，不训练新策略，不接 QP。

输入：

```text
src/rl_risk_sac/utils/risk.py
src/rl_risk_sac/collision/detectors.py
tests/test_risk.py
```

建议新增：

```text
src/rl_risk_sac/utils/predictive_risk.py
tests/test_predictive_risk.py
```

验证：

- 静止障碍物且机械臂静止时，预测距离应与当前距离一致。
- 障碍物匀速接近时，`d_pred <= d_now`，`T_enter` 有限。
- 障碍物远离时，预测风险不应高于接近场景。
- 对相同当前距离、不同接近速度的场景，预测风险应区分危险程度。

通过标准：

- 单元测试通过。
- 输出字段至少包含 `d_pred`、`T_enter`、`risk_pred_per_link`、`risk_pred_body`、`critical_link`。

失败定位：

- 若 `d_pred` 异常，先查胶囊体外推和障碍物外推。
- 若风险跳变严重，先查最近点投影、连杆切换和 softmax 聚合。

- [x] **P1.2 在固定轨迹上离线评估预测风险**

目标：不改变策略动作，只在已有或新采样轨迹上对比当前风险与预测风险。

输入：

```text
已有 `link_fixed_penalty1` checkpoint
random_crossing / upper_arm / elbow / forearm / wrist 场景
```

输出建议：

```text
outputs/current_results/predictive_risk_offline_eval/
```

指标：

```text
Warning Lead Time
未来最小距离误差
风险连杆识别率
误报率
漏报率
```

通过标准：

- 至少在接近型场景中得到正的平均 Warning Lead Time。
- 对“近但远离”的障碍物，误报率不能明显高于当前风险。
- 风险最高连杆与实际最近/越界连杆大体一致。

失败定位：

- 如果 Lead Time 没有提升，先调预测时域和 `d_safe`，不要直接进入训练。
- 如果误报过多，先检查 `w_min`、`w_enter`、`R_horizon` 的权重。

## P2：预测风险进入策略的失败消融

- [x] **P2.1 扩展 observation schema**

目标：让环境能输出预测风险状态，但先不训练大规模模型。

建议 schema：

```text
link_risk_pred_v1
```

验证：

- observation 维度固定。
- 所有元素有限值。
- 无障碍物或远离障碍物时，预测风险处于低值。
- 旧 `link_risk_v1` baseline 不受影响。

通过标准：

- smoke test 和 observation 单元测试通过。
- `Instant-Link-SAC` 旧配置仍可运行。

失败定位：

- 如果旧 checkpoint 无法加载，说明 schema 与旧配置没有隔离好。
- 如果训练刚开始就 NaN，先查归一化、clip 范围和风险指数项。

- [x] **P2.2 训练与门控 `Predictive-Link-SAC`**

目标：先确认预测风险状态不会破坏学习，再进行正式三 seed 训练。

建议设置：

```text
train seeds: 101
eval seeds: 1001
训练步数: 10k 至 30k
场景: random_crossing
```

验证：

- reward、cost、success、collision 曲线没有明显异常。
- action、risk、observation 均为有限值。
- 成功率不应明显低于随机或完全停滞策略。

通过标准：

- 小规模训练能稳定结束。
- evaluation CSV 字段完整。

失败定位：

- 如果策略停滞，先降低预测风险奖励权重。
- 如果碰撞率异常升高，先检查预测风险输入方向和尺度。

当前结果：seed 2001 上已完成 13 组、1,300 episodes 的 10k/20k/30k checkpoint validation。Instant-Link 30k 的 success/collision 为 16%/17%；任务表现最好的 predictive `h=0.5` 30k 为 20%/30%，安全性明显下降；能保持 collision 的 predictive 候选 success 仅 5%--7%。随后完成的 100k seed 101 probe 训练稳定，但 40k--100k checkpoint validation 仍未过门控：Instant-Link 100k 为 62%/7%，最佳 Predictive 100k 为 44%/17%，配对 success 改善/恶化 11/29、collision 引入/消除 11/1。因此停止 observation-only 方案，不扩展三 seed。

代码审计确认此前 predictive risk 只进入 observation，固定惩罚 reward 仍只使用 current risk。现已加入默认值为 0 的 `sac.predictive_risk_penalty`，仅对 `predictive_link` 提供 dense early-warning shaping，旧方法和旧配置目标保持不变。下一步用权重 0.1/0.25 做两个 30k seed 101 探针。

raw predicted-risk shaping validation 已完成。相对无 shaping 30k，权重 0.1/30k 将 collision 30% 降至 16%，但 success 20% 降至 13%；相对 Instant-Link 30k 的 16%/17%，也没有净任务收益。权重 0.25/30k 为 9%/23%，更差。该结果与 observation-only 100k 门控共同说明：预测信号在当前任务中直接耦合 SAC 未形成稳定净收益。`Predictive-Link-SAC` 到此停止扩张；已实现但尚未运行的 excess shaping 仅保留为可复现实验资产，不再占用主线预算。

```text
configs/experiments/p2_predictive_checkpoint_validation.yaml
configs/experiments/random_crossing_predictive_link_100k_probe.yaml
configs/experiments/random_crossing_predictive_link_30k_reward0p1.yaml
configs/experiments/random_crossing_predictive_link_30k_reward0p25.yaml
configs/experiments/random_crossing_predictive_link_30k_excess0p25.yaml
configs/experiments/random_crossing_predictive_link_30k_excess0p5.yaml
configs/experiments/p2_predictive_100k_checkpoint_validation.yaml
outputs/current_results/p2_predictive_checkpoint_validation/summary/
outputs/current_results/p2_predictive_100k_checkpoint_validation/summary/
outputs/current_results/p2_predictive_reward_shaping_validation/summary/
```

## P3：动作条件预测进入安全层

- [ ] **P3.0 修复最终主实验的测量与求解语义**

这是进入新方法前的必要工程门，不改变论文主线：

- 在每个 240 Hz 物理子步推进后读取关节位置/速度、胶囊体距离和 PyBullet contact，而不是只在 20 Hz 控制周期末采样。
- 将命令导数与反馈导数分开记录，例如 `command_substep_*` 和 `measured_substep_*`。
- 将现有 `safety_qp_infeasible` 重命名为投影残差失败；修正 `active_constraints` 当前把“满足约束数量”当成“活跃约束数量”的实现。
- 为最终方法显式加入并固定 OSQP 依赖，记录 solver status、iterations、primal/dual residual、slack、solve time 和 cycle overrun。
- 全部 P6 变体使用同一新监测口径；旧结果保留历史字段，不重写原 CSV。

- [ ] **P3.1 实现候选动作条件的逐连杆子步预测**

目标：固定 Instant-Link-SAC actor，评价候选名义动作诱导的机器人子步轨迹与障碍物运动，生成逐连杆、逐子步预测间隙和线性化梯度。

实现边界：候选动作先经过现有 rate limiter 与 Butterworth 得到名义 endpoint；本周期使用 quintic，周期外短时保持 endpoint。先对全部连杆做名义 rollout，再只对活跃连杆—时刻对计算梯度，避免对 `全部连杆 x 全时域 x 全关节` 做无差别有限差分。预测函数不得通过 `resetJointState` 后遗漏恢复仿真状态；优先封装无副作用的运动学查询并增加状态恢复测试。

最低对比：

```text
当前时刻距离/距离变化率
当前速度趋势预测
候选动作条件预测
```

建议设置：

```text
validation seed: 2001
held-out eval seeds: 1004、1005、1006
场景: random_crossing，外加至少一个定向压力场景
```

指标：

```text
Success Rate
Collision Rate
Safety Violation Rate
Minimum Distance
Final Position Error
Warning Lead Time
未来最小距离误差
风险连杆识别率
RMS/Peak Jerk
```

通过标准：

- 候选动作变化时预测轨迹与风险约束方向相应变化。
- 相较 action-independent 预测，未来最小距离误差或危险连杆识别至少一项稳定改善。
- 当前周期使用实际 quintic 过渡，周期外使用终端速度保持假设；滚动到下一周期后必须由新观测重新预测。
- 距离梯度通过有限差分检查，输出能线性化为关于安全终点速度的仿射约束，且数值有限。
- 在预设信赖域内报告距离线性化残差分布，并仅用 validation 分位数确定保守裕量。
- 在 20 Hz 控制预算内报告全流程 prediction time；超时优化顺序是减少冗余预测时刻、约束并集和有限差分次数，不改变核心比较定义。

失败定位：

- 如果预测对候选动作不敏感，检查机器人子步状态是否真正由候选 endpoint 生成。
- 如果线性化误差过大，缩短时域、增加重线性化频率或加入验证集标定的保守裕量。
- 如果 Top-k 漏掉 QP 修正后变危险的连杆，使用当前危险、名义预测 Top-k 和近阈值连杆的并集，并收紧信赖域。

## P4：反应式迭代投影基线（历史配置名 minimal_qp）

- [x] **P4.1 实现最小干预安全 QP**

定位更新：当前迭代半空间投影器与单时刻 closest-link 约束是反应式工程基线，不是最终 Proposed 的凸 QP 实现。配置名和 CSV 字段沿用 `minimal_qp`/`safety_qp_*`，但论文正文称为 `Reactive-Projection`。其计算时间与有限迭代残差不能解释为标准 QP 求解时间和数学不可行率。

目标：先只验证安全约束方向和可行性，不加入 jerk 自适应复杂度。

QP 初始版本：

```text
min ||q_dot - q_dot_nom||^2 + rho ||xi||^2

subject to:
距离变化率安全约束
关节速度约束
关节位置一步预测约束
xi >= 0
```

建议新增：

```text
src/rl_risk_sac/control/safety_qp.py
tests/test_safety_qp.py
```

验证：

- 远离障碍物时 QP 透传名义动作。
- 接近障碍物且名义动作危险时 QP 修改动作。
- 约束不可行时松弛变量为正，并能触发降级标记。

通过标准：

- 单元测试通过。
- QP 输出速度不越界。
- 约束方向经过构造场景验证。

失败定位：

- 如果越修正越危险，优先检查 `n_i` 方向和 `J_i q_dot` 符号。
- 如果经常不可行，先放宽安全阈值或检查速度/位置边界。

- [x] **P4.2 固定 actor 反事实评估最小 QP**

目标：只检验安全层，不重训 actor。

当前工程状态：

- 已完成最小 QP 接入环境与评估指标记录。
- 初版“投影器直接修正已生成子步轨迹”在 random_crossing seed 101/1004 100 episode 中将控制周期采样 collision 从 4% 降到 0%，但子步命令 RMS jerk 从约 110 升到约 280。
- 已改为“QP 修正策略周期目标速度，再交给 RTB 生成子步轨迹”，并通过 `tests/test_safety_qp.py`、`tests/test_safety_qp_env.py`、`tests/test_execution_pipeline.py`、`tests/test_smoke.py`、`tests/test_rtb_env.py`。
- random_crossing seed 101/1004 初步 recheck 已完成：平滑投影器相比 baseline 将控制周期采样 collision 从 4% 降到 0%，safety violation rate 从 1.3865% 降到 0.2015%，子步命令 RMS jerk 基本不变，109.61 -> 109.77。
- 已修复 `episode_seed = eval_seed + episode` 导致的评估样本重叠，改用一一映射的 episode seed，并在 CSV 与汇总 manifest 中强制记录和检查唯一性。
- 修正版三 actor、五场景、三个 held-out eval seed 反事实共 9,000 次 episode 运行；每个执行变体包含 4,500 个无重复实验单元，两个变体按相同 episode seed 成对比较。完整性检查为 90/90 文件、变体内 0 重复 episode seed。
- 五场景宏平均中，反应式投影器相比 RTB baseline：success 61.67% -> 63.71%，控制周期采样 collision 6.91% -> 1.49%，safety violation rate 1.519% -> 0.238%，minimum distance 0.1928 m -> 0.2034 m，子步命令 RMS jerk 109.21 -> 108.76。
- 平均安全层介入率为 3.15% 控制周期，有限迭代后残差失败率为 0.248%，平均投影计算时间为 0.012 ms；仍需诊断 67 个按控制周期采样的残余碰撞 episode。上述数值不代表最终 OSQP 的不可行率或耗时。

事实源与输出：

```text
configs/experiments/minimal_qp_heldout_counterfactual.yaml
outputs/current_results/minimal_qp_heldout_counterfactual_independent_seeds/
outputs/current_results/minimal_qp_heldout_counterfactual_independent_seeds/summary/manifest.json
outputs/current_results/minimal_qp_heldout_counterfactual_independent_seeds/summary/eval_summary_across_train_seeds.csv
outputs/current_results/minimal_qp_heldout_counterfactual_independent_seeds/summary/eval_summary_macro_across_train_seeds.csv
```

本阶段实际对比：

```text
Instant-Link-SAC + RTB baseline
Instant-Link-SAC + RTB + Reactive-Projection（配置历史名 minimal_qp）
```

动作条件预测与最终 `Proposed` 留到 P3/P6；两者固定使用同一批 `Instant-Link-SAC` actors，以隔离“预测安全约束”相对反应式安全层的增益。

指标：

```text
Collision Rate
Safety Violation Rate
Minimum Distance
Success Rate
Intervention Rate
CorrectionNorm
Projection Residual Failure Rate
Projection Compute Time
```

通过标准：

- QP 能降低安全违反或碰撞。
- 成功率下降在可解释范围内。
- 介入不是长期保持或频繁停止。
- 子步命令 RMS jerk 不应明显高于同 checkpoint baseline；直接修正子步的 jerk 尖峰结果只作为失败诊断。

失败定位：

- 如果安全改善来自大量停止，先调整降级策略和安全阈值。
- 如果成功率崩掉，先检查 QP 是否过度保守。

## P5：执行轨迹连续性过渡消融

- [x] **P5.1 endpoint/quintic 加速度和 jerk 边界消融**

目标：验证运动连续性约束本身，不立即加入风险自适应权重。

当前工程状态：子步命令边界、峰值指标和三组消融配置已实现，当前完整测试为 45 passed。validation 使用 seed 2001、三个固定 actor、五场景、每组 20 episodes，共 75/75 文件和 1,500 个无重复实验单元。`8/400` 保持控制周期采样碰撞结果并限制命令峰值，但宏平均投影残差失败率由 0.208% 升至 1.202%，random crossing 达 5.11%。放宽到 `12/500` 后碰撞仍为 2.33%，成功率由 61.33% 变为 62.67%，子步命令最大 acceleration/jerk 为 8.335/500.005；残差失败率仍为 1.196%。这表明 jerk 边界与残差尾部相关，但不能由启发式投影结果断言标准 QP 可行域为空。参数冻结为 `a_max=12 rad/s^2`、`j_max=500 rad/s^3`，正式 held-out 保留无运动边界、仅 `a_max=12`、`a_max=12 + j_max=500` 三组以量化该权衡。

validation 事实源：

```text
outputs/current_results/p5_motion_bounds_validation/summary/manifest.json
outputs/current_results/p5_motion_bounds_validation/summary/eval_summary_macro_across_train_seeds.csv
```

held-out 矩阵：

```text
configs/experiments/p5_motion_bounds_heldout.yaml
```

held-out 已完成：135/135 文件、13,500/13,500 行、0 个重复 episode 单元。`12/500` 相比 minimal QP endpoint 将子步命令全局最大 acceleration 从 51.536 限制到 8.335，将子步命令全局最大 jerk 从 3091.683 限制到 500.005，平均 episode 命令 peak jerk 从 292.41 降至 265.25。宏平均 success 为 63.49% -> 63.18%，collision 为 1.56% -> 1.67%，safety violation 为 0.215% -> 0.203%，投影残差失败率为 0.225% -> 0.274%。该过渡方案只证明 quintic endpoint 能约束发出的命令轨迹；它没有逐子步反馈测量，也不能写成最终统一轨迹 QP 已完成或声称改善碰撞。

held-out 事实源：

```text
outputs/current_results/p5_motion_bounds_heldout/summary/manifest.json
outputs/current_results/p5_motion_bounds_heldout/summary/eval_summary_macro_across_train_seeds.csv
```

对比：

```text
minimal QP
QP + acceleration bound
QP + acceleration bound + jerk bound
```

指标：

```text
RMS/Peak Acceleration
RMS/Peak Jerk
Collision Rate
Safety Violation Rate
Success Rate
Projection Residual Failure Rate
```

通过标准：

- jerk 峰值明显下降或被硬边界限制。
- 安全指标不明显恶化。
- 投影残差失败率可解释，不将其误称为标准 QP 不可行率。

失败定位：

- 如果 jerk 降了但碰撞升高，说明运动边界压缩了避障响应，需要调 `a_max`、`j_max` 或提前预警阈值。
- 如果投影残差失败率高，先检查历史速度初始化、jerk 公式和边界交集。

- [ ] **P5.2 实现轨迹一致的 One-Step-QP 基线**

目标：使用 OSQP 等标准求解器，使安全终点速度唯一确定本周期下发的 quintic 子步命令；对全部子步施加位置、速度、acceleration 和 jerk 约束，并将当前时刻的距离/Jacobian/障碍物速度冻结后施加到各子步速度。该基线解决执行一致性，但不更新未来几何，因此与 P6 的动作条件多时刻预测约束有清晰差别。

对比：

```text
endpoint-only motion bounds
trajectory-embedded motion bounds
One-Step-QP: frozen-current-geometry safety + trajectory motion bounds
```

指标：

```text
每个子步 Motion-Limit Violation Rate
每个子步 Safety Constraint Residual
QP Status / Slack / Solve Time
Collision / Success / Peak Jerk
```

通过标准：

- 优化器内部子步命令与实际下发命令逐项一致。
- 命令 motion bounds 在数值容差内全部满足，并分别报告关节反馈边界。
- 标准 solver status、残差和 slack 定义正确，求解时间低于控制周期。
- 相比 endpoint-only，逐子步监测安全指标不恶化。

失败定位：

- 如果安全约束导致不可行，增加有界独立 safety slack；命令运动边界保持硬约束，求解失败时进入明确降级路径。
- 如果求解时间过高，先利用仿射结构预计算矩阵、warm start，并在不漏掉峰值系数的前提下减少冗余运动约束。

## P6：动作条件预测轨迹 QP 主实验

- [ ] **P6.1 运行最终主比较**

目标：形成新论文第五章的核心结果。

方法矩阵：

```text
EE-SAC
Instant-Link-SAC
Predictive-Link-SAC（失败消融）
Instant-Link-SAC+Reactive-Projection（已有）
Instant-Link-SAC+One-Step-QP
Proposed: Instant-Link-SAC+Predictive-Trajectory-QP
LDRC-SAC
```

最低必须完成：

```text
Instant-Link-SAC
Instant-Link-SAC+Reactive-Projection
Instant-Link-SAC+One-Step-QP
Proposed: Predictive-Trajectory-QP
```

建议设置：

```text
actor seeds: 既有 101、202、303 checkpoints（checkpoint step 按归档配置，不统称为全部 100k）
held-out eval seeds: 与调参完全分离
场景: random_crossing，upper_arm，elbow，forearm，wrist
```

通过标准：

- 每个创新结论都有对应消融。
- `One-Step-QP` 与 `Proposed` 使用相同 actor、RTB 参数、OSQP、运动边界、slack 形式和 episode seeds；核心差异仅为冻结当前几何与动作条件多时刻几何。
- `Proposed` 相比 `One-Step-QP` 降低逐子步碰撞、安全距离违反或高 slack 尾部中的至少一项，且成功率没有不可接受下降。
- `Proposed` 保持命令 acceleration/jerk 边界，并报告反馈侧的实际跟踪结果。
- QP 求解时间满足控制周期。

失败定位：

- 如果 Proposed 不优于 One-Step-QP，只回到 validation 检查动作条件外推误差、Top-k、信赖域和保守裕量，不修改 held-out 参数，也不重新更换论文主问题。
- 如果 One-Step-QP 与现有投影器差异异常，先核对约束符号、slack 语义和逐子步监测，不能把求解器差异误写成预测收益。
- 风险自适应权重不参与本轮成败判定；固定参数主方法完成后，才决定是否值得做该附加消融。

## P7：鲁棒性与泛化

- [ ] **P7.1 逐项加入扰动**

目标：一次只改变一个扰动因素，避免无法判断失败来源。

顺序：

```text
位置噪声
速度误差
控制延迟
障碍物速度突变
未见速度范围
双球形障碍物
```

通过标准：

- 每个扰动都有单独表格。
- 报告性能下降曲线，而不是只给最终平均值。
- 明确哪些失效来自预测模型，哪些来自 QP 可行域。

失败定位：

- 噪声失败：优先查速度估计和低通滤波。
- 延迟失败：优先查预测时域和安全裕度。
- 双障碍失败：优先查 Top-k 约束数量和 QP 求解时间。

## P8：真实 UR5 低速验证

- [ ] **P8.1 离线部署预检复核**

目标：在接触真实机器人前，使用 P6 冻结的 `Proposed` 配置和既有 Instant-Link actor，确认模型、观测、标准 QP、降级逻辑和最终指令链路一致。实机是条件性验证项，不是仿真主线完成的前置条件。

输入：

```text
P6 最终选定的 Proposed 配置
P6 使用的既有 Instant-Link actor checkpoints
scripts/deployment_preflight.py（需先扩展为支持 Proposed）
outputs/archive_old/deployment_preflight_legacy/offline_report.json
```

候选约束：

- 既有 `link_fixed_penalty1` checkpoints 在 EMA 环境下训练；只有在 P6 的统一 RTB 评估和离线预检通过后，才可作为低速实机候选，且必须披露执行器迁移。
- 当前归档中的 `offline_report.json` 仅是旧 `link_fixed_penalty1` 基线预检记录；`Proposed` 的新预检报告尚未生成。
- 部署候选必须与论文主实验使用相同的 observation schema、预测风险参数、安全 QP、速度/位置/加速度/jerk 约束和子步监测管线。
- `scripts/deployment_preflight.py` 当前仍绑定旧 `link_fixed_penalty1` 三 seed 候选；完成 P6 后必须先解除该绑定，再生成新的 Proposed 预检报告。

通过标准：

- checkpoint 可加载。
- 观测维度和训练配置一致。
- 策略输出、预测风险、QP 输出和最终执行指令均为有限值。
- 归一化动作、关节速度、关节位置、加速度和 jerk 不超过配置边界。
- QP 不可行、松弛超限、感知丢失、无效深度、障碍物状态过期或风险超阈时，能够在指令发往机器人前进入安全停止。
- 离线报告记录配置版本、checkpoint、observation schema、约束参数、QP 状态和全部失败项。

离线预检只验证无硬件推理和仿真指令链路，不构成实机安全验收或形式化安全保证。

- [ ] **P8.2 现场安全签核**

在真实控制器与安全单元上逐项确认：

- 将仿真速度映射为更保守的真实关节速度、加速度、jerk 和工作空间限制，并由控制器强制执行。
- 在安全姿态下实测物理急停、保护停、控制器断连和命令超时的停止路径。
- 实测 RGB-D 检测丢失、无效深度、障碍物状态过期、风险超阈和 QP 失败时的停止行为。
- 完成相机—机械臂外参、TCP、胶囊体保守包络和距离监控的实物复核。
- 先空载、无障碍、最低速度完成单步和短轨迹测试，再允许轻质球体低速实验。

- [ ] **P8.3 真实低速可执行性验证**

目标：只验证低速部署可行性，不故意制造碰撞。

最低对比：

```text
Instant-Link-SAC
Proposed
```

建议次数：

```text
每组 10 至 20 次
```

指标：

```text
Minimum Distance
Safety Violation Rate
Intervention Rate
CorrectionNorm
Task Success
QP Solve Time
感知丢失次数
人工急停/保护停次数
```

通过标准：

- 能稳定完成低速执行。
- QP 或降级逻辑在高风险时能触发。
- 不宣称绝对安全或高速安全。
- 只报告可执行性、轨迹、风险响应和人工/保护停止，不安排故意碰撞的高风险基线对比。

失败定位：

- 如果实机比仿真保守，优先查相机外参、安全距离和速度估计延迟。
- 如果执行抖动，优先查底层速度伺服周期和 jerk 边界。

## 最小可写论文路径

如果时间有限，优先完成：

1. P3.0：逐物理子步监测和历史指标语义修复。
2. P3.1：候选动作条件间隙与线性化误差验证。
3. P4：已完成的 Reactive-Projection 固定 actor 反事实基线。
4. P5.2：标准 One-Step-QP 与轨迹一致命令边界。
5. P6：Instant、Reactive-Projection、One-Step-QP、Predictive-Trajectory-QP 主比较。

暂缓：

- 双障碍物泛化。
- 大规模真实实验。
- LDRC 深度改进。
- 自适应权重的大范围搜索。

## 与进展文档的绑定规则

`docs/current/thesis_experiment_progress.md` 中的状态表应使用本文档编号更新：

| 计划编号 | 对应进展项 |
| --- | --- |
| P0 | 已有结果固化与使用边界 |
| P1 | 预测性连杆风险 |
| P2 | `Predictive-Link-SAC` 失败消融与方法选择依据 |
| P3 | 动作条件预测与约束线性化 |
| P4 | 反应式迭代投影工程基线 |
| P5 | 轨迹一致 One-Step-QP 与运动命令约束 |
| P6 | 动作条件 Predictive-Trajectory-QP 完整方法 |
| P7 | 鲁棒性与泛化 |
| P8 | 真实 UR5 低速验证 |

每完成一个编号，应在进展文档中追加：

```text
完成日期
输出目录
关键指标
是否通过
是否进入下一步
```
