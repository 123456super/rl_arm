# 研究重构：不确定性连杆预测风险、安全过滤器与协同安全强化学习

> 状态快照（2026-08-04）：阶段一已冻结为历史基线；P1/P2 实现与测试完成；P3 B1--B5 已按修正口径完成比较并形成“未冻结”决策。当前状态和结果以 [docs/current](../current/research_status.md) 为准。

## 0. 日期化阶段状态

| 日期/阶段 | 状态 | 说明 |
| --- | --- | --- |
| 2026-07-27--07-31 / P0 | 已冻结 | 阶段一 held-out 主比较及 link_fixed_penalty1 候选不变 |
| 2026-07-29--07-31 / P1-P2 | 开发验证完成 | 风险、过滤器、safe-stop、trace 和测试链路可运行，但不是端到端安全保证 |
| 2026-08-03--08-04 / P3 | 统一比较完成，设计未冻结 | 固定障碍物 0.05 m/s、机械臂 1.0 rad/s；严格 B4 保留为诊断基线，recovery 已拒绝 |
| 后续 / P4-P5 | 未开始 | 等 P3 统一几何/动态可行性/任务性能问题收敛后再决定；实时性仍是后置冻结门槛 |

### 2026-08-03 的研究决策

P3 先按低速双速度统一配置重新建立 B1--B5：障碍物固定 0.05 m/s，机械臂关节速度上限固定 1.0 rad/s；同时固定 frame-corrected 几何、`geometry_margin_m=0.03`、strict QP、recovery/relaxation/maximin 关闭。连杆线速度改为 `J_point qdot` 的 m/s 结果，P3 仅以 PyBullet physical contact 终止，capsule overlap 独立统计。修正前 P3 数据不能作为新方法性能结论。

### 2026-08-03 口径修正

1. `action_scale` 保持为 rad/s，不再直接作为预测模型的 m/s 连杆速度上界。
2. 逐连杆预测速度由胶囊最近点平移 Jacobian与实际关节速度计算；延迟裕度使用当前姿态、关节速度盒下的局部 m/s 上界。
3. 新实验分别报告 `collision_capsule_overlap`、`collision_pybullet_contact`、`collision_any` 和 `termination_collision`。
4. 阶段一继续使用旧综合碰撞定义以保证复现；新 P3 不与阶段一 collision rate 直接横向比较。

### 2026-08-04 冻结后决策

冻结包 `outputs/p3_postfix_dev_100k/p3_freeze_decision.json` 的决策为 `do_not_freeze_p3_or_expand_recovery`。严格 B4 在共享最终集上为 4/432 次 physical contact、78/432 次 success；worst-link recovery 为 22/432 次 physical contact、79/432 次 success，因接触增加而拒绝。B4/B5 无条件初始不安全率分别为 3.5% 和 18.0%。因此本设计文档继续保留方法目标和长期实验矩阵，但不把 P3 结果写成已冻结方法，也不启动 M、OOD 或真机。

## 1. 定位与阶段衔接

本分支的新论文主题为：**面向感知不确定性动态障碍物的机械臂连杆级预测风险约束控制与协同安全强化学习**。

阶段一工作（`ee_fixed`、`link_fixed_penalty1`、`ldrc_fixed`、`ldrc_adaptive`）不废弃。它提供 UR5/PyBullet 环境、连杆胶囊体、动态风险、SAC 训练与评估管线，以及一个重要发现：只替换风险惩罚或约束 SAC 形式，不能稳定解决碰撞控制、任务效率和执行行为之间的取舍。阶段一结果从最终论文结论调整为**预研究基线、问题动机和可复现材料**。

本文件是新主题的研究设计基准。历史结果只描述阶段一，不能和新论文的最终结论混用。

## 2. 研究问题和目标

核心问题：

> 当动态障碍物状态来自存在位置、速度、时延和跟踪误差的感知系统时，如何利用连杆级短时预测风险，对学习策略的关节速度指令实施可解释、可验证的安全约束，同时保持任务效率？

目标如下：

1. 建立考虑连杆几何、相对运动、感知误差与执行时延的预测安全裕度。
2. 设计在线安全过滤器，尽量保留策略动作，必要时修正、限速或安全停止。
3. 将过滤器干预量纳入强化学习训练，使策略主动避险而非长期依赖兜底。
4. 通过因子化消融、独立随机种子、OOD 扰动与低速真机验证安全、效率、保守性和实时性。

## 3. 预期贡献与非主张

1. **风险建模**：提出连杆级预测安全裕度，统一几何距离、相对速度、预测时域、感知误差、感知时延和控制跟踪误差。
2. **安全执行**：提出最小修正策略动作的连杆级安全过滤器，以运动约束、预测安全约束和不可行时安全停止作为命令出口。
3. **协同学习**：用预测风险和过滤器干预代价训练策略，减小策略与执行安全层的失配。
4. **证据链**：分离验证风险表示、过滤器和协同训练的贡献，并报告失效边界。

不宣称对任意障碍物运动、任意视觉误差或硬件故障的绝对安全；不主动进行真机碰撞；不把单一仿真设置外推为广泛泛化。

## 4. 方法架构

```text
RGB-D / 状态估计（位置、速度、时间戳、有效性、误差界）
                         ↓
连杆胶囊体 + 不确定性预测风险 / 保守安全裕度
                         ↓
SAC 产生期望关节速度 qdot_rl
                         ↓
在线安全过滤器（修正、限速、停止）
                         ↓
控制器独立限速、围栏、急停与真实 UR5
```

### 4.1 状态估计接口

每周期使用 `(c_hat, v_hat)`，并传递位置/速度误差界、观测时间戳、检测有效标志和控制时延估计。状态无效、过期或误差界不可用时，不得伪造低风险输入；过滤器必须进入保守降速或安全停止分支。

### 4.2 不确定性连杆预测风险

阶段一的连杆胶囊体继续使用。在短时预测窗口内估计每条连杆与障碍物的最小表面间距 `d_pred,i`，构造：

```text
d_robust,i = d_pred,i - m_geometry,i - m_perception - m_delay - m_tracking
h_i = d_robust,i - d_safe
```

各裕度必须可追溯到几何包络、感知误差、延迟窗口内的最大位移和速度跟踪误差。`h_i` 是安全函数，不能与训练 reward、碰撞事件或统计指标混用。

### 4.3 在线安全过滤器

策略输出 `qdot_rl` 后，过滤器求解与其距离最小的安全关节速度。第一版可采用 QP 或等价投影，至少包括：关节位置/速度/加速度限制、工作空间限制、每条有效连杆的一阶预测安全约束、动作连续性限制；约束不可行、感知失效或状态过期时确定性安全停止。安全过滤器是策略输出后的唯一命令出口，不能旁路。

### 4.4 协同安全强化学习

训练执行过滤后的动作，并记录原始动作、过滤后动作、干预范数、干预类型与停止事件。除任务奖励外，训练应抑制持续预测风险与频繁/大幅干预。首版固定惩罚 SAC 即可；只有在过滤器基线稳定后，才比较约束 SAC 等额外机制，避免无法归因。

## 5. 理论与安全边界

理论目标是在明确假设下给出保守安全条件，而非宣称泛化的 RL 收敛证明：

1. 明确障碍物位置/速度误差、感知和控制时延、连杆模型误差、控制输入与障碍物速度的有界假设；
2. 说明安全裕度如何覆盖最坏情况误差；
3. 在过滤器约束可行、控制器跟踪误差受界时，论证保守安全集的保持条件；
4. 说明约束不可行或感知失效时，安全停止能做什么、不能做什么。

每一条保证都必须列出假设，并用仿真扰动与现场标定检查假设是否合理。

## 6. 实验矩阵

所有方法共享机器人、动作限幅、场景、训练预算和独立 checkpoint 选择。最低因子化比较如下：

| 编号 | 风险表示 | 执行层 | 训练 | 目的 |
| --- | --- | --- | --- | --- |
| B1 | 末端距离 | 无过滤器 | 固定惩罚 SAC | 历史弱基线 |
| B2 | 连杆当前距离 | 无过滤器 | 固定惩罚 SAC | 连杆建模收益 |
| B3 | 连杆预测风险 | 无过滤器 | 固定惩罚 SAC | 动态预测收益 |
| B4 | 连杆预测风险 | 非鲁棒过滤器 | 固定惩罚 SAC | 过滤器收益 |
| B5 | 不确定性预测风险 | 鲁棒过滤器 | 固定惩罚 SAC | 不确定性裕度收益 |
| M | 不确定性预测风险 | 鲁棒过滤器 | 协同安全 RL | 完整方法 |

阶段一 `link_fixed_penalty1` 和 `ldrc_fixed` 可作为补充历史/算法基线，但不能替代 B1--B5 的归因实验。

### 6.1 历史开发与失效诊断（截至 2026-08-03 前）

`configs/experiments/p3/` 的早期 smoke 与旧口径开发结果仅用于链路和失效诊断；B4 的碰撞率为 25%、安全距离违反率为 9.40%，B5 成功率为 0%。这些数字不属于当前冻结包，不能据此声称预测风险、过滤器或不确定性裕度带来整体增益。

点雅可比实现已使常规 PyBullet 内过滤器计算满足当前 50 ms 仿真周期；当前 100k 开发轮次已完成独立 checkpoint validation。过滤器现支持循环投影、Dykstra、主动集回退和可选 OSQP QP。常规 recovery 诊断的 OSQP 平均求解约 2.6--2.7 ms；新 maximin recovery 的单步最高求解约 277 ms，尚不具备实时性。上述测量均不包含感知、通信和控制器延迟。

已增加显式 recovery mode；最新诊断以 `h_min <= 0.06 m` 触发、`h_min >= 0.08 m` 退出，并将障碍物自身运动加入屏障约束 `J_h qdot + h_drift + kappa(h-m) >= 0`。seed 5301 的同一 B4 checkpoint 20-episode trace 显示：原多连杆 recovery 为 4 次碰撞，提前 recovery 为 3 次，放宽不可行预测约束为 2 次，maximin recovery 为 1 次；最后一轮成功 14/20。maximin 分支仅在预测约束不可行时执行，保持关节和工作空间硬约束、最大化危险连杆的最小 clearance 导数，并标记为 `recovery_relaxed`，不代表满足预测安全约束。episode 8 仍碰撞，且 episode 0 需 4.05 s recovery 后仍未完成任务。因此 recovery、约束放宽和 maximin 都未冻结，不得作为安全或实时性证据。

### 6.2 bounded escape 复核结论（2026-07-30）

`primal infeasible` 状态识别和 recovery 参数错位问题已修复，并通过 focused/integration `23 passed` 与 smoke test。使用同一 step 95000 checkpoint、eval seeds 5101/5201、每 seed 20 episodes 的 bounded escape 复核结果为：

| eval seed | success | collision | safety violation steps | recovery triggered/success | mean solve | max solve |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5101 | 10/20 | 1/20 | 27 | 15/14 | 4.32 ms | 176.24 ms |
| 5201 | 13/20 | 1/20 | 0 | 16/15 | 5.23 ms | 198.50 ms |

加入 50 ms simulator-side watchdog 后，seed 5101 有 1 次预算停止、seed 5201 有 7 次预算停止，最大记录耗时仍为 154.59 ms 和 221.09 ms；seed 5201 的安全违反增加到 21 步，碰撞没有下降。该 watchdog 是 post-return 检查，不能替代可中断求解器或独立硬件 watchdog。

由此淘汰 bounded escape、`recovery_relaxed` 和 maximin recovery，不将其作为实时安全机制或论文主结果。随后使用 frame-corrected 几何和 TTC-triggered deterministic escape 完成了 4 组、每组 20 episodes 的复核；合计 success `4/80`、collision `34/80`，且最大过滤耗时 `410.4 ms`。碰撞全部为 `capsule_overlap`，未观察到 PyBullet physical contact；完整漂移审计显示 dynamic-drift 不可行停止碰撞率为 `29/34`。该结果仍未通过 P3 门槛，当前不启动新的训练、P4/OOD 或真机工作。后续优先做根因隔离，不把 deterministic escape 视为已通过的候选方案。

冻结设计后，使用至少 3 个、推荐 5 个未参与开发的 train seeds 和新 held-out eval seeds。训练重复单位是 train seed，不能把 episode 当作独立训练重复。OOD 至少包括障碍物速度/半径/方位、目标位置、位置和速度噪声、观测和控制时延、相机外参扰动。

除成功率、碰撞率、最小距离、最终误差和 jerk 外，还要报告过滤器干预率/范数、停止率、近失事件率、风险检出率、漏检率、误停率、风险提前量、求解时间、超时率和不可行率。

### 6.3 几何修正后的 strict safe-stop 诊断（2026-07-30，历史基线）

UR5 胶囊映射已修正，重点修复 `upper_arm` 的近零长度错误映射，并补齐前臂与腕部段。两个几何修正后的 20k B4 开发训练均已完成：train seed `4102` 选择 step `20000`，train seed `4103` 选择 step `15000`，validation seed 均为 `6201`。

此前使用 `configs/experiments/p3_diagnostics/b4_osqp_strict_margin30_budget.yaml`、eval seed `6101`、每个 checkpoint 20 episodes 得到如下历史结果；后续 deterministic escape 复核见本节之后的结论：

| train seed | checkpoint | success | collision | capsule overlap | PyBullet contact | violation steps | mean solve | max solve |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4102 | 20000 | 7/20 | 1/20 | 0/20 | 1/20 | 123 | 4.42 ms | 206.69 ms |
| 4103 | 15000 | 6/20 | 1/20 | 0/20 | 1/20 | 4 | 4.43 ms | 146.43 ms |

两次碰撞均为 PyBullet 对 `upper_arm_link` 的实际接触，胶囊重叠字段为 false；这是后续 frame-corrected deterministic escape 之前的历史 strict 结果。两组结果存在超过 50 ms 的求解长尾，不能冻结 P3，也不能作为实时安全保证；当前状态以本节后面的 4×20 deterministic escape 和完整漂移审计为准。

对提前降速 trace 的 safe-stop 漂移审计显示，11 个不可行停止 episode 中 4 个为 `h` 下降超过 `0.05 m/s` 的动态漂移型，3 次碰撞全部位于动态漂移型；其余 7 个为静态或缓慢漂移型。该结果说明障碍物持续运动时零速度 safe-stop 不能保证保持安全集；后续方案必须区分静态不可行与动态漂移不可行，并明确任何逃逸动作不再属于严格安全保证。

该阶段不继续训练，不启动 P4/OOD 或真机。后续已完成 frame-corrected deterministic escape 首轮复核，但未通过门槛；当前下一步仅保留 geometry mismatch、dynamic drift、碰撞判据和 infeasible 原因的隔离分析，详见 `docs/archive/p3_pre_20260803/p3_failure_root_cause_analysis.md`。

### 6.4 2026-07-31 TTC 与固定 link Jacobian 复核后的决策

修正固定 link Jacobian 后，relaxed TTC recovery 在 train seed 4102、eval seed 6301 的复核为 success 5%、collision 35%，最大过滤耗时 582 ms；关闭约束放宽的 strict 对照为 success 5%、collision 55%，其中历史标签 `unavoidable_collision=55%`，平均 safe-stop 率约 43%。这里的 `unavoidable_collision` 是当时基于动态漂移和 safe-stop 条件的离线分类，不等同于已证明的物理不可避免碰撞；这组对照说明：

1. 放宽预测约束会把部分碰撞变成可避免但不满足硬约束的碰撞，不能作为安全通过。
2. strict safe-stop 不会因机器人停止而阻止动态障碍物继续接近。
3. 平均求解时间不能掩盖超过 50 ms 的单步长尾。

因此截至 2026-07-31，P3 仍为“诊断完成、设计未冻结”。deterministic escape 已完成首轮复核但未通过固定预算、碰撞分类和任务退化门槛，不能进入新的训练或 P4/OOD。

### 6.5 2026-07-31 V2 根因隔离后的当前状态

V2 2x2 在 80 episodes 上完成了几何与障碍物运动的因子化比较：legacy/moving 为 success `11/80`、collision `6/80`；corrected/moving 为 `8/80`、`39/80`；legacy/static 为 `8/80`、`0/80`；corrected/static 为 `7/80`、`0/80`。corrected/moving 的 collision 中 38 次为 capsule-only、1 次为 PyBullet contact；static 条件无碰撞，支持“动态障碍物运动是主要交互因素”的判断，但不支持把胶囊重叠直接写成物理碰撞。

漂移审计显示 corrected/moving 有 59 个 infeasible-stop episodes，其中 39 个 dynamic drift、20 个 static/slow；dynamic drift 碰撞 34/39（87.2%），static/slow 碰撞 5/20（25.0%）。这确认 zero-velocity safe-stop 不能保持动态安全集。OSQP 归因覆盖 1,823 个 infeasible steps：`predictive_barrier` 1,815 次、`joint_acceleration` 1,103 次，主因是预测屏障与加速度可达性的联合冲突。

四个条件的 mean filter time 为 4.36--5.25 ms、P99 为 6.55--7.47 ms，但 max 为 197--351 ms；当前 post-return budget check 不是硬截止。V2 success 仅 8.8%--13.8%，因此任务性能、实时性和安全闭环都未达到冻结门槛。

**当前决策：** P3 诊断已完成但设计未冻结。允许在低速双速度统一配置下重训/评估，以解决几何一致性、动态可行性、碰撞分类和任务收敛；不进入 P4/OOD、不做真机。实时耗时和硬截止暂时后置。


### 6.6 速度边界与统一配置结果（2026-07-31）

速度边界诊断覆盖实际障碍物速度 0.05/0.10/0.20/0.30/0.38 m/s，每档 80 episodes。原始 CSV 的 speed_mps 展示值错误地除以 1000，已修正为配置实际值。

| speed (m/s) | success | collision | capsule overlap | PyBullet contact | infeasible rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 5.0% | 15.0% | 12 | 0 | 17.8% |
| 0.10 | 6.25% | 33.75% | 27 | 0 | 24.1% |
| 0.20 | 3.75% | 37.5% | 30 | 0 | 23.4% |
| 0.30 | 7.5% | 53.75% | 43 | 3 | 38.4% |
| 0.38 | 6.25% | 65.0% | 52 | 2 | 48.7% |

速度升高会使 collision 和 infeasible rate 总体上升；0.05 m/s 是已测试的最低非零障碍物速度。该边界结果用于选择低速控制点，不代表任何速度是最优安全边界。

### 6.7 低速双速度主控制点（1.0 rad/s，2026-07-31）

为把速度从失败原因中尽量摘出，本轮固定两个独立速度变量：

- 障碍物线速度：speed_range=[0.05,0.05] m/s；
- 机械臂关节速度上限：action_scale=1.0 rad/s。

1.0 rad/s 比历史 0.7 rad/s 上限略高，避免机械臂动作能力不足成为失败原因；joint_acceleration_limit_radps2=4.0 仍固定，单独保留加速约束变量。两种速度分别配置、分别记录，不使用一个数值替代另一个。

该设置已被当前 P3 修正后实验采用；结果和冻结判断以 `outputs/p3_postfix_dev_100k/p3_freeze_decision.json` 为准。robot025 与旧 fixed_speed010 仍仅作历史对照，不与当前主结果混合。

## 7. 真机验证边界

真机只在限速、围栏、急停、保护停、超时停止、相机标定和感知失效停止签核后测试，不主动制造碰撞。以无障碍到达、低风险穿越、上臂/肘部/前臂接近、感知短时丢失和受控时延为主，记录检测状态、时间戳、障碍物估计、风险/裕度、策略与过滤后动作、停止事件、`d_min` 和控制时延。

实机重点量化风险检出、响应提前量、漏检和误停，而不是以真实碰撞率替代安全验证。

## 8. 实施门槛与章节结构

| 阶段 | 工作 | 通过条件 |
| --- | --- | --- |
| P0 | 固化阶段一 | 历史结果可复跑，不再修改其结论 |
| P1 | 预测风险与误差裕度 | **已完成开发验证**：单元测试覆盖几何、预测窗口、误差裕度、无效/未来/过期观测和边界值 |
| P2 | 仿真安全过滤器 | **开发验证完成，诊断已完成多轮**：约束投影、Dykstra/主动集回退、OSQP QP、端到端命令出口、无效/过期状态注入和逐连杆 trace 已具备；常规过滤器通过 smoke test，仍未构成端到端或实机安全保证 |
| P3 | 协同训练与消融 | **因子化比较完成但未冻结**：修正后 B1--B5 已归档；strict B4 为诊断基线，recovery 已拒绝，M 尚未实现 |
| P4 | 独立复核与 OOD | 新 train/eval seeds 和预定义统计完成 |
| P5 | 低速真机 | 现场签核和安全受控试验记录齐全 |

论文建议章节为：绪论；相关工作；预测风险与不确定性裕度；安全过滤和协同学习；理论/系统/失效安全；仿真消融与 OOD；真机验证；结论与局限。

### 6.7 低速双速度主控制点（1.0 rad/s，2026-07-31）

当前主控制点固定两个独立速度变量：

- 障碍物线速度：speed_range=[0.05,0.05] m/s；
- 机械臂关节速度上限：action_scale=1.0 rad/s。

0.05 m/s 是已完成速度边界测试的最低非零障碍物速度；1.0 rad/s 比历史 0.7 rad/s 上限略高，避免机械臂动作过慢成为失败原因。joint_acceleration_limit_radps2=4.0 仍保持不变，因此速度上限与加速约束仍可分别归因。

该设置已完成对应 P3 修正后实验；当前主结果不得回写到旧 `p3_unified_geometry` 目录。robot025 仅作低机械臂速度历史对照。
