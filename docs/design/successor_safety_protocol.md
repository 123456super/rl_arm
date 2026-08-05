# 后继方案协议草案：可行安全集感知的连杆预测风险控制

> 状态：`approve_g0_only`，2026-08-05。本文定义 P3 冻结后的独立后继方案；当前只授权可行性标签、严格性测试和 V0/V1 动作等价性验证的实现。coverage audit、训练、评估、OOD 和真机仍未授权。

## 1. 决策与范围

P3 冻结包已拒绝 predictive-barrier recovery：在共享最终集上，严格 B4 为 4/432 次 physical contact、78/432 次 success；worst-link recovery 为 22/432 次 physical contact、79/432 次 success。因此后继方案不得把放宽预测约束、escape、maximin 或任何 recovery 动作作为安全机制。

本协议提出的后继方向为：**可行安全集感知的连杆预测风险控制**（viability-aware link predictive safety control，简称 VAPS）。它首先回答“当前状态在既定不确定性与动力学约束下是否仍存在严格安全动作”，再让策略学习避免进入模型判定的不可行区域；它不在不可行后发出违反预测约束的补救动作。

论文问题保持不变：在位置、速度、时延和跟踪误差受界的动态障碍物场景中，如何以连杆级预测风险约束机械臂命令，同时保持任务效率。本文只定义仿真内研究协议，不作硬实时、广泛泛化、绝对安全或真机部署主张。

## 2. 已知失败与研究假设

### 2.1 已知事实

1. `initial_h_min_m > 0` 只说明当前裕度为正，不能说明在障碍物持续运动、关节速度/加速度约束下存在后续严格安全动作。
2. 严格 B4 的 safe-stop 会在动态漂移场景中保持零关节速度，但不能阻止障碍物继续接近。
3. recovery 虽可减少部分 QP 不可行，却增加了 physical contact；因此“不可行后逃逸”不能作为安全主张。
4. 300 ms 是过滤器返回后的仿真侧检查，不是可中断求解器、外部 watchdog 或端到端实时保证。

### 2.2 待检验假设

- H1：基于严格约束的短时可行性判定能将“当前裕度正但后续无严格安全动作”的状态单独标注，不再误称为 safe-stop 可保证安全。
- H2：策略在训练中感知可行性余量和过滤器干预后，可以在不增加 physical contact 的前提下减少进入模型不可行区的频率，并提高任务成功率。
- H3：在固定不确定性界内，严格约束、可行性标签和执行动作的联合日志足以解释失效边界；无法满足假设时应报告模型不可行，而不是声称物理不可避免。

## 3. 安全目标与术语

### 3.1 固定符号

对每条胶囊连杆 `i`，沿用现有预测风险定义：

```text
d_robust,i = d_pred,i - m_geometry,i - m_perception - m_delay - m_tracking
h_i = d_robust,i - d_safe
```

`h_i` 的单位为 m；连杆线速度和上界分别使用 `*_mps`，关节命令和加速度分别使用 `*_radps`、`*_radps2`。不得把 `action_scale` 的 rad/s 数值写入任何 m/s 字段。

### 3.2 运行时状态

每个控制周期必须产生以下互斥状态之一：

| 状态 | 定义 | 可作出的主张 | 控制动作 |
| --- | --- | --- | --- |
| `certified_viable` | 在本协议的不确定性界和预测窗口内，严格约束存在可行命令序列 | 仅可称“模型内严格可行” | 输出严格 QP 的最小修正命令 |
| `model_infeasible_dynamic` | 严格约束不可行，且预测裕度漂移低于 `-0.05 m/s` | 不得称物理不可避免 | 零速度 fail-safe，并记录事件 |
| `model_infeasible_static_or_slow` | 严格约束不可行，但未满足动态漂移条件 | 不得称物理不可避免 | 零速度 fail-safe，并记录事件 |
| `invalid_or_stale_observation` | 感知无效、过期或误差界缺失 | 不能评价风险模型 | 零速度 fail-safe |
| `unknown_compute_budget` | 在仿真侧预算检查或求解器未给出可信严格结果时无法判定严格可行性 | 不得称实时通过或严格不可行 | 零速度 fail-safe，并单列统计 |

`model_infeasible_*` 是模型和约束集合下的结论，不等同于物理碰撞不可避免。任何违反严格预测约束的动作都必须标记为诊断动作，不能计入 VAPS 方法或安全通过。

### 3.3 可行性定义

令控制周期为 `dt = 0.05 s`，可行性窗口为 `H_v = 0.15 s`，即三个控制周期。状态 `x_t` 被标为 `certified_viable`，当且仅当存在命令序列 `(u_t, u_{t+1}, u_{t+2})`，使得在整个窗口内同时满足：

1. 关节位置、关节速度、关节加速度、工作空间和动作连续性约束；
2. 所有有效连杆的严格预测安全约束，且不使用 slack、relaxation 或 recovery；
3. 障碍物速度、观测年龄、控制时延、跟踪误差和几何裕度均不超过本协议给出的界；
4. 使用常速度障碍物外推作为模型假设。该假设失效时，状态必须降级为 `invalid_or_stale_observation` 或单独报告模型失配。

实现允许采用滚动多步 QP、等价可达集检查或经验证的一步保守充分条件；任何近似都必须说明其是充分条件、必要条件还是启发式分类器。只有严格可行结果可以产生 `certified_viable` 标签。

G0 的 V1 监视器实现采用 `one_step_strict_sufficient`：当前周期的预测风险有效且严格过滤器已给出无 slack、无 relaxation 的可行命令时，记录为可执行的一步保守充分条件。它不新增三步优化器、不修改 `qdot_cmd`，且在其求解器状态字段中明确标识；进入 V2 前仍须按本节定义完成多步可行性实现审查。

## 4. 固定研究设置

以下设置属于后继协议的固定条件；后续仅可在新的、预先声明的敏感性实验中改变，不能为某个方法或 checkpoint 回调。

| 项目 | 固定值 | 依据 |
| --- | ---: | --- |
| 机器人与几何 | UR5，frame-corrected capsules | P3 修正后统一模型 |
| 控制周期 | `0.05 s` | 现有 20 Hz 仿真控制 |
| 物理步长 | `0.0041666667 s` | 现有 PyBullet 设置 |
| 障碍物半径 | `0.07 m` | 现有场景定义 |
| 障碍物速度 | `0.05 m/s` | 已测试的最低非零动态点 |
| 关节速度上限 | `1.0 rad/s` | P3 修正后统一口径 |
| 关节加速度上限 | `4.0 rad/s2` | 当前严格可达性约束 |
| `d_safe` | `0.12 m` | 当前风险定义 |
| 几何裕度 | `0.03 m` | P3 修正后统一口径 |
| 预测/可行性窗口 | `0.15 s` | 与现有预测窗口一致 |
| 最大观测年龄 | `0.10 s` | 现有感知有效性边界 |
| 控制时延 | `0.05 s` | 论文不确定性目标的名义延迟 |
| 跟踪误差界 | `0.01 m` | 论文不确定性目标的名义界 |
| 碰撞终止 | `physical_contact` | capsule overlap 独立保守统计 |
| 过滤器 | strict QP | 保持安全约束含义清晰 |
| recovery/relaxation/maximin | 全部关闭 | P3 已拒绝分支 |
| 仿真侧耗时门槛 | `300 ms`，预算停止率为零 | 只作离线筛选，不是硬实时 |

`rad/s2` 仅用于表格可读性；代码字段必须使用 `_radps2`。VAPS 的第一轮不得新增 position/velocity noise、外参扰动、OOD 速度或真机条件；这些属于后续阶段，必须单独预注册。

## 5. 方法和对照

| 编号 | 方法 | 是否训练 | 用途 |
| --- | --- | --- |
| V0 | 严格 B4 复现：预测风险 + strict QP | 是 | 在新 seed/新无条件 reset 清单上的主对照 |
| V1 | V0 + 可行性监视器，只记录标签，不改变 `qdot_cmd` | 否，复用 V0 actor | 验证标签、日志与严格控制链路隔离 |
| V2 | VAPS：策略观察可行性余量，训练执行 `qdot_cmd` 并惩罚进入模型不可行区/过大干预 | 是 | 候选后继方法 |

V1 与 V0 在相同 actor、seed 和确定性执行条件下，`qdot_cmd` 必须逐步一致到数值容差；否则先修复隔离性，不进入 V2。V2 只可在 V1 通过后实现。

V2 的训练记录必须区分 `qdot_requested` 与严格过滤后的 `qdot_cmd`。训练代价至少包括连续可行性余量、过滤器干预范数和 safe-stop 事件；其权重只能用独立 validation 清单预先定义的有限候选集合选择，最终清单不得参与调参。

## 6. 种子、初始状态与数据划分

为消除 P3 的共享可行初始集覆盖问题，本协议以**无条件 reset 分布**为主评估对象；`certified_viable` 子集只用于条件安全主张，绝不替代全分布结果。

| 角色 | 固定集合 | 用途 |
| --- | --- | --- |
| train seeds | `4301, 4302, 4303, 4304, 4305` | V0、V2 的独立训练重复 |
| validation manifest | `8201`--`8240`，40 episodes | checkpoint 和有限超参数候选选择 |
| final manifest | `9001`--`9200`，200 episodes | 一次性最终比较，禁止调参 |
| coverage audit | `10001`--`11000`，1000 resets | 无条件初始安全/可行覆盖率审计 |

四类种子互斥。所有方法使用同一 validation/final 清单；不得按方法、checkpoint 或 `certified_viable` 标签重新采样。训练重复的统计单位是 train seed，episode 只用于计算每个 train seed 的比率。

### 6.1 任务可行性预检查标签

在任何后继训练或性能比较前，必须对每个 validation/final reset 生成以下三个**离线标签**。标签用于解释样本与分层报告，不得用于删除 episode、重新采样或改变全分布主结果。

1. `ik`：以固定 URDF、关节限位和末端位置容差做多初值 IK。`reachable` 表示至少找到一个满足限位、末端误差和自碰撞检查的关节解；`not_reachable` 表示当前 IK 搜索未找到，不可写成工作空间中的绝对不可达。
2. `no_obstacle_task`：对 IK 候选生成满足同一关节位置、速度、加速度和末端工作空间约束的平滑关节路径。`candidate_path_found` 表示无障碍时找到候选任务路径；`not_found` 仅表示该固定候选搜索未找到。
3. `dynamic_obstacle_path`：沿 reset 时确定的障碍物轨迹和反弹边界，对同一候选路径检查全部胶囊的保守净距。净距必须包含 `d_safe`、几何裕度、跟踪误差和控制时延位移。`candidate_path_found` 只是存在性下界；`not_found` 不得解释为物理上绝对无路可走。

预检查实现为 `configs/experiments/vaps/v3_task_feasibility_precheck.yaml` 和 `scripts/audit_vaps_task_feasibility.py`。它不加载 actor、不执行策略命令、不启用 recovery/relaxation，也不修改运行时控制。输出必须保留每个 seed 的目标、初始关节状态、障碍物初始状态、三个标签、候选数和失败原因。

正式报告必须同时给出全部 reset 分布及按这三层标签分组的结果。不能因某一标签为 `not_found` 而将其从 physical contact、success、不可行率或覆盖率的全分布统计中移除。

每次 final 评估必须同时输出：

1. 全部 200 个无条件 reset episode 的结果；
2. 协议参考可行性检查标记为 `certified_viable` 的子集结果及覆盖率；
3. `invalid_or_stale_observation`、`model_infeasible_dynamic`、`model_infeasible_static_or_slow` 和 `unknown_compute_budget` 的 episode/step 比率。

参考可行性检查必须固定为 V1 的严格模型和固定参数，不能使用 V2 的学习输出或任何候选方法私有参数决定样本归属。

## 7. 执行顺序与门槛

### G0：协议与实现审查

- 审查可行性定义、模型假设、所有不确定性界和状态标签。
- 为标签互斥性、严格约束无 slack、无效/过期观测停机、动态/静态漂移分类和 V1 动作等价性增加单元/集成测试。
- 在 G0 通过前，不创建训练 YAML，不运行 `train.py` 或 `evaluate.py`。

### G1：无学习的覆盖与标签验证

- 在 1000 个 coverage seeds 上执行 reset 与可行性检查，不加载或训练新 actor。
- 报告初始 `h_min`、`certified_viable` 覆盖率、初始不安全率和各不可行类别。
- 对每个自然出现的状态抽样人工复核至少 20 个 trace，检查单位、连杆名称、障碍物速度和状态标签一致；自然计数为零的状态不得伪造样本，必须记录零计数并引用 G0 故障注入测试作为行为证据。
- 通过条件：无 NaN/Inf、状态标签互斥、观测无效必停机、V1 不改变 V0 命令；G1 不形成性能结论。

### G1-T：无学习的任务可行性预检查

- 在 validation、final（以及需要报告覆盖率时的 1000-reset coverage）清单上运行三层任务可行性预检查，不加载 actor，不执行策略动作。
- 固定记录 `ik`、`no_obstacle_task` 和 `dynamic_obstacle_path` 的状态、候选数量、失败原因和使用的几何/运动学边界。
- 通过条件是所有 seed 均有完整标签、输入和路径检查无 NaN/Inf、候选搜索失败被标记为 `not_found` 而不是绝对不可达；G1-T 不形成性能结论，也不授权 G3。
- 任一方法不得根据这些标签删除 final episode。条件子集只用于解释性分组，全分布结果必须保留。

### G2：V0/V1 严格执行层比较

- 使用同一组 V0 actor 和 validation/final seeds 比较 V0 与 V1。
- 通过条件：在确定性条件下两者的每步 `qdot_cmd`、终止原因和碰撞字段一致；V1 仅新增可行性标签和诊断字段。
- G2 的锁步配置必须设 `max_filter_compute_time_s: null`。该字段是 post-return 墙钟诊断，V0/V1 顺序执行时可因调度抖动只在一侧触发零速度，不能作为确定性动作等价性的输入；G1 固定 `.30 s` 的 coverage audit 仍是时间预算的唯一证据。此隔离不改变 QP、几何、速度、加速度约束，亦不启用 recovery 或 relaxation。
- 不通过时只能修复实现或协议，不得通过放宽约束、escape 或恢复动作掩盖差异。

### G3：V2 开发与 checkpoint 选择

- V2 使用五个新的 train seeds，训练执行严格过滤后的 `qdot_cmd`。
- 仅使用 40 个 validation seeds 选择固定 checkpoint 和有限候选权重；选择规则、候选集合和 tie-break 必须在运行前写入配置。
- 开发轮次只检查日志、稳定性和失败分类，不写入论文主结论。

### G4：一次性最终比较

- 对 V0 与 V2 使用相同的 5 个 train seeds、同一 200-seed final manifest 和固定 checkpoint 选择规则。
- 主结果同时报告全分布和 `certified_viable` 条件分布；两者不得混合排名。
- 最终清单运行完成后，任何参数、代码或样本修改都必须开启新协议版本和新输出根目录。

## 8. 冻结通过条件

V2 只有同时满足下列条件，才可进入“后继方案候选”评审；任一条件失败即保持未冻结。

| 维度 | 条件 |
| --- | --- |
| 安全非劣 | 以 train seed 为单位的 physical-contact rate 相对 V0 的差值，其单侧 95% bootstrap 上界不超过 `+0.5` 个百分点；全分布 pooled physical-contact rate 不得超过 `1.5%` |
| 任务改进 | mean success 相对 V0 至少提高 `5` 个百分点，且配对 train-seed bootstrap 95% CI 下界大于 `0` |
| 可行性 | `model_infeasible_dynamic + model_infeasible_static_or_slow` 的 episode rate 不高于 V0 `+1` 个百分点；不得通过 relaxed action 降低该指标 |
| 覆盖 | 1000-reset audit 的初始不安全率、可行集覆盖率和全部 reset 分布结果完整报告；不得只引用条件子集 |
| 几何与碰撞 | `collision_capsule_overlap`、`collision_pybullet_contact`、`collision_any`、`termination_collision` 和 `termination_reason` 均完整报告，并解释 capsule/physical 差异 |
| 时间 | budget-stop rate 为零；报告 mean/P95/P99/max。`P99 <= 50 ms` 且 max `<= 300 ms` 只构成仿真筛选，不构成硬实时结论 |
| 可复现 | 每个 run 保存完整 config、git revision、train/validation/final seed、checkpoint selection、逐 episode CSV 和需要时的逐步 trace |

出现一次 physical contact 不会被描述为“安全保证失败以外的可接受事件”；它必须单独审计连杆、状态标签、是否初始不安全、过滤器状态、求解时间和模型假设是否被满足。

## 9. 必须记录的字段

除现有 P3 字段外，VAPS 逐步 trace 和 episode 汇总必须记录：

```text
viability_status
viability_horizon_s
viability_min_h_m
viability_strict_feasible
viability_solver_status
viability_model_assumptions_valid
qdot_requested_radps
qdot_cmd_radps
predictive_link_velocity_norms_mps
predictive_max_link_speed_bound_mps
collision_capsule_overlap
collision_pybullet_contact
collision_any
termination_collision
termination_reason
```

若存在 safe-stop，还必须记录其首末 step、`safe_stop_h_drift_mps`、`safe_stop_drift_class`、可行性状态、活动/不可行约束类别、QP 状态和所有计时字段。episode CSV 必须保存实际 `seed` 与 `seed_manifest`；逐步速度诊断保存在 trace/progress，不能因 episode 汇总而丢失。

## 10. 明确禁止事项

1. 不沿用或重开 recovery、constraint relaxation、maximin、bounded escape 或 deterministic escape 分支。
2. 不按候选方法重新筛选初始可行 seeds，不把条件子集结果当作全 reset 分布结果。
3. 不以 capsule overlap 单独替代 physical contact，也不单独引用含义不明的 `collision` 字段。
4. 不把 post-return 的 300 ms 检查写成硬实时、可中断或部署保证。
5. 在 G4 通过前，不启动 OOD、真实机器人、主动制造碰撞或完整真机验证。

## 11. 审查结论模板

每次协议评审仅可给出以下之一：

- `approve_g0_only`：允许实现标签和测试，不允许 coverage audit、训练或评估；
- `approve_g1_g2`：允许无学习 coverage audit 与 V0/V1 严格链路验证；
- `approve_g3_g4`：允许按本协议运行 V2 训练与最终比较；
- `do_not_advance`：任一安全、覆盖、日志或可复现门槛未满足。

## 12. 审查记录

| 日期 | 决议 | 授权范围 | 未授权范围 | 决议依据 |
| --- | --- | --- | --- | --- |
| 2026-08-05 | `approve_g0_only` | 可行性标签、严格约束实现、单元/集成测试、V0/V1 动作等价性测试 | coverage audit、训练、评估、OOD、真机、recovery/relaxation | 可行性定义、固定条件和冻结门槛已明确；实现尚未验证，不能进入数据采集或方法比较 |
| 2026-08-05 | `approve_g1_g2` | 无学习的 1000-reset coverage audit，以及使用既有 V0 actor 的 V0/V1 严格链路比较 | V2 训练、checkpoint 选择、最终比较、OOD、真机、recovery/relaxation | G0 的标签、严格 fail-safe、无效/过期停机和 V0/V1 动作等价性测试已通过；G1/G2 仍不得形成 V2 性能结论 |
| 2026-08-05 | `approve_g1_g2` | 继续执行使用既有 V0 actor 的 V0/V1 严格链路比较 | V2 训练、checkpoint 选择、最终比较、OOD、真机、recovery/relaxation | G1 的 1000-reset audit 完整覆盖固定清单；标签计数完整、无 NaN/Inf、unknown/invalid/budget-stop 为零，P99/max 为 9.86/214.95 ms，且 40 个确定性 reset trace 的结构复核与 G0 零计数类别证据均已完成 |
| 2026-08-05 | `do_not_advance` | 保留 G2 v1 失败 trace；不得把其余通过文件合并为 G2 通过 | G3/G4、V2 训练、checkpoint 选择、最终比较、OOD、真机、recovery/relaxation | v1 在 train seed `4108`、validation seed `8216`、step `129` 出现单侧 `safe_stop_compute_budget`，使 `qdot_cmd` 和后续观测分叉；这证明 post-return 墙钟预算不是锁步确定性输入 |
| 2026-08-05 | `approve_g1_g2` | 仅运行新协议 `vaps_g2_strict_execution_v2` 的既有 V0 actor 锁步验证；G2 v2 禁用 post-return 计时回退并额外比较 filter status | G3/G4、V2 训练、checkpoint 选择、最终比较、OOD、真机、recovery/relaxation | 变更只隔离已由 G1 审计的非确定性计时诊断，不改变严格约束或 fail-safe；失败 seed `8216` 的 v2 单 seed 复测四类数值差异均为零 |
| 2026-08-05 | `do_not_advance` | 仅重放并审计 G2 v2 trace 中冻结的 11 次 `safe_stop_projection_failed` | G3/G4、V2 训练、checkpoint 选择、最终比较、OOD、真机、recovery/relaxation | G2 v2 的 V0/V1 执行等价性通过，但 11 次双方同步投影失败尚未完成残差、约束和零命令 fail-safe 审计；不得将同步性等同于严格可行性 |
| 2026-08-05 | `do_not_advance` | 对已冻结的 11 次投影失败执行原矩阵、无时限的离线 OSQP 迭代上限归因（`2000/10000/50000`）及独立残差复核 | G3/G4、V2 训练、checkpoint 选择、最终比较、OOD、真机、recovery/relaxation，以及运行时控制改动 | 定点回放已确认 11/11 为零命令 fail-safe，但零命令均不满足瞬时严格约束；必须区分严格问题实际不可行与当前 OSQP 迭代上限不足，且该数值诊断不构成阶段授权 |
| 2026-08-05 | `do_not_advance` | 仅对 10 个经 50000 次 OSQP 迭代判为 primal infeasible 的冻结事件进行离线约束冲突归因 | G3/G4、V2 训练、checkpoint 选择、最终比较、OOD、真机、recovery/relaxation、运行时迭代上限调整及环境命令改动 | 数值归因完整覆盖 11 个冻结事件：`2000` 次均未定，`10000` 次为 9 个不可行/2 个未定，`50000` 次为 10 个严格不可行/1 个严格可行；唯一可行事件 `(4110,9119,155)` 在 34625 次迭代获得最大原约束残差 `1.60e-08`。因此运行时投影失败不能全归咎于迭代上限，且提高上限不能解除多数严格约束冲突或授权 G3 |
| 2026-08-05 | `do_not_advance` | 实现离线三层任务可行性预检查标签，不运行训练、策略评估或运行时控制改动 | G3/G4、V2 训练、checkpoint 选择、最终比较、OOD、真机、recovery/relaxation、按标签删样本 | 当前随机目标仅按笛卡尔工作空间采样，尚未记录 IK、无障碍候选路径和给定动态障碍轨迹下的候选路径可行性；必须先将目标/场景问题与局部严格控制不可行区分 |
| 2026-08-05 | `do_not_advance` | 将三层任务可行性标签与既有 G2 v2 trace 做离线解释性对照 | G3/G4、V2 训练、checkpoint 选择、最终比较、OOD、真机、recovery/relaxation、按标签删样本 | final 交叉分组显示“动态候选路径未找到”组的 safe-stop 为 `97/120=80.8%`，而三层候选均找到组为 `227/447=50.8%`；但 G2 trace 没有 `success` 字段，不能据此声称策略到达率或推进训练 |
| 2026-08-05 | `do_not_advance` | 对三个已冻结 V0 actor 在 G2 final seed 上执行无障碍到达能力诊断 | G3/G4、V2 训练、checkpoint 选择、动态障碍最终比较、OOD、真机、recovery/relaxation、按标签删样本 | 用户要求补齐既有策略的真实无障碍 `success` 证据；诊断固定关闭障碍物和安全过滤器，保留原目标/初始状态及 200-seed 清单，只用于区分基础到达能力与动态避障问题 |
| 2026-08-05 | `do_not_advance` | 归档冻结 B4 actor 的无障碍实际执行结果 | G3/G4、V2 训练、checkpoint 选择、动态避障主比较、OOD、真机、recovery/relaxation | 三个 actor 在 600 个无障碍 episode 中仅 `31/600=5.17%` success，`569/600` 超时未到达，且无 physical contact；基础 reaching 未通过，不能在其上评估或宣称安全协同收益 |
