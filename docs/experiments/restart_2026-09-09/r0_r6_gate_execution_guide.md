# R0–R6 Gate 实验执行指南

> 整理日期：2026-09-10  
> 适用实验：`restart-2026-09-09-v1` 及其明确登记的后续修订  
> 定位：本文件说明阶段顺序、检查项、证据和失败分流，不替代已冻结的数值协议。

## 1. 文档关系与使用原则

本指南依据当前论文主线整理：

```text
名义 SAC actor
→ 动作条件全身间隙预测
→ 轨迹一致 One-Step / Predictive QP
→ 全身非线性验收与反例增广
→ 逐子步监测和已验收 fallback
```

发生冲突时，文档优先级如下：

1. [论文大纲](../../thesis/thesis_outline.md)：研究问题、方法边界和论文结构；
2. [实验跟踪](experiment_tracker.md)：当前阶段、冻结协议、任务状态和硬门禁；
3. [结果事实源](experiment_results.md)：允许引用的实验事实和失败结果；
4. 本指南：如何执行和审计每个 Gate。

基本原则：

- 每一阶段只引入一个待验证的新因素；前一硬门禁未通过，不进入依赖它的下游阶段。
- 开发训练 seed、validation seed 和最终 held-out seed 严格隔离。开发 Gate 不得使用最终 held-out 数据。
- 未通过的运行和负结果必须保留，不得降低既有门槛后把同一运行改写为通过。
- 新发现的设计缺陷可以修复，但必须建立新的阶段编号、在运行前写清单因素变化和选择规则。
- R2 actor 冻结后，R3–R8 原则上不再训练或微调 actor，以维持方法比较的因果可解释性。
- 训练内 episode 指标只用于诊断，checkpoint 只能按预注册 validation 规则选择。

### 1.1 R2 顺序补正的生效边界

本指南形成于 R2a--R2d 已经完成之后，不能把后来定义的子 Gate 追溯性写成当时已经执行。统一口径如下：

```text
R2a--R2d：指南形成前的历史诊断链，只保留失败与修复证据
R2e：补做 R2-A 无障碍基础容量
R2-B：只做当前 0.055 m 终止定义下的稳定性诊断
R2-C：R2e 通过后新预注册的风险课程或名义 actor 职责验证
R2-D：最终路线的三个正式 seeds 与 actor 冻结
```

因此，“R2a--R2d 已完成”不等于“R2-A--R2-D 已完成”。R3 仍由补正后的 R2-D Gate 阻塞。

## 2. 原指南中需要修正的三点

### 2.1 不把示例精度直接写成硬门禁

“5 cm 成功率 100%，再依次通过 3/2/1 cm”适合作为能力分层建议，但不是当前协议。当前环境的冻结成功阈值是 `0.055 m`，R2 的硬门禁是：

```text
无障碍 Success Rate >= 0.80
五个障碍场景宏平均 Success Rate >= 0.60
```

在当前版本中，3/2/1 cm 只能作为后续新协议候选，不能从进入 5.5 cm 即终止的现有 rollout 推断，也不能看过结果后追加为 Gate。若论文确实需要厘米级定位，应先说明其研究必要性，再建立新版本并预注册目标集合、终止定义、阈值和容忍率。

### 2.2 开发 Gate 使用 validation，不使用最终 held-out

指南中的 “IK-reachable held-out goal set” 应改称固定 validation goal set。最终 held-out seeds `{2001,2002,2003}` 只在方法、actor 和全部参数冻结后使用一次，不能参与 R2–R6 的开发决策。

### 2.3 不得事后筛掉当前 validation 目标

对训练/测试目标先做 IK 可达性筛选是合理的新任务设计，但当前 v1 已冻结笛卡尔工作空间均匀采样。不能根据 R2 失败事后删除困难目标。若采用 IK-filtered goal pool，必须建立 v2 协议，并冻结：

- IK 是否约束末端姿态；
- joint limits、self-collision 和安全裕量；
- IK 初值、迭代上限与残差阈值；
- 训练、validation、held-out 三个互不重叠的固定目标池；
- 拒绝采样比例及其对任务分布的影响。

当前对 90 个无障碍 validation 目标的 IK 检查只属于诊断证据，不改变原分母。

## 3. 阶段总览

| 阶段 | 只验证什么 | SAC 训练 | 主要输出 | 失败后的动作 |
| --- | --- | --- | --- | --- |
| R0 | 协议、版本、seed 和事实源可追溯 | 否 | 冻结配置与 manifest 规则 | 禁止正式实验 |
| R1 | 测量、执行语义和几何近似可信 | 否 | 一致性测试、几何校准与裕量 | 修环境，不训练 actor |
| R2-A | SAC 是否能完成基础无障碍 reaching | 是 | 基础容量诊断 | 修 SAC/任务 MDP |
| R2-B | 基础定位精度与稳定性 | 可选 | 分阈值精度曲线 | 不盲目加障碍 |
| R2-C | 风险耦合训练或名义 actor 职责路线 | 是 | 成功/碰撞/超时/平滑性 | 修 curriculum 或职责划分 |
| R2-D | 多 seed actor 选择和冻结 | 是，随后冻结 | actor/checkpoint 哈希 | 不进入 R3 |
| R3 | 动作条件预测本身是否可信 | 否 | 预测误差、coverage、LeadTime | 禁止把预测接入 QP 主张 |
| R4 | One-Step-QP 和运动边界是否正确 | 否 | 求解、轨迹一致和安全基线 | 禁止进入多时刻 QP |
| R5 | 固定 Top-k 多时刻约束的净作用和漏检 | 否 | PTQP 对照及 nonlinear rejection | 不声称细化必要性 |
| R6 | 全身验收、反例增广与 fallback 是否有效 | 否 | 完整 VG-PTQP validation | 不进入最终 held-out |

## 4. R0：协议和可复现基座

### 必须冻结

- UR5 URDF、基座、受控关节、工具坐标系和胶囊定义；
- 240 Hz 物理频率、20 Hz 策略频率及每周期 12 个物理子步；
- `0.7 rad/s` 动作范围和 `0.1 rad/s` 相邻策略速度变化上限；
- Butterworth `omega_c=30 rad/s` 与 quintic RTB；
- goal/obstacle 分布、episode 上限和 success/collision/truncation 定义；
- reward、cost、observation schema 和所有训练超参数；
- 开发、validation、held-out seeds 与 `episode_seed` 派生规则；
- 指标字段、聚合分母、checkpoint 选择规则和输出目录。

### 证据

- resolved config、源码/URDF/胶囊/checkpoint SHA-256；
- Python、PyBullet、Torch、NumPy、SciPy、OSQP 和硬件版本；
- dirty worktree 的完整源码快照哈希；
- 训练入口拒绝覆盖非空 run 目录。

### Gate

配置可复现、数据划分无交叉、输出事实源唯一。缺少任一项时不启动正式长训练。

## 5. R1：环境、测量、几何和执行接口

R1 不训练 SAC。

### 5.1 FK/IK 与目标审计

- 对正式目标分布报告 IK 残差分布和不可达比例；
- IK 结果重新做 FK，不能只相信 solver status；
- 明确是否约束姿态和 joint limits；
- 不从当前 validation 分母中事后删除失败目标。

### 5.2 逐子步测量

- 每次 `stepSimulation` 后读取关节反馈、胶囊距离和 PyBullet contact；
- 碰撞事实只来自 PyBullet contact；胶囊间隙 `<=0` 只属于近似几何违反；
- 分开记录 240 Hz 命令与反馈的速度、acceleration 和 jerk；
- transition cost 使用 12 个子步中的最大风险、最小距离和任一 contact。

### 5.3 执行链一致性

- rate limiter 和有状态 Butterworth 每个策略周期只更新一次；
- quintic 数学序列与实际下发序列逐项一致；
- 跨周期分别继承命令侧和反馈侧 `a_prev`；
- 位置、速度、acceleration 和 jerk 的单位与采样频率明确。

名义执行测量已经通过；QP 内轨迹与实际下发的一致性必须在 R4 实现后再次通过，不能把尚未实现的 QP 检查写成 R1 已完成。

### 5.4 胶囊几何

- 按连杆、关节构型和近表面间隙分层采样；
- 对比胶囊距离与 PyBullet collision shape/contact；
- 报告单侧误差和危险漏检，而不仅是平均误差；
- 使用独立校准集冻结 `m_geom`。

当前证据为 `m_geom=0.04 m` 后危险漏检 `0/1597`；这只校准模型裕量，不等于形式化碰撞保证。

## 6. R2：训练和冻结名义 actor

R2 是主流程中集中训练 SAC 的阶段。建议内部采用 A–D 分解，但每个子阶段仍需单独登记。

### 6.1 R2-A：无障碍 reaching 容量

目的：回答当前 SAC、reward、RTB 和动作限制能否完成最基础的静态目标到达。

执行要求：

- 无 QP、无障碍、相同 `link_risk_v1` 维度和执行管线；
- deterministic actor 在固定 validation seeds 上评估；
- 报告 Success Rate、Final Position Error、Timeout Rate、Completion Time；
- 同时用 IK 或解析控制器确认环境任务可解，但这些控制器结果不计作 SAC 成绩。

当前待执行的 R2e 正是该容量诊断。它不能直接冻结为正式 actor。

失败分流：

```text
R2-A 未通过
├─ 目标/执行器也不可解 → 修环境或建立新任务协议
└─ 解析控制器可解但 SAC 失败 → 审查 reward、SAC 稳定性、归一化和训练设计
```

### 6.2 R2-B：定位精度分层

当前 v1 episode 一进入 `5.5 cm` 就终止，因此不能用同一 rollout 声称已经测试 5/3/2/1 cm。R2-B 在 v1 中只报告 `5.5 cm` Gate、Final Position Error、超时轨迹的最近/最终误差和阈值附近稳定性。只有论文目标明确要求更高精度时，才建立新的 success/termination 配置、重新预注册并从头评估；更小阈值不能从现有终止轨迹事后推断。

需要区分：

- 从未接近目标；
- 接近但在阈值外振荡；
- 一度进入阈值但执行/终止逻辑未正确识别；
- 关节或 rate limit 导致剩余时间不足。

### 6.3 R2-C：有障碍 current-risk 名义策略

只有基础 reaching 通过后才进入。

可采用预注册 curriculum：

1. 障碍存在但远离 nominal path；
2. 静态或极低速轻干扰；
3. 低速动态横穿；
4. 正式 random crossing 分布；
5. 指定连杆和快速接近只作为后期少量训练或 validation stress。

课程的 level、切换 step、采样比例和 replay 行为必须在训练前冻结。不得依据同一 validation 曲线临时切换。困难 stress 场景不应高比例混入训练，否则会把名义 actor 训练成停滞策略。

必须联合报告：

- Success Rate、Collision Rate、Timeout Rate；
- Final Position Error、Completion Time；
- Safety Violation Rate、Minimum Distance、current-risk 响应；
- 命令/反馈平滑性和 rate-limit 触发率；
- 按 obstacle-enabled 与场景拆分的训练诊断。

### 6.4 terminal shortcut Gate

碰撞立即终止时，必须比较 success、collision、timeout 三类 episode 的长度和回报。若提前碰撞能规避后续逐步负回报，R2 直接失败，不能依靠下游 QP 掩盖。

R2b 已发现该问题；R2c 将 `collision_penalty` 从 2 修正为 100 后，碰撞回报不再优于 timeout。该修复属于任务 MDP 正确性，不是论文创新。

### 6.5 名义 actor 职责复审

论文核心创新是独立的预测安全 QP，而不是风险 SAC。如果 R2-A 通过、但 current-risk 多任务训练持续破坏 reaching，应在新协议中明确选择以下一种路线：

- 保留 Instant-Link-SAC，但使用预注册 curriculum/分层 replay；
- 使用只负责任务的 SAC nominal actor，把障碍安全完全交给独立 QP；
- 将 current-risk SAC 保留为算法基线，而非主方法的必要前置。

这属于论文设计修改，必须同步修改论文大纲、R2 Gate 和全部方法名称，不能静默替换 actor。

### 6.6 R2-D：多 seed 冻结

- 开发 seed 只选择训练设计，不进入正式主比较；
- 设计冻结后从头训练 seeds `{101,202,303}`；
- 每个 seed 独立按同一 validation 规则选 checkpoint；
- 检查训练稳定性、terminal shortcut 和异常 seed；
- 写入 actor/checkpoint SHA-256；
- 从 R3 开始禁止更新 actor。

## 7. R3：动作条件预测

R3 固定 actor、不启用 QP，只验证预测。

### 真值协议

- 每周期只计算一次并缓存名义 endpoint；
- 从相同机器人/障碍物状态复制仿真；
- 对预测和反事实真值输入同一 endpoint；
- 执行同一周期 quintic 子步，周期外按预注册规则保持 endpoint；
- 不能让后续 SAC 重规划混入开环预测真值。

### 指标

- 逐连杆、逐时刻 future distance MAE/尾部误差；
- future minimum-gap error；
- `T_enter` 检测率、误报率和时间误差；
- 最危险连杆/临界时刻识别；
- 事件条件 LeadTime 和 coverage；
- 距离—endpoint 梯度的中心差分方向导数误差；
- 几何、跟踪、感知、速度、线性化和时间网格裕量。

### Gate

动作条件预测必须相对“当前距离”和“当前速度趋势”至少改善一个预注册主指标，同时满足误报、梯度、coverage 和耗时门槛。否则不得声称预测有效，也不进入预测 QP 主张。

## 8. R4：One-Step-QP

R4 只建立最简单的当前几何 QP，用于隔离 solver 和运动边界作用。

必须验证：

- 距离梯度符号与有限差分一致；
- 危险动作经 QP 后当前间隙方向总体改善；
- QP endpoint 对应的 quintic 子步就是实际下发序列；
- 所有子步满足位置、速度、acceleration 和跨周期 jerk 边界；
- OSQP status、iterations、primal/dual residual 和 slack 语义正确；
- 可行、不可行、NaN、超时和 fallback 分支都有测试；
- 报告端到端 mean/p95/p99/max，而非只报告 solver 平均时间。

Gate：逐子步安全不得劣于无安全层，任务损失、修正量和实时性在预注册容忍范围内，且 QP/实际执行轨迹一致性通过。

## 9. R5：固定 Top-k Predictive-Trajectory-QP

R5 在与 R4 相同的 actor、solver、运动边界、slack 和 episodes 上，只增加动作条件多时刻约束。

初始约束包含：

- 当前近阈值集合；
- 名义预测 Top-k 连杆的临界时刻；
- 各连杆首次越界时刻及预注册相邻时间索引。

全身非线性回放在 R5 中只作为诊断器，不把反例反馈回 QP。必须记录：

```text
ModelCheckRejectionRate
= 固定 Top-k QP 判定可执行、但全连杆 × 全验证时刻回放不通过的比例
```

还需记录遗漏的连杆/时刻、线性化残差、修正量和 fallback。若多时刻预测相对 R4 没有预注册净收益，预测约束不能进入核心正结论；若 rejection 接近零，也必须重新评估 R6 反例细化的必要性。

## 10. R6：VG-Predictive-Trajectory-QP

R6 在固定 Top-k 初始集合上增加：

```text
稀疏 QP
→ 候选 endpoint
→ 全连杆 × 全验证时刻 nonlinear replay
→ 最严重反例批量增广
→ 新参考点重新线性化并求解
→ 通过、预算终止或 fallback
```

候选在以下任一条件发生时不得正常执行：

- 达到 `R_max` 仍未通过；
- deadline 不足；
- solver/slack 状态不合格；
- NaN、无效梯度或迭代停滞；
- nonlinear verification 失败。

优先切换到预构造且通过同一全身验收的 jerk-limited braking fallback；fallback 也未通过时记录 `fallback_unverified` 并触发急停语义。

### 指标组

1. 任务：Success Rate、Goal Error、Completion Time；
2. 安全：Collision Rate、Minimum Clearance、Violation Rate、LeadTime；
3. 干预：Intervention Rate、`||u_safe-u_nom||_2`、停止比例；
4. 验证：ModelCheckRejectionRate、增广轮数、反例数；
5. 恢复：

```text
RefinementRecoveryRate
= 首轮 nonlinear verification 拒绝后，经反例增广最终通过的比例
```

6. 求解与降级：solver failure、slack、fallback、unverified fallback；
7. 实时性：prediction、gradient、QP、verification、fallback-check 和 controller 总时间的 mean/p95/p99/max，以及 deadline 命中率。

Gate：相对 R5 降低最终模型内漏检，并在最终安全或同等安全下体现修正/fallback 的净收益；不能只靠更多停止获得通过。

## 11. R6 之后的 held-out 交接

R0–R6 全部通过后才进入 R7：

- 冻结 actor、预测、QP、裕量、Top-k、验证网格、细化和 deadline 参数；
- 锁定全部配置与哈希；
- 一次性运行 held-out seeds `{2001,2002,2003}`；
- 至少比较 Instant-Link-SAC+RTB、Reactive-Projection、One-Step-QP、固定 Top-k PTQP 和 VG-PTQP；
- 使用相同 actor、RTB 和 episode seeds 做配对反事实；
- held-out 后不调参，修改必须新建实验版本。

## 12. 当前项目位置

截至 2026-09-10：

- R0 大部分协议与可复现基座已建立；dirty 源码通过逐 run 哈希保存；
- R1 名义测量和几何 Gate 已通过，QP 专属轨迹一致性留待 R4；
- R2a：固定风险 penalty 网格失败；
- R2b：补充 20% 无障碍训练覆盖，仍失败；
- R2c：修复 collision terminal shortcut，碰撞显著下降但任务 Gate 失败；
- R2d：扩展到 150k 后策略波动，证明不能继续以“训练不够久”解释；
- R2e：无障碍 SAC 容量诊断已预注册，待训练与 validation；
- R3–R6：仍被 R2 Gate 阻塞，没有可写入论文的性能结论。

当前决策树：

```text
R2e 无障碍容量诊断
├─ 未达到 0.80
│  └─ 审查 SAC/reward/normalization；不得增加障碍或进入 R3
└─ 达到 0.80
   └─ current-risk 多任务耦合为主要嫌疑
      ├─ 新协议：预注册 curriculum / replay 设计
      └─ 或修改论文：任务 SAC nominal actor + 独立预测安全 QP
```

无论选择哪条路线，都必须先更新论文大纲与 tracker，再启动新的正式 actor 训练。
