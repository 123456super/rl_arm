# 论文实验执行与进展跟踪

> **归档状态：** 本文档于 2026-09-09 封存，其中状态和通过判定不得用于新实验或论文结论。
> 初始计划日期：2026-09-08；方法路线修订：2026-09-09；最近整理：2026-09-09  
> 本文档是封存前 P0--P8 的执行状态快照，不再更新。  
> 封存时对应的研究方案现见 [论文大纲](../../thesis/thesis_outline.md)。  
> 已确认的实验数据、统计口径和可引用结果见 [实验结果事实源](thesis_experiment_results.md)。

## 1. 维护规则

- `thesis_outline.md` 回答“研究什么、为什么、方法和实验如何定义”。
- 本文档回答“做到哪里、下一步做什么、何时算通过”。
- `thesis_experiment_results.md` 回答“实际得到什么数据、哪些结论可以引用”。
- 本文档不复制详细公式和结果表；需要解释方法或引用数值时链接到相应事实源。
- 每一步只验证一个关键环节。结果异常时停在当前步骤定位，不把训练、预测、QP 和实机验证混在同一轮实验中。
- validation 只用于选择冻结参数；held-out 测试不得参与方法选择。
- 所有新比较使用无重复 `episode_seed`，并由 manifest 检查文件数、行数、实验单元和配对关系。

每次状态更新至少记录：

```text
计划编号:
完成日期:
代码版本:
配置与 checkpoint:
输出目录:
train/eval/episode seeds:
状态: todo | doing | blocked | passed | failed-result
是否通过预注册门控:
主要异常:
下一步:
```

## 2. 总状态

| 编号 | 状态 | 当前结论 | 下一门槛 |
| --- | --- | --- | --- |
| P0 | passed | 独立 episode seed 的三方法正式基线已冻结 | 只作为已有方法背景，不重复调参 |
| P1 | passed | action-independent 预测风险模块与离线预警已验证，但误报偏高 | 动作条件预测由 P3 单独验证 |
| P2 | failed-result | prediction observation 与 reward shaping 未获得任务/安全共同收益 | 作为负消融冻结，不继续扩大训练 |
| P3 | todo | 动作条件间隙、几何标定、逐子步监测和距离梯度尚未完成 | 通过 P3.0/P3.1 后才能进入最终 QP |
| P4 | passed | Reactive-Projection 工程基线已完成固定 actor 反事实 | 不解释为标准 QP 证据 |
| P5 | partial | endpoint/quintic 运动边界过渡消融完成；One-Step-QP 未完成 | 完成 P5.2 标准 OSQP 基线 |
| P6 | todo | VG-PTQP 尚未实现和验证 | 完成约束生成循环与五组主比较 |
| P7 | todo | 鲁棒性与泛化尚未完成 | P6 参数冻结后逐项运行 |
| P8 | conditional | 真实 UR5 验证尚未开始 | 以 P6 通过、硬件接口和现场签核为前提 |

## 3. 当前阻塞与近期任务

当前阻塞：

- 现有安全结果只在 20 Hz 控制周期末采样，可能漏掉 240 Hz 子步间的最小距离和接触。
- 历史 `minimal_qp` 是有限轮半空间投影器；`safety_qp_infeasible` 和 `active_constraints` 字段语义不适用于标准 QP。
- P5 只验证命令轨迹峰值，没有反馈侧运动导数或逐子步安全证据。
- 动作条件预测、时变裕量、标准 OSQP、全身非线性验收、反例增广和可验收制动 fallback 尚未实现。
- 既有 actor 在 EMA 下训练，安全层主比较统一使用 RTB；只能依靠同 actor、同 RTB、同 episode 的配对反事实识别安全层作用。

按顺序执行：

1. 完成 P3.0：逐物理子步测量、字段语义修复和几何模型标定。
2. 完成 P3.1：动作条件离散 quintic rollout、开环反事实真值和距离梯度检查。
3. 完成 P5.2：接入 OSQP，建立轨迹一致 One-Step-QP。
4. 完成 P6.0：固定 Top-k PTQP、全身验收、反例增广、deadline 和制动 fallback。
5. 在 validation 冻结参数后运行 P6.1 held-out 主比较，再进入 P7；P8 保持条件性。

## 4. 分阶段执行清单

### P0：冻结当前基线状态

状态：`passed`

- [x] 修复 episode seed 重叠并完成正式三方法复评估。
- [x] 冻结 `link_fixed_penalty1` 作为 `Instant-Link-SAC` 名义策略起点。
- [x] 将旧 seed 重叠表、adaptive beta 和 LDRC 结果限制为历史背景或次要比较。
- [x] 在结果事实源中记录完整性检查、统计单位和结论边界。

验收：正式基线文件、行数和实验单元无重复；后续新方法不借用旧表证明预测或 QP 收益。

事实与输出：见 [结果事实源第 1--5 节](thesis_experiment_results.md#1-统计口径数据审计与使用状态)。

### P1：离线验证已有预测风险

状态：`passed`

- [x] 实现 `d_pred`、`T_enter`、逐连杆预测风险和危险连杆输出。
- [x] 覆盖静止、接近、远离和相同距离不同速度的单元场景。
- [x] 在固定 actor 轨迹上评估预警提前量、未来最小距离、危险连杆、误报和漏报。

验收：模块输出有限值；接近场景具有正预警提前量；误报问题如实进入后续动作条件对照。

主要资产：

```text
src/rl_risk_sac/utils/predictive_risk.py
tests/test_predictive_risk.py
scripts/evaluate_predictive_risk_offline.py
outputs/archive_pre_restart_2026-09-09/current_results/predictive_risk_offline_eval/
```

### P2：预测风险进入策略的失败消融

状态：`failed-result`，已冻结

- [x] 增加隔离的 `link_risk_pred_v1` observation schema。
- [x] 完成短训练、100k probe 和 checkpoint validation。
- [x] 完成 raw predicted-risk reward shaping validation。
- [x] 判定 observation/reward 路线未获得稳定净收益，停止三 seed 扩张和 excess shaping 运行。

验收：负结果可复现，旧 `link_risk_v1` 与旧 checkpoint 不受 schema 影响；不得把该路线改写为正贡献。

配置与事实：

```text
configs/experiments/p2_predictive_checkpoint_validation.yaml
configs/experiments/p2_predictive_100k_checkpoint_validation.yaml
configs/experiments/p2_predictive_reward_shaping_validation.yaml
outputs/archive_pre_restart_2026-09-09/current_results/p2_predictive_checkpoint_validation/summary/
outputs/archive_pre_restart_2026-09-09/current_results/p2_predictive_100k_checkpoint_validation/summary/
outputs/archive_pre_restart_2026-09-09/current_results/p2_predictive_reward_shaping_validation/summary/
```

详细数值见 [结果事实源第 9--11 节](thesis_experiment_results.md#9-p22-predictive-link-sac-checkpoint-validation)。

### P3：动作条件预测与测量基础

状态：`todo`

#### P3.0 测量、几何与求解语义

- [ ] 每个 240 Hz 物理子步后读取关节位置/速度、胶囊距离和 PyBullet contact。
- [ ] 分开记录 `command_substep_*` 与 `measured_substep_*` acceleration/jerk。
- [ ] 将历史“QP infeasible”改为投影残差失败，修正 active-constraint 统计。
- [ ] 显式固定 OSQP 依赖和 solver status、iterations、primal/dual residual、slack、solve time、cycle overrun 字段。
- [ ] 对工作空间分层采样，以 PyBullet collision shape/contact 标定每根胶囊的距离误差、阈值混淆矩阵和危险漏检率。
- [ ] 仅在 validation 上冻结 `m_geom`；无法由裕量覆盖的系统性漏检必须通过修改胶囊定义解决。

门控：全部 P6 变体可以使用同一逐子步口径；旧 CSV 保留原语义，不回写历史数据。

#### P3.1 动作条件预测与梯度

- [ ] 每周期只调用一次 rate limiter/Butterworth，并缓存同一名义 endpoint。
- [ ] 使用与实际下发一致的离散 quintic 子步模型；周期外采用 endpoint 保持并在下一控制周期滚动重规划。
- [ ] 提供无副作用的运动学查询，禁止 `resetJointState` 后遗漏状态恢复。
- [ ] 对全部建模连杆 rollout，仅对活跃 `(link,time)` 计算梯度。
- [ ] 用中心差分方向导数检查距离—endpoint 梯度和投影区段切换。
- [ ] 在信赖域内统计线性化残差，分解并冻结几何、跟踪、位置/速度估计、线性化和网格裕量。
- [ ] 建立开环反事实真值：同一初始状态、同一缓存 endpoint、同一执行轨迹；闭环只将一周期误差作为部署指标。
- [ ] 分别报告事件条件 LeadTime、warning coverage、误报率、未来最小距离误差和危险连杆识别率。
- [ ] 报告完整 prediction time，并满足 20 Hz 预算。

门控：相对当前速度趋势预测，未来最小距离或危险连杆识别至少一项稳定改善；梯度误差、误报和耗时通过预注册阈值。

### P4：Reactive-Projection 工程基线

状态：`passed`

- [x] 验证距离约束方向、关节速度/位置边界和危险动作修正方向。
- [x] 将安全修正放在 RTB endpoint 前，避免直接修改子步造成 jerk 尖峰。
- [x] 完成三 actor、五场景、独立 held-out seed 的固定 actor 配对反事实。
- [x] 记录介入、修正量、投影残差和计算时间。

边界：论文名称固定为 `Reactive-Projection`；历史配置名 `minimal_qp` 不表示标准 QP，残差失败也不表示数学不可行。

资产与结果：

```text
src/rl_risk_sac/control/safety_qp.py
tests/test_safety_qp.py
configs/experiments/minimal_qp_heldout_counterfactual.yaml
outputs/archive_pre_restart_2026-09-09/current_results/minimal_qp_heldout_counterfactual_independent_seeds/
```

详细数值见 [结果事实源第 6 节](thesis_experiment_results.md#6-minimal-qp-独立-seed-五场景反事实结果)。

### P5：轨迹一致运动边界与 One-Step-QP

状态：`partial`

#### P5.1 endpoint/quintic 过渡消融

- [x] 完成无运动边界、acceleration-only、acceleration+jerk 的 validation 和 held-out。
- [x] 冻结进入下一阶段的命令边界参数。
- [x] 将证据限制为“下发命令峰值受控”，不外推为反馈边界或逐子步安全。

资产与结果：

```text
configs/experiments/p5_motion_bounds_validation.yaml
configs/experiments/p5_motion_bounds_heldout.yaml
outputs/archive_pre_restart_2026-09-09/current_results/p5_motion_bounds_validation/summary/
outputs/archive_pre_restart_2026-09-09/current_results/p5_motion_bounds_heldout/summary/
```

详细数值见 [结果事实源第 7--8 节](thesis_experiment_results.md#7-endpointquintic-运动边界-validation过渡消融)。

#### P5.2 轨迹一致 One-Step-QP

- [ ] 使用 OSQP 优化单一安全 endpoint。
- [ ] 对本周期全部实际下发子步施加位置、速度、acceleration 和跨周期 jerk 硬边界。
- [ ] 使用冻结的当前几何构造 One-Step 安全基线，不引入动作条件未来几何。
- [ ] 保证优化器子步与实际下发子步逐项一致，QP 后不再滤波。
- [ ] 使用有界独立 safety slack；运动边界保持硬约束。
- [ ] 分别验证命令和反馈边界，并记录 solver 状态、残差、slack 与端到端耗时。

门控：命令边界在数值容差内全部满足；solver 语义正确；端到端耗时低于控制周期；逐子步安全指标不劣于 endpoint-only 过渡版本。

### P6：VG-Predictive-Trajectory-QP 主实验

状态：`todo`

#### P6.0 约束生成、验收与降级循环

- [ ] 初始约束集使用 current-near、名义 Top-k、首次越界及冻结的相邻时刻规则。
- [ ] 每轮在新参考 endpoint 处重新计算累计约束的距离和梯度。
- [ ] 每个候选解在全部建模连杆、全部验证时刻上执行非线性胶囊回放。
- [ ] 每轮加入至多 `B_add` 个最严重新反例；已有反例持续违反时强制重新线性化。
- [ ] 实现 `R_max`、deadline、solver failure、slack、NaN、停滞和无有效梯度的独立降级原因。
- [ ] 在细化前构造满足运动硬边界的 jerk-limited braking trajectory，并对完整制动时域预验收；只执行其当前周期片段并保存后续制动计划。
- [ ] 制动 fallback 未通过时触发硬件级急停并记录 `fallback_unverified`，不宣称停止必然避碰。
- [ ] 为网格间隔提供可信 Lipschitz 裕量；否则结论明确限制为离散验证时刻。
- [ ] 日志能够重建每轮参考点、约束集、候选解、反例、验收结果和最终执行来源。

最低单元场景：一轮通过、危险连杆迁移、同索引重线性化、达到最大轮数、deadline、solver failure、高 slack、NaN、梯度无效、fallback 通过和 fallback 未通过。

#### P6.1 validation 与 held-out 主比较

最低五组：

```text
Instant-Link-SAC + RTB
Instant-Link-SAC + Reactive-Projection
Instant-Link-SAC + One-Step-QP
Predictive-Trajectory-QP（固定 Top-k）
VG-Predictive-Trajectory-QP
```

- [ ] 三个 QP 变体使用相同 actor、RTB、OSQP、运动边界、slack 和 episode seeds。
- [ ] One-Step-QP 与固定 Top-k PTQP 只改变是否采用动作条件多时刻几何。
- [ ] 固定 Top-k PTQP 与 VG-PTQP 只改变是否把全身验收反例反馈到下一轮。
- [ ] 固定 Top-k 候选也运行同一验收器，但不反馈反例，用于测量 model-check rejection。
- [ ] validation 冻结时域、验证网格、Top-k、`B_add`、`R_max`、deadline、裕量、信赖域和 QP 权重后，再运行独立 held-out seeds。
- [ ] 报告任务、安全、干预、命令/反馈平滑性、约束细化、fallback 和端到端实时性。

门控：

- 固定 Top-k PTQP 相对 One-Step-QP 改善至少一个预注册预测安全主指标，且任务/干预代价在容忍范围内。
- VG-PTQP 相对固定 Top-k 降低最终模型内漏检，并在最终安全违反或相同安全水平下的 fallback/修正代价上体现净收益。
- 只靠增加停止或 `fallback_unverified` 消除正常动作漏检不算通过。
- 命令运动边界满足配置，反馈误差与违反单独报告；端到端安全层时间满足 50 ms 周期并保留通信余量。

### P7：鲁棒性与泛化

状态：`todo`，依赖 P6 参数冻结

- [ ] 位置噪声。
- [ ] 速度尺度/方向误差。
- [ ] 控制与观测延迟。
- [ ] 障碍物速度或方向突变。
- [ ] 略超训练分布的速度。
- [ ] 双球形障碍物附加泛化。

门控：一次只改变一个因素，报告性能下降曲线，并区分预测模型失配、QP 可行域、网格覆盖和 fallback 失败。

### P8：真实 UR5 低速验证

状态：`conditional`

#### P8.1 Proposed 离线预检

- [ ] 扩展 `scripts/deployment_preflight.py`，解除旧基线硬绑定。
- [ ] 使用 P6 冻结配置和同一 actor/checkpoint 生成新的 Proposed 报告。
- [ ] 检查 observation、预测、QP、运动边界、逐子步监测、状态时效和全部降级路径。

#### P8.2 现场安全签核

- [ ] 在真实控制器上强制更保守的速度、加速度、jerk 和工作空间限制。
- [ ] 实测物理急停、保护停、断连、命令超时和感知失效路径。
- [ ] 复核相机外参、TCP、胶囊保守包络和距离监控。
- [ ] 依次完成空载、无障碍、最低速度单步和短轨迹测试。

#### P8.3 低速可执行性验证

- [ ] 在相同条件下比较冻结的 Instant-Link-SAC 与 Proposed。
- [ ] 不故意制造接触，不安排高风险无保护基线。
- [ ] 报告最小距离、安全违反、介入、修正量、任务、求解耗时、感知丢失和人工/保护停止。

门控：只支持低速部署可行性与风险响应结论，不外推为高速、形式化或认证级安全。

## 5. 最小可写论文路径

时间有限时，按以下顺序完成：

1. P3.0：逐子步测量、几何标定和字段语义。
2. P3.1：动作条件间隙、开环真值和线性化误差。
3. P5.2：标准 One-Step-QP 与轨迹一致运动边界。
4. P6.0：固定 Top-k、全身验收、反例增广和制动 fallback。
5. P6.1：五组配对主比较。

可暂缓：双障碍泛化、大规模真实实验、LDRC 深度改进、风险自适应权重和 RTB actor 重训。

## 6. 当前论文写作边界

当前可以写：

- 正式三方法基线已按独立 episode seed 协议完成，具体数值引用结果事实源。
- `Instant-Link-SAC` 是当前主线的冻结名义 actor。
- action-independent 预测风险存在预警收益但误报偏高。
- Reactive-Projection 在既有控制周期采样口径下具有工程收益。
- endpoint/quintic 过渡实验支持命令峰值边界，不支持反馈边界或最终预测 QP 结论。
- Predictive-Link-SAC observation/reward 是已冻结的负消融。

当前不能写：

- 动作条件预测已经优于当前趋势预测。
- One-Step-QP、固定 Top-k PTQP 或 VG-PTQP 已经实现或验证。
- 现有 P5 已证明反馈运动边界或逐子步碰撞安全。
- VG-PTQP 已降低真实漏检、碰撞或安全违反。
- 制动 fallback、连续时间安全或递归可行性已经得到保证。
- 真实 UR5 已经验证 Proposed，或方法提供形式化无碰撞保证。

所有具体数值、样本量和分场景结论只从 [实验结果事实源](thesis_experiment_results.md) 引用；封存后仍在维护的方法定义、论文结构和原创性表述见 [论文大纲](../../thesis/thesis_outline.md)。
