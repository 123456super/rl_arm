# 从零实验执行跟踪

> 启动日期：2026-09-09  
> 状态：全部实验结论清零，从协议与测量审计重新开始。  
> 本文件只维护任务、状态、依赖和验收门槛；详细方法见 [论文大纲](../../thesis/thesis_outline.md)，所有新数值只写入 [新实验结果事实源](experiment_results.md)，阶段执行与失败分流见 [R0–R6 Gate 实验执行指南](r0_r6_gate_execution_guide.md)。

## 1. 重启原则

- 旧实验、旧 checkpoint 评估和旧参数只能帮助定位代码，不作为新实验先验、门控证据或论文结果。
- 不从旧 held-out 结果反推参数；所有参数搜索重新限制在新 validation 集。
- 先冻结论文主问题、执行器、指标、seed 和统计协议，再训练或评估。
- 先验证测量与几何，再验证预测，再实现安全层；不并行改变多个核心因素。
- 所有方法使用同一 RTB 执行管线、同一逐物理子步测量、同一 actor 集和成对 episodes。
- 每阶段未通过时保留负结果并停止进入下游，不临时改写核心创新。

## 2. 状态总览

| 阶段 | 状态 | 目标 | 前置条件 |
| --- | --- | --- | --- |
| R0 | development gate passed; clean freeze pending | 冻结代码、环境、数据和统计协议 | 无 |
| R1 | nominal gate passed | 验证逐子步测量、执行轨迹与胶囊几何 | R0；QP 专属检查在 R4 前完成 |
| R2 | R2d gate failed; R2e diagnostic ready | 从头建立执行器一致的 SAC 名义基线 | 隔离 SAC 基础到达能力与风险多任务耦合 |
| R3 | todo | 验证动作条件预测和距离梯度 | R2 |
| R4 | todo | 建立标准 One-Step-QP | R3 |
| R5 | todo | 建立固定 Top-k Predictive-Trajectory-QP | R4 |
| R6 | todo | 建立 VG-Predictive-Trajectory-QP | R5 |
| R7 | todo | 冻结参数并运行 held-out 主比较 | R6 |
| R8 | todo | 鲁棒性与泛化 | R7 |
| R9 | conditional | 真实 UR5 低速验证 | R7 与现场签核 |

## 3. R0：预注册与可复现基座

- [x] 建立独立输出根目录 `outputs/restart_2026-09-09/`，并将全部旧输出封存到 `outputs/archive_pre_restart_2026-09-09/`。
- [x] 建立重启实验的独立配置命名空间 `configs/restart_2026-09-09/`，禁止复用会写入归档目录的旧配置。
- [x] 记录基准 Git commit、Python 环境、PyBullet、OSQP、NumPy、SciPy 和硬件状态；每个训练 run 另写完整源码哈希。当前 worktree 尚未提交，因此 manifest 必须保留 `git_dirty=true`。
- [x] 冻结 UR5 URDF、修正后的胶囊定义、动作范围 `0.7 rad/s`、相邻策略步限幅 `0.1 rad/s`、RTB `omega_c=30 rad/s`、20/240 Hz 频率和 episode 上限 240 个控制步。碰撞或安全到达正常终止，达到步数上限截断。
- [x] 冻结开发训练 seed `11`、正式 actor seeds `{101,202,303}`、validation base seeds `{1001,1002,1003}`、held-out base seeds `{2001,2002,2003}`；`episode_seed` 使用 `derive_episode_seed(base_seed, episode_index)` 的 Cantor 配对，禁止加法映射。
- [x] 冻结五个主场景为 random、upper_arm_crossing、elbow_crossing、forearm_crossing、wrist_crossing；validation 每个 base seed/场景 30 episodes，held-out 每个 base seed/场景 100 episodes，以相同 actor、方法、base seed、episode index 为配对单位。无障碍场景另作任务健全性门禁。
- [x] 冻结任务、安全、干预、命令/反馈平滑性、求解、fallback 和端到端实时性指标，字段口径见下方“冻结协议”。
- [x] 预注册 R2a 权重与 checkpoint 选择；R3--R7 的预测/QP 容忍区间须在对应 validation 开始前补齐。
- [x] 训练入口建立 `run_manifest.json`，记录配置、URDF、胶囊、源码、依赖和全部 checkpoint SHA-256，并拒绝覆盖非空 run 目录；评估总 manifest 检查在 R2 validation 前补齐。
- [x] 同 seed、同动作的双环境逐子步 observation、反馈、距离、风险和 contact 位级一致测试通过。

通过标准：两次小规模 dry-run 的状态、动作、障碍轨迹和指标逐项一致；协议完整写入新结果事实源。开发运行已用逐 run dirty 源码哈希和评估 plan manifest 满足可追溯 Gate；进入 R2-D 正式多 seed 训练前仍须建立干净 commit 或明确的不可变源码快照。因此 R0 不再阻塞 R2e 诊断，但尚未达到正式冻结状态。

### 冻结协议（v1）

下列 R2a--R2d 条目记录指南形成前已经预注册并执行的历史诊断链；它们的失败事实保留，但不映射为已完成的 R2-A/R2-C/R2-D。自 R2e 起启用“R2 顺序补正 v1.1”：R2e=`R2-A` 基础容量，R2-B 仅做 `0.055 m` 定位稳定性诊断，R2-C 在 R2-A 通过后另行预注册风险课程或正式修改名义 actor 职责，R2-D 才执行三个正式 seeds 并冻结 actor。

- R2a 只比较 `fixed_risk_penalty in {1,2,4,8}`，共享开发训练 seed 11、100000 steps 和每 10000 steps checkpoint；候选集合不是从封存结果缩小得到。
- 只评估 60000--100000 step checkpoints。候选必须同时达到无障碍 Success Rate >= 0.80、五场景宏平均 Success Rate >= 0.60；未达到则 R2 失败，不降低门槛。合格项依次按宏平均 Collision Rate、Safety Violation Rate 升序，Success Rate 降序、Final Position Error 升序选择；完全相同时选较小 penalty，同一 penalty 的 checkpoints 仍完全并列时选较早 step。最后一项仅补足 validation 前原规则未覆盖的确定性 tie-break，不增加新的性能选择指标。
- 选定 penalty 后才生成正式 seed 101/202/303 配置并从头训练，不把 seed 11 actor 放入主比较。每个正式 actor 的 checkpoint 使用相同 validation 规则单独选择并冻结哈希。
- R2a 失败后的 R2b 只修复一个已观测的训练支持集缺口：以 `episode_enable_probability=0.8` 固定 80% random-crossing、20% no-obstacle episodes，保持 `link_risk_v1`、奖励、RTB、动作限制和 seed 11 不变。先训练 penalty 1 单候选 100000 steps，仍只验证 60000--100000 checkpoints，并沿用 R2a 的两项成功率门槛和排序规则。若 R2b 仍失败则再次停止，不降低门槛；若通过，正式 seeds 采用同一混合比例。
- R2b 失败后确认第二个单独缺口：负的逐步位置回报与碰撞立即终止共同产生“提前碰撞优于困难超时”的 terminal shortcut。R2c 只把任务侧 `collision_penalty` 从 2 提高到 100；固定 20% no-obstacle、penalty 1、seed 11、100000 steps、checkpoint 范围、门槛和排序规则均不变。数值 100 在训练前冻结：它高于 R2b 后半程碰撞与超时 53.4 的未折扣回报差，并为 `gamma=0.99` 对约 51--72 步终止惩罚的折扣留出余量；这是一项待 R2c 验证的修复而非理论保证。R2c 若仍失败则停止并重新审查任务 MDP，不继续堆叠调参。
- R2c 失败后的 MDP 复审确认碰撞捷径已消失，剩余失败主要为 timeouts，且 60000--100000 steps 的无障碍与障碍宏成功率仍总体上升。R2d 因此只把从头训练预算从 100000 扩到 150000 steps；不无 replay 地续训、不改奖励/分布/门槛，只验证新增的 110000--150000 checkpoints。若 R2d 仍失败则不再以“训练尚未收敛”为由继续延长。
- R2d 失败后不再延长同一训练。R2e 是 R2-A 且不参与正式 actor 选择：关闭障碍，仅训练相同 SAC、RTB、`link_risk_v1` 接口和任务 reward 100000 steps，再验证 60000--100000 checkpoints 的无障碍 Success Rate 是否达到 0.80。若失败，修复 SAC/任务学习本身；若通过，只能证明基础容量成立，随后必须另行预注册 R2-C，才能区分 current-risk 多任务耦合并选择 curriculum 或任务 SAC 路线。
- `d_min`、`risk_global` 是 20 Hz 步末/下一 observation 的胶囊状态；`control_min_distance`、`control_max_risk` 是该 transition 内 12 个 240 Hz 子步的胶囊最小值和风险最大值，`control_collision` 是任一子步的 PyBullet contact。训练 cost 使用 `control_min_distance/control_max_risk/control_collision`，碰撞终止只使用 `control_collision`；胶囊间隙小于等于零仍只属于近似几何违反，不冒充真实接触。
- 达到 240 个控制步只产生 Gymnasium `truncated`，用于重置 episode；SAC Bellman target 仍从该 transition 的最终 observation bootstrap。只有碰撞或安全到达产生 `terminated` 并屏蔽 bootstrap。
- `qdot_substeps`、`command_acc_substeps`、`command_jerk_substeps` 是 240 Hz 下发命令，单位依次为 rad/s、rad/s^2、rad/s^3；`measured_qdot_substeps`、`measured_acc_substeps`、`measured_jerk_substeps` 是每次 `stepSimulation` 后的同单位反馈。旧 `physics_*` 字段只保留为命令侧兼容别名，不得再称为实测值。
- 安全距离物理阈值为 `d_safe=0.12 m`。R1 冻结单侧几何裕量 `m_geom=0.04 m`，供后续预测/QP 约束使用；名义 actor 仍报告原始胶囊距离和 PyBullet contact，二者不得混写。

## 4. R1：测量、执行与几何正确性

- [x] 每个 240 Hz 子步后读取机器人反馈、胶囊距离和 PyBullet contact。
- [x] 分开记录命令与反馈的速度、acceleration 和 jerk。
- [x] 验证 rate limiter 和有状态 Butterworth 名义执行每周期只调用一次。
- [ ] 验证离散 quintic 预测、QP 内轨迹和实际下发序列逐项一致。
- [x] 验证跨周期命令与反馈各自保留 `a_prev`，首子步 jerk 包含周期边界。
- [x] 按六个目标胶囊、随机关节构型和近表面间隙分层抽样 2400 点，以 PyBullet collision shape/contact 标定距离误差和危险漏检。
- [x] 修复原胶囊连杆错位、局部 offset 未随 link 姿态旋转以及半径偏小；校正后单侧最大误差 0.0352 m，冻结 `m_geom=0.04 m` 后危险漏检为 0/1597。
- [x] 明确本阶段全部测量字段单位、采样频率和聚合分母。

QP 内离散轨迹逐项一致性属于 R4 实现后才能完成的条件，不能作为 R2 名义 actor 训练的伪前置结果；它仍是进入 R4 主比较前的硬门禁。

通过标准：无已知子步漏采；轨迹一致性在数值容差内通过；胶囊误差/漏检满足预注册阈值。

## 5. R2：重新建立名义策略基线

### 5.1 历史诊断链（R2a--R2d，不计作补正后的子 Gate）

- [x] R2a 开发网格 `fixed_risk_penalty={1,2,4,8}`、seed 11 均完成 100000 steps；四个 run 的源码快照一致，manifest、20 个 checkpoint 文件及哈希完整，训练日志无 NaN/Inf。
- [x] 按冻结规则完成 R2a 的 60000--100000 step checkpoint validation；360/360 文件、10800/10800 episodes 完整且无重复 episode seed，但 20 个候选均未达到门槛，R2a gate failed，未选定 penalty。
- [x] 诊断确认 validation 目标可由 UR5 IK 达到，而 R2a actor 从未在训练中看到无障碍零风险占位 observation；保持 observation schema 不变，加入固定 20% no-obstacle episode coverage，并建立 R2b 单因素配置与测试。
- [x] 完成 R2b penalty 1 开发探针训练：100000 steps、10 组 step checkpoints、完整 manifest，686 个 episode 中障碍物启用比例 79.15%，日志 finite。
- [x] 按原门槛完成 R2b 的 60000--100000 checkpoints validation：90/90 文件完整，但 5 个候选均失败；最佳无障碍 Success Rate 0.7000，最佳五障碍宏 Success Rate 0.3711，均低于 0.80/0.60 门槛。
- [x] 诊断 R2b terminal shortcut：后 50000 步中，障碍场景 collision episode 平均 50.99 步、回报 -72.42，timeout episode 平均 240 步、回报 -125.84；碰撞成为优于继续尝试的错误捷径。
- [x] 预注册 R2c 单因素修复及 checkpoint validation 矩阵：只将 `reward.collision_penalty` 从 2 改为 100，其余沿用 R2b；仍验证 60000--100000 steps、原六场景与原 validation seeds。
- [x] 完成 R2c 开发训练：100000 steps、37.64 分钟、527 个完整 episodes、障碍物启用比例 79.13%、22 个 checkpoint/agent-state 哈希和 finite 日志完整；后 50000 步障碍场景碰撞 episode 从 R2b 的 124 个降为 7 个。
- [x] 按原门槛完成 R2c validation：90/90 文件完整；step 100000 无障碍 Success Rate 0.6556、五障碍宏 Success Rate 0.4800、Collision Rate 0.0067，5 个候选均未通过。
- [x] 完成 R2c 后 MDP 复审并预注册 R2d：R2c 100k 的五障碍失败中 233/450 为非碰撞未成功，碰撞仅 3/450；R2d 仅将从头训练预算增至 150000 steps。
- [x] 从头完成 R2d 150000-step 训练：56.88 分钟、824 个完整 episodes、障碍物启用比例 78.88%、32 个 checkpoint/agent-state 哈希和 finite 日志完整；R2d/R2c 的 100k actor SHA-256 完全一致。
- [x] 按原门槛完成 R2d validation：90/90 文件完整；5 个候选全部失败，最高无障碍 Success Rate 0.5222、最高五障碍宏 Success Rate 0.4600。
- [x] 审计全部 90 个无障碍 validation 目标：PyBullet IK 有 88/90 可进入 0.055 m 成功阈值；相同执行管线下阻尼最小二乘控制器抽查 10/10 成功，未发现任务/时限本身解释约半数失败的证据。

### 5.2 顺序补正 v1.1（从 R2e 开始）

- [x] 预注册 R2e 无障碍 SAC 容量诊断及 5-checkpoint validation；该 run 不可直接冻结为正式 actor。
- [ ] 训练并验证 R2e；在诊断结论前暂停正式 R2。
- [ ] R2-B 只分析当前 `0.055 m` success Gate 下的定位稳定性；现有 episode 在 5.5 cm 终止，禁止据此声称完成 3/2/1 cm Gate。
- [ ] R2e 通过后，先根据证据预注册 R2-C 的风险 curriculum，或同步修改论文为任务 SAC nominal actor；不得在结果出来前把路线写死。
- [ ] R2-C 通过后执行 R2-D：使用与部署相同的 RTB、rate limit、最终冻结 observation 和任务定义，从头训练 seeds `{101,202,303}`。
- [ ] 旧 EMA checkpoint 与 R2a--R2d actors 只作诊断，不参与主比较。
- [ ] validation 选择 checkpoint 的规则在训练前冻结，禁止按 held-out 表现选模型。
- [ ] 运行无障碍、随机穿越和四个重点连杆场景的 validation 健全性检查。
- [ ] checkpoint 冻结后生成哈希；R3--R8 不再更新 actor。
- [ ] 在独立 held-out 前只记录 validation 结论，不提前运行最终测试集。

通过标准：训练稳定、任务能力通过预注册下限、三个种子无系统性异常；否则先修复基线，不进入安全层。

## 6. R3：动作条件预测

- [ ] 用缓存 endpoint 生成执行一致的离散 quintic/终端保持 rollout。
- [ ] 建立同状态、同 endpoint 的仿真副本开环反事实真值。
- [ ] 对比当前反应式、当前速度趋势和动作条件预测。
- [ ] 检查所有建模连杆和验证时刻的表面间隙。
- [ ] 用中心差分方向导数验证距离—endpoint 梯度及最近特征切换。
- [ ] 在 validation 信赖域中估计几何、跟踪、感知、速度、线性化和网格裕量。
- [ ] 报告事件条件 LeadTime、coverage、误报、未来最小距离误差、危险连杆识别和耗时。

通过标准：动作条件预测相对当前速度趋势至少改善一个预注册预测主指标，且误报、梯度误差和耗时通过门槛。

## 7. R4：One-Step-QP

- [ ] 接入标准 OSQP，记录 status、iterations、primal/dual residual、slack 和端到端时间。
- [ ] 只使用冻结的当前几何构造安全约束，避免混入未来预测收益。
- [ ] 同一 endpoint 对全部实际子步施加位置、速度、acceleration 和跨周期 jerk 硬边界。
- [ ] 有界 safety slack 只用于诊断和降级；未通过验收的候选不正常执行。
- [ ] 为可行、不可行、数值失败和运动边界冲突建立单元测试。

通过标准：求解语义、轨迹一致性和命令边界正确；逐子步安全不劣于无安全层；端到端时间满足周期预算。

## 8. R5：固定 Top-k Predictive-Trajectory-QP

- [ ] 在与 R4 相同的 solver、运动边界、slack、actor 和 episodes 上加入动作条件多时刻几何。
- [ ] 冻结 Top-k、临界/首次越界相邻时刻和信赖域规则。
- [ ] 候选解运行全身非线性验收用于测量漏检，但不把反例反馈进 QP。
- [ ] 记录模型内拒绝率、遗漏连杆/时刻、线性化残差、修正量和 fallback。

通过标准：相对 R4 改善预注册预测安全主指标，且任务与干预代价在容忍范围；否则预测约束不进入核心贡献。

## 9. R6：VG-Predictive-Trajectory-QP

- [ ] 每轮在新参考 endpoint 重线性化累计约束。
- [ ] 在全部建模连杆、全部验证时刻上执行非线性回放。
- [ ] 每轮加入至多 `B_add` 个最严重反例；已有索引持续违反时仍重新线性化。
- [ ] 实现 `R_max`、deadline、solver/slack、NaN、无效梯度和停滞终止。
- [ ] 预构造并验证完整 jerk-limited 制动计划；失败时记录 `fallback_unverified` 并触发急停语义。
- [ ] 日志能够完整重建每轮约束、候选、反例、验收和执行来源。

通过标准：相对 R5 降低最终模型内漏检，并在最终安全或同等安全下的修正/fallback 代价上体现净收益；不能靠更多停止通过。

## 10. R7：冻结与 held-out 主比较

最低五组：

```text
Instant-Link-SAC + RTB
Reactive-Projection
One-Step-QP
固定 Top-k Predictive-Trajectory-QP
VG-Predictive-Trajectory-QP
```

- [ ] 只在 validation 冻结所有预测、QP、裕量、网格、反例和 deadline 参数。
- [ ] 锁定 actor/checkpoint 哈希和全部配置后一次性运行 held-out。
- [ ] 使用相同 actor、RTB、episode seeds 和逐子步测量进行成对比较。
- [ ] 报告分场景、分 actor、宏平均、配对效应量和分层置信区间。
- [ ] 同时报告成功、碰撞/违反、干预、命令/反馈平滑性、solver、细化、fallback 和端到端时间。
- [ ] held-out 后不调参；任何修改建立新版本并重新预注册。

通过标准：论文大纲中的每个核心主张都有隔离良好的正向消融；未通过的主张降级或删除。

## 11. R8：鲁棒性与泛化

- [ ] 位置噪声。
- [ ] 速度大小/方向误差。
- [ ] 感知与控制延迟。
- [ ] 障碍物速度/方向突变。
- [ ] 略超训练分布的速度。
- [ ] 双球形障碍物附加实验。

通过标准：一次只改变一个因素，报告退化曲线，并区分预测失配、QP 可行域、连续时间覆盖和 fallback 失败。

## 12. R9：真实 UR5 条件性验证

- [ ] 使用 R7 冻结的 actor、配置和完整安全链路重新生成离线预检报告。
- [ ] 完成控制器限幅、断连/超时、保护停、物理急停、感知失效和状态过期测试。
- [ ] 现场复核相机外参、TCP、胶囊包络和距离监控。
- [ ] 从空载、无障碍、最低速度单步逐级进入轻质球体实验。
- [ ] 只比较低速可执行性，不故意制造碰撞。

通过标准：只支持低速部署与风险响应结论，不外推为形式化、高速或认证级安全。

## 13. 当前写作边界

在 R7 通过前：

- 不在论文中引用归档实验数值作为当前方法证据；
- 不写“动作条件预测有效”“标准 QP 有效”或“VG-PTQP 已降低碰撞”；
- 不写连续时间安全、递归可行性或已验证制动保证；
- 只可描述研究问题、方法设计、历史问题如何促成本次重启，以及已通过的纯单元/一致性检查。
