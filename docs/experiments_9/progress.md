# 连续实验进度（v8 → v9 → v10 → v11）

更新日期：2026-09-18

历史基线协议：`continuous_pose_rfm_mean_smooth_then_safety_homotopy_v8`  
当前协议：`task_first_persistent_eta_replay_homotopy_v11`

除“V11 协议决策与实现状态”外，本文前述训练数值均为 V8/V9/V10 历史证据，不构成当前执行命令。当前命令只以 [experiment_order.md](experiment_order.md) 为准。

## 版本状态摘要

- V8：正式 S0 seed 11001 完成三个 100k Block，最终 `eta=0.65`、`K_pose=0`，按当时协议失败停止。
- V9：正式 S0 完成累计 200k，最终 `eta=0.4` recovery；SAC 数值稳定，但原剩余预算低于课程理论下界，因此放弃 Block3。
- V10：已完成 600k S0 并按 `hard_budget_stop` 归档；V11：正式 S0 Block1 已完成，Block2 在一次 eta 换档同步缺陷后从累计 150k checkpoint 恢复，下一命令见 [experiment_order.md](experiment_order.md)。

## V8 已完成

| 项目 | 结果 |
| --- | --- |
| observation | 从 71 维更新为 98 维；旋转 6D、相对旋转 6D、末端 twist、统一尺度的 `qdot`、逐连杆 approach/risk 已接通环境 |
| reward | 使用非线性乘法位姿质量和折扣势函数；测地角继续用于奖励/成功；平滑量使用六关节归一化速度增量的均方；安全项仍只由 `xi` 缩放 |
| replay 重标 | reward 参数随 replay 保存，并在安全同伦改变 `xi` 时按同一 v8 契约重算 |
| 产物隔离 | v8/v9/V10/V11 输出分别在 `outputs/experiments_9_16/`、`outputs/experiments_9_17/`、`outputs/experiments_9/v10/`、`outputs/experiments_9/v11/`；文档统一在本目录维护 |
| 针对性测试 | `25 passed`；包含旋转 6D 符号不变性、98 维索引、速度尺度、逐连杆字段、位置优先样例、闭环无刷分与环境平滑量均方契约 |
| 全量回归测试 | `83 passed`；平滑量修复未破坏训练、恢复、课程、奖励重标或环境接口 |
| 奖励抽样预检 | 扩大到 1000 个经运动学校验的可达目标后，99.8% 满足 `理想到达 > 静止超时 > 严格碰撞/硬失败` |
| 回报中位数 | 理想 60 步到达 `31.356`；静止超时 `-1.442`；立即严格碰撞/硬失败 `-10.016`；低 `xi` 宽容碰撞单步 `-0.216` |
| PyBullet/contact 一致性 | 10 构型 × 6 胶囊 × 2 方向，共 120/120 一致；接触时最大胶囊距离 `0.02944 m < 0.12 m`，最小风险 `0.80 >= 0.79` |
| 短训练/恢复冒烟（修复前协议，已终止） | seed 91016 训练 300 transition、完成 293 次更新并保存 98 维完整 checkpoint；恢复后继续 50 transition，global step `300→350`、replay `300→350`、update `293→343`；只保留恢复链路证据，不得用于当前协议续训 |
| 正式 S0 block1 | seed 11001 完成 `100000` transition、`99001` 次更新和 4 个完整 checkpoint；16 档位置范围课程全部推进到 `g=1`，`K_position=21844/25000`，尚未进入姿态课程 |
| 正式 S0 block2 | 从 block1 完整恢复并训练至 global step `200000`；完成位置巩固，姿态课程推进到 `eta=0.5`，但在线姿态窗口与确定性位置探针均未满足晋级条件，block2 末尾为 recovery 模式 |
| 正式 S0 block3 | 从 block2 完整恢复并训练至 global step `300000`；通过 `eta=0.5/0.6` 后停在 `eta=0.65`，最终 probe `0.94/0.04/0.02`（success/collision/timeout），无 Gate 资格并按三 block 规则停止 |

1000 样本中有 0.2% 未满足完整排序，表现为立即严格碰撞略优于该样本的静止超时。它没有重现旧方案中 99% 样本立即碰撞更优的阻断，也未违反碰撞有界与宽容期可继续的设计；仍需在短训练的真实轨迹分布上复核，不能把独立样本审计当作收敛证明。

## 350 transition 真实轨迹奖励审计（修复前诊断，已终止）

数据来自 `preflight_v8_smoke` 的 300 步及其完整恢复后的 50 步。全部 350 步都处于前 3000 步随机 warm-up，因此该审计检验的是“PyBullet 真实转移上的奖励方向”，不是已学习策略的收敛效果。

| 指标 | 结果 |
| --- | --- |
| 单步 reward | min `-1.640`，p05 `-0.685`，median `-0.092`，p95 `0.514`，max `20.885` |
| 非成功步 | 347 步；均值 `-0.088`，中位数 `-0.100`，正奖励比例 `37.18%` |
| 成功步 | 3 步；奖励范围 `20.604～20.885`，均值 `20.739` |
| 靠近目标 | 158 步；79.75% 为正，奖励中位数 `0.107` |
| 远离目标 | 185 步；0% 为正，奖励中位数 `-0.259` |
| 奖励与位置改善相关性 | `0.887` |
| 完成 episode | 1 个 timeout，未折扣/折扣回报 `-21.234/-9.737`；3 个成功，折扣回报 `18.460/20.727/11.201` |
| critic 后 50 次更新 | `Q1=1.619`、`Q2=1.617`、target=`1.636`；无非有限值，没有发现早期 Q 发散 |

32 个“位置略微改善但奖励仍为负”的 transition，中位位置改善只有 `1.48 mm`，中位奖励 `-0.025`。其原因是微小改善不足以抵消折扣势函数的时间压力、约 `0.012` 的关节速度代价和接近饱和的 `0.010` 平滑代价。这种符号不是方向错配：改善较明显的 126 步均得到正奖励，而所有位置恶化步均为负奖励。

当前结论是 S0-P 的位置奖励方向通过真实转移检查，成功奖励和非终止奖励也处于不同数量级；没有重新出现“立即失败比持续尝试更有利”或早期 Q 爆炸。局限是当前 `eta=0`、无障碍且动作来自 warm-up，尚未验证姿态乘法项和已学习 actor 的分布。v8 transition 日志现已增加 `Phi`、质量、进展、速度和平滑代价等原始分项，正式 S0 可直接复查而无需反算。

### 越过 warm-up 的补充审计（修复前诊断，已终止）

另运行 `preflight_v8_trajectory_audit` 共 3500 transition：前 3000 步为随机 warm-up，后 500 步为 SAC 随机策略动作。该 run 使用 `validation-mode` 的 512 容量 replay 和 batch 8，只用于实现审计，不能作为正式超参数下的性能实验。

| 指标 | warm-up 3000 步 | actor 500 步 |
| --- | ---: | ---: |
| reward 中位数 / 均值 | `-0.057 / 0.038` | `-0.057 / -0.072` |
| reward p05 / p95 | `-0.516 / 0.383` | `-0.389 / 0.195` |
| 正奖励比例 | `34.73%` | `31.80%` |
| 靠近步中正奖励比例 | `73.73%` | `67.37%` |
| 远离步中正奖励比例 | `0%` | `0%` |
| reward 与位置改善相关性 | `0.804` | `0.727` |
| task reached | `15` | `0` |
| 速度代价均值 | `0.0132` | `0.0140` |
| 平滑代价均值 | `0.00992` | `0.00995` |

actor 段唯一完成的 240 步 episode 超时，未折扣/折扣回报为 `-18.394/-8.550`。因此短训练后的 actor 尚未形成可声明的到达能力；这符合只有 500 个策略 transition、极小 replay/batch 的预检性质。奖励方向仍保持正确：actor 的 264 个位置恶化步全部为负奖励，没有发现策略通过远离目标、提高速度或来回摆动获得正奖励。

该轨迹暴露出平滑代价退化：warm-up `97.2%`、actor `98.8%` 的 transition 达到 `0.01` 上限。根因是环境把六关节归一化速度增量的平方**求和**后再裁剪；普通多关节动作也很容易超过 1。当前协议已改为逐关节**均方**后再裁剪。上述旧轨迹及其 checkpoint 仅用于定位问题，不再代表当前奖励契约。

3500 步末尾 `Q1/Q2/target=9.188/9.157/9.161`，最后一次 critic loss 为 `0.0114`，最近 100 次三者均值为 `8.958/8.946/8.874`，且没有非有限值。数值上 critic 与自身 soft target 同步，没有出现旧实验中 Q 约 107 且与 target/回报继续分离的现象；但 soft Q 包含熵项，当前数值不能直接与环境回报比较，也不能用该预检排除长期高估。

## 平滑代价修复后真实轨迹审计

修复后重新独立运行 `preflight_v8_mean_smooth_trajectory`：seed 91019，共 3500 transition，其中前 3000 步为随机 warm-up，后 500 步为早期 actor。该 run 仍使用 `validation-mode`，目的只是验证新公式在真实 PyBullet 转移中的数值分布，不用于判断策略是否收敛。日志总 reward 的分项重构最大误差为 `3.55e-15`。

| 指标 | warm-up 3000 步 | actor 500 步 |
| --- | ---: | ---: |
| 平滑代价达到 `0.01` 上限 | `15.0%` | `13.8%` |
| 平滑代价 min / p05 | `0.00020 / 0.00201` | `0.00058 / 0.00188` |
| 平滑代价 median / mean | `0.00623 / 0.00628` | `0.00610 / 0.00622` |
| 平滑代价 p95 / max | `0.010 / 0.010` | `0.010 / 0.010` |
| reward median / mean | `-0.0558 / 0.0159` | `-0.0344 / -0.0604` |
| reward p05 / p95 | `-0.554 / 0.407` | `-0.268 / 0.052` |
| 靠近步中正奖励比例 | `75.57%` | `42.93%` |
| 远离步中正奖励比例 | `0%` | `0%` |
| reward 与位置改善相关性 | `0.815` | `0.618` |
| task reached | `12` | `0` |

饱和率由修复前的 `97.2%/98.8%` 降至 `15.0%/13.8%`，同时保留极端抖动的 `0.01` 最大惩罚。p05、中位数、p95 已形成明显间隔，说明该项现在能区分不同程度的速度突变，不再近似常数。actor 仅经历 500 个策略 transition，0 次到达不能用于否定收敛；方向检查仍成立，所有远离目标的 transition 奖励均非正。最后 100 次更新的 `Q1/Q2/target` 均值为 `8.523/8.539/8.417`，最后一次 critic loss 为 `0.0620`，未出现非有限值。

## V8 正式 S0 seed 11001 block1

正式 run `s0_v8_seed11001_block1` 从随机初始化完成 100000 transition，使用正式 replay、batch 和更新配置。末尾 checkpoint 为 `checkpoints/step_0100000.pt`，协议、seed、98 维 observation、完整 replay、optimizer、课程、RNG 和未完成 episode 状态均已保存。

| 指标 | 结果 |
| --- | --- |
| 训练计数 | `100000` transition，`99001` 次更新，`8965` 个完整 episode，replay `60000` |
| 位置课程 | 16 档全部推进到 `g=1`；完整范围累计 `21844/25000` 个合格 transition，尚差 `3156` |
| 姿态课程 | `eta=0`、姿态容差 `pi`；位置巩固尚未完成，因此姿态课程尚未开始，probe 尚未触发 |
| 全部合格 episode | `8943` 个；位置到达率 `99.698%`，timeout `0.157%`，自碰撞 `0.123%`，关节越界 `0.022%` |
| 完整范围 episode | `1371/1371` 位置到达；timeout、自碰撞、环境碰撞和关节越界均为 `0` |
| 最近 1000 个合格 episode | 到达率 `100%`；平均/中位 episode 长度 `15.894/15` 步，位置终点误差 p95 `0.05427 m` |
| critic 末值 | `Q1/Q2/target=24.441/24.435/24.260`，critic loss `0.03966`；完整范围 episode 实际折扣回报均值/中位数 `25.255/25.307`，未见 Q 高估 |
| 温度 | `alpha` 从 `0.2` 下降后稳定在约 `0.0066~0.0069`，末值 `0.006872`，没有继续单调坍缩 |
| 奖励方向 | 完整范围所有位置恶化的非终止 transition 均为非正奖励；最近 10000 步同样无反例 |
| 平滑代价 | 最近 10000 步中位数 `0.00182`、p95 `0.00664`、上限 `0.01`，没有重新退化为近似常数 |

各位置范围档均在达到最低 5000 个合格 transition 后及时晋级；成功率最低的是 `g=0.32` 档的 `96.04%`，仍高于 `0.80` 晋级门槛，之后恢复。block1 的限制不是位置学习失败，而是 100000 步预算在完整范围位置巩固达到规定的 25000 步前结束。不得降低巩固阈值或提前进入姿态/S1；应使用最终完整 checkpoint 原样恢复 block2。

## V8 正式 S0 seed 11001 block2

正式 run `s0_v8_seed11001_block2` 从 block1 的 `step_0100000.pt` 以 `same_stage_continuation` 完整恢复，并继续训练 100000 transition。global step、update、episode、replay、课程与 RNG 均连续；末尾 checkpoint 为 `checkpoints/step_0100000.pt`。

| 指标 | 结果 |
| --- | --- |
| 训练计数 | 累计 global step `200000`、update `199001`、完整 episode `11700`，replay `60000` |
| 课程状态 | `g=1`、`K_position=121844`、位置阶段完成；姿态课程推进到 `eta=0.5`，容差 `1.6208 rad`，当前档累计 `28378` transition |
| 已通过姿态档 | `eta=0,0.1,0.2,0.3` 的确定性位置探针均一次通过；`eta=0.4` 首次探针 `0.96` success、`0.02` collision，恢复后以 `1.00/0.00` 通过并晋级 |
| 当前姿态窗口 | `eta=0.5` 最近 100 个当前任务 episode 成功率 `0.79`、位置到达率 `0.82`；累计该档 350 个 episode 成功率 `0.731` |
| 位置保持 | 最近 100 个在线位置锚点 `100/100` 到达；但当前档两次固定确定性探针为 `0.96/0.02` 与 `0.94/0.02`（success/collision），末尾保持失败状态 |
| recovery | 最后一次 probe 后启用 anchor episode 概率 `0.50`，replay 使用 anchor/current/historical=`50/35/15`；末尾窗口仍未恢复到晋级条件 |
| 姿态学习信号 | 当前任务平均姿态改善随 `eta` 从 `0.1` 的 `0.073 rad` 增至 `0.5` 的 `0.674 rad`，说明策略确实响应姿态目标，并非姿态奖励完全失效 |
| `eta=0.5` 失败组成 | 350 个 episode 中 256 成功、49 自碰撞、43 timeout、2 关节越界；最近 100 个为 79 成功、12 自碰撞、9 timeout |
| critic/温度末值 | `Q1/Q2/target=18.181/18.217/18.010`，critic loss `0.0695`，`alpha=0.008589`；课程变难后价值下降但 critic/target 保持贴合，无非有限值或发散 |
| replay 抽样 | recovery 末尾每 batch 为 anchor/current/historical=`128/90/38`，与 `50/35/15` 目标一致 |

block2 证明 6D 姿态输入与乘法位姿势函数能够推动姿态误差下降，并已越过旧方案较早的姿态停滞区；但 `eta=0.5` 同时出现当前任务成功率不足和固定位置探针退化，不能视为通过。训练锚点 100% 与固定确定性探针失败并不矛盾：前者来自随机目标和随机策略动作，后者使用冻结的 50 个私有固定目标与确定性动作，后者是晋级事实源。当时按既定协议从末尾 checkpoint 继续 block3，并保持 reward、Gate 与 replay 比例不变；block3 的最终结果如下。

## V8 正式 S0 seed 11001 block3

正式 run `s0_v8_seed11001_block3` 从 block2 的 `step_0100000.pt` 以 `same_stage_continuation` 完整恢复，并继续训练 100000 transition。末尾累计 global step `300000`、update `299001`、完整 episode `13678`，replay 保持 `60000`；25k、50k、75k 和 100k checkpoint 均已保存。

| 指标 | 结果 |
| --- | --- |
| 课程推进 | `eta=0.5` 经第二次 block3 probe 通过；`eta=0.6` 前三次 probe 失败、第四次以 `1.00` success 和零碰撞通过；随后到达 `eta=0.65` |
| `eta=0.5` 当前任务 | 187 个 episode，success `82.89%`、position reached `83.96%`、self-collision `11.76%`、timeout `4.81%`、joint-limit `0.53%`；平均姿态改善 `0.702 rad` |
| `eta=0.6` 当前任务 | 524 个 episode，success `75.57%`、position reached `78.44%`、self-collision `11.83%`、timeout `12.60%`、joint-limit `0%`；平均姿态改善 `0.910 rad` |
| `eta=0.65` 当前任务 | 434 个 episode，success `80.18%`、position reached `81.11%`、self-collision `11.75%`、timeout `7.83%`、joint-limit `0.23%`；平均姿态改善 `1.049 rad` |
| `eta=0.65` 趋势 | 前/后半段 success 为 `81.11%/79.26%`，self-collision 为 `10.14%/13.36%`；末 100 个 episode success 回到 `82%`，但没有形成单调、稳健改善 |
| 位置锚点 | block3 共 833 个，position reached `98.80%`、self-collision `0.72%`、timeout `0.48%`；末尾 100 个窗口为 `0.99` |
| 最终固定探针 | `eta=0.65` 两次均为 `0.94` success；第一次 collision/timeout=`0.02/0.04`，第二次为 `0.04/0.02`，均无 joint-limit，故保持 recovery |
| 最终资格 | `eta=0.65`、`K_pose=0`、`full_pose_steps=0`、最终 probe 未通过、`s0_goal_gate_eligible=false` |
| critic/温度末值 | `Q1/Q2/target=17.986/18.035/17.791`，critic loss `0.12663`，`alpha=0.009255` |
| 数值稳定性 | 100000 次 update 的记录字段均为有限值；Q1/Q2 全程平均差 `0.043`，末 10000 update 的双 critic 均值相对 target 高约 `0.285`，未见发散 |
| recovery 抽样 | 末尾每 batch anchor/current/historical=`128/90/38`，严格符合 `50/35/15` |

block3 显示方法仍有学习能力，而不是在 `eta=0.5` 完全停滞：它恢复并通过了 `eta=0.5` 和 `0.6`，姿态改善量也随难度增加。但在第三个正式 block 结束时只到 `eta=0.65`，离 `eta=1`、25000 个完整位姿巩固 transition 和 S0 Gate 仍有实质距离。在线随机位置锚点保持接近 99%，不能覆盖固定探针持续发现的少数困难目标；在 `eta=0.65` 最后两次 probe 中，失败仍由自碰撞和 timeout 构成。与此同时 critic、target、温度、梯度和 replay 比例均正常，因此主要问题是策略在姿态训练下的位置安全保持与固定目标泛化，而不是 SAC 数值崩溃。实现复核还显示：S0 无障碍时 98 维 observation 的障碍几何/risk 段固定为无风险占位值，self-collision 只在实际接触后以 `-10` 硬失败进入奖励，没有连续的自碰撞间隙或接近风险信号；这与当前任务约 `11.7%` 的自碰撞率相符，是下一版必须优先验证的设计缺口，但不能在本 run 中途修改。

按第 3.4.7 节预先固定的停止规则，本条 seed 11001 v8 S0 在累计三个 block 后正式失败。保留 block3 最终 checkpoint 仅供诊断，不再续训 block4，不降低确定性探针或 Gate 阈值，也不进入 S1。下一版协议应先定位固定 probe 的碰撞构型，并在不使用私有 probe 样本训练的前提下改进自碰撞几何信号或困难位置保持机制；任何 reward、observation、replay 或课程修改都必须升级协议版本并从头训练。

## V8 问题优先级（历史）

1. **系统级瓶颈：困难目标上的位置保持与泛化。** 在线随机锚点末窗为 `0.99`，固定确定性 probe 却连续两次只有 `0.94`；平均锚点表现不能保证少数固定困难目标安全。
2. **最大直接失败类型：自碰撞。** `eta=0.65` 的 434 个当前任务 episode 中有 51 个 self-collision、34 个 timeout、1 个 joint-limit；最终 probe 的 3 个失败中有 2 个碰撞、1 个 timeout。自碰撞数量最多，但不是全部失败。
3. **首要设计缺口：缺少连续 self-clearance/self-risk。** S0 的外部障碍风险字段为空场景占位值，策略只能从 `q` 隐式学习自碰撞，reward 只在接触后给 `-10` hard failure。该机制与观测到的碰撞率一致，应作为下一版首要验证假设，但尚未通过消融证明是唯一根因。
4. **次要但不可忽略：timeout。** `eta=0.6` timeout 达 `12.60%`，`eta=0.65` 为 `7.83%`；即使完全消除自碰撞，也不能据此推断课程会自动达到 `eta=1`。
5. **已排除的主要方向：数值发散和 replay 配比错误。** critic/target、alpha、梯度、有限值检查及 recovery 的 `50/35/15` 抽样均正常。

## V8 结束时的后续判断（历史）

- 对 block3 最终 actor 复现并定位固定 probe 中的自碰撞与 timeout 构型，不把私有 probe 目标回灌训练；
- 基于诊断定义新协议版本，重新审计 observation、连续自碰撞信号和位置保持机制后从随机初始化开始；
- 当前 v8 不执行 S1/S2 或多 seed 扩展，因为主 seed 已在 S0 按规则失败。

## V8 兼容性说明（历史）

v8 改变了输入维度和奖励语义。旧 actor/critic 的第一层形状不匹配，旧 replay 也缺少新状态量；即使强行补零，旧 critic 的目标值仍对应另一套奖励。因此不得使用 `--restart-pose-from`、`--restart-retention-from` 或 `--resume` 加载 v2～v7。平滑量仍使用平方和的旧 `continuous_pose_rfm_then_safety_homotopy_v8` checkpoint 也不得续训；只有当前 `continuous_pose_rfm_mean_smooth_then_safety_homotopy_v8` 协议的完整 checkpoint 才能恢复。

## V8 → V9 交接（历史）

2026-09-17 将连续自碰撞信号实现为同一研究线下的新协议版本 v9。v9 将输入扩展为 122 维、replay 原始字段扩展为 22 维，并改变奖励与课程状态，因此 v8 checkpoint 都不得恢复到 v9；代码已验证会在加载网络前拒绝该协议混用。

## V9 正式 S0 Block1（历史）

原始进程在 `stage_step=66646` 意外中断；从最后完整 checkpoint 恢复后，恢复 run `outputs/experiments_9_17/s0_v9_seed11001_block1_resume50000/` 正常完成剩余 `50000` steps。两个 v9 run 目录按 `global_step` 去重合并后，共计 `100000` transitions、`8945` episodes 和 `99001` SAC updates。

| 指标 | 结果 |
| --- | ---: |
| position/task reached | `8907/8945 = 99.575%` |
| safe success | `99.575%` |
| self-collision | `8/8945 = 0.0894%` |
| joint-limit | `3/8945 = 0.0335%` |
| timeout | `27/8945 = 0.3018%` |
| environment/obstacle collision | `0` |

最近 100、500、1000 个 episode 均为 100% 到达、零 self-collision、零 timeout。后期 alpha 约 `0.0064`、Q 值约 `24`，未见 NaN/Inf 或发散。前后半段 reward 均值为 `3.106/1.716`，与目标尺度课程升至 `g=1` 相符；最后 10k 回升至 `1.900`。

当前课程状态为 `goal_scale=1`、`K_position=21709/25000`、`eta=0`、`lambda_self=0.02`、`s0_goal_gate_eligible=false`。还差约 `3291` 个合格 transition，不能开始姿态课程、self-safety ramp 或 S1/S2。下一步命令和预计耗时见 [experiment_order.md](experiment_order.md)。

## V9 正式 S0 Block2（历史）

Block2 正常完成 100000 stage steps，累计 `global_step=200000`。最终 checkpoint
为 `outputs/experiments_9_17/s0_v9_seed11001_block2/checkpoints/step_0100000.pt`。

| 课程状态 | 结果 |
| --- | ---: |
| goal scale / `K_position` | `1.0 / 121709` |
| `lambda_self / K_self` | `1.0 / 96556` |
| orientation | `eta=0.4`，level index `4` |
| current pose / anchor window | `0.81 / 1.00` |
| retention | `recovery`，anchor probability `0.50` |
| `K_pose` / Gate eligible | `0 / false` |

Block2 共完成 2293 episodes：safe success `91.71%`、self-collision `2.57%`、
joint-limit `0.26%`、timeout `5.45%`。由于数据跨越多个姿态档，总体比例只用于
描述训练过程，不能替代分档性能：

| eta（非锚点） | episodes | safe success | self-collision | timeout |
| ---: | ---: | ---: | ---: | ---: |
| `0.0` | `530` | `99.81%` | `0.19%` | `0%` |
| `0.1` | `288` | `98.96%` | `0.35%` | `0.35%` |
| `0.2` | `370` | `87.30%` | `3.51%` | `9.19%` |
| `0.3` | `248` | `75.40%` | `8.47%` | `15.32%` |
| `0.4` | `372` | `80.38%` | `5.65%` | `13.17%` |

`eta=0.4` 最后 100 个非锚点 episode 为 success/self-collision/joint-limit/timeout
=`0.81/0.09/0.01/0.09`。v8 同档最后 100 个为 `0.86/0.09/0/0.05`；目前
不能证明 v9 已在同难度显著降低自碰撞，且失败的一部分转移成了 timeout。

固定位置 probe 在 global step `192410` 得到 `0.88/0.02/0/0.10`，在
`199475` 得到 `0.98/0.02/0/0`（success/collision/joint-limit/timeout）。第二次
success 已达标，但仍有 1/50 collision，违反 collision 必须为 0 的条件，因此
训练器正确冻结在 `eta=0.4` 并进入 recovery。

数值训练稳定：最终 alpha 约 `0.0097`；最近 1000 updates 的
Q1/Q2/target 均值约 `19.43/19.45/19.04`，critic loss 均值约 `0.245`；没有
NaN、Inf 或 Q 发散。Block2 的主要瓶颈不是位置能力或数值稳定性，而是姿态要求
下的安全动作选择与困难位置泛化。完整 self 权重已经生效，仍存在碰撞和较高
timeout，因此下一 Block 必须同时观察 collision、timeout 和 safe success。

### 固定预算下的可达性结论

v9 的正式停止规则是每 stage 最多三个 100k block。Block2 结束时仍为
`eta=0.4` recovery。当前 probe 后至少还需
约 4654 个本档非锚点 transition 才能重试；通过后，从 `eta=0.5` 到 `0.98` 的
11 档至少需要 `55000` 个非锚点 transition，进入 `eta=1` 后还需 `25000` 个
非锚点完整位姿巩固。忽略任何额外失败，非锚点下界已经约为 `84654`；正常模式
25% 锚点和当前 recovery 50% 锚点折算后的总 transition 下界约为 `116000`，严格
超过第三个 block 剩余的 100k 预算。

这不表示三 block 预算从训练起点就数学不可达：理想的一次通过轨迹可以在 300k 内
完成。它表示实际 v9 在前 200k 的学习速度和 probe 失败已经消耗了过多预算，因此
在预先固定的 300k 上限内现已不可能取得 Gate 资格。若坚持原协议，Block3 只能作为
最后一个正式诊断 block，结束后应按规则报告 v9 S0 失败；不能进入 S1。

若研究目标改为继续探索最终可收敛性，可以在 Block3 前单独登记“扩展预算”协议并
从当前 checkpoint 继续，无需因预算变化重训网络；但它属于事后协议修订，必须与原
v9 固定预算结果分开报告，并对后续所有训练 seed 使用同一新上限。

原 v9 协议下的最后一个 S0 Block3 命令为：

```bash
python scripts/train_thesis_homotopy.py \
  --config configs/experiments/thesis_homotopy.yaml \
  --stage s0 --seed 11001 \
  --resume outputs/experiments_9_17/s0_v9_seed11001_block2/checkpoints/step_0100000.pt \
  --run-name s0_v9_seed11001_block3
```

该命令现仅作为历史记录，**不得执行**。V9 已被新的奖励与课程契约取代。

## V10 历史协议决策与 600k 结果

对首轮三篇参考论文完成原文方法审查后，V10 于 2026-09-17 冻结为新的候选协议。详细证据见 [reward_literature_analysis.md](reward_literature_analysis.md)。

V10 解决 V9 的两个可识别设计问题：

1. V9 在完成位置任务后同时推进姿态难度和连续 self penalty，无法保证先学会完整位姿基础任务。
2. V9 的 `2 R_self+8 V_self` 位于 `[0,10]`，在 `lambda_self=1` 时每步连续约束可与一次 `-10` 硬失败同量级，可能造成保守 timeout。

新契约保留 122 维 observation、collision-mesh self-clearance/TTC/approach/risk、逐子步真实 self-contact 硬终止与 `-10`，但作如下修改：

| 项目 | V10 定义 |
| --- | --- |
| 任务顺序 | 位置课程与巩固 → 完整位姿课程与巩固 → self-safety ramp → 满权重巩固 |
| 任务阶段 self 权重 | `lambda_self=0`；硬碰撞安全机制始终开启 |
| safety ramp 门槛 | `eta=1`、`K_pose>=25000`、位姿窗口≥0.80、锚点窗口≥0.95、确定性位置 probe 通过 |
| 性能下降 | 冻结 `lambda_self`；不继续加约束，也不降低已经达到的权重 |
| 连续安全尺度 | `(2R+8V)/10`，每个 external/self 组均在 `[0,1]` |
| 最终巩固 | `lambda_self=1` 后再完成至少 25000 个合格 transitions |
| 网络 | 122→256→256 MLP；不加入 LSTM，不使用专家 replay |
| checkpoint | V8/V9 全部不兼容；V10 必须随机初始化 |

已完成的实现验证：

- V10 协议标识、配置和独立输出目录已更新；
- curriculum checkpoint 增加 `orientation_full_scale_min_steps`、`self_full_weight_min_steps` 和 `full_weight_steps`；
- 训练日志增加 `s0_phase`、`pose_phase_complete` 和满权重巩固计数；
- evaluator 与 S0→S1 阶段切换均检查满权重巩固；
- 当前 V10、停止分析器和 evaluator 定向测试为 `39 passed`，包括安全代价有界、完整位姿前不推进、性能下降后冻结、累计 stage 预算、相邻 Block 校验和恢复状态测试；全量回归为 `97 passed`。reward/goal/contact/self-contact/feasibility 五项 PyBullet 预检均通过；外部接触审计构造出 1199/1200 个真实浅接触 witness，已构造样本的一致率为 1199/1199，另 1 个无真实接触的生成样本不计入一致率；
- 200-step 真实轨迹的 observation 恒为 122 维、22 个奖励原子量齐全、reward 最大重构误差为 0；
- 24-step 保存加 12-step 恢复后 replay 从 24 连续增长到 36，V9 checkpoint 会在加载网络前被 V10 协议拒绝。

V10 seed 11001 已完成六个 Block、累计 `600000` transitions，按停止分析记为 `hard_budget_stop`。最终 `eta=0.75`、`pose_phase_complete=false`、`lambda_self=0`、`s0_goal_gate_eligible=false`。V10 的 SAC 更新全程有限且无发散；失败主要表现为 `eta=0.75` 当前任务成功率下降、timeout 增长和位置锚点遗忘。Block6 当前档成功率为 `60.2%`，timeout 为 `33.6%`，位置锚点窗口为 `92%`。

对最终 checkpoint 的 replay 审计显示：无障碍 FIFO 的 `60000` 条数据中只有 `eta=0` 锚点 `13558` 条和 `eta=0.75` 当前档 `46442` 条，`eta=0.1..0.7` 历史数据为 0。该事实直接触发 V11 的 replay storage 修订。

## V11 协议决策与实现状态

V11 于 2026-09-18 冻结。V11 不改变 observation、奖励原子量、奖励课程、self-contact 硬终止或网络；只修复持久历史 replay：

| 存储池 | 默认容量 | 不变量 |
| --- | ---: | --- |
| position anchor FIFO | 15000 | 所有 `eta=0` 或显式位置锚点写入；只保留自身最新数据 |
| current eta FIFO | 30000 | 只写当前非零 eta；换档时冻结，不被新档覆盖 |
| historical eta snapshot | 每档 3000 | 换档时由上一 current 池抽样冻结；只读、按 eta 分层采样 |

checkpoint 保存 `storage_version=2`、三个池、历史 eta 映射、当前 eta、各池写指针和 replay RNG。V10 或更早 replay 在加载时拒绝，V11 S0 必须从随机初始化开始。

V11 初始验证：`tests/test_thesis_homotopy.py` 定向 `34 passed`，全量回归 `98 passed`；300-step 真实轨迹和 50-step 恢复使累计 step 由 300 连续到 350；V10 checkpoint 因 protocol 不匹配被拒绝。

### V11 正式 S0 与 eta 换档中断

Block1 完成 100k transitions：`g=1`，`K_position=21904/25000`，完整目标范围位置成功率为 `99.77%`，尚未进入姿态课程。SAC 数值有限且稳定。

Block2 从 Block1 恢复后推进至 `eta=0.3`。在累计 step `160879`，`eta=0.3` 的确定性位置 probe 以 success/collision/joint-limit/timeout=`1/0/0/0` 通过，课程晋级到 `eta=0.4`。随后首个新档 episode 恰好被抽为 `eta=0` 位置 anchor：transition 只写入 anchor 池，current replay 仍标记为 `eta=0.3`，而梯度采样已请求 `eta=0.4`，一致性检查因此终止训练。这是 replay 换档原子性缺陷，不是 PyBullet inertial warning、网络发散或 checkpoint 损坏。

修复后，训练器在课程晋级时立即调用 `begin_orientation_level(new_eta)`，把旧 current 池冻结为 historical，并建立新档空 current 池；新档首个 episode 即使为 anchor，也可由 anchor 与 historical 数据补足 batch。新增精确回归测试后，`tests/test_thesis_homotopy.py` 为 `35 passed`，全量为 `99 passed`；用真实 150k checkpoint 模拟 `0.3→0.4` 后得到 historical `{0.1:3000,0.2:3000,0.3:3000}`，可正常抽取 122 维 batch。

最近的完整且一致 checkpoint 是 Block2 `step_0050000.pt`，对应累计 `150000` steps、`eta=0.3`。崩溃前累计 `150001..160879` 的 10879 steps 没有完整 checkpoint，保留为缺陷诊断日志，但不得计入连续训练、Block 汇总或平台判断。恢复运行必须使用新目录并训练 50k，使正式累计预算回到 200k。

### V10 训练预算与停止协议修订

V9 在 200k 时仍有明确档位推进且 SAC 数值稳定，但从当时状态到 Gate 的课程下界约 116k，原 300k 上限只剩 100k。因而“累计 300k 未过 Gate即失败”混淆了学习停滞与预算不足。V10 不回溯改写 V8/V9 的历史结论，但正式协议改为：Block 仅是 100k 的分析单位；每 stage 统一硬上限 600k；每 25k 保存 checkpoint；每 100k 完整分析。

新增 `stage_total_step` 贯穿 checkpoint、summary 及五类训练 CSV，用于跨 run/中断恢复审计累计 stage 预算。新增 `scripts/analyze_thesis_plateau.py`，只比较两个相邻完整 S0 Block 的同一 `eta` 数据，并按配置中的最小样本量和实质改善阈值输出 `continue`、`plateau_stop`、`numerical_failure`、`run_validation` 或 `hard_budget_stop`。Gate eligible 只表示应运行 validation；三个固定 validation seed 对同一 checkpoint 全部通过才成功。该规则对所有正式训练 seed 和后续消融相同。

### 补充文献审查：动态惩罚与六自由度机械臂奖励

2026-09-17 补充审查 `动态惩罚.pdf` 与 `奖励函数.pdf`。动态惩罚论文支持从低到高增加约束权重，但其基于历史最大 critic loss 的倍增触发会受 SAC 熵温度、replay 分布和 reward 重标影响，不能替代 V10 的任务成功率、锚点和 probe 门槛。自由漂浮机械臂论文中的四元数姿态、关节速度平方、平滑和碰撞机制已由 V10 等价或更严格实现；基座动量耦合不适用于固定基座 UR5，未标定 observation 白噪声不加入 nominal。训练契约保持不变；同时修复文档与评价器的指标缺口，validation 现在输出 command/measured velocity、acceleration、jerk 的逐 episode RMS/peak 和分场景 mean/p95/max，无需重新训练 checkpoint。
