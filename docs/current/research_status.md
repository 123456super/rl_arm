# 当前研究状态

> 更新时间：2026-08-07。本页是项目当前状态的唯一入口。

## 研究方向

当前论文主题为：**面向感知不确定性动态障碍物的机械臂连杆级预测风险约束控制与协同安全强化学习**。

阶段一已经冻结为预研究基线。P3 的统一口径 B1--B5 已完成并形成冻结决策包，但**P3 方法本身未冻结**；完整协同方法 M、OOD 和真机验证均未启动。

## 进展总结

| 方面 | 已完成 | 当前结论 |
| --- | --- | --- |
| 阶段一基线 | held-out 复核、3 方法比较、材料归档 | 可作为历史预研究基线，不能与新 P3 数字混合 |
| 预测风险（P1） | 单元测试、单位修正、逐连杆 Jacobian 速度 | 实现链路成立，尚无泛化安全保证 |
| 安全过滤器（P2） | strict QP、投影/回退、safe-stop、审计链路 | 可运行但不是硬实时或安全证明 |
| P3 因子化比较 | B1--B5、3 train seeds、共享 final manifest | 完成诊断，未形成可冻结主方法 |
| 基础 reaching v1/v2 | 独立无障碍训练、checkpoint 选择、3×200 final 评估 | v2 基础 reaching actor 已冻结；全量 97.0%，固定可达子集 100% |
| 归档与决策 | 冻结包、recovery 对照、初始可行性审计 | 停止当前 recovery 分支扩张 |

## 2026-08-03 修正

1. 预测模型不再把关节速度上限 `rad/s` 直接作为连杆速度 `m/s`。
2. 预测中的逐连杆速度由胶囊最近点平移 Jacobian `m/rad` 乘实际关节速度 `rad/s` 得到。
3. 延迟裕度的连杆速度上界由当前姿态下胶囊端点 Jacobian和关节速度盒计算，单位为 `m/s`；这是控制周期内的局部运动学上界，不是全工作空间全局界。
4. 碰撞事件拆为胶囊包络重叠、PyBullet 物理接触、二者并集和实际终止事件。统一 P3 配置仅以物理接触终止，胶囊重叠继续作为保守风险事件统计。

这些修正改变预测裕度、回合长度、成功率、reward/cost 分布和 checkpoint 选择口径。因此所有 2026-07-31 及以前的 P3 数字只能作为失效诊断，不能与修正后的实验直接合并。

## 阶段状态

| 阶段 | 状态 | 可引用结论 |
| --- | --- | --- |
| P0 阶段一 | 已冻结 | 连杆级固定惩罚 SAC 在旧仿真口径下改善任务完成能力 |
| P1 预测风险 | 实现与测试完成，实验待重跑 | 单元级功能成立；旧 P3 性能数字作废为主结论 |
| P2 安全过滤器 | 实现与测试完成，实验待重跑 | 唯一命令出口和失效停止链路可运行，不构成安全保证 |
| P3 因子化实验 | 已完成、未冻结 | B1--B5 的统一比较完成；严格 B4 保留为诊断基线，不扩展 recovery |
| P4 OOD | 未开始 | 无 |
| P5 真机 | 未开始 | 无 |

## P3 冻结决策（2026-08-04）

- 决策包：`outputs/p3_postfix_dev_100k/p3_freeze_decision.json`。
- 决策：`do_not_freeze_p3_or_expand_recovery`。这不是 P3 方法冻结，而是停止当前预测 barrier recovery 分支的实验扩张。
- 共享最终清单：3 个 train seeds（4108/4109/4110）各 144 回合，共 432 回合；仿真侧耗时筛选阈值为 300 ms，且该检查不是硬实时截止。
- 严格 B4 基线为 4 次 physical contact、78 次 success；冻结 actor 后的 worst-link recovery 为 22 次 physical contact、79 次 success，故拒绝该 recovery。
- 无条件 reset 初始不安全覆盖率：B4 为 3.5%（7/200），B5 为 18.0%（36/200）；二者不混入共享可行初始集的主比较。

## 基础 reaching v2 冻结决策（2026-08-07）

- 独立协议 `docs/current/reaching_recovery_protocol.md` 已完成 v2 训练、checkpoint 选择和最终评估；该协议关闭动态障碍物、安全过滤器、viability monitor 和 recovery/relaxation 分支。
- 三个新 train seeds（`4301/4302/4303`）各完成 `300000` steps；选中 checkpoint 分别为 step `240000/280000/220000`，validation success 均为 `39/40=97.5%`。
- 固定 final manifest `9001--9200` 上，三个 actor 各为全量 `194/200=97.0%`、固定 IK 可达且无障碍候选路径子集 `194/194=100%`；pooled 为 `582/600=97.0%` 与 `582/582=100%`。
- 三个 actor 共同失败的 6 个 reset 为 `9021、9065、9095、9098、9120、9142`，均为 12 s 超时，collision、capsule overlap 和 physical contact 均为零。由于这 6 个 reset 的固定 IK 搜索未找到候选，全量口径不能诚实宣称 99%。
- 冻结范围仅为三个 v2 actor、配置、checkpoint 选择规则、validation manifest 和 final manifest；这不是 P3/VAPS 方法冻结，也不授权 G3/G4、OOD、真机或安全结论。

## 当前阻塞

- **历史 B4 基础 actor 诊断（已由独立 v2 基线补齐）**：三个冻结 B4 V0 actor 在同一无障碍 final 清单上共运行 600 回合，仅成功 31 次（5.17%）；该旧 actor 结果仍作为 P3/VAPS 失效诊断保留。独立 reaching recovery v2 已在新 actor 上达到全量 582/600（97.0%）和固定可达子集 582/582（100%），因此基础 reaching 基线门槛已解决，但不能把新基线结果写成 P3 安全性能。
- 当前 P3 比较未形成可冻结的主方法：B4 严格过滤器仍有不可行停止，B5 的不可行/安全停止占比更高；恢复放宽会恶化 physical contact。
- G2 v2 的冻结投影失败已完成数值归因：在 11 个原矩阵事件中，10 个在 50000 次 OSQP 迭代内被判为 primal infeasible，另 1 个只在 34625 次迭代后获得严格可行候选。这同时暴露严格约束集合的真实冲突和当前运行时迭代上限的数值不足，不能以单纯提高迭代上限消除问题。
- 完整方法 M 尚未实现；当前 replay buffer 仍保存策略原始动作，过滤器干预尚未进入训练代价。根据本轮冻结决策，不进入 M 的实现或评估。
- 实际感知噪声、观测时延和外参扰动尚未形成 OOD 数据。
- 求解器只有事后耗时检查，尚无硬截止保证；300 ms 通过不构成部署实时性结论。

## 问题清单

1. **基础策略问题（已解决但保留历史边界）**：旧冻结 B4 actor 在无障碍、无安全过滤器下的真实 success 为 31/600（5.17%）；该结果仍是 P3/VAPS 的历史诊断。新独立 reaching v2 基线为全量 582/600（97.0%）、固定可达子集 582/582（100%），可作为后续基础到达基线，但不等于安全过滤器或动态避障性能。
2. **可行性问题**：动态障碍物漂移与预测屏障、关节加速度约束存在联合冲突，safe-stop 不能阻止障碍物继续接近。
3. **覆盖问题**：无条件 reset 的初始不安全率 B4/B5 为 3.5%/18.0%；共享可行集只适用于主比较，不能代表全 reset 分布。
4. **实时性问题**：当前 300 ms 是 simulator-side post-return 检查；缺少可中断求解、外部 watchdog 和端到端延迟预算。
5. **证据边界问题**：未完成协同训练、OOD 扰动和真机签核，不能宣称泛化、部署或绝对安全。

## 下一步

基础 reaching v2 已完成并可冻结为独立无障碍到达基线；严格 B4 仍仅保留为 P3/VAPS 诊断基线。该进展不改变 `do_not_freeze_p3_or_expand_recovery` 决议：不得把 reaching v2 结果写成动态避障、安全、OOD、真机或完整方法 M 结果。任何后继安全方案仍必须遵循独立安全目标与评估协议，不能沿用已拒绝的 recovery 分支。

静态障碍物 S1 冻结 actor 已完成复核但未通过：pooled success `428/600=71.3%`，其中 `156/172` 个失败为 timeout，另有 4 次 capsule overlap 和 13 次 physical contact。按渐进协议，当前下一步是 S1-R 静态障碍物迁移训练：从三个 v2 selected actor 及其 SAC state 各继续 `200000` steps，仅加入零速度随机静态障碍物和 `fixed_risk_penalty=1.0`，保持安全过滤器、viability、recovery 和 relaxation 全部关闭。训练、选点和 final 评估命令见[后续渐进式实验协议](successor_incremental_experiment_protocol.md)；S1-R 未通过前不运行 S2 动态障碍物。

S1-R 训练、validation checkpoint 选择和 final 评估均已完成。选中 checkpoint 为：4301 step `240000`、4302 step `480000`、4303 step `320000`。final pooled success 为 `491/600=81.8%`，timeout `105`，capsule overlap `2`，physical contact `4`。相较冻结 S1 的 `428/600=71.3%`、timeout `156`、capsule overlap `4`、physical contact `13`，迁移训练有效改善静态障碍物表现；但 pooled success 仍低于无障碍 S0 的 `97.0%`，且 actor 4301 没有改善（仍 `174/200`）。S1-R 按原门槛通过，但由于静态候选子集仍未达到目标，当前暂停 S2，先进行 S1-R2 静态混合场景迁移训练。

S1-R2 actor-only 迁移已完成 validation：4301 选中起点 step `240000`（35/40=87.5%），4302 选中起点 step `480000`（30/40=75.0%，较 S1-R 起点下降 3 回合），4303 选中 step `580000`（33/40=82.5%，较起点增加 1 回合）。pooled validation 为 `98/120=81.7%`，低于 S1-R 起点 `100/120=83.3%`。该结果不能区分 mixed-static 覆盖和 dense reward 的真实效果，因为 actor-only 迁移同时重置了 critic/alpha/replay，并从较大 `start_step` 跳过了原 warmup，造成随机 critic 立即更新已训练 actor。当前不进入 S2；先做 S1-R2.1 匹配 SAC state 的迁移重跑。

S1-R2.1 三个 seed 使用同一步匹配的 actor 与 SAC critic/target/alpha state（旧 checkpoint 不含 optimizer moments，因此优化器仍重新初始化），并在 resumed training 中按本次迁移重新计数：前 `10000` 步收集新 replay、禁止更新，之后再开始 SAC 更新。4301 的 actor 与 S1-R 起点完全相同，因此使用 v2 的匹配 `agent_state_step_240000.pt`；4302/4303 使用 S1-R 对应的 `agent_state_step_480000.pt`/`agent_state_step_320000.pt`。S1-R2.1 仍保持静态 mixed 场景、零速度、`fixed_risk_penalty=2.0`，安全过滤器、动态障碍物、viability、recovery 和 relaxation 全部关闭。

S1-R2.1 已完成 validation 和 full final。validation 选中 step 为 4301=`520000`、4302=`480000`、4303=`620000`，分别为 `38/40=95.0%`、`30/40=75.0%`、`35/40=87.5%`，pooled `103/120=85.8%`。在完整 `9001--9200` 原始 random 静态障碍物 final 上，成功分别为 `168/200=84.0%`、`164/200=82.0%`、`165/200=82.5%`，pooled `497/600=82.8%`；相较 S1-R 的 `491/600=81.8%` 提升 6 个 episode。physical contact 从 `4` 降至 `1`，capsule overlap 保持 `2`，collision_any 从 `5` 降至 `2`。静态候选路径子集从 `434/483=89.9%` 提升到 `443/483=91.7%`，但仍不能宣称 99%。

该提升主要由 4303 贡献：`153/200→165/200`，候选子集 `136/161→148/161`；4302 选回完全相同的 S1-R 起点，结果不变；4301 full random 从 `174/200→168/200` 反而下降。配对 episode 为 `35` 个失败转成功、`29` 个成功转失败，说明不是所有 reset 都稳定改善。当前结论是 warm-start 修复有效地消除了 actor-only 的灾难性退化，但 mixed-static 覆盖和 dense reward 尚未形成跨 seed 稳定收益；不进入动态障碍物，下一步只针对 4301/4303 做静态随机分布上的低学习率/短迁移确认，并继续保留完整 final manifest 和所有失败 reset。

静态 S1-R 的离线有限候选预检查已完成，输出为 `outputs/reaching_incremental/s1_static_obstacle_finetune/static_feasibility_precheck.json`。200 个 reset 中 IK reachable 为 190、无障碍候选路径找到为 189、静态障碍物候选路径找到为 161。与三个 S1-R final CSV 按 reset 对齐后，静态候选子集 pooled success 为 `434/483=89.9%`；其中 49 个失败均为 12 s timeout，且该子集中无 capsule overlap 或 physical contact。由于预检查的 `not_found` 是有限搜索标签而非数学无解证明，且候选子集仍显著低于 99%，后续应优先改进静态到达/脱困训练与动作响应，不应把剩余失败全部剔除为“无解”。

固定动作响应诊断已完成：原始 `fixed_beta=0.35`、`0.50`、`0.65` 的全量 success 分别为 `491/600`、`485/600`、`489/600`；静态候选子集分别为 `434/483`、`428/483`、`432/483`。`beta=0.65` 仅改善 actor 4301，反而降低 4302/4303，不能作为统一修复。当前暂停 S2，优先进行带静态障碍物覆盖增强和 `fixed_risk_penalty=2.0` 的第二轮迁移训练；训练仍保持完整 final manifest，安全过滤器、动态速度、viability、recovery 和 relaxation 全部关闭。

独立的[基础 reaching 恢复协议](reaching_recovery_protocol.md)已完成 v2 验证并冻结基础 actor；它与 VAPS G3/G4 完全隔离。该协议只证明无障碍 reaching 执行链路恢复，不授权动态避障、安全比较、OOD、真机或任何 recovery/relaxation 分支。

已新增[后继安全协议](../design/successor_safety_protocol.md)，其方向为可行安全集感知的严格预测控制。G0 已完成：viability 标签、严格约束与 V0/V1 动作等价性测试通过。G1 的 1000-reset coverage audit 已完成，结果保存在 `outputs/vaps_g1/coverage_10001_11000_v3.json`：`certified_viable=83.3%`、严格模型不可行 `16.7%`、unknown/invalid/budget-stop 均为零；求解 P99 `9.86 ms`、最大 `214.95 ms`，仅构成离线筛选证据。确定性抽样的 20 条严格可行和 20 条严格不可行 reset trace 已完成单位、连杆、状态与 safe-stop 命令的结构复核；零自然样本类别由 G0 故障注入测试覆盖。2026-08-05 决议保持为 `approve_g1_g2`：下一步只授权使用既有 V0 actor 的 V0/V1 严格链路比较；仍不授权安全方法 V2 训练、checkpoint 选择、最终比较、OOD、真机或任何 recovery/relaxation 分支。

G2 v1 的 `4108` validation 在 `seed=8216, step=129` 出现单侧 `safe_stop_compute_budget`，因此整套 v1 结果未通过且保留在 `outputs/vaps_g2/`，不得用于推进决议。该回退由 post-return 墙钟计时触发，不具备锁步确定性；新建的 G2 v2 仅移除此计时诊断作为比较输入，保留全部严格 QP/几何/速度/加速度约束、无效观测停机和不可行 safe-stop。G1 的 `.30 s` 时间审计仍有效且不可由 G2 替代；v2 结果完成独立审计前，仍不得进入 G3。

G2 v2 的六个固定结果已通过完整性审计：`3×40` validation 与 `3×200` final episode 均完成，所有逐步 V0/V1 观测、动作、请求/执行命令、碰撞、终止和 filter status 差异均为零。trace 中同时出现 `11` 次双方一致的 `safe_stop_projection_failed`，已冻结这 11 个 `(train seed, reset seed, step)` 及原始 trace 哈希。定点回放确认 11/11 保持该状态、V0/V1 和 source 命令精确一致且均为零；零命令在 11/11 个事件中不满足全部瞬时约束，因此它只是拒绝原策略动作的 fail-safe，不能被解释为严格可行。

冻结矩阵的离线数值可行性归因已完成，结果为 `outputs/vaps_g2_v2/projection_feasibility_audit.json`。所有诊断均保留原始 box、predictive 与 workspace 约束，关闭 solver time limit、不开启 recovery/relaxation，并对返回候选独立复核最大残差：

| OSQP 最大迭代 | 11 个事件的结果 | 含义 |
| ---: | --- | --- |
| 2000 | 11 个 `indeterminate_max_iterations` | 该上限不足以区分可行与不可行 |
| 10000 | 9 个 `primal_infeasible`、2 个仍未定 | 多数不是低迭代上限造成的失败 |
| 50000 | 10 个 `primal_infeasible`、1 个 `strict_feasible` | 原严格约束集合同时存在真实不可行与数值收敛不足 |

唯一严格可行事件为 `(train_seed=4110, seed=9119, step=155)`：OSQP 在 34625 次迭代后返回 `solved`，原约束独立复核最大残差为 `1.60e-08`，低于 `1e-6` 容差。其余 10 个事件在 50000 次上限内获得 primal-infeasible 证书；证书向量出现的约 `2e9` 残差不是候选控制命令或物理违约幅度，不能作为执行量解释。

当前决议为 `do_not_advance`：不得进入 G3、不得训练 V2 或修改运行时控制。把运行时上限直接提高到 50000 只能消除 1 个数值性投影失败且计算代价不可接受，不能解决另 10 个严格不可行事件。当前只允许两项离线诊断：对这 10 个事件做原矩阵约束冲突归因；以及对 validation/final reset 生成目标 IK、无障碍候选任务路径和给定动态障碍轨迹下候选路径的三层标签。后者只解释目标/场景/局部控制的失败边界，不能按标签删除样本，也不授权 G3。两项诊断均不得放宽约束、启用 recovery 或把诊断候选用于环境执行。

## 三层任务可行性与 G2 执行事件对照（离线诊断）

三层预检查已经完成，并与既有 G2 v2 的三份 V0/V1 锁步 trace 按相同 reset seed 对齐。报告文件为：

- `outputs/vaps_task_feasibility_v1/final_vs_g2_execution_audit.json`（200 个 reset × 3 个 train seed = 600 个 episode）
- `outputs/vaps_task_feasibility_v1/validation_vs_g2_execution_audit.json`（40 个 reset × 3 个 train seed = 120 个 episode）

G2 trace 没有环境 `success` 字段，因此本节只报告 physical contact、capsule overlap、安全停机和投影失败；不能从这些文件推导策略真实到达率。

final 三层交叉分组的 pooled 结果如下：

| 分组 | episode | physical contact | safe-stop | 解释 |
| --- | ---: | ---: | ---: | --- |
| IK 可达 + 无障碍候选路径 + 动态候选路径找到 | 447 | 77（17.2%） | 227（50.8%） | 场景和候选路径均具备，但局部严格控制仍会不可行或碰撞 |
| IK 可达 + 无障碍候选路径找到 + 动态候选路径未找到 | 120 | 33（27.5%） | 97（80.8%） | 动态障碍轨迹是主要困难，不能把失败全归因于策略 |
| IK 可达 + 无障碍候选路径未找到 | 3 | 0（0.0%） | 1（33.3%） | 无障碍候选搜索本身没有找到路径，样本很少 |
| IK 搜索未找到 | 30 | 10（33.3%） | 21（70.0%） | 目标可能过远/过高，也可能只是有限 IK 搜索未命中 |

上述结果是解释性证据，不是成功率，也不表示 `not_found` 是数学上的绝对无解。P3 正式评估的 144-seed 清单为 `7127--7300`，与本次 G2 final 的 `9001--9200` 不同，不能直接和三层标签拼接；如果论文需要“无障碍策略真实到达率”，必须另行使用同一 seed manifest 跑正式评估。

历史 B4 actor 的无障碍到达能力诊断使用 G2 v2 已冻结的三个 B4 actor、同一 `9001--9200` final seed、原有随机目标和初始关节状态，关闭障碍物与安全过滤器，记录策略真实 `success`、最终位置误差和目标坐标。配置为 `configs/experiments/vaps/v3_no_obstacle_policy_eval.yaml`，汇总器为 `scripts/summarize_no_obstacle_reachability.py`。该诊断不训练、不选 checkpoint、不修改运行时控制，结果只回答旧 actor 在无障碍下能否到达；它已由独立 reaching recovery v2 基线补充，但不能替代动态障碍全分布主结果。

诊断结果已完成，汇总位于 `outputs/vaps_no_obstacle_policy_eval/summary.json`，逐 episode 结果位于 `outputs/vaps_no_obstacle_policy_eval/episodes_joined.csv`：

| train seed | episode | success | success rate | 平均最终误差 |
| ---: | ---: | ---: | ---: | ---: |
| 4108 | 200 | 7 | 3.5% | 0.582 m |
| 4109 | 200 | 1 | 0.5% | 0.803 m |
| 4110 | 200 | 23 | 11.5% | 0.585 m |
| pooled | 600 | 31 | 5.17% | 0.656 m |

其中 569/600 回合运行到 12 s 上限仍未成功，physical contact 和 termination collision 均为零；200 个 reset 中只有 27 个（13.5%）被至少一个 actor 成功到达。即使离线标签为“IK 可达 + 无障碍候选路径找到”的 189 个 reset，仍只有 27 个被至少一个 actor 到达。该段是旧冻结 B4 actor 的历史失效证据；独立 reaching recovery v2 已在同一 final manifest 上达到全量 582/600（97.0%）和固定可达子集 582/582（100%）。两者不能混为同一 actor 或同一方法结论。
