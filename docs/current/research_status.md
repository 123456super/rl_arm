# 当前研究状态

> 更新时间：2026-08-05。本页是项目当前状态的唯一入口。

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

## 当前阻塞

- **基础到达策略未通过**：三个冻结 B4 V0 actor 在关闭障碍物和安全过滤器的同一 200-seed final 清单上共运行 600 回合，仅成功 31 次（5.17%）；569 次跑满 12 s 仍未到达，且零 physical contact。即使在 IK 可达、无障碍候选路径找到的 567 个 actor--episode 中，也只成功 31 次（5.47%）。这说明当前 actor 的基础无障碍到达能力不足，不能把后续动态障碍失败主要归因于安全约束、动态轨迹或可行性标签。
- 当前 P3 比较未形成可冻结的主方法：B4 严格过滤器仍有不可行停止，B5 的不可行/安全停止占比更高；恢复放宽会恶化 physical contact。
- G2 v2 的冻结投影失败已完成数值归因：在 11 个原矩阵事件中，10 个在 50000 次 OSQP 迭代内被判为 primal infeasible，另 1 个只在 34625 次迭代后获得严格可行候选。这同时暴露严格约束集合的真实冲突和当前运行时迭代上限的数值不足，不能以单纯提高迭代上限消除问题。
- 完整方法 M 尚未实现；当前 replay buffer 仍保存策略原始动作，过滤器干预尚未进入训练代价。根据本轮冻结决策，不进入 M 的实现或评估。
- 实际感知噪声、观测时延和外参扰动尚未形成 OOD 数据。
- 求解器只有事后耗时检查，尚无硬截止保证；300 ms 通过不构成部署实时性结论。

## 问题清单

1. **基础策略问题（首要）**：当前冻结 B4 actor 在无障碍、无安全过滤器下的真实 success 仅为 5.17%，三位 train seed 分别为 3.5%、0.5%、11.5%，且 200 个目标中仅 27 个至少被一个 actor 到达。必须先恢复稳定的基础到达能力，不能直接把安全过滤器包装到一个未学会 reaching 的策略上。
2. **可行性问题**：动态障碍物漂移与预测屏障、关节加速度约束存在联合冲突，safe-stop 不能阻止障碍物继续接近。
3. **覆盖问题**：无条件 reset 的初始不安全率 B4/B5 为 3.5%/18.0%；共享可行集只适用于主比较，不能代表全 reset 分布。
4. **实时性问题**：当前 300 ms 是 simulator-side post-return 检查；缺少可中断求解、外部 watchdog 和端到端延迟预算。
5. **证据边界问题**：未完成协同训练、OOD 扰动和真机签核，不能宣称泛化、部署或绝对安全。

## 下一步

本轮只归档和引用冻结决策包，不再运行或扩展 recovery、M、OOD 或真机实验。严格 B4 仅保留为诊断基线；当前已确认基础无障碍到达能力不合格，不能进入 V2、动态避障性能比较或任何安全方法推进。任何后继方案都必须先单独定义安全目标与评估协议，不能沿用已拒绝的 recovery 分支。

已按基础策略问题新增独立的[基础 reaching 恢复协议](reaching_recovery_protocol.md)：仅在关闭障碍物和安全过滤器的条件下，使用三个新 train seeds 恢复并验证策略到达能力。它与 VAPS G3/G4 完全隔离，不授权动态避障、安全比较、OOD、真机或任何 recovery/relaxation 分支；只有达到协议门槛后才能重新审查后续安全研究。

已新增[后继安全协议](../design/successor_safety_protocol.md)，其方向为可行安全集感知的严格预测控制。G0 已完成：viability 标签、严格约束与 V0/V1 动作等价性测试通过。G1 的 1000-reset coverage audit 已完成，结果保存在 `outputs/vaps_g1/coverage_10001_11000_v3.json`：`certified_viable=83.3%`、严格模型不可行 `16.7%`、unknown/invalid/budget-stop 均为零；求解 P99 `9.86 ms`、最大 `214.95 ms`，仅构成离线筛选证据。确定性抽样的 20 条严格可行和 20 条严格不可行 reset trace 已完成单位、连杆、状态与 safe-stop 命令的结构复核；零自然样本类别由 G0 故障注入测试覆盖。2026-08-05 决议保持为 `approve_g1_g2`：下一步只授权使用既有 V0 actor 的 V0/V1 严格链路比较；仍不授权 V2 训练、checkpoint 选择、最终比较、OOD、真机或任何 recovery/relaxation 分支。

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

根据本轮明确请求，新增一项**冻结 actor 的无障碍到达能力诊断**：使用 G2 v2 已冻结的三个 B4 actor、同一 `9001--9200` final seed、原有随机目标和初始关节状态，关闭障碍物与安全过滤器，记录策略真实 `success`、最终位置误差和目标坐标。配置为 `configs/experiments/vaps/v3_no_obstacle_policy_eval.yaml`，汇总器为 `scripts/summarize_no_obstacle_reachability.py`。该诊断不训练、不选 checkpoint、不修改运行时控制，结果只回答“既有策略在无障碍下能否到达”，不授权 G3/G4，也不能替代动态障碍全分布主结果。

诊断结果已完成，汇总位于 `outputs/vaps_no_obstacle_policy_eval/summary.json`，逐 episode 结果位于 `outputs/vaps_no_obstacle_policy_eval/episodes_joined.csv`：

| train seed | episode | success | success rate | 平均最终误差 |
| ---: | ---: | ---: | ---: | ---: |
| 4108 | 200 | 7 | 3.5% | 0.582 m |
| 4109 | 200 | 1 | 0.5% | 0.803 m |
| 4110 | 200 | 23 | 11.5% | 0.585 m |
| pooled | 600 | 31 | 5.17% | 0.656 m |

其中 569/600 回合运行到 12 s 上限仍未成功，physical contact 和 termination collision 均为零；200 个 reset 中只有 27 个（13.5%）被至少一个 actor 成功到达。即使离线标签为“IK 可达 + 无障碍候选路径找到”的 189 个 reset，仍只有 27 个被至少一个 actor 到达。这是当前已定位的首要问题：**冻结 B4 基础策略没有学会稳定 reaching**。结论仅适用于这三个冻结的 B4 actor 和该无障碍观测设置；它不等于证明 SAC 或机械臂本身无法完成 reaching，也不应与动态障碍结果混合为安全指标。
