# 从 legacy direct-SAC 到分层 Residual 架构的证据链

## 1. 结论

历史结果足以支持“停止把主要精力放在 direct-SAC 参数搜索，改为拆分全局可达、名义收敛、局部学习修正和最终安全约束”。它们不构成新方法性能证明；新架构仍必须在同一 fixed manifest 上完成 nominal-only、Residual 成对消融和动态鲁棒性实验。

## 2. 历史性能事实

### S0 无障碍基线

固定 final manifest 上，三个 actor 均为 full-set `194/200=97.0%`，pooled `582/600=97.0%`；有限 IK 候选子集均为 `194/194=100%`，pooled `582/582=100%`。六个共同失败 reset 为 `9021、9065、9095、9098、9120、9142`。

这说明旧 actor 在有限预检查标记为可达的无障碍任务上具备强基线，但 full-set 中仍混入 IK 候选搜索失败，不能把所有失败都解释为策略控制失败。

### S1 静态障碍物

| 阶段 | success | timeout | collision_any | 可得结论 |
| --- | ---: | ---: | ---: | --- |
| frozen S0 actor 静态迁移 | `428/600=71.33%` | 156 | 16 | 无障碍 actor 不能直接承担被挡路径与避障 |
| S1 fine-tune | `501/600=83.50%` | 99 | 0 | 训练可消除该批碰撞，但仍留下明显到达失败 |
| extend +100k | `501/600=83.50%` | 99 | 0 | 单纯增加训练步数没有 full-final 收益 |
| candidate terminal refine 历史最好 | `521/600=86.83%` | 79 | 0 | terminal shaping 只修复了一部分失败 |
| candidate terminal refine 当前复跑 | `511/600=85.17%` | 89 | 0 | 改动/复跑没有稳定超过历史最好 |
| R3/R4 blind | `505/600=84.17%` | 95 | 0 | 结果可复核，但不能作 curriculum 因果结论 |

历史最好来自[旧协议快照](../documents/s1_static_obstacle_protocol_2026-08-18.md)，当前复跑来自[归档 JSON](../results/s1_static_candidate_terminal_refine_current_rerun_summary.json)。两条记录不得合并。

## 3. 失败结构，而不只是成功率

当前 511/600 复跑的 89 个失败分为：

| failure mode | 数量 | 对控制职责的含义 |
| --- | ---: | --- |
| nonconvergent timeout | 36 | actor 没有稳定进入有效路径或收敛 basin |
| near-goal regression | 23 | 曾接近目标但无法保持终端收敛 |
| near-goal timeout | 16 | 终端推进不足 |
| low-motion stall | 14 | 动作不足或任务与避障目标相互抵消 |

其中 67/89 的最终误差 `>=0.12 m`。因此仅调 success threshold、terminal radius 或末端小范围增益，最多覆盖 22 个低于 `0.12 m` 的失败，不能解释大多数远端不收敛。

### 为什么参数修补已经到达决策边界

- 延长 100k steps 后 full-final 仍为 `501/600`，没有收益。
- terminal shaping 能修复部分 near-goal timeout，但覆盖不了全局 basin、被挡直连和大误差 nonconvergent。
- 参数与恢复路径改动后的复跑从历史 `521/600` 回退到 `511/600`，没有形成稳定改进。
- seed 4302 长期明显落后，说明同一任务分布上的 direct actor 解对训练随机性敏感。
- 碰撞降为零后 timeout 仍高，说明到达与安全之间的职责冲突没有靠 reward tuning 消失。

这些事实不证明任何参数都不可能再提高某一 manifest，但足以说明继续搜索参数不是建立“静态到动态可扩展方法”的优先路线。

## 4. Feasibility 结果揭示了什么

旧 precheck 在 200 个 reset 上得到：175 个静态障碍直连候选找到、15 个直连未找到、10 个 IK 未找到。它最多尝试有限 IK 候选，并检查若干关节空间三次曲线直连，且文件本身声明 `candidate_search_is_not_completeness_proof=true`。

因此正确解释是：

- 10 个 `IK not found` 支持扩展候选生成与验证，即多初值 IK；不证明目标数学不可达。
- 15 个 `path not found` 支持引入真正的配置空间搜索，即 RRT-Connect；不证明完整规划也会失败。
- full-set 指标必须保留，同时另报 IK-found、plan-found 和 plan-conditioned success，避免通过条件筛选抬高结果。

## 5. R3/R4 为什么不能被过度解释

R3/R4 配置均设置 `env.residual_control.enabled=false`，所以历史 timeout 不能归因于 legacy residual hold 或 link-avoidance 分支。两条恢复训练又先经过 20k collect-only 和 20k critic-only；selected checkpoint 仅在恢复后 `+10k`，actor 尚未进入 repair 更新。对应 R3/R4 selected actor 逐 tensor 相同，blind episode 结果相同是 checkpoint 身份的预期结果。

这段历史真正支持的是：后续 Residual 贡献必须在同一 planner/tracker/safety 基座上，用零 residual 与训练 residual 做成对消融，并验证实际加载的 actor signature 和更新步数。它不支持“curriculum 已被证明无效”。

## 6. 历史证据到新模块的映射

| 历史观测 | 新架构职责 | 新实验必须报告 |
| --- | --- | --- |
| 有限搜索中 10/200 IK 未找到 | 多初值 IK 与严格候选有效性检查 | IK found、候选数、TCP error |
| 15/200 障碍物直连未找到 | RRT-Connect 全局绕行 | direct/RRT plan found、planning time、clearance |
| 36 nonconvergent + 14 stall | waypoint tracker 提供稳定名义推进 | tracking error、path progress、plan-conditioned success |
| 23 regression + 16 near-goal timeout | terminal DLS servo 单独负责末端收敛 | servo 进入次数、退出/回退、terminal success |
| 静态碰撞可降为零但 timeout 持续 | 到达与安全职责解耦 | nominal 候选、过滤后命令、intervention rate |
| 动态障碍会使静态路径暂时或持续失效 | predictive risk + active `AVOID_HOLD` + event-triggered `REPLAN` | 状态占比、min predicted margin、replan success |
| R3/R4 actor 身份混淆 | risk-conditioned Residual 独立成对消融 | actor signature、residual norm、相对 nominal 增益 |

对应的冻结链路是：

```text
多初值 IK + RRT-Connect
        ↓
joint waypoint tracker + terminal DLS servo
        ↓
risk-conditioned Residual SAC
        ↓
link-level robust predictive risk + QP/CBF safety filter
        ↓
qdot_cmd
```

这个映射的论文价值不在于把经典模块并列，而在于提出并验证明确的职责边界：RL 只做受风险限制的局部 residual；所有候选命令经过同一连杆级预测安全出口；短时冲突主动恢复，持续路径失效再重规划。

## 7. 可以与不可以据此声称的结论

历史证据可以支持：

- direct-SAC 的剩余瓶颈不是一个已知参数即可解释，继续盲目调参优先级低。
- 新架构每个核心模块都对应已观测的独立失败类型，而非任意堆叠。
- 静态与动态应共用控制层次，并通过状态机把局部恢复与全局重规划分开。
- 论文应以职责分解、连杆级预测风险条件 Residual、最终安全过滤和事件触发协同为方法贡献，并逐项消融。

历史证据不能支持：

- 新架构已经超过历史 `86.83%`；尚需固定 manifest 的正式 hierarchical final。
- dynamic obstacle 下已经有良好性能；尚需 S2 nominal-only 和 Residual 结果。
- QP/CBF 对整个混合状态机具有严格递归安全保证；`RECOVERY_RELAXED` 明确不属于严格 barrier 可行命令。
- RRT、DLS、SAC 或 QP 单个经典模块本身构成硕士论文创新。

## 8. 对后续实验的约束

1. 先在固定 S1 manifest 上验证 nominal-only 的 IK、规划、跟踪和 servo，不用 Residual 掩盖基座问题。
2. 达到 plan-conditioned success 门槛后，在同一任务和安全层上训练 Residual，并与零 residual 成对比较。
3. 动态 S2 只改变障碍物运动与预测不确定性，不再更换 planner、tracker、动作语义、状态机或安全出口。
4. 最终论文同时保留 legacy direct-SAC 表、新 hierarchical 表和模块消融；旧结果只作架构调整前基线。
