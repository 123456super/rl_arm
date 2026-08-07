# 实验结果总表

> 更新时间：2026-08-07。结果按“正式证据、独立冻结基线、开发证据、失效诊断、待完成”分层。

## 独立冻结基线：基础 reaching recovery v2

该协议与 P3/VAPS 安全方法完全隔离：关闭动态障碍物、安全过滤器、viability monitor 和 recovery/relaxation 分支，固定使用 `9001--9200` final manifest。三个新 train seeds 均完成 `300000` steps，并按独立 40-seed validation manifest 选择 checkpoint。

| train seed | selected step | final 全量 success | final 可达子集 success | collision / capsule overlap / physical contact |
| ---: | ---: | ---: | ---: | ---: |
| 4301 | 240000 | 194/200 = 97.0% | 194/194 = 100% | 0 / 0 / 0 |
| 4302 | 280000 | 194/200 = 97.0% | 194/194 = 100% | 0 / 0 / 0 |
| 4303 | 220000 | 194/200 = 97.0% | 194/194 = 100% | 0 / 0 / 0 |
| pooled | — | 582/600 = 97.0% | 582/582 = 100% | 0 / 0 / 0 |

三个 actor 共同失败的 reset 为 `9021、9065、9095、9098、9120、9142`，均为 12 s 超时。固定 IK 搜索在这些 reset 上未找到候选，因此全量结果不能宣称 99%；报告必须同时保留全量 `97.0%` 和条件可达 `100%`。

冻结范围：三个 v2 actor、配置、checkpoint selection 规则、validation manifest 和 final manifest。该冻结只表示基础 reaching 基线可复现，不表示 P3/VAPS 安全方法、动态避障、泛化、OOD、真机或完整方法 M 已冻结。

数据源：`outputs/reaching_recovery_v2/eval/seed_430{1,2,3}_final.csv`、对应 `checkpoint_selection/selected_checkpoint.csv` 和 [基础 reaching 恢复协议](reaching_recovery_protocol.md)。

## S1 静态障碍物：冻结 actor 未通过

在相同 `9001--9200` final manifest、随机零速度障碍物、关闭安全过滤器的条件下，冻结 v2 actor 的 pooled success 为 `428/600=71.3%`。失败中 `156/172` 为 timeout；另有 `4` 次 capsule overlap 和 `13` 次 physical contact。该分布说明主要问题是策略没有稳定学会绕开静态障碍物，而不是 safety filter/QP 造成的失败。

| actor | success | timeout | capsule overlap | physical contact |
| ---: | ---: | ---: | ---: | ---: |
| 4301 | 174/200 | 22 | 1 | 3 |
| 4302 | 106/200 | 89 | 1 | 4 |
| 4303 | 148/200 | 45 | 2 | 6 |
| pooled | 428/600 | 156 | 4 | 13 |

当前处理路径是 S1-R 静态障碍物迁移训练：从三个 v2 selected actor 及 SAC state 各继续 `200000` steps，只加入零速度随机障碍物和 `fixed_risk_penalty=1.0`，安全过滤器、viability、recovery、relaxation 全部关闭。S1-R 完成前不进入动态障碍物 S2。命令和停止规则见[后续渐进式实验协议](successor_incremental_experiment_protocol.md)。

## 历史失效诊断：冻结 B4 基础到达策略

为区分“动态障碍/安全过滤器导致失败”和“策略本身不会 reaching”，对 G2 v2 固定的三个 B4 actor（train seeds `4108/4109/4110`）进行了无障碍实际执行诊断。固定 final reset 清单 `9001--9200`，保留原目标和初始关节状态，关闭障碍物与安全过滤器；每个 actor 运行 200 回合。结果为策略真实 `success`，不是 IK 或候选路径存在率。

| train seed | episode | success | success rate | 平均最终误差 | 到 12 s 上限仍未成功 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4108 | 200 | 7 | 3.5% | 0.582 m | 193 |
| 4109 | 200 | 1 | 0.5% | 0.803 m | 199 |
| 4110 | 200 | 23 | 11.5% | 0.585 m | 177 |
| pooled | 600 | 31 | 5.17% | 0.656 m | 569 |

- 三个 actor 均无 physical contact，说明失败形式是长期未到达目标，不是碰撞终止。
- 200 个目标中仅 27 个（13.5%）被至少一个 actor 到达；而离线预检查在 189/200 个 reset 中找到 IK 和无障碍候选路径。
- 该结果仍作为旧 B4 actor 的 P3/VAPS 失效诊断保留；基础 reaching 门槛已由独立 v2 基线重新验证，但不得把 v2 的无障碍结果与 P3 动态安全指标混合。

数据源：`outputs/vaps_no_obstacle_policy_eval/summary.json` 和 `outputs/vaps_no_obstacle_policy_eval/episodes_joined.csv`。本结论仅针对这三个冻结 B4 actor 在无障碍观测下的表现；它不等于 SAC、UR5 或目标空间在数学上不可达。

## 可作为正式基线

阶段一 held-out 数据位于 `outputs/rechecks/heldout_1004_1006/final_3methods/`，覆盖 3 train seeds、5 场景、3 方法、3 eval seeds，共 13,500 episodes。

| 方法 | 宏平均 success | 宏平均 collision | 使用边界 |
| --- | ---: | ---: | --- |
| `ee_fixed` | 16.4% | 7.69% | 末端风险弱基线 |
| `link_fixed_penalty1` | 66.7% | 6.51% | 阶段一综合候选 |
| `ldrc_fixed` | 59.3% | 5.91% | 部分场景碰撞率略低 |

数据源：`outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_macro_across_train_seeds.csv`。

这些数字沿用旧 `collision = capsule_overlap OR pybullet_contact` 终止定义，只能作为阶段一预研究基线，不能与新 P3 物理接触终止结果直接比较。

## 仅作为开发证据

`outputs/p3_dev/` 和 `outputs/p3_dev_100k/` 完成了 B1--B5 单 train seed 链路检查。100k 开发评估曾出现 B3 45% success/0% collision、B5 50%/0%，但使用旧预测速度、旧碰撞终止和后续确认不一致的几何包络。

结论仅限于：B1--B5 训练、checkpoint 选择、评估和过滤器指标链路可运行。不得据此声称预测风险、鲁棒裕度或过滤器带来稳定收益。

## P3 修正口径冻结包：未形成方法冻结

数据源：`outputs/p3_postfix_dev_100k/p3_freeze_decision.json`。主比较使用 B5 共享可行最终清单：3 个 train seeds（4108/4109/4110）、每个 144 回合，共 432 回合；耗时筛选阈值为 300 ms。下表为各 train seed 均值再汇总的均值，不应与阶段一旧口径结果直接比较。

| 方法 | success | capsule overlap | physical contact | 关键结论 |
| --- | ---: | ---: | ---: | --- |
| B1 `ee_current` | 20.37% | 6.02% | 4.86% | 无过滤器基线 |
| B2 `link_current` | 17.59% | 6.94% | 4.86% | 连杆当前风险未改善接触率 |
| B3 `link_predictive` | 11.57% | 2.08% | 1.16% | 接触较低，但任务性能下降 |
| B4 `predictive_nonrobust_filter` | 18.06% | 2.55% | 0.93% | 严格过滤诊断基线 |
| B5 `robust_predictive_filter` | 11.57% | 4.40% | 3.24% | 不可行/安全停止更高 |

B4 的严格 QP 在相同 432 回合上为 4 次 physical contact、78 次 success。冻结 actor 后启用 worst-link predictive-barrier relaxation recovery 虽得到 79 次 success，却产生 22 次 physical contact（并触发 recovery 343 次），因此明确拒绝。P3 决策为 `do_not_freeze_p3_or_expand_recovery`：严格 B4 可保留作诊断基线，但 P3 不作为已冻结主方法，也不据此开展 recovery、M、OOD 或真机实验。

无条件 reset 审计应与主比较分开报告：B4 初始不安全 7/200（3.5%），B5 为 36/200（18.0%）。共享可行初始集不消除这一覆盖问题。

## 仅作为失效诊断

| 目录 | 主要发现 | 使用方式 |
| --- | --- | --- |
| `outputs/p3_diagnostics/root_cause_2x2_v2/` | 动态漂移与严格几何共同放大不可行；38/39 collision 为 capsule-only | 解释旧设计失效 |
| `outputs/p3_diagnostics/root_cause_attribution/` | predictive barrier 与 joint acceleration 是主要联合冲突 | 约束设计依据 |
| `outputs/p3_diagnostics/speed_boundary/` | 障碍物速度提高时旧口径 collision/infeasible 总体上升 | 选择低速开发点 |
| `outputs/p3_diagnostics/deterministic_escape/` | success 4/80、旧口径 collision 34/80、最大求解 410.4 ms | 淘汰旧 escape 方案 |
| `outputs/p3_geometry_fix/` | 胶囊覆盖、frame/Jacobian 和 TTC 修复过程 | 几何历史审计 |

这些结果全部早于 2026-08-03 单位与碰撞口径修正，只能用于根因分析。

## 尚无结果／未授权扩展

- 完整协同方法 M、OOD、实时硬截止和真机均无可引用结果，且本轮冻结决策不授权继续这些实验。

## 引用规则

1. 阶段一论文数字引用阶段一正式基线目录。
2. 新论文主结果必须来自 2026-08-03 之后、记录代码版本和 `collision_termination_mode` 的新目录。
3. 新结果必须同时报告 physical contact、capsule overlap、termination collision，不再单独报告含义不明的 collision rate。
4. P3 当前只能引用冻结决策包中的“未冻结／拒绝 recovery”结论，不能声称获得可部署或已冻结的预测风险控制方法。
