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

### S1-R 静态障碍物迁移训练结果

S1-R 使用独立 validation manifest 选点后，在完整 `9001--9200` final manifest 上完成评估。三个选中 checkpoint 分别为 4301 step `240000`、4302 step `480000`、4303 step `320000`。

| actor | selected step | success | timeout | capsule overlap | physical contact | mean final error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4301 | 240000 | 174/200 = 87.0% | 22 | 1 | 3 | 0.0660 m |
| 4302 | 480000 | 164/200 = 82.0% | 36 | 1 | 1 | 0.0795 m |
| 4303 | 320000 | 153/200 = 76.5% | 47 | 0 | 0 | 0.1014 m |
| pooled | — | 491/600 = 81.8% | 105 | 2 | 4 | 0.0823 m |

相较冻结 S1 的 pooled `428/600=71.3%`、timeout `156`、capsule overlap `4`、physical contact `13`，S1-R 成功数增加 `63`，timeout 减少 `51`，physical contact 减少 `9`。因此静态障碍物迁移训练通过预设 S1-R 门槛，但仍明显低于无障碍 S0 的 `582/600=97.0%`。4301 选中了迁移起点，说明该 seed 的训练没有带来收益；总体改善主要来自 4302 和 4303。

数据源：`outputs/reaching_incremental/s1_static_obstacle_finetune/eval/seed_430{1,2,3}_final.csv`。

### S1-R2 actor-only 迁移结果：不作为最终改进证据

S1-R2 使用 `mixed_static` 和 `fixed_risk_penalty=2.0`，但三组均通过 `--reset-agent-state` 只加载 actor，重新初始化 critic/target/alpha/replay。validation 结果如下：

| seed | S1-R 起点 | S1-R2 选中点 | validation 结果 | physical contact / capsule overlap |
| ---: | ---: | ---: | ---: | ---: |
| 4301 | 35/40 = 87.5% | step 240000 | 35/40 = 87.5% | 0 / 0 |
| 4302 | 33/40 = 82.5% | step 480000 | 30/40 = 75.0% | 0 / 0 |
| 4303 | 32/40 = 80.0% | step 580000 | 33/40 = 82.5% | 0 / 0 |
| pooled | 100/120 = 83.3% | — | 98/120 = 81.7% | 0 / 0 |

4301/4302 的最佳点都是迁移起点，4303 仅增加 1 个成功。该轮不能作为 mixed-static 或 dense reward 的独立效果证据，因为 actor-only 恢复方式使随机 critic 在大 `start_step` 下立即参与更新，存在灾难性遗忘混杂因素。当前应先完成匹配 SAC state 的 S1-R2.1 重跑，再进行 `9001--9200` final 评估。

### S1-R2.1 匹配 SAC state 迁移结果

S1-R2.1 使用匹配的 actor/SAC state、`10000` 步 replay 收集和 `10000` 步 critic-only warmup。validation 只用于 mixed-static 选点，final 仍使用原始 random 静态障碍物分布和完整 `9001--9200` manifest。

| seed | selected step | validation | final success | final physical / capsule / any |
| ---: | ---: | ---: | ---: | ---: |
| 4301 | 520000 | 38/40 = 95.0% | 168/200 = 84.0% | 0 / 0 / 0 |
| 4302 | 480000 | 30/40 = 75.0% | 164/200 = 82.0% | 1 / 1 / 1 |
| 4303 | 620000 | 35/40 = 87.5% | 165/200 = 82.5% | 0 / 1 / 1 |
| pooled | — | 103/120 = 85.8% | 497/600 = 82.8% | 1 / 2 / 2 |

相较 S1-R pooled `491/600=81.8%`，成功增加 6 个，physical contact 从 4 降到 1；静态候选路径子集从 `434/483=89.9%` 提升到 `443/483=91.7%`。但 4301 从 `174/200` 降到 `168/200`，4302 完全回到 S1-R 起点，提升主要来自 4303 的 `153/200→165/200`。因此这是有方向但跨 seed 不稳定的改进，不能作为 99% 静态避障能力证据，也不授权进入动态障碍物。

### S1-R 静态可行性与失败归因（离线）

为区分“候选路径未找到”和策略执行失败，使用相同 `9001--9200` reset manifest、零速度障碍物运行了离线有限候选预检查，结果保存在 `outputs/reaching_incremental/s1_static_obstacle_finetune/static_feasibility_precheck.json`。该脚本不加载 actor、不执行策略、不修改运行时控制；`not_found` 仅表示有限搜索未找到 witness，不是数学上的无解证明。

| 标签 | reset 数 | S1-R pooled episode success |
| --- | ---: | ---: |
| IK reachable | 190/200 | 487/570 = 85.4% |
| 无障碍候选路径找到 | 189/200 | 484/567 = 85.4% |
| 静态障碍物候选路径找到 | 161/200 | 434/483 = 89.9% |
| 静态候选路径未找到（含 IK 未找到） | 39/200 | 53/117 = 45.3% |

在静态候选路径已找到的 483 个 pooled episode 中仍有 49 个超时失败，且没有 capsule overlap 或 physical contact；因此当前主要瓶颈是策略到达/脱困，而不是可以全部归因于无解静态场景。按 reset 统计，161 个静态候选 reset 中 116 个由三个 actor 全部成功，另有 45 个至少一个 actor 失败。当前不能诚实宣称“排除无解后 99%”：候选子集观测值为 89.9%，远低于 99%。

该预检查的 `required_clearance_m=0.16` 包含 `d_safe=0.12`、几何裕度 `0.03` 和跟踪误差界 `0.01`；静态速度为零时没有延迟漂移项。

### S1-R 动作响应诊断：`fixed_beta` 不构成稳定修复

在相同 S1-R selected actor、相同 `9001--9200` final manifest 和相同静态场景上，只把固定动作平滑系数改为 `0.50` 或 `0.65`，结果如下。三组均为 `3×200=600` episode；静态候选子集固定为 161 个 reset、483 个 episode。

| 设置 | 全量 success | 静态候选子集 success | 全量 timeout | capsule / physical contact |
| --- | ---: | ---: | ---: | ---: |
| S1-R 原始 `beta=0.35` | 491/600 = 81.8% | 434/483 = 89.9% | 105 | 2 / 4 |
| 诊断 `beta=0.50` | 485/600 = 80.8% | 428/483 = 88.6% | 111 | 2 / 3 |
| 诊断 `beta=0.65` | 489/600 = 81.5% | 432/483 = 89.4% | 107 | 2 / 4 |

配对到相同 actor/reset 后，`beta=0.50` 在候选子集出现 10 个原始成功变失败、4 个失败变成功；`beta=0.65` 出现 13 个原始成功变失败、11 个失败变成功。`beta=0.65` 虽使 actor 4301 从 `174/200` 提升到 `178/200`，但 4302 从 `164/200` 降至 `162/200`、4303 从 `153/200` 降至 `149/200`。因此动作响应只改变失败 reset 的分配，没有形成跨 seed 的稳定增益；不建议把 beta=0.50 或 0.65 直接作为下一轮统一训练配置。下一轮改为静态混合场景迁移训练：保留 50% 原始 random，另外 50% 均匀覆盖 upper-arm/elbow/forearm/wrist crossing，并把 `fixed_risk_penalty` 从 1.0 提高到 2.0；S2 暂停。

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
