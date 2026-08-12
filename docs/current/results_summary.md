# 实验结果总表

> 更新时间：2026-08-11。结果按“正式证据、独立冻结基线、开发证据、失效诊断、待完成”分层。

## 统一报告口径

- 静态障碍物 full final 固定为 `9001--9200`、三个 actor 各 200 回合、一个 `random` 零速度障碍物、12 s 上限和 `success_tolerance=0.055 m`；安全过滤器及所有 recovery/relaxation 分支关闭。
- validation 固定为 `9301--9340`，只用于 checkpoint selection。没有运行 full final 的阶段只报告 validation，不外推为 final 性能。
- “静态候选子集”固定为有限搜索找到静态候选路径的 161 个 reset，即 pooled 483 个 actor×reset episode；`not_found` 不是数学无解证明，也不是删除 final reset 的依据。
- collision 必须分别报告 `collision_any`、capsule overlap 和 physical contact；三者可能重叠，不能直接相加。
- S1-R2.1 统一称为“同一步部分训练状态 warm-start”。旧 checkpoint 没有 target critic、optimizer、replay、训练步数或 RNG，不能称为完整 SAC 恢复。
- 表中数字是观测结果；只有排除恢复状态、数据分布和选择偏差后，才能写成某个 reward、learning rate 或采样机制的因果结论。

## 独立冻结基线：基础 reaching recovery v2

该协议与 P3/VAPS 安全方法完全隔离：关闭动态障碍物、安全过滤器、viability monitor 和 recovery/relaxation 分支，固定使用 `9001--9200` final manifest。三个新 train seeds 均完成 `300000` steps，并按独立 40-seed validation manifest 选择 checkpoint。

| train seed | selected step | final 全量 success | final 可达子集 success | collision_any / capsule overlap / physical contact |
| ---: | ---: | ---: | ---: | ---: |
| 4301 | 240000 | 194/200 = 97.0% | 194/194 = 100% | 0 / 0 / 0 |
| 4302 | 280000 | 194/200 = 97.0% | 194/194 = 100% | 0 / 0 / 0 |
| 4303 | 220000 | 194/200 = 97.0% | 194/194 = 100% | 0 / 0 / 0 |
| pooled | — | 582/600 = 97.0% | 582/582 = 100% | 0 / 0 / 0 |

三个 actor 共同失败的 reset 为 `9021、9065、9095、9098、9120、9142`，均为 12 s 超时。固定 IK 搜索在这些 reset 上未找到候选，因此全量结果不能宣称 99%；报告必须同时保留全量 `97.0%` 和条件可达 `100%`。

冻结范围：三个 v2 actor、配置、checkpoint selection 规则、validation manifest 和 final manifest。该冻结只表示基础 reaching 基线可复现，不表示 P3/VAPS 安全方法、动态避障、泛化、OOD、真机或完整方法 M 已冻结。

数据源：`outputs/reaching_recovery_v2/eval/seed_430{1,2,3}_final.csv`、对应 `checkpoint_selection/selected_checkpoint.csv` 和 [基础 reaching 恢复协议](reaching_recovery_protocol.md)。

## S1 静态障碍物阶段总览

| 阶段 | 状态 | full final success | timeout | collision_any / capsule / physical | 静态候选子集 | 当前解释 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| S1 冻结 actor | 已完成、未通过 | 428/600 = 71.3% | 156 | 16 / 4 / 13 | 未统计 | 需要静态场景适应 |
| S1-R | 已完成、通过原门槛 | 491/600 = 81.8% | 105 | 5 / 2 / 4 | 434/483 = 89.9% | 相对 S1 有效，仍远低于 99% |
| S1-R2 | validation 诊断 | 未运行 | — | — | — | actor-only 恢复存在随机 critic 混杂 |
| S1-R2.1 | 旧 final 参考 | 497/600 = 82.8% | 101 | 2 / 2 / 1 | 443/483 = 91.7% | 小幅总体提升，跨 seed 不稳定 |
| S1-R2.2 | 未产生新 actor | 与 R2.1 相同 | 与 R2.1 相同 | 与 R2.1 相同 | 与 R2.1 相同 | 只否定旧恢复链路下的该组低学习率设置 |
| S1-R3 | validation 未通过且受混杂 | 未运行 | — | — | — | 不能证明 terminal progress 无效 |
| S1-R4 | 已完成、当前 final 参考 | 516/600 = 86.0% | 84 | 0 / 0 / 0 | 461/483 = 95.4% | 方向有效但未达 99%；剩余失败集中在策略终端收敛/回退和非候选 reset 边界 |

## S1 静态障碍物：冻结 actor 未通过

在相同 `9001--9200` final manifest、随机零速度障碍物、关闭安全过滤器的条件下，冻结 v2 actor 的 pooled success 为 `428/600=71.3%`。失败中 `156/172` 为 timeout；另有 `4` 次 capsule overlap 和 `13` 次 physical contact。该分布说明主要问题是策略没有稳定学会绕开静态障碍物，而不是 safety filter/QP 造成的失败。

| actor | success | timeout | collision_any | capsule overlap | physical contact |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4301 | 174/200 | 22 | 4 | 1 | 3 |
| 4302 | 106/200 | 89 | 5 | 1 | 4 |
| 4303 | 148/200 | 45 | 7 | 2 | 6 |
| pooled | 428/600 | 156 | 16 | 4 | 13 |

S1-R 已按该诊断完成，后续结果见下文。当前进度已经推进到 S1-R4 final 分析完成；S2 动态障碍物仍暂停。当前下一步按三步走：终端伺服诊断、failure-neighborhood jitter 训练和静态 waypoint teacher 导出。命令归档和停止规则见[后续渐进式实验协议](successor_incremental_experiment_protocol.md)。

### S1-R 静态障碍物迁移训练结果

S1-R 使用独立 validation manifest 选点后，在完整 `9001--9200` final manifest 上完成评估。三个选中 checkpoint 分别为 4301 step `240000`、4302 step `480000`、4303 step `320000`。

| actor | selected step | success | timeout | collision_any | capsule overlap | physical contact | mean final error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4301 | 240000 | 174/200 = 87.0% | 22 | 4 | 1 | 3 | 0.0660 m |
| 4302 | 480000 | 164/200 = 82.0% | 36 | 1 | 1 | 1 | 0.0795 m |
| 4303 | 320000 | 153/200 = 76.5% | 47 | 0 | 0 | 0 | 0.1014 m |
| pooled | — | 491/600 = 81.8% | 105 | 5 | 2 | 4 | 0.0823 m |

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

4301/4302 的最佳点都是迁移起点，4303 仅增加 1 个成功。该轮不能作为 mixed-static 或 dense reward 的独立效果证据，因为 actor-only 恢复方式使随机 critic 在大 `start_step` 下立即参与更新，存在灾难性遗忘混杂因素。当时据此执行了 S1-R2.1；该后续阶段现已完成。

### S1-R2.1 同一步部分训练状态 warm-start 结果

S1-R2.1 使用同一步 actor 和旧 `agent_state` 中存在的 online critic、alpha、lambda、cost EMA，并执行 `10000` 步 replay 收集和 `10000` 步 critic-only warmup。旧 checkpoint 不含 target critic、optimizer、replay、训练步数或 RNG，因此本轮是部分状态 warm-start，不是完整 SAC 恢复。validation 只用于 mixed-static 选点，final 仍使用原始 random 静态障碍物分布和完整 `9001--9200` manifest。

| seed | selected step | validation | final success | final collision_any / capsule / physical |
| ---: | ---: | ---: | ---: | ---: |
| 4301 | 520000 | 38/40 = 95.0% | 168/200 = 84.0% | 0 / 0 / 0 |
| 4302 | 480000 | 30/40 = 75.0% | 164/200 = 82.0% | 1 / 1 / 1 |
| 4303 | 620000 | 35/40 = 87.5% | 165/200 = 82.5% | 1 / 1 / 0 |
| pooled | — | 103/120 = 85.8% | 497/600 = 82.8% | 2 / 2 / 1 |

相较 S1-R pooled `491/600=81.8%`，成功增加 6 个，physical contact 从 4 降到 1；静态候选路径子集从 `434/483=89.9%` 提升到 `443/483=91.7%`。但 4301 从 `174/200` 降到 `168/200`，4302 完全回到 S1-R 起点，提升主要来自 4303 的 `153/200→165/200`。因此这是有方向但跨 seed 不稳定的改进，不能作为 99% 静态避障能力证据，也不授权进入动态障碍物。

### S1-R2.2 静态 random 低学习率短迁移：无新增收益

为检查 S1-R2.1 的不稳定性是否可能仅由更新幅度造成，4301 和 4303 从各自 selected checkpoint 继续 `100000` steps；训练场景改为原始 random 静态分布，actor/critic/alpha learning rate 降为 `5e-5/1e-4/5e-5`，其余沿用同一步部分状态 warm-start、静态零速度和关闭过滤器。checkpoint selection 把迁移起点 actor 一并纳入候选，避免被迫选择退化的新 checkpoint。

| seed | 起点 validation | 最佳继续点 validation | 选择结果 | final |
| ---: | ---: | ---: | --- | ---: |
| 4301 | 35/40 = 87.5% | step 540000, 35/40 = 87.5% | 起点 step 520000 | 168/200 = 84.0% |
| 4303 | 32/40 = 80.0% | step 640000, 32/40 = 80.0% | 起点 step 620000 | 165/200 = 82.5% |

其余继续训练 checkpoint 均退化：4301 为 `24/40--31/40`，4303 为 `26/40--31/40`，后者还在 step `680000`、`720000` 出现 validation collision。最终选中 actor 与 S1-R2.1 相同，因此同一 final reset 的 success、reward 和轨迹指标逐行一致。该结果否定的是旧部分恢复链路下这组低学习率配置，不能外推为修复后的 SAC/replay 续训必然无效。

### S1 静态失败轨迹归因

数据源为 `outputs/reaching_incremental/s1_static_failure_diagnostics/report.json`、`actor_episode_diagnostics.csv` 和 `reset_diagnostics.csv`。该离线报告连接 S1-R2.1 的完整 final CSV/trace 与零速度有限候选路径预检查；候选路径标签仅是 witness 搜索结果，不是数学可解性的证明，所有失败 reset 均保留。

| 范围 | episode | success | timeout | collision_any | 关键失败形态 |
| --- | ---: | ---: | ---: | ---: | --- |
| 全量三 actor | 600 | 497 | 101 | 2 | nonconvergent timeout 44；near-goal timeout/regression 43；low-motion stall 14 |
| 静态候选路径找到 | 483 | 443 | 40 | 0 | near-goal timeout 21；near-goal regression 11；其余 8 |

候选路径失败的 32/40 在轨迹中至少进入 `0.08 m` 目标误差内，但未满足原始 `0.055 m` 成功条件。`near-goal timeout` 的平均末段关节速度范数仅 `0.147 rad/s`、平均最终误差 `0.083 m`；`near-goal regression` 曾到达平均 `0.066 m`，随后回退到平均 `0.272 m`。该证据把训练目标聚焦到终端收敛和回退控制；它不支持放松碰撞口径、success threshold 或删除失败 reset。

固定的 36-reset 候选失败并集清单为 `configs/experiments/reaching_incremental/manifests/s1_static_candidate_failure_union_v1.json`。S1-D1 仅把这些 reset 的时间上限从 12 s 扩展到 24 s，用于判定失败是时间窗不足还是策略停滞；它不是 validation/final manifest，不能选 checkpoint 或报告为全量性能。

### S1-D1 候选失败时间窗诊断：时间窗不是主因

在固定 36-reset 诊断清单上，三个 S1-R2.1 selected actor 仅将 `max_episode_steps` 从 `240` 改为 `480`，即从 12 s 改为 24 s；checkpoint、静态 random 观测、成功阈值和碰撞定义均未改变。原 12 s 的 108 条诊断 episode 中有 40 条 timeout，延长时间后仅救回 2 条。

| actor | 原 12 s success | 原 timeout | 24 s success | 原 timeout 转成功 | 24 s 仍 timeout | 新 collision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4301 | 22/36 | 14 | 22/36 | 0 | 14 | 0 |
| 4302 | 23/36 | 13 | 25/36 | 2 | 11 | 0 |
| 4303 | 23/36 | 13 | 23/36 | 0 | 13 | 0 |
| pooled | 68/108 | 40 | 70/108 | 2 | 38 | 0 |

两个延迟成功 reset 仅属于 actor 4302：`9116` 在 `16.95 s` 成功、`9200` 在 `14.25 s` 成功。其余失败均完整运行至 24 s，故该结果拒绝“时间窗不足是主因”的假设，也不能用延长 episode 来报告原 12 s 任务的 success 提升。该诊断随后用于设计 R3 的终端到达 reward；R3 因恢复状态混杂，未形成该 reward 的有效性结论。

### S1-R3 终端到达 shaping：validation 未通过，恢复状态混杂

S1-R3 仅增加 `0.12 m` 终端区域内的有符号 progress reward（`w_terminal_progress=30.0`），其余保持 S1-R2.1 不变。checkpoint selection 显式纳入迁移起点：

| seed | 起点 validation | 训练后最佳点 | selected step | 训练后退化范围 |
| ---: | ---: | ---: | ---: | ---: |
| 4301 | 38/40 = 95.0% | step 540000, 38/40 = 95.0% | 520000 | 40.0%--67.5% |
| 4302 | 30/40 = 75.0% | step 500000, 30/40 = 75.0% | 480000 | 20.0%--45.0% |
| 4303 | 35/40 = 87.5% | step 640000, 35/40 = 87.5% | 620000 | 7.5%--75.0% |

三个 selected actor 都是原 S1-R2.1 起点，因此不产生新的 final 证据。后续 checkpoint 审计确认，本轮在修改 reward 后仍加载了旧 reward critic，而且旧 checkpoint 不含 target critic、optimizer、replay 或训练 RNG；因此该结果只能说明原恢复方式发生退化，不能单独证明 terminal progress reward 无效。R3 不进入动态障碍物，也不作为新 final 结果引用。

### S1-R4 训练基础设施修复与静态终端迁移结果

已完成的修复包括：完整 SAC/optimizer/target critic/RNG checkpoint，压缩 replay 保存恢复，reward 与完整 risk 配置签名校验，不兼容时自动重置对应 critic，time-limit truncation 保持 Bellman bootstrap，以及成功/失败轨迹标记、分层采样和 checkpoint step 一致性检查。真实的三个旧起点 checkpoint 已验证均被识别为 `reward_signature_match=False`、`cost_signature_match=True`，故 R4 不会再让旧 reward critic 学习新 terminal reward。

R4 从 R2.1 selected actor 的 step `520000/480000/620000` 各继续 `200000` steps。由于旧 checkpoint 没有 replay 和 optimizer，第一轮固定使用 `50000` 步 collect-only 加 `50000` 步 critic-only 重建；actor/alpha learning rate 为 `1e-5`，actor output anchor 权重为 `10.0`。训练环境保持 static random 和零障碍物速度，25% episode 复现固定 36 个候选失败 reset，75% 保持原随机覆盖；50% replay batch 配额按成功/失败完整轨迹分层。

R4 已完成训练、checkpoint selection、完整 final 评估和失败诊断。validation 选中 4301 step `660000`、4302 step `620000`、4303 step `740000`，validation success 分别为 `36/40=90.0%`、`33/40=82.5%`、`33/40=82.5%`。完整 final 结果如下：

| actor | selected step | success | timeout | collision_any | capsule overlap | physical contact | mean final error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4301 | 660000 | 176/200 = 88.0% | 24 | 0 | 0 | 0 | 0.0714 m |
| 4302 | 620000 | 173/200 = 86.5% | 27 | 0 | 0 | 0 | 0.0738 m |
| 4303 | 740000 | 167/200 = 83.5% | 33 | 0 | 0 | 0 | 0.0821 m |
| pooled | — | 516/600 = 86.0% | 84 | 0 | 0 | 0 | 0.0758 m |

相较 S1-R2.1，R4 full final 从 `497/600=82.8%` 提升到 `516/600=86.0%`，collision_any/capsule/physical 从 `2/2/1` 降为 `0/0/0`；静态候选路径子集从 `443/483=91.7%` 提升到 `461/483=95.4%`。配对到相同 actor/reset 后，R4 相比 R2.1 救回 27 个 episode，同时丢失 8 个原本成功 episode，净增 19 个成功；相比 S1-R 净增 25 个成功。因此 R4 证明修复后的训练链路和终端/失败样本强化方向有效，但仍不能声称静态障碍物达到 99%，也不是 focused reset、stratified replay、anchor 或 terminal reward 的单因素贡献证明。

R4 当前失败共 84 条，全部为 timeout 且无碰撞：`nonconvergent_timeout` 42、`near_goal_timeout` 16、`near_goal_regression` 13、`low_motion_stall` 13。静态候选路径找到的 483 条 episode 中仍有 22 条失败，全部无碰撞，其中 `near_goal_timeout` 9、`near_goal_regression` 9、`nonconvergent_timeout` 4；候选路径找到的失败 reset 没有三 actor 全部失败的情况，说明这部分更像策略局部速度场和终端收敛问题，而不是场景硬不可解。另有 16 个 reset 为三 actor 全部 timeout，均属于 `not_found` 或 `not_checked_due_to_ik`，应作为目标/场景可行性边界与策略问题分开报告。

当前基于 R4 的统一判断是：静态障碍物失败已经从碰撞问题收敛为 actor 自身在局部观测下的速度场问题。由于评估关闭 safety filter，失败不是过滤器拦截或执行器没有跟随命令；候选子集的主要瓶颈是进入 `0.055--0.08 m` 附近后停滞或回退，非候选 reset 的主要瓶颈是目标更远/更高、障碍物更贴近目标或有限 IK/候选路径未找到。当前不进入 S2，下一步按三步走：终端伺服诊断、failure-neighborhood jitter 训练和静态 waypoint teacher 导出。

数据源：`outputs/reaching_incremental/s1_static_terminal_repaired_finetune/eval/seed_430{1,2,3}_final.csv`、`outputs/reaching_incremental/s1_static_terminal_repaired_finetune/diagnostics/report.json`、`actor_episode_diagnostics.csv` 和 `reset_diagnostics.csv`。

### S1-R 静态可行性与失败归因（离线）

为区分“候选路径未找到”和策略执行失败，使用相同 `9001--9200` reset manifest、零速度障碍物运行了离线有限候选预检查，结果保存在 `outputs/reaching_incremental/s1_static_obstacle_finetune/static_feasibility_precheck.json`。该脚本不加载 actor、不执行策略、不修改运行时控制；`not_found` 仅表示有限搜索未找到 witness，不是数学上的无解证明。

| 标签 | reset 数 | S1-R pooled episode success |
| --- | ---: | ---: |
| IK reachable | 190/200 | 487/570 = 85.4% |
| 无障碍候选路径找到 | 189/200 | 484/567 = 85.4% |
| 静态障碍物候选路径找到 | 161/200 | 434/483 = 89.9% |
| 静态候选路径未找到（含 IK 未找到） | 39/200 | 53/117 = 45.3% |

在 S1-R 的静态候选路径已找到的 483 个 pooled episode 中仍有 49 个超时失败，且没有 capsule overlap 或 physical contact；因此当时主要瓶颈是策略到达/脱困，而不是可以全部归因于无解静态场景。R4 后候选子集提升到 `461/483=95.4%`，但仍低于 99%；因此当前仍不能诚实宣称“排除无解后 99%”。候选路径标签只能用于解释失败边界，不能用于删除 final reset 或放宽 success threshold。

该预检查的 `required_clearance_m=0.16` 包含 `d_safe=0.12`、几何裕度 `0.03` 和跟踪误差界 `0.01`；静态速度为零时没有延迟漂移项。

### S1-R 动作响应诊断：`fixed_beta` 不构成稳定修复

在相同 S1-R selected actor、相同 `9001--9200` final manifest 和相同静态场景上，只把固定动作平滑系数改为 `0.50` 或 `0.65`，结果如下。三组均为 `3×200=600` episode；静态候选子集固定为 161 个 reset、483 个 episode。

| 设置 | 全量 success | 静态候选子集 success | 全量 timeout | capsule / physical contact |
| --- | ---: | ---: | ---: | ---: |
| S1-R 原始 `beta=0.35` | 491/600 = 81.8% | 434/483 = 89.9% | 105 | 2 / 4 |
| 诊断 `beta=0.50` | 485/600 = 80.8% | 428/483 = 88.6% | 111 | 2 / 3 |
| 诊断 `beta=0.65` | 489/600 = 81.5% | 432/483 = 89.4% | 107 | 2 / 4 |

配对到相同 actor/reset 后，`beta=0.50` 在候选子集出现 10 个原始成功变失败、4 个失败变成功；`beta=0.65` 出现 13 个原始成功变失败、11 个失败变成功。`beta=0.65` 虽使 actor 4301 从 `174/200` 提升到 `178/200`，但 4302 从 `164/200` 降至 `162/200`、4303 从 `153/200` 降至 `149/200`。因此动作响应只改变失败 reset 的分配，没有形成跨 seed 的稳定增益；不建议把 beta=0.50 或 0.65 作为统一修复。该诊断之后的 R2/R3/R4 已执行完毕，当前进度统一为 R4 final 分析完成，S2 继续暂停。

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
