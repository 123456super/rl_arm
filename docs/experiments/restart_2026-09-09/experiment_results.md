# 从零实验结果事实源

> 启动日期：2026-09-09  
> 当前状态：已有 R1 测量/几何校准证据；R2a--R2d validation 均完整结束但门禁失败，尚无可冻结的名义 actor；R2e 基础容量诊断待运行。  
> 本文件是重启后唯一的数据与统计事实源；归档目录中的旧数值不得复制为新结果。
> 方法、公式和论文结构见 [论文大纲](../../thesis/thesis_outline.md)；执行状态见 [从零实验跟踪](experiment_tracker.md)。

## 1. 数据准入规则

只有同时满足以下条件的数据才能写入本文件：

- 实验属于重启后的预注册阶段编号 `R0`--`R9`。
- 代码 commit、完整配置、checkpoint 哈希、环境版本和输出目录可追溯。
- train、validation、held-out seed 集合在运行前冻结并互不混用。
- `episode_seed` 一一映射，manifest 检查文件数、行数、重复单元和配对关系。
- 所有方法使用相同逐 240 Hz 子步状态、距离、接触、命令导数和反馈导数口径。
- validation 只用于参数选择；任何看过 held-out 后的修改必须新建实验版本，不得覆盖原结果。
- 失败、超时、NaN、solver failure、fallback 和急停均保留在分母中，不静默删除。

## 2. 冻结协议

R0 v1 已冻结大部分协议；源码提交与评估总 manifest 尚待完成：

```text
experiment_version: restart-2026-09-09-v1
git_commit: 2942a2c14e4d404db8614dd82b2e51a7d11ad003 + dirty source snapshot hashes
environment_lock: outputs/restart_2026-09-09/r0_protocol/environment_lock.txt
output_root: outputs/restart_2026-09-09/
train_seeds: development={11}, final={101,202,303}
validation_seeds: {1001,1002,1003}
heldout_seeds: {2001,2002,2003}
episode_seed_rule: CantorPair(base_seed, episode_index)
scenarios: {random, upper_arm_crossing, elbow_crossing, forearm_crossing, wrist_crossing}
primary_metrics: Success/Collision/SafetyViolation/MinimumDistance + command/measured smoothness + intervention/solver/fallback/time
acceptance_tolerances: R2 frozen in tracker; later stage tolerances pending preregistration
manifest_schema: restart_run_manifest_v1
```

## 3. 实验登记表

| 编号 | 日期 | 代码版本 | 配置 | 输出目录 | 完整性 | 状态 | 可用于论文 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1-geometry-diagnostic | 2026-09-09 | dirty snapshot | corrected capsule config | `outputs/restart_2026-09-09/r1_geometry/diagnostic_corrected_seed71002/` | 600 rows | diagnostic | 否 |
| R1-geometry-calibration | 2026-09-09 | dirty snapshot | corrected capsule config, seed 71003 | `outputs/restart_2026-09-09/r1_geometry/calibration_seed71003/` | 2400 rows | nominal gate passed | 仅方法校准证据 |
| R2a-penalty-training | 2026-09-09 | `2942a2c` + identical dirty source snapshots | penalty `{1,2,4,8}`, seed 11, 100000 steps | `outputs/restart_2026-09-09/r2a_penalty_selection/train/` | 4/4 complete manifests；每个 run 10 组 step checkpoints + final；日志 finite | training complete | 否 |
| R2a-checkpoint-validation | 2026-09-10 | frozen evaluation plan manifest | 20 candidates；6 scenarios；validation seeds `{1001,1002,1003}`；30 episodes/cell | `outputs/restart_2026-09-09/r2a_penalty_selection/validation/` | 360/360 files；10800/10800 rows；0 duplicate units；plan verified | gate failed | 否 |
| R2b-dropout-training | 2026-09-10 | dirty source snapshot | penalty 1；80% random crossing + 20% no obstacle；seed 11；100000 steps | `outputs/restart_2026-09-09/r2b_obstacle_dropout/train/penalty1_dropout20_dev_seed11_steps100000/` | complete manifest；10 组 step checkpoints + final；686 episodes；79.15% obstacle-enabled；日志 finite | training complete | 否 |
| R2b-checkpoint-validation | 2026-09-10 | frozen evaluation plan manifest | 5 candidates；6 scenarios；validation seeds `{1001,1002,1003}`；30 episodes/cell | `outputs/restart_2026-09-09/r2b_obstacle_dropout/validation/` | 90/90 files；2700/2700 rows；plan verified | gate failed | 否 |
| R2c-terminal-correction | 2026-09-10 | complete dirty source snapshot | R2b + `collision_penalty=100`；seed 11；100000 steps | `outputs/restart_2026-09-09/r2c_collision_terminal_penalty/train/penalty1_dropout20_collision100_dev_seed11_steps100000/` | complete manifest；10 组 step checkpoints + final；22 hashes；527 完整 episodes；日志 finite | training complete | 否 |
| R2c-checkpoint-validation | 2026-09-10 | frozen evaluation plan manifest | 5 candidates；6 scenarios；validation seeds `{1001,1002,1003}`；30 episodes/cell | `outputs/restart_2026-09-09/r2c_collision_terminal_penalty/validation/` | 90/90 files；2700/2700 rows；plan verified | gate failed | 否 |
| R2d-training-budget | 2026-09-10 | complete dirty source snapshot | R2c MDP；从头训练 150000 steps；只验证 110000--150000 | `outputs/restart_2026-09-09/r2d_training_budget_150k/train/penalty1_dropout20_collision100_dev_seed11_steps150000/` | complete manifest；15 组 step checkpoints + final；32 hashes；824 完整 episodes；日志 finite | training complete | 否 |
| R2d-checkpoint-validation | 2026-09-10 | frozen evaluation plan manifest | 5 candidates；6 scenarios；validation seeds `{1001,1002,1003}`；30 episodes/cell | `outputs/restart_2026-09-09/r2d_training_budget_150k/validation/` | 90/90 files；2700/2700 rows；plan verified | gate failed | 否 |
| R2e-no-obstacle-capacity | 2026-09-10 | preregistered diagnostic | no obstacle；相同 SAC/RTB/reaching task；seed 11；100000 steps | `outputs/restart_2026-09-09/r2e_no_obstacle_capacity/` | 尚未运行 | diagnostic ready | 否 |

## 4. 已确认结果

R1 校准在每个胶囊 400 个样本、共 2400 个近表面分层样本上得到：胶囊距离减 PyBullet collision-shape 距离的 p50=-0.00780 m、p95=0.00905 m、p99=0.01617 m、最大单侧误差 0.03513 m。原始 `d_safe=0.12 m` 分类有 15 个危险漏检（0.939%）；应用预先冻结的单侧 `m_geom=0.04 m` 后为 0。PyBullet contact 仍作为每物理子步碰撞事实源，不能用胶囊零间隙替代。

逐子步确定性测试与当前完整测试集通过：49 tests passed。该结论只证明名义执行测量门禁和已登记的环境修复，不证明策略性能或后续 QP 有效。

## 5. 失败与无效运行

初始胶囊设计存在系统性错误：`upper_arm` 胶囊接近零长度，后续胶囊整体错位，且 offset 未按 link 姿态变换。600 点修复前诊断出现危险漏检 38 次（9.27%）、胶囊碰撞漏报 57 次（30.3%），因此旧胶囊及其全部旧结果无效。诊断输出保留在 `outputs/restart_2026-09-09/r1_geometry/diagnostic_seed71000/`。另有两个配置引用已重命名的中间诊断目录，均不作为准入证据。

R2 启动前代码审计还发现两处协议实现偏差，并在任何 R2 长训练开始前修复：其一，环境曾把胶囊近似间隙 `<=0` 与 PyBullet contact 合并为碰撞终止，这与“PyBullet contact 是碰撞事实源”的 R1 结论冲突；修复后胶囊间隙仅进入安全距离违反与 risk cost，碰撞只由逐子步 contact 判定。其二，训练入口曾把时间上限 `truncated` 写成 SAC replay 的 terminal mask；修复后仅碰撞/成功产生的 `terminated` 屏蔽 Bellman bootstrap。此前只有 12-step 入口 smoke run 使用旧语义，不构成策略性能证据；R2 正式运行不得复用它。

R2a checkpoint validation 完整通过数据门禁，但策略门禁失败。20 个候选中，无障碍 Success Rate 达到 `0.80` 的数量为 0，五障碍场景宏平均 Success Rate 达到 `0.60` 的数量也为 0。最接近门槛的 `penalty1_step100000` 无障碍 Success Rate 为 0.0778、五场景宏平均 Success Rate 为 0.4800、Collision Rate 为 0.0844。按预注册协议不得降低门槛或选用该 checkpoint，须先诊断无障碍 observation 分布、任务奖励/终止和名义策略训练能力，再重新预注册 R2b。

R2b 前诊断表明，抽样目标经 PyBullet IK 可进入 `success_tolerance`，未发现目标空间整体不可达证据；主要实现缺口是 R2a 的训练 episode 始终启用障碍物，而无障碍门禁使用每连杆距离 1.5 m、零方向、零风险的占位 observation，形成 actor 未见的输入分布。R2b 因此不修改 `link_risk_v1`、奖励或门槛，只预注册 20% no-obstacle episode coverage，并先运行 penalty 1 单因素开发探针。该诊断不把 R2a 失败结果改写为正证据。

R2b checkpoint validation 数据完整，但 5 个候选仍全部失败。step 100000 的无障碍 Success Rate 为 0.7000，五障碍场景宏平均 Success Rate 为 0.3711、Collision Rate 为 0.2644；全候选最高无障碍 Success Rate 为 0.7000，最高宏 Success Rate 为 0.3711。与 R2a 最优候选的无障碍 0.0778 相比，训练支持集修复确实改善了基础到达，但尚未达到 0.80，且障碍任务没有通过 0.60 门槛，因此不得冻结 checkpoint。

进一步按 R2b 后 50000 训练步的真实 episode 终局拆分：障碍启用时，124 个 collision episodes 平均长度 50.99、平均回报 -72.42；112 个 timeout episodes 平均长度 240、平均回报 -125.84；74 个 success episodes 平均回报 -18.53。当前 `-w_p ||e||^2` 逐步回报、碰撞立即终止和仅 2 分任务侧碰撞惩罚的组合，使困难状态下提前碰撞可规避大量后续负回报。R2c 将此判定为原任务 MDP 的 terminal-shortcut 设计缺陷，而非继续改变 dropout：只把 `collision_penalty` 预注册为 100，其余 R2b 因素不变。该修改是名义策略训练正确性修复，不属于论文核心创新。

R2c 训练在 37.64 分钟内完成 100000 steps，manifest、22 个 actor/agent-state 哈希及数值日志完整；527 个完整 episodes 中障碍物启用比例为 0.7913。后 50000 步的障碍场景中，collision/success/timeout 分别为 7/76/139 个；collision episode 平均回报降至 -229.02，不再优于 timeout 的 -130.67。该训练内诊断支持 terminal shortcut 已被压制，但不能替代预注册 validation，也尚不能证明 R2 门禁通过。

R2c validation 的 90/90 文件和 2700/2700 episodes 完整，但所有 checkpoints 仍未通过门禁。step 100000 达到无障碍 Success Rate 0.6556、五障碍宏 Success Rate 0.4800、宏 Collision Rate 0.0067。分障碍场景的 450 episodes 中成功 214、碰撞 3、其余未成功 233；失败已主要表现为到时未到达，而非碰撞。与此同时，宏 Success Rate 从 step 60000 的 0.2444 总体升至 step 100000 的 0.4800，无障碍从 0.1444 升至 0.6556。因此 R2d 只检验“100k 预算不足”：保持 R2c MDP 和全部门槛，从头训练 150000 steps，预注册只评价 110000--150000 checkpoints；这是最后一次仅凭未收敛趋势延长预算。

R2d 从头训练在 56.88 分钟内完成 150000 steps，manifest、32 个 actor/agent-state 哈希及数值日志完整；824 个完整 episodes 中障碍物启用比例为 0.7888。R2d step 100000 actor 的 SHA-256 与 R2c step 100000 完全一致，证明预算字段未改变前 100k 的随机训练轨迹。110k--150k 的训练内障碍成功率在 0.3261--0.5400 间波动，不能据此选择最后 checkpoint，须执行已冻结的独立 validation。

R2d validation 的 90/90 文件和 2700/2700 episodes 完整，但 5 个新增 checkpoints 全部失败。最高无障碍 Success Rate 为 step 140000 的 0.5222；最高五障碍宏 Success Rate 为 step 150000 的 0.4600（同 checkpoint Collision Rate 0.0044）。延长训练没有延续 R2c step 100000 的 0.6556/0.4800，表现为策略波动而非单调未收敛，因此禁止再次仅增加 steps。

随后对无障碍任务做独立可行性审计：全部 90 个 validation targets 中，PyBullet IK 有 88 个可进入 0.055 m 阈值，p95 IK 位置误差为 0.0186 m；相同 RTB/rate-limit 执行下的阻尼最小二乘控制器抽查 10 episodes 为 10/10 成功。这排除了工作空间或 12 s 时限足以解释当前约半数失败的假设。R2e 将只检验 SAC 在无障碍到达任务上的容量，不作为正式 actor 结果。

## 6. 当前结论边界

截至本文件建立时：

- 没有任何重启后的基线、预测、QP、鲁棒性或实机结果；
- R2a--R2d 开发训练及 validation 已完整结束且均失败；R2b/R2c 分别验证了训练覆盖和碰撞终止修复方向，但都不构成合格策略性能证据；
- R2a--R2d 是指南形成前的历史诊断链，不得表述为已经依次完成 R2-A、R2-C 或 R2-D；补正后的正式顺序从 R2e=`R2-A` 开始；
- 旧实验不能证明当前论文主线中的任何数值结论；
- 只有新 tracker 中相应门控通过且在本文件登记的数据，才能进入论文正文。
