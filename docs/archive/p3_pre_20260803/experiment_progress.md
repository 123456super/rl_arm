# 阶段一进展与新研究分支计划（项目状态更新至 2026-07-31）

> 文档版本与口径（2026-07-31）：本页按“日期—阶段—证据—结论边界”记录进度。阶段一正式主比较保持冻结；P1/P2 仅表示开发验证完成；P3 进入低速双速度统一配置训练阶段，实时耗时暂不作为本轮阻塞条件。

> **2026-08-03 说明：本文件改为只追加的历史时间线，不再代表当前状态。当前状态和可引用结果分别见 [current/research_status.md](../../current/research_status.md) 与 [current/results_summary.md](../../current/results_summary.md)。2026-07-31 及以前 P3 数字采用旧速度/碰撞口径，仅用于失效诊断。**

## 2026-08-03：速度单位与碰撞口径修正

- 连杆预测速度由胶囊最近点 Jacobian `m/rad` 乘实际关节速度 `rad/s` 得到 `m/s`，不再将 `action_scale` 直接作为线速度。
- 延迟裕度使用当前姿态下 Jacobian 和关节速度盒计算的局部 `max_link_speed_bound_mps`。
- 碰撞事件拆分为 capsule overlap、PyBullet contact、二者并集和实际终止事件；统一 P3 配置仅以物理接触终止。
- 因修正会改变预测裕度、episode 长度、reward/cost 和 checkpoint 选择，旧 P3 数据不再用于新方法性能比较；下一步按新协议重跑 B1--B5。

## 项目阶段进度总览（截至 2026-07-31）

| 阶段 | 时间/日期 | 当前状态 | 可引用内容 | 进入下一阶段的门槛 |
| --- | --- | --- | --- | --- |
| P0：阶段一基线 | 2026-07-27 至 2026-07-31 | 已冻结 | held-out 三方法主表；link_fixed_penalty1 为仿真候选 | 只允许复现和勘误，不再改写主结论 |
| P1：预测风险与不确定性 | 2026-07-29 前完成首轮开发验证 | 开发验证完成 | 几何、预测窗口、误差裕度和观测有效性测试 | 需要在冻结配置上补齐独立复核 |
| P2：仿真安全过滤器 | 2026-07-29 至 2026-07-31 | 开发验证完成，未构成安全保证 | QP/投影/OSQP、命令唯一出口、safe-stop 和 trace 诊断 | 先解决几何漏检、动态漂移和计算长尾 |
| P3：协同训练与安全失效分析 | 2026-07-29--07-31 | 当时诊断完成并计划统一训练；现已被 2026-08-03 口径修正取代 | B1--B5、mesh/frame 审计、V2 2x2、速度边界、漂移审计、OSQP 约束归因 | 仅作修正前历史诊断 |
| P4：独立复核与 OOD | 原计划后置 | 未开始 | 无 | P3 冻结后才可开始 |
| P5：低速真机 | 原计划后置 | 未开始 | 无 | P4 通过且现场安全签核完成 |

### 当时决策（2026-07-31，已由 2026-08-03 决策取代）

- 阶段一结果与 P3 诊断严格分开；P3 的任何 success/collision 数值不得回写阶段一主表。
- 当时计划启动一轮低速双速度统一配置训练/评估；该计划未产出结果，现按 2026-08-03 新口径重建基线。
- V2 2x2、速度边界、漂移审计和 OSQP 约束归因已完成；当前优先做统一几何/碰撞口径下的训练收敛、动态可达性和不可行原因分析。
- deterministic escape、relaxed recovery、bounded/maximin escape 和 mesh-fit 胶囊均只保留为离线失效/几何敏感性诊断，不得写成安全成功。

### 当前阻塞问题（2026-07-31）

1. 几何一致性：frame-corrected 顶点和三角形内部采样覆盖已通过，但 corrected/moving 的 capsule-only collision 显著增加，mesh/capsule 仍不能视为完整动态碰撞判据。
2. 动态漂移：V2 corrected/moving 的 dynamic drift 碰撞为 34/39（87.2%）；strict safe-stop 停止机器人后，障碍物仍会压缩安全裕度。
3. 联合不可行：1,823 个 infeasible steps 中 predictive barrier 出现 1,815 次、joint acceleration 出现 1,103 次，主因是两者联合可达性冲突，不是单纯 workspace/velocity 限制。
4. 实时预算（后置）：V2 mean 4.36--5.25 ms、P99 6.55--7.47 ms、max 197--351 ms；本轮只记录，不把它作为离线训练和非实时根因修复的阻塞条件。
5. 任务与证据规模：V2 success 仅 7--11/80（8.8%--13.8%），结果仍来自开发 checkpoint 和 80 episode 诊断，不足以支持跨 seed 泛化、P4/OOD 或真机结论。

## 0. 最新结果归档（截至 2026-07-31，基于 outputs）

### 当前结论

阶段一的 held-out 主比较已冻结，`link_fixed_penalty1` 仍是阶段一仿真部署候选。P3 的几何修正、严格安全停止和补充诊断均已完成一轮开发验证，但**P3 设计未冻结，不进入 P4/OOD、实时链路或真机**。本次结果只更新研究分支状态，不改变阶段一主表和既有论文结论。

### 已完成的最新诊断

UR5 胶囊映射已修正并补齐后续前臂/腕部段；以下是 eval seed `6101`、每个 train seed 20 episodes 的**历史**严格 OSQP safe-stop 复核：

| train seed | selected checkpoint | success | collision | safety violation steps | mean solve | max solve |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4102 | `actor_step_20000.pt` | 7/20 | 1/20 | 123 | 4.42 ms | 206.69 ms |
| 4103 | `actor_step_15000.pt` | 6/20 | 1/20 | 4 | 4.43 ms | 146.43 ms |

两次碰撞均为 PyBullet 实际接触，胶囊重叠均为 false；这说明该历史几何包络存在漏检。两组严格配置的平均求解时间较低，却仍出现 146--207 ms 的单步长尾，超过 50 ms 控制周期。后续 frame-corrected deterministic escape 的最新结果以本页后文和 `docs/archive/p3_pre_20260803/p3_failure_root_cause_analysis.md` 为准。

补充方案没有解决该问题：同一开发分支的 bounded projection（train seed 4103）为 5/20 success、2/20 collisions、137 个违反步；提前降速为 3/20 success、1/20 collision、132 个违反步，并出现 248.41 ms 最大求解时间。因此，bounded projection 和 preemptive speed scale 均只保留为故障诊断，不作为候选方案。

早期 11 个 `safe_stop_infeasible` trace 的审计表明，7 个静态或缓慢漂移案例没有碰撞，4 个动态漂移案例发生 3 次碰撞。该结果已被后续完整审计取代：58 个不可行停止中 34 个为 dynamic drift，碰撞 29 个。零速度安全停止无法阻止障碍物自身持续接近；这是当前的主要失效机理，而非策略命令投影可单独解决的问题。

### 当前问题与下一步

1. 上臂、前臂及腕部真实接触与胶囊距离模型仍存在不一致，需先完成离线碰撞几何变换审计和包络校准。
2. 动态障碍物漂移可在策略停止后继续压缩预测安全裕度；必须将静态不可行和动态漂移不可行分开建模、记录和评估，不能宣称 safe-stop 提供动态安全保证。
3. OSQP/主动集路径存在远超 50 ms 的长尾；现有 watchdog 在计算返回后才检查，不能构成实时截止保障。
4. 当前 2x20 仅为开发复核。只有在几何、动态漂移和计算预算问题得到可验证处理后，才能使用新的 train seeds 与独立 held-out eval seeds 再次评估；在此之前不启动 P4/OOD 或真机。

### 可追溯数据

```text
outputs/p3_geometry_fix/b4_seed4102_20k/strict_eval_seed6101_metrics.csv
outputs/p3_geometry_fix/b4_seed4103_20k/strict_eval_seed6101_metrics.csv
outputs/p3_geometry_fix/b4_seed4103_20k/projection_bounded_eval_seed6101_metrics.csv
outputs/p3_geometry_fix/b4_seed4103_20k/preemptive_speedscale_eval_seed6101_metrics.csv
outputs/p3_geometry_fix/safe_stop_drift_audit.json
```

### 2026-07-31 几何覆盖审计结果

对现有 UR5 collision mesh 做 256 个关节姿态采样后，**早期严格配置**的胶囊并未覆盖 mesh 顶点：`upper_arm_link` 所需最大半径约 `0.2056 m`，而配置为 `0.0600 m`；`forearm_link` 所需约 `0.1503 m`，而配置为 `0.0520 m`。这解释了先前 PyBullet 接触而胶囊重叠为 false 的现象。该结论已由后续 frame-corrected 几何审计补充，不能把早期配置称为当前配置。

已新增离线诊断配置 `configs/experiments/p3_diagnostics/b4_osqp_strict_mesh_capsules_heldout.yaml`。它使用 collision-mesh 顶点拟合的保守胶囊（附加约 3 mm 顶点余量），同样的 256 姿态审计中六个 link 均通过顶点覆盖检查。该检查仅证明采样顶点覆盖，不证明三角形表面、动态运动或实时安全；配置只用于下一轮离线 held-out 诊断，不得与历史 B4 数值混合。

当时的推荐顺序是先完成 mesh-fit 几何复核、再做 held-out safe-stop 和漂移分类；这些步骤已由后续 frame-corrected 几何审计和 deterministic escape 诊断推进。当前结论以本页后文的最新 4×20 复核和完整漂移审计为准，仍不训练、不进入 P4/OOD、不做真机测试。

### 2026-07-31 mesh-fit 胶囊 300 回合复核

使用 mesh-fit 胶囊配置、train seeds `4102/4103` 的已选 checkpoint、eval seeds `6301/6401/6501`，每组 50 episodes，共 300 episodes，结果如下：

| 指标 | 汇总结果 |
| --- | ---: |
| success | 19/300（6.3%） |
| collision | 105/300（35.0%） |
| safety violation steps | 4,283 |
| infeasible-stop episodes | 186/300（62.0%） |
| mean filter solve time | 4.99 ms |
| max filter solve time | 302.04 ms |

碰撞审计进一步显示：105 个碰撞 episode 中，82 个触发 capsule overlap，24 个触发 PyBullet contact，只有 1 个同时触发；因此当前 mesh-fit 胶囊既造成大量保守误报（81 个 capsule-only），仍有 23 个物理接触未被胶囊重叠捕获。顶点采样覆盖不等于三角面覆盖、动态接触覆盖或正确的 link-frame 变换，不能据此宣称几何安全。

safe-stop 漂移审计覆盖的 186 个不可行停止 episode 中，115 个被分类为 dynamic drift、71 个为 static_or_slow。动态漂移案例碰撞 89/115（77.4%），静态/缓慢漂移案例碰撞 16/71（22.5%）。换言之，约 85% 的审计碰撞来自动态漂移类别；零速度 safe-stop 在障碍物持续运动时不能保持安全集。

六组结果均出现明显任务退化：success 为 1--6/50（2%--12%），碰撞为 13--23/50（26%--46%）；其中一组最大过滤耗时 302.04 ms，全部超过当前 50 ms 控制周期上限。该结果不是可冻结的安全改进，mesh-fit 胶囊配置应淘汰为主方案，仅保留作几何敏感性/误报诊断。

**P3 决策：不冻结。** mesh-fit 结果只保留为几何敏感性/误报诊断，不再作为主方案。后续应继续做独立的 link-frame/三角面碰撞审计，以及带障碍物运动边界的离线失效分析；历史 mesh-fit 的物理接触漏检数值不能直接套用到后续 frame-corrected 配置。

### 2026-07-31 坐标系修正与三角面审计

复核发现，原 mesh-fit 生成流程将 collision shape 的惯性坐标系直接当作 link 坐标系，导致端点/半径与运行时 `getLinkState()[4]/[5]` 不一致。已新增 `scripts/audit_collision_frames.py`，显式执行 inertial-frame -> world -> link-frame 变换，并生成 `b4_osqp_strict_frame_corrected_heldout.yaml`。

修正后的 2048 姿态顶点覆盖和 64 姿态、每三角形 8 个内部采样点（约 3.3M 点）均通过，六个 link 的最大半径余量约为 3 mm。该结果只证明采样点在 capsule 内，不证明完整三角面、障碍物运动或控制实时性；因此下一步先用修正后的配置做小规模单回合 sanity check，再决定是否重做 held-out。

### 2026-07-31 坐标修正后的 sanity eval

使用 `b4_osqp_strict_frame_corrected_heldout.yaml`、train seed `4102` 的 step `20000` checkpoint 和 eval seed `6301` 完成 1 回合 sanity eval。几何审计本身通过：256 个姿态、每个三角形 8 个内部采样点，六个 link 的 `all_sampled_points_covered=true`，最大半径余量约 3 mm。

但策略执行未通过 sanity 门槛：回合在第 16 步结束，`success=0`，`collision=1`，`collision_capsule_overlap=1`，`collision_pybullet_contact=0`，最终误差 `0.2382 m`，最小 capsule 距离 `-0.00158 m`。过滤器干预率和 safe-stop 率均为 100%，不可行率为 93.75%，动作变化为 0；预测安全裕度从初始 `0.0490 m` 降至最小 `-0.1740 m`。平均过滤耗时 `23.98 ms`，最大耗时 `325.11 ms`，仍远超 50 ms 控制周期。

这说明坐标系修正消除了几何审计中的 frame mismatch，但没有解决控制链路的核心失效：预测约束很快不可行，机器人停止后障碍物仍继续运动并压缩裕度；同时 mesh-fit capsule overlap 会先于 PyBullet physical contact 触发，属于保守模型碰撞而非实际接触。该回合不能作为安全成功，也不能据此扩大 held-out 评估。

**当前决策保持不变：P3 不冻结。** 已完成的几何审计可作为模型一致性证据；sanity eval 暴露出的动态漂移、过高不可行率和 325 ms 长尾仍未解决。下一步只应做离线动态可达性/漂移边界分析和过滤器超时路径审计，不重新训练、不进入 P4/OOD、不做真机。

### 2026-07-31 TTC 提前逃逸仿真候选

针对动态漂移导致的零速度 safe-stop 失效，新增配置 `configs/experiments/p3_diagnostics/b4_osqp_ttc_escape_frame_corrected.yaml`。恢复模式现在基于候选命令下的安全裕度导数计算最小 TTC，在 `recovery_ttc_threshold_s` 内提前进入有界逃逸；逃逸仍通过 joint/workspace 约束滤波。评估 trace 新增 `unavoidable_collision`、`avoidable_collision` 和触发原因字段。

使用修正坐标系胶囊、train seed `4102` 的 `actor_step_20000.pt`、eval seed `6301` 做 1 回合 sanity：完整运行 240 步，`collision=0`、`capsule_overlap=0`、`pybullet_contact=0`、`safety_violation_steps=0`，平均滤波耗时 `7.30 ms`，动作变化恢复为非零；任务仍未到达目标（`success=0`，最终位置误差 `0.309 m`），且最大单步耗时 `379.18 ms`，超过 50 ms 控制周期。因此该结果只能标记为“仿真候选闭环通过”，不能作为实时安全或任务性能结论。

验证：安全滤波与环境集成测试 `25 passed`。全量测试 `49 passed, 1 failed`；唯一失败为既有 `tests/test_analytic_safety_jacobian.py`：第 6 个胶囊在 `p2_safety_filter_dev` 配置下解析 Jacobian 为零，而有限差分约 `0.45`，与 TTC 改动无关，需单独修复/确认几何模型后再冻结。

随后按同一配置完成 `20 episodes × 3 eval seeds`（6301/6401/6501）。成功率分别为 5%/20%/20%，胶囊碰撞率为 20%/15%/20%，平均安全违反步数为 20.0/15.05/14.65；平均求解时间为 7.17/6.28/7.23 ms，但最大长尾为 258/166/200 ms。三组均无 PyBullet 实际接触，碰撞均被标为 `avoidable_collision`，说明当前恢复/几何包络仍不足以作为仿真安全通过标准。

额外筛选了更保守的 `h=0.10 m`、TTC `0.30 s` 变体（seed 6301，20 回合），结果仍为 success 5%、collision 20%、平均安全违反 20.7 步；因此继续调 recovery 阈值不是当前最优方向。安全相关测试（排除既有 Jacobian 基准）为 `50 passed`。未通过 `max_solve <= 50 ms`、无碰撞且 success 不退化前，不进入 OOD、P4 或真机；下一步应改为固定时间预算的确定性逃逸/投影。

随后修复了固定 link 的 Jacobian：`_link_point_jacobian` 现在把 `tool0` 等固定链上的点转换到最近可动祖先 link，再计算 PyBullet Jacobian；退化为点胶囊的末端也补充了点 Jacobian。全量测试现为 `51 passed`。修复后的同一 TTC + relaxed recovery 复核为 success 5%、collision 35%、最大耗时 582 ms；关闭约束放宽的严格对照为 success 5%、collision 55%，其中 `unavoidable_collision=55%`、`avoidable_collision=0`，平均安全停止率 43%。这证实：严格停止对动态漂移无能为力，而放宽预测约束会产生可避免碰撞；两者都不能冻结为方案。

当前唯一推荐方向是实现固定时间预算的硬约束优先 escape controller（不可行时安全停止并分类审计），再重新跑仿真回归；该建议属于当时的阶段记录，后续 deterministic escape 复核未通过，因此当前只做根因隔离，不训练、不进入 OOD/P4、不做真机。

### 当前口径修正（2026-07-31）

上一节的 strict safe-stop、bounded escape 和“当前唯一推荐方向”段落属于当时的阶段性记录，已由后续 frame-corrected deterministic escape 复核取代。最新 4×20 结果为 success `4/80`、collision `34/80`、最大过滤耗时 `410.4 ms`；完整漂移审计为 58 个不可行停止，其中 dynamic drift 34 个、碰撞 29 个。当前只做根因隔离和失效分析，不把 deterministic escape 视为已通过方案；不可行停止按 `dynamic_drift`/`static_or_slow` 分类，不能预先全部标记为 `unavoidable_collision`。

### 2026-07-31 V2 根因隔离与审计结果（当前最新）

- V2 2x2：legacy/moving `success=11/80, collision=6/80`；corrected/moving `8/80, 39/80`；legacy/static `8/80, 0/80`；corrected/static `7/80, 0/80`。同一 corrected geometry 在 static 条件无碰撞，说明动态障碍物运动是关键交互因素。
- 漂移审计：legacy/moving 为 7 个 dynamic drift（碰撞 4）和 8 个 static/slow（碰撞 0）；corrected/moving 为 39 个 dynamic drift（碰撞 34，87.2%）和 20 个 static/slow（碰撞 5，25.0%）。
- 约束归因：1,823 个 infeasible steps 中 1,818 个可由单类别松弛解释；`predictive_barrier` 1,815 次、`joint_acceleration` 1,103 次，主因是联合可达性冲突。
- 时间与性能：四个条件 max filter time 为 197--351 ms，超过 50 ms；success 仅 7--11/80，不能冻结策略或实时方案。
- 现状与决策：评估、约束归因和审计均已完成；P3 为“诊断完成、设计未冻结”。当前不重训、不进入 P4/OOD、不做真机；下一步只做统一几何/碰撞口径、动态可达性和可中断/硬截止路径验证。

## 1. 当前状态

已完成的仿真实验包括：四种初始方法（`ee_fixed`、`link_fixed`、`ldrc_fixed`、`ldrc_adaptive`）的 100k 三-seed训练、validation seed 2001 的 checkpoint selection、五场景统一评估与三-seed汇总；`link_fixed` 固定风险惩罚、`C_safe` 和事件代价的小规模参数敏感性筛选；`link_fixed_penalty1`（`fixed_risk_penalty=1.0`）的 100k 三-seed重训、checkpoint selection，以及与 `ee_fixed`、`ldrc_fixed` 的五场景 eval-seed held-out 评估和统一汇总；并已完成 `ldrc_adaptive` actor / execution 反事实诊断。

最终 eval-seed held-out 主对比表明，`link_fixed_penalty1` 获得最高任务完成表现，并在安全距离违反率、最小距离、最终误差和 jerk 上优于 `ldrc_fixed`；`ldrc_fixed` 仅保留略低的平均碰撞率。`ldrc_adaptive` 没有稳定增益，不作为核心方法。

### 当前进度标记

**阶段一已闭环；P1/P2 已完成开发验证；P3 的 B1--B5、OSQP、recovery 与多连杆逃逸已完成开发诊断，但设计尚未冻结。几何修正后的 strict safe-stop 已完成一轮 2x20 episode 诊断，仍未达到冻结门槛。**

### 最新进度（2026-07-29，Recovery、动态漂移与 maximin 逃逸诊断）

在 B4 OSQP 诊断基础上，已增加显式 recovery mode：当预测安全裕度 `h_min <= 0` 时暂停策略任务推进，生成远离障碍物的逃逸关节速度，并在 `h_min >= 0.02 m` 后退出；回合级 CSV 单独记录 recovery 触发、成功、步数和持续时间。默认仿真配置仍关闭 recovery，诊断入口为 `configs/experiments/p3_diagnostics/b4_osqp_recovery.yaml`。

随后将单一最危险连杆的逃逸方向改为多连杆加权方向：所有负裕度连杆按 clearance deficit 加权合并安全 Jacobian；恢复阶段同时保留尚未达到退出裕度的连杆，避免恢复后立即重触发。实现位于 `src/rl_risk_sac/envs/ur5_dynamic_obstacle_env.py`，并新增集成测试。

`recovery_speed_radps=0.35` 的 20 回合诊断相较 `0.25` 基线将安全违规步数从 120 降至 88，触发回合平均恢复时间从 1.73 s 降至 1.53 s；任务成功率仍为 11/20，碰撞仍为 1/20，故速度提升只能作为诊断基线，不能视为安全保证。

多连杆加权方向已用 3 个评估 seed、每 seed 20 回合复核：

| eval seed | success | collision | safety violation steps | min predictive `h` |
| ---: | ---: | ---: | ---: | ---: |
| 5101 | 11/20 | 1/20 | 90 | -0.0788 m |
| 5201 | 15/20 | 1/20 | 112 | -0.0499 m |
| 5301 | 11/20 | 4/20 | 60 | -0.1595 m |

为复现 seed 5301 的 4 次碰撞，已使用 B4 选中的 step 95000 checkpoint、20 episodes 和带 trace 的 OSQP 重跑。原过滤器即使显示 `solved` 也会发生碰撞：它只约束机器人命令对 `h` 的变化，漏掉障碍物自身速度引起的裕度漂移。现将约束修正为 `J_h qdot + h_drift + kappa(h - m) >= 0`，并在 trace 中记录逐连杆预测/鲁棒距离、`h_i`、`J_h qdot`、`h_drift`、约束残差、障碍物状态及初始不安全标记。

在同一 seed/checkpoint 的逐步诊断中，将 recovery 触发/退出改为 `h_min <= 0.06 m` / `h_min >= 0.08 m`。碰撞数从原多连杆 recovery 的 4/20 依次变为提前 recovery 的 3/20、计入漂移但零速度回退的 3/20、放宽不可行预测约束的 2/20，以及 maximin recovery 的 1/20；最后一轮成功率为 14/20。初始 `h=-0.0677 m` 的 episode 16 被明确标记为 `initially_unsafe=1`，并在 maximin recovery 下脱离后成功完成；不能据此将初始安全集外恢复表述为安全保证。

maximin recovery 在预测约束不可行时，保持关节和工作空间硬约束，最大化所有未达退出裕度连杆的最小 clearance 导数，并以 `recovery_relaxed` 明确记录预测约束放宽。它避免了 episode 0 的碰撞，但 episode 8 仍碰撞，且最危险连杆残差仍为负；episode 0 还出现 4.05 s recovery 后未完成任务。该分支单步最高求解时间约 277 ms，超过 50 ms 控制周期，因而只可作为离线故障诊断，不能进入实时链路、P4、OOD 或真机。该历史诊断随后已完成限时/预算复核，并在 2026-07-30 的 bounded escape 结论中淘汰。

### 最新进度（2026-07-30，bounded escape 复核与方案收敛）

此前 `primal infeasible` 识别修复后，bounded escape 的两个入口均已完成同一 checkpoint、同一两个 eval seed 的 20 episode 复核。恢复分支的参数错位也已修复：`SafetyFilterInput` 不再被误传为 `projection_iterations`；安全过滤器 focused/integration 测试为 `23 passed`，目标配置 smoke test 通过。

不带预算 watchdog 的 bounded escape 结果如下：

| eval seed | success | collision | safety violation steps | recovery triggered/success | mean solve | max solve |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5101 | 10/20 | 1/20 | 27 | 15/14 | 4.32 ms | 176.24 ms |
| 5201 | 13/20 | 1/20 | 0 | 16/15 | 5.23 ms | 198.50 ms |

对应指标文件为 `outputs/p3_diagnostics/b4_bounded_escape_margin30_fixarg_seed5101_metrics.csv` 和 `outputs/p3_diagnostics/b4_bounded_escape_margin30_fixarg_seed5201_metrics.csv`；预算复核文件为同名 `budget_seed5101/5201_metrics.csv`。

加入 `max_filter_compute_time_s=0.05` 的预算诊断没有改善任务或安全结果：seed 5101 仍为 10/20、1/20、27 个违反步，发生 1 次预算停止；seed 5201 仍为 13/20、1/20，但出现 21 个违反步和 7 次预算停止。最大记录耗时仍为 154.59 ms 和 221.09 ms。该 watchdog 在过滤计算返回后才执行，不能真正中断超时的 OSQP/主动集计算。

因此，bounded escape、`recovery_relaxed` 和 maximin recovery 均不进入实时链路，也不进入 P4/OOD 或真机验证。该阶段曾暂时保留 strict safe-stop 入口；随后已完成 frame-corrected deterministic escape 的 4×20 复核（success `4/80`、collision `34/80`、最大过滤耗时 `410.4 ms`）。因此这里的 strict 入口应理解为历史诊断配置，不是已通过的部署方案，P3 仍不冻结。

> **口径更新（2026-07-31）**：本节及前面的 bounded/strict 段落是阶段性历史记录。后续 frame-corrected deterministic escape 的 4×20 复核和完整漂移审计已经完成，当前状态以本页“当前口径修正”和 `docs/archive/p3_pre_20260803/p3_failure_root_cause_analysis.md` 为准。

### 最新进度（2026-07-30，几何修正后的 strict safe-stop 诊断）

UR5 胶囊映射已修正：`upper_arm` 从 `shoulder_link -> upper_arm_link` 改为 `shoulder_link -> forearm_link`，并补齐后续前臂/腕部段。几何修正后的 B4 开发训练已完成两个 20k train seed，checkpoint selection 使用 validation seed `6201`：seed `4102` 选择 `actor_step_20000.pt`，seed `4103` 选择 `actor_step_15000.pt`。

在严格 OSQP 配置 `b4_osqp_strict_margin30_budget.yaml`、eval seed `6101`、每个 seed 20 episodes 下，结果为：

| train seed | selected checkpoint | success | collision | capsule overlap | PyBullet contact | violation steps | mean solve | max solve |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4102 | 20000 | 7/20 | 1/20 | 0/20 | 1/20 | 123 | 4.42 ms | 206.69 ms |
| 4103 | 15000 | 6/20 | 1/20 | 0/20 | 1/20 | 4 | 4.43 ms | 146.43 ms |

两次碰撞均为 PyBullet 实际接触，接触连杆为 `upper_arm_link`，胶囊重叠检测为 false；这说明碰撞审计已能区分“胶囊模型漏检”和物理引擎接触。两组结果都出现超过 50 ms 的单步长尾，且成功率仅为 30--35%，因此不能冻结 P3，也不能把严格 safe-stop 描述为已满足实时安全保证。当前结果只覆盖 eval seed `6101`，不能替代新的 held-out eval seeds `5101/5201` 的完整复核。

对提前降速诊断 trace 的漂移审计显示：11 个进入 `safe_stop_infeasible` 的 episode 中，4 个属于 `h` 下降超过 `0.05 m/s` 的动态漂移型，3 次碰撞全部集中在该类别；静态或缓慢漂移型共 7 个。结论是障碍物持续运动会使零速度 safe-stop 离开安全集，后续必须把“静态不可行”和“动态漂移不可行”分开处理。该审计是故障机理诊断，不是安全保证。

当前主要问题：上臂实际接触仍未被胶囊重叠可靠覆盖；预测安全裕度在动态漂移下会下降；严格 OSQP 平均耗时较低但存在 146--207 ms 长尾；过滤器高干预/不可行时任务成功率明显下降。下一步只做离线碰撞几何变换审计、动态漂移故障分析和独立 held-out 复核，不启动新的训练、P4/OOD 或真机。

主比较的模型训练、checkpoint selection、eval-seed held-out 评估和三-seed汇总均已完成；它们被冻结为阶段一基线，当前不需要继续训练或重跑。新研究的实施基准见 [research_direction.md](../../design/research_direction.md)：B1--B5 已在一个开发 train seed 和一个开发 eval seed 上完成 10k step 链路检查，但没有稳定的任务--安全增益，不能进入 P4 或作为论文证据。

## 2. 已完成工作

| 工作项 | 状态 | 结果 |
| --- | --- | --- |
| 原四种方法的 100k 训练与评估 | 已完成 | 提供初始消融、checkpoint selection 和 adaptive 诊断；旧主表不再作为最终方法排序依据 |
| 固定惩罚参数敏感性筛选 | 已完成 | 在两 seed、50 episodes 的筛选中，`fixed_risk_penalty=1.0` 是 `link_fixed` 的候选最优值 |
| `link_fixed_penalty1` 100k 重训 | 已完成 | train seeds 为 101、202、303；与原训练预算相同 |
| checkpoint selection | 已完成 | 每个 train seed 使用 validation seed 2001、20 episodes，独立选择 checkpoint |
| held-out 三方法统一评估 | 已完成 | eval seeds 为 1004、1005、1006；135 个 CSV、13,500 episodes |
| 三-seed 汇总 | 已完成 | 已生成按 train seed、跨 seed 和宏平均三类表格 |
| adaptive actor / execution 反事实诊断 | 已完成 | adaptive actor 训练退化，且自适应执行器显著增加 jerk |
| P1 不确定性连杆预测风险 | 已完成（开发验证） | 已实现时间戳状态估计接口、短时最小预测距离、几何/感知/时延/跟踪裕度和 `h`；无效、未来或过期观测不会产生低风险结果 |
| P2 仿真安全过滤器 | 已完成（开发验证） | 策略命令可经连杆预测约束、关节位置/速度/加速度和 TCP 工作空间约束投影；不可用风险、无效输入或不可行约束均输出零速度 |
| P2 端到端失效注入与指标 | 已完成（开发验证） | 可注入有效、无效或过期障碍物估计；运行时记录预测状态、`h_min`、原始/过滤后命令、干预量、停止状态、违反量与求解耗时 |
| P2 解析点雅可比与仿真实时性 | 已完成（开发基准） | 以 PyBullet 点雅可比替代有限差分状态保存/恢复；20 episode、2,693 个控制步中滤波器均值 2.68 ms、P95 4.94 ms，无步超过 50 ms。该基准不代表后续 OSQP/deterministic escape 的硬实时预算 |
| P3 B1--B5 开发轮次 | 已完成（未冻结） | 共用 train seed 4101、100k step、validation seed 5201、eval seed 5101；B4/B5 无稳定综合优势 |
| P3 过滤器约束诊断 | 已完成（未冻结） | 已完成 OSQP 状态修复、逐连杆诊断、障碍物漂移项、recovery 与几何修正后的 strict safe-stop 诊断；strict 2x20 结果仍有碰撞、长尾和任务退化，不能冻结 |
| OSQP 依赖与后端 | 已完成（开发验证） | `osqp=1.1.3`、`scipy=1.18.0` 已安装；常规 OSQP 路径约 2.6--2.7 ms，但 maximin recovery 单步最高约 277 ms，尚不满足当前 50 ms 控制周期 |

## 3. 最终主对比口径

| 维度 | 设置 |
| --- | --- |
| 主比较方法 | `ee_fixed`、`link_fixed_penalty1`、`ldrc_fixed` |
| `link_fixed_penalty1` | 连杆级风险 + 固定风险惩罚 SAC + 固定平滑，`sac.fixed_risk_penalty=1.0` |
| train seeds | 101、202、303 |
| checkpoint validation | validation seed 2001，每 checkpoint 20 episodes |
| held-out eval seeds | 1004、1005、1006 |
| episodes | 每个 eval seed 100；每个场景-方法-train seed 合并 300 episodes |
| 场景 | `random_crossing`、`upper_arm_crossing`、`elbow_crossing`、`forearm_crossing`、`wrist_crossing` |
| 正式评估规模 | 3 methods x 3 train seeds x 3 eval seeds x 100 episodes x 5 scenarios = 13,500 episodes |

参数 `fixed_risk_penalty=1.0` 在早期 train seeds 101、202 的小规模筛选中提出；主结论使用未参与该筛选的 held-out eval seeds 1004--1006，但重训仍复用了 train seeds 101、202。因此，这一评估仅对 eval seeds 独立，尚未对完整的超参数选择和训练随机性完全独立。每个表格单元先对同一 train seed 的 300 episodes 求均值，再在 3 个 train seed 间计算 mean +/- sample std（`n=3`）。

最终主表数据源：

```text
outputs/rechecks/heldout_1004_1006/final_3methods/all_eval_episodes.csv
outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_by_train_seed.csv
outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_across_train_seeds.csv
outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_macro_across_train_seeds.csv
```

旧文件 `outputs/formal/summary/` 及原 `link_fixed`（惩罚 4.0）结果只保留作历史和敏感性分析，不能再用于主方法排序或正文主表。

## 4. 核心结果

跨五个场景的宏平均，统计量为 3 个 train seed 的 mean +/- std：

| 方法 | success rate | collision rate | final position error | min distance | safety violation rate | RMS jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ee_fixed` | 16.4% +/- 1.4% | 7.69% +/- 2.68% | 0.396 +/- 0.014 m | 0.189 +/- 0.013 m | 1.69% +/- 0.61% | 24.7 +/- 1.5 |
| `link_fixed_penalty1` | 66.7% +/- 15.3% | 6.51% +/- 3.63% | 0.125 +/- 0.041 m | 0.202 +/- 0.011 m | 1.25% +/- 0.63% | 33.0 +/- 3.1 |
| `ldrc_fixed` | 59.3% +/- 20.6% | 5.91% +/- 1.27% | 0.160 +/- 0.067 m | 0.169 +/- 0.012 m | 3.58% +/- 0.98% | 34.4 +/- 0.4 |

主场景 `random_crossing` 中，`link_fixed_penalty1` 达到 64.2% +/- 12.5% 成功率、6.89% +/- 2.22% 碰撞率和 0.125 +/- 0.029 m 最终误差；`ldrc_fixed` 为 56.3% +/- 18.0%、5.56% +/- 2.91% 和 0.172 +/- 0.054 m。

按场景观察，`link_fixed_penalty1` 在五个场景的平均成功率均高于 `ldrc_fixed`；尤其在 forearm 和 wrist 场景分别达到 72.8% 和 76.1%。在 upper arm 和 elbow 场景，`ldrc_fixed` 的碰撞率低于 `link_fixed_penalty1`，因此不能宣称固定惩罚方法在所有安全指标上严格占优。

## 5. Adaptive 诊断

以下诊断沿用原正式矩阵的 `random_crossing` 结果，不属于本次 held-out 三方法主表。在 seed 202、303 上，对 actor 与执行器交叉评估，每个组合 300 episodes：

| train seed | actor / execution | success rate | collision rate | RMS jerk |
| --- | --- | ---: | ---: | ---: |
| 202 | fixed / fixed | 77.7% | 2.0% | 32.6 |
| 202 | fixed / adaptive | 71.3% | 5.0% | 93.5 |
| 202 | adaptive / adaptive | 79.3% | 9.0% | 75.2 |
| 202 | adaptive / fixed | 79.7% | 9.0% | 27.8 |
| 303 | fixed / fixed | 50.3% | 6.0% | 32.7 |
| 303 | fixed / adaptive | 57.0% | 7.0% | 73.2 |
| 303 | adaptive / adaptive | 30.3% | 26.0% | 74.3 |
| 303 | adaptive / fixed | 26.3% | 26.0% | 31.4 |

结论：seed 303 的退化来自 adaptive actor 本身；固定执行器不能恢复其任务和碰撞表现。自适应执行器会显著提高 jerk，且没有稳定的性能收益。因此 `ldrc_adaptive` 保留为失败消融，不纳入当前主对比或真实部署候选。

## 6. 可支持的结论

1. 连杆级风险建模配合恰当标定的固定风险惩罚能够提高单动态障碍物场景下的目标到达能力。`link_fixed_penalty1` 在 eval-seed held-out 的五个场景中均获得最高的三-seed平均成功率，并降低最终位置误差。
2. `link_fixed_penalty1` 与 `ldrc_fixed` 存在碰撞控制折中：后者的宏平均碰撞率略低（5.91% vs 6.51%），但前者在成功率、最终误差、最小距离、安全距离违反率、jerk 和平均风险代价上更好。
3. 当前证据不支持“风险约束 SAC 相对经过调参的固定惩罚 baseline 带来整体增益”的主张。约束方法的优势应限定为部分高风险区域中的略低碰撞率，而非综合性能最优。
4. 固定惩罚权重是强敏感超参数。原 `link_fixed` 取值 4.0 导致过度保守，不能作为固定惩罚方法的代表性最终比较配置。
5. `ldrc_adaptive` 没有显示稳定增益；当前不支持自适应平滑提升任务能力、安全性或平滑性的主张。

## 7. 局限与风险

| 风险 | 影响 | 处理方式 |
| --- | --- | --- |
| 仅 3 个 train seed | 不能开展可靠显著性检验 | 报告 mean +/- std（n=3），不使用“统计显著”措辞 |
| `fixed_risk_penalty=1.0` 来自小规模筛选 | 可能存在配置选择偏差；筛选与重训复用了 train seeds 101、202 | 最终比较使用未参与筛选的 eval seeds 1004--1006；后续应增加未参与筛选的新 train seeds 复核 |
| 固定惩罚与约束方法各有安全指标取舍 | 不能宣称任一方法全安全指标最优 | 同时报告碰撞率、违反率、最小距离和任务指标 |
| adaptive actor seed 303 退化 | 不支持完整方法主张 | 将 adaptive 作为失败消融，不作真实部署候选 |
| 单球形障碍物仿真 | 外部泛化有限 | 限定为本文仿真设定，不作真实安全保证 |

## 8. 已固化事项与后续工作

### 8.1 已完成：阶段一结果固化

1. 已使用 held-out 三方法汇总表生成正文 Table 1（`random_crossing`）和 Table 2（四个定向压力场景），统计单位为 train seed，方法名称统一为 `link_fixed_penalty1` 或“连杆级固定风险惩罚 SAC（`w_R=1.0`）”。
2. 阶段一结论文件已统一为“`link_fixed_penalty1` 是历史综合候选；`ldrc_fixed` 仅在部分场景保留较低碰撞率”的口径；原四方法表格仅作为历史附录。
3. `docs/archive/stage1/paper_materials.md` 已列出正文/附录图表、数据源与一键重建命令，并明确 `w_R=1.0` 筛选与 train seed 复用的局限。

### 8.2 已完成：阶段一策略离线预检；实机签核待现场完成

1. 三个 `link_fixed_penalty1` checkpoint 已固化为部署候选：train seeds 101、202、303 分别选择 step 100000、50000、100000。
2. `scripts/deployment_preflight.py` 已完成无硬件预检：确认 checkpoint 可加载、推理输入和命令均为有限值、归一化动作未越界，并在 PyBullet 中确认关节速度命令不超过 `0.7 rad/s` 仿真限幅。
3. 真实机器人控制接口、控制器限速/工作空间、急停与保护停、RGB-D 失效安全停止、相机外参和碰撞包络仍需在现场签核；完成后才可使用轻质球体开展 10--20 次低速可执行性验证。
4. 阶段一策略不可作为新论文的最终部署方法。新系统须先完成安全过滤器、感知失效分支和独立离线预检，才可进入现场签核；真实实验仍不进行高风险碰撞性基线对比。

### 8.3 已完成：P1/P2 仿真开发验证

1. [预测风险模块](../src/rl_risk_sac/utils/predictive_risk.py) 已实现连杆胶囊体短时最小表面距离、逐连杆几何裕度，以及感知、时延和跟踪误差裕度。输出包括 `d_pred`、`d_robust`、安全函数 `h`、状态年龄和可用性状态。
2. [安全过滤器](../src/rl_risk_sac/utils/safety_filter.py) 已以半空间投影实现 `J_h qdot + kappa h >= 0`，并同时处理关节位置、速度、加速度/命令连续性与外部工作空间约束。风险不可用、输入无效或约束不可行时，过滤器确定性输出零速度。
3. [UR5 仿真环境](../src/rl_risk_sac/envs/ur5_dynamic_obstacle_env.py) 在 `env.safety_filter.enabled=true` 时将过滤器作为策略命令的唯一出口。通过 PyBullet 点雅可比和固定最优投影参数构造连杆安全函数与 TCP 工作空间雅可比；默认配置保持关闭，不改变阶段一结果。
4. 已提供感知估计注入接口，可在仿真中复现无效和过期状态。端到端指标包括预测状态、`h_min`、原始与过滤后动作、干预范数、停止状态、活动约束数、最大违反量和求解时间。
5. 当前回归结果为 `51 passed`，且 P2、B4、B5 的 PyBullet smoke test 已通过。过滤器已支持循环投影、Dykstra、主动集回退、可选 OSQP QP、显式 recovery 和多连杆加权逃逸方向；新增的 OSQP infeasible/recovery 回归 focused suite 为 `23 passed`。bounded escape、strict safe-stop 和 deterministic escape 均出现任务退化、碰撞或长尾，不能作为实时方案。所有耗时数字只覆盖当前 PyBullet 进程内的求解，不覆盖感知、通信、控制器和真实硬件延迟。

### 8.4 当前状态：P3 因子化开发轮次已完成，设计未冻结

1. 已在 `configs/experiments/p3/` 下完成 B1--B5：共用 train seed `4101`、10k step 和 eval seed `5101` 的 20 episode 检查，输出隔离在 `outputs/p3_dev/b*/`。开发汇总如下，统计单位是 episode，仅用于发现问题：

| 方法 | success | collision | safety violation rate | 说明 |
| --- | ---: | ---: | ---: | --- |
| B1 末端当前距离 | 10% | 20% | 4.65% | 历史弱基线 |
| B2 连杆当前距离 | 15% | 20% | 4.48% | 任务误差改善，但无安全增益 |
| B3 连杆预测风险 | 5% | 15% | 6.58% | 碰撞略低但任务和距离违反变差 |
| B4 预测风险 + 非鲁棒过滤器 | 15% | 25% | 9.40% | 介入率 34.96%，不可行停止率 4.60%，未显示安全收益 |
| B5 鲁棒预测风险 + 鲁棒过滤器 | 0% | 20% | 4.55% | 介入率 47.96%，不可行停止率 8.23%，任务过于保守或训练不足 |

2. B4/B5 的过滤器在 PyBullet 内分别达到 P95 3.89 ms 和 4.88 ms，均无控制步超过 50 ms；实时性不再是当前开发瓶颈，但这不是端到端或实机实时性结论。
3. 100k 开发评估仍不支持冻结 B1--B5 参数、不支持进入 P4/OOD，也不支持把 B4/B5 写成降低碰撞率的证据。OSQP 状态记录和 `primal infeasible` 识别问题已修复，并已用同一 checkpoint、eval seeds 5101/5201 完成带 trace 的 bounded escape 复核。
4. 复核表明 bounded escape 的碰撞仍为各 1/20，seed 间安全违反不稳定，且预算诊断仍出现 154--221 ms 长尾；因此淘汰 relaxed/maximin recovery。几何修正后的 strict safe-stop 在 eval seed 6101 上仍为 1/20 碰撞、146--207 ms 长尾，且成功率为 6--7/20；因此继续停留在开发诊断。
5. 设计冻结后，才可用新的 train seeds 和 held-out eval seeds 完成独立复核与 OOD 扰动；当前 P1/P2 测试和单 seed P3 结果不得替代此证据。真机工作仍停留在现场签核前。

### 2026-07-31 低速双速度主控制点（当前执行口径）

为排除机械臂过慢导致的失败，本轮固定两个独立低速变量：

| 变量 | 固定值 | 单位 |
| --- | ---: | --- |
| 障碍物线速度 | 0.05 | m/s |
| 机械臂关节速度上限（action_scale） | 1.0 | rad/s |

1.0 rad/s 比历史 0.7 rad/s 上限略高，避免动作能力不足成为失败原因；joint_acceleration_limit_radps2=4.0 仍固定，单独保留加速约束变量。0.05 m/s 为已测试的最低非零障碍物速度。两者分别配置、分别记录，不用一个速度值替代另一个。

当前入口为 b4_fixed_obstacle005_robot100_*，新 train seeds 为 4108/4109，eval seeds 为 5101/5201。b4_fixed_obstacle005_robot025_* 保留为慢机械臂速度对照；旧 fixed_speed010 结果不与当前主控制点混合。

本轮优先级仍为几何一致性 -> 动态漂移与联合不可行 -> collision 分类 -> 任务收敛与跨 seed 稳定性；实时耗时、硬截止和 watchdog 只做被动记录。


### 2026-07-31 速度边界与低速双速度主控制点（当前执行口径）

速度边界诊断实际覆盖 0.05/0.10/0.20/0.30/0.38 m/s，每档 80 episodes。原始 speed_boundary_summary.csv 的 speed_mps 标签错误地除以 1000，已修正。

| 实际速度 (m/s) | success | collision | capsule overlap | PyBullet contact | infeasible rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 5.0% | 15.0% | 12 | 0 | 17.8% |
| 0.10 | 6.25% | 33.75% | 27 | 0 | 24.1% |
| 0.20 | 3.75% | 37.5% | 30 | 0 | 23.4% |
| 0.30 | 7.5% | 53.75% | 43 | 3 | 38.4% |
| 0.38 | 6.25% | 65.0% | 52 | 2 | 48.7% |

当前主控制点固定两个独立速度：障碍物线速度 0.05 m/s，机械臂关节速度上限 action_scale=1.0 rad/s。1.0 rad/s 比历史 0.7 rad/s 略高，避免机械臂动作能力不足成为失败原因；joint_acceleration_limit_radps2=4.0 仍固定。旧 robot025 配置和 fixed_speed010 结果只作对照，不与当前主结果混合。

当前入口为 b4_fixed_obstacle005_robot100_*，新 train seeds 为 4108/4109，eval seeds 为 5101/5201。实时耗时、硬截止和 watchdog 只做被动记录，待统一配置下的非实时问题解决后再处理。
