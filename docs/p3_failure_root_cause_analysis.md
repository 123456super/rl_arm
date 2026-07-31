# P3 近期实验失败根因分析

> 文档状态：2026-07-31 诊断版。本文统一整理当前 P3 仿真失效证据，不改写阶段一（P0）已冻结的主结果。

## 1. 结论摘要

近期实验持续失败不是单一阈值、求解器或训练种子造成的，而是以下闭环叠加：

```text
训练几何/安全包络与评估几何不一致
        -> 策略进入评估时才出现的严格边界
        -> recovery 命令经过 strict filter 后不可行
        -> 输出零速度 safe-stop
        -> 动态障碍物继续运动，安全裕度继续下降
        -> capsule overlap 或真实接触，回合终止
```

根因优先级：

1. **训练与评估的几何和安全 envelope 不一致**，造成策略分布与执行约束失配。
2. **strict safe-stop 不能在动态障碍物运动时保持安全集**，零速度不等于安全保持。
3. **recovery 不是独立的可行性求解器**，候选动作仍需通过 strict filter；不可行时仍停止。
4. **胶囊模型同时承担风险约束和回合终止**，既产生保守误报，也存在与 PyBullet 接触不一致的漏检。
5. **预测风险的一阶局部模型没有回答联合约束下是否存在可达逃逸动作**。
6. **OSQP/投影计算存在远超 50 ms 的长尾**；预算检查是返回后的事后检查，不是硬截止。

训练步数少、独立 seed 少和指标口径混杂会放大问题，但不是唯一根因。当前不能把失败解释为“继续调 recovery 参数即可解决”，也不能把 capsule collision rate 直接解释为物理碰撞率。

## 2. 阶段边界

- P0 阶段一主比较已冻结；本文不修改其结论。
- P1/P2 仅表示风险、过滤器和诊断链路完成开发验证，不代表端到端安全保证。
- P3 尚未冻结。本轮先在统一配置下解决几何一致性、动态可行性、碰撞口径和任务退化问题；实时性/耗时暂不作为本轮阻塞条件。
- 在 P3 通过统一几何、碰撞分类、动态可行性、任务性能和不可行率门槛前，不进入 P4/OOD 或真机验证。实时性仍是最终部署门槛，但本轮只记录，不用它阻止离线训练和诊断。
- `recovery_relaxed`、bounded/maximin escape 只能作为离线诊断，不能作为安全保证或论文主结果。

## 3. 结果总览

### 3.1 最新 deterministic escape

frame-corrected 几何、strict OSQP、20 ms 求解限时、50 ms 诊断预算；4 组各 20 回合，共 80 回合：

| train/eval | success | collision | safety violation steps | max solve |
| --- | ---: | ---: | ---: | ---: |
| 4102/5101 | 2/20 | 7/20 | 401 | 378.7 ms |
| 4102/5201 | 2/20 | 12/20 | 435 | 368.4 ms |
| 4103/5101 | 0/20 | 6/20 | 188 | 401.4 ms |
| 4103/5201 | 0/20 | 9/20 | 380 | 410.4 ms |
| **合计** | **4/80（5%）** | **34/80（42.5%）** | **1404** | **410.4 ms** |

补充：recovery triggered `76/80`，recovery success `48/80`，平均过滤耗时约 `8.1 ms`。34 次 collision 全部为 capsule overlap，PyBullet physical contact 为 0。

后续报告必须拆分：`capsule_overlap`（保守胶囊重叠）和 `pybullet_contact`（物理接触）。两者不能合并解释为真实碰撞率；physical contact 为 0 也不等于安全，因为模型仍可能漏检。

### 3.2 不可行停止漂移审计

完整 `safe_stop_drift_audit.json` 覆盖 58 个不可行停止回合：

| 类别 | episode 数 | 其中碰撞 | 碰撞率 |
| --- | ---: | ---: | ---: |
| dynamic drift | 34 | 29 | 85.3% |
| static or slow | 24 | 5 | 20.8% |

这直接表明动态障碍物持续运动是 safe-stop 失效的主要物理机制。不能把全部碰撞预先标记为 `unavoidable_collision`。

### 3.3 历史 mesh-fit 结果

历史 300 回合 mesh-fit 复核为 success `19/300`、collision `105/300`；其中 82 次 capsule-only、24 次触发 PyBullet contact、仅 1 次同时触发。该结果说明胶囊模型会误报，也可能漏掉物理接触；几何采样覆盖不能替代完整动态碰撞判据。该历史数据不与最新 deterministic 数值混合。

### 3.3 固定速度边界诊断（2026-07-31）

在相同 frame-corrected 几何、strict QP、安全 margin 0.03 m、两个历史 B4 checkpoint、两个 eval seed 的条件下，完成了 5 个固定障碍物速度，每个速度 80 episodes（共 400 episodes）。速度边界汇总文件为 outputs/p3_diagnostics/speed_boundary/speed_boundary_summary.csv。

注意：该 CSV 的 speed_mps 展示字段存在除以 1000 的标签错误。配置后缀 speed_005/010/020/030/038 对应的实际速度分别是 0.05/0.10/0.20/0.30/0.38 m/s，以下表格按配置实际值修正。

| 实际速度 (m/s) | episodes | success | collision | capsule overlap | PyBullet contact | avoidable | infeasible steps | mean solve | max solve |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 80 | 4/80 (5.0%) | 12/80 (15.0%) | 12 | 0 | 12 | 17.8% | 4.88 ms | 325 ms |
| 0.10 | 80 | 5/80 (6.25%) | 27/80 (33.75%) | 27 | 0 | 7 | 24.1% | 4.97 ms | 338 ms |
| 0.20 | 80 | 3/80 (3.75%) | 30/80 (37.5%) | 30 | 0 | 3 | 23.4% | 5.23 ms | 329 ms |
| 0.30 | 80 | 6/80 (7.5%) | 43/80 (53.75%) | 43 | 3 | 3 | 38.4% | 4.94 ms | 337 ms |
| 0.38 | 80 | 5/80 (6.25%) | 52/80 (65.0%) | 52 | 2 | 7 | 48.7% | 4.91 ms | 294 ms |

结论：速度从 0.10 m/s 起已能暴露 dynamic drift；继续提高速度时 collision 和 infeasible rate 总体上升。0.10 m/s 是为了减少变量而选的统一控制点，不是宣称的最优安全速度。capsule overlap 和 PyBullet contact 必须继续分开；success 在所有速度下仍只有 3.75%--7.5%，首要问题是统一训练/评估包络、联合不可行和 safe-stop 后的动态漂移。

## 4. 根因分析

### 4.1 训练几何与评估几何不一致

训练入口 `configs/experiments/p3_geometry_fix/b4_seed4102_20k.yaml` 和 `b4_seed4103_20k.yaml` 只 include 旧 `p3/b4_predictive_nonrobust_filter.yaml`。对应 checkpoint 的 `config.json` 仍主要使用旧 capsule 结构和较小半径，训练时 `geometry_margin_m=0`，没有最终 frame-corrected local endpoints。

最新评估使用 frame-corrected endpoints、更大的 mesh-fit capsule、`geometry_margin_m=0.03 m` 以及 strict predictive/workspace/joint constraints。策略是在较宽松安全模型上训练，却在更严格 envelope 上执行，导致动作频繁进入评估新增边界，干预、不可行和任务失败一起增加。

几何覆盖审计通过只说明采样点在胶囊内，不证明训练/评估模型一致，也不证明胶囊与真实 mesh 的动态接触判定一致。

### 4.2 safe-stop 不能保持动态安全集

环境在每个仿真子步都调用 `_move_obstacle()`；不可行时令机器人速度为零，只能停止机器人，不能停止障碍物。实际链路是：

1. 预测约束不可行；
2. filter 输出零速度；
3. 障碍物继续移动；
4. `h` 继续下降；
5. 发生 capsule overlap 或物理接触。

dynamic-drift 不可行回合碰撞率 85.3%，说明这是结构性缺口，而非偶发异常。`unavoidable_collision` 只能在满足动态漂移和 safe-stop 条件时使用。

### 4.3 recovery 不能绕过不可行 filter

`_recovery_command()` 根据各连杆 clearance deficit 加权安全 Jacobian，生成远离障碍物的有界方向；它不是联合约束可行性求解器。`env.step()` 随后仍将该命令交给 strict filter。当 `recovery_allow_constraint_relaxation=false` 且约束不可行时，`safety_filter.py` 返回 `SAFE_STOP_INFEASIBLE`，最终命令仍为零速度。

因此 deterministic escape 的准确描述是：

> TTC/裕度触发的候选 escape + strict filter + 不可行时 safe-stop。

它不是“约束不可行后仍保证逃逸”的控制器。

### 4.4 胶囊模型直接影响碰撞和成功

环境使用 `capsule_overlap OR pybullet_contact` 作为 collision，并要求 `d_min > d_safe` 才算 success。几何包络变化会同时改变风险约束、过滤器行为、回合终止、success、reward/cost 和训练分布。因此更保守的模型会直接降低 success 并增加 capsule-only collision；包络不足又会漏掉 physical contact。

### 4.5 预测风险与控制可行性存在断层

预测使用 constant-velocity obstacle 和有限窗口；机器人 link velocity 在风险计算中置零，动作影响随后通过局部 `J_h qdot` 一阶约束修正。这能估计局部动作对 `h` 的导数，但没有求解多连杆、关节位置/速度/加速度、workspace、障碍物漂移和可达逃逸轨迹的联合可行性。

所以 `h` 下降或 filter infeasible 不等价于“没有任何物理可行动作”；反过来，filter solved 也不等价于未来一定不会接触。

### 4.6 求解时间没有硬截止

OSQP 的 `time_limit=0.02 s` 和环境的 `max_filter_compute_time_s=0.05 s` 都不能中断已经运行的 OSQP、Python 调用或投影。后者是在 solver 返回后检查，超时才改为 safe-stop。

最新 deterministic 结果平均约 8.1 ms，但最大单步耗时 368--410 ms；历史 TTC/strict 诊断还出现 146--582 ms 长尾。平均值不能证明满足 50 ms 控制周期。

### 4.7 放大因素

- deterministic eval 主要使用 20k-step checkpoint；
- 只有两个 train seeds；
- checkpoint 与最终 corrected eval geometry 不一致；
- 早期 B1--B5 success 仅 0--15%，说明尚未形成稳定策略；
- 当前结果是开发诊断，不是跨 seed 泛化或 OOD 证据。

增加训练步数可能改变数值，但在几何、不可行处理和动态漂移未统一前，不能证明根因消失。

## 4.8 2026-07-31 V2 2x2 根因隔离结果

在相同 checkpoint、评估口径和 20 episodes x 2 train seeds x 2 eval seeds 的设置下，完成了 `legacy/corrected geometry` x `moving/static obstacle` 的 V2 2x2 对照。该实验用于隔离几何包络和障碍物运动因素，不是 P0 正式主表，也不是安全通过验证。

| 条件 | success | collision | capsule overlap | PyBullet contact | unavoidable | avoidable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy geometry + moving | 11/80 (13.8%) | 6/80 (7.5%) | 3 | 5 | 4 | 2 |
| corrected geometry + moving | 8/80 (10.0%) | 39/80 (48.8%) | 39 | 1 | 34 | 5 |
| legacy geometry + static | 8/80 (10.0%) | 0/80 | 0 | 0 | 0 | 0 |
| corrected geometry + static | 7/80 (8.8%) | 0/80 | 0 | 0 | 0 | 0 |

补充统计：legacy/moving 有 259 个 violation steps、271 个 infeasible steps 和 14 次 budget stop；corrected/moving 有 1,534 个 violation steps、1,889 个 infeasible steps 和 19 次 budget stop；legacy/static 有 16 个 infeasible steps 和 4 次 budget stop；corrected/static 有 537 个 infeasible steps 和 5 次 budget stop。corrected/moving 的 39 次 collision 中，38 次为 capsule-only，1 次为 PyBullet physical contact；因此 capsule collision 与物理接触必须分开解释。

这组对照支持两个边界明确的判断：

1. geometry 从 legacy 切换为 corrected 后，moving 条件的 collision 从 7.5% 上升到 48.8%，但同一 corrected geometry 在 static 条件下 collision 为 0；动态障碍物运动是关键交互因素，不能把变化归因于策略单独退化。
2. corrected geometry 的采样覆盖审计已通过，但其更严格包络显著增加 infeasible 和 capsule-only 事件；采样覆盖证明模型内部一致性，不证明胶囊与真实 mesh 的完整动态碰撞等价。

## 4.9 V2 safe-stop 漂移审计

文件：`outputs/p3_diagnostics/root_cause_2x2_v2/safe_stop_drift_audit.json`。审计只针对发生 infeasible stop 的 episode，并按停止后 `h` 是否持续下降分类；`unavoidable_collision` 是离线标签，不是物理不可避免性的证明。

| 条件 | infeasible episodes | dynamic drift | dynamic drift 中碰撞 | static/slow | static/slow 中碰撞 |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy + moving | 15 | 7 | 4/7 (57.1%) | 8 | 0/8 |
| corrected + moving | 59 | 39 | 34/39 (87.2%) | 20 | 5/20 (25.0%) |
| legacy + static | 0* | 0 | 0 | 0* | 0 |
| corrected + static | 0* | 0 | 0 | 0* | 0 |

`*` static 条件没有动态漂移，也没有碰撞；其少量 infeasible steps 不是进入漂移审计的 episode。V2 与统一 trace/metrics 的标签已对齐：legacy/moving 为 4 unavoidable、2 avoidable；corrected/moving 为 34 unavoidable、5 avoidable。

结论是：corrected/moving 的碰撞主要发生在 dynamic drift（34/39，87.2%）。机器人停止后障碍物仍继续运动，zero-velocity safe-stop 不能保持动态安全集；不能将所有 infeasible 后碰撞预先标为 unavoidable，也不能把 safe-stop 写成动态安全保证。

## 4.10 OSQP 约束归因

目录：`outputs/p3_diagnostics/root_cause_attribution/`。对 1,823 个 infeasible steps 执行离线单类别松弛探针：1,818 个可由单类别松弛解释，5 个属于 combined_or_unresolved。类别存在重叠，出现次数不能直接相加：

| 类别 | 出现次数 |
| --- | ---: |
| `predictive_barrier` | 1,815 |
| `joint_acceleration` | 1,103 |
| `workspace` | 17 |
| `joint_velocity` | 8 |

主要组合为 `predictive_barrier` 单独 708 次、`predictive_barrier + joint_acceleration` 1,084 次、`predictive_barrier + joint_acceleration + workspace` 14 次，另有少量 velocity/workspace/未解析组合。由此，当前 infeasibility 的主因是 predictive barrier 与 joint acceleration 的联合可达性冲突，而不是单纯 workspace 或 joint velocity 限制。该归因是诊断证据，不等价于已经找到可部署的修复。

## 4.11 V2 时间预算审计

V2 filter timing（仅统计当前仿真进程内过滤器，不含感知、通信和控制器）如下：

| 条件 | mean | P95 | P99 | max |
| --- | ---: | ---: | ---: | ---: |
| legacy + moving | 4.76 ms | 6.33 ms | 6.88 ms | 223.47 ms |
| corrected + moving | 5.25 ms | 6.72 ms | 7.47 ms | 350.70 ms |
| legacy + static | 4.36 ms | 5.89 ms | 6.55 ms | 197.15 ms |
| corrected + static | 4.73 ms | 6.37 ms | 6.94 ms | 256.17 ms |

均值、P95/P99 接近毫秒级，但 max 仍达到 197--351 ms，不能声称满足 50 ms 控制周期。现有 `max_filter_compute_time_s` 是 solver 返回后的检查，无法中断已经运行的 OSQP/主动集路径；硬截止仍未解决。

## 4.12 基于 V2 的现状与决策

已完成 V2 2x2 几何/运动隔离、safe-stop 漂移审计、统一 collision 标签、OSQP 约束归因、速度边界诊断，以及安全过滤器回归（全量测试 55 passed）。

当前已确认：几何失配和动态漂移是主要问题；infeasible 主要涉及 predictive barrier 与 joint acceleration 的联合冲突；任务 success 仍很低，不足以支持策略性能结论。实时性存在长尾，但本轮按当前决策只记录，不把它作为离线训练/诊断的阻塞条件。

本轮问题优先级：
1. 训练/评估 frame-corrected 几何、安全 margin 和动作约束统一；
2. 固定障碍物 0.05 m/s、机械臂关节 1.0 rad/s 下的动态漂移、safe-stop 失效和联合不可行；
3. capsule overlap 与 PyBullet contact 独立统计；
4. 统一配置下的任务性能、训练收敛和跨 train seed 稳定性；
5. 上述问题收敛后，再处理求解耗时、硬截止和实时链路。

## 4.13 统一配置与本轮执行口径（2026-07-31）

新增配置目录：configs/experiments/p3_unified_geometry/。

基础入口为 b4_fixed_speed010_base.yaml；新训练 seeds 为 4104 和 4105，各 100000 steps；评估 seeds 为 5101 和 5201。训练和评估共享最终 frame-corrected local capsule endpoints、geometry_margin_m=0.03、strict QP、recovery/relaxation/maximin 关闭、动作/工作空间/关节加速度约束和 speed_range=[0.10,0.10]。

历史 configs/experiments/p3_geometry_fix/b4_seed4102_20k.yaml、b4_seed4103_20k.yaml 不修改，旧 checkpoint provenance 继续可复现。统一配置中的 qp_time_limit_s 和 max_filter_compute_time_s 为 null，表示本轮是离线 feasibility/behavior 轮次，不是实时性验证。

本轮执行顺序：先验证几何和安全 envelope 一致，再验证动态漂移和联合不可行，再验证碰撞分类与任务收敛，最后才处理耗时、硬截止和实时控制链路。

### 4.14 低速双速度隔离配置（2026-07-31，当前控制点）

为把速度因素从失败原因中尽量摘除，本轮将障碍物和机械臂速度分别固定，不再使用速度范围或默认动作上限：

| 变量 | 配置 | 单位 | 含义 |
| --- | --- | --- | --- |
| 障碍物速度 | speed_range=[0.05,0.05] | m/s | 固定的球形障碍物线速度 |
| 机械臂速度 | action_scale=1.0 | rad/s | 策略、过滤器和执行层共享的关节速度上限 |

选择 0.05 m/s 障碍物速度，是因为速度边界诊断中它是已测试的最低非零速度；选择 1.0 rad/s 机械臂关节上限，是为了避免 0.25 rad/s 的人为限速成为失败原因，同时略高于历史 0.7 rad/s 上限，保留足够的避障动作能力。两者是不同物理量，不能比较数值大小，也不能用一个变量代替另一个变量。

新增当前主控制点配置：
configs/experiments/p3_unified_geometry/b4_fixed_obstacle005_robot100_base.yaml
configs/experiments/p3_unified_geometry/b4_fixed_obstacle005_robot100_seed4108_100k.yaml
configs/experiments/p3_unified_geometry/b4_fixed_obstacle005_robot100_seed4109_100k.yaml
configs/experiments/p3_unified_geometry/b4_fixed_obstacle005_robot100_eval_seed5101.yaml
configs/experiments/p3_unified_geometry/b4_fixed_obstacle005_robot100_eval_seed5201.yaml

该配置继续使用最终 frame-corrected capsules、geometry_margin_m=0.03、strict QP、recovery/relaxation/maximin 关闭，qp_time_limit_s 和 max_filter_compute_time_s 为 null。fixed_obstacle005_robot025 配置保留为较慢机械臂速度对照，不与当前主控制点结果混合。

## 5. 统一指标和标签

1. collision 同时报告 `capsule_overlap`、`pybullet_contact`，并可报告二者同时触发。
2. success 同时受 goal error 和 `d_min > d_safe` 限制，不能只解释为到达率。
3. `unavoidable_collision` 仅用于 safe-stop 后仍存在 dynamic drift 的离线分类。
4. `safe_stop_infeasible` 与 `safe_stop_compute_budget` 分开统计。
5. 同时报告 mean、P95/P99、max solve time；最大值超过控制周期时不能声称实时满足。
6. episode 是评估样本，不是独立 train seed；跨训练重复按 train seed 报告。
7. 几何顶点/三角形采样覆盖只作为模型一致性证据，不作为动态安全证明。

## 6. 下一步验证顺序

### A. 统一配置训练与评估（当前）

使用 b4_fixed_obstacle005_robot100 配置下的两个新 train seeds 4108/4109 和 eval seeds 5101/5201。训练、checkpoint selection 和评估都使用同一组 frame-corrected capsules、margin 0.03 m、action/workspace/joint constraints、strict QP 和固定 speed_range=[0.05,0.05] 和 action_scale=1.0。

### B. 统一配置下的动态可行性

记录 safe_stop_infeasible 后的 h 漂移、dynamic/static 分类、碰撞分类，以及 predictive barrier 与 joint acceleration 的约束归因。

### C. 分离碰撞判据

同时记录总 collision、capsule_overlap、pybullet_contact、link name 和最小接触距离，不把 capsule-only 事件写成物理碰撞。

### D. 任务性能和训练收敛

按 train seed 报告 success、collision、infeasible rate、violation steps、最终误差和干预量。episode 只作为回合样本，不冒充独立 train seed。统一配置达到可解释收敛前，不进入 P4/OOD 或真机。

### E. 实时性与硬截止（后置）

本轮只保留 mean/P95/P99/max 记录，不以 max_solve <= 50 ms 阻止离线训练和根因诊断。统一配置的非实时问题解决后，再单独处理可中断求解器、硬截止和外部 watchdog。
## 7. 当前决策

当前 P3 结论：诊断已完成，设计仍未冻结；本轮进入统一变量的离线训练/评估阶段。

本轮明确不修改历史 4102/4103 配置，不扩大速度、几何或 recovery 变量，不把 0.05 m/s 或 1.0 rad/s 宣称为最优安全边界，不把 capsule-only collision 写成 physical contact，不把 safe-stop 写成动态安全保证，也不因暂不处理耗时而进入 P4/OOD 或真机。

## 8. 证据索引

- 阶段决策：`docs/research_direction.md`、`docs/experiment_progress.md`
- 最新 metrics：`outputs/p3_diagnostics/deterministic_escape/*metrics.csv`
- 最新 trace：`outputs/p3_diagnostics/deterministic_escape/*_traces/`
- 漂移审计：`outputs/p3_diagnostics/deterministic_escape/safe_stop_drift_audit.json`
- V2 2x2 漂移审计：`outputs/p3_diagnostics/root_cause_2x2_v2/safe_stop_drift_audit.json`
- V2 约束归因：`outputs/p3_diagnostics/root_cause_attribution/*/metrics.csv`
- 几何审计：`outputs/p3_diagnostics/deterministic_escape/collision_coverage_2048.json`、`collision_triangles_p256_s8.json`
- 训练配置：`configs/experiments/p3_geometry_fix/b4_seed4102_20k.yaml`、`b4_seed4103_20k.yaml`
- deterministic 配置：`configs/experiments/p3_diagnostics/b4_deterministic_escape_frame_corrected_budget.yaml`
- 环境流程：`src/rl_risk_sac/envs/ur5_dynamic_obstacle_env.py`
- 过滤器和预算：`src/rl_risk_sac/utils/safety_filter.py`
