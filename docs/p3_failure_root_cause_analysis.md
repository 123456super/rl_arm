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
- P3 尚未冻结。当前证据只支持固定预算、硬约束优先的仿真失效分析。
- 在 P3 通过固定时间预算、碰撞分类、任务性能和不可行率门槛前，不启动新训练、P4/OOD、实时链路或真机验证。
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

**已完成：** V2 2x2 几何/运动隔离、safe-stop 漂移审计、统一 collision 标签、OSQP 约束归因、V2 timing 统计，以及相关安全过滤器回归（29 passed）。

**当前现状：** P3 已完成诊断，但设计未冻结。几何失配和动态漂移已被证据支持；infeasible 主要是 predictive barrier 与 acceleration 的联合问题；任务 success 仍只有 8.8%--13.8%，不足以支持策略性能结论；过滤器 max 仍超 50 ms。

**当前问题：** 训练/评估几何和安全 margin 尚未形成统一冻结配置；strict safe-stop 对动态障碍物没有持续安全保证；联合可行性没有得到可达逃逸求解；实时路径没有硬截止；样本仍是开发 checkpoint 的 80 episode 量级，不能支持跨 seed 泛化。

**当前决策：** 不重训、不进入 P4/OOD、不做真机验证。下一轮只允许在统一几何和碰撞口径下做离线可达性、障碍物运动边界、硬截止求解器和最小回归；只有通过固定预算、碰撞分类、不可行率和任务性能门槛后，才重新评估是否冻结 P3。

## 5. 统一指标和标签

1. collision 同时报告 `capsule_overlap`、`pybullet_contact`，并可报告二者同时触发。
2. success 同时受 goal error 和 `d_min > d_safe` 限制，不能只解释为到达率。
3. `unavoidable_collision` 仅用于 safe-stop 后仍存在 dynamic drift 的离线分类。
4. `safe_stop_infeasible` 与 `safe_stop_compute_budget` 分开统计。
5. 同时报告 mean、P95/P99、max solve time；最大值超过控制周期时不能声称实时满足。
6. episode 是评估样本，不是独立 train seed；跨训练重复按 train seed 报告。
7. 几何顶点/三角形采样覆盖只作为模型一致性证据，不作为动态安全证明。

## 6. 下一步验证顺序

### A. 隔离 geometry mismatch

固定 checkpoint、eval seed 和环境随机序列，仅切换训练几何/`geometry_margin` 与最终 corrected 几何/`geometry_margin`，比较 success、两类 collision、infeasible rate、violation steps 和干预量。

### B. 隔离 dynamic drift

固定策略和几何，分别运行 obstacle velocity = 0 与原始 moving obstacle，比较 safe-stop 后 `h`、碰撞分类和恢复触发。

### C. 分离碰撞判据

每次评估同时记录 capsule overlap、PyBullet contact、link name 和最小接触距离。

### D. 分类 infeasible 原因

按 predictive barrier、joint bound、acceleration、workspace、projection failure、OSQP infeasible 和 compute budget 分类。

### E. 审计真实时间预算

记录完整过滤周期的 mean、P95/P99、max 及各 phase 时间。若需要硬截止，使用可中断求解器或独立外部 watchdog，不能依赖 post-return 检查。

### F. 统一配置后再决定是否重训

训练和评估完全统一几何、安全 margin、动作约束和碰撞口径后，再决定是否投入新的训练预算。

## 7. 当前决策

当前 P3 结论：**诊断完成，设计未冻结**。

近期失败的主因不是“策略还差一点”，而是训练分布、几何模型、可行性处理、动态障碍物假设和实时预算没有形成一致闭环。在这些结构性问题解决前：

- 不把 deterministic escape 写成安全通过；
- 不把 capsule-only collision 写成物理碰撞；
- 不把 safe-stop 写成动态安全保证；
- 不继续扩大训练、P4/OOD 或真机实验。

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
