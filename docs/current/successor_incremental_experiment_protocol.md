# 后续渐进式实验协议

> 更新时间：2026-08-07。本文把已完成的无障碍 reaching v2 作为唯一基础策略，规定后续逐层增加环境和安全机制的实验顺序。每一级只改变一个主要因素；前一级未通过时，不进入后一级，也不同时修改训练、障碍物和过滤器。

## 1. 总原则

当前最可靠的结果是独立的 reaching v2：三个 actor 在固定 `9001--9200` final manifest 上 pooled success 为 `582/600=97.0%`，固定 IK 可达且无障碍候选路径子集为 `582/582=100%`，无 collision、capsule overlap 或 physical contact。

后续实验的第一目标不是立即训练出新方法，而是回答每次失败究竟来自哪一层：

1. 基础 actor 是否仍能完成 reaching；
2. 障碍物出现后，任务和观测分布是否改变；
3. 障碍物运动是否造成额外失败；
4. strict safety filter 是否造成干预、不可行或 safe-stop；
5. 只有上述链路可解释后，才考虑训练新的安全策略。

禁止把多个新因素放在同一次实验中。尤其不得在尚未完成冻结 actor 评估前同时加入动态障碍物、predictive risk、strict QP、viability monitor、recovery 或新的训练目标。

## 2. 固定不变的基线

### 2.1 Actor 与 checkpoint

后续冻结 actor 只使用以下三个 checkpoint，不重新选择、不静默替换：

| train seed | checkpoint |
| ---: | --- |
| 4301 | `outputs/reaching_recovery_v2/train/seed_4301/link_fixed_no_obstacle_speed100_seed4301_steps300000/actor_step_240000.pt` |
| 4302 | `outputs/reaching_recovery_v2/train/seed_4302/link_fixed_no_obstacle_speed100_seed4302_steps300000/actor_step_280000.pt` |
| 4303 | `outputs/reaching_recovery_v2/train/seed_4303/link_fixed_no_obstacle_speed100_seed4303_steps300000/actor_step_220000.pt` |

checkpoint 选择依据保持为独立的 40 个 validation seeds `9301--9340`，三个 actor 的 validation success 均为 `39/40=97.5%`。

### 2.2 Reset、目标和控制条件

- final reset manifest：`configs/experiments/reaching_recovery/manifests/v1_final.json`，即 `9001--9200`，每个 actor 运行 200 个 episode。
- validation manifest：只用于需要检查配置/链路时的 40 个 seeds；不得用 final seeds 调参。
- UR5、随机目标、初始关节状态、20 Hz 控制、12 s episode 上限、`0.055 m` success tolerance 和 `action_scale: 1.0` 保持不变。
- 先关闭安全过滤器、viability monitor、diagnostic logging、recovery、relaxation 和 maximin。
- 每一级写入新的日期化输出目录，不覆盖 `outputs/reaching_recovery_v2/` 或历史 P3 输出。

### 2.3 统一结果字段

所有阶段都必须报告：

- success、timeout、最终位置误差、完成时间；
- `collision_capsule_overlap`、`collision_pybullet_contact`、`collision_any`、`termination_collision` 和 `termination_reason`；
- 最小距离、风险、目标和障碍物初始状态/速度；
- 实际 `seed` 与 `seed_manifest`。

开启过滤器后，额外报告 intervention rate/norm、safe-stop rate、infeasible rate、projection failure rate、fallback rate，以及 solve time 的 mean/P95/P99/max 和 compute-budget stop rate。

## 3. 阶梯实验顺序

### S0：冻结 actor 无障碍复核

**目的**：确认后续运行环境、checkpoint 加载和 manifest 没有改变基础结果。

**设置**：沿用 v2，无障碍、无安全过滤器。

**通过条件**：三个 actor 的 pooled success 与冻结结果一致到统计口径允许的误差；collision、capsule overlap、physical contact 均为零。若 S0 不通过，停止并先修复运行链路。

### S1：静态障碍物，仍关闭安全过滤器

**唯一新增因素**：打开障碍物，但设置 `speed_range: [0.0, 0.0]`；场景先使用 `random`。

**目的**：区分“障碍物本身改变观测/任务”与“动态运动造成的困难”。此阶段不使用 predictive risk 或过滤器。

**通过条件**：

- 先看三 actor 是否仍有可解释的到达能力；
- 记录碰撞和 capsule-only 事件，不能只看 success；
- 若失败主要是目标/静态障碍物几何不可行，应转入离线任务可行性分层，不直接训练。

S1 未完成前，不进入动态障碍物实验。

### S1-R：静态障碍物迁移训练（S1 未通过时的修复路径）

当前 S1 的冻结无障碍 actor pooled success 只有 `428/600=71.3%`，其中 `156/172` 个失败是 timeout，说明主要缺少绕障策略，而不是过滤器问题。若 S1 未通过，采用迁移训练：从三个 v2 selected actor 及其 agent state 继续训练，只在训练环境加入静态随机障碍物，并启用当前连杆风险的固定惩罚作为 dense learning signal。

S1-R 只改变训练适应，不加入动态速度、安全过滤器、viability monitor、recovery 或 relaxation。每个 seed 从对应 v2 selected checkpoint 继续 `200000` steps；新 actor 只能在独立 validation manifest 上选择，final 仍使用完整 `9001--9200` 清单。

S1-R 的通过条件：三个 seed 在 final 全量分布上均重新评估；pooled success 必须高于冻结 actor 的 `71.3%`，且任何单个 actor 不得低于其对应冻结基线（`87.0%/53.0%/74.0%`），physical contact 总数不得高于旧 S1 的 `13/600`。任何提升都必须同时报告 capsule overlap、timeout 和最终误差，不能只报告 success。若仍失败，先分析静态场景的任务可行性和失败 reset，不进入 S2。

本轮 S1-R final 已满足上述门槛：pooled `491/600=81.8%`，三个 actor 分别为 `87.0%/82.0%/76.5%`，physical contact `4/600`。按原门槛本可进入 S2；但静态候选子集仍只有 `89.9%`，因此当前决策是先执行 S1-R2，不启动 S2。S2 若后续获准，只把 `speed_range` 改为 `[0.05, 0.05]`，继续关闭安全过滤器、viability、recovery 和 relaxation，且仍使用相同的 actor、validation 规则和 final manifest。

### S1-R 后续诊断：先检查动作响应，再决定是否二次迁移训练

离线有限候选预检查显示静态候选路径存在的 reset 为 161/200，但其策略执行成功率只有 434/483=89.9%；49 个失败均为 timeout，且没有碰撞。因此“排除无解后 99%”目前没有证据支持，不能按标签删除失败 reset。建议先在相同 S1-R selected actor、相同 final manifest、相同静态场景下做只改变 `env.fixed_beta` 的响应性诊断：`0.50` 与 `0.65`，不训练、不启用过滤器、不改变 success threshold。若某个 beta 在三个 actor 上都降低候选子集 timeout 且 collision_any 不增加，再以该 beta 作为下一轮静态迁移训练的唯一新增变量；若无改善，则停止盲目加训，转向逐 reset 轨迹和目标/障碍物几何归因。

诊断配置：`configs/experiments/reaching_incremental/s1_static_beta050_seed430{1,2,3}.yaml`、`s1_static_beta065_seed430{1,2,3}.yaml`。这些配置只覆盖固定动作平滑系数，继承 S1-R 的静态零速度障碍物和关闭安全过滤器设置。

诊断评估命令（六个终端可并行；每个命令使用各配置中继承的 GPU）如下。checkpoint 固定为 S1-R validation 已选结果，不能重新选点：

```bash
python scripts/evaluate.py --config configs/experiments/reaching_incremental/s1_static_beta050_seed4301.yaml --checkpoint outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4301/link_fixed_static_obstacle_finetune_seed4301_steps440000/actor_step_240000.pt --seed-manifest configs/experiments/reaching_recovery/manifests/v1_final.json --episodes 200 --output outputs/reaching_incremental/s1_static_obstacle_beta_diagnostic/beta050/seed_4301_final.csv --trace-output outputs/reaching_incremental/s1_static_obstacle_beta_diagnostic/beta050/seed_4301_traces
python scripts/evaluate.py --config configs/experiments/reaching_incremental/s1_static_beta050_seed4302.yaml --checkpoint outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4302/link_fixed_static_obstacle_finetune_seed4302_steps480000/actor_step_480000.pt --seed-manifest configs/experiments/reaching_recovery/manifests/v1_final.json --episodes 200 --output outputs/reaching_incremental/s1_static_obstacle_beta_diagnostic/beta050/seed_4302_final.csv --trace-output outputs/reaching_incremental/s1_static_obstacle_beta_diagnostic/beta050/seed_4302_traces
python scripts/evaluate.py --config configs/experiments/reaching_incremental/s1_static_beta050_seed4303.yaml --checkpoint outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4303/link_fixed_static_obstacle_finetune_seed4303_steps420000/actor_step_320000.pt --seed-manifest configs/experiments/reaching_recovery/manifests/v1_final.json --episodes 200 --output outputs/reaching_incremental/s1_static_obstacle_beta_diagnostic/beta050/seed_4303_final.csv --trace-output outputs/reaching_incremental/s1_static_obstacle_beta_diagnostic/beta050/seed_4303_traces
python scripts/evaluate.py --config configs/experiments/reaching_incremental/s1_static_beta065_seed4301.yaml --checkpoint outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4301/link_fixed_static_obstacle_finetune_seed4301_steps440000/actor_step_240000.pt --seed-manifest configs/experiments/reaching_recovery/manifests/v1_final.json --episodes 200 --output outputs/reaching_incremental/s1_static_obstacle_beta_diagnostic/beta065/seed_4301_final.csv --trace-output outputs/reaching_incremental/s1_static_obstacle_beta_diagnostic/beta065/seed_4301_traces
python scripts/evaluate.py --config configs/experiments/reaching_incremental/s1_static_beta065_seed4302.yaml --checkpoint outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4302/link_fixed_static_obstacle_finetune_seed4302_steps480000/actor_step_480000.pt --seed-manifest configs/experiments/reaching_recovery/manifests/v1_final.json --episodes 200 --output outputs/reaching_incremental/s1_static_obstacle_beta_diagnostic/beta065/seed_4302_final.csv --trace-output outputs/reaching_incremental/s1_static_obstacle_beta_diagnostic/beta065/seed_4302_traces
python scripts/evaluate.py --config configs/experiments/reaching_incremental/s1_static_beta065_seed4303.yaml --checkpoint outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4303/link_fixed_static_obstacle_finetune_seed4303_steps420000/actor_step_320000.pt --seed-manifest configs/experiments/reaching_recovery/manifests/v1_final.json --episodes 200 --output outputs/reaching_incremental/s1_static_obstacle_beta_diagnostic/beta065/seed_4303_final.csv --trace-output outputs/reaching_incremental/s1_static_obstacle_beta_diagnostic/beta065/seed_4303_traces
```

### S1-R2：静态混合场景迁移训练（actor-only 诊断，已完成）

由于 S1-R 的静态候选子集仍只有 `434/483=89.9%`，且 beta 响应诊断没有跨 seed 稳定收益，当前先做第二轮静态迁移训练。S1-R2 只改变训练分布和 dense 风险惩罚：障碍物保持零速度，`scenario: mixed_static` 以 50% 原始 `random`、12.5% `upper_arm_crossing`、12.5% `elbow_crossing`、12.5% `forearm_crossing`、12.5% `wrist_crossing` 采样，`fixed_risk_penalty: 2.0`。安全过滤器、动态速度、viability、recovery、relaxation 和 maximin 全部关闭。

S1-R2 从 S1-R selected actor 继续训练 300000 steps；起始 checkpoint 也作为不得退化候选复制到新 run 目录。为保持三个 train seed 的迁移口径一致，三组均使用 actor-only 迁移并重新初始化 critic、target critic、alpha 和优化器状态，不恢复旧 SAC state。由于 actor-only 迁移不能让随机 critic 立即接管已训练 actor，训练入口必须按本次迁移重新计数：先用新场景数据填充 replay，并执行默认 `10000` 步 critic-only warmup，期间冻结 actor 和 alpha；之后才开启完整 SAC 更新。4301 的起点 `actor_step_240000.pt` 没有同一步保存的 agent state，不能拿 `step_260000` 或最终 state 冒充匹配状态。validation 仍使用 `9301--9340`，final 仍使用完整 `9001--9200`，不能删除 reset。训练输出目录与 S1-R 分开，不覆盖既有结果。

训练配置为 `configs/experiments/reaching_incremental/s1_static_mixed_finetune_seed430{1,2,3}.yaml`。三个终端分别运行：

```bash
RUN=outputs/reaching_incremental/s1_static_mixed_finetune/train/seed_4301/link_fixed_static_mixed_finetune_seed4301_steps540000
mkdir -p "$RUN"
cp -n outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4301/link_fixed_static_obstacle_finetune_seed4301_steps440000/actor_step_240000.pt "$RUN/actor_step_240000.pt"
python scripts/train.py --config configs/experiments/reaching_incremental/s1_static_mixed_finetune_seed4301.yaml --resume-actor outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4301/link_fixed_static_obstacle_finetune_seed4301_steps440000/actor_step_240000.pt --reset-agent-state --reset-state-warmup-steps 10000 --start-step 240000
```

```bash
RUN=outputs/reaching_incremental/s1_static_mixed_finetune/train/seed_4302/link_fixed_static_mixed_finetune_seed4302_steps780000
mkdir -p "$RUN"
cp -n outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4302/link_fixed_static_obstacle_finetune_seed4302_steps480000/actor_step_480000.pt "$RUN/actor_step_480000.pt"
python scripts/train.py --config configs/experiments/reaching_incremental/s1_static_mixed_finetune_seed4302.yaml --resume-actor outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4302/link_fixed_static_obstacle_finetune_seed4302_steps780000/actor_step_480000.pt --reset-agent-state --reset-state-warmup-steps 10000 --start-step 480000
```

```bash
RUN=outputs/reaching_incremental/s1_static_mixed_finetune/train/seed_4303/link_fixed_static_mixed_finetune_seed4303_steps620000
mkdir -p "$RUN"
cp -n outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4303/link_fixed_static_obstacle_finetune_seed4303_steps420000/actor_step_320000.pt "$RUN/actor_step_320000.pt"
python scripts/train.py --config configs/experiments/reaching_incremental/s1_static_mixed_finetune_seed4303.yaml --resume-actor outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4303/link_fixed_static_obstacle_finetune_seed4303_steps620000/actor_step_320000.pt --reset-agent-state --reset-state-warmup-steps 10000 --start-step 320000
```

训练完成后，只在 `mixed_static` 的 `9301--9340` validation 上选择 checkpoint：

```bash
python scripts/select_checkpoint.py --config configs/experiments/reaching_incremental/s1_static_mixed_finetune_seed4301.yaml --run-dir outputs/reaching_incremental/s1_static_mixed_finetune/train/seed_4301/link_fixed_static_mixed_finetune_seed4301_steps540000 --seed-manifest configs/experiments/reaching_recovery/manifests/v1_validation.json --episodes 40 --metric success_rate
python scripts/select_checkpoint.py --config configs/experiments/reaching_incremental/s1_static_mixed_finetune_seed4302.yaml --run-dir outputs/reaching_incremental/s1_static_mixed_finetune/train/seed_4302/link_fixed_static_mixed_finetune_seed4302_steps780000 --seed-manifest configs/experiments/reaching_recovery/manifests/v1_validation.json --episodes 40 --metric success_rate
python scripts/select_checkpoint.py --config configs/experiments/reaching_incremental/s1_static_mixed_finetune_seed4303.yaml --run-dir outputs/reaching_incremental/s1_static_mixed_finetune/train/seed_4303/link_fixed_static_mixed_finetune_seed4303_steps620000 --seed-manifest configs/experiments/reaching_recovery/manifests/v1_validation.json --episodes 40 --metric success_rate
```

选择完成后，再读取 `selected_checkpoint.csv`，使用完整 `9001--9200` final manifest 在原始 `random` 静态障碍物分布上评估；评估配置为 `s1_static_mixed_eval_random_seed430{1,2,3}.yaml`。不能用 final manifest 选点，也不能删失败 reset。

### S1-R2.1：匹配 SAC state 的静态混合迁移（已完成）

S1-R2 的 actor-only 结果显示，随机初始化的 critic 在跳过原始 `start_step` warmup 后立即接管已训练 actor，导致 4301/4302 退化。下一轮先修复迁移初始化，不改变 mixed-static 覆盖或 `fixed_risk_penalty=2.0`：恢复 actor 与同一步匹配的 SAC state，完整恢复 critic/target critic/alpha；不使用 `--reset-agent-state`。4301 的 actor 与 S1-R 起始时完全相同，因此使用原始 v2 的 `agent_state_step_240000.pt` 作为严格匹配 state；4302/4303 使用 S1-R 对应的 `agent_state_step_480000.pt`/`agent_state_step_320000.pt`。训练输出使用新目录，不覆盖 S1-R 或 S1-R2。

三个训练命令如下：

```bash
python scripts/train.py --config configs/experiments/reaching_incremental/s1_static_mixed_stateful_finetune_seed4301.yaml --resume-actor outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4301/link_fixed_static_obstacle_finetune_seed4301_steps440000/actor_step_240000.pt --resume-state outputs/reaching_recovery_v2/train/seed_4301/link_fixed_no_obstacle_speed100_seed4301_steps300000/agent_state_step_240000.pt --start-step 240000
```

```bash
python scripts/train.py --config configs/experiments/reaching_incremental/s1_static_mixed_stateful_finetune_seed4302.yaml --resume-actor outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4302/link_fixed_static_obstacle_finetune_seed4302_steps480000/actor_step_480000.pt --resume-state outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4302/link_fixed_static_obstacle_finetune_seed4302_steps480000/agent_state_step_480000.pt --start-step 480000
```

```bash
python scripts/train.py --config configs/experiments/reaching_incremental/s1_static_mixed_stateful_finetune_seed4303.yaml --resume-actor outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4303/link_fixed_static_obstacle_finetune_seed4303_steps420000/actor_step_320000.pt --resume-state outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4303/link_fixed_static_obstacle_finetune_seed4303_steps420000/agent_state_step_320000.pt --start-step 320000
```

完成后仍只使用 `9301--9340` validation 选点，再用完整 `9001--9200` final manifest 在原始 `random` 静态分布评估。S1-R2.1 期间不进入动态障碍物、不启用安全过滤器、viability、recovery 或 relaxation。

S1-R2.1 结果：validation 选中 4301=`520000`、4302=`480000`、4303=`620000`；full final pooled success 为 `497/600=82.8%`，静态候选路径子集为 `443/483=91.7%`，physical contact `1/600`。相较 S1-R，成功增加 6 个、physical contact 减少 3 个，但 4301 退化、4302 回退起点、提升主要由 4303 贡献，暂不进入 S2。

### S2：低速动态障碍物，仍关闭安全过滤器

**唯一新增因素**：在 S1 设置上把障碍物速度改为固定 `0.05 m/s`，其余不变。

**目的**：单独测量障碍物运动相对于静态障碍物带来的增量影响。

**通过条件**：对比 S1/S2 的 success、physical contact、capsule overlap、timeout、最小距离和最终误差；按相同 reset seed 配对分析。若 S2 失败，先判断是动态场景不可行、策略不具备避障能力，还是碰撞定义/环境执行问题。

### S3：受控障碍物场景或速度敏感性

只有 S2 结果可解释后，才选择一个变量继续：

- 固定速度不变，依次测试 `upper_arm_crossing`、`elbow_crossing`、`forearm_crossing`、`wrist_crossing` 中的一个；或
- 固定 `random` 场景，只把速度改为一个更高的固定值，如 `0.10 m/s`。

不能在同一轮同时改场景和速度。每次只跑一个新条件，并与 S2 使用同一 reset manifest。

### S4：在已通过的障碍物条件上开启 strict safety filter

**唯一新增因素**：保持 S3 最后一个已通过的障碍物条件和冻结 actor，只开启 strict predictive filter。

固定关闭：

- `recovery_mode_enabled`；
- `recovery_allow_constraint_relaxation`；
- `recovery_maximize_min_clearance`；
- maximin、escape 和任何 relaxation。

第一轮只使用确定性、可审计的 strict 配置；不要把 V1 viability monitor 或新的训练代价同时加入。

**通过条件**：

- V0（过滤器关闭）与 strict filter 的所有碰撞字段可对照；
- 过滤器干预、safe-stop、不可行和投影失败均有逐步 trace；
- 求解时间和预算停止率完整记录；
- 任何 physical contact 都要回看对应 link、初始安全状态、过滤器状态和 termination reason。

若 S4 产生大量不可行或 safe-stop，停止扩展，不启用 recovery；先做现有离线约束冲突归因和任务可行性分析。

### S5：只增加 viability monitor（可选）

仅当 S4 的 strict 链路稳定后，才在完全相同 actor、场景、seed 和过滤器设置上打开 `viability_monitor_enabled`。该监视器第一阶段只记录标签，不得改变 `qdot_cmd`。

通过条件是 V0/V1 每一步请求动作、执行命令、终止原因和碰撞字段一致；不一致就修复隔离性，不进入训练。

### S6：小规模安全策略训练（最后进行）

只有 S0--S5 的失败边界已经明确，才另行定义训练协议。训练时必须：

- 使用新的输出根目录和新的 train seeds；
- 明确区分 `qdot_requested` 与 strict filter 后的 `qdot_cmd`；
- 只在 validation manifest 上选 checkpoint；
- final manifest 一次性评估，不按可行性标签删样本；
- 不加入 recovery/relaxation 作为训练或执行补救。

S6 不是当前默认下一步，不能因为 S4 失败就直接通过训练“修掉”过滤器不可行。

## 4. 每一级的停止规则

出现以下任一情况，立即停在当前级别：

1. 基线 actor 在 S0 就无法复现；
2. 新增因素未能被配置哈希、实际 seed 或 trace 证明只改变了一项；
3. 发现 NaN/Inf、单位错误、seed manifest 错位或 checkpoint 被重新选择；
4. collision 事件定义、终止原因或 physical contact 与 capsule overlap 无法区分；
5. 过滤器出现不可行/safe-stop，但没有保留原始请求命令、约束类别和求解状态；
6. 结果需要删除 reset、改变 success threshold 或启用 recovery 才能“通过”。

停止意味着保留失败证据并分析原因，不是继续增加配置或调整多个超参数。

## 5. 推荐的第一轮执行

下一步先执行 S1-R（S0 已通过）：

1. 从三个 v2 selected actor 继续静态障碍物训练 `200000` steps；
2. 只使用 `9301--9340` validation manifest 选择 checkpoint；
3. 三个新 actor 各跑固定 `9001--9200` final；
4. 汇总全量结果和失败 reset；
5. 只有 S1-R 通过后才进入 S2 低速动态障碍物。

本轮不启用安全过滤器、不启用 recovery；v2 actor 原文件不修改。

### S1-R 可直接执行的命令

以下命令假设从仓库根目录执行，使用当前环境中的 `python`，不需要 `conda run`。三个训练可以分别放在三个终端并行运行；每个终端固定一张 GPU，不能把同一个输出目录交给多个进程。

终端 1（GPU 1，seed 4301）：

```bash
python scripts/train.py \
  --config configs/experiments/reaching_incremental/s1_static_finetune_seed4301.yaml \
  --resume-actor outputs/reaching_recovery_v2/train/seed_4301/link_fixed_no_obstacle_speed100_seed4301_steps300000/actor_step_240000.pt \
  --resume-state outputs/reaching_recovery_v2/train/seed_4301/link_fixed_no_obstacle_speed100_seed4301_steps300000/agent_state_step_240000.pt \
  --start-step 240000
```

终端 2（GPU 4，seed 4302）：

```bash
python scripts/train.py \
  --config configs/experiments/reaching_incremental/s1_static_finetune_seed4302.yaml \
  --resume-actor outputs/reaching_recovery_v2/train/seed_4302/link_fixed_no_obstacle_speed100_seed4302_steps300000/actor_step_280000.pt \
  --resume-state outputs/reaching_recovery_v2/train/seed_4302/link_fixed_no_obstacle_speed100_seed4302_steps300000/agent_state_step_280000.pt \
  --start-step 280000
```

终端 3（GPU 0，seed 4303）：

```bash
python scripts/train.py \
  --config configs/experiments/reaching_incremental/s1_static_finetune_seed4303.yaml \
  --resume-actor outputs/reaching_recovery_v2/train/seed_4303/link_fixed_no_obstacle_speed100_seed4303_steps300000/actor_step_220000.pt \
  --resume-state outputs/reaching_recovery_v2/train/seed_4303/link_fixed_no_obstacle_speed100_seed4303_steps300000/agent_state_step_220000.pt \
  --start-step 220000
```

三个训练都完成后，再分别执行 checkpoint 选择。选择只看独立的 `9301--9340` validation manifest，不得把 final seeds 用于选点：

先把各自的迁移起点 actor 作为一个“不得退化”候选放入新 run 目录；这样如果迁移训练在 validation 上变差，选择器可以诚实地保留起点，而不会被迫选择一个更差的新 checkpoint：

```bash
cp -n \
  outputs/reaching_recovery_v2/train/seed_4301/link_fixed_no_obstacle_speed100_seed4301_steps300000/actor_step_240000.pt \
  outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4301/link_fixed_static_obstacle_finetune_seed4301_steps440000/actor_step_240000.pt
```

```bash
cp -n \
  outputs/reaching_recovery_v2/train/seed_4302/link_fixed_no_obstacle_speed100_seed4302_steps300000/actor_step_280000.pt \
  outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4302/link_fixed_static_obstacle_finetune_seed4302_steps480000/actor_step_280000.pt
```

```bash
cp -n \
  outputs/reaching_recovery_v2/train/seed_4303/link_fixed_no_obstacle_speed100_seed4303_steps300000/actor_step_220000.pt \
  outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4303/link_fixed_static_obstacle_finetune_seed4303_steps420000/actor_step_220000.pt
```

若选择器最终选中起点 step，说明本轮迁移没有在独立 validation 上带来可接受改善；仍需保留该结果，不能改用 final seeds 重新选点。

```bash
python scripts/select_checkpoint.py \
  --config configs/experiments/reaching_incremental/s1_static_finetune_seed4301.yaml \
  --run-dir outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4301/link_fixed_static_obstacle_finetune_seed4301_steps440000 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_validation.json \
  --episodes 40 --metric success_rate
```

```bash
python scripts/select_checkpoint.py \
  --config configs/experiments/reaching_incremental/s1_static_finetune_seed4302.yaml \
  --run-dir outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4302/link_fixed_static_obstacle_finetune_seed4302_steps480000 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_validation.json \
  --episodes 40 --metric success_rate
```

```bash
python scripts/select_checkpoint.py \
  --config configs/experiments/reaching_incremental/s1_static_finetune_seed4303.yaml \
  --run-dir outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4303/link_fixed_static_obstacle_finetune_seed4303_steps420000 \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_validation.json \
  --episodes 40 --metric success_rate
```

选择完成后，从每个 `checkpoint_selection/selected_checkpoint.csv` 读取 `checkpoint` 列，显式评估完整 `9001--9200` final manifest。示例（4301；其余两组只替换 seed、配置和输出路径）：

```bash
CKPT=$(python - <<'PY'
import csv
with open("outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4301/link_fixed_static_obstacle_finetune_seed4301_steps440000/checkpoint_selection/selected_checkpoint.csv", newline="", encoding="utf-8") as f:
    print(next(csv.DictReader(f))["checkpoint"])
PY
)
python scripts/evaluate.py \
  --config configs/experiments/reaching_incremental/s1_static_finetune_seed4301.yaml \
  --checkpoint "$CKPT" \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_final.json \
  --episodes 200 \
  --output outputs/reaching_incremental/s1_static_obstacle_finetune/eval/seed_4301_final.csv \
  --trace-output outputs/reaching_incremental/s1_static_obstacle_finetune/eval/seed_4301_traces
```

另外两组 final 评估可在独立终端并行执行：

```bash
CKPT=$(python - <<'PY'
import csv
with open("outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4302/link_fixed_static_obstacle_finetune_seed4302_steps480000/checkpoint_selection/selected_checkpoint.csv", newline="", encoding="utf-8") as f:
    print(next(csv.DictReader(f))["checkpoint"])
PY
)
python scripts/evaluate.py \
  --config configs/experiments/reaching_incremental/s1_static_finetune_seed4302.yaml \
  --checkpoint "$CKPT" \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_final.json \
  --episodes 200 \
  --output outputs/reaching_incremental/s1_static_obstacle_finetune/eval/seed_4302_final.csv \
  --trace-output outputs/reaching_incremental/s1_static_obstacle_finetune/eval/seed_4302_traces
```

```bash
CKPT=$(python - <<'PY'
import csv
with open("outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4303/link_fixed_static_obstacle_finetune_seed4303_steps420000/checkpoint_selection/selected_checkpoint.csv", newline="", encoding="utf-8") as f:
    print(next(csv.DictReader(f))["checkpoint"])
PY
)
python scripts/evaluate.py \
  --config configs/experiments/reaching_incremental/s1_static_finetune_seed4303.yaml \
  --checkpoint "$CKPT" \
  --seed-manifest configs/experiments/reaching_recovery/manifests/v1_final.json \
  --episodes 200 \
  --output outputs/reaching_incremental/s1_static_obstacle_finetune/eval/seed_4303_final.csv \
  --trace-output outputs/reaching_incremental/s1_static_obstacle_finetune/eval/seed_4303_traces
```

最终汇总必须保留 pooled success、timeout、capsule overlap、physical contact、collision_any、最终误差和失败 reset；不得删除失败 reset，也不得在结果不理想时改用 final manifest 选 checkpoint。

冻结 actor 的评估命令统一使用 final manifest 和显式 checkpoint；示例：

```bash
python scripts/evaluate.py \
  --config configs/experiments/reaching_incremental/s0_seed4301.yaml \
  --checkpoint outputs/reaching_recovery_v2/train/seed_4301/link_fixed_no_obstacle_speed100_seed4301_steps300000/actor_step_240000.pt \
  --output outputs/reaching_incremental/s0_no_obstacle/seed_4301_final.csv \
  --trace-output outputs/reaching_incremental/s0_no_obstacle/seed_4301_traces
```

S0/S1 的独立配置已经建立在 `configs/experiments/reaching_incremental/` 下。S1 基础配置为 `configs/experiments/reaching_incremental/s1_static_obstacle.yaml`，只覆盖：

```yaml
includes:
  - ../reaching_recovery/v2_base.yaml

env:
  obstacle:
    enabled: true
    scenario: random
    speed_range: [0.0, 0.0]
  safety_filter:
    enabled: false
    viability_monitor_enabled: false
    diagnostic_logging: false
    recovery_mode_enabled: false
    recovery_allow_constraint_relaxation: false
    recovery_maximize_min_clearance: false
```

S1 的三个 actor 仍分别用上表 checkpoint；只改变 `--config` 和输出目录。S2 只把该配置的 `speed_range` 改为 `[0.05, 0.05]`，不得同时修改场景、actor 或 manifest。

## 6. 结果解释边界

- S0 只能证明基础 reaching 链路仍可复现。
- S1--S3 只能说明冻结基础 actor 在逐级障碍物条件下的行为，不能称为安全方法结果。
- S4--S5 只能验证 strict filter 的执行链路和标签隔离性，不能据此宣称安全保证或硬实时。
- 只有经过独立协议、预先声明门槛和完整 final 评估的新训练方法，才可讨论后继方法候选；不得把无障碍 v2 success 写成动态避障或安全 success。
