# 后续渐进式实验协议

> 更新时间：2026-08-12。当前进度为 S1-R4 已完成训练、validation 选点、full final 评估和失败诊断；S2 及以后尚未授权。本文把无障碍 reaching v2 作为唯一基础策略，规定后续逐层增加环境和安全机制的顺序。

## 1. 总原则

当前最可靠的结果是独立的 reaching v2：三个 actor 在固定 `9001--9200` final manifest 上 pooled success 为 `582/600=97.0%`，固定 IK 可达且无障碍候选路径子集为 `582/582=100%`，无 collision、capsule overlap 或 physical contact。

后续实验的第一目标不是立即训练出新方法，而是回答每次失败究竟来自哪一层：

1. 基础 actor 是否仍能完成 reaching；
2. 障碍物出现后，任务和观测分布是否改变；
3. 障碍物运动是否造成额外失败；
4. strict safety filter 是否造成干预、不可行或 safe-stop；
5. 只有上述链路可解释后，才考虑训练新的安全策略。

禁止把多个新因素放在同一次实验中。尤其不得在尚未完成冻结 actor 评估前同时加入动态障碍物、predictive risk、strict QP、viability monitor、recovery 或新的训练目标。

S1-R4 是这一规则的显式例外：它用于修复已确认的 checkpoint/replay/timeout 训练正确性问题，并同时验证保守稳定机制，因此只能判断修复后整体方案是否可继续，不能归因各组件贡献。R4 已证明整体方向有效，但未达到 99% 静态避障目标；后续若要声明 focused reset、stratified replay、anchor 或 terminal reward 的贡献，仍需单因素消融。

现在按三步继续：

1. 先做终端伺服诊断，确认终端段的局部速度场和 clearance gate。
2. 再做失败邻域 jitter 训练，只修近目标停滞和回退。
3. 最后导出静态 waypoint teacher，只给后续蒸馏/模仿用。

## 2. 固定不变的基线

### 2.1 Actor 与 checkpoint

S0/S1 冻结基线只使用以下三个 v2 checkpoint，不重新选择、不静默替换。S1-R 及以后若使用迁移 actor，必须在对应阶段显式记录起点、selected checkpoint 和评估配置，不能把迁移 actor 冒充以下冻结基线：

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

本轮 S1-R final 已满足上述门槛：pooled `491/600=81.8%`，三个 actor 分别为 `87.0%/82.0%/76.5%`，physical contact `4/600`。按原门槛本可进入 S2；但静态候选子集仍只有 `89.9%`，因此当时的决议是先执行 S1-R2，不启动 S2。S1-R2 及其后续诊断、R4 修复后训练现已完成，当前仍停留在静态障碍物 S1 内分析和修复剩余失败。S2 若后续获准，只把 `speed_range` 改为 `[0.05, 0.05]`，继续关闭安全过滤器、viability、recovery 和 relaxation，且仍使用相同的 actor、validation 规则和 final manifest。

### S1-R 后续诊断：先检查动作响应，再决定是否二次迁移训练

离线有限候选预检查显示静态候选路径存在的 reset 为 161/200，但其策略执行成功率只有 434/483=89.9%；49 个失败均为 timeout，且没有碰撞。因此“排除无解后 99%”没有证据支持，不能按标签删除失败 reset。当时先在相同 S1-R selected actor、相同 final manifest、相同静态场景下执行了只改变 `env.fixed_beta` 的 `0.50/0.65` 响应性诊断；两组都没有跨 seed 稳定改善，随后才进入 S1-R2。

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

由于 S1-R 的静态候选子集仍只有 `434/483=89.9%`，且 beta 响应诊断没有跨 seed 稳定收益，当时执行了第二轮静态迁移训练。S1-R2 只改变训练分布和 dense 风险惩罚：障碍物保持零速度，`scenario: mixed_static` 以 50% 原始 `random`、12.5% `upper_arm_crossing`、12.5% `elbow_crossing`、12.5% `forearm_crossing`、12.5% `wrist_crossing` 采样，`fixed_risk_penalty: 2.0`。安全过滤器、动态速度、viability、recovery、relaxation 和 maximin 全部关闭。

S1-R2 从 S1-R selected actor 继续训练 300000 steps；三组均使用 actor-only 迁移并重新初始化 critic、target critic、alpha 和优化器状态。实际 run 的 `config.json` 只记录 `agent_state_reset=true`，`progress.csv` 在迁移起点后 500 步时 alpha 已经变化，证明当次执行没有 transfer-relative collect-only/critic-only warmup，随机 critic 很早就参与了 actor 更新。后来协议中补写的 `--reset-state-warmup-steps 10000` 不代表已产生的 R2 结果，现已删除，避免错误复现。

R2 的事实来源固定为 `outputs/reaching_incremental/s1_static_mixed_finetune/train/seed_430{1,2,3}/.../config.json`、`progress.csv` 和 `checkpoint_selection/selected_checkpoint.csv`。validation 使用 `9301--9340`；本轮没有形成可引用的 full final。当前训练入口已修复恢复调度，重新执行同名配置会产生不同训练语义，因此不得与历史 R2 数字合并。

### S1-R2.1：同一步部分训练状态的静态混合迁移（已完成）

S1-R2 的 actor-only 结果显示，随机初始化的 critic 在跳过原始 `start_step` warmup 后立即接管已训练 actor，导致 4301/4302 退化。R2.1 因此恢复了与 actor 同一步的旧 `agent_state`，不使用 `--reset-agent-state`。后续审计确认旧文件只含 online reward/cost critic、alpha、lambda 和 cost EMA，不含 target critic、optimizer、replay、训练步数或 RNG；故本轮统一定义为“同一步部分训练状态 warm-start”，不是完整 SAC 恢复。4301 使用 v2 的 `agent_state_step_240000.pt`，4302/4303 使用 S1-R 的 `agent_state_step_480000.pt`/`agent_state_step_320000.pt`。训练输出使用新目录，不覆盖 S1-R 或 S1-R2。

R2.1 的实际起点 actor/state 路径和恢复参数保存在三个 run 的 `config.json` 中：`resume_replay_warmup_steps=10000`、`resume_critic_warmup_steps=10000`。完成后只使用 `9301--9340` validation 选点，再用完整 `9001--9200` final manifest 在原始 `random` 静态分布评估。当前代码增加了签名校验、timeout bootstrap 和完整 checkpoint 语义，重新运行旧命令不会复现同一训练链路；R2.1 仅按现有产物归档。

S1-R2.1 结果：validation 选中 4301=`520000`、4302=`480000`、4303=`620000`；full final pooled success 为 `497/600=82.8%`，静态候选路径子集为 `443/483=91.7%`，collision_any/capsule overlap/physical contact 为 `2/2/1`。相较 S1-R，成功增加 6 个、physical contact 减少 3 个，但 4301 退化、4302 回退起点、提升主要由 4303 贡献，暂不进入 S2。

### S1-D1：候选失败时间窗诊断（已完成）

S1-R2.2 在旧部分状态恢复链路下没有产生可选收益：4301/4303 的短周期低学习率 random-static 续训都选回迁移起点。该结果只否定当时的恢复方式和超参数组合，不能外推为修复后的续训必然无效。对 S1-R2.1 full final trace 的离线归因显示，静态候选路径存在的 40 条失败均为 timeout，其中 32 条曾进入 `0.08 m` 目标误差内且无碰撞。随后只把 episode 上限从 `240` 控制步（12 s）改为 `480`（24 s），形成 S1-D1 时间窗诊断。

固定清单 `configs/experiments/reaching_incremental/manifests/s1_static_candidate_failure_union_v1.json` 包含 36 个 reset：它们在静态候选路径预检查中有 witness，且至少一个 S1-R2.1 actor 在 final 中失败。该清单是失败诊断样本，不得参与 checkpoint selection、不得替代完整 final manifest、不得用于宣称全量 success。S1-D1 保持 static random、零障碍物速度、相同 actor、`success_tolerance=0.055 m`、同一碰撞定义，安全过滤器、viability、recovery、relaxation 均保持关闭。

可在三个终端并行运行：

```bash
python scripts/evaluate.py --config configs/experiments/reaching_incremental/s1_static_horizon_diagnostic_seed4301.yaml --checkpoint outputs/reaching_incremental/s1_static_mixed_stateful_finetune/train/seed_4301/link_fixed_static_mixed_stateful_finetune_seed4301_steps540000/actor_step_520000.pt --seed-manifest configs/experiments/reaching_incremental/manifests/s1_static_candidate_failure_union_v1.json --episodes 36 --output outputs/reaching_incremental/s1_static_horizon_diagnostic/eval/seed_4301.csv --trace-output outputs/reaching_incremental/s1_static_horizon_diagnostic/eval/seed_4301_traces
```

```bash
python scripts/evaluate.py --config configs/experiments/reaching_incremental/s1_static_horizon_diagnostic_seed4302.yaml --checkpoint outputs/reaching_incremental/s1_static_mixed_stateful_finetune/train/seed_4302/link_fixed_static_mixed_stateful_finetune_seed4302_steps780000/actor_step_480000.pt --seed-manifest configs/experiments/reaching_incremental/manifests/s1_static_candidate_failure_union_v1.json --episodes 36 --output outputs/reaching_incremental/s1_static_horizon_diagnostic/eval/seed_4302.csv --trace-output outputs/reaching_incremental/s1_static_horizon_diagnostic/eval/seed_4302_traces
```

```bash
python scripts/evaluate.py --config configs/experiments/reaching_incremental/s1_static_horizon_diagnostic_seed4303.yaml --checkpoint outputs/reaching_incremental/s1_static_mixed_stateful_finetune/train/seed_4303/link_fixed_static_mixed_stateful_finetune_seed4303_steps620000/actor_step_620000.pt --seed-manifest configs/experiments/reaching_incremental/manifests/s1_static_candidate_failure_union_v1.json --episodes 36 --output outputs/reaching_incremental/s1_static_horizon_diagnostic/eval/seed_4303.csv --trace-output outputs/reaching_incremental/s1_static_horizon_diagnostic/eval/seed_4303_traces
```

结果：原 12 s 的 40 条候选 timeout 中，仅 2 条（4302 的 reset `9116`/`9200`）在 24 s 内成功，耗时 `16.95 s`/`14.25 s`；其余 38 条均至 24 s 仍 timeout，且三 actor 无新增 collision。因此未达到“超过半数被延长时间救回”的判据，时间窗不是主因。24 s 结果不能写成原 12 s final 提升，后续保持 `max_episode_steps=240`。

S1-D1 完成后，当时只新增终端到达 dense reward，用于成功阈值附近的停滞与回退，形成 S1-R3；没有改动 obstacle distribution、`fixed_risk_penalty`、速度、平滑响应、安全过滤器、viability、recovery 或 relaxation。R3 仍使用原 40-reset validation manifest 选点，并计划在通过后使用完整 200-reset final manifest 复核。

### S1-R3：终端到达 shaping（已完成，恢复状态存在混杂）

**唯一新增因素**：原 reward 保持不变；当上一步或当前末端目标误差进入 `0.12 m` 时，额外加入有符号 `w_terminal_progress * (e_{t-1} - e_t)`。本轮固定 `w_terminal_progress=30.0`：继续接近目标会加分，回退会等量扣分。默认值为 `0.0`，所以不会改变所有历史实验。该项直接针对 21 条 near-goal timeout 和 11 条 near-goal regression，不更改成功阈值。

训练相对 S1-R2.1 只改变这一个 reward shaping 项：继续使用 `mixed_static`、零速度、`fixed_risk_penalty=2.0`、原 SAC learning rate、同一步部分训练状态 warm-start、`10000` replay warmup 和 `10000` critic-only warmup；安全过滤器、viability、recovery、relaxation 保持关闭。final 仍计划在原始 random 静态分布和完整 `9001--9200` manifest 评估。4302 的 S1-R2.1 起点没有在 stateful 输出目录重存 `agent_state_step_480000.pt`，因此使用了与该 actor 同一步的 S1-R state。

R3 的实际 actor/state 起点和 `10000+10000` warmup 参数保存在三个 run 的 `config.json` 中。由于当前代码会识别 reward 签名变化并重置旧 reward critic，重新执行旧命令将成为一个不同实验，不能覆盖或补写 R3 结果。

选择时先把三个迁移起点 actor 复制到各自 R3 run directory，再执行原 40-reset validation selection；这保证所有继续训练 checkpoint 都必须胜过起点才能进入 final。当时预注册的通过条件是：selected checkpoint 在完整 final 上提高候选路径子集 success、减少 near-goal timeout/regression，且 pooled collision_any 不高于 S1-R2.1 的 `2/600`。R3 validation 未通过；后续训练状态审计又确认其恢复链路存在混杂，因此实际决议改为先修复训练基础设施并执行 R4，而不是把 R3 解释为 shaping 失败。

S1-R3 validation 实际结果：4301 选回 `520000`=`38/40`，4302 选回 `480000`=`30/40`，4303 选回 `620000`=`35/40`；三个起点均与 S1-R2.1 selected actor 相同。4301 的 `540000`、4302 的 `500000`、4303 的 `640000` 虽与各自起点 validation 成功率持平，但 deterministic tie-break 选择更早 step；其余点明显退化。因此 R3 未通过，不运行新的 full final。

训练状态审计随后发现，R3 在 reward 定义改变后继续加载了按旧 reward 训练的 reward critic；旧 checkpoint 同时缺少 target critic、optimizer、replay、训练步数和 RNG。故上述结果保留为旧恢复链路的失效证据，不能独立归因为 terminal progress reward 无效。

### S1-R4：修复后的静态终端迁移（已完成，当前进度）

R4 先修复训练正确性，再验证整体方案。新的 checkpoint 格式保存 online/target critic、全部 optimizer、alpha/lambda/cost EMA、actor reference 和训练 RNG；replay 另存为压缩 `.npz`。加载时对 reward 和完整 risk 配置生成签名，不兼容的 critic 自动重置。time-limit truncation 只结束 episode，不再切断 Bellman bootstrap。旧 checkpoint 无法补出缺失的 optimizer/replay，因此首轮 R4 必须重建；新 checkpoint 恢复完整学习状态，但中途 checkpoint 恢复会从新的 episode 继续，不宣称仿真器逐步状态等价。

R4 固定设置如下：

- 三个 actor 从 R2.1 selected step `520000/480000/620000` 各训练 `200000` steps；
- terminal progress reward 保持 radius `0.12 m`、weight `30.0`，旧 reward critic因签名不匹配自动重置；
- 前 `50000` 步只收集 replay，接着 `50000` 步只更新 critic，最后 `100000` 步才更新完整 SAC；
- actor/critic/alpha learning rate 为 `1e-5/1e-4/1e-5`，actor output anchor 权重为 `10.0`；
- 一个零速度静态障碍物，场景为原始 `random`；25% episode 从固定 36-reset 失败清单抽样，75% 保持完整随机采样；
- replay 中 50% batch 配额按完整成功/失败轨迹分层，其余保持随机采样；
- 安全过滤器、动态障碍物、viability、recovery、relaxation 和 maximin 全部关闭。

R4 是基础设施、数据覆盖和稳定性机制的整体恢复实验，不是单因素消融。实际训练、validation 选点、完整 `9001--9200` final 和失败诊断均已完成；结果显示整体方向有效，但仍不足以进入 S2。后续若要声明 focused reset、stratified replay、anchor 或 terminal reward 的单独贡献，仍需单因素消融。

R4 validation 选中 4301 step `660000`、4302 step `620000`、4303 step `740000`，validation success 分别为 `36/40=90.0%`、`33/40=82.5%`、`33/40=82.5%`。full final 结果如下：

| actor | selected step | full final success | timeout | collision_any / capsule / physical | mean final error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4301 | 660000 | 176/200 = 88.0% | 24 | 0 / 0 / 0 | 0.0714 m |
| 4302 | 620000 | 173/200 = 86.5% | 27 | 0 / 0 / 0 | 0.0738 m |
| 4303 | 740000 | 167/200 = 83.5% | 33 | 0 / 0 / 0 | 0.0821 m |
| pooled | — | 516/600 = 86.0% | 84 | 0 / 0 / 0 | 0.0758 m |

相较 S1-R2.1，R4 full final 净增 19 个成功，collision_any/capsule/physical 从 `2/2/1` 降为 `0/0/0`；静态候选路径子集从 `443/483=91.7%` 提升到 `461/483=95.4%`。因此 R4 是当前静态障碍物的最新 final 参考，但仍不能宣称 99% 静态避障能力。

R4 的剩余失败全部为 timeout：全量失败包含 `nonconvergent_timeout` 42、`near_goal_timeout` 16、`near_goal_regression` 13、`low_motion_stall` 13。静态候选路径找到的 483 条 episode 中仍有 22 条失败，其中 `near_goal_timeout` 9、`near_goal_regression` 9、`nonconvergent_timeout` 4，且没有任何候选 reset 是三 actor 全部失败。另有 16 个 reset 为三 actor 全部 timeout，均属于 `not_found` 或 `not_checked_due_to_ik`。统一解释为：候选子集失败主要是 actor 局部速度场、终端精度和回退控制问题；全量中的三 actor 全失败 reset 应作为目标/场景可行性边界单独保留。

基于 R4 的当前决议：S2 动态障碍物继续暂停；不得删除 reset、放宽 `success_tolerance=0.055 m`、启用 safety filter/recovery 或把候选标签当作筛选条件。当前下一步不再只是泛泛的“修静态失败”，而是按三步走：

1. 先跑终端伺服诊断，确认近目标段的执行几何是否可收敛。
2. 再跑 failure-neighborhood jitter 训练，专门压 `0.055--0.08 m` 附近的停滞和回退。
3. 再从 `static_feasibility_precheck.json` 导出 waypoint teacher，作为后续示范数据。

这三步都只是在静态阶段内收口，不改变 final 口径，也不授权 S2。

### 当前三步命令

#### 1. 终端伺服诊断

```bash
python scripts/evaluate_terminal_servo.py \
  --config configs/experiments/reaching_incremental/s1_static_terminal_failure_refine_seed4301.yaml \
  --checkpoint outputs/reaching_incremental/s1_static_terminal_repaired_finetune/train/seed_4301/link_fixed_static_terminal_repaired_finetune_seed4301_steps720000/actor_step_660000.pt \
  --seed-manifest configs/experiments/reaching_incremental/manifests/s1_r4_static_candidate_failure_union_v1.json \
  --episodes 21 \
  --output outputs/reaching_incremental/s1_static_terminal_failure_refine/diagnostics_terminal_servo/seed_4301.csv \
  --trace-output outputs/reaching_incremental/s1_static_terminal_failure_refine/diagnostics_terminal_servo/seed_4301_traces
python scripts/evaluate_terminal_servo.py \
  --config configs/experiments/reaching_incremental/s1_static_terminal_failure_refine_seed4302.yaml \
  --checkpoint outputs/reaching_incremental/s1_static_terminal_repaired_finetune/train/seed_4302/link_fixed_static_terminal_repaired_finetune_seed4302_steps680000/actor_step_620000.pt \
  --seed-manifest configs/experiments/reaching_incremental/manifests/s1_r4_static_candidate_failure_union_v1.json \
  --episodes 21 \
  --output outputs/reaching_incremental/s1_static_terminal_failure_refine/diagnostics_terminal_servo/seed_4302.csv \
  --trace-output outputs/reaching_incremental/s1_static_terminal_failure_refine/diagnostics_terminal_servo/seed_4302_traces
python scripts/evaluate_terminal_servo.py \
  --config configs/experiments/reaching_incremental/s1_static_terminal_failure_refine_seed4303.yaml \
  --checkpoint outputs/reaching_incremental/s1_static_terminal_repaired_finetune/train/seed_4303/link_fixed_static_terminal_repaired_finetune_seed4303_steps820000/actor_step_740000.pt \
  --seed-manifest configs/experiments/reaching_incremental/manifests/s1_r4_static_candidate_failure_union_v1.json \
  --episodes 21 \
  --output outputs/reaching_incremental/s1_static_terminal_failure_refine/diagnostics_terminal_servo/seed_4303.csv \
  --trace-output outputs/reaching_incremental/s1_static_terminal_failure_refine/diagnostics_terminal_servo/seed_4303_traces
```

#### 2. failure-neighborhood jitter 训练

```bash
python scripts/train.py \
  --config configs/experiments/reaching_incremental/s1_static_terminal_failure_refine_seed4301.yaml \
  --resume-actor outputs/reaching_incremental/s1_static_terminal_repaired_finetune/train/seed_4301/link_fixed_static_terminal_repaired_finetune_seed4301_steps720000/actor_step_660000.pt \
  --reset-agent-state \
  --start-step 660000
python scripts/train.py \
  --config configs/experiments/reaching_incremental/s1_static_terminal_failure_refine_seed4302.yaml \
  --resume-actor outputs/reaching_incremental/s1_static_terminal_repaired_finetune/train/seed_4302/link_fixed_static_terminal_repaired_finetune_seed4302_steps680000/actor_step_620000.pt \
  --reset-agent-state \
  --start-step 620000
python scripts/train.py \
  --config configs/experiments/reaching_incremental/s1_static_terminal_failure_refine_seed4303.yaml \
  --resume-actor outputs/reaching_incremental/s1_static_terminal_repaired_finetune/train/seed_4303/link_fixed_static_terminal_repaired_finetune_seed4303_steps820000/actor_step_740000.pt \
  --reset-agent-state \
  --start-step 740000
```

#### 3. 静态 waypoint teacher 导出

```bash
python scripts/export_static_waypoint_teacher.py \
  --feasibility outputs/reaching_incremental/s1_static_obstacle_finetune/static_feasibility_precheck.json \
  --output-json outputs/reaching_incremental/s1_static_terminal_failure_refine/teacher/static_waypoint_teacher_summary.json \
  --output-jsonl outputs/reaching_incremental/s1_static_terminal_failure_refine/teacher/static_waypoint_teacher.jsonl \
  --dt 0.05
```

以下为 R4 已执行训练命令归档。三个训练当时可以并行运行；命令只用于复现本轮，不是当前要运行的步骤：

```bash
RUN=outputs/reaching_incremental/s1_static_terminal_repaired_finetune/train/seed_4301/link_fixed_static_terminal_repaired_finetune_seed4301_steps720000
mkdir -p "$RUN"
cp -n outputs/reaching_incremental/s1_static_mixed_stateful_finetune/train/seed_4301/link_fixed_static_mixed_stateful_finetune_seed4301_steps540000/actor_step_520000.pt "$RUN/actor_step_520000.pt"
python scripts/train.py --config configs/experiments/reaching_incremental/s1_static_terminal_repaired_finetune_seed4301.yaml --resume-actor outputs/reaching_incremental/s1_static_mixed_stateful_finetune/train/seed_4301/link_fixed_static_mixed_stateful_finetune_seed4301_steps540000/actor_step_520000.pt --resume-state outputs/reaching_incremental/s1_static_mixed_stateful_finetune/train/seed_4301/link_fixed_static_mixed_stateful_finetune_seed4301_steps540000/agent_state_step_520000.pt --start-step 520000
```

```bash
RUN=outputs/reaching_incremental/s1_static_terminal_repaired_finetune/train/seed_4302/link_fixed_static_terminal_repaired_finetune_seed4302_steps680000
mkdir -p "$RUN"
cp -n outputs/reaching_incremental/s1_static_mixed_stateful_finetune/train/seed_4302/link_fixed_static_mixed_stateful_finetune_seed4302_steps780000/actor_step_480000.pt "$RUN/actor_step_480000.pt"
python scripts/train.py --config configs/experiments/reaching_incremental/s1_static_terminal_repaired_finetune_seed4302.yaml --resume-actor outputs/reaching_incremental/s1_static_mixed_stateful_finetune/train/seed_4302/link_fixed_static_mixed_stateful_finetune_seed4302_steps780000/actor_step_480000.pt --resume-state outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4302/link_fixed_static_obstacle_finetune_seed4302_steps480000/agent_state_step_480000.pt --start-step 480000
```

```bash
RUN=outputs/reaching_incremental/s1_static_terminal_repaired_finetune/train/seed_4303/link_fixed_static_terminal_repaired_finetune_seed4303_steps820000
mkdir -p "$RUN"
cp -n outputs/reaching_incremental/s1_static_mixed_stateful_finetune/train/seed_4303/link_fixed_static_mixed_stateful_finetune_seed4303_steps620000/actor_step_620000.pt "$RUN/actor_step_620000.pt"
python scripts/train.py --config configs/experiments/reaching_incremental/s1_static_terminal_repaired_finetune_seed4303.yaml --resume-actor outputs/reaching_incremental/s1_static_mixed_stateful_finetune/train/seed_4303/link_fixed_static_mixed_stateful_finetune_seed4303_steps620000/actor_step_620000.pt --resume-state outputs/reaching_incremental/s1_static_mixed_stateful_finetune/train/seed_4303/link_fixed_static_mixed_stateful_finetune_seed4303_steps620000/agent_state_step_620000.pt --start-step 620000
```

运行开始时三组日志要求显示 `reward_critics_loaded=False`、`cost_critics_loaded=True`、`optimizers_loaded=False`、`replay=fresh`，随后依次显示 50000 步 collect-only 和 50000 步 critic-only。该要求作为 R4 训练链路审计条件保留。

训练完成后已只用 `9301--9340` validation manifest 选点，并把起点 actor 作为候选；完整 final 已保留所有 `9001--9200` reset。R4 的 full final 虽有净提升且碰撞为零，但候选子集仍只有 `461/483=95.4%`，未达到进入动态障碍物的可信门槛；因此 R4 之后仍不进入 S2。

### S2：低速动态障碍物，仍关闭安全过滤器

**当前状态**：暂停。

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

## 5. 已执行命令归档：S1-R

本节命令已执行完成，仅用于复现 S1-R，不是当前下一步。S1-R4 也已执行完成并在第 3 节归档；当前下一步仍停留在静态障碍物内分析和修复 R4 剩余失败。

S1-R 当时的执行顺序为：

1. 从三个 v2 selected actor 继续静态障碍物训练 `200000` steps；
2. 只使用 `9301--9340` validation manifest 选择 checkpoint；
3. 三个新 actor 各跑固定 `9001--9200` final；
4. 汇总全量结果和失败 reset；
5. 只有 S1-R 通过后才进入 S2 低速动态障碍物。

本轮不启用安全过滤器、不启用 recovery；v2 actor 原文件不修改。

### S1-R 历史命令

以下命令仅用于历史复现，假设从仓库根目录执行并使用当前环境中的 `python`。不要用它们代替 R4 结果或后续静态失败修复命令。

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
