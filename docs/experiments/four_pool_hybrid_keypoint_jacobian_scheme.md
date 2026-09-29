# 四池 Hybrid Keypoint + Jacobian + Auto-PCR 实现说明

更新日期：2026-09-29。本文以当前工作区代码与配置为准，区分配置默认行为、L0 四池定向续训、升档后的 Current/History 行为，以及已有实验结果。本文更新不启动训练或评估，不修改算法。

## 1. 主线与实际可达路径

正式实验配置只有 `configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml`，其直接继承 `../default.yaml`；不再继承已删除的旧 thesis 实验配置。训练入口 `validate_training_architecture()` 限制为 Hybrid Keypoint + Jacobian + Auto-PCR：166 维 Hybrid observation、unified keypoint reward、开启 CHAIN-PCR 和 Auto-PCR。

但“架构唯一”不等于“四池 replay 默认开启”。当前启动方式仍决定不同的数据路径：

| 启动方式/阶段 | 实际行为 |
| --- | --- |
| 直接使用当前 YAML 新建 S0，或恢复非四池 checkpoint | 默认 joint-pose replay：Recent + Success + Semantic，可包含 Previous；不是四池 |
| 原始 1M 完整 checkpoint + `--bad-state-starts`，不提供 seed | 增加定向 rollout，但 replay 仍为旧配比；不是四池 |
| 原始 1M 完整 checkpoint + `--bad-state-starts` + `--four-pool-seed` | 初始化 L0 四池，Recent 清空重建，Success 不再单独采样 |
| 恢复 L0 四池完整 checkpoint | 从 checkpoint 继承四池，仍须提供原 bad-state bank，不重复加载 seed |
| 四池 L0 用 `--promote-to-level 1` 升档 | 转成 Current/History 模式，不再使用四池固定配额 |
| `--initialize-actor-from` + `--start-level` | 只导入 Actor；Critic、replay、计数器重新开始，不是完整继承式续训 |

`configs/default.yaml` 及其基础配置、配置校验中仍有旧 `method`、predictive risk、执行平滑等字段。它们并未全部清理，但不能据此认为正式 thesis 入口仍运行旧算法。S0/S1/S2 是同一架构的串行课程阶段，不是三个旧方案。

实际训练链路为：

```text
实验 YAML + default.yaml
  -> train_thesis_homotopy.py：参数/架构检查，构造或恢复训练状态
  -> HomotopyCurriculum：生成场景、精度、采样范围与安全权重合同
  -> ThesisHomotopyEnv / ParallelThesisEnvPool：reset、仿真、观测、成功判断
  -> GaussianActor：6 维关节速度动作
  -> HomotopyReplayBuffer：写入、分池、抽样、重新计算 reward
  -> ThesisSACAgent：双 Critic、Actor + Auto-PCR、alpha、Target 更新
  -> 完整 checkpoint / Actor 权重
  -> evaluate_thesis_homotopy.py：按指定档位冻结评估
```

## 2. 来源与文件约定

以下为已有实验使用的路径，文件名不证明当前文件存在、池已满或训练已完成：

| 项目 | 路径（相对项目根目录） |
| --- | --- |
| 原始完整 1M checkpoint | `outputs/serial_hybrid_keypoint_jacobian_auto_chain/s0_seed11001_hybrid_keypoint_jacobian_auto_chain_4000k/checkpoints/step_1000000.pt` |
| 85 个局部坏状态 | `outputs/serial_hybrid_keypoint_jacobian_auto_chain/bad_state_starts_1m_seed51001.pt` |
| 冻结种子池 | `outputs/serial_hybrid_keypoint_jacobian_auto_chain/four_pool_seed_frozen_1m_20260929.pt` |
| L0 四池 run | `outputs/serial_hybrid_keypoint_jacobian_auto_chain/s0_seed11001_four_pool_from_1m_200k/` |
| L1 升档来源 | 上述 L0 run 的 `checkpoints/step_0025000.pt` |
| L1 当前数据专用 run | `outputs/serial_hybrid_keypoint_jacobian_auto_chain/s0_seed11001_four_pool_25k_inherited_level1_1000k/` |

完整 `step_*.pt` 保存 Actor、Q1/Q2、Target Q1/Q2、alpha、优化器、Auto-PCR 参考网络与 EMA、replay、curriculum、计数器及 RNG/环境恢复信息。`actor_step_*.pt` 保存 Actor state dict，用于冻结评估，不能替代 `--resume` 的完整 checkpoint。

`step_0025000.pt` 表示该训练块新增 25k transition，不是从零训练 25k；`--steps` 是该块请求新增的 transition 数，受阶段剩余预算约束。8 个环境每轮最多贡献 8 条 transition，不是 8 倍的独立训练预算。

## 3. 环境、课程与成功判定

| 参数 | 当前配置/实现 |
| --- | --- |
| 机器人 | UR5 六关节，工具参考 link 为 `wrist_3_link` |
| 仿真/控制 | 物理步长 `1/240 s`，每动作 12 个子步，控制周期 `0.05 s` |
| Episode horizon | `240` 控制 step，最多 12 s；冻结评估可单独覆盖 |
| 动作尺度/电机力 | 每关节目标速度尺度 `0.7 rad/s`；motor force `90` |
| 外部安全距离 | `d_safe=0.12 m`；障碍半径 `0.075 m`，速度配置 `0.1 m/s` |
| 自碰撞 | 安全距离 `0.005 m`、查询距离 `0.25 m`、TTC 上限 `3 s` |
| 自安全课程 | `lambda_self` 从 `0.2` 到 `1.0`，ramp `50k`，满权重最少 `25k` transition；实际进度由课程门控 |
| 自安全 QP | `self_safety_projection.enabled: false`，当前不投影动作 |
| 保存/训练 | 每 `25k` 保存；配置每块 `200k`，阶段硬预算 `5M`；正式 S0 默认 8 环境 |

S0 的 rollout 场景全为 `none`；S1 为 25% none + 75% static；S2 为 20% none + 30% static + 50% dynamic。S1/S2 必须从上一阶段完整 checkpoint 继承，当前多环境训练入口只支持 S0。场景 rollout 比例不等于 replay 的场景 batch 比例。

当前精度课程按 `--level-index` 使用以下五档。全部档位的目标距离范围都是 `[0.03,0.70] m`，目标姿态差范围都是 `[0.03,3.141592] rad`，goal scale 都为 `1.0`：

| 索引 | 位置阈值 eps_p（m） | 姿态阈值 eps_R（rad） | 内部 orientation scale/分池标记 |
| --- | ---: | ---: | ---: |
| L0 / 0 | 0.1000 | 0.3000 | 0.00 |
| L1 / 1 | 0.0800 | 0.2400 | 0.25 |
| L2 / 2 | 0.0640 | 0.1920 | 0.50 |
| L3 / 3 | 0.0512 | 0.1536 | 0.75 |
| L4 / 4 | 0.0500 | 0.1000 | 1.00 |

这里升档主要收紧精度，不扩大目标范围。内部 `orientation_scale=0` 不表示 L0 只采零姿态差；task-space 合同已经明确完整角度范围。采样器使用位置和姿态各 10 档的 100 cell；正常训练的固定 mixture 为 50% 全 cell 均匀 + 25% O5–O9 + 25% O7–O9，adaptive 采样关闭。这是采样目标分布，实际可达性拒绝采样仍可能影响接受分布。

`ThesisHomotopyEnv.step()` 的成功条件为：无硬失败/障碍失败，位置误差 `<= eps_p` 且姿态误差 `<= eps_R`，连续满足 `5` 个控制 step。正常合同达到成功即终止；硬失败也终止，未终止且达到 horizon 则截断。**当前成功判定没有额外要求关节或末端速度低于 `stable_success` 的配置值**，不能把这些遗留速度阈值写成终止条件。目标速度命令限制为 `+-0.7 rad/s`；这不是对仿真测得实际 qdot 的硬钳制。

升档为手动模式，最低当前档训练预算 `100k` transition；`--allow-early-promotion` 可显式绕过预算并记录。`--approve-orientation-promotion` 使用最近 probe 的通过结论；`--promote-to-level` 是显式档位决策，不等于已经通过测量 gate。

## 4. 网络输入与缩放

输入为 `166` 维 float32，Box 声明范围 `[-1,1]`。以下索引采用 Python 左闭右开区间，拼接顺序与 `build_thesis_observation()` 一致。`clip` 指按表中范围裁剪；代码不是拼接后统一再做一次全局 clip。

| 索引 | 维数 | 量与当前处理 |
| --- | ---: | --- |
| `[0:6]` | 6 | q：`clip(2*(q-lower)/(upper-lower)-1,-1,1)`，按 URDF 关节上下限归一化 |
| `[6:12]` | 6 | qdot：`clip(qdot/0.7,-1,1)` |
| `[12:21]` | 9 | 三个非共线关键点的目标位置减末端位置，除 `1.0 m` 后 clip；立方体边长 `0.50 m` |
| `[21:75]` | 54 | `J_KP`（9 x 6），除 `1.0` 后 clip，按行展开 |
| `[75:78]` | 3 | 显式位置误差：直接以米输入并 clip，**没有另除位置尺度** |
| `[78:81]` | 3 | SO(3) 旋转误差向量：除 pi 后 clip |
| `[81:82]` | 1 | 位置误差范数（米）：直接 clip 到 `[0,1]` |
| `[82:83]` | 1 | 姿态误差范数：除 pi，clip 到 `[0,1]` |
| `[83:86]` | 3 | 末端线速度：除 `1.0 m/s` 后 clip |
| `[86:89]` | 3 | 末端角速度：除 `4.2 rad/s` 后 clip |
| `[89:90]` | 1 | goal scale：`2*clip(scale,0,1)-1` |
| `[90:91]` | 1 | eps_p：`2*clip(eps_p/0.20,0,1)-1` |
| `[91:92]` | 1 | eps_R：`2*clip(eps_R/pi,0,1)-1` |
| `[92:93]` | 1 | 剩余时间：`2*clip((horizon-step_count)/horizon,0,1)-1` |
| `[93:111]` | 18 | 六 link 的障碍相对向量（米）：直接 clip，无另外尺度除法 |
| `[111:114]` | 3 | 障碍位置：low=`[.22,-.62,.18]`、high=`[.72,.62,.62]`，映射到 `[-1,1]` 后 clip |
| `[114:117]` | 3 | 障碍速度：除 `0.1 m/s`；该项没有独立 clip |
| `[117:123]` | 6 | 外部距离：`2*(clip(d,-.20,.80)+.20)-1` |
| `[123:129]` | 6 | 外部 TTC：`clip(ttc,0,3)/3` |
| `[129:135]` | 6 | 外部接近速度：除 observation 常量 `1.0 m/s`，clip 到 `[0,1]` |
| `[135:141]` | 6 | 外部风险：clip 到 `[0,1]` |
| `[141:142]` | 1 | 障碍 presence：0/1，不做缩放 |
| `[142:148]` | 6 | 自碰撞距离：`2*(clip(d,-.02,.25)+.02)/.27-1` |
| `[148:154]` | 6 | 自碰撞 TTC：`clip(ttc,0,3)/3` |
| `[154:160]` | 6 | 自碰撞接近速度：除 observation 常量 `1.0 m/s`，clip 到 `[0,1]` |
| `[160:166]` | 6 | 自碰撞风险：clip 到 `[0,1]` |

S0 没有外部障碍，但仍保留 49 维外部槽位：相对向量/位置/速度/接近速度/风险/presence 为零，距离和 TTC 为一。自碰撞槽位仍由实际几何计算。接近速度 observation 的 `1.0` 尺度不要与自碰撞风险计算配置的 `approach_velocity_scale: 0.7` 混淆。

`J_KP` 由几何 Jacobian 和关键点偏移计算，是 Actor 可见特征；它不作为解析 IK 控制器输出，也没有 IK teacher 或残差动作。`include_orientation_error_vector: false` 只禁止旧独立观测模式，不意味着 Hybrid 中不包含 `[78:81]` 的旋转误差向量。

## 5. 输出、网络与 SAC/Auto-PCR

Actor 输出六维 tanh-Gaussian 动作：训练从高斯重参数化采样后 tanh，冻结评估为 `tanh(mean)`。环境执行：

```text
a = clip(actor_action, -1, 1)
precision_scale = clip(max(rho_p/0.05, rho_R/0.15), 0.10, 1.0)
joint_target_velocity = clip(0.7*a*precision_scale, -0.7, 0.7)
```

precision scale 当前开启；其 `.05/.15` 是固定控制尺度，不随 L0/L1 成功阈值自动替换。随后经过自安全投影入口，但当前投影关闭，返回未投影命令。

| 项目 | 当前实现/有效配置 |
| --- | --- |
| Actor backbone | `166 -> 256 ReLU -> 256 ReLU -> 256 Linear`，最后一层无激活；mean/log_std 两个 `256 -> 6` 头 |
| Actor 分布 | log_std 限制 `[-20,2]`；tanh 动作并做 log probability Jacobian 修正 |
| Critic | Q1/Q2 独立，各为 `172 -> 256 ReLU -> 256 ReLU -> 1`；输入为 observation + action；各有 Target |
| Critic loss/target | 两个 Huber loss 之和；target 使用 Target 双 Q 最小值减 entropy 项，terminal mask 屏蔽 bootstrap |
| SAC | `gamma=.99`，`tau=.005`，target entropy `-6`；alpha 初始 `.2`，自动学习 |
| Adam 学习率 | Actor `1e-4`，Critic `1e-4`，alpha `3e-4` |
| 更新 | batch 1024；UTD `.25`；Actor 每 2 次 Critic 更新一次，alpha 与 Actor 同次更新；梯度裁剪 Actor/Critic 各 10 |
| Warmup | 初始随机动作 3000 transition；`update_after=1000`，同时必须有完整 batch；完整恢复不重新从零 warmup |
| Auto-PCR | 初始系数 1；目标惩罚/SAC loss 比 `.003`，EMA `.99`，自动系数范围 `[1e-4,100]`；随后还限制当次惩罚不超过 `abs(SAC loss)` 的 10% |

Auto-PCR 对 Actor 加 `KL(old Gaussian || current Gaussian)`，比较 tanh 前的对角高斯；Critic 和 alpha 不加此项。参考 Actor 是递进的链式快照：每次 Actor 更新后，reference 被换成该次更新前的 Actor，**不是固定的 1M Actor**。Actor 更新时额外按同一 replay 规则抽一批 1024 observation 供 PCR 使用，再抽训练 batch；额外抽样不额外增加一次 UTD 梯度更新。

## 6. 实际生效的奖励

启用 `pose_objective: unified_keypoint`。令三个关键点的平均距离为 D_KP，下一状态跟踪质量 `T=mean_i(exp(-d_i(next)/.05))`，本 episode 成功阈值为 eps_p/eps_R：

```text
precision_quality = exp(-max(rho_p(next)/eps_p, rho_R(next)/eps_R))
velocity_cost = clip(mean((qdot_after/0.7)^2), 0, 1)
smooth_cost = clip(mean(((qdot_after-qdot_before)/0.7)^2), 0, 1)
r_goal = .20*T + 10*(D_KP(current)-D_KP(next))
         + .20*precision_quality - .04*velocity_cost - .01*smooth_cost
         + 20*task_reached
r = r_goal - hard_penalty - timeout_guard
           - external_safety_penalty - self_safety_penalty - terminal_guard
```

精度项 temperature 为 1，`use_episode_tolerance_for_joint_precision: true`，所以升档会改变该项的 eps_p/eps_R，但不改变公式及系数。关键点 progress 是直接距离差，没有在此项乘 gamma。

硬失败（自碰撞、环境碰撞、关节越界）罚 20；未成功且无硬失败的 timeout 罚 2。外部/自碰撞安全组都为 `(2*risk+8*clearance_violation)/10`，分别再乘 `xi*.05`、`lambda_self*.05`；风险与距离违例均裁剪到 `[0,1]`。`clearance_violation=clip((safe_distance-min_distance)/safe_distance,0,1)`，外部/自安全距离分别 `.12/.005 m`。终止障碍碰撞且非硬失败的 terminal guard 罚 20。

配置和日志仍保留 position/orientation shaping、fine reward、joint-precision progress、partial precision、leave-tolerance 等字段；当前 `r_goal` 不把这些诊断项相加。不能把配置出现的所有系数都当成生效奖励。

Replay 每条保存 observation、action、next observation、done、episode/step 和 29 个 raw 特征。训练抽样时重新计算 reward，安全权重采用当前 curriculum；精度阈值读取该 transition 保存的 episode 合同，**不是把所有历史样本重新标成新档位成功**。当前 SAC 是 reward-only，Batch 的 cost 不对应另一套 cost Critic。

## 7. L0 定向采集与四池

从原始 1M 完整 checkpoint 继续，正常 S0 rollout 占约 75% transition，bad-state 占约 25%。85 个中心是审计识别的低 alignment、停滞、crossing 后回弹附近完整状态，不是任意 `0.6–1.0 rad` 状态。训练入口要求中心 rho_R 在该范围，并检查关节、目标、观测维度、场景与 episode 时钟。

`reset_bad_state()` 继承中心 q、qdot、目标和原 step_count，保留剩余 episode 预算；恢复训练 RNG/合同后做局部扰动：q 每维 `+-0.015 rad`，qdot 每维 `+-0.02 rad/s`，目标位置每维 `+-0.004 m`，目标 Euler 扰动每维 `+-0.012 rad`。最多尝试 64 次，检查关节/速度范围、自碰撞，以及位姿误差和 q/qdot/J_KP 的归一化邻域距离。Jacobian 由扰动后的状态重新计算，不是独立随机扰动矩阵。正常/定向轨迹长度 EMA 用于调节起点选择概率，目标是 transition 占比，而非固定 25% episode。

`build_four_pool_seed.py` 用原始 Actor 做冻结 deterministic rollout：正常 episode 按三个姿态层 40%/30%/30% 调度，默认总数 1200，只接纳成功 episode；每中心另做一次扰动 bad-state rollout 生成初始 Frontier。种子来源路径、checkpoint SHA256、bad-state bank 路径在训练入口校验，thesis/SAC 合同也要求匹配。

| 池 | 容量与保存 | 每 1024 batch |
| --- | --- | ---: |
| Recent / Plasticity | 30,000 transition FIFO；启用四池时重建，接收全部当前正常/定向数据 | 384 |
| Anchor / Stability | 目标 24,000：O0–O4 9,600，O5–O6 7,200，O7–O9 7,200；原 Actor 正常成功 episode，训练期间固定 | 256（102/77/77） |
| Frontier / Bad-state | 85 中心各 176 FIFO，上限 14,960；保存前 40 step、困难事件和成功恢复末段 | 192 |
| Semantic / Coverage | 100 cell，每格最多 8 episode，每条最多 64 个线性稀疏 transition；上限 51,200 | 192 |

Anchor 按完整 episode 写入且不超容量，因此实际大小不保证恰好满池；每层至少 256、每中心至少一条是 seed 接纳检查。Recent/Anchor 采用 episode-balanced 抽样；Frontier 每中心随机最多取 3 条候选，再打乱取 192；不是动态按成功率加权。Semantic 按非空 cell 轮转，再选择 episode/transition；保存质量是末态 `max(rho_p/.05,rho_R/.10)`，分母固定，不随 L1 阈值变化，允许较好的失败轨迹而非只收成功。

Frontier 保存条件包括：起点后前 40 step；rho_R 在 `[.6,1]` 且单步改善 `<=.0005` 的停滞；从 `<.6` 到 `>=.6` 的回弹；同角度范围 alignment `<.25`；成功恢复时补充起点后至少 40 step 且属于末 20 step 的 transition。训练期间这一额外写入只对标记的 targeted episode 生效，不是扫描所有正常 S0 轨迹自动新增中心。

四池实际是固定 `384+256+192+192=1024`，不会因为 YAML 的 semantic fraction `.20` 而改成 20%；这里是 18.75%。Success 独立池重建为容量 1 的占位池，不再写入成功 episode，也不占 batch。一个 transition 可同时出现在 Recent、Frontier、Semantic 中；跨池并未全局去重。四池若无法凑齐配额会报错，不会自动回退到旧 success replay。

默认旧 replay 与四池不同：Success 配额按成功 episode 数调度，少于 5 个为 0，5/20/50 个门槛对应 `.05/.15/.30`，Semantic 目标 `.20`，Previous 配置为 0，剩余由 Recent 填充；冷启动有补齐逻辑。保留此路径意味着代码当前还不是“四池唯一数据实现”。

## 8. L1 完整继承与 Current/History

四池 L0 显式升 L1 时，网络、alpha、优化器及 PCR 状态完整继承。旧 Recent 首先归档到 History，余量按约 3/4 Anchor、剩余 Frontier、再用可用 Semantic 补齐；History 容量提升至至少 60,000。空间有限，不能说所有池中的所有旧数据都完整搬入。随后清掉独立 Anchor/Frontier 引用，`four_pool_enabled=false`、`four_pool_history_mode=true`，新档 Current 从空池开始。

默认 `--promoted-current-fraction` 为 `.5`，即 Current/History 约 50:50；Current 不足时可增加历史样本比例。已使用的 L1 分支设置为 `1.0`：每 batch 1024 条全部为当前档数据，History 保存但不抽样；Current 未满 1024 条则不更新。Semantic 仍可更新，但在该采样模式没有独立配额。

L0 的 bad-state bank 不继续用于 L1 rollout，升档入口不允许同时传 `--bad-state-starts`。因此“25% 定向数据 + 四池固定 batch”仅描述当前 L0 实验，不能直接套到 L1。以下命令与既有 L1 分支对应，不在本文更新过程中执行：

```bash
conda activate rl
python scripts/train_thesis_homotopy.py \
  --config configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml \
  --stage s0 \
  --resume outputs/serial_hybrid_keypoint_jacobian_auto_chain/s0_seed11001_four_pool_from_1m_200k/checkpoints/step_0025000.pt \
  --promote-to-level 1 \
  --promoted-current-fraction 1.0 \
  --steps 1000000 --seed 11001 --num-envs 8 \
  --run-name s0_seed11001_four_pool_25k_inherited_level1_1000k
```

S1/S2 的 replay 则按场景分配：1024 batch 在 S1 为 none/static/dynamic=`256/768/0`，S2 为 `204/308/512`，不足可跨场景重分配；none 样本走保留的无障碍数据采样，不沿用 L0 四池配额。它们仍是同一训练架构的后续阶段。

## 9. 冻结评估与已有结果

评估使用当前 YAML 构造环境与网络，再加载 Actor。完整 checkpoint 可推断档位/阶段；Actor-only checkpoint 必须提供 `--level-index`，且 S1/S2 Actor-only 应显式指定场景。评估不更新网络、不采训练 replay；正常 frozen probe 按 100 cell 均衡采样，不能用在线训练成功率替代。

标准 L1 评估为 240 step、连续 5 step 保持、eps_p=.08 m/eps_R=.24 rad，例如：

```bash
conda activate rl
python scripts/evaluate_thesis_homotopy.py \
  --config configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml \
  --checkpoint outputs/serial_hybrid_keypoint_jacobian_auto_chain/s0_seed11001_four_pool_25k_inherited_level1_1000k/checkpoints/actor_step_0300000.pt \
  --level-index 1 --scene none --episodes 1000 --seed 51001 --num-envs 8 \
  --output outputs/serial_hybrid_keypoint_jacobian_auto_chain/s0_seed11001_four_pool_25k_inherited_level1_1000k/evaluations/actor_step_0300000_level1_seed51001.json
```

加 `--max-episode-steps 500` 并使用不同 output 路径可做 500-step 诊断，精度仍是 L1；这也改变 observation 的剩余时间比例，不只是额外允许 260 step。

当前保留的两个 JSON 记录了同一个 +300k L1 Actor、1000 episode、seed 51001 的已有结果，本次没有重新评估：

| 评估 horizon | 成功率 | Timeout | 最差 cell 成功率 | 文件名（上述 run 的 evaluations 下） |
| --- | ---: | ---: | ---: | --- |
| 240 | 82.7% | 16.8% | 30.0% | `actor_step_0300000_level1_seed51001.json` |
| 500 | 83.9% | 15.3% | 30.0% | `actor_step_0300000_level1_seed51001_horizon500.json` |

独立评估脚本的 `passed` 条件还区分场景：

| 场景 | 总体成功率下限 | Collision 上限 | Timeout 上限 | 共同要求 |
| --- | ---: | ---: | ---: | --- |
| none（S0） | 85% | 5% | 5% | 每 cell 至少 60%，joint_limit_rate 必须为 0 |
| static | 90% | 3% | 10% | 同上 |
| dynamic | 80% | 5% | 20% | 同上 |

课程升档另有上一档保持检查（配置成功率下限 80%、碰撞上限 5%），自安全完成另有 1% collision ceiling；这些不应与独立评估的单个 `passed` 混为一谈。上述两次结果仅总体和最差 cell 已不足 S0 阈值，Timeout 和关节越界也未满足标准，不构成课程通过证明；500-step 结果也不等同标准 240-step 通过率。这里只记录已有单次实验，不能据此声称四池改善/遗忘的因果结论。

## 10. 代码核对入口

| 内容 | 源码 |
| --- | --- |
| 正式参数、精度档与场景课程 | `configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml` |
| 启动分支、完整恢复、手动升档、定向概率、PCR 参考 batch | `scripts/train_thesis_homotopy.py` |
| 166 维顺序/缩放、J_KP、实际 reward | `src/rl_risk_sac/tasks/thesis_reaching.py` |
| 动作执行、成功/超时、坏状态邻域 reset | `src/rl_risk_sac/envs/thesis_homotopy_env.py` |
| 并行环境采集 | `src/rl_risk_sac/envs/parallel_thesis_env.py` |
| Actor/Critic 结构 | `src/rl_risk_sac/algorithms/networks.py` |
| SAC/Auto-PCR、alpha 与优化器恢复 | `src/rl_risk_sac/algorithms/thesis_sac.py` |
| 课程合同与门控 | `src/rl_risk_sac/algorithms/homotopy_curriculum.py` |
| 四池、旧 replay、升档 History、Semantic、重算 reward | `src/rl_risk_sac/algorithms/homotopy_replay.py` |
| 原始 frozen Actor 建 Anchor/Frontier | `scripts/build_four_pool_seed.py` |
| 冻结评估档位、场景与 horizon 覆盖 | `scripts/evaluate_thesis_homotopy.py` |

结论：当前正式模型主线是 Hybrid Keypoint + Jacobian + Auto-PCR；L0 四池是需要显式启用的数据方案，L1 Current/History 是其升档行为，代码仍保留默认旧 replay 与 Actor-only 初始化入口。描述当前实现时必须同时说明架构和实际 replay 模式，不能只凭 YAML 名称或 run 名称认定四池正在生效。
