# 连续实验方案（v8/v9/V10 历史与 V11 当前协议）

更新日期：2026-09-18

执行状态：V8/V9 已失败；V10 完成 600000 transitions 但在 `eta=0.75` 退化并硬预算停止；V11 已实现，必须从随机初始化开始。以下第 1～7 节保留版本演化依据，第 8 节是当前协议。

## 1. V8 状态空间契约（历史）

V8 策略 observation 为 98 维，所有分量均裁剪或缩放到可控范围：

| 区间 | 维数 | 内容 | 归一化 |
| --- | ---: | --- | --- |
| `0:6` | 6 | 关节角 `q` | 关节上下限映射到 `[-1,1]` |
| `6:12` | 6 | 关节速度 `qdot` | 除以动作上限 `0.7 rad/s` |
| `12:21` | 9 | 末端位置 + 末端旋转 6D | 位置按 1 m；旋转矩阵前两列 |
| `21:30` | 9 | 目标位置 + 目标旋转 6D | 同上 |
| `30:39` | 9 | 三维位置误差 + 相对旋转 6D | 位置按 1 m；旋转矩阵前两列 |
| `39:41` | 2 | 位置误差范数 + 姿态测地角 | 分别除以 1 m 与 `pi` |
| `41:47` | 6 | 末端线速度 + 角速度 | 分别除以 1 m/s 与 4.2 rad/s |
| `47:49` | 2 | 姿态课程系数 `eta` + 剩余时域 | 映射到 `[-1,1]` |
| `49:85` | 36 | 障碍相对向量、位置、速度、逐连杆距离、TTC | 固定物理尺度 |
| `85:97` | 12 | 六段连杆接近速度 + 六段连杆风险 | 各自裁剪到 `[0,1]` |
| `97` | 1 | 障碍存在标志 | `0/1` |

旋转 6D 的列顺序为 `[R[:,0], R[:,1]]`。它对四元数 `q/-q` 不变，避免四元数双覆盖的不连续；策略不再接收三维 `SO(3)` log。姿态误差仍按
`rho_R = ||Log(R_goal R_ee^T)||` 计算，因为测地角适合作为奖励标量和成功阈值，不承担连续姿态坐标的角色。

S0 中障碍相关 49 维使用确定的空场景占位值；这样 S0/S1/S2 始终共享同一个 actor，不在阶段切换时扩展网络。正式结果表明这里存在一个需要在下一版重新设计的边界：这些字段只描述外部障碍，不描述机器人自身连杆之间的间隙；S0 中策略只能由 `q` 隐式推断自碰撞风险，并在真实接触后收到稀疏硬失败信号。

## 2. V8 奖励契约（历史）

定义误差质量

```text
u_p = exp(-rho_p / sigma_p)
u_R = exp(-rho_R / sigma_R)
H(u) = 0.5 * (u + u^4)
Phi(s) = H(u_p) * (1 + beta * eta * H(u_R))
```

默认 `sigma_p=0.20 m`、`sigma_R=1.0 rad`、`beta=1`。`H` 同时保留远距离梯度并增强近目标分辨率；姿态质量以乘法形式受位置质量调制，不再使用分段的远/近姿态奖励门。

单步到达奖励为

```text
r_goal = 20 * (gamma * Phi(s') - Phi(s))
         + 20 * I(success)
         - 0.04 * clip(mean((qdot'/0.7)^2), 0, 1)
         - 0.01 * clip(mean_j(((qdot'[j]-qdot[j])/0.7)^2), 0, 1)
```

势函数使用与 SAC 相同的 `gamma=0.99`，避免来回运动通过未折扣差分刷取正奖励。成功条件仍为位置容差和姿态测地角容差同时满足。

安全部分保持独立：

```text
r = r_goal
    - xi * (2 * risk_max + 8 * clearance_violation)
    - 10 * I(hard_failure)
```

因此在 S1/S2 的宽容期，低 `xi` 只减弱安全代价，不降低到达奖励；碰撞仍不终止，使策略先保持穿过障碍也能到达的能力。随后 `xi` 逐步增至 1，再切换严格碰撞终止。这与“到达优先的安全同伦”机制兼容。

## 3. V8 实验问题（历史）

1. 连续旋转输入与 twist 是否让 S0 的完整位姿课程达到 `eta=1`，同时保持位置能力？
2. 折扣势函数奖励是否消除旧奖励的局部门控突变、闭环刷分和明显 Q 高估？
3. 新增逐连杆接近速度/风险后，S1/S2 是否能在不牺牲到达率的情况下提前减速或绕障？
4. 从宽容碰撞到严格碰撞的转换是否仍会崩溃？若崩溃发生在何个 `xi`、何种场景和哪一连杆？

### v8 实测回答

1. 连续旋转输入和 twist 能推动姿态学习，但未在预算内达到完整位姿：课程通过 `eta=0.6` 后停在 `eta=0.65`，因此答案是“有改善但未达成”。
2. 折扣势函数未出现旧实验式闭环刷分或明显 Q 发散；block3 末尾 `Q1/Q2/target=17.986/18.035/17.791`。这只能说明数值健康，不能抵消任务失败。
3. 未回答。S0 未通过，按协议不得进入 S1/S2 检验逐连杆外部障碍风险。
4. 未回答。当前失败发生在 S0 姿态课程和位置保持阶段，尚未进入安全同伦严格化。

## 4. V8 通过条件与诊断（历史）

- 每次正式训练前必须通过单元测试、奖励排序、接触一致性和短训练/恢复冒烟。
- S0 仍执行位置范围课程、至少 25k 的 `g=1` 位置巩固、离散姿态课程和至少 25k 的 `eta=1` 完整位姿巩固。
- 每个课程档同时记录位置误差、姿态测地角、`Phi`、势函数进展、速度代价、成功率、碰撞/关节越界和 Q 均值。
- 训练成功率不能替代固定目标确定性探针；位置保持探针失败时冻结姿态晋级。
- validation 和 held-out 规则沿用原协议，但 v8 不能与旧网络或旧 replay 混用。

## 5. V8 失败判据（历史）

- 任何旧 checkpoint/replay 被加载到 v8；
- observation 出现非有限值或越界，真实 contact 与胶囊风险不一致；
- 课程在一个档位持续两个 100k block 仍无提升；
- 固定目标位置保持率连续两次低于 0.95；
- Q 值持续增长而实际折扣回报不增长；
- 严格化后到达率断崖式下降且一个完整恢复 block 无法恢复。

触发失败判据时先保存诊断，不通过临时放宽 Gate 把失败模型带入下一阶段。

## 6. V8 最终失败诊断（历史）

- 三个 block 累计 `300000` transition 后仍为 `eta=0.65`、`K_pose=0`、`s0_goal_gate_eligible=false`，触发正式三 block 停止规则。
- `eta=0.65` 当前任务的 434 个 episode 中，success `80.18%`；51 个 self-collision、34 个 timeout、1 个 joint-limit。自碰撞是最大的直接失败类型，timeout 次之。
- 最终固定位置 probe 为 47/50 成功、2/50 碰撞、1/50 timeout；在线位置锚点窗口却为 0.99。因而更深层问题是困难目标上的位置保持/泛化，不能只写成“平均成功率不足”。
- S0 对 self-collision 没有显式连续几何输入或接近惩罚，只有接触后的 `-10` hard failure。这与约 `11.7%` 的当前任务自碰撞率一致，是下一版的首要设计假设，但尚不能仅凭相关性断言它是唯一根因。
- update 日志无非有限值，critic、target、温度和 recovery replay 比例正常；没有证据支持通过调整学习率、延长同一 run 或放宽 Gate 解决问题。

下一版必须先在不泄漏私有 probe 目标的前提下定位失败构型，再决定是否加入 self-clearance/self-risk observation、连续自碰撞代价或更强的困难位置保持机制。任一修改都会改变契约，必须使用新协议标识并从头训练。

## 7. V9 修正协议（历史）

v8 结果显示完整姿态阶段存在 self-collision，但 observation 和 reward 只有接触后的稀疏硬失败信号。v9 保留 v8 的前 98 维 observation，在末尾追加：

| 区间 | 内容 |
| --- | --- |
| `98:104` | 六个可控 link 的最小 collision-mesh self-clearance |
| `104:110` | 六个可控 link 的最小 self-TTC |
| `110:116` | 六个可控 link 的最大 self-approach |
| `116:122` | 六个可控 link 的最大 self-risk |

自碰撞几何使用 URDF collision mesh 的 PyBullet `getClosestPoints`，排除运动链相邻 link pair；不使用会在合法腕部构型中永久重叠的胶囊距离。reset、目标生成和真实 contact 判定使用同一套非相邻 pair 集合。

奖励在 v8 位姿势函数基础上增加独立自碰撞项：

```text
r = r_goal - 10 I_hard
    - xi_scene (2 R_external + 8 V_external)
    - lambda_self (2 R_self + 8 V_self)
```

`lambda_self` 在 S0-P 固定为 `0.02`；位置完整范围巩固完成后按 50000 个合格 transition 线性升至 `1`，S1/S2 固定为 `1`。真实 self-contact 在所有权重下仍立即硬终止。replay 保存 22 个奖励原始字段，并在抽样时按当前 `lambda_self`、`xi_scene` 和 transition 自身的姿态系数重算 reward。

v9 使用 122 维输入，隐藏层仍为 `[256,256]`；v8 actor、critic、optimizer 和 replay 均不兼容，不能迁移。

v9 预检结果：针对性测试 `31 passed`、全量测试 `89 passed`、500 构型 mesh/contact 一致率 `1.0`、真实 200-transition reward 重构误差 `0`、保存恢复冒烟 `24→36`，且 v8 checkpoint 会被协议校验拒绝。预检只证明实现自洽，不证明完整 S0 收敛。

## 8. V11 当前协议

当前协议标识为 `task_first_persistent_eta_replay_homotopy_v11`，输出根目录为 `outputs/experiments_9/v11/`。V11 保留 V10 的 122 维 observation、mesh self-clearance、22 个 replay 原子字段和 `[256,256]` 网络；改变的是无障碍 replay 的物理存储契约。

V10 的 600k 结果表明，单一 60000 FIFO 在 `eta=0.75` 长时间训练后覆盖了 `eta=0.1..0.7` 的全部历史数据。V11 将无障碍 replay 分为 anchor FIFO、current-eta FIFO 和每个已完成 eta 档的冻结 snapshot，默认容量分别为 `15000`、`30000` 和每档 `3000`。历史 snapshot 只读，不参与当前档写入；checkpoint 保存并恢复所有池、当前档标识、历史档位和 replay RNG。旧 replay storage version 或旧协议必须拒绝恢复。

奖励改为有界安全组：

```text
c_external = (2 R_external + 8 V_external) / 10
c_self     = (2 R_self + 8 V_self) / 10
r = r_goal - xi_scene c_external - lambda_self c_self
    - 10 I_hard - 10 I_strict_obstacle_terminal
```

S0 状态机按以下顺序单向推进：

| 状态 | 任务与门槛 | `lambda_self` |
| --- | --- | ---: |
| `position` | `g:0.03->1`，完成 `K_position>=25000` | `0` |
| `pose` | `eta:0->1`，完成 `K_pose>=25000`、位姿/锚点窗口和 probe | `0` |
| `self_safety_ramp` | 上述任务门槛持续满足时，累计 50000 合格 transitions；跌破即冻结 | `0->1` |
| `self_safety_consolidation` | 满权重下再累计 25000 合格 transitions | `1` |
| `complete` | 允许运行三组 validation Gate | `1` |

真实 self-contact 在所有状态始终立即终止并扣 `10`。`lambda_self=0` 只关闭碰撞前的连续 margin penalty，不允许碰撞，也不关闭 24 维 self observation。

Replay 在抽样时使用 transition 自身的 `eta`、当前 `lambda_self` 和当前 `xi_scene` 重算 reward，因此阶段切换保留网络、optimizer 和历史数据，不混用旧标量 reward。V8/V9 checkpoint 因协议不匹配必须拒绝。

Block 固定为 100000 transitions 的执行、保存和分析单位，每 25000 transitions 保存完整 checkpoint；它不是“最多三个”的停止上限。每个 stage 对所有正式 seed 和消融统一设置累计 600000 transitions 的硬预算，预算只保证计算量可比，不把“达到某个 step”定义为学习停滞。

S0 的平台期必须由两个相邻完整 Block 判定：两个 endpoint 的 `orientation_level_index/eta` 相同，且两个 Block 的非锚点合格 episode 都没有跨档；每个 Block 至少包含 100 个当前档 episode、50 个位置锚点 episode和一次当前档固定 probe。在此前提下，safe success 未提高 0.02、self-collision 未下降 0.01、timeout 未下降 0.02、anchor success 未提高 0.01，同时 probe success/collision/timeout 分别未改善 0.02/0.01/0.02，才记为 `plateau_stop`。任一项达到阈值都继续训练，跨 `eta` 混合的总体成功率不参与判定。

任一更新诊断出现 NaN/Inf 时立即记 `numerical_failure`。持续 Q 发散定义为当前 Block 尾部 `Q abs p95>1000` 且超过前一 Block 5 倍，并且 `critic loss p95>1000` 且超过前一 Block 5 倍；单次峰值不触发停止。`s0_goal_gate_eligible=true` 只触发三个 validation seeds，并不等于成功；`41001/42002/43003` 必须对同一 checkpoint 分别通过。若 validation 未全通过且尚未平台或达到 600k，则从当前 Block 最后 checkpoint 继续。上述预算、平台阈值、数值规则和 Gate 规则不得因训练 seed 或消融组改变。

论文分析与拒绝 LSTM/专家 replay 的理由见 [reward_literature_analysis.md](reward_literature_analysis.md)。
