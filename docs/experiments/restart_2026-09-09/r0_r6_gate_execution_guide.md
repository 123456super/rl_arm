# R0–R6 Gate 训练与验证流程

> 适用于 `conservative_capsule_gap_v2`。
> 核心规则：每个阶段只增加一个新难点；前一阶段未通过 Gate，不进入下一阶段。

## 1. 总体训练流程

| 阶段 | 核心目的 | 主要输入/结构 | 是否训练 SAC | 障碍难度 | 关键通过指标 | 不通过时 |
| --- | --- | --- | --- | --- | --- | --- |
| **R0** | 冻结实验规则与环境 | UR5、控制频率、目标分布、reward、success、数据划分 | 否 | 无/未正式启用 | 配置完整，train/validation/held-out 分离，参数全部记录 | **禁止训练** |
| **R1** | 验证机器人、IK、几何和执行链 | FK/IK、joint limit、胶囊、240 Hz 子步、quintic、rate limit、Butterworth | 否 | 无障碍为主；单障碍只做几何测试 | IK 可达、FK/IK 一致、胶囊误差合格、命令连续、速度/acc/jerk 不越界 | **禁止训练 SAC** |
| **R2-A** | 学会最基础的无障碍 reaching | SAC → rate limit → Butterworth → `u=q̇_nom` → quintic | **是** | 无障碍 | IK-reachable validation 目标集上 5 cm 成功率达到 Gate | 继续解决 reaching，不加障碍 |
| **R2-B** | 提高无障碍定位精度 | 与 R2-A 相同 | **继续训练/微调** | 无障碍 | 依次通过 3 cm、2 cm、1 cm 精度 Gate | 不进入有障碍训练 |
| **R2-C** | 学会当前风险条件下的动态避障 nominal policy | 当前几何/风险 → Instant-Link-SAC → rate limit → Butterworth → quintic | **是** | 从弱到强逐级增加 | success、collision、timeout、风险响应、平滑性达标 | 不冻结 actor，不进入预测层 |
| **R2-D** | 冻结最终 nominal actor | 与部署相同，但没有 QP | 否 | 正式训练分布 + 独立 validation | 多 seed 稳定，无 terminal shortcut，指标稳定 | 回到 R2-C |
| **R3** | 单独验证动作条件预测是否可信 | 冻结 actor + `q̂_k(u)`、`ĉ_k`、`d̂_i,k` | 否 | 动态障碍逐级测试 | 未来最小间隙误差、首次越界、危险连杆、提前量达标 | **禁止加 QP** |
| **R4** | 验证最简单 QP 能否正确修改危险动作 | 冻结 actor + One-Step-QP | 否 | 低/中难度动态障碍 | QP 可行率、间隙改善、碰撞降低、修正幅度、实时性 | 禁止进入多时刻 PTQP |
| **R5** | 验证轨迹嵌入和固定 Top-k 多时刻约束 | 冻结 actor + 固定 Top-k PTQP | 否 | 正式动态障碍 | 优于 R4，acc/jerk 满足，固定 Top-k 漏检可量化 | 禁止进入 VG-PTQP |
| **R6** | 验证完整 VG-PTQP | Top-k 初始化 + 全身非线性验收 + 反例增广 + fallback | 否 | 全难度 + 压力场景 | 验收率、RecoveryRate、安全、任务、deadline、fallback 全达标 | 不进入 held-out 最终实验 |

训练主线可以概括为：

```text
先证明环境正确
    ↓
先让 SAC 在无障碍下学会 reaching
    ↓
再提高 reaching 精度
    ↓
再让 SAC 学习当前风险下的动态避障
    ↓
冻结 actor，之后不再训练 SAC
    ↓
依次验证预测、单步 QP、多时刻 QP、验证驱动 QP
```

## 2. R0：冻结实验规则

### 本阶段目的

R0 不追求性能，也不训练网络。它只负责冻结后续实验的共同规则，避免训练过程中不断改变目标。

### 必须冻结的内容

- UR5 模型、基座位置、6 个关节及 joint limits；
- PyBullet 物理频率 240 Hz；
- SAC 控制频率 20 Hz；
- 每个控制周期 12 个物理子步；
- 动作范围和 policy rate limit；
- Butterworth 参数和 quintic 公式；
- 目标工作空间、初始关节分布和 IK 可达性筛选；
- episode 最大长度；
- reward、cost、success、collision、timeout 定义；
- 动态障碍物的位置、方向、速度和半径分布；
- train、validation、held-out test 的目标和 seed；
- 每个 Gate 的数值阈值；
- checkpoint 的候选步数和选择规则。

当前项目必须保持以下几何语义：

```text
d_raw = center_distance - capsule_radius - obstacle_radius
d     = d_raw - 0.04 m
```

其中 `d` 用于 observation、risk、TTC、cost、预测和 QP；PyBullet contact 是碰撞事实源。

### R0 Gate

只有以下条件全部成立才通过：

- 所有配置有唯一版本；
- train、validation、held-out 完全分离；
- 所有 Gate 阈值已在看结果前写定；
- 源码、配置、URDF、胶囊和依赖可由 manifest 复现；
- 完整测试通过；
- 几何修订前的 actor 和 replay buffer 明确禁止复用。

不通过时：停止，不运行正式标定，也不训练 SAC。

## 3. R1：验证环境、几何与执行链

### 本阶段目的

先证明环境本身正确，避免 SAC 学习环境 bug。R1 全程不训练 SAC。

### R1.1 FK/IK 和目标可达性

对随机目标 `p_g` 求解 IK，再用 FK 回代：

```text
q_IK = IK(p_g)
p_FK = FK(q_IK)
e_IK = ||p_FK - p_g||₂
```

执行要求：

1. 在整个目标工作空间分层采样，而不是只测中心区域。
2. 将目标分为可达、不可达、IK 失败、越 joint limit 四类。
3. 只有通过 IK、joint limit 和 FK 回代检查的目标才能进入训练和评估目标池。
4. 报告 `e_IK` 的 mean、p95、p99、max，并保存最差样本。

### R1.2 rate limit、Butterworth 和 quintic

不用 SAC，人工输入零动作、阶跃、正负交替和随机速度：

```text
人工动作
 → 动作缩放
 → rate limit
 → Butterworth endpoint
 → quintic 12 子步
 → PyBullet 240 Hz 执行
```

重点检查：

- Butterworth 每个控制周期只更新一次；
- 预测、QP 和执行共用同一个缓存 endpoint；
- 实际下发满足 `q̇_k=(1-s_k)q̇_prev+s_k u`；
- 上一周期末速度和加速度正确传到下一周期；
- 每周期恰好执行 12 个子步；
- command 和 measured velocity/acceleration/jerk 分开记录；
- position、velocity、acceleration、jerk 均不超过冻结限制。

### R1.3 胶囊几何标定

使用一个静态障碍球，不训练策略。对六个胶囊分别在随机关节构型和近表面方向采样：

```bash
conda run -n rl python scripts/calibrate_capsule_geometry.py \
  --config configs/default.yaml \
  --samples-per-link 400 \
  --seed 71001 \
  --output-dir outputs/restart_2026-09-09/r1_geometry/calibration_seed71001
```

比较胶囊距离与 PyBullet collision-shape 距离，报告：

- 总体和逐连杆距离误差；
- `d_safe` 危险判断的 false positive / false negative；
- 胶囊碰撞与 PyBullet contact 的漏检；
- 扣除 0.04 m margin 前后的结果；
- `wrist_3` 球形近似的独立结果；
- 最严重误差样本对应的关节状态和障碍物位置。

### R1 Gate

- 正式目标全部 IK 可达；
- FK/IK 回代误差达到预注册阈值；
- 数学生成的 quintic 与实际下发命令一致；
- joint、velocity、acceleration、jerk 均不越界；
- 0.04 m margin 后危险漏检和碰撞漏检为零；
- `wrist_3` 独立检查通过。

任一项失败：修复环境或几何并重新执行 R1，禁止训练 SAC。

## 4. R2：训练并冻结 nominal SAC

R2 是整个 R0–R6 中唯一训练 SAC 的阶段，分为 R2-A、R2-B、R2-C、R2-D。

所有训练必须：

- actor 和 replay buffer 从零初始化；
- 从第一步就使用部署时相同的 rate limit、Butterworth 和 quintic；
- 只用 validation 选择 checkpoint；
- 不查看 held-out test 结果；
- 不加载几何修订前 checkpoint。

### 4.1 R2-A：无障碍 5 cm reaching

#### 控制结构

```text
q、q̇、goal、零风险占位
       ↓
      SAC
       ↓
  rate limit
       ↓
  Butterworth
       ↓
    q̇_nom
       ↓
  u = q̇_nom
       ↓
quintic 12 子步执行
```

没有障碍物，没有 QP，但 observation 仍保持 `link_risk_v1` 的固定维度。

#### 训练步骤

1. 使用冻结的 IK-reachable train 目标池训练。
2. 只在预注册 checkpoint steps 暂停并评估。
3. 使用固定 validation 目标集和 deterministic actor。
4. 记录 success、timeout、final error、episode 内 minimum error、完成时间和平滑性。
5. 按冻结规则选择候选 checkpoint。

#### R2-A Gate

核心指标是固定 IK-reachable validation 集合上的 `SuccessRate@5cm`。推荐目标是 100%，但正式判定以 R0 写定的阈值为准。

不通过时：继续解决基础 reaching，包括 observation、reward、网络容量、探索和执行链问题；不允许先加入障碍物。

### 4.2 R2-B：逐步提高定位精度

R2-B 保持与 R2-A 完全相同的控制结构和无障碍环境，只改变目标精度：

```text
5 cm → 3 cm → 2 cm → 1 cm
```

每一级都应：

1. 在冻结的 validation 目标集上单独报告成功率。
2. 检查到达后是否稳定，而不是短暂穿过阈值。
3. 比较 final error 和 episode 内 minimum error。
4. 将 timeout 分为从未靠近、靠近后振荡、越过目标、动作饱和。
5. 检查目标附近的 action、acceleration 和 jerk。

#### R2-B Gate

- 5 cm：应达到基础 reaching Gate；
- 3 cm：成功率接近基础水平；
- 2 cm：成功率高且跨 seed 稳定；
- 1 cm：达到 R0 预注册的最终论文要求。

不要求事后强行把 1 cm 调到 100%。若某一级失败，停留在该级分析和训练，不进入有障碍阶段。

注意：当前配置的正式 success tolerance 是 `0.055 m`。实施上述精度课程前，应在 R0 将各级 success 判定、稳定窗口和阈值写入新版协议。

### 4.3 R2-C：训练动态避障 nominal policy

#### 控制结构

```text
当前 q、q̇、goal
障碍物中心和速度
胶囊 d_i、approach velocity、TTC、Risk_now
                 ↓
         Instant-Link-SAC
                 ↓
            rate limit
                 ↓
            Butterworth
                 ↓
              q̇_nom
                 ↓
          quintic 12 子步
```

仍然没有 QP。SAC 必须自己学会基本动态避障。

#### 障碍课程

| Level | 障碍设置 | 本级唯一目的 |
| --- | --- | --- |
| **0** | 障碍存在但远离 reaching 路径 | 确认增加风险 observation 后 reaching 没有被破坏 |
| **1** | 静态或极低速轻度阻挡 | 学会基本绕行 |
| **2** | 低速动态横穿 | 学会响应距离、接近速度和 TTC |
| **3** | 正式随机动态障碍分布 | 完成正式 nominal policy 训练 |
| **4** | 肘部、前臂、腕部、快速接近等困难场景 | 做 validation stress，不默认大量加入训练 |

前一级没有通过，不提高速度、阻挡程度或场景复杂度。课程比例和切换条件必须提前冻结。

#### 每一级都要检查

- SuccessRate；
- CollisionRate；
- TimeoutRate；
- final/minimum goal error；
- completion time；
- `d_min`、`d_min_raw` 和 safety violation；
- 当前风险上升时 actor 是否产生合理避让；
- policy action、240 Hz command 和 measured trajectory 的平滑性；
- 各胶囊结果，尤其是非末端连杆碰撞。

#### terminal shortcut Gate

由于碰撞会提前结束 episode，必须检查策略有没有学会“主动碰撞比困难地继续运动更划算”：

- 比较立即碰撞和继续至 timeout 的累计 reward；
- 检查碰撞前动作是否持续朝障碍物方向；
- 检查碰撞前是否仍在获得任务 progress；
- 保存各 seed 的典型碰撞轨迹；
- 确认 collision penalty 足以抵消提前终止规避的未来负 reward。

存在 terminal shortcut 时，R2-C 失败。必须修复 reward/termination 并重新训练，不能交给 QP 掩盖。

#### R2-C Gate

success、collision、timeout、目标误差、平滑性和当前风险响应均达到 R0 阈值；没有 terminal shortcut；各连杆和各场景没有系统性失败。

不通过时：留在对应课程级别继续处理，不冻结 actor，不进入 R3。

### 4.4 R2-D：重训并冻结最终 actor

R2-D 不再探索新结构。先冻结 R2-C 的 observation、reward、课程和超参数，再从零训练至少 3 个预注册 random seeds。

每个 seed：

1. 只按相同 validation 规则选择 checkpoint。
2. 运行无障碍和正式动态障碍 validation。
3. 检查 success/collision/timeout 和 terminal shortcut。
4. 保存 actor SHA-256、train seed、checkpoint step、配置和源码哈希。

如果 Instant-Link-SAC 持续破坏 reaching，可根据预注册规则改用 task-only SAC 作为 nominal actor，并保留 Instant-Link-SAC 为策略基线；这一路线必须在进入 R3 前冻结。

#### R2-D Gate

- 至少 3 个从零训练 seed 达标；
- seed 间性能波动在阈值内；
- 无异常 terminal shortcut；
- checkpoint 选择没有接触 held-out；
- actor、observation、动作顺序和执行链均可由 hash 唯一识别。

通过后冻结 actor。R3–R6 原则上不再训练或微调 SAC。

## 5. R3：验证动作条件预测

### 本阶段目的

只验证预测，不加 QP。回答：给定候选 endpoint `u` 后，能否提前预测各连杆未来间隙？

### 预测流程

```text
冻结 actor 生成 q̇_nom
        ↓
令候选 u = q̇_nom
        ↓
使用真实 quintic 得到 q̂_k(u)
        +
障碍运动模型得到 ĉ_k
        ↓
每个未来姿态重新计算全部胶囊
        ↓
d̂_i,k、d_i^pred、T_i^enter、critical link
```

预测模型必须使用实际将执行的 quintic 子步，不能只把当前胶囊按速度直线外推。

### 真值生成

从相同机器人和障碍状态创建仿真副本，执行同一个缓存候选 endpoint：

- 本控制周期执行完全相同的 12 个 quintic 子步；
- 之后按冻结的 endpoint-hold 规则展开；
- 不允许后续 SAC 重新规划改变真值轨迹。

### 指标和 Gate

- `d̂_i,k` 的 MAE、p95、最大误差；
- 未来最小间隙误差；
- 首次进入 `d_safe` 的 recall、false negative 和时间误差；
- 危险连杆识别准确率；
- 预警 LeadTime、coverage 和 false-positive rate；
- prediction mean/p95/p99/max time。

只有预测距离、首次越界、危险连杆和提前量全部达到预注册阈值，才进入 R4。

当前仓库已有的线性 predictive-risk 只能作为“当前运动趋势预测”基线；正式 R3 需要候选动作条件 rollout 和开环反事实真值。

## 6. R4：One-Step-QP

### 本阶段目的

用最简单的单时刻 QP 检查距离梯度和动作修正方向，不直接跳到多时刻 PTQP。

### 控制流程

```text
冻结 actor → rate limit → Butterworth → q̇_nom
→ One-Step convex QP → u* → quintic → 执行
```

QP 后不能再次 Butterworth 滤波，否则实际执行动作与 QP 验证动作不一致。

### 先做人工危险动作测试

- 动作朝向障碍物；
- 动作背离障碍物；
- 动作与障碍物切向；
- joint-limit 邻域；
- 无危险但接近 activation threshold。

验证 distance Jacobian、QP correction 方向、速度/position/acceleration/jerk 约束和 nonlinear rollout 的实际间隙。

### R4 Gate

- QP optimal/feasible rate 达标；
- 危险动作修正后实际最小间隙整体改善；
- collision/violation 相对无 QP 降低；
- 安全动作不过度干预；
- CorrectionNorm 和任务性能下降在阈值内；
- solver failure 和 deadline miss 有安全处理；
- mean/p95/p99/max controller time 达标。

当前 `SafetyQP` 是迭代半空间投影，只能作为 Reactive-Projection 基线。正式 R4 必须使用真正的 convex QP solver 并报告 status、residual 和 slack。

不通过时：修复梯度、约束、solver 或执行接口，不进入 R5。

## 7. R5：固定 Top-k 多时刻 PTQP

### 本阶段目的

在 R4 基础上只增加“动作条件 + 多时刻 + 固定 Top-k”约束，验证轨迹嵌入是否有价值。

### 控制流程

```text
冻结 actor 生成 q̇_nom
        ↓
名义动作条件多时刻 rollout
        ↓
选择固定 Top-k 危险连杆/时刻
        ↓
Predictive-Trajectory-QP
        ↓
u* → 同一 quintic 子步执行
```

Top-k 按连杆选择，但实际 QP 约束索引是 `(link_id, time_index)`。R5 不做反例增广。

### 公平比较

R4 与 R5 必须使用相同 actor、episode seeds、solver、运动边界、quintic 和时间预算。唯一新增变量是多时刻预测约束。

### 全身检查只作诊断

对每个 R5 候选做全连杆、全验证时刻 nonlinear rollout，计算：

```text
ModelCheckRejectionRate
= 固定 Top-k QP 判定可执行、但全身 nonlinear rollout 拒绝的候选数
  / 全部被检查候选数
```

记录漏掉的连杆、时刻、违反量和危险连杆迁移。R5 不能根据检查结果重解或 fallback，否则会混入 R6 功能。

### R5 Gate

- 相对 R4 至少改善一项预注册预测安全主指标；
- collision/violation 或未来最小间隙有实际改善；
- success 和 CorrectionNorm 的退化在容忍范围；
- 所有 quintic 子步满足 acceleration/jerk；
- ModelCheckRejectionRate 和遗漏类型已量化；
- 实时性达标。

没有相对 R4 的净收益时，不得进入 VG-PTQP 结论阶段。

## 8. R6：完整 VG-PTQP

### 本阶段目的

验证全身 nonlinear 验收和反例增广能否修复固定 Top-k 的遗漏。

### 控制流程

```text
冻结 actor
    ↓
q̇_nom
    ↓
名义动作条件全身预测
    ↓
固定规则生成初始稀疏集合 A₀
    ↓
轨迹嵌入 convex QP
    ↓
候选 u*^(r)
    ↓
全连杆 × 全验证时刻 nonlinear rollout
    ↓
通过？
 ┌──┴────────────────────┐
是                       否
↓                        ↓
执行 verified 子步       找出最严重反例 C_r
                         ↓
                    加入新约束
                         ↓
                  重新线性化并求解
```

每一轮固定 active set 和线性化点，因此每个子问题仍是 convex QP。整个算法受最大细化轮数和控制周期时间预算限制。

### 终止和 fallback

出现以下任一情况时，候选动作禁止执行：

- 达到最大细化轮数；
- deadline 剩余时间不足；
- solver fail；
- slack 超阈；
- nonlinear verification fail；
- 新一轮没有继续改善；
- 状态过期或出现 NaN。

此时进入提前通过相同全身验收的 jerk-limited braking fallback。若 braking 也不能通过，则仿真 emergency stop；未来硬件阶段对应硬件急停。

### R6 Gate

任务指标：

- SuccessRate、GoalError、EpisodeTime、TimeoutRate。

安全指标：

- CollisionRate、MinClearance、SafetyViolationRate、LeadTime。

干预指标：

- InterventionRate、CorrectionNorm、fallback rate。

验收指标：

- 首轮 ModelCheckRejectionRate；
- 最终 verification pass rate；
- RefinementRecoveryRate；
- 各轮新增反例数和危险连杆迁移率；
- 未验收候选执行次数必须为零。

实时指标：

- prediction、Jacobian、QP、verification、fallback check；
- controller 总耗时的 mean、p95、p99、max；
- `P(t_controller ≤ T_budget)` 和 deadline miss rate。

完整方法必须相对 R5 降低最终模型漏检，并且确有一部分首轮拒绝候选由反例增广恢复，而不是仅靠增加 fallback。任务性能、干预幅度和实时性也必须在预注册范围内。

不通过时：不得开启 held-out test，也不得宣称 VG-PTQP 的核心创新已经验证。

## 9. Gate 执行顺序速查

```text
R0 未过 → 禁止做任何正式训练
R1 未过 → 禁止训练 SAC
R2-A 未过 → 不加障碍
R2-B 未过 → 不进入动态障碍训练
R2-C 未过 → 不冻结 actor
R2-D 未过 → 不进入预测层
R3 未过 → 不启用 QP
R4 未过 → 不实现/评估多时刻 PTQP
R5 未过 → 不宣称 VG refinement 的必要性
R6 未过 → 不打开 held-out test
```

每完成一个阶段，只做三件事：

1. 将运行和指标写入 [experiment_tracker.md](experiment_tracker.md)。
2. 根据预注册阈值明确写 `pass` 或 `fail`。
3. 只有 `pass` 才解除下一阶段阻塞。

任何涉及胶囊、margin、success、collision、observation、reward、预测真值、执行顺序或 QP 后处理的语义变化，都必须提高实验版本，并从最早受影响的阶段重新开始。
