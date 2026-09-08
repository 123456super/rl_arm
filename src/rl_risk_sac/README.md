# `src/rl_risk_sac` 代码导读

这份目录是论文方法的核心实现。可以先把它理解成三层：

```text
训练脚本 scripts/train.py
  |
  v
Gym 环境 envs/ur5_dynamic_obstacle_env.py
  |
  +-- scene/      目标之外的动态场景对象，例如障碍物
  +-- tasks/      observation、reward、cost、目标点状态
  +-- robots/     UR5 连杆胶囊体近似
  +-- collision/  胶囊体与障碍物之间的风险检测
  +-- control/    actor 动作到可执行关节速度轨迹
  |
  v
算法 algorithms/sac.py
```

如果你已经大致理解论文大纲，读代码时最重要的是抓住这条线：

```text
策略 actor 输出归一化动作
  -> 环境把动作变成关节速度
  -> PyBullet 推进一步机械臂、障碍物和目标
  -> 胶囊体模型读取当前连杆几何
  -> 风险模块计算每根连杆的距离、接近速度、TTC 和风险
  -> 任务模块拼 observation，并计算 reward/cost
  -> 训练脚本把 transition 放入 replay buffer
  -> SACAgent 用 replay buffer 更新 actor、reward critic、cost critic
```

## 推荐阅读顺序

建议不要从神经网络文件开始读。这个项目的难点不在 PyTorch 写法，而在“环境状态怎么变成强化学习训练信号”。

1. `envs/ur5_dynamic_obstacle_env.py`

   先读 `UR5DynamicObstacleEnv.__init__()`、`reset()`、`step()`。这是所有模块的汇合点，也是理解一回合 episode 如何运行的入口。

2. `tasks/reaching.py`

   读 `ReachingObservationBuilder.build()` 和 `ReachingObjective.reward()/cost()`。这里能看到 SAC 到底观察什么、优化什么。

3. `utils/risk.py`

   读 `compute_link_risk()`。这里对应论文中的连杆级动态风险，包含表面距离、接近速度、TTC 和风险加权融合。

4. `robots/ur5_capsules.py` 和 `collision/detectors.py`

   这两个文件解释“为什么能算连杆风险”：机械臂先被近似成胶囊体，检测器再计算障碍物到每根胶囊体的风险。

5. `algorithms/sac.py`

   最后读 SAC。此时你已经知道 reward/cost 从哪里来，再看 actor、reward critic、cost critic 和拉格朗日乘子会容易很多。

6. `control/execution.py` 和 `utils/trajectory_blending.py`

   这些文件解释 actor 动作不是直接丢给电机，而是会经过速度变化率限制、EMA 或 Butterworth + 五次插值。

## 一次 `env.step(action)` 内部发生什么

`UR5DynamicObstacleEnv.step()` 是最值得反复看的函数。它基本就是论文仿真实验的一步控制周期。

```text
输入 action
  |
  | 1. action 被裁剪到 [-1, 1]
  v
计算执行前风险 pre_risk
  |
  | 2. ExecutionPipeline 根据 action、上一条速度命令、risk_global 生成速度轨迹
  v
qdot_trajectory: [物理子步数, 关节数]
  |
  | 3. 每个物理子步：
  |    - 障碍物前进一步
  |    - 动态目标前进一步
  |    - PyBullet 执行一小步速度控制
  v
读取新的关节状态、末端位置、胶囊体状态
  |
  | 4. RiskDetector 计算连杆风险
  v
ReachingObservationBuilder 拼 observation
  |
  | 5. 判断 success / collision / truncated
  v
ReachingObjective 计算 reward 和 cost
  |
  | 6. 固定惩罚方法把 cost 扣进 reward；LDRC 方法保留单独 cost
  v
返回 obs, reward, cost, terminated, truncated, info
```

几个关键变量：

- `action`：actor 输出的归一化动作，范围是 `[-1, 1]`。
- `qdot_cmd`：一个控制周期结束时的关节速度命令。
- `qdot_trajectory`：一个控制周期内分给 PyBullet 多个物理子步执行的速度轨迹。
- `pre_risk`：执行本步动作之前的风险，用于自适应平滑。
- `risk` / `self.last_risk`：执行本步之后重新计算的风险，用于 observation、reward、cost 和日志。
- `terminated`：碰撞或成功导致 episode 正常结束。
- `truncated`：达到最大步数导致 episode 被截断。

## Observation 由哪些量组成

observation 在 `tasks/reaching.py` 的 `ReachingObservationBuilder.build()` 中拼接。顺序很重要，因为训练好的 checkpoint 默认依赖这个输入 schema。

```text
q / pi
q_dot / action_scale
goal_error
goal_velocity_error
risk.distances
risk.directions
risk.approach_velocities / v_max
risk.ttc / ttc_max
risk.risks
previous_command / action_scale
beta
```

直观解释：

- `q`：当前关节角，让策略知道机械臂姿态。
- `q_dot`：当前关节速度，让策略知道运动状态。
- `goal_error`：末端到目标点的位置误差，是到达任务的核心。
- `goal_velocity_error`：动态目标实验中使用，表示目标速度和末端速度的差。
- `risk.distances`：每根胶囊连杆到障碍物表面的距离。
- `risk.directions`：每根连杆最近点指向障碍物球心的方向。
- `risk.approach_velocities`：障碍物和连杆沿最近方向的接近速度，只保留“正在靠近”的部分。
- `risk.ttc`：time-to-collision 风格的时间量，越小越危险。
- `risk.risks`：每根连杆融合后的风险分数。
- `previous_command`：上一条关节速度命令，用于让策略感知动作连续性。
- `beta`：当前平滑系数，主要和自适应平滑方法有关。

## Reward 和 Cost 的区别

这个项目里 `reward` 和 `cost` 是故意分开的。

`reward` 主要服务于“把任务做好”：

- 离目标越近越好。
- 比上一步更接近目标会得到 progress 奖励。
- 速度命令变化太大会被平滑项惩罚。
- 成功到达有 bonus。
- 碰撞有 penalty。

`cost` 主要服务于“保持安全”：

- `risk_global` 越大，cost 越大。
- `d_min < d_safe` 时产生安全距离 violation cost。
- 真实碰撞时产生 collision cost。

固定风险惩罚方法：

```text
reward = reward - fixed_risk_penalty * cost
```

LDRC 约束方法：

```text
reward 仍然表示任务收益
cost 单独进入 cost critic
actor_loss 额外加上 lagrange_multiplier * cost_q
```

所以看实验结果时要分清：固定惩罚是“把安全揉进 reward”，约束 SAC 是“reward 和安全约束分开优化”。

## 方法名称和代码行为

项目里常见方法名在 `envs/ur5_dynamic_obstacle_env.py` 的 `METHODS` 中定义：

- `ee_fixed`

  末端风险固定惩罚基线。`LinkRiskDetector(end_effector_only=True)` 会只让最后一根胶囊体风险参与打分，然后环境把 `cost` 按固定权重扣进 `reward`。

- `link_fixed`

  连杆级风险固定惩罚方法。所有胶囊连杆都参与风险计算，环境同样把 `cost` 按固定权重扣进 `reward`。

- `ldrc_fixed`

  连杆级动态风险约束 SAC。环境返回单独的 `cost`，`SACAgent` 训练 cost critic，并用 `lagrange_multiplier` 约束策略。执行平滑使用固定参数。

- `ldrc_adaptive`

  在 `ldrc_fixed` 基础上增加自适应平滑：风险越高，`beta` 越大，当前策略命令占比越高；风险低时更多沿用上一条命令，使动作更平滑。根据当前 README 的实验结论，这个历史消融不是最终部署候选。

## 模块逐个看

### `envs/`

`UR5DynamicObstacleEnv` 是 Gymnasium 环境入口。它负责：

- 连接和重置 PyBullet。
- 加载 UR5 URDF。
- 解析配置里的关节名和 link 名称。
- 重置机械臂初始关节角。
- 采样目标和障碍物。
- 每一步执行关节速度控制。
- 调用风险、任务和执行管线模块。
- 汇总训练和评估用的 `info` 指标。

这个文件里的私有函数大致分三类：

- `_joint_state()`、`_end_effector_state()`、`_capsules()`：从 PyBullet 读取状态。
- `_advance_obstacle()`、`_move_target()`：推进场景对象。
- `_compute_risk()`、`_has_collision()`、`_get_obs_and_info()`：生成训练信号和诊断信息。

### `algorithms/`

`sac.py` 是算法主体：

- `actor`：输入 observation，输出归一化动作。
- `reward_q1/reward_q2`：估计奖励回报。
- `cost_q1/cost_q2`：估计安全代价回报。
- `reward_target_q*`、`cost_target_q*`：目标网络，使用 soft update 慢慢跟随在线网络。
- `alpha`：SAC 的熵温度，控制探索强度。
- `lagrange_multiplier`：LDRC 方法里的安全约束权重。

`networks.py` 只是神经网络结构：

- `GaussianActor` 使用 tanh-squashed Gaussian policy。
- `QNetwork` 输入 `(obs, action)`，输出一个标量 Q 值。

`replay_buffer.py` 是 off-policy 经验池：

- 存储 `(observation, action, reward, cost, next_observation, done)`。
- 训练时随机采样 batch。

### `robots/`

`UR5CapsuleModel` 把机械臂主要连杆近似成胶囊体：

```text
CapsuleState = start 点 + end 点 + radius + name
```

胶囊体的两端来自 PyBullet link 的世界坐标。这样做比直接用复杂 mesh 算距离简单得多，也更适合快速构造连杆级风险 observation。

### `collision/`

`LinkRiskDetector` 是风险检测入口。当前实现支持球形障碍物：

```text
多个 ObstacleState
  -> 对每个障碍物分别调用 compute_link_risk()
  -> 聚合成一个 LinkRisk
```

聚合时：

- 每根连杆的距离取最近障碍物对应的距离。
- 每根连杆的风险取所有障碍物下的最大风险。
- `risk_global` 取所有连杆风险最大值。
- `d_min` 取所有连杆和所有障碍物之间的最小表面距离。

`NullRiskDetector` 用于关闭障碍物时的无风险占位，保证 observation 维度不变。

### `utils/risk.py`

这是当前帧风险公式的核心文件。对每根胶囊连杆：

1. 找障碍物球心到胶囊中心线段的最近点。
2. 计算球表面到胶囊表面的距离 `surface_distance`。
3. 用上一帧胶囊状态估计该最近点的连杆速度。
4. 计算障碍物相对连杆的接近速度 `approach_velocity`。
5. 根据距离和接近速度估计 `ttc`。
6. 分别计算距离风险、速度风险、TTC 风险。
7. 加权融合成每根连杆的 `risk`。

最终返回的 `LinkRisk` 同时包含 per-link 信息和全局信息，是 observation、cost、碰撞判断和日志的共同来源。

### `utils/predictive_risk.py`

这是预测式风险的工具模块，不是主环境当前帧风险的替代品。它在一个短时间窗内做线性外推：

```text
当前胶囊体 + 上一帧胶囊体 -> 估计连杆端点速度
障碍物当前位置 + 障碍物速度 -> 估计未来球心位置
在多个未来时间点重复计算距离
```

它关注三个未来量：

- `d_pred`：预测窗内最小距离。
- `t_enter`：首次进入安全距离的时间。
- `approach_velocities`：预测窗内最大接近速度。

这些量再融合成 `risk_pred_per_link` 和 `risk_pred_body`，主要适合离线评估、诊断或后续扩展。

### `scene/`

`SphericalObstacleProvider` 只负责障碍物运动状态，不负责 PyBullet 可视化物体。

支持两类场景：

- `random`：从工作空间一侧随机穿越到另一侧。
- 命名场景：例如 `upper_arm_crossing`、`elbow_crossing`、`forearm_crossing`、`wrist_crossing`，用于让障碍物更常经过特定连杆区域。

每次 `advance(dt)` 会用匀速模型推进障碍物，并在配置边界处反弹。

### `tasks/`

`reaching.py` 定义到达任务：

- `ReachingObservationBuilder`：把物理状态、目标误差和风险信息拼成固定长度 observation。
- `ReachingObjective`：计算 reward 和 cost。

`targets.py` 定义目标点状态：

- `static`：目标固定。
- `linear_bounce`：目标以随机速度运动，并在 workspace 边界反弹。

### `control/`

`ExecutionPipeline` 负责动作执行前的处理：

```text
归一化 action
  -> 乘 action_scale 得到策略速度
  -> JointVelocityRateLimiter 限制相邻策略步速度差
  -> EMA 或 ButterworthQuinticRTB 平滑
  -> 输出 PyBullet 子步速度轨迹
```

关键点：

- actor 网络只输出一个控制周期的动作。
- PyBullet 可能需要多个更小的物理子步。
- RTB 会把一个策略速度目标变成一段子步轨迹。
- `physics_rms_acceleration` 和 `physics_rms_jerk` 用于检查真实执行轨迹是否平滑。

### `utils/config.py`

配置加载支持 `includes`。典型结构是：

```text
configs/default.yaml
  includes:
    - robot/ur5.yaml
    - environment/sim.yaml
    - algo/sac.yaml
    - run/dev.yaml
```

`deep_update()` 会递归合并配置，后加载的当前 YAML 会覆盖 include 进来的基础配置。`validate_config()` 会提前检查必需字段、关节数量、平滑模式、速度范围、设备选择权重等。

### `utils/device.py`

`resolve_device()` 根据配置选择运行设备：

- `cpu`：直接使用 CPU。
- `cuda:0`：使用指定 GPU。
- `cuda:auto` / `auto`：在候选 GPU 中按空闲显存和算力加权打分。

如果 CUDA 不可用且 `fallback_to_cpu` 为 true，就自动回退 CPU。

### `utils/seeding.py`

`set_seed()` 同时设置 Python、NumPy、PyTorch 和 `PYTHONHASHSEED`。这能提高复现实验的稳定性，但在 GPU 和 PyBullet 场景下仍不能保证绝对逐位一致。

## `scripts/train.py` 和 src 的关系

虽然 `train.py` 不在 `src` 目录里，但它是使用 `src` 的主要脚本。

训练循环可以简化为：

```text
load_config()
resolve_device()
set_seed()
env = UR5DynamicObstacleEnv(...)
agent = SACAgent(...)
replay = ReplayBuffer(...)

for step in total_steps:
    action = 随机动作或 agent.select_action(observation)
    next_observation, reward, cost, terminated, truncated, info = env.step(action)
    replay.add(...)
    if 满足更新条件:
        batch = replay.sample(...)
        agent.update(batch)
    if episode 结束:
        agent.update_lagrange(mean_cost)
        写 train_metrics.csv
        env.reset()
    if 到保存间隔:
        agent.save(...)
```

输出文件一般包括：

- `config.json`：本次训练实际使用的完整配置。
- `train_metrics.csv`：episode 级训练指标。
- `progress.csv`：step 级训练进度。
- `actor*.pt`：策略网络权重。
- `agent_state*.pt`：critic、alpha、lambda、cost_ema 等训练状态。

## 修改代码时最容易踩的点

- 不要随意改变 observation 字段顺序，否则旧 checkpoint 基本不能继续使用。
- 不要把 `reward` 和 `cost` 混成一个概念；固定惩罚和 LDRC 的核心差别就在这里。
- 不要把 actor 输出直接理解成真实电机命令；它还要经过 `ExecutionPipeline`。
- 不要把 PyBullet mesh 碰撞和论文里的连杆级风险混为一谈；风险主要来自胶囊体近似。
- 如果增加多障碍物，要优先看 `collision/detectors.py` 的聚合逻辑和 observation 维度是否仍固定。
- 如果增加新 observation，需要同步更新 schema version，并重新训练模型。
