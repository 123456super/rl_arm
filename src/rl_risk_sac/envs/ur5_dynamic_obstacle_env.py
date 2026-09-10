from __future__ import annotations

"""UR5 动态障碍物避障环境的整体架构说明。

这个文件很长，因为它是项目里最核心的“装配层”：它不是单纯实现某一个公式，
而是把 PyBullet 物理仿真、机械臂 URDF、连杆胶囊体、动态障碍物、目标点、
风险计算、observation 拼接、reward/cost 计算、动作平滑和 episode 终止条件
全部接在一起。读这份文件时，建议把它看成“强化学习环境主循环”，而不是
普通工具函数。

一、这个文件在项目中的位置
--------------------------------

训练脚本 `scripts/train.py` 不直接操作 PyBullet，也不直接计算风险。它只做：

    env = UR5DynamicObstacleEnv(config, method=method)
    observation, info = env.reset()
    next_observation, reward, cost, terminated, truncated, info = env.step(action)

也就是说，本文件负责把“一个动作 action”变成“一个 transition”：

    action
      -> PyBullet 中机械臂和障碍物前进一步
      -> 计算新的 observation
      -> 计算 reward 和 cost
      -> 判断 episode 是否结束
      -> 返回给 SAC 训练循环

二、它自己负责什么，不负责什么
--------------------------------

本环境负责：

1. 读取 config 中的 env/risk/smoothing/reward/robot 配置。
2. 连接 PyBullet，加载 UR5 的 URDF。
3. 把配置里的关节名、link 名解析成 PyBullet 的整数 id。
4. 在 reset 时重置机械臂、目标点、动态障碍物和上一时刻状态。
5. 在 step 时把 actor 的动作转成关节速度轨迹并执行多个物理子步。
6. 调用胶囊体模型得到每根连杆的近似几何。
7. 调用风险检测器计算当前帧风险，必要时计算预测风险。
8. 调用任务模块拼出 SAC observation，并计算 reward/cost。
9. 把训练、评估、画图需要的诊断数据放入 info。

本环境不负责：

1. 不实现 SAC 网络更新。那在 `algorithms/sac.py`。
2. 不直接写 replay buffer。那在 `scripts/train.py`。
3. 不直接定义神经网络结构。那在 `algorithms/networks.py`。
4. 不把 URDF mesh 当作论文风险几何。论文风险几何主要来自 `UR5CapsuleModel`。
5. 不真正控制实机。这里的 backend 是 PyBullet 仿真。

三、核心对象之间的关系
--------------------------------

可以把 `__init__()` 中创建的对象按下面方式理解：

    UR5DynamicObstacleEnv
      |
      +-- UR5CapsuleModel
      |     从 PyBullet 读取 link 位姿，把机械臂近似成多根胶囊体。
      |
      +-- SphericalObstacleProvider
      |     维护球形障碍物的位置、速度、随机穿越场景和边界反弹。
      |
      +-- WorkspaceTargetProvider
      |     维护目标点的位置和速度，支持静态目标或 linear_bounce 动态目标。
      |
      +-- LinkRiskDetector / NullRiskDetector
      |     根据胶囊体和障碍物计算 LinkRisk。
      |
      +-- ReachingObservationBuilder
      |     把关节状态、目标误差、风险信息、上一动作、beta 拼成 observation。
      |
      +-- ReachingObjective
      |     根据目标误差、动作平滑、碰撞和风险计算 reward/cost。
      |
      +-- ExecutionPipeline
            把 actor 输出的归一化动作变成 PyBullet 子步关节速度轨迹。

四、一次 reset() 做什么
--------------------------------

`reset()` 是一个 episode 的起点，大致流程如下：

    1. 如果传了 seed，就重建 numpy 随机数生成器。
    2. p.resetSimulation() 清空 PyBullet 世界。
    3. 重新设置 time_step、gravity。
    4. 创建地面。
    5. 加载 URDF 机器人。
    6. 解析 joint_ids 和 tool_link_id。
    7. 按配置的默认关节角加随机噪声重置机械臂。
    8. 采样目标点状态。
    9. 采样障碍物状态，并创建可视化/碰撞球。
    10. 创建目标 marker。
    11. 清空上一条速度命令、上一帧加速度、上一帧 capsule 等记忆。
    12. 根据 method 初始化 beta 和执行平滑管线。
    13. 计算第一帧 observation 和 info。

注意 `prev_capsules` 很重要：动态风险里的连杆速度不是从 PyBullet 直接读的，
而是用当前胶囊体和上一帧胶囊体差分估计出来的。reset 后会先记录初始
capsule，下一步才能估算连杆最近点速度。

五、一次 step(action) 做什么
--------------------------------

`step()` 是最核心的函数，可以按下面顺序读：

    输入 action，范围理论上是 [-1, 1]
      |
      v
    1. clip action，防止策略输出越界
      |
      v
    2. 计算执行前风险 pre_risk
       - adaptive smoothing 需要用这个风险决定 beta
       - 使用执行前风险，是为了避免“动作执行后的结果反过来决定已执行动作”
      |
      v
    3. ExecutionPipeline.process(...)
       - action * action_scale 得到策略关节速度
       - 可选限制相邻策略步速度差
       - EMA 或 Butterworth+quintic 生成子步速度轨迹
      |
      v
    4. 对 qdot_trajectory 中每个物理子步：
       - 障碍物 advance(time_step)
       - 目标 advance(time_step)
       - PyBullet 速度控制
       - p.stepSimulation()
       - 统计物理子步加速度和 jerk
      |
      v
    5. 物理执行结束后重新计算 observation 和 info
       - 读取关节状态和末端状态
       - 计算当前帧 LinkRisk
       - 如果 schema 要求，计算 PredictiveLinkRisk
      |
      v
    6. 判断 collision/success/truncated
      |
      v
    7. 计算 reward 和 cost
       - reward: 到达目标、进展、动作平滑、成功/碰撞奖惩
       - cost: 风险、安全距离 violation、碰撞代价
      |
      v
    8. 如果 method 是固定惩罚类，把 cost 扣进 reward
      |
      v
    9. 更新 prev_qdot_cmd、prev_capsules、prev_goal_error_norm 等记忆
      |
      v
    返回 obs, reward, cost, terminated, truncated, info

六、method 对行为的影响
--------------------------------

`METHODS` 中的方法名会影响风险使用方式和训练信号：

1. `ee_fixed`

   末端风险基线。风险检测时 `end_effector_only=True`，非末端连杆的风险
   不参与全局 risk_global。环境会把 cost 乘固定权重扣进 reward。

2. `link_fixed`

   连杆级固定惩罚方法。所有胶囊连杆都参与风险计算，环境同样把 cost
   乘固定权重扣进 reward。

3. `predictive_link`

   使用预测式 observation schema 时的固定惩罚方法。它保留固定惩罚训练
   逻辑，但 observation/info 中可以包含预测时间窗内的未来风险信息。

4. `ldrc_fixed`

   约束 SAC 方法。环境返回原始 reward 和单独 cost，不把 cost 直接扣进
   reward；`SACAgent` 会用 cost critic 和拉格朗日乘子处理安全约束。

5. `ldrc_adaptive`

   在 LDRC 基础上使用自适应 EMA 平滑。风险越高，beta 越接近 beta_max，
   当前策略命令占比越高；风险低时 beta 更小，动作更平滑。

七、当前风险和预测风险的区别
--------------------------------

当前风险 `LinkRisk`：

    当前胶囊体 + 上一帧胶囊体 + 当前障碍物
      -> 当前距离、当前接近速度、当前 TTC、当前 per-link risk

预测风险 `PredictiveLinkRisk`：

    当前胶囊体 + 上一帧胶囊体 + 当前障碍物速度
      -> 在 horizon 内线性外推多次
      -> 未来最小距离 d_pred
      -> 首次进入安全距离时间 t_enter
      -> 预测 per-link risk

代码里 `_compute_risk()` 返回当前帧风险，`_compute_predictive_risk()` 返回
预测风险。是否把预测风险拼进 observation，取决于 observation schema，
例如 `link_risk_pred_v1`。

八、几个容易混淆的状态变量
--------------------------------

- `time_step`

  PyBullet 的物理积分步长，通常比较小。

- `control_dt`

  SAC 策略输出动作的周期，通常是多个 `time_step`。

- `sim_substeps`

  一次 `env.step()` 内包含多少个 PyBullet 物理子步：

      sim_substeps = round(control_dt / time_step)

- `action`

  actor 输出的归一化动作，范围是 [-1, 1]。

- `qdot_policy`

  `action * action_scale` 后的原始策略速度。

- `qdot_policy_limited`

  经过策略步速度变化率限制后的速度。

- `qdot_cmd`

  本控制周期最终命令。对于 RTB，它等于子步轨迹最后一个速度点。

- `qdot_substeps`

  PyBullet 每个物理子步下发的目标速度轨迹；实际反馈是
  `measured_qdot_substeps`，二者不能混写。

- `prev_qdot_cmd`

  上一个控制周期结束时的速度命令，用于 reward 平滑项和执行管线。

- `prev_physics_qdot_cmd`

  上一个物理子步执行的速度，用于统计物理频率下的加速度和 jerk。

- `prev_capsules`

  上一控制时刻的胶囊体状态，用于估计连杆运动速度。

- `beta`

  平滑系数。固定方法使用 fixed_beta，自适应方法根据风险动态更新。

- `last_risk`

  最近一次 `_get_obs_and_info()` 计算得到的风险结果，`step()` 后半段会复用它。

九、info 字典为什么这么大
--------------------------------

`info` 不是神经网络输入，而是训练、评估、画图和诊断用的指标集合。常见字段：

- `goal`、`goal_velocity`、`ee_pos`：目标和末端状态。
- `goal_error_norm`：末端到目标距离。
- `obstacle_center`、`obstacle_velocity`：兼容旧脚本的第一个障碍物状态。
- `obstacle_centers`、`obstacle_velocities`：多障碍物数组形式。
- `risk_global`、`d_min`、`closest_link`：当前帧核心风险指标。
- `risk_body`：每根连杆的当前风险。
- `risk_pred_body`、`d_pred`、`t_enter_pred`：预测风险诊断字段。
- `reward`、`cost`、`success`、`collision`：训练结果。
- `qdot_cmd`、`qdot_policy`、`qdot_substeps`：动作执行信号。
- `joint_acc`、`joint_jerk`：控制周期 endpoint 的命令导数。
- `command_*_substeps`、`measured_*_substeps`：240 Hz 命令与反馈导数。
- `control_min_distance`、`control_max_risk`、`substep_contact`：控制周期内逐子步安全测量。

十、读这份文件的建议
--------------------------------

第一次读可以只看这些函数：

    __init__()
    reset()
    step()
    _get_obs_and_info()
    _compute_risk()
    _compute_predictive_risk()

第二次再看 PyBullet 细节函数：

    _joint_state()
    _end_effector_state()
    _capsules()
    _reset_robot()
    _resolve_robot_references()
    _move_target()
    _advance_obstacle()

最后再看可视化创建函数：

    _create_floor()
    _create_obstacle()
    _create_goal_marker()

这样读会顺很多：先理解强化学习 transition，再理解几何和仿真实现细节。
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from rl_risk_sac.collision import LinkRiskDetector, NullRiskDetector, RiskDetector
from rl_risk_sac.control import ExecutionPipeline, SafetyQP, quintic_endpoint_bounds
from rl_risk_sac.envs.pybullet_scene import PyBulletScene
from rl_risk_sac.robots.pybullet_robot import PyBulletRobot
from rl_risk_sac.robots.ur5_capsules import CapsuleState
from rl_risk_sac.scene import ObstacleState, SphericalObstacleProvider
from rl_risk_sac.tasks import ReachingObservationBuilder, ReachingObjective, WorkspaceTargetProvider
from rl_risk_sac.utils.predictive_risk import (
    PredictiveLinkRisk,
    compute_predictive_link_risk,
)
from rl_risk_sac.utils.runtime_config import RuntimeConfig, resolve_robot_urdf


METHODS = {"ee_fixed", "link_fixed", "predictive_link", "ldrc_fixed", "ldrc_adaptive"}


class UR5DynamicObstacleEnv(gym.Env):
    """UR5 joint-velocity reaching task with dynamic spherical obstacles.

    这个类是整套项目的组装入口：PyBullet 只负责物理仿真，风险、
    observation、reward/cost 和动作平滑都委托给 src 下的其他模块。
    读代码时可以把它当成论文实验的“主循环”。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: dict[str, Any], method: str, render_mode: str | None = None):
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {sorted(METHODS)}")
        runtime = RuntimeConfig.from_mapping(config)
        self.runtime_config = runtime
        self.method = method
        self.render_mode = render_mode

        env_cfg = runtime.env
        smoothing_cfg = runtime.smoothing
        self.reward_cfg = runtime.reward

        self.time_step = env_cfg.time_step
        self.control_dt = env_cfg.control_dt
        # SAC 策略按 control_dt 输出一次动作；PyBullet 用更小的 time_step
        # 积分，所以一次 env.step 会包含多个物理子步。
        self.sim_substeps = max(1, int(round(self.control_dt / self.time_step)))
        self.max_episode_steps = env_cfg.max_episode_steps
        self.action_scale = env_cfg.action_scale
        self.fixed_beta = env_cfg.fixed_beta
        self.success_tolerance = env_cfg.success_tolerance
        self.workspace = env_cfg.workspace
        self.obstacle_cfg = env_cfg.obstacle
        self.goal_cfg = env_cfg.goal
        self.robot_cfg = runtime.robot
        self.observation_cfg = env_cfg.observation
        self.visual_cfg = env_cfg.visual
        self.execution_cfg = env_cfg.execution
        self.fixed_smoothing_mode = self.execution_cfg.fixed_smoothing_mode
        self.safety_qp_cfg = self.execution_cfg.safety_qp
        self.safety_qp_enabled = self.safety_qp_cfg.enabled
        self.safety_qp_gain = self.safety_qp_cfg.gain
        self.safety_qp_activation_distance = self.safety_qp_cfg.activation_distance
        self.safety_qp_fd_epsilon = self.safety_qp_cfg.fd_epsilon
        self.safety_qp_trajectory_mode = self.safety_qp_cfg.trajectory_mode
        self.safety_qp_max_acceleration = self.safety_qp_cfg.motion_bounds.max_acceleration
        self.safety_qp_max_jerk = self.safety_qp_cfg.motion_bounds.max_jerk
        self.obstacle_enabled = self.obstacle_cfg.enabled

        self.beta_min = smoothing_cfg.beta_min
        self.beta_max = smoothing_cfg.beta_max
        self.risk_high = smoothing_cfg.risk_high
        self.lambda_beta = smoothing_cfg.lambda_beta

        self.risk_config = runtime.risk
        self.cost_cfg = runtime.risk_cost

        # 每个环境实例自己持有随机数生成器，保证 reset(seed=...) 可复现实验。
        self.rng = np.random.default_rng(runtime.seed)
        self.physics_client_id = p.connect(p.GUI if render_mode == "human" or env_cfg.gui else p.DIRECT)
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client_id)
        p.setGravity(*env_cfg.gravity, physicsClientId=self.physics_client_id)

        self.repo_root = Path(__file__).resolve().parents[3]
        self.robot_urdf = resolve_robot_urdf(self.repo_root, self.robot_cfg)
        self.robot = PyBulletRobot(self.robot_cfg, self.robot_urdf, self.physics_client_id)
        self.capsule_model = self.robot.capsule_model
        self.scene = PyBulletScene(self.physics_client_id, self.obstacle_cfg, self.visual_cfg)
        self.joint_ids: list[int] = []
        self.tool_link_id = -1
        self.joint_count = len(self.robot_cfg.joint_names)
        self.execution_pipeline = ExecutionPipeline(
            joint_count=self.joint_count,
            action_scale=self.action_scale,
            control_dt=self.control_dt,
            smoothing_mode=self.fixed_smoothing_mode,
            fixed_beta=self.fixed_beta,
            cutoff_angular_frequency=self.execution_cfg.cutoff_angular_frequency,
            max_policy_velocity_delta=self.execution_cfg.max_policy_velocity_delta,
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            risk_high=self.risk_high,
            lambda_beta=self.lambda_beta,
        )
        self.safety_qp = SafetyQP(self.safety_qp_cfg.solver)
        # Kept as a compatibility alias for analysis scripts that inspected the
        # RTB filter state directly before the execution pipeline was extracted.
        self.fixed_rtb = self.execution_pipeline.rtb
        distance_clip = self.observation_cfg.distance_clip
        self.observation_schema = self.observation_cfg.schema_version
        self.predictive_risk_config = self.observation_cfg.predictive_risk
        self.observation_builder = ReachingObservationBuilder(
            action_scale=self.action_scale,
            distance_clip=(float(distance_clip[0]), float(distance_clip[1])),
            v_max=self.risk_config.v_max,
            ttc_max=self.risk_config.ttc_max,
            schema_version=self.observation_schema,
            prediction_horizon=self.predictive_risk_config.horizon,
            include_predictive_per_link_score=self.observation_cfg.include_predictive_per_link_score,
        )
        self.objective = ReachingObjective(self.reward_cfg, self.cost_cfg, self.risk_config.d_safe)
        self.target_provider = WorkspaceTargetProvider(self.goal_cfg, self.workspace)
        self.obstacle_provider = SphericalObstacleProvider(self.obstacle_cfg)
        # 风险检测只接收“胶囊体状态 + 障碍物状态”，不直接依赖 PyBullet。
        # ee_fixed 在检测器内部把非末端连杆风险置零，用作论文中的末端基线。
        detector_args = {
            "config": self.risk_config,
            "obstacle_radius": self.obstacle_cfg.radius,
            "dt": self.control_dt,
            "end_effector_only": self.method == "ee_fixed",
            "no_obstacle_distance": self.observation_cfg.no_obstacle_distance,
        }
        self.risk_detector: RiskDetector = (
            LinkRiskDetector(**detector_args)
            if self.obstacle_enabled
            else NullRiskDetector(
                config=self.risk_config,
                no_obstacle_distance=self.observation_cfg.no_obstacle_distance,
            )
        )

        self.robot_id: int | None = None
        self.obstacle_id: int | None = None
        self.obstacle_ids: list[int] = []
        self.goal_marker_id: int | None = None
        self.prev_capsules: list[CapsuleState] | None = None
        self.prev_goal_error_norm = 0.0
        self.prev_qdot_cmd = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_joint_acc = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_physics_qdot_cmd = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_physics_joint_acc = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_measured_qdot = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_measured_joint_acc = np.zeros(self.joint_count, dtype=np.float32)
        self.beta = self.fixed_beta
        self.step_count = 0
        self.goal = np.zeros(3, dtype=np.float32)
        self.goal_velocity = np.zeros(3, dtype=np.float32)
        self.obstacle_center = np.zeros(3, dtype=np.float32)
        self.obstacle_velocity = np.zeros(3, dtype=np.float32)
        self.obstacle_states: tuple[ObstacleState, ...] = ()
        self.last_risk = None
        self.last_info: dict[str, Any] = {}

        obs_dim = self.observation_builder.dimension(self.joint_count, self.capsule_model.count)
        obs_bound = self.observation_cfg.space_bound
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.joint_count,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_bound, high=obs_bound, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Start a new episode and return the first observation.

        reset 做四件事：重建仿真场景、重置机械臂、采样目标和障碍物、
        初始化上一时刻状态。后续风险速度项会用到 prev_capsules。
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.action_space.seed(seed)

        p.resetSimulation(physicsClientId=self.physics_client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client_id)
        p.setGravity(*self.runtime_config.env.gravity, physicsClientId=self.physics_client_id)
        self.scene.create_floor()
        self.robot_id = self.robot.load(self.rng)
        self.joint_ids = self.robot.joint_ids
        self.tool_link_id = self.robot.tool_link_id
        # 目标和障碍物由 provider 管理，环境只负责把它们同步到 PyBullet 可视物体。
        target_state = self.target_provider.reset(self.rng)
        self.goal = target_state.position
        self.goal_velocity = target_state.velocity
        self.obstacle_states = self.obstacle_provider.reset(self.rng)
        self._sync_legacy_obstacle_state()
        self.scene.populate(self.obstacle_states if self.obstacle_enabled else (), self.goal)
        self.obstacle_ids = self.scene.obstacle_ids
        self.obstacle_id = self.obstacle_ids[0] if self.obstacle_ids else None
        self.goal_marker_id = self.scene.goal_marker_id

        self.prev_qdot_cmd = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_joint_acc = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_physics_qdot_cmd = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_physics_joint_acc = np.zeros(self.joint_count, dtype=np.float32)
        _, measured_qdot = self._joint_state()
        self.prev_measured_qdot = measured_qdot.copy()
        self.prev_measured_joint_acc = np.zeros(self.joint_count, dtype=np.float32)
        # 约束自适应方法从 beta_min 起步；固定方法直接使用配置里的 fixed_beta。
        self.execution_pipeline.reset(adaptive=self.method.endswith("adaptive"))
        self.beta = self.fixed_beta if self.method.endswith("fixed") else self.beta_min
        self.step_count = 0
        self.prev_capsules = self._capsules()
        self.prev_goal_error_norm = float(np.linalg.norm(self._goal_error()))
        obs, info = self._get_obs_and_info()
        return obs, info

    def step(self, action: np.ndarray):
        """Advance one policy step.

        action 是 actor 输出的 [-1, 1] 归一化关节速度。这里先根据执行管线
        变成物理子步速度轨迹，再推进 PyBullet，最后计算 observation、reward、
        cost 和终止条件。
        """
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # 用执行动作前的风险决定自适应平滑强度，避免把本步执行后的结果
        # 反过来影响已经发生的控制命令。
        pre_risk = self._compute_risk()
        raw_policy_velocity, limited_policy_velocity, rate_limited = self.execution_pipeline.prepare_policy_velocity(
            action
        )
        if self.safety_qp_enabled and self.safety_qp_trajectory_mode == "filtered_endpoint_qp":
            qp_nominal_velocity = self.execution_pipeline.filter_rtb_target(limited_policy_velocity)
            qp_info = self._apply_safety_qp(pre_risk, qp_nominal_velocity)
            execution = self.execution_pipeline.interpolate_safe_endpoint(
                raw_policy_velocity=raw_policy_velocity,
                limited_policy_velocity=limited_policy_velocity,
                safe_endpoint_velocity=qp_info["safe_policy_velocity"],
                previous_command=self.prev_qdot_cmd,
                sample_count=self.sim_substeps,
                rate_limited=rate_limited,
            )
        else:
            qp_info = self._apply_safety_qp(pre_risk, limited_policy_velocity)
            execution = self.execution_pipeline.smooth_prepared_velocity(
                raw_policy_velocity=raw_policy_velocity,
                limited_policy_velocity=limited_policy_velocity,
                smoothing_target_velocity=qp_info["safe_policy_velocity"],
                previous_command=self.prev_qdot_cmd,
                risk_global=pre_risk.risk_global,
                sample_count=self.sim_substeps,
                adaptive=self.method.endswith("adaptive"),
                rate_limited=rate_limited,
            )
        qdot_trajectory = execution.trajectory
        qdot_cmd = execution.command
        beta = execution.beta

        command_accelerations = []
        command_jerks = []
        measured_positions = []
        measured_velocities = []
        measured_accelerations = []
        measured_jerks = []
        substep_distances = []
        substep_raw_distances = []
        substep_risks = []
        substep_contacts = []
        previous_substep_capsules = self._capsules()
        for qdot_substep in qdot_trajectory:
            # 每个物理子步都移动障碍物和动态目标，使控制频率低于仿真频率时
            # 场景运动仍然连续。
            self._advance_obstacle(self.time_step)
            self._move_target(self.time_step)
            p.setJointMotorControlArray(
                self.robot_id,
                self.joint_ids,
                p.VELOCITY_CONTROL,
                targetVelocities=qdot_substep.tolist(),
                forces=[self.execution_cfg.joint_motor_force] * self.joint_count,
                physicsClientId=self.physics_client_id,
            )
            p.stepSimulation(physicsClientId=self.physics_client_id)
            command_acc = (qdot_substep - self.prev_physics_qdot_cmd) / self.time_step
            command_jerk = (command_acc - self.prev_physics_joint_acc) / self.time_step
            command_accelerations.append(command_acc.copy())
            command_jerks.append(command_jerk.copy())
            self.prev_physics_qdot_cmd = qdot_substep.copy()
            self.prev_physics_joint_acc = command_acc.copy()

            # R1 measurement contract: every 240 Hz physics step is followed by
            # simulator feedback, capsule geometry and contact sampling.  These
            # are measurements of the executed transition, not aliases for the
            # interpolated command trajectory.
            measured_q, measured_qdot = self._joint_state()
            measured_acc = (measured_qdot - self.prev_measured_qdot) / self.time_step
            measured_jerk = (measured_acc - self.prev_measured_joint_acc) / self.time_step
            current_capsules = self._capsules()
            substep_risk = self.risk_detector.detect(
                capsules=current_capsules,
                previous_capsules=previous_substep_capsules,
                obstacles=self.obstacle_states,
                dt=self.time_step,
            )
            contact = self._has_contact()

            measured_positions.append(measured_q.copy())
            measured_velocities.append(measured_qdot.copy())
            measured_accelerations.append(measured_acc.copy())
            measured_jerks.append(measured_jerk.copy())
            substep_distances.append(float(substep_risk.d_min))
            substep_raw_distances.append(float(substep_risk.d_min_raw))
            substep_risks.append(float(substep_risk.risk_global))
            substep_contacts.append(bool(contact))
            self.prev_measured_qdot = measured_qdot.copy()
            self.prev_measured_joint_acc = measured_acc.copy()
            previous_substep_capsules = current_capsules

        self.step_count += 1
        obs, info = self._get_obs_and_info()
        risk = self.last_risk
        # 碰撞和成功是 episode 的两个正常终止原因；步数耗尽是 truncated。
        control_min_distance = min(substep_distances) if substep_distances else float(risk.d_min)
        control_min_distance_raw = (
            min(substep_raw_distances) if substep_raw_distances else float(risk.d_min_raw)
        )
        control_max_risk = max(substep_risks) if substep_risks else float(risk.risk_global)
        # Collision truth comes from PyBullet contact.  Capsule overlap is an
        # approximate geometry signal and is already represented by the safety
        # violation/cost; treating it as contact would contradict the R1
        # calibration contract and terminate episodes on capsule false positives.
        collision = bool(any(substep_contacts))
        control_safety_violation = control_min_distance < self.risk_config.d_safe
        success = (
            info["goal_error_norm"] < self.success_tolerance
            and not control_safety_violation
            and not collision
        )
        terminated = bool(collision or success)
        truncated = self.step_count >= self.max_episode_steps

        reward = self.objective.reward(
            goal_error_norm=float(info["goal_error_norm"]),
            previous_goal_error_norm=self.prev_goal_error_norm,
            command=qdot_cmd,
            previous_command=self.prev_qdot_cmd,
            success=success,
            collision=collision,
        )
        cost = self.objective.cost(
            risk,
            collision,
            risk_global=control_max_risk,
            d_min=control_min_distance,
        )
        predictive_reward_penalty = 0.0
        if self.method in {"ee_fixed", "link_fixed", "predictive_link"}:
            # 固定惩罚方法把风险代价直接扣进 reward；LDRC 方法把 cost
            # 单独交给 SACAgent 的 cost critic 和拉格朗日乘子处理。
            reward -= self.runtime_config.sac.fixed_risk_penalty * cost
        if self.method == "predictive_link":
            # Observation-only predictive SAC did not learn to exploit its early
            # warning reliably. Keep this shaping term opt-in so all historical
            # checkpoints and link_fixed baselines retain their exact objective.
            predictive_signal = float(info.get("risk_pred_body", 0.0))
            if self.runtime_config.sac.predictive_risk_penalty_mode == "excess":
                # Isolate early-warning information instead of paying twice for
                # risk already represented by the current-risk cost.
                predictive_signal = max(predictive_signal - float(info["risk_global"]), 0.0)
            predictive_reward_penalty = self.runtime_config.sac.predictive_risk_penalty * predictive_signal
            reward -= predictive_reward_penalty

        # policy 频率导数、240 Hz 命令导数和 240 Hz 反馈导数分开记录；
        # 不再把插值命令误称为物理反馈。
        joint_acc = (qdot_cmd - self.prev_qdot_cmd) / self.control_dt
        jerk = (joint_acc - self.prev_joint_acc) / self.control_dt
        command_acceleration = np.asarray(command_accelerations, dtype=np.float32)
        command_jerk = np.asarray(command_jerks, dtype=np.float32)
        measured_position = np.asarray(measured_positions, dtype=np.float32)
        measured_velocity = np.asarray(measured_velocities, dtype=np.float32)
        measured_acceleration = np.asarray(measured_accelerations, dtype=np.float32)
        measured_jerk = np.asarray(measured_jerks, dtype=np.float32)
        info.update(
            {
                "reward": float(reward),
                "cost": float(cost),
                "predictive_reward_penalty": float(predictive_reward_penalty),
                "predictive_reward_signal": float(
                    predictive_reward_penalty
                    / max(self.runtime_config.sac.predictive_risk_penalty, 1e-12)
                )
                if self.method == "predictive_link"
                else 0.0,
                "collision": bool(collision),
                "success": bool(success),
                "control_min_distance": float(control_min_distance),
                "control_min_distance_raw": float(control_min_distance_raw),
                "control_max_risk": float(control_max_risk),
                "control_safety_violation": bool(control_safety_violation),
                "control_collision": bool(collision),
                "substep_distance": np.asarray(substep_distances, dtype=np.float32),
                "substep_distance_raw": np.asarray(substep_raw_distances, dtype=np.float32),
                "substep_risk": np.asarray(substep_risks, dtype=np.float32),
                "substep_contact": np.asarray(substep_contacts, dtype=np.bool_),
                "qdot_cmd": qdot_cmd.copy(),
                "qdot_policy": execution.policy_velocity.copy(),
                "qdot_policy_limited": execution.limited_policy_velocity.copy(),
                "qdot_policy_safe_target": qp_info["safe_policy_velocity"].copy(),
                "policy_rate_limited": bool(execution.rate_limited),
                "joint_acc": joint_acc.copy(),
                "joint_jerk": jerk.copy(),
                "command_rms_acceleration": float(np.sqrt(np.mean(np.square(command_acceleration)))),
                "command_rms_jerk": float(np.sqrt(np.mean(np.square(command_jerk)))),
                "command_peak_acceleration": float(np.max(np.abs(command_acceleration))),
                "command_peak_jerk": float(np.max(np.abs(command_jerk))),
                "measured_rms_acceleration": float(np.sqrt(np.mean(np.square(measured_acceleration)))),
                "measured_rms_jerk": float(np.sqrt(np.mean(np.square(measured_jerk)))),
                "measured_peak_acceleration": float(np.max(np.abs(measured_acceleration))),
                "measured_peak_jerk": float(np.max(np.abs(measured_jerk))),
                # Deprecated aliases retained for existing analysis scripts.
                # Their semantics are explicitly command-side, not feedback.
                "physics_rms_acceleration": float(np.sqrt(np.mean(np.square(command_acceleration)))),
                "physics_rms_jerk": float(np.sqrt(np.mean(np.square(command_jerk)))),
                "physics_peak_acceleration": float(np.max(np.abs(command_acceleration))),
                "physics_peak_jerk": float(np.max(np.abs(command_jerk))),
                "qdot_substeps": qdot_trajectory.copy(),
                "command_acc_substeps": command_acceleration.copy(),
                "command_jerk_substeps": command_jerk.copy(),
                "measured_q_substeps": measured_position.copy(),
                "measured_qdot_substeps": measured_velocity.copy(),
                "measured_acc_substeps": measured_acceleration.copy(),
                "measured_jerk_substeps": measured_jerk.copy(),
                # Deprecated command-side aliases.
                "joint_acc_substeps": command_acceleration.copy(),
                "joint_jerk_substeps": command_jerk.copy(),
                "beta": float(beta),
                "fixed_smoothing_mode": self.fixed_smoothing_mode,
                "safety_qp_enabled": bool(self.safety_qp_enabled),
                "safety_qp_intervened": bool(qp_info["intervened"]),
                "safety_qp_infeasible": bool(qp_info["infeasible"]),
                "safety_qp_correction_norm": float(qp_info["correction_norm"]),
                "safety_qp_max_slack": float(qp_info["max_slack"]),
                "safety_qp_active_constraints": int(qp_info["active_constraints"]),
                "safety_qp_solve_time_ms": float(qp_info["solve_time_ms"]),
            }
        )
        self.prev_qdot_cmd = qdot_cmd
        self.prev_joint_acc = joint_acc
        self.prev_capsules = self._capsules()
        self.prev_goal_error_norm = info["goal_error_norm"]
        self.beta = float(beta)
        self.last_info = info
        return obs, float(reward), float(cost), terminated, truncated, info

    def _apply_safety_qp(self, risk, policy_velocity: np.ndarray) -> dict[str, Any]:
        target_velocity = np.asarray(policy_velocity, dtype=np.float32)
        default = {
            "safe_policy_velocity": target_velocity.copy(),
            "intervened": False,
            "infeasible": False,
            "correction_norm": 0.0,
            "max_slack": 0.0,
            "active_constraints": 0,
            "solve_time_ms": 0.0,
        }
        if not self.safety_qp_enabled:
            return default
        motion_bounds_enabled = (
            self.safety_qp_max_acceleration is not None or self.safety_qp_max_jerk is not None
        )
        safety_constraint_enabled = (
            self.obstacle_enabled
            and bool(self.obstacle_states)
            and risk.d_min <= self.safety_qp_activation_distance
            and 0 <= risk.closest_link < self.capsule_model.count
        )
        if not safety_constraint_enabled and not motion_bounds_enabled:
            return default
        if safety_constraint_enabled:
            jacobian = self._surface_distance_jacobian(risk.closest_link, self.obstacle_center)
            direction = risk.directions[risk.closest_link]
            obstacle_distance_rate = float(np.dot(direction, self.obstacle_velocity))
            margin = float(risk.distances[risk.closest_link]) - self.risk_config.d_safe
            matrix = np.asarray([-jacobian], dtype=np.float32)
            bound = np.asarray([obstacle_distance_rate + self.safety_qp_gain * margin], dtype=np.float32)
        else:
            matrix = np.empty((0, self.joint_count), dtype=np.float32)
            bound = np.empty(0, dtype=np.float32)
        lower = np.full(self.joint_count, -self.action_scale, dtype=np.float32)
        upper = np.full(self.joint_count, self.action_scale, dtype=np.float32)
        if motion_bounds_enabled:
            motion_lower, motion_upper = quintic_endpoint_bounds(
                previous_velocity=self.prev_qdot_cmd,
                previous_acceleration=self.prev_physics_joint_acc,
                physics_dt=self.time_step,
                sample_count=self.sim_substeps,
                max_acceleration=self.safety_qp_max_acceleration,
                max_jerk=self.safety_qp_max_jerk,
            )
            lower = np.maximum(lower, motion_lower)
            upper = np.minimum(upper, motion_upper)
        result = self.safety_qp.solve(
            nominal_command=target_velocity,
            constraint_matrix=matrix,
            constraint_bound=bound,
            lower_bound=lower,
            upper_bound=upper,
        )
        return {
            "safe_policy_velocity": result.command.copy(),
            "intervened": result.intervened,
            "infeasible": result.infeasible,
            "correction_norm": result.correction_norm,
            "max_slack": result.max_slack,
            "active_constraints": result.active_constraints,
            "solve_time_ms": result.solve_time_ms,
        }

    def _surface_distance_jacobian(self, capsule_index: int, obstacle_center: np.ndarray) -> np.ndarray:
        return self.robot.surface_distance_jacobian(
            capsule_index,
            obstacle_center,
            self.safety_qp_fd_epsilon,
            self.risk_config.eps,
        )

    def close(self) -> None:
        if p.isConnected(self.physics_client_id):
            p.disconnect(self.physics_client_id)

    def get_metrics(self) -> dict[str, Any]:
        return dict(self.last_info)

    def _get_obs_and_info(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Collect simulator state, compute risk, and build the SAC observation."""
        q, q_dot = self._joint_state()
        ee_pos, ee_vel = self._end_effector_state()
        goal_error = self.goal - ee_pos
        goal_velocity_error = self.goal_velocity - ee_vel
        risk = self._compute_risk()
        predictive_risk = self._compute_predictive_risk() if self.observation_schema == "link_risk_pred_v1" else None
        self.last_risk = risk

        obs = self.observation_builder.build(
            q=q,
            q_dot=q_dot,
            goal_error=goal_error,
            goal_velocity_error=goal_velocity_error,
            risk=risk,
            previous_command=self.prev_qdot_cmd,
            beta=self.beta,
            predictive_risk=predictive_risk,
        )
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(f"Observation shape {obs.shape} does not match {self.observation_space.shape}")

        info = {
            "observation_schema": self.observation_schema,
            "goal": self.goal.copy(),
            "goal_velocity": self.goal_velocity.copy(),
            "ee_pos": ee_pos.copy(),
            "goal_error_norm": float(np.linalg.norm(goal_error)),
            # This is episode-level presence, not merely the global configuration
            # switch; R2b uses a frozen no-obstacle episode mixture during training.
            "obstacle_enabled": bool(any(state.enabled for state in self.obstacle_states)),
            "obstacle_configured": bool(self.obstacle_enabled),
            "obstacle_center": self.obstacle_center.copy(),
            "obstacle_velocity": self.obstacle_velocity.copy(),
            "obstacle_centers": np.asarray([state.center for state in self.obstacle_states], dtype=np.float32),
            "obstacle_velocities": np.asarray([state.velocity for state in self.obstacle_states], dtype=np.float32),
            "obstacle_count": int(sum(state.enabled for state in self.obstacle_states)),
            "risk_global": float(risk.risk_global),
            "d_min": float(risk.d_min),
            "d_min_raw": float(risk.d_min_raw),
            "geometry_margin": float(self.risk_config.geometry_margin),
            "closest_link": int(risk.closest_link),
            "safety_violation": bool(risk.d_min < self.risk_config.d_safe),
            "risk_body": risk.risks.copy(),
        }
        if predictive_risk is not None:
            info.update(
                {
                    "risk_pred_body": float(predictive_risk.risk_pred_body),
                    "d_pred": predictive_risk.d_pred.copy(),
                    "d_pred_raw": (
                        predictive_risk.d_pred.copy()
                        if predictive_risk.d_pred_raw is None
                        else predictive_risk.d_pred_raw.copy()
                    ),
                    "t_enter_pred": predictive_risk.t_enter.copy(),
                    "risk_pred_per_link": predictive_risk.risk_pred_per_link.copy(),
                    "critical_link_pred": int(predictive_risk.critical_link),
                }
            )
        return obs, info

    def _compute_risk(self):
        """Compute link-level risk from current capsule and obstacle states."""
        return self.risk_detector.detect(
            capsules=self._capsules(),
            previous_capsules=self.prev_capsules,
            obstacles=self.obstacle_states,
        )

    def _compute_predictive_risk(self) -> PredictiveLinkRisk:
        capsules = self._capsules()
        active_obstacles = [obstacle for obstacle in self.obstacle_states if obstacle.enabled]
        if not active_obstacles:
            count = len(capsules)
            return PredictiveLinkRisk(
                d_pred=np.full(count, self.observation_cfg.no_obstacle_distance, dtype=np.float32),
                t_enter=np.full(count, np.inf, dtype=np.float32),
                risk_pred_per_link=np.zeros(count, dtype=np.float32),
                risk_pred_body=0.0,
                critical_link=0 if count else -1,
                closest_points_pred=np.zeros((count, 3), dtype=np.float32),
                closest_times=np.zeros(count, dtype=np.float32),
                approach_velocities=np.zeros(count, dtype=np.float32),
                d_pred_raw=np.full(count, self.observation_cfg.no_obstacle_distance, dtype=np.float32),
            )

        risks = [
            compute_predictive_link_risk(
                capsules=capsules,
                prev_capsules=self.prev_capsules,
                obstacle_center=obstacle.center,
                obstacle_velocity=obstacle.velocity,
                obstacle_radius=self.obstacle_cfg.radius,
                dt=self.control_dt,
                config=self.predictive_risk_config,
                use_end_effector_only=self.method == "ee_fixed",
            )
            for obstacle in active_obstacles
        ]
        return self._aggregate_predictive_risks(risks)

    @staticmethod
    def _aggregate_predictive_risks(risks: list[PredictiveLinkRisk]) -> PredictiveLinkRisk:
        if len(risks) == 1:
            return risks[0]

        d_stack = np.stack([risk.d_pred for risk in risks])
        raw_d_stack = np.stack(
            [risk.d_pred if risk.d_pred_raw is None else risk.d_pred_raw for risk in risks]
        )
        nearest_obstacle_indices = np.argmin(d_stack, axis=0)
        link_indices = np.arange(d_stack.shape[1])
        per_link_risk = np.max(np.stack([risk.risk_pred_per_link for risk in risks]), axis=0)
        return PredictiveLinkRisk(
            d_pred=d_stack[nearest_obstacle_indices, link_indices].astype(np.float32),
            t_enter=np.min(np.stack([risk.t_enter for risk in risks]), axis=0).astype(np.float32),
            risk_pred_per_link=per_link_risk.astype(np.float32),
            risk_pred_body=float(np.max(per_link_risk)),
            critical_link=int(np.argmax(per_link_risk)),
            closest_points_pred=np.stack([risk.closest_points_pred for risk in risks])[
                nearest_obstacle_indices, link_indices
            ].astype(np.float32),
            closest_times=np.stack([risk.closest_times for risk in risks])[
                nearest_obstacle_indices, link_indices
            ].astype(np.float32),
            approach_velocities=np.max(np.stack([risk.approach_velocities for risk in risks]), axis=0).astype(
                np.float32
            ),
            d_pred_raw=raw_d_stack[nearest_obstacle_indices, link_indices].astype(np.float32),
        )

    def _joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.robot.joint_state()

    def _end_effector_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.robot.end_effector_state()

    def _goal_error(self) -> np.ndarray:
        ee_pos, _ = self._end_effector_state()
        return self.goal - ee_pos

    def _capsules(self) -> list[CapsuleState]:
        """Return the current world-space capsule approximation of the arm."""
        return self.robot.capsules()

    def _has_contact(self) -> bool:
        """Return whether robot-obstacle contact exists at the current physics substep."""
        if not self.obstacle_enabled or not self.obstacle_ids:
            return False
        return self.scene.has_robot_contact(self.robot_id)

    def _move_target(self, dt: float) -> None:
        target_state = self.target_provider.advance(dt)
        self.goal = target_state.position
        self.goal_velocity = target_state.velocity
        self.scene.sync_goal(self.goal)

    def _advance_obstacle(self, dt: float) -> None:
        self.obstacle_states = self.obstacle_provider.advance(dt)
        self._sync_legacy_obstacle_state()
        if not self.obstacle_enabled or not self.obstacle_ids:
            return
        self.scene.sync_obstacles(self.obstacle_states)

    def _sync_legacy_obstacle_state(self) -> None:
        if not self.obstacle_states:
            self.obstacle_center = np.asarray(self.obstacle_cfg.disabled_position, dtype=np.float32)
            self.obstacle_velocity = np.zeros(3, dtype=np.float32)
            return
        self.obstacle_center = self.obstacle_states[0].center
        self.obstacle_velocity = self.obstacle_states[0].velocity
