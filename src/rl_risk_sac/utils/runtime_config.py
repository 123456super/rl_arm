from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

FloatRange: TypeAlias = tuple[float, float]
Vector3: TypeAlias = tuple[float, float, float]
Color: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True)
class CapsuleSpec:
    name: str
    parent_link_name: str
    child_link_name: str
    radius: float
    parent_offset: Vector3
    child_offset: Vector3
    allow_degenerate: bool


@dataclass(frozen=True)
class RiskConfig:
    d_safe: float
    geometry_margin: float
    sigma_d: float
    v_max: float
    ttc_max: float
    tau: float
    eps: float
    eps_v: float
    w_distance: float
    w_velocity: float
    w_ttc: float


@dataclass(frozen=True)
class PredictiveRiskConfig:
    horizon: float
    step: float
    d_safe: float
    geometry_margin: float
    sigma_d: float
    tau_enter: float
    v_max: float
    eps: float
    w_min_distance: float
    w_enter_time: float
    w_approach: float


@dataclass(frozen=True)
class SafetyQPConfig:
    max_iterations: int
    eps: float
    violation_tolerance: float


def _range(value: Any) -> FloatRange:
    return float(value[0]), float(value[1])


def _vector3(value: Any) -> Vector3:
    return float(value[0]), float(value[1]), float(value[2])


def _color(value: Any) -> Color:
    return float(value[0]), float(value[1]), float(value[2]), float(value[3])


@dataclass(frozen=True)
class RobotResetConfig:
    default_joint_positions: tuple[float, ...]
    joint_noise_range: float


@dataclass(frozen=True)
class RobotRuntimeConfig:
    urdf: str
    base_position: Vector3
    joint_names: tuple[str, ...]
    tool_link_name: str
    reset: RobotResetConfig
    capsules: tuple[CapsuleSpec, ...]


@dataclass(frozen=True)
class ObservationRuntimeConfig:
    schema_version: str
    space_bound: float
    distance_clip: FloatRange
    no_obstacle_distance: float
    predictive_risk: PredictiveRiskConfig
    include_predictive_per_link_score: bool


@dataclass(frozen=True)
class MotionBoundsConfig:
    max_acceleration: float | None
    max_jerk: float | None


@dataclass(frozen=True)
class SafetyQPRuntimeConfig:
    enabled: bool
    activation_distance: float
    gain: float
    fd_epsilon: float
    trajectory_mode: str
    solver: SafetyQPConfig
    motion_bounds: MotionBoundsConfig


@dataclass(frozen=True)
class ExecutionRuntimeConfig:
    joint_motor_force: float
    max_policy_velocity_delta: float | None
    fixed_smoothing_mode: str
    cutoff_angular_frequency: float
    safety_qp: SafetyQPRuntimeConfig


@dataclass(frozen=True)
class VisualRuntimeConfig:
    floor_position: Vector3
    floor_rgba: Color
    obstacle_rgba: Color
    goal_marker_radius: float
    goal_rgba: Color


@dataclass(frozen=True)
class RandomObstacleConfig:
    x_range: FloatRange
    z_range: FloatRange
    start_y_abs_range: FloatRange
    target_y_abs_range: FloatRange


@dataclass(frozen=True)
class NamedObstacleConfig:
    x_range: FloatRange
    z_range: FloatRange
    start_y_abs: float
    target_y_abs: float


@dataclass(frozen=True)
class ObstacleRuntimeConfig:
    enabled: bool
    episode_enable_probability: float
    scenario: str
    count: int
    radius: float
    speed_range: FloatRange
    disabled_position: Vector3
    bounds: Mapping[str, FloatRange]
    random: RandomObstacleConfig
    scenarios: Mapping[str, NamedObstacleConfig]


@dataclass(frozen=True)
class GoalRuntimeConfig:
    mode: str
    fixed: bool
    position: Vector3
    speed_range: FloatRange


@dataclass(frozen=True)
class EnvironmentRuntimeConfig:
    gui: bool
    gravity: Vector3
    time_step: float
    control_dt: float
    max_episode_steps: int
    action_scale: float
    fixed_beta: float
    success_tolerance: float
    workspace: Mapping[str, FloatRange]
    observation: ObservationRuntimeConfig
    execution: ExecutionRuntimeConfig
    visual: VisualRuntimeConfig
    obstacle: ObstacleRuntimeConfig
    goal: GoalRuntimeConfig


@dataclass(frozen=True)
class RiskCostConfig:
    k_risk: float
    k_violation: float
    k_collision: float
    c_safe: float


@dataclass(frozen=True)
class SmoothingRuntimeConfig:
    beta_min: float
    beta_max: float
    risk_high: float
    lambda_beta: float


@dataclass(frozen=True)
class RewardRuntimeConfig:
    w_position: float
    w_progress: float
    w_smooth: float
    success_bonus: float
    collision_penalty: float


@dataclass(frozen=True)
class DeviceSelectionConfig:
    fallback_to_cpu: bool
    candidate_ids: tuple[int, ...] | None
    min_free_memory_gb: float
    memory_weight: float
    compute_weight: float
    print_summary: bool


@dataclass(frozen=True)
class SACRuntimeConfig:
    gamma: float
    tau: float
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    lambda_lr: float
    cost_ema_rho: float
    batch_size: int
    replay_size: int
    warmup_steps: int
    update_after: int
    update_every: int
    hidden_dims: tuple[int, ...]
    fixed_risk_penalty: float
    predictive_risk_penalty: float
    predictive_risk_penalty_mode: str
    initial_alpha: float
    initial_lambda: float


@dataclass(frozen=True)
class TrainRuntimeConfig:
    method: str
    total_steps: int
    progress_interval: int
    log_interval: int
    save_interval: int
    output_dir: str
    run_name: str | None


@dataclass(frozen=True)
class EvalRuntimeConfig:
    method: str
    checkpoint: str | None
    episodes: int
    seed: int
    output: str | None
    trace_output: str | None


@dataclass(frozen=True)
class SmokeRuntimeConfig:
    method: str
    seed: int
    no_obstacle_seed: int
    max_episode_steps: int
    rollout_steps: int
    replay_capacity: int
    batch_size: int


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    device: str
    device_selection: DeviceSelectionConfig
    robot: RobotRuntimeConfig
    env: EnvironmentRuntimeConfig
    risk: RiskConfig
    risk_cost: RiskCostConfig
    smoothing: SmoothingRuntimeConfig
    reward: RewardRuntimeConfig
    sac: SACRuntimeConfig
    train: TrainRuntimeConfig
    eval: EvalRuntimeConfig
    smoke: SmokeRuntimeConfig

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> RuntimeConfig:
        """Convert a validated resolved YAML mapping into immutable runtime config."""
        robot = config["robot"]
        reset = robot["reset"]
        capsules = tuple(
            CapsuleSpec(
                name=str(item["name"]),
                parent_link_name=str(item["parent_link_name"]),
                child_link_name=str(item["child_link_name"]),
                radius=float(item["radius"]),
                parent_offset=_vector3(item["parent_offset"]),
                child_offset=_vector3(item["child_offset"]),
                allow_degenerate=bool(item["allow_degenerate"]),
            )
            for item in robot["capsules"]
        )
        robot_config = RobotRuntimeConfig(
            urdf=str(robot["urdf"]),
            base_position=_vector3(robot["base_position"]),
            joint_names=tuple(str(name) for name in robot["joint_names"]),
            tool_link_name=str(robot["tool_link_name"]),
            reset=RobotResetConfig(
                default_joint_positions=tuple(float(value) for value in reset["default_joint_positions"]),
                joint_noise_range=float(reset["joint_noise_range"]),
            ),
            capsules=capsules,
        )

        risk = config["risk"]
        weights = risk["weights"]
        risk_config = RiskConfig(
            d_safe=float(risk["d_safe"]),
            geometry_margin=float(risk["geometry_margin"]),
            sigma_d=float(risk["sigma_d"]),
            v_max=float(risk["v_max"]),
            ttc_max=float(risk["ttc_max"]),
            tau=float(risk["tau"]),
            eps=float(risk["eps"]),
            eps_v=float(risk["eps_v"]),
            w_distance=float(weights["distance"]),
            w_velocity=float(weights["velocity"]),
            w_ttc=float(weights["ttc"]),
        )

        env = config["env"]
        observation = env["observation"]
        prediction = observation["predictive_risk"]
        prediction_config = PredictiveRiskConfig(
            horizon=float(prediction["horizon"]),
            step=float(prediction["step"]),
            d_safe=risk_config.d_safe,
            geometry_margin=risk_config.geometry_margin,
            sigma_d=risk_config.sigma_d,
            tau_enter=risk_config.tau,
            v_max=risk_config.v_max,
            eps=risk_config.eps,
            w_min_distance=float(prediction["w_min_distance"]),
            w_enter_time=float(prediction["w_enter_time"]),
            w_approach=float(prediction["w_approach"]),
        )
        execution = env["execution"]
        safety_qp = execution["safety_qp"]
        motion_bounds = safety_qp["motion_bounds"]
        max_acceleration = motion_bounds["max_acceleration"]
        max_jerk = motion_bounds["max_jerk"]
        obstacle = env["obstacle"]
        random_obstacle = obstacle["random"]
        scenarios = MappingProxyType(
            {
                str(name): NamedObstacleConfig(
                    x_range=_range(value["x_range"]),
                    z_range=_range(value["z_range"]),
                    start_y_abs=float(value["start_y_abs"]),
                    target_y_abs=float(value["target_y_abs"]),
                )
                for name, value in obstacle["scenarios"].items()
            }
        )
        obstacle_config = ObstacleRuntimeConfig(
            enabled=bool(obstacle["enabled"]),
            episode_enable_probability=float(obstacle["episode_enable_probability"]),
            scenario=str(obstacle["scenario"]),
            count=int(obstacle["count"]),
            radius=float(obstacle["radius"]),
            speed_range=_range(obstacle["speed_range"]),
            disabled_position=_vector3(obstacle["disabled_position"]),
            bounds=MappingProxyType({axis: _range(value) for axis, value in obstacle["bounds"].items()}),
            random=RandomObstacleConfig(
                x_range=_range(random_obstacle["x_range"]),
                z_range=_range(random_obstacle["z_range"]),
                start_y_abs_range=_range(random_obstacle["start_y_abs_range"]),
                target_y_abs_range=_range(random_obstacle["target_y_abs_range"]),
            ),
            scenarios=scenarios,
        )
        visual = env["visual"]
        env_config = EnvironmentRuntimeConfig(
            gui=bool(env["gui"]),
            gravity=_vector3(env["gravity"]),
            time_step=float(env["time_step"]),
            control_dt=float(env["control_dt"]),
            max_episode_steps=int(env["max_episode_steps"]),
            action_scale=float(env["action_scale"]),
            fixed_beta=float(env["fixed_beta"]),
            success_tolerance=float(env["success_tolerance"]),
            workspace=MappingProxyType({axis: _range(value) for axis, value in env["workspace"].items()}),
            observation=ObservationRuntimeConfig(
                schema_version=str(observation["schema_version"]),
                space_bound=float(observation["space_bound"]),
                distance_clip=_range(observation["distance_clip"]),
                no_obstacle_distance=float(observation["no_obstacle_distance"]),
                predictive_risk=prediction_config,
                include_predictive_per_link_score=bool(prediction["include_per_link_score"]),
            ),
            execution=ExecutionRuntimeConfig(
                joint_motor_force=float(execution["joint_motor_force"]),
                max_policy_velocity_delta=(
                    None
                    if execution["max_policy_velocity_delta"] is None
                    else float(execution["max_policy_velocity_delta"])
                ),
                fixed_smoothing_mode=str(execution["fixed_smoothing_mode"]),
                cutoff_angular_frequency=float(execution["rtb"]["cutoff_angular_frequency"]),
                safety_qp=SafetyQPRuntimeConfig(
                    enabled=bool(safety_qp["enabled"]),
                    activation_distance=float(safety_qp["activation_distance"]),
                    gain=float(safety_qp["gain"]),
                    fd_epsilon=float(safety_qp["fd_epsilon"]),
                    trajectory_mode=str(safety_qp["trajectory_mode"]),
                    solver=SafetyQPConfig(
                        max_iterations=int(safety_qp["max_iterations"]),
                        eps=float(safety_qp["eps"]),
                        violation_tolerance=float(safety_qp["violation_tolerance"]),
                    ),
                    motion_bounds=MotionBoundsConfig(
                        max_acceleration=None if max_acceleration is None else float(max_acceleration),
                        max_jerk=None if max_jerk is None else float(max_jerk),
                    ),
                ),
            ),
            visual=VisualRuntimeConfig(
                floor_position=_vector3(visual["floor_position"]),
                floor_rgba=_color(visual["floor_rgba"]),
                obstacle_rgba=_color(visual["obstacle_rgba"]),
                goal_marker_radius=float(visual["goal_marker_radius"]),
                goal_rgba=_color(visual["goal_rgba"]),
            ),
            obstacle=obstacle_config,
            goal=GoalRuntimeConfig(
                mode=str(env["goal"]["mode"]),
                fixed=bool(env["goal"]["fixed"]),
                position=_vector3(env["goal"]["position"]),
                speed_range=_range(env["goal"]["speed_range"]),
            ),
        )

        cost = risk["cost"]
        smoothing = config["smoothing"]
        reward = config["reward"]
        sac = config["sac"]
        device_selection = config["device_selection"]
        candidate_ids = device_selection["candidate_ids"]
        train = config["train"]
        evaluation = config["eval"]
        smoke = config["smoke"]
        return cls(
            seed=int(config["seed"]),
            device=str(config["device"]),
            device_selection=DeviceSelectionConfig(
                fallback_to_cpu=bool(device_selection["fallback_to_cpu"]),
                candidate_ids=(
                    None if candidate_ids is None else tuple(int(value) for value in candidate_ids)
                ),
                min_free_memory_gb=float(device_selection["min_free_memory_gb"]),
                memory_weight=float(device_selection["memory_weight"]),
                compute_weight=float(device_selection["compute_weight"]),
                print_summary=bool(device_selection["print_summary"]),
            ),
            robot=robot_config,
            env=env_config,
            risk=risk_config,
            risk_cost=RiskCostConfig(
                k_risk=float(cost["k_risk"]),
                k_violation=float(cost["k_violation"]),
                k_collision=float(cost["k_collision"]),
                c_safe=float(cost["c_safe"]),
            ),
            smoothing=SmoothingRuntimeConfig(
                beta_min=float(smoothing["beta_min"]),
                beta_max=float(smoothing["beta_max"]),
                risk_high=float(smoothing["risk_high"]),
                lambda_beta=float(smoothing["lambda_beta"]),
            ),
            reward=RewardRuntimeConfig(
                w_position=float(reward["w_position"]),
                w_progress=float(reward["w_progress"]),
                w_smooth=float(reward["w_smooth"]),
                success_bonus=float(reward["success_bonus"]),
                collision_penalty=float(reward["collision_penalty"]),
            ),
            sac=SACRuntimeConfig(
                gamma=float(sac["gamma"]),
                tau=float(sac["tau"]),
                actor_lr=float(sac["actor_lr"]),
                critic_lr=float(sac["critic_lr"]),
                alpha_lr=float(sac["alpha_lr"]),
                lambda_lr=float(sac["lambda_lr"]),
                cost_ema_rho=float(sac["cost_ema_rho"]),
                batch_size=int(sac["batch_size"]),
                replay_size=int(sac["replay_size"]),
                warmup_steps=int(sac["warmup_steps"]),
                update_after=int(sac["update_after"]),
                update_every=int(sac["update_every"]),
                hidden_dims=tuple(int(value) for value in sac["hidden_dims"]),
                fixed_risk_penalty=float(sac["fixed_risk_penalty"]),
                predictive_risk_penalty=float(sac["predictive_risk_penalty"]),
                predictive_risk_penalty_mode=str(sac["predictive_risk_penalty_mode"]),
                initial_alpha=float(sac["initial_alpha"]),
                initial_lambda=float(sac["initial_lambda"]),
            ),
            train=TrainRuntimeConfig(
                method=str(train["method"]),
                total_steps=int(train["total_steps"]),
                progress_interval=int(train["progress_interval"]),
                log_interval=int(train["log_interval"]),
                save_interval=int(train["save_interval"]),
                output_dir=str(train["output_dir"]),
                run_name=None if train["run_name"] is None else str(train["run_name"]),
            ),
            eval=EvalRuntimeConfig(
                method=str(evaluation["method"]),
                checkpoint=(
                    None if evaluation["checkpoint"] is None else str(evaluation["checkpoint"])
                ),
                episodes=int(evaluation["episodes"]),
                seed=int(evaluation["seed"]),
                output=None if evaluation["output"] is None else str(evaluation["output"]),
                trace_output=(
                    None
                    if evaluation["trace_output"] is None
                    else str(evaluation["trace_output"])
                ),
            ),
            smoke=SmokeRuntimeConfig(
                method=str(smoke["method"]),
                seed=int(smoke["seed"]),
                no_obstacle_seed=int(smoke["no_obstacle_seed"]),
                max_episode_steps=int(smoke["max_episode_steps"]),
                rollout_steps=int(smoke["rollout_steps"]),
                replay_capacity=int(smoke["replay_capacity"]),
                batch_size=int(smoke["batch_size"]),
            ),
        )


def resolve_robot_urdf(repo_root: Path, robot: RobotRuntimeConfig) -> Path:
    path = Path(robot.urdf)
    return path if path.is_absolute() else repo_root / path
