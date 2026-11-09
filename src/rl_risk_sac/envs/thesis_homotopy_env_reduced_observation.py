from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

from rl_risk_sac.envs.thesis_goal_sampling import GoalSamplingMixin
from rl_risk_sac.envs.thesis_dynamic_obstacles import DynamicObstacleMixin
from rl_risk_sac.envs.thesis_static_obstacles import (
    ObstacleSamplingError,
    StaticObstacleMixin,
)
from rl_risk_sac.envs.thesis_reward import compute_step_reward
from rl_risk_sac.prediction import LSTMObstaclePredictor, load_lstm_predictor
from rl_risk_sac.robots.pybullet_robot import PyBulletRobot
from rl_risk_sac.tasks.reference_reaching import (
    OBSTACLE_CLEARANCE_SCALE_M,
    SelfCollisionGeometry,
    ThesisGeometry,
    build_reference_observation,
    closest_points_segment_obb,
    compute_self_collision_geometry,
    compute_thesis_geometry,
    pose_error,
    quaternion_to_matrix,
    reference_point_errors,
    reference_ppo_observation_dim,
)
from rl_risk_sac.utils.risk import closest_point_on_segment
from rl_risk_sac.utils.runtime_config import RuntimeConfig


@dataclass
class EpisodeContract:
    scene: str = "none"
    terminate_on_success: bool = True


class ThesisHomotopyEnv(
    GoalSamplingMixin,
    StaticObstacleMixin,
    DynamicObstacleMixin,
    gym.Env,
):
    """Reference-PPO reaching environment with masked obstacle-set input."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: dict[str, Any], render_mode: str | None = None) -> None:
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        self.render_mode = render_mode
        self.config = config
        self.runtime = RuntimeConfig.from_mapping(config)
        thesis = config["thesis"]
        self.simple_reaching_mode = bool(thesis["simple_reaching_mode"])
        self.terminate_on_collision = bool(thesis["terminate_on_collision"])
        observation = thesis["observation"]
        reward = thesis["reward"]
        configured_reward_mode = str(reward["reward_mode"])
        if configured_reward_mode != "reference_ppo":
            raise ValueError("only reward.reward_mode=reference_ppo is supported")
        self.reward_mode = "reference_ppo"
        self.reference_dynamics_contract = True
        self.physics_dt = float(thesis["physics_dt"])
        self.control_dt = float(thesis["control_dt"])
        self.substeps = int(round(self.control_dt / self.physics_dt))
        if self.substeps != 100 or not np.isclose(
            self.substeps * self.physics_dt, self.control_dt
        ):
            raise ValueError(
                "thesis protocol requires 100 physics substeps per control period"
            )
        self.obstacle_geometry_check_interval = int(
            thesis["obstacle_geometry_check_interval_substeps"]
        )
        if not 1 <= self.obstacle_geometry_check_interval <= self.substeps:
            raise ValueError(
                "obstacle_geometry_check_interval_substeps must be in [1, substeps]"
            )
        self.horizon = int(thesis["horizon"])
        self.action_mode = str(thesis["action_mode"])
        if self.action_mode != "joint_delta":
            raise ValueError("reference PPO requires action_mode=joint_delta")
        self.action_limit = np.full(
            6, float(thesis["action_limit_rad"]), dtype=np.float32
        )
        if np.any(self.action_limit <= 0.0):
            raise ValueError("action_limit_rad must be positive")
        self.joint_target_margin = float(thesis["joint_target_margin_rad"])
        self.joint_limit_tolerance = float(thesis["joint_limit_tolerance_rad"])
        if self.joint_target_margin < 0.0 or self.joint_limit_tolerance <= 0.0:
            raise ValueError("invalid joint target margin or limit tolerance")
        self.qdot_observation_scale = np.full(
            6, float(thesis["qdot_observation_scale"]),
            dtype=np.float32,
        )
        if np.any(self.qdot_observation_scale <= 0.0):
            raise ValueError("qdot observation scales must be positive")
        self.success_hold_steps = int(thesis["success_hold_steps"])
        if self.success_hold_steps < 1:
            raise ValueError("success_hold_steps must be positive")
        self.ee_linear_velocity_scale = float(thesis["ee_linear_velocity_scale"])
        self.ee_angular_velocity_scale = float(thesis["ee_angular_velocity_scale"])
        if self.ee_linear_velocity_scale <= 0.0 or self.ee_angular_velocity_scale <= 0.0:
            raise ValueError("end-effector velocity scales must be positive")
        reference_settings = thesis["reference_ppo"]
        self.reference_success_threshold = float(
            reference_settings["success_threshold"]
        )
        if self.reference_success_threshold <= 0.0:
            raise ValueError("reference_ppo.success_threshold must be positive")
        reference_gravity = reference_settings["gravity"]
        self.gravity = np.asarray(reference_gravity, dtype=np.float64)
        if self.gravity.shape != (3,) or not np.isfinite(self.gravity).all():
            raise ValueError("reference_ppo.gravity must contain three finite values")
        configured_motor_forces = reference_settings["motor_forces"]
        self.motor_forces = np.asarray(configured_motor_forces, dtype=np.float64)
        if self.motor_forces.shape != (6,) or np.any(self.motor_forces <= 0.0):
            raise ValueError("reference_ppo.motor_forces must contain six positive values")
        plane_settings = reference_settings["mounting_plane"]
        self.mounting_plane_enabled = bool(plane_settings["enabled"])
        self.mounting_plane_height = float(plane_settings["height_m"])
        if not np.isfinite(self.mounting_plane_height):
            raise ValueError("reference_ppo.mounting_plane.height_m must be finite")
        ignored_plane_links = plane_settings["ignored_link_names"]
        if not isinstance(ignored_plane_links, list) or not all(
            isinstance(name, str) for name in ignored_plane_links
        ):
            raise ValueError(
                "reference_ppo.mounting_plane.ignored_link_names must be a string list"
            )
        self.mounting_plane_ignored_link_names = set(ignored_plane_links)
        self.mounting_plane_id: int | None = None
        self._mounting_plane_ignored_link_ids: set[int] = set()
        self.obstacle_half_extents_min = np.asarray(
            thesis["obstacle_half_extents_min_m"],
            dtype=np.float32,
        )
        self.obstacle_half_extents_max = np.asarray(
            thesis["obstacle_half_extents_max_m"],
            dtype=np.float32,
        )
        if (
            self.obstacle_half_extents_min.shape != (3,)
            or self.obstacle_half_extents_max.shape != (3,)
            or not np.isfinite(self.obstacle_half_extents_min).all()
            or not np.isfinite(self.obstacle_half_extents_max).all()
            or np.any(self.obstacle_half_extents_min <= 0.0)
            or np.any(self.obstacle_half_extents_max < self.obstacle_half_extents_min)
        ):
            raise ValueError(
                "obstacle OBB half-extents bounds must be finite positive 3-vectors "
                "with min <= max"
            )
        self.obstacle_speed = float(thesis["obstacle_speed"])
        if not np.isfinite(self.obstacle_speed) or self.obstacle_speed <= 0.0:
            raise ValueError("obstacle_speed must be finite and positive")
        dynamic_motion = thesis["dynamic_motion"]
        self.dynamic_motion_mode = str(dynamic_motion["mode"])
        if self.dynamic_motion_mode != "smooth_linear_oscillation":
            raise ValueError(
                "dynamic_motion.mode must be smooth_linear_oscillation"
            )
        dynamic_half_extents = dynamic_motion["half_extents_m"]
        self.dynamic_obstacle_half_extents = np.asarray(
            dynamic_half_extents, dtype=np.float32
        )
        if (
            self.dynamic_obstacle_half_extents.shape != (3,)
            or not np.isfinite(self.dynamic_obstacle_half_extents).all()
            or np.any(self.dynamic_obstacle_half_extents <= 0.0)
        ):
            raise ValueError(
                "dynamic_motion.half_extents_m must contain three finite "
                "positive values"
            )
        dynamic_prediction = thesis["dynamic_prediction"]
        self.dynamic_prediction_enabled = bool(dynamic_prediction["enabled"])
        self.dynamic_prediction_mode = str(dynamic_prediction["mode"])
        if self.dynamic_prediction_enabled and self.dynamic_prediction_mode != "lstm":
            raise ValueError("dynamic_prediction.mode must be lstm")
        self.dynamic_prediction_margin = float(
            dynamic_prediction["margin_m"]
        )
        if (
            not np.isfinite(self.dynamic_prediction_margin)
            or self.dynamic_prediction_margin < 0.0
        ):
            raise ValueError(
                "dynamic_prediction.margin_m must be finite and non-negative"
            )
        self.dynamic_history_sample_hz = float(
            dynamic_prediction["history_sample_hz"]
        )
        self.dynamic_history_length = int(
            dynamic_prediction["history_length"]
        )
        if (
            not np.isfinite(self.dynamic_history_sample_hz)
            or self.dynamic_history_sample_hz <= 0.0
        ):
            raise ValueError(
                "dynamic_prediction.history_sample_hz must be finite and positive"
            )
        if self.dynamic_history_length < 2:
            raise ValueError(
                "dynamic_prediction.history_length must be at least 2"
            )
        history_interval = 1.0 / self.dynamic_history_sample_hz
        self.dynamic_history_interval_substeps = int(
            round(history_interval / self.physics_dt)
        )
        if (
            self.dynamic_history_interval_substeps < 1
            or not np.isclose(
                self.dynamic_history_interval_substeps * self.physics_dt,
                history_interval,
                rtol=0.0,
                atol=1.0e-9,
            )
        ):
            raise ValueError(
                "dynamic prediction history frequency must be an integer "
                "multiple of the physics step"
            )
        horizon_fractions = np.asarray(
            dynamic_prediction["horizon_fractions"],
            dtype=np.float32,
        )
        if (
            horizon_fractions.shape != (3,)
            or not np.isfinite(horizon_fractions).all()
            or np.any(horizon_fractions <= 0.0)
            or np.any(np.diff(horizon_fractions) <= 0.0)
            or not np.isclose(float(horizon_fractions[-1]), 1.0)
        ):
            raise ValueError(
                "dynamic_prediction.horizon_fractions must contain three "
                "strictly increasing positive values ending at 1"
            )
        self.dynamic_prediction_times = (
            horizon_fractions * self.control_dt
        ).astype(np.float32)
        self.dynamic_predictor: LSTMObstaclePredictor | None = None
        if self.dynamic_prediction_enabled:
            checkpoint = dynamic_prediction["checkpoint"]
            if not checkpoint:
                raise ValueError(
                    "dynamic_prediction.checkpoint is required when LSTM prediction is enabled"
                )
            self.dynamic_predictor = load_lstm_predictor(
                checkpoint,
                expected_history_length=self.dynamic_history_length,
                expected_history_hz=self.dynamic_history_sample_hz,
                expected_horizon_times=self.dynamic_prediction_times,
                device=str(dynamic_prediction["device"]),
            )
        self.blocking_obstacle_mode = bool(thesis["blocking_obstacle_mode"])
        self.d_safe = float(thesis["d_safe"])
        self_collision_config = thesis["self_collision"]
        self.d_self_safe = float(self_collision_config["safe_distance_m"])
        self.self_query_distance = float(self_collision_config["query_distance_m"])
        self.self_ttc_max = float(self_collision_config["ttc_max_s"])
        self.self_approach_velocity_scale = float(
            self_collision_config["approach_velocity_scale"]
        )
        self.self_ttc_tau = float(self_collision_config["ttc_tau_s"])
        if not 0.0 < self.d_self_safe < self.self_query_distance:
            raise ValueError("self collision distances must satisfy 0 < safe < query")
        self.position_tolerance = float(thesis["position_tolerance"])
        self.orientation_tolerance = float(thesis["orientation_tolerance"])
        if not 0.0 < self.position_tolerance <= 0.20:
            raise ValueError("position_tolerance must be in (0, 0.20]")
        if not 0.0 < self.orientation_tolerance <= np.pi:
            raise ValueError("orientation_tolerance must be in (0, pi]")
        self.reward_parameters = {
            "reference_obstacle_threshold": float(
                reward["reference_obstacle_threshold_m"]
            ),
            "reference_obstacle_clearance_weight": float(
                reward["reference_obstacle_clearance_weight"]
            ),
            "reference_obstacle_collision_penalty": float(
                reward["reference_obstacle_collision_penalty"]
            ),
            "reference_self_collision_penalty": float(
                reward["reference_self_collision_penalty"]
            ),
            "reference_environment_collision_penalty": float(
                reward["reference_environment_collision_penalty"]
            ),
        }
        for name, value in self.reward_parameters.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.goal_sample_max_attempts = int(thesis["goal_sample_max_attempts"])
        if self.goal_sample_max_attempts < 1:
            raise ValueError("goal_sample_max_attempts must be positive")
        self.scene_sample_max_attempts = int(thesis["scene_sample_max_attempts"])
        if self.scene_sample_max_attempts < 1:
            raise ValueError("scene_sample_max_attempts must be positive")
        fixed_goal = thesis["fixed_orientation_goal"]
        self.fixed_orientation_goal_enabled = bool(fixed_goal["enabled"])
        if not self.fixed_orientation_goal_enabled:
            raise ValueError(
                "the active S0/S1/S2 pipeline requires fixed_orientation_goal.enabled=true"
            )
        self.fixed_goal_euler = np.asarray(
            fixed_goal["euler_rad"], dtype=np.float64
        )
        self.fixed_initial_joint_positions = np.asarray(
            fixed_goal["initial_joint_positions_rad"],
            dtype=np.float64,
        )
        if self.fixed_goal_euler.shape != (3,) or not np.isfinite(self.fixed_goal_euler).all():
            raise ValueError("fixed_orientation_goal.euler_rad must contain 3 finite angles")
        if (
            self.fixed_initial_joint_positions.shape != (6,)
            or not np.isfinite(self.fixed_initial_joint_positions).all()
        ):
            raise ValueError("fixed_orientation_goal.initial_joint_positions_rad must contain 6 finite angles")
        self.fixed_goal_quaternion = np.asarray(
            p.getQuaternionFromEuler(self.fixed_goal_euler.tolist()), dtype=np.float32
        )
        self.contract = EpisodeContract()
        self.rng = np.random.default_rng(int(config["seed"]))

        self.client_id = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError(
                "failed to connect to PyBullet; human rendering requires an available display"
            )
        self.robot = PyBulletRobot(
            self.runtime.robot,
            Path(config["robot"]["urdf"]),
            self.client_id,
        )
        self.robot_id = -1
        self.obstacle_id: int | None = None
        self._obstacle_body_id: int | None = None
        self._obstacle_body_ids: list[int] = []
        self.obstacle_positions = np.empty((0, 3), dtype=np.float32)
        self.obstacle_velocities = np.empty((0, 3), dtype=np.float32)
        self.obstacle_half_extents_all = np.empty((0, 3), dtype=np.float32)
        self.obstacle_quaternions = np.empty((0, 4), dtype=np.float32)
        self.obstacle_dynamic_mask = np.empty(0, dtype=bool)
        self.dynamic_motion_centers = np.empty((0, 3), dtype=np.float32)
        self.dynamic_motion_directions = np.empty((0, 3), dtype=np.float32)
        self.dynamic_motion_amplitudes = np.empty(0, dtype=np.float32)
        self.dynamic_motion_angular_frequencies = np.empty(0, dtype=np.float32)
        self.dynamic_motion_phases = np.empty(0, dtype=np.float32)
        self.dynamic_motion_time = 0.0
        self.dynamic_history_substeps = 0
        self.dynamic_position_history: list[list[tuple[float, np.ndarray]]] = []
        self._dynamic_prediction_cache: dict[int, tuple[float, np.ndarray]] = {}
        self.obstacle_position = np.zeros(3, dtype=np.float32)
        self.obstacle_velocity = np.zeros(3, dtype=np.float32)
        self.obstacle_half_extents = self.obstacle_half_extents_min.copy()
        self.obstacle_quaternion = np.asarray(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32
        )
        self.goal_position = np.zeros(3, dtype=np.float32)
        self.goal_quaternion = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.goal_joint_positions = np.zeros(6, dtype=np.float32)
        self.goal_ik_validation: dict[str, Any] = {}
        self.goal_sample_attempts_last = 0
        self.step_count = 0
        self.success_hold_count = 0
        self.obstacle_contact_seen = False
        self.previous_rho_position = 0.0
        self.previous_rho_orientation = 0.0
        observation_config = thesis["observation"]
        self.reference_point_offset_m = float(
            observation_config["reference_point_offset_m"]
        )
        obstacle_set_config = observation_config["obstacle_set"]
        self.max_obstacles = int(obstacle_set_config["max_obstacles"])
        if self.max_obstacles < 1:
            raise ValueError("observation.obstacle_set.max_obstacles must be positive")
        raw_count_probabilities = thesis["static_obstacle_count_probabilities"]
        if not isinstance(raw_count_probabilities, dict) or not raw_count_probabilities:
            raise ValueError(
                "static_obstacle_count_probabilities must be a non-empty mapping"
            )
        count_probabilities: list[tuple[int, float]] = []
        for raw_count, raw_probability in raw_count_probabilities.items():
            count = int(raw_count)
            probability = float(raw_probability)
            if count < 1 or count > self.max_obstacles:
                raise ValueError(
                    f"static obstacle counts must be in [1, {self.max_obstacles}]"
                )
            if not np.isfinite(probability) or probability < 0.0:
                raise ValueError(
                    "static obstacle count probabilities must be finite and non-negative"
                )
            count_probabilities.append((count, probability))
        count_probabilities.sort()
        probability_sum = sum(probability for _, probability in count_probabilities)
        if probability_sum <= 0.0:
            raise ValueError(
                "static obstacle count probabilities must contain positive mass"
            )
        self.static_obstacle_counts = np.asarray(
            [count for count, _ in count_probabilities], dtype=np.int64
        )
        self.static_obstacle_count_probabilities = np.asarray(
            [probability / probability_sum for _, probability in count_probabilities],
            dtype=np.float64,
        )
        self.hybrid_static_obstacle_count = int(
            thesis["hybrid_static_obstacle_count"]
        )
        if (
            self.hybrid_static_obstacle_count < 1
            or self.hybrid_static_obstacle_count + 1 > self.max_obstacles
        ):
            raise ValueError(
                "hybrid_static_obstacle_count must leave room for one "
                "dynamic obstacle"
            )
        allowed_layouts = {
            "legacy_blocking", "path_offset", "mixed", "structured",
        }
        raw_layout_probabilities = thesis["static_obstacle_layout_probabilities"]
        if not isinstance(raw_layout_probabilities, dict) or not raw_layout_probabilities:
            raise ValueError(
                "static_obstacle_layout_probabilities must be a non-empty mapping"
            )
        layout_probabilities: list[tuple[str, float]] = []
        for raw_layout, raw_probability in raw_layout_probabilities.items():
            layout = str(raw_layout)
            probability = float(raw_probability)
            if layout not in allowed_layouts:
                raise ValueError(
                    f"unknown static obstacle layout {layout!r}; expected one of "
                    f"{sorted(allowed_layouts)}"
                )
            if not np.isfinite(probability) or probability < 0.0:
                raise ValueError(
                    "static obstacle layout probabilities must be finite and non-negative"
                )
            layout_probabilities.append((layout, probability))
        layout_probability_sum = sum(
            probability for _, probability in layout_probabilities
        )
        if layout_probability_sum <= 0.0:
            raise ValueError(
                "static obstacle layout probabilities must contain positive mass"
            )
        self.static_obstacle_layouts = tuple(
            layout for layout, _ in layout_probabilities
        )
        self.static_obstacle_layout_probabilities = np.asarray(
            [
                probability / layout_probability_sum
                for _, probability in layout_probabilities
            ],
            dtype=np.float64,
        )
        self.path_offset_min = float(thesis["path_offset_min_m"])
        self.path_offset_max = float(thesis["path_offset_max_m"])
        self.obstacle_surface_clearance_min = float(
            thesis["obstacle_surface_clearance_min_m"]
        )
        self.structured_offset_min = float(
            thesis["structured_offset_min_m"]
        )
        self.structured_offset_max = float(
            thesis["structured_offset_max_m"]
        )
        if not (
            0.0 <= self.path_offset_min <= self.path_offset_max
            and self.obstacle_surface_clearance_min >= 0.0
            and 0.0 <= self.structured_offset_min <= self.structured_offset_max
        ):
            raise ValueError("invalid static obstacle layout distance settings")
        self.obstacle_layout = "none"
        if self.reference_point_offset_m <= 0.0:
            raise ValueError("observation.reference_point_offset_m must be positive")
        self._link_name_to_id: dict[str, int] = {}
        self._collision_link_ids: set[int] = set()
        self._self_link_order: list[int] = []
        self._self_link_rank: dict[int, int] = {}
        self._self_pair_link_ids: list[tuple[int, int]] = []
        self._self_pair_slots: list[tuple[int, int]] = []

        # Loading the URDF dominates short curriculum episodes. Keep one world
        # alive for the lifetime of the environment and reset only its state.
        self._initialize_world()

        observation_dim = reference_ppo_observation_dim(self.max_obstacles)
        low = np.full(observation_dim, -np.inf, dtype=np.float32)
        high = np.full(observation_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low, high,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            shape=(6,),
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
        )

    def configure_episode(
        self, scene: str, *, terminate_on_success: bool = True,
    ) -> None:
        if scene not in {"none", "static", "dynamic", "hybrid"}:
            raise ValueError(
                "scene must be one of: none, static, dynamic, hybrid"
            )
        self.contract = EpisodeContract(
            scene=scene,
            terminate_on_success=bool(terminate_on_success),
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        total_obstacle_proposals = 0
        total_obstacle_sampling_time = 0.0
        last_sampling_error: ObstacleSamplingError | None = None
        for scene_sample_attempt in range(1, self.scene_sample_max_attempts + 1):
            self._disable_obstacle()
            self._reset_fixed_initial_state()
            self._sample_fixed_orientation_goal_pose()
            try:
                self._create_obstacle()
            except ObstacleSamplingError as error:
                total_obstacle_proposals += self.obstacle_sampling_proposals
                total_obstacle_sampling_time += self.obstacle_sampling_time_s
                last_sampling_error = error
                continue
            total_obstacle_proposals += self.obstacle_sampling_proposals
            total_obstacle_sampling_time += self.obstacle_sampling_time_s
            self.obstacle_sampling_proposals = total_obstacle_proposals
            self.obstacle_sampling_time_s = total_obstacle_sampling_time
            self.scene_sampling_attempts = scene_sample_attempt
            break
        else:
            raise ObstacleSamplingError(
                "failed to sample a feasible goal and obstacle scene after "
                f"{self.scene_sample_max_attempts} goal attempts; "
                f"last error: {last_sampling_error}"
            ) from last_sampling_error
        self.step_count = 0
        self.success_hold_count = 0
        self.obstacle_contact_seen = False
        observation, info, geometry, self_geometry, position_error, orientation_error = self._observe()
        self.previous_rho_position = float(np.linalg.norm(position_error))
        self.previous_rho_orientation = float(np.linalg.norm(orientation_error))
        info.update(self._contract_info())
        info.update(self._geometry_info(geometry))
        info.update(self._self_geometry_info(self_geometry))
        return observation, info





    def _initialize_world(self) -> None:
        p.resetSimulation(physicsClientId=self.client_id)
        p.setTimeStep(self.physics_dt, physicsClientId=self.client_id)
        p.setGravity(*self.gravity.tolist(), physicsClientId=self.client_id)
        # Robot.load currently initializes its joints. A private generator
        # prevents that one-time initialization from consuming episode RNG.
        self.robot_id = self.robot.load(
            np.random.default_rng(0), enable_self_collision=True,
        )
        self._resolve_collision_links()
        if self.mounting_plane_enabled:
            plane_shape = p.createCollisionShape(
                p.GEOM_PLANE,
                planeNormal=[0.0, 0.0, 1.0],
                physicsClientId=self.client_id,
            )
            self.mounting_plane_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=plane_shape,
                basePosition=[0.0, 0.0, self.mounting_plane_height],
                physicsClientId=self.client_id,
            )
        self._disable_obstacle()


    def _advance_success_hold(self, strict_pose_reached: bool) -> bool:
        """Advance the consecutive strict-pose counter and report success."""
        self.success_hold_count = (
            self.success_hold_count + 1 if strict_pose_reached else 0
        )
        return bool(
            strict_pose_reached and self.success_hold_count >= self.success_hold_steps
        )

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("action must contain six finite joint-angle increments")
        q_before, qdot_before = self.robot.joint_state()
        lower, upper = self.robot.joint_lower_limits, self.robot.joint_upper_limits
        command_lower = np.maximum(lower, -self.action_limit)
        command_upper = np.minimum(upper, self.action_limit)
        target_positions = np.clip(q_before + action, command_lower, command_upper)
        projection_info = {
            "projection_applied": False,
            "projection_iterations": 0,
            "projection_max_violation_before": 0.0,
            "projection_max_violation_after": 0.0,
        }
        contact_seen_before = self.obstacle_contact_seen
        obstacle_collision = False
        self_collision = False
        environment_collision = False
        joint_limit = False
        max_risk = 0.0
        min_distance = float("inf")
        self_max_risk = 0.0
        self_min_distance = self.self_query_distance
        self_min_ttc = self.self_ttc_max
        self_max_approach = 0.0
        executed_substeps = 0
        max_joint_limit_violation = 0.0

        # POSITION_CONTROL targets persist in Bullet until they are replaced.
        # Sending the same command before every one of the 100 physics steps
        # adds Python/C-API overhead without changing the simulated control.
        p.setJointMotorControlArray(
            self.robot_id,
            self.robot.joint_ids,
            p.POSITION_CONTROL,
            targetPositions=target_positions.tolist(),
            targetVelocities=[0.0] * 6,
            positionGains=[0.05] * 6,
            forces=self.motor_forces.tolist(),
            physicsClientId=self.client_id,
        )
        for substep_index in range(self.substeps):
            # Move kinematic obstacles to the same physical timestamp before
            # Bullet resolves robot contacts for this substep. Moving them
            # afterwards would leave contact queries one 240 Hz step behind.
            self._advance_obstacle()
            p.stepSimulation(physicsClientId=self.client_id)
            executed_substeps += 1
            if self.simple_reaching_mode:
                events = self._collision_events()
                self_collision = self_collision or events["self_collision"]
                q_now, _ = self.robot.joint_state()
                joint_limit = joint_limit or bool(
                    np.any(q_now < lower - self.joint_limit_tolerance)
                    or np.any(q_now > upper + self.joint_limit_tolerance)
                )
                max_joint_limit_violation = max(max_joint_limit_violation, float(np.max(lower - q_now)), float(np.max(q_now - upper)))
                continue
            # Physics/contact detection remains at 240 Hz. Predictive capsule
            # geometry is diagnostic shaping, so sample obstacle clearance at
            # 48 Hz and compute the exact endpoint again in _observe(). The
            # expensive 15-pair self geometry is endpoint-only; real self
            # contacts are still checked on every physics step below.
            should_sample_obstacle_geometry = (
                self.obstacle_id is not None
                and (substep_index + 1) < self.substeps
                and (substep_index + 1)
                % self.obstacle_geometry_check_interval == 0
            )
            if should_sample_obstacle_geometry:
                geometry = self._geometry()
                if geometry is not None:
                    max_risk = max(max_risk, geometry.risk_max)
                    min_distance = min(min_distance, geometry.distance_min)
            events = self._collision_events()
            obstacle_collision = obstacle_collision or events["obstacle_collision"]
            self_collision = self_collision or events["self_collision"]
            environment_collision = environment_collision or events["environment_collision"]
            q_now, _ = self.robot.joint_state()
            max_joint_limit_violation = max(max_joint_limit_violation, float(np.max(lower - q_now)), float(np.max(q_now - upper)))
            joint_limit = joint_limit or max_joint_limit_violation > self.joint_limit_tolerance
            terminal_contact = bool(
                self.terminate_on_collision
                and not self.simple_reaching_mode
                and (obstacle_collision or self_collision)
            )
            # A nonterminal collision must not silently shorten the physical
            # duration of a policy action. Only an explicitly terminal contact
            # may interrupt the 100-substep control period.
            if terminal_contact:
                break

        self.obstacle_contact_seen = self.obstacle_contact_seen or obstacle_collision
        self.step_count += 1
        observation, info, endpoint_geometry, endpoint_self_geometry, position_error, orientation_error = self._observe()
        next_rho_position = float(np.linalg.norm(position_error))
        next_rho_orientation = float(np.linalg.norm(orientation_error))
        next_reference_point_distances = np.asarray(
            info["reference_point_distances"], dtype=np.float32
        )
        raw_obstacle_collision = bool(obstacle_collision)
        raw_self_collision = bool(self_collision)
        raw_environment_collision = bool(environment_collision)
        terminal_obstacle_collision = bool(
            self.terminate_on_collision
            and not self.simple_reaching_mode
            and raw_obstacle_collision
        )
        terminal_self_collision = bool(
            self.terminate_on_collision
            and not self.simple_reaching_mode
            and raw_self_collision
        )
        # S0 preserves the FTP4RM nonterminal contact behavior.  In obstacle
        # stages, configured robot/obstacle or self contact is terminal.
        any_collision = bool(
            raw_obstacle_collision or raw_self_collision
            or raw_environment_collision
        )
        hard_failure = bool(
            terminal_obstacle_collision or terminal_self_collision
        )
        reference_error = float(
            0.5 * np.sum(next_reference_point_distances)
            + 0.25 * next_rho_orientation
        )
        strict_pose_reached = bool(reference_error < self.reference_success_threshold)
        position_reached = strict_pose_reached
        task_reached = self._advance_success_hold(strict_pose_reached)
        terminated = bool(
            terminal_obstacle_collision or terminal_self_collision
            or (task_reached and self.contract.terminate_on_success)
        )
        timeout = bool(self.step_count >= self.horizon and not terminated)
        truncated = bool(timeout)
        _, qdot_after = self.robot.joint_state()
        # Average across joints before the bounded reward mapping. Summing six
        # normalized squared changes made the smoothness term saturate on
        # nearly every transition.
        smooth_velocity = float(
            np.mean(np.square((qdot_after - qdot_before) / self.qdot_observation_scale))
        )
        velocity_magnitude = float(
            np.mean(np.square(qdot_after / self.qdot_observation_scale))
        )
        if endpoint_geometry is None:
            min_distance = 0.80
            max_risk = 0.0
        else:
            min_distance = min(min_distance, endpoint_geometry.distance_min)
            max_risk = max(max_risk, endpoint_geometry.risk_max)
        self_min_distance = min(self_min_distance, endpoint_self_geometry.distance_min)
        self_max_risk = max(self_max_risk, endpoint_self_geometry.risk_max)
        self_min_ttc = min(self_min_ttc, endpoint_self_geometry.ttc_min)
        self_max_approach = max(self_max_approach, endpoint_self_geometry.approach_max)
        if self.simple_reaching_mode:
            max_risk = 0.0
            min_distance = 0.80
            self_max_risk = 0.0
            self_min_distance = self.self_query_distance
            self_min_ttc = self.self_ttc_max
            self_max_approach = 0.0
        reward_result = compute_step_reward(
            reference_error=reference_error,
            obstacle_collision=raw_obstacle_collision,
            self_collision=raw_self_collision,
            environment_collision=raw_environment_collision,
            obstacle_clearances=(
                () if endpoint_geometry is None
                else tuple(float(value) for value in endpoint_geometry.distances)
            ),
            parameters=self.reward_parameters,
            obstacle_risk=max_risk,
            self_risk=self_max_risk,
            obstacle_distance=min_distance,
            self_distance=self_min_distance,
            obstacle_safe_distance=self.d_safe,
            self_safe_distance=self.d_self_safe,
        )
        reward = reward_result.reward
        reward_fields = reward_result.fields
        self_collision_step_penalty = reward_result.self_collision_step_penalty
        collision_assisted_reach = bool(task_reached and self.obstacle_contact_seen)
        safe_success = bool(
            task_reached and not self.obstacle_contact_seen
            and not raw_self_collision and not raw_environment_collision
            and not joint_limit
        )
        cost = reward_result.cost
        info.update(
            {
                **self._contract_info(),
                **self._geometry_info(endpoint_geometry),
                **self._self_geometry_info(endpoint_self_geometry),
                **reward_fields,
                "reward": reward,
                "self_collision_step_penalty": self_collision_step_penalty,
                "hard_penalty": reward_fields["hard_penalty"],
                "safety_penalty": reward_fields["safety_penalty"],
                "terminal_guard_penalty": reward_fields["terminal_guard_penalty"],
                "cost": cost,
                "policy_joint_delta": action.copy(),
                "policy_joint_target": target_positions.copy(),
                "policy_joint_velocity": np.zeros(6, dtype=np.float32),
                "commanded_joint_delta": (target_positions - q_before).copy(),
                "target_joint_positions": target_positions.copy(),
                "scaled_policy_joint_velocity": np.zeros(6, dtype=np.float32),
                "commanded_joint_velocity": np.zeros(6, dtype=np.float32),
                "joint_velocity": qdot_after.copy(),
                **projection_info,
                "task_reached": task_reached,
                "strict_pose_reached": strict_pose_reached,
                "success_hold_count": self.success_hold_count,
                "success_hold_steps_required": self.success_hold_steps,
                "position_reached": position_reached,
                "timeout": timeout,
                "termination_reason": (
                    "obstacle_collision" if terminal_obstacle_collision
                    else "self_collision" if terminal_self_collision
                    else "success" if task_reached and terminated
                    else "timeout" if timeout else "running"
                ),
                "collision_assisted_reach": collision_assisted_reach,
                "safe_success": safe_success,
                "success": safe_success,
                "collision": any_collision,
                "obstacle_collision": raw_obstacle_collision,
                "self_collision": raw_self_collision,
                "environment_collision": raw_environment_collision,
                "joint_limit": joint_limit,
                "joint_limit_violation_max_rad": max_joint_limit_violation,
                "hard_failure": bool(
                    terminal_obstacle_collision or terminal_self_collision
                ),
                "obstacle_contact_seen_before": contact_seen_before,
                "obstacle_contact_seen": self.obstacle_contact_seen,
                "control_max_risk": max_risk,
                "control_min_distance": min_distance,
                "control_safety_violation": bool(min_distance < self.d_safe),
                "control_self_max_risk": self_max_risk,
                "control_self_min_distance": self_min_distance,
                "control_self_min_ttc": self_min_ttc,
                "control_self_max_approach": self_max_approach,
                "control_self_safety_violation": bool(
                    self_min_distance < self.d_self_safe
                ),
                "executed_substeps": executed_substeps,
                "qdot_cmd": np.zeros(6, dtype=np.float32),
                "qdot_measured": qdot_after.copy(),
                "smooth_velocity": smooth_velocity,
                "velocity_magnitude": velocity_magnitude,
                "rho_position": self.previous_rho_position,
                "next_rho_position": next_rho_position,
                "rho_orientation": self.previous_rho_orientation,
                "next_rho_orientation": next_rho_orientation,
            }
        )
        self.previous_rho_position = next_rho_position
        self.previous_rho_orientation = next_rho_orientation
        return observation, reward, cost, terminated, truncated, info
















    def _observe(self):
        if np.any(self.obstacle_dynamic_mask):
            # Include the exact policy-decision timestamp even when it falls
            # between regular 30 Hz tracker samples.
            self._record_dynamic_history()
        q, qdot = self.robot.joint_state()
        ee_position, ee_quaternion, ee_linear_velocity, ee_angular_velocity = (
            self.robot.end_effector_kinematics()
        )
        position_error, orientation_error = pose_error(
            self.goal_position, self.goal_quaternion, ee_position, ee_quaternion
        )
        point_errors = reference_point_errors(
            self.goal_position,
            self.goal_quaternion,
            ee_position,
            ee_quaternion,
            offset_m=self.reference_point_offset_m,
        )
        point_distances = np.linalg.norm(point_errors, axis=1).astype(np.float32)
        obstacle_geometries = self._obstacle_geometries()
        geometry = self._aggregate_obstacle_geometries(obstacle_geometries)
        self_geometry = self._self_geometry()
        obstacle_features = None
        if obstacle_geometries:
            if any(item.distances.shape != (6,) for item in obstacle_geometries):
                raise RuntimeError(
                    "obstacle encoding requires six ordered UR5 capsule clearances"
                )
            obstacle_features = np.stack(
                [
                    np.concatenate(
                        [
                            item.distances / OBSTACLE_CLEARANCE_SCALE_M,
                            item.nearest_direction,
                        ]
                    )
                    for item in obstacle_geometries
                ]
            ).astype(np.float32)
        observation = build_reference_observation(
            q=q,
            ee_position=ee_position,
            ee_quaternion=ee_quaternion,
            goal_position=self.goal_position,
            goal_quaternion=self.goal_quaternion,
            position_error=position_error,
            orientation_error=orientation_error,
            reference_point_offset_m=self.reference_point_offset_m,
            obstacle_features=obstacle_features,
            max_obstacles=self.max_obstacles,
        )
        info = {
            "goal_sampling_mode": (
                "fixed_orientation" if self.fixed_orientation_goal_enabled
                else "random_pose"
            ),
            "goal_position": self.goal_position.copy(),
            "goal_quaternion": self.goal_quaternion.copy(),
            "goal_joint_positions": self.goal_joint_positions.copy(),
            "goal_ik_validation": self.goal_ik_validation.copy(),
            "goal_sample_attempts": self.goal_sample_attempts_last,
            "ee_position": ee_position.copy(),
            "ee_quaternion": ee_quaternion.copy(),
            "ee_linear_velocity": ee_linear_velocity.copy(),
            "ee_angular_velocity": ee_angular_velocity.copy(),
            "goal_error_norm": float(np.linalg.norm(position_error)),
            "orientation_error_norm": float(np.linalg.norm(orientation_error)),
            "position_error_vector": position_error.copy(),
            "orientation_error_vector": orientation_error.copy(),
            "reference_point_errors": point_errors.copy(),
            "reference_point_distances": point_distances.copy(),
        }
        if np.any(self.obstacle_dynamic_mask):
            predicted_positions = []
            envelope_centers = []
            envelope_half_extents = []
            tracked_velocities = []
            for obstacle_index in np.flatnonzero(self.obstacle_dynamic_mask):
                center, half_extents, positions = (
                    self._predicted_obstacle_envelope(obstacle_index)
                )
                predicted_positions.append(positions)
                envelope_centers.append(center)
                envelope_half_extents.append(half_extents)
                tracked_velocities.append(
                    self._tracked_obstacle_velocity(obstacle_index)
                )
            info.update({
                "dynamic_prediction_mode": self.dynamic_prediction_mode,
                "dynamic_prediction_enabled": self.dynamic_prediction_enabled,
                "dynamic_prediction_times_s": self.dynamic_prediction_times.copy(),
                "dynamic_tracked_velocities": np.asarray(
                    tracked_velocities, dtype=np.float32
                ),
                "dynamic_predicted_positions": np.asarray(
                    predicted_positions, dtype=np.float32
                ),
                "dynamic_envelope_centers": np.asarray(
                    envelope_centers, dtype=np.float32
                ),
                "dynamic_envelope_half_extents": np.asarray(
                    envelope_half_extents, dtype=np.float32
                ),
            })
        return observation, info, geometry, self_geometry, position_error, orientation_error

    def pose_crossing_diagnostics(self) -> dict[str, object]:
        """Return expensive kinematic diagnostics for the current state.

        This is intentionally an on-demand evaluation hook: computing SVDs and
        self geometry on every training transition would slow the normal PPO
        loop.  Crossing-state evaluators call it at most once per episode.
        """
        q, _ = self.robot.joint_state_exact()
        lower = np.asarray(self.robot.joint_lower_limits, dtype=np.float64)
        upper = np.asarray(self.robot.joint_upper_limits, dtype=np.float64)
        margins = np.minimum(q - lower, upper - q)
        normalized_margins = margins / np.maximum(upper - lower, 1.0e-12)
        ee_position, ee_quaternion = self.robot.end_effector_pose()
        position_error, orientation_error = pose_error(
            self.goal_position, self.goal_quaternion,
            ee_position, ee_quaternion,
        )
        ee_jacobian = np.asarray(
            self.robot.end_effector_jacobian(), dtype=np.float64,
        )
        ee_singular_values = np.linalg.svd(
            ee_jacobian, compute_uv=False,
        )
        self_distance = float(self._self_geometry().distance_min)
        return {
            "step": int(self.step_count),
            "remaining_steps": int(max(0, self.horizon - self.step_count)),
            "remaining_time_fraction": float(max(
                0.0, (self.horizon - self.step_count) / self.horizon,
            )),
            "rho_position": float(np.linalg.norm(position_error)),
            "rho_orientation": float(np.linalg.norm(orientation_error)),
            "joint_positions": q.tolist(),
            "joint_limit_margin_min_rad": float(np.min(margins)),
            "joint_limit_margin_mean_rad": float(np.mean(margins)),
            "joint_limit_margin_min_fraction": float(
                np.min(normalized_margins)
            ),
            "ee_jacobian_sigma_min": float(np.min(ee_singular_values)),
            "ee_jacobian_sigma_max": float(np.max(ee_singular_values)),
            "ee_jacobian_condition_number": float(
                np.max(ee_singular_values)
                / max(float(np.min(ee_singular_values)), 1.0e-12)
            ),
            "ee_manipulability": float(np.prod(ee_singular_values)),
            "ee_log_manipulability": float(np.sum(np.log(
                np.maximum(ee_singular_values, 1.0e-12)
            ))),
            "self_distance": self_distance,
        }

    def _obstacle_geometry_for(
        self,
        capsules: list[Any],
        position: np.ndarray,
        velocity: np.ndarray,
        half_extents: np.ndarray | None = None,
        quaternion: np.ndarray | None = None,
        capsule_endpoint_velocities: np.ndarray | None = None,
    ) -> ThesisGeometry:
        obstacle_half_extents = (
            self.obstacle_half_extents if half_extents is None else half_extents
        )
        obstacle_quaternion = (
            self.obstacle_quaternion if quaternion is None else quaternion
        )
        return compute_thesis_geometry(
            capsules,
            position,
            velocity,
            self._closest_point_velocities(
                position,
                capsules,
                capsule_endpoint_velocities=capsule_endpoint_velocities,
            ),
            obstacle_half_extents=obstacle_half_extents,
            obstacle_rotation=quaternion_to_matrix(obstacle_quaternion),
            d_safe=self.d_safe,
        )

    def _minimum_obstacle_clearance_for(
        self,
        capsules: list[Any],
        position: np.ndarray,
        half_extents: np.ndarray | None = None,
        quaternion: np.ndarray | None = None,
    ) -> float:
        """Return the exact minimum capsule-to-current-OBB signed clearance."""
        obstacle_half_extents = (
            self.obstacle_half_extents if half_extents is None else half_extents
        )
        obstacle_quaternion = (
            self.obstacle_quaternion if quaternion is None else quaternion
        )
        rotation = quaternion_to_matrix(obstacle_quaternion)
        minimum = float("inf")
        for capsule in capsules:
            _, _, _, centerline_distance, _ = closest_points_segment_obb(
                capsule.start,
                capsule.end,
                position,
                obstacle_half_extents,
                rotation,
            )
            minimum = min(
                minimum, centerline_distance - float(capsule.radius)
            )
        return float(minimum)

    def _obstacle_geometries(self) -> list[ThesisGeometry]:
        if self.obstacle_id is None:
            return []
        capsules = self.robot.capsules()
        # Link state is shared by all obstacles at one policy timestamp.
        capsule_endpoint_velocities = self._capsule_endpoint_velocities(capsules)
        geometries: list[ThesisGeometry] = []
        for obstacle_index, (
            position,
            velocity,
            half_extents,
            quaternion,
        ) in enumerate(zip(
            self.obstacle_positions,
            self.obstacle_velocities,
            self.obstacle_half_extents_all,
            self.obstacle_quaternions,
            strict=True,
        )):
            if (
                self.obstacle_dynamic_mask[obstacle_index]
                and self.dynamic_prediction_enabled
            ):
                position, half_extents, _ = self._predicted_obstacle_envelope(
                    obstacle_index
                )
                velocity = self._tracked_obstacle_velocity(obstacle_index)
            geometries.append(self._obstacle_geometry_for(
                capsules,
                position,
                velocity,
                half_extents,
                quaternion,
                capsule_endpoint_velocities=capsule_endpoint_velocities,
            ))
        return geometries

    @staticmethod
    def _aggregate_obstacle_geometries(
        geometries: list[ThesisGeometry],
    ) -> ThesisGeometry | None:
        if not geometries:
            return None
        distances = np.stack([item.distances for item in geometries])
        nearest_obstacle = np.argmin(distances, axis=0)
        link_indices = np.arange(distances.shape[1])
        nearest_distances = distances[nearest_obstacle, link_indices]
        relative_vectors = np.stack(
            [item.relative_vectors for item in geometries]
        )[nearest_obstacle, link_indices]
        ttc = np.min(np.stack([item.ttc for item in geometries]), axis=0)
        approaches = np.max(
            np.stack([item.approach_velocities for item in geometries]), axis=0
        )
        risks = np.max(np.stack([item.risk_per_link for item in geometries]), axis=0)
        closest_geometry = min(geometries, key=lambda item: item.distance_min)
        return ThesisGeometry(
            relative_vectors=relative_vectors.astype(np.float32),
            distances=nearest_distances.astype(np.float32),
            nearest_direction=closest_geometry.nearest_direction.copy(),
            ttc=ttc.astype(np.float32),
            approach_velocities=approaches.astype(np.float32),
            risk_per_link=risks.astype(np.float32),
            risk_max=float(np.max(risks)),
            distance_min=float(np.min(nearest_distances)),
        )

    def _geometry(self) -> ThesisGeometry | None:
        return self._aggregate_obstacle_geometries(self._obstacle_geometries())

    def _self_geometry(self) -> SelfCollisionGeometry:
        pair_distances, pair_approaches = self._self_pair_geometry()
        return compute_self_collision_geometry(
            pair_distances,
            pair_approaches,
            self._self_pair_slots,
            d_safe=self.d_self_safe,
            query_distance=self.self_query_distance,
            ttc_max=self.self_ttc_max,
            approach_velocity_scale=self.self_approach_velocity_scale,
            ttc_tau=self.self_ttc_tau,
        )

    def _self_pair_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        pair_distances = np.full(
            len(self._self_pair_link_ids), self.self_query_distance, dtype=np.float32
        )
        pair_approaches = np.zeros(len(self._self_pair_link_ids), dtype=np.float32)
        # A whole-body self query returns contacts for adjacent and otherwise
        # ignored links, then Python discards most of them.  Asking Bullet for
        # only the 15 monitored pairs preserves the exact geometry contract
        # while avoiding that broad-phase/result-conversion overhead.
        for pair_index, (first, second) in enumerate(self._self_pair_link_ids):
            points = p.getClosestPoints(
                bodyA=self.robot_id,
                bodyB=self.robot_id,
                distance=self.self_query_distance,
                linkIndexA=first,
                linkIndexB=second,
                physicsClientId=self.client_id,
            )
            for point in points:
                point_first = np.asarray(point[5], dtype=np.float32)
                point_second = np.asarray(point[6], dtype=np.float32)
                if float(point[8]) >= pair_distances[pair_index]:
                    continue
                vector = point_second - point_first
                norm = float(np.linalg.norm(vector))
                if norm > 1e-8:
                    normal = vector / norm
                else:
                    normal = -np.asarray(point[7], dtype=np.float32)
                velocity_first = self._link_world_point_velocity(first, point_first)
                velocity_second = self._link_world_point_velocity(second, point_second)
                distance_rate = float(
                    np.dot(normal, velocity_second - velocity_first)
                )
                pair_distances[pair_index] = float(point[8])
                pair_approaches[pair_index] = max(-distance_rate, 0.0)
        return pair_distances, pair_approaches

    def _link_world_point_velocity(
        self, link_id: int, world_point: np.ndarray
    ) -> np.ndarray:
        state = p.getLinkState(
            self.robot_id,
            link_id,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        center_of_mass = np.asarray(state[0], dtype=np.float32)
        linear = np.asarray(state[6], dtype=np.float32)
        angular = np.asarray(state[7], dtype=np.float32)
        return linear + np.cross(angular, np.asarray(world_point) - center_of_mass)

    def _capsule_endpoint_velocities(
        self, capsules: list[Any],
    ) -> np.ndarray:
        """Query start/end world velocities once for every capsule."""
        link_states: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

        def point_velocity(link_name: str, local_offset: np.ndarray) -> np.ndarray:
            link_id = self.robot.capsule_model.link_name_to_id[link_name]
            if link_id < 0:
                return np.zeros(3, dtype=np.float32)
            if link_id not in link_states:
                state = p.getLinkState(
                    self.robot_id,
                    link_id,
                    computeLinkVelocity=True,
                    computeForwardKinematics=True,
                    physicsClientId=self.client_id,
                )
                link_states[link_id] = (
                    np.asarray(state[0], dtype=np.float32),
                    np.asarray(p.getMatrixFromQuaternion(state[5]), dtype=np.float32).reshape(3, 3),
                    np.asarray(state[6], dtype=np.float32),
                    np.asarray(state[7], dtype=np.float32),
                )
            _, rotation, linear, angular = link_states[link_id]
            radius = rotation @ np.asarray(local_offset, dtype=np.float32)
            return linear + np.cross(angular, radius)

        endpoint_velocities = []
        for capsule, spec in zip(capsules, self.robot.capsule_model.specs):
            start_velocity = point_velocity(
                spec.parent_link_name, spec.parent_offset
            )
            end_velocity = point_velocity(
                spec.child_link_name, spec.child_offset
            )
            endpoint_velocities.append((start_velocity, end_velocity))
        return np.asarray(endpoint_velocities, dtype=np.float32)

    def _closest_point_velocities(
        self,
        obstacle_position: np.ndarray,
        capsules: list[Any] | None = None,
        *,
        capsule_endpoint_velocities: np.ndarray | None = None,
    ) -> np.ndarray:
        if capsules is None:
            capsules = self.robot.capsules()
        if capsule_endpoint_velocities is None:
            capsule_endpoint_velocities = self._capsule_endpoint_velocities(
                capsules
            )
        endpoint_velocities = np.asarray(
            capsule_endpoint_velocities, dtype=np.float32
        )
        expected_shape = (len(capsules), 2, 3)
        if endpoint_velocities.shape != expected_shape:
            raise ValueError(
                "capsule endpoint velocities must have shape "
                f"{expected_shape}"
            )
        velocities = []
        for capsule, (start_velocity, end_velocity) in zip(
            capsules, endpoint_velocities, strict=True,
        ):
            _, fraction = closest_point_on_segment(
                obstacle_position, capsule.start, capsule.end
            )
            velocities.append(
                (1.0 - fraction) * start_velocity + fraction * end_velocity
            )
        return np.asarray(velocities, dtype=np.float32)

    def _resolve_collision_links(self) -> None:
        self._link_name_to_id = {}
        for joint_id in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.robot_id, joint_id, physicsClientId=self.client_id)
            self._link_name_to_id[info[12].decode("utf-8")] = joint_id
        collision_link_names = {
            link_name
            for spec in self.robot.capsule_model.specs
            for link_name in (spec.parent_link_name, spec.child_link_name)
        }
        self._collision_link_ids = {
            self._link_name_to_id[link_name]
            for link_name in collision_link_names
            if self._link_name_to_id[link_name] >= 0
        }
        self._mounting_plane_ignored_link_ids = {
            self._link_name_to_id[name]
            for name in self.mounting_plane_ignored_link_names
            if name in self._link_name_to_id
        }
        names = [
            "base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link",
            "wrist_1_link", "wrist_2_link", "wrist_3_link",
        ]
        self._self_link_order = [self._link_name_to_id[name] for name in names]
        self._self_link_rank = {
            link_id: index for index, link_id in enumerate(self._self_link_order)
        }
        self._self_pair_link_ids = []
        self._self_pair_slots = []
        for first_index, first_link in enumerate(self._self_link_order):
            for second_index in range(first_index + 2, len(self._self_link_order)):
                second_link = self._self_link_order[second_index]
                self._self_pair_link_ids.append(tuple(sorted((first_link, second_link))))
                self._self_pair_slots.append(
                    (first_index - 1, second_index - 1)
                )

    def _collision_events(self) -> dict[str, bool]:
        obstacle_collision = False
        self_collision = False
        environment_collision = False
        # One robot-wide query is equivalent to the former three filtered
        # queries and preserves per-physics-step collision detection at 240 Hz.
        contacts = p.getContactPoints(
            bodyA=self.robot_id,
            physicsClientId=self.client_id,
        )
        for point in contacts:
            if point[8] > 0.0:
                continue
            other_body = int(point[2])
            robot_link = int(point[3])
            other_link = int(point[4])
            if (
                other_body in self._obstacle_body_ids
                and robot_link in self._collision_link_ids
            ):
                obstacle_collision = True
            elif other_body == self.robot_id:
                if (
                    robot_link in self._self_link_rank
                    and other_link in self._self_link_rank
                    and abs(
                        self._self_link_rank[robot_link]
                        - self._self_link_rank[other_link]
                    ) > 1
                ):
                    self_collision = True
            elif (
                self.mounting_plane_id is not None
                and other_body == self.mounting_plane_id
                and robot_link in self._collision_link_ids
                and robot_link not in self._mounting_plane_ignored_link_ids
            ):
                environment_collision = True
            if (
                (not self._obstacle_body_ids or obstacle_collision)
                and self_collision
                and (self.mounting_plane_id is None or environment_collision)
            ):
                break
        return {
            "obstacle_collision": bool(obstacle_collision),
            "self_collision": bool(self_collision),
            "environment_collision": bool(environment_collision),
        }

    def _set_joint_state(self, q: np.ndarray, qdot: np.ndarray | None = None) -> None:
        velocity = np.zeros(6, dtype=np.float64) if qdot is None else np.asarray(qdot, dtype=np.float64)
        for joint_id, position, speed in zip(self.robot.joint_ids, q, velocity):
            p.resetJointState(
                self.robot_id,
                joint_id,
                float(position),
                targetVelocity=float(speed),
                physicsClientId=self.client_id,
            )

    def _contract_info(self) -> dict[str, Any]:
        return {
            "scene": self.contract.scene,
            "position_tolerance": self.position_tolerance,
            "orientation_tolerance": self.orientation_tolerance,
            "terminate_on_success": self.contract.terminate_on_success,
            "obstacle_enabled": self.obstacle_id is not None,
            "obstacle_count": int(len(self._obstacle_body_ids)),
            "dynamic_obstacle_count": int(np.count_nonzero(
                self.obstacle_dynamic_mask
            )),
            "static_obstacle_count": int(
                len(self._obstacle_body_ids)
                - np.count_nonzero(self.obstacle_dynamic_mask)
            ),
            "obstacle_layout": self.obstacle_layout,
            "obstacle_sampling_proposals": self.obstacle_sampling_proposals,
            "obstacle_sampling_time_s": self.obstacle_sampling_time_s,
            "scene_sampling_attempts": self.scene_sampling_attempts,
        }

    @staticmethod
    def _geometry_info(geometry: ThesisGeometry | None) -> dict[str, Any]:
        if geometry is None:
            return {"risk_body": np.zeros(6, dtype=np.float32), "d_min": 0.80, "risk_global": 0.0}
        return {
            "risk_body": geometry.risk_per_link.copy(),
            "distance_body": geometry.distances.copy(),
            "ttc_body": geometry.ttc.copy(),
            "d_min": geometry.distance_min,
            "risk_global": geometry.risk_max,
        }

    @staticmethod
    def _self_geometry_info(geometry: SelfCollisionGeometry) -> dict[str, Any]:
        return {
            "self_distance_body": geometry.distances.copy(),
            "self_ttc_body": geometry.ttc.copy(),
            "self_approach_body": geometry.approach_velocities.copy(),
            "self_risk_body": geometry.risk_per_link.copy(),
            "self_d_min": geometry.distance_min,
            "self_risk_global": geometry.risk_max,
            "self_ttc_min": geometry.ttc_min,
            "self_approach_max": geometry.approach_max,
            "self_closest_pair": geometry.closest_pair,
        }

    def close(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)
