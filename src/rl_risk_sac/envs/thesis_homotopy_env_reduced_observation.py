from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

from rl_risk_sac.envs.thesis_goal_sampling import GoalSamplingMixin
from rl_risk_sac.envs.joint_motion_controller import BoundedJointMotionController
from rl_risk_sac.envs.thesis_dynamic_obstacles import DynamicObstacleMixin
from rl_risk_sac.envs.thesis_static_obstacles import (
    ObstacleSamplingError,
    StaticObstacleMixin,
    _sample_physical_layout_plan,
)
from rl_risk_sac.envs.thesis_reward import compute_step_reward
from rl_risk_sac.prediction import LSTMObstaclePredictor, load_lstm_predictor
from rl_risk_sac.robots.pybullet_robot import PyBulletRobot
from rl_risk_sac.tasks.reference_reaching import (
    BASE_CLEARANCE_DELTA_FEATURE_DIM,
    OBSTACLE_CLEARANCE_SCALE_M,
    PLANE_CLEARANCE_SCALE_M,
    SelfCollisionGeometry,
    ThesisGeometry,
    build_reference_observation,
    closest_points_segment_obb,
    compute_self_collision_geometry,
    compute_thesis_geometry,
    pose_error,
    quaternion_to_matrix,
    obstacle_feature_dim,
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
        # Visualization-only wall-clock pacing.  Normal training/evaluation
        # leaves this at zero, so it cannot change simulation semantics or
        # throughput.  Human replay may set it to physics_dt / playback_speed
        # to make all 240 Hz substeps visible in the PyBullet GUI.
        self._visualization_substep_delay_s = 0.0
        self._visualization_input_callback: Callable[[], None] | None = None
        self._evaluation_counterfactual_variant: str | None = None
        self._evaluation_forced_goal: tuple[
            np.ndarray, np.ndarray, np.ndarray
        ] | None = None
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
        if not np.isclose(self.substeps * self.physics_dt, self.control_dt):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
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
        # Explicit absolute joint-position bounds.  These are intentionally
        # independent from the Actor's per-decision joint-delta range.
        self.configured_joint_lower_limits = np.asarray(
            thesis.get("joint_position_lower_limits_rad", [-2.0 * np.pi] * 6),
            dtype=np.float64,
        )
        self.configured_joint_upper_limits = np.asarray(
            thesis.get("joint_position_upper_limits_rad", [2.0 * np.pi] * 6),
            dtype=np.float64,
        )
        if (
            self.configured_joint_lower_limits.shape != (6,)
            or self.configured_joint_upper_limits.shape != (6,)
            or not np.isfinite(self.configured_joint_lower_limits).all()
            or not np.isfinite(self.configured_joint_upper_limits).all()
            or np.any(
                self.configured_joint_upper_limits
                <= self.configured_joint_lower_limits
            )
        ):
            raise ValueError("invalid configured joint-position limits")
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
        motion = thesis["joint_motion_controller"]
        self.bounded_joint_motion_enabled = bool(motion.get("enabled", True))
        self.real_control_frequency_hz = float(motion["real_control_frequency_hz"])
        self.real_control_dt = 1.0 / self.real_control_frequency_hz
        self.controller_updates_per_policy_step = int(round(
            self.control_dt * self.real_control_frequency_hz
        ))
        smoothing = motion["trajectory_smoothing"]
        self.motion_controller = (
            BoundedJointMotionController(
                joint_velocity_limits=np.asarray(
                    motion["joint_velocity_limits_rad_s"], dtype=np.float64
                ),
                joint_acceleration_limits=np.asarray(
                    motion["joint_acceleration_limits_rad_s2"], dtype=np.float64
                ),
                position_gain=float(motion["position_gain_s_inv"]),
                position_tolerance=float(motion["position_tolerance_rad"]),
                trajectory_smoothing_enabled=bool(smoothing["enabled"]),
                butterworth_cutoff_hz=float(smoothing["butterworth_cutoff_hz"]),
                quintic_blend_duration_s=float(
                    smoothing["quintic_blend_duration_s"]
                ),
            )
            if self.bounded_joint_motion_enabled
            else None
        )
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
            # Use the checkpoint's exact timestamps in the swept-volume
            # metadata.  A legacy horizon may extend at most 5% beyond the
            # 0.4 s action period and is therefore conservative.
            self.dynamic_prediction_times = (
                self.dynamic_predictor.horizon_times.copy()
            )
        self.blocking_obstacle_mode = bool(thesis["blocking_obstacle_mode"])
        self.mesh_clearance_diagnostics_enabled = False
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
            "reference_obstacle_clearance_power": float(
                reward["reference_obstacle_clearance_power"]
            ),
            "reference_obstacle_collision_penalty": float(
                reward["reference_obstacle_collision_penalty"]
            ),
            # Old checkpoint-embedded configurations predate the separate
            # first-contact cost. Falling back to the per-step value preserves
            # their historical evaluation semantics.
            "reference_obstacle_collision_entry_penalty": float(
                reward.get(
                    "reference_obstacle_collision_entry_penalty",
                    reward["reference_obstacle_collision_penalty"],
                )
            ),
            "reference_self_collision_penalty": float(
                reward["reference_self_collision_penalty"]
            ),
            "reference_environment_collision_penalty": float(
                reward["reference_environment_collision_penalty"]
            ),
            "reference_environment_collision_entry_penalty": float(
                reward.get(
                    "reference_environment_collision_entry_penalty",
                    reward["reference_environment_collision_penalty"],
                )
            ),
            "reference_tcp_path_length_weight": float(
                reward.get("reference_tcp_path_length_weight", 0.0)
            ),
            "reference_base_clearance_decrease_start": float(
                reward.get("reference_base_clearance_decrease_start_m", 0.001)
            ),
            "reference_base_clearance_decrease_full": float(
                reward.get("reference_base_clearance_decrease_full_m", 0.010)
            ),
        }
        for name, value in self.reward_parameters.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.reward_parameters["reference_obstacle_clearance_power"] <= 0.0:
            raise ValueError("reference_obstacle_clearance_power must be positive")
        if not (
            self.reward_parameters["reference_base_clearance_decrease_start"]
            < self.reward_parameters["reference_base_clearance_decrease_full"]
        ):
            raise ValueError(
                "Base clearance-decrease gate must satisfy start < full"
            )
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
        self.environment_contact_seen = False
        if self.motion_controller is not None:
            self.motion_controller.reset()
        self.previous_rho_position = 0.0
        self.previous_rho_orientation = 0.0
        observation_config = thesis["observation"]
        self.reference_point_offset_m = float(
            observation_config["reference_point_offset_m"]
        )
        obstacle_set_config = observation_config["obstacle_set"]
        self.max_obstacles = int(obstacle_set_config["max_obstacles"])
        self.obstacle_feature_mode = str(
            obstacle_set_config.get("feature_mode", "clearance_direction")
        )
        self.obstacle_feature_dim = obstacle_feature_dim(
            self.obstacle_feature_mode
        )
        self.route_size_scale = float(
            obstacle_set_config.get("route_size_scale_m", 0.25)
        )
        self.route_position_scale = float(
            obstacle_set_config.get("route_position_scale_m", 0.75)
        )
        if (
            not np.isfinite(self.route_size_scale)
            or not np.isfinite(self.route_position_scale)
            or self.route_size_scale <= 0.0
            or self.route_position_scale <= 0.0
        ):
            raise ValueError("route feature scales must be finite and positive")
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
            "legacy_blocking", "workspace_random", "path_offset", "mixed",
            "structured", "grounded_path", "grounded_random",
            "suspended_path", "suspended_random", "physical_mixed",
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
        physical_mixture = thesis["physical_obstacle_mixture"]
        self.physical_challenging_episode_probability = float(
            physical_mixture["challenging_episode_probability"]
        )
        self.physical_grounded_probability = float(
            physical_mixture["grounded_probability"]
        )
        self.physical_extra_path_probability = float(
            physical_mixture["extra_path_probability"]
        )
        self.physical_max_path_obstacles = int(
            physical_mixture["max_path_obstacles"]
        )
        self.physical_suspended_bottom_min = float(
            physical_mixture["suspended_bottom_min_m"]
        )
        self.physical_obstacle_top_max = float(
            physical_mixture["obstacle_top_max_m"]
        )
        self.physical_path_clearance_max = float(
            physical_mixture["path_clearance_max_m"]
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
        self.obstacle_layouts: tuple[str, ...] = ()
        # Reset selects one requested layout per episode. It remains fixed
        # while infeasible goals are rejected, so accepted episodes preserve
        # the configured layout mixture instead of biasing toward layouts
        # that are easier to sample.
        self._requested_obstacle_layout: str | None = None
        self._requested_static_obstacle_count: int | None = None
        self._requested_physical_layout_plan: tuple[str, ...] = ()
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

        observation_dim = reference_ppo_observation_dim(
            self.max_obstacles, self.obstacle_feature_dim
        )
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
        self,
        scene: str,
        *,
        terminate_on_success: bool = True,
        counterfactual_variant: str | None = None,
        forced_goal_position: np.ndarray | None = None,
        forced_goal_quaternion: np.ndarray | None = None,
        forced_goal_joint_positions: np.ndarray | None = None,
    ) -> None:
        if scene not in {"none", "static", "dynamic", "hybrid"}:
            raise ValueError(
                "scene must be one of: none, static, dynamic, hybrid"
            )
        self.contract = EpisodeContract(
            scene=scene,
            terminate_on_success=bool(terminate_on_success),
        )
        variants = {"low_wide", "tall_narrow", "offset_left", "offset_right"}
        if counterfactual_variant is not None:
            if scene != "static":
                raise ValueError("counterfactual obstacle evaluation requires static scene")
            if counterfactual_variant not in variants:
                raise ValueError(
                    f"unknown counterfactual variant {counterfactual_variant!r}"
                )
        goal_values = (
            forced_goal_position,
            forced_goal_quaternion,
            forced_goal_joint_positions,
        )
        if any(value is not None for value in goal_values) and not all(
            value is not None for value in goal_values
        ):
            raise ValueError("forced counterfactual goal requires position, quaternion, and joints")
        self._evaluation_counterfactual_variant = counterfactual_variant
        self._evaluation_forced_goal = None
        if forced_goal_position is not None:
            position = np.asarray(forced_goal_position, dtype=np.float32)
            quaternion = np.asarray(forced_goal_quaternion, dtype=np.float32)
            joints = np.asarray(forced_goal_joint_positions, dtype=np.float32)
            if position.shape != (3,) or quaternion.shape != (4,) or joints.shape != (6,):
                raise ValueError("invalid forced counterfactual goal dimensions")
            if not all(np.isfinite(value).all() for value in (position, quaternion, joints)):
                raise ValueError("forced counterfactual goal must be finite")
            self._evaluation_forced_goal = (
                position.copy(), quaternion.copy(), joints.copy()
            )

    def configure_human_playback(
        self,
        playback_speed: float,
        *,
        input_callback: Callable[[], None] | None = None,
    ) -> None:
        """Pace GUI physics substeps without changing simulated time."""
        if self.render_mode != "human":
            raise RuntimeError("human playback requires render_mode='human'")
        speed = float(playback_speed)
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError("playback_speed must be finite and positive")
        self._visualization_substep_delay_s = self.physics_dt / speed
        self._visualization_input_callback = input_callback

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        total_obstacle_proposals = 0
        total_obstacle_sampling_time = 0.0
        last_sampling_error: ObstacleSamplingError | None = None
        self._requested_obstacle_layout = None
        self._requested_static_obstacle_count = None
        self._requested_physical_layout_plan = ()
        if (
            self.contract.scene in {"static", "hybrid"}
            and self.blocking_obstacle_mode
        ):
            self._requested_obstacle_layout = str(self.rng.choice(
                self.static_obstacle_layouts,
                p=self.static_obstacle_layout_probabilities,
            ))
            if self._requested_obstacle_layout == "physical_mixed":
                if self.contract.scene == "static":
                    self._requested_static_obstacle_count = int(self.rng.choice(
                        self.static_obstacle_counts,
                        p=self.static_obstacle_count_probabilities,
                    ))
                else:
                    self._requested_static_obstacle_count = (
                        self.hybrid_static_obstacle_count
                    )
                self._requested_physical_layout_plan = (
                    _sample_physical_layout_plan(
                        self._requested_static_obstacle_count,
                        challenging_probability=(
                            self.physical_challenging_episode_probability
                        ),
                        grounded_probability=self.physical_grounded_probability,
                        extra_path_probability=self.physical_extra_path_probability,
                        max_path_obstacles=self.physical_max_path_obstacles,
                        rng=self.rng,
                    )
                )
        scene_attempt_limit = (
            1 if self._evaluation_forced_goal is not None
            else self.scene_sample_max_attempts
        )
        for scene_sample_attempt in range(1, scene_attempt_limit + 1):
            self._disable_obstacle()
            self._reset_fixed_initial_state()
            self.episode_start_ee_position = np.asarray(
                self.robot.end_effector_pose()[0], dtype=np.float32
            )
            # A real controller starts a new episode from the measured, static
            # initial state.  Carrying the previous episode's final velocity
            # into the acceleration limiter would create a command that cannot
            # occur after the robot has been reset.
            if self.motion_controller is not None:
                _, initial_velocity = self.robot.joint_state()
                self.motion_controller.reset(initial_velocity)
            if self._evaluation_forced_goal is None:
                self._sample_fixed_orientation_goal_pose()
            else:
                position, quaternion, joints = self._evaluation_forced_goal
                self.goal_position = position.copy()
                self.goal_quaternion = quaternion.copy()
                self.goal_joint_positions = joints.copy()
                self.goal_ik_validation = {
                    "q_ik": joints.astype(np.float64),
                    "reachable": True,
                    "finite": True,
                    "within_limits": True,
                    "collision_free": True,
                    "position_error_m": 0.0,
                    "orientation_error_rad": 0.0,
                }
                self.goal_sample_attempts_last = 0
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
            self._requested_obstacle_layout = None
            self._requested_static_obstacle_count = None
            self._requested_physical_layout_plan = ()
            raise ObstacleSamplingError(
                "failed to sample a feasible goal and obstacle scene after "
                f"{scene_attempt_limit} goal attempts; "
                f"last error: {last_sampling_error}"
            ) from last_sampling_error
        self._requested_obstacle_layout = None
        self._requested_static_obstacle_count = None
        self._requested_physical_layout_plan = ()
        self.step_count = 0
        self.success_hold_count = 0
        self.obstacle_contact_seen = False
        self.environment_contact_seen = False
        self.tcp_path_length_m = 0.0
        self.joint_path_length_rad = 0.0
        self.direct_tcp_distance_m = float(np.linalg.norm(
            self.goal_position - self.episode_start_ee_position
        ))
        self.route_lateral_min_m = 0.0
        self.route_lateral_max_m = 0.0
        self.route_vertical_deviation_min_m = 0.0
        self.route_vertical_deviation_max_m = 0.0
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

    def base_clearance_changes(self, base_action: np.ndarray) -> np.ndarray:
        """Predict five moving-link safety-clearance changes from Base.

        The query is counterfactual.  With the bounded CB3.1 controller it
        evaluates the robot at the joint position reachable after one policy
        period under the same 125 Hz velocity/acceleration contract used by
        :meth:`step`; it must not jump directly to an unreachable ``q +
        action`` target.  The legacy unbounded route still evaluates its full
        position target.  Each link uses its minimum signed clearance over
        external obstacles and the mounting plane before and after the Base
        command.  Values use the same 0.20 m scale as the set observations.

        PyBullet state is restored exactly before returning, so this query has
        no effect on the subsequent physical transition.
        """
        action = np.asarray(base_action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("base_action must contain six finite joint increments")
        current_geometries = self._safety_geometries()
        if not current_geometries:
            return np.zeros(BASE_CLEARANCE_DELTA_FEATURE_DIM, dtype=np.float32)
        current = self._aggregate_obstacle_geometries(current_geometries)
        if current is None or current.distances.shape != (6,):
            raise RuntimeError("invalid current safety clearances")

        q, qdot = self.robot.joint_state_exact()
        command_lower, command_upper = self.effective_joint_position_limits()
        target = np.clip(q + action, command_lower, command_upper)
        if self.bounded_joint_motion_enabled:
            if self.motion_controller is None:
                raise RuntimeError("bounded joint controller was not initialized")
            predicted_q = self.motion_controller.predict_reachable_position(
                q,
                target,
                self.real_control_dt,
                self.controller_updates_per_policy_step,
            )
            predicted_q = np.clip(predicted_q, command_lower, command_upper)
        else:
            predicted_q = target
        state_id = p.saveState(physicsClientId=self.client_id)
        try:
            for joint_id, position, velocity in zip(
                self.robot.joint_ids, predicted_q, qdot, strict=True
            ):
                p.resetJointState(
                    self.robot_id,
                    joint_id,
                    float(position),
                    targetVelocity=float(velocity),
                    physicsClientId=self.client_id,
                )
            predicted_geometries = self._safety_geometries()
            predicted = self._aggregate_obstacle_geometries(predicted_geometries)
            if predicted is None or predicted.distances.shape != (6,):
                raise RuntimeError("invalid predicted safety clearances")
            changes = (
                np.asarray(predicted.distances, dtype=np.float64)
                - np.asarray(current.distances, dtype=np.float64)
            ) / OBSTACLE_CLEARANCE_SCALE_M
        finally:
            p.restoreState(stateId=state_id, physicsClientId=self.client_id)
            p.removeState(stateUniqueId=state_id, physicsClientId=self.client_id)
        if not np.isfinite(changes).all():
            raise RuntimeError("non-finite Base clearance-change prediction")
        # The first ordered capsule is the shoulder.  Its rotation is about
        # the vertical base axis, so its counterfactual clearance change is
        # not a useful avoidance-control feature.  Keep shoulder clearance in
        # the obstacle set, but omit its Base-induced delta here.
        result = changes[1:].astype(np.float32)
        if result.shape != (BASE_CLEARANCE_DELTA_FEATURE_DIM,):
            raise RuntimeError("invalid moving-link clearance-change vector")
        return result

    def _plane_link_clearances(self) -> np.ndarray:
        """Return six ordered signed link clearances to the mounting plane.

        Distances use Bullet's collision geometry rather than capsule AABBs,
        so a rotated link does not produce a false plane penetration.  The
        fixed base links are intentionally excluded, matching the physical
        collision contract.  Values are raw metres; the observation builder
        applies the shared 0.20 m safety-object scale.
        """
        if self.mounting_plane_id is None:
            return np.full(6, PLANE_CLEARANCE_SCALE_M, dtype=np.float32)
        names = (
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        )
        # Shoulder rotation is about the vertical mounting axis and its low
        # fixed geometry is not an avoidance degree of freedom.  Preserve the
        # six-link row contract with a safe sentinel in slot zero.
        distances: list[float] = [PLANE_CLEARANCE_SCALE_M]
        query_distance = PLANE_CLEARANCE_SCALE_M
        for name in names:
            link_id = self._link_name_to_id[name]
            points = p.getClosestPoints(
                bodyA=self.robot_id,
                bodyB=self.mounting_plane_id,
                distance=query_distance,
                linkIndexA=link_id,
                physicsClientId=self.client_id,
            )
            distance = min((float(point[8]) for point in points), default=query_distance)
            distances.append(distance)
        result = np.asarray(distances, dtype=np.float32)
        if (
            result.shape != (6,)
            or not np.isfinite(result).all()
        ):
            raise RuntimeError("invalid link-to-plane clearance vector")
        return result

    def _plane_geometry(self) -> ThesisGeometry | None:
        """Represent the rigid mounting plane as one obstacle-set element."""
        if self.mounting_plane_id is None:
            return None
        distances = self._plane_link_clearances()
        direction = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        relative_vectors = np.repeat(direction[None, :], 6, axis=0)
        ttc = np.full(6, self.self_ttc_max, dtype=np.float32)
        approach = np.zeros(6, dtype=np.float32)
        risk = np.clip(
            (self.d_safe - distances) / self.d_safe, 0.0, 1.0
        ).astype(np.float32)
        return ThesisGeometry(
            relative_vectors=relative_vectors,
            distances=distances,
            nearest_direction=direction,
            ttc=ttc,
            approach_velocities=approach,
            risk_per_link=risk,
            risk_max=float(np.max(risk)),
            distance_min=float(np.min(distances)),
        )

    def _safety_geometries(self) -> list[ThesisGeometry]:
        """Return external obstacles and the plane under one geometry contract."""
        geometries = self._obstacle_geometries()
        plane = self._plane_geometry()
        if plane is not None:
            geometries.append(plane)
        if len(geometries) > self.max_obstacles:
            raise RuntimeError(
                "unified safety set exceeds observation obstacle capacity"
            )
        return geometries

    def residual_safety_context(self, base_action: np.ndarray) -> np.ndarray:
        """Return Base-induced minimum-clearance changes for the safety set."""
        context = self.base_clearance_changes(base_action).astype(np.float32)
        if (
            context.shape != (BASE_CLEARANCE_DELTA_FEATURE_DIM,)
            or not np.isfinite(context).all()
        ):
            raise RuntimeError("invalid Residual safety context")
        return context

    def effective_joint_position_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the one authoritative command range for all six joints.

        The configured CB3.1 limits may be narrowed to match an installation's
        PolyScope safety settings, but can never extend beyond the robot URDF.
        Execution, prediction, IK filtering and diagnostics all use this same
        intersection so a policy cannot be silently blocked by a second range.
        """
        lower = np.maximum(
            self.robot.joint_lower_limits.astype(np.float64),
            self.configured_joint_lower_limits,
        )
        upper = np.minimum(
            self.robot.joint_upper_limits.astype(np.float64),
            self.configured_joint_upper_limits,
        )
        if np.any(upper <= lower):
            raise RuntimeError("configured and URDF joint-position limits do not overlap")
        return lower, upper

    def step(
        self,
        action: np.ndarray,
        *,
        base_clearance_changes: np.ndarray | None = None,
    ):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("action must contain six finite joint-angle increments")
        clearance_gates = None
        if base_clearance_changes is not None:
            changes = np.asarray(base_clearance_changes, dtype=np.float64)
            if changes.shape != (BASE_CLEARANCE_DELTA_FEATURE_DIM,) or not np.isfinite(changes).all():
                raise ValueError(
                    "base_clearance_changes must contain five finite values"
                )
            gate_config = self.reward_parameters
            decrease_start = gate_config[
                "reference_base_clearance_decrease_start"
            ]
            decrease_full = gate_config[
                "reference_base_clearance_decrease_full"
            ]
            # Context values use the shared 0.20 m observation scale. Convert
            # the predicted decrease back to metres before applying the gate.
            decrease_m = np.maximum(
                -changes * OBSTACLE_CLEARANCE_SCALE_M, 0.0
            )
            phase = np.clip(
                (decrease_m - decrease_start)
                / max(decrease_full - decrease_start, 1.0e-12),
                0.0,
                1.0,
            )
            moving_link_gates = phase * phase * (3.0 - 2.0 * phase)
            # The shoulder has no useful avoidance degree of freedom and has
            # no corresponding Base-delta input. Never let its fixed absolute
            # clearance create a perpetual proximity cost.
            clearance_gates = np.concatenate((
                np.zeros(1, dtype=np.float64), moving_link_gates,
            ))
        q_before, qdot_before = self.robot.joint_state()
        ee_position_before = np.asarray(
            self.robot.end_effector_pose()[0], dtype=np.float32
        )
        command_lower, command_upper = self.effective_joint_position_limits()
        lower, upper = command_lower, command_upper
        requested_target_positions = q_before + action
        target_positions = np.clip(
            requested_target_positions, command_lower, command_upper
        )
        joint_position_clip = np.abs(
            requested_target_positions - target_positions
        )
        projection_info = {
            "projection_applied": False,
            "projection_iterations": 0,
            "projection_max_violation_before": 0.0,
            "projection_max_violation_after": 0.0,
        }
        contact_seen_before = self.obstacle_contact_seen
        environment_contact_seen_before = self.environment_contact_seen
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

        peak_joint_speed = 0.0
        peak_joint_acceleration = 0.0
        peak_joint_jerk = 0.0
        peak_commanded_joint_jerk = 0.0
        joint_jerk_square_sum = 0.0
        joint_jerk_sample_count = 0
        commanded_joint_jerk_square_sum = 0.0
        commanded_joint_jerk_sample_count = 0
        peak_tcp_linear_speed = 0.0
        peak_tcp_angular_speed = 0.0
        velocity_limited_substeps = 0
        acceleration_limited_substeps = 0
        tcp_limited_substeps = 0
        controller_updates = 0
        last_measured_velocity = qdot_before.astype(np.float64)
        last_measured_acceleration = np.zeros(6, dtype=np.float64)
        last_commanded_velocity = np.zeros(6, dtype=np.float32)
        if self.bounded_joint_motion_enabled:
            if self.motion_controller is None:
                raise RuntimeError("bounded joint controller was not initialized")
            self.motion_controller.begin_policy_step(
                q_before, target_positions, self.control_dt
            )
        if not self.bounded_joint_motion_enabled:
            # Exact execution route used by the archived S0/S1 reference-PPO
            # experiments: a joint-position target persists for all physics
            # substeps in one policy period.
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
            if self.bounded_joint_motion_enabled:
                if self.motion_controller is None:
                    raise RuntimeError("bounded joint controller was not initialized")
                # Bullet integrates at 240 Hz, whereas CB3.1 accepts commands
                # at 125 Hz.  Update at the first physics instant at or after
                # each real-controller tick, then hold the command until the
                # next tick.  For a 0.4 s policy period this produces exactly
                # 50 controller updates across 96 physics substeps.
                physics_time = substep_index * self.physics_dt
                next_control_time = controller_updates * self.real_control_dt
                if physics_time + 1.0e-12 >= next_control_time:
                    q_control, _ = self.robot.joint_state()
                    motion_command = self.motion_controller.command(
                        q_control,
                        target_positions,
                        self.real_control_dt,
                    )
                    last_commanded_velocity = motion_command.velocity
                    peak_commanded_joint_jerk = max(
                        peak_commanded_joint_jerk,
                        float(np.max(np.abs(motion_command.jerk))),
                    )
                    commanded_joint_jerk_square_sum += float(
                        np.sum(np.square(motion_command.jerk, dtype=np.float64))
                    )
                    commanded_joint_jerk_sample_count += int(
                        motion_command.jerk.size
                    )
                    controller_updates += 1
                    velocity_limited_substeps += int(
                        motion_command.joint_velocity_limited
                    )
                    acceleration_limited_substeps += int(
                        motion_command.joint_acceleration_limited
                    )
                    tcp_limited_substeps += int(
                        motion_command.tcp_velocity_limited
                    )
                    p.setJointMotorControlArray(
                        self.robot_id,
                        self.robot.joint_ids,
                        p.VELOCITY_CONTROL,
                        targetVelocities=motion_command.velocity.tolist(),
                        forces=self.motor_forces.tolist(),
                        physicsClientId=self.client_id,
                    )
            # Move kinematic obstacles to the same physical timestamp before
            # Bullet resolves robot contacts for this substep. Moving them
            # afterwards would leave contact queries one 240 Hz step behind.
            self._advance_obstacle()
            p.stepSimulation(physicsClientId=self.client_id)
            if self._visualization_input_callback is not None:
                self._visualization_input_callback()
            if self._visualization_substep_delay_s > 0.0:
                time.sleep(self._visualization_substep_delay_s)
            executed_substeps += 1
            _, measured_velocity = self.robot.joint_state()
            measured_acceleration = (
                measured_velocity.astype(np.float64) - last_measured_velocity
            ) / self.physics_dt
            measured_jerk = (
                measured_acceleration - last_measured_acceleration
            ) / self.physics_dt
            last_measured_velocity = measured_velocity.astype(np.float64)
            last_measured_acceleration = measured_acceleration
            _, _, tcp_linear_velocity, tcp_angular_velocity = (
                self.robot.end_effector_kinematics()
            )
            peak_joint_speed = max(
                peak_joint_speed, float(np.max(np.abs(measured_velocity)))
            )
            peak_joint_acceleration = max(
                peak_joint_acceleration,
                float(np.max(np.abs(measured_acceleration))),
            )
            peak_joint_jerk = max(
                peak_joint_jerk, float(np.max(np.abs(measured_jerk)))
            )
            joint_jerk_square_sum += float(np.sum(np.square(measured_jerk)))
            joint_jerk_sample_count += int(measured_jerk.size)
            peak_tcp_linear_speed = max(
                peak_tcp_linear_speed, float(np.linalg.norm(tcp_linear_velocity))
            )
            peak_tcp_angular_speed = max(
                peak_tcp_angular_speed, float(np.linalg.norm(tcp_angular_velocity))
            )
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
        self.environment_contact_seen = (
            self.environment_contact_seen or environment_collision
        )
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
        q_after, qdot_after = self.robot.joint_state()
        ee_position_after = np.asarray(info["ee_position"], dtype=np.float32)
        tcp_path_increment_m = float(np.linalg.norm(
            ee_position_after - ee_position_before
        ))
        joint_path_increment_rad = float(np.linalg.norm(q_after - q_before))
        self.tcp_path_length_m += tcp_path_increment_m
        self.joint_path_length_rad += joint_path_increment_rad
        tcp_path_efficiency_ratio = self.tcp_path_length_m / max(
            self.direct_tcp_distance_m, 1.0e-6
        )
        route_lateral_m, route_vertical_deviation_m = (
            self._tcp_route_deviation(ee_position_after)
        )
        self.route_lateral_min_m = min(
            self.route_lateral_min_m, route_lateral_m
        )
        self.route_lateral_max_m = max(
            self.route_lateral_max_m, route_lateral_m
        )
        self.route_vertical_deviation_min_m = min(
            self.route_vertical_deviation_min_m, route_vertical_deviation_m
        )
        self.route_vertical_deviation_max_m = max(
            self.route_vertical_deviation_max_m, route_vertical_deviation_m
        )
        route_candidates = {
            "left": self.route_lateral_max_m,
            "right": -self.route_lateral_min_m,
            "over": self.route_vertical_deviation_max_m,
            "under": -self.route_vertical_deviation_min_m,
        }
        selected_route = max(route_candidates, key=route_candidates.get)
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
            obstacle_first_contact=bool(
                raw_obstacle_collision and not contact_seen_before
            ),
            self_collision=raw_self_collision,
            environment_collision=raw_environment_collision,
            environment_first_contact=bool(
                raw_environment_collision
                and not environment_contact_seen_before
            ),
            obstacle_clearances=(
                () if endpoint_geometry is None
                else tuple(float(value) for value in endpoint_geometry.distances)
            ),
            obstacle_clearance_gates=(
                None
                if endpoint_geometry is None or clearance_gates is None
                else tuple(float(value) for value in clearance_gates)
            ),
            parameters=self.reward_parameters,
            obstacle_risk=max_risk,
            self_risk=self_max_risk,
            obstacle_distance=min_distance,
            self_distance=self_min_distance,
            obstacle_safe_distance=self.d_safe,
            self_safe_distance=self.d_self_safe,
            tcp_path_increment_m=tcp_path_increment_m,
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
                "requested_joint_target": requested_target_positions.copy(),
                "policy_joint_velocity": last_commanded_velocity.copy(),
                "commanded_joint_delta": (q_after - q_before).copy(),
                "target_joint_positions": target_positions.copy(),
                "scaled_policy_joint_velocity": last_commanded_velocity.copy(),
                "commanded_joint_velocity": last_commanded_velocity.copy(),
                "joint_velocity": qdot_after.copy(),
                "joint_speed_peak_rad_s": peak_joint_speed,
                "joint_acceleration_peak_rad_s2": peak_joint_acceleration,
                "joint_jerk_peak_rad_s3": peak_joint_jerk,
                "commanded_joint_jerk_peak_rad_s3": peak_commanded_joint_jerk,
                "joint_jerk_rms_rad_s3": float(np.sqrt(
                    joint_jerk_square_sum / max(1, joint_jerk_sample_count)
                )),
                "commanded_joint_jerk_rms_rad_s3": float(np.sqrt(
                    commanded_joint_jerk_square_sum
                    / max(1, commanded_joint_jerk_sample_count)
                )),
                "tcp_linear_speed_peak_m_s": peak_tcp_linear_speed,
                "tcp_angular_speed_peak_rad_s": peak_tcp_angular_speed,
                "controller_updates": controller_updates,
                "controller_updates_expected": self.controller_updates_per_policy_step,
                "real_control_frequency_hz": self.real_control_frequency_hz,
                "physics_frequency_hz": 1.0 / self.physics_dt,
                "joint_velocity_limiter_fraction": (
                    velocity_limited_substeps / max(1, controller_updates)
                ),
                "joint_acceleration_limiter_fraction": (
                    acceleration_limited_substeps / max(1, controller_updates)
                ),
                "tcp_velocity_limiter_fraction": (
                    tcp_limited_substeps / max(1, controller_updates)
                ),
                "joint_position_clip_fraction": float(
                    np.mean(joint_position_clip > 1.0e-12)
                ),
                "joint_position_clip_max_rad": float(
                    np.max(joint_position_clip)
                ),
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
                "qdot_cmd": last_commanded_velocity.copy(),
                "qdot_measured": qdot_after.copy(),
                "smooth_velocity": smooth_velocity,
                "velocity_magnitude": velocity_magnitude,
                "tcp_path_increment_m": tcp_path_increment_m,
                "tcp_path_length_m": self.tcp_path_length_m,
                "joint_path_increment_rad": joint_path_increment_rad,
                "joint_path_length_rad": self.joint_path_length_rad,
                "direct_tcp_distance_m": self.direct_tcp_distance_m,
                "tcp_path_efficiency_ratio": tcp_path_efficiency_ratio,
                "route_lateral_min_m": self.route_lateral_min_m,
                "route_lateral_max_m": self.route_lateral_max_m,
                "route_vertical_deviation_min_m": (
                    self.route_vertical_deviation_min_m
                ),
                "route_vertical_deviation_max_m": (
                    self.route_vertical_deviation_max_m
                ),
                "selected_avoidance_route": selected_route,
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
        external_geometries = self._obstacle_geometries()
        external_geometry = self._aggregate_obstacle_geometries(
            external_geometries
        )
        plane_geometry = self._plane_geometry()
        obstacle_geometries = list(external_geometries)
        if plane_geometry is not None:
            obstacle_geometries.append(plane_geometry)
        geometry = self._aggregate_obstacle_geometries(obstacle_geometries)
        self_geometry = self._self_geometry()
        obstacle_features = None
        route_sizes_m: list[np.ndarray] = []
        route_positions_m: list[np.ndarray] = []
        if obstacle_geometries:
            if any(item.distances.shape != (6,) for item in obstacle_geometries):
                raise RuntimeError(
                    "obstacle encoding requires six ordered UR5 capsule clearances"
                )
            rows = []
            for obstacle_index, item in enumerate(external_geometries):
                row = [
                    item.distances / OBSTACLE_CLEARANCE_SCALE_M,
                    item.nearest_direction,
                ]
                if self.obstacle_feature_mode == "route_geometry":
                    position, half_extents, quaternion = (
                        self._policy_obstacle_obb(obstacle_index)
                    )
                    projected_size, path_position = self._route_geometry_features(
                        position, half_extents, quaternion
                    )
                    row.extend((projected_size, path_position))
                    route_sizes_m.append(
                        projected_size * self.route_size_scale
                    )
                    route_positions_m.append(
                        path_position * self.route_position_scale
                    )
                rows.append(np.concatenate(row))
            if plane_geometry is not None:
                plane_row = [
                    plane_geometry.distances / OBSTACLE_CLEARANCE_SCALE_M,
                    plane_geometry.nearest_direction,
                ]
                if self.obstacle_feature_mode == "route_geometry":
                    # The infinite mounting plane has no finite OBB centre or
                    # route-dependent extent. Its clearance/direction fields
                    # already identify its safety role without inventing a box.
                    plane_row.extend((np.zeros(3), np.zeros(3)))
                rows.append(np.concatenate(plane_row))
            obstacle_features = np.stack(rows).astype(np.float32)
        observation = build_reference_observation(
            q=q,
            ee_position=ee_position,
            ee_quaternion=ee_quaternion,
            goal_position=self.goal_position,
            goal_quaternion=self.goal_quaternion,
            position_error=position_error,
            orientation_error=orientation_error,
            normalized_joint_velocity=qdot / self.qdot_observation_scale,
            reference_point_offset_m=self.reference_point_offset_m,
            obstacle_features=obstacle_features,
            max_obstacles=self.max_obstacles,
            obstacle_feature_dim=self.obstacle_feature_dim,
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
            "external_d_min": (
                0.80
                if external_geometry is None
                else float(external_geometry.distance_min)
            ),
            "external_mesh_d_min": (
                self._external_mesh_clearance()
                if self.mesh_clearance_diagnostics_enabled
                else (
                    0.80
                    if external_geometry is None
                    else float(external_geometry.distance_min)
                )
            ),
        }
        if route_sizes_m:
            primary_size = route_sizes_m[0]
            primary_position = route_positions_m[0]
            lateral_half_width = float(primary_size[1])
            vertical_half_height = float(primary_size[2])
            if vertical_half_height < 0.8 * lateral_half_width:
                route_shape = "low_wide"
            elif lateral_half_width < 0.8 * vertical_half_height:
                route_shape = "tall_narrow"
            else:
                route_shape = "balanced"
            info.update({
                "primary_obstacle_route_shape": route_shape,
                "primary_obstacle_route_half_extents_m": primary_size.copy(),
                "primary_obstacle_route_position_m": primary_position.copy(),
            })
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

    def _policy_obstacle_obb(
        self, obstacle_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the same current/predicted OBB represented by the encoder."""
        position = self.obstacle_positions[obstacle_index]
        half_extents = self.obstacle_half_extents_all[obstacle_index]
        if (
            self.obstacle_dynamic_mask[obstacle_index]
            and self.dynamic_prediction_enabled
        ):
            position, half_extents, _ = self._predicted_obstacle_envelope(
                obstacle_index
            )
        return (
            np.asarray(position, dtype=np.float32),
            np.asarray(half_extents, dtype=np.float32),
            np.asarray(self.obstacle_quaternions[obstacle_index], dtype=np.float32),
        )

    def _route_geometry_features(
        self,
        position: np.ndarray,
        half_extents: np.ndarray,
        quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project an OBB into a stable start-to-goal route frame.

        The three size values are the OBB half-widths along the nominal path,
        across it, and vertically.  The three location values are the centre's
        coordinates in that same frame.  Unlike local OBB extents, these
        features remain meaningful when a horizontal box is yawed.
        """
        start = np.asarray(self.episode_start_ee_position, dtype=np.float64)
        goal = np.asarray(self.goal_position, dtype=np.float64)
        horizontal = goal[:2] - start[:2]
        norm = float(np.linalg.norm(horizontal))
        if norm <= 1.0e-8:
            along = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            along = np.asarray(
                [horizontal[0] / norm, horizontal[1] / norm, 0.0],
                dtype=np.float64,
            )
        lateral = np.asarray([-along[1], along[0], 0.0], dtype=np.float64)
        vertical = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        frame = np.stack((along, lateral, vertical), axis=0)
        rotation = quaternion_to_matrix(quaternion)
        projected_half_extents = (
            np.abs(frame @ rotation) @ np.asarray(half_extents, dtype=np.float64)
        )
        relative_position = frame @ (
            np.asarray(position, dtype=np.float64) - start
        )
        return (
            (projected_half_extents / self.route_size_scale).astype(np.float32),
            (relative_position / self.route_position_scale).astype(np.float32),
        )

    def _tcp_route_deviation(
        self, ee_position: np.ndarray,
    ) -> tuple[float, float]:
        """Return lateral and vertical deviation from the straight TCP route."""
        start = np.asarray(self.episode_start_ee_position, dtype=np.float64)
        goal = np.asarray(self.goal_position, dtype=np.float64)
        current = np.asarray(ee_position, dtype=np.float64)
        horizontal = goal[:2] - start[:2]
        horizontal_length = float(np.linalg.norm(horizontal))
        if horizontal_length <= 1.0e-8:
            along = np.asarray([1.0, 0.0], dtype=np.float64)
            progress = 0.0
        else:
            along = horizontal / horizontal_length
            progress = float(np.clip(
                np.dot(current[:2] - start[:2], along) / horizontal_length,
                0.0,
                1.0,
            ))
        lateral_axis = np.asarray([-along[1], along[0]], dtype=np.float64)
        lateral = float(np.dot(current[:2] - start[:2], lateral_axis))
        direct_height = float(start[2] + progress * (goal[2] - start[2]))
        return lateral, float(current[2] - direct_height)

    def pose_crossing_diagnostics(self) -> dict[str, object]:
        """Return expensive kinematic diagnostics for the current state.

        This is intentionally an on-demand evaluation hook: computing SVDs and
        self geometry on every training transition would slow the normal PPO
        loop.  Crossing-state evaluators call it at most once per episode.
        """
        q, _ = self.robot.joint_state_exact()
        lower, upper = self.effective_joint_position_limits()
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

    def _external_mesh_clearance(self, query_distance: float = 0.20) -> float:
        """Return exact Bullet mesh clearance to external obstacle bodies.

        Capsule distances remain the policy's shape-independent OBB feature.
        This independent metric is used to verify whether a generated path
        scene genuinely challenges the frozen Base rather than merely
        intersecting a conservative capsule approximation.
        """
        distance_limit = float(query_distance)
        if not np.isfinite(distance_limit) or distance_limit <= 0.0:
            raise ValueError("mesh-clearance query distance must be positive")
        minimum = distance_limit
        for obstacle_body_id in self._obstacle_body_ids:
            points = p.getClosestPoints(
                bodyA=self.robot_id,
                bodyB=obstacle_body_id,
                distance=distance_limit,
                physicsClientId=self.client_id,
            )
            for point in points:
                if int(point[3]) not in self._collision_link_ids:
                    continue
                minimum = min(minimum, float(point[8]))
        return float(minimum)

    def set_mesh_clearance_diagnostics(self, enabled: bool) -> None:
        """Enable exact mesh clearance only during Base scene validation."""
        self.mesh_clearance_diagnostics_enabled = bool(enabled)

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
            "obstacle_layouts": self.obstacle_layouts,
            "grounded_obstacle_count": int(sum(
                layout.startswith("grounded_")
                for layout in self.obstacle_layouts
            )),
            "suspended_obstacle_count": int(sum(
                layout.startswith("suspended_")
                for layout in self.obstacle_layouts
            )),
            "path_relevant_obstacle_count": int(sum(
                layout.endswith("_path")
                for layout in self.obstacle_layouts
            )),
            "obstacle_positions": self.obstacle_positions.copy(),
            "obstacle_half_extents": self.obstacle_half_extents_all.copy(),
            "obstacle_quaternions": self.obstacle_quaternions.copy(),
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
