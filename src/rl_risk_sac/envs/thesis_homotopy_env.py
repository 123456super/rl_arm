from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

from rl_risk_sac.control import SafetyQP, SafetyQPConfig, distance_rate_constraint
from rl_risk_sac.robots.pybullet_robot import PyBulletRobot
from rl_risk_sac.tasks.thesis_reaching import (
    SelfCollisionGeometry,
    ThesisGeometry,
    build_thesis_observation,
    compute_self_collision_geometry,
    compute_thesis_geometry,
    homotopy_reward,
    keypoint_position_jacobian,
    pose_keypoint_errors,
    pose_error,
    thesis_observation_dim,
)
from rl_risk_sac.utils.risk import closest_point_on_segment
from rl_risk_sac.utils.runtime_config import RuntimeConfig


@dataclass
class EpisodeContract:
    scene: str = "none"
    xi: float = 1.0
    strict_obstacle_collision: bool = True
    goal_scale: float = 1.0
    goal_scale_history_min: float | None = None
    goal_scale_frontier_min: float | None = None
    goal_scale_history_probability: float = 0.5
    target_distance_min_m: float | None = None
    target_distance_max_m: float | None = None
    target_orientation_min_rad: float | None = None
    target_orientation_max_rad: float | None = None
    target_history_distance_max_m: float | None = None
    target_history_orientation_max_rad: float | None = None
    target_history_probability: float = 0.5
    orientation_scale: float = 1.0
    position_tolerance: float = 0.055
    orientation_tolerance: float = 0.10
    lambda_self: float = 1.0
    terminate_on_success: bool = True


class ThesisHomotopyEnv(gym.Env):
    """Hybrid Keypoint + Jacobian environment for S0/S1/S2.

    Observations combine keypoint errors, their joint Jacobian, explicit pose
    errors, task-space twist, and collision geometry in 162 dimensions.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: dict[str, Any], render_mode: str | None = None) -> None:
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        self.render_mode = render_mode
        self.config = config
        self.runtime = RuntimeConfig.from_mapping(config)
        thesis = config["thesis"]
        observation = thesis.get("observation", {})
        reward = thesis.get("reward", {})
        thesis_observation_dim(
            observation.get("include_orientation_error_vector", False),
            keypoint_jacobian_pose=observation.get("keypoint_jacobian_pose", True),
            hybrid_explicit_pose_error=observation.get("hybrid_explicit_pose_error", True),
        )
        if (
            reward.get("pose_objective", "unified_keypoint") != "unified_keypoint"
            or not reward.get("keypoint_pose_reward", True)
            or reward.get("rtpc_orientation_reward", False)
            or reward.get("pose_balanced_rtpc_reward", False)
        ):
            raise ValueError("only unified keypoint pose reward is supported")
        self.physics_dt = float(thesis.get("physics_dt", 1.0 / 240.0))
        self.control_dt = float(thesis.get("control_dt", 0.05))
        self.substeps = int(round(self.control_dt / self.physics_dt))
        if self.substeps != 12 or not np.isclose(self.substeps * self.physics_dt, self.control_dt):
            raise ValueError("thesis protocol requires 12 substeps at 1/240 s for a 0.05 s control period")
        self.horizon = int(thesis.get("horizon", 500))
        self.action_scale = np.full(6, float(thesis.get("action_scale", 0.7)), dtype=np.float32)
        self.qdot_observation_scale = np.full(
            6, float(thesis.get("qdot_observation_scale", thesis.get("action_scale", 0.7))),
            dtype=np.float32,
        )
        if np.any(self.qdot_observation_scale <= 0.0):
            raise ValueError("qdot observation scales must be positive")
        precision_control = thesis.get("precision_control", {})
        self.precision_control_enabled = bool(precision_control.get("enabled", False))
        self.precision_min_action_scale = float(precision_control.get("min_action_scale", 0.05))
        self.precision_strict_action_scale = float(precision_control.get("strict_action_scale", 0.10))
        self.success_hold_steps = int(thesis.get("success_hold_steps", 1))
        if not 0.0 < self.precision_min_action_scale <= self.precision_strict_action_scale <= 1.0:
            raise ValueError("precision action scales must satisfy 0 < minimum <= strict <= 1")
        if self.success_hold_steps < 1:
            raise ValueError("success_hold_steps must be positive")
        self.velocity_limits = np.full(6, np.pi, dtype=np.float32)
        self.ee_linear_velocity_scale = float(thesis.get("ee_linear_velocity_scale", 1.0))
        self.ee_angular_velocity_scale = float(thesis.get("ee_angular_velocity_scale", 4.2))
        if self.ee_linear_velocity_scale <= 0.0 or self.ee_angular_velocity_scale <= 0.0:
            raise ValueError("end-effector velocity scales must be positive")
        self.motor_force = float(thesis.get("joint_motor_force", 90.0))
        self.obstacle_radius = float(thesis.get("obstacle_radius", 0.075))
        self.obstacle_speed = float(thesis.get("obstacle_speed", 0.1))
        self.d_safe = float(thesis.get("d_safe", 0.12))
        self_collision_config = thesis.get("self_collision", {})
        self.d_self_safe = float(self_collision_config.get("safe_distance_m", 0.005))
        self.self_query_distance = float(self_collision_config.get("query_distance_m", 0.25))
        self.self_ttc_max = float(self_collision_config.get("ttc_max_s", 3.0))
        self.self_approach_velocity_scale = float(
            self_collision_config.get("approach_velocity_scale", 0.7)
        )
        self.self_ttc_tau = float(self_collision_config.get("ttc_tau_s", 1.0))
        if not 0.0 < self.d_self_safe < self.self_query_distance:
            raise ValueError("self collision distances must satisfy 0 < safe < query")
        projection = thesis.get("self_safety_projection", {})
        self.self_projection_enabled = bool(projection.get("enabled", False))
        self.self_projection_activation_distance = float(
            projection.get("activation_distance_m", 0.05)
        )
        self.self_projection_gain = float(projection.get("gain", 5.0))
        self.self_projection_fd_epsilon = float(projection.get("fd_epsilon", 0.001))
        if min(
            self.self_projection_activation_distance,
            self.self_projection_gain,
            self.self_projection_fd_epsilon,
        ) <= 0.0:
            raise ValueError("self safety projection parameters must be positive")
        self.self_safety_qp = SafetyQP(SafetyQPConfig(
            max_iterations=int(projection.get("max_iterations", 16)),
            eps=float(projection.get("eps", 1.0e-8)),
            violation_tolerance=float(projection.get("violation_tolerance", 1.0e-6)),
        ))
        self.position_tolerance = float(thesis.get("position_tolerance", 0.055))
        self.orientation_tolerance = float(thesis.get("orientation_tolerance", 0.10))
        reward_config = thesis.get("reward", {})
        self.reward_parameters = {
            "d_safe": self.d_safe,
            "d_self_safe": self.d_self_safe,
            "position_sigma": float(reward_config.get("position_sigma_m", 0.20)),
            "micro_power": float(reward_config.get("micro_power", 4.0)),
            "fine_position_sigma": float(reward_config.get("fine_position_sigma_m", 0.02)),
            "fine_orientation_sigma": float(reward_config.get("fine_orientation_sigma_rad", 0.06)),
            "fine_position_progress_scale": float(
                reward_config.get("fine_position_progress_scale", 1.0)
            ),
            "fine_orientation_progress_scale": float(
                reward_config.get("fine_orientation_progress_scale", 1.0)
            ),
            "precision_position_scale": float(
                reward_config.get("precision_position_scale_m", 0.03)
            ),
            "precision_orientation_scale": float(
                reward_config.get("precision_orientation_scale_rad", 0.09)
            ),
            "precision_stop_cost_weight": float(
                reward_config.get("precision_stop_cost_weight", 0.20)
            ),
            "joint_precision_progress_scale": float(
                reward_config.get("joint_precision_progress_scale", 0.0)
            ),
            "joint_precision_temperature": float(
                reward_config.get("joint_precision_temperature", 2.0)
            ),
            "joint_position_tolerance": float(
                reward_config.get("joint_position_tolerance_m", self.position_tolerance)
            ),
            "joint_orientation_tolerance": float(
                reward_config.get(
                    "joint_orientation_tolerance_rad", self.orientation_tolerance
                )
            ),
            "leave_tolerance_multiplier": float(
                reward_config.get("leave_tolerance_multiplier", 2.0)
            ),
            "leave_joint_tolerance_penalty": float(
                reward_config.get("leave_joint_tolerance_penalty", 0.0)
            ),
            "hold_reward_scale": float(reward_config.get("hold_reward_scale", 0.0)),
            "partial_precision_reward_scale": float(
                reward_config.get("partial_precision_reward_scale", 0.0)
            ),
            "position_progress_scale": float(reward_config.get("position_progress_scale", 2.0)),
            "orientation_progress_scale": float(reward_config.get("orientation_progress_scale", 2.0)),
            "position_state_cost_weight": float(
                reward_config.get("position_state_cost_weight", 0.02)
            ),
            "orientation_state_cost_weight": float(
                reward_config.get("orientation_state_cost_weight", 0.02)
            ),
            "rtpc_orientation_reward": bool(
                reward_config.get("rtpc_orientation_reward", False)
            ),
            "pose_balanced_rtpc_reward": bool(
                reward_config.get("pose_balanced_rtpc_reward", False)
            ),
            "orientation_absolute_error_scale": float(
                reward_config.get("orientation_absolute_error_scale", 0.125)
            ),
            "orientation_error_progress_scale": float(
                reward_config.get("orientation_error_progress_scale", 0.15)
            ),
            "orientation_completion_reward_scale": float(
                reward_config.get("orientation_completion_reward_scale", 0.10)
            ),
            "orientation_absolute_position_gate_sigma": float(
                reward_config.get(
                    "orientation_absolute_position_gate_sigma_m", 0.20
                )
            ),
            "orientation_completion_position_gate_multiplier": float(
                reward_config.get(
                    "orientation_completion_position_gate_multiplier", 1.0
                )
            ),
            "keypoint_pose_reward": bool(
                reward_config.get("keypoint_pose_reward", True)
            ),
            "keypoint_tracking_scale": float(
                reward_config.get("keypoint_tracking_scale", 0.20)
            ),
            "keypoint_tracking_sigma": float(
                reward_config.get("keypoint_tracking_sigma_m", 0.05)
            ),
            "keypoint_progress_scale": float(
                reward_config.get("keypoint_progress_scale", 10.0)
            ),
            "keypoint_precision_reward_scale": float(
                reward_config.get("keypoint_precision_reward_scale", 0.0)
            ),
            "success_bonus": float(reward_config.get("success_bonus", 20.0)),
            "velocity_cost_weight": float(reward_config.get("velocity_cost_weight", 0.04)),
            "smooth_cost_weight": float(reward_config.get("smooth_cost_weight", 0.01)),
            "hard_failure_penalty": float(
                reward_config.get("hard_failure_penalty", 20.0)
            ),
            "timeout_penalty": float(reward_config.get("timeout_penalty", 2.0)),
            "safety_risk_weight": float(reward_config.get("safety_risk_weight", 2.0)),
            "safety_clearance_weight": float(
                reward_config.get("safety_clearance_weight", 8.0)
            ),
            "self_risk_weight": float(reward_config.get("self_risk_weight", 2.0)),
            "self_clearance_weight": float(
                reward_config.get("self_clearance_weight", 8.0)
            ),
            "external_safety_scale": float(
                reward_config.get("external_safety_scale", 0.05)
            ),
            "self_safety_scale": float(reward_config.get("self_safety_scale", 0.05)),
        }
        self.goal_sample_max_attempts = int(thesis.get("goal_sample_max_attempts", 10000))
        if self.goal_sample_max_attempts < 1:
            raise ValueError("goal_sample_max_attempts must be positive")
        task_space_sampling = thesis.get("joint_pose_curriculum", {}).get(
            "task_space_sampling", {}
        )
        self.task_space_balanced_bins = bool(
            task_space_sampling.get("balanced_bins", False)
        )
        self.task_space_position_bins = int(
            task_space_sampling.get("position_bins", 10)
        )
        self.task_space_orientation_bins = int(
            task_space_sampling.get("orientation_bins", 10)
        )
        self.task_space_reset_max_attempts = int(
            task_space_sampling.get("reset_max_attempts", 32)
        )
        if self.task_space_position_bins < 1 or self.task_space_orientation_bins < 1:
            raise ValueError("task-space sampling bin counts must be positive")
        if self.task_space_reset_max_attempts < 1:
            raise ValueError("task-space reset_max_attempts must be positive")
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
        self.obstacle_position = np.zeros(3, dtype=np.float32)
        self.obstacle_velocity = np.zeros(3, dtype=np.float32)
        self.goal_position = np.zeros(3, dtype=np.float32)
        self.goal_quaternion = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.goal_joint_positions = np.zeros(6, dtype=np.float32)
        self.goal_ik_validation: dict[str, Any] = {}
        self.goal_sample_attempts_last = 0
        self.sampled_goal_scale = 1.0
        self.sampled_target_distance_m = 0.0
        self.sampled_target_orientation_rad = 0.0
        self.sampled_target_position_bin = -1
        self.sampled_target_orientation_bin = -1
        total_task_space_bins = self.task_space_position_bins * self.task_space_orientation_bins
        self.goal_proposal_counts = np.zeros(total_task_space_bins, dtype=np.int64)
        self.goal_accept_counts = np.zeros(total_task_space_bins, dtype=np.int64)
        self.goal_resample_counts = np.zeros(total_task_space_bins, dtype=np.int64)
        self._task_space_bin_order = np.empty(0, dtype=np.int64)
        self._task_space_bin_cursor = 0
        self._selected_task_space_position_bin = -1
        self._selected_task_space_orientation_bin = -1
        self.task_space_bin_probabilities: np.ndarray | None = None
        self.step_count = 0
        self.success_hold_count = 0
        self.obstacle_contact_seen = False
        self.previous_rho_position = 0.0
        self.previous_rho_orientation = 0.0
        self.previous_keypoint_distance = 0.0
        observation_config = thesis.get("observation", {})
        self.include_orientation_error_vector = bool(
            observation_config.get("include_orientation_error_vector", False)
        )
        self.keypoint_jacobian_pose = bool(
            observation_config.get("keypoint_jacobian_pose", True)
        )
        self.hybrid_explicit_pose_error = bool(
            observation_config.get("hybrid_explicit_pose_error", True)
        )
        if self.hybrid_explicit_pose_error and not self.keypoint_jacobian_pose:
            raise ValueError(
                "hybrid_explicit_pose_error requires keypoint_jacobian_pose"
            )
        self.keypoint_cube_side_m = float(
            observation_config.get("keypoint_cube_side_m", 0.30)
        )
        self.keypoint_error_scale_m = float(
            observation_config.get("keypoint_error_scale_m", 1.0)
        )
        self.keypoint_jacobian_scale = float(
            observation_config.get("keypoint_jacobian_scale", 1.0)
        )
        self._link_name_to_id: dict[str, int] = {}
        self._collision_link_ids: set[int] = set()
        self._self_link_order: list[int] = []
        self._self_pair_link_ids: list[tuple[int, int]] = []
        self._self_pair_slots: list[tuple[int, int]] = []

        # Loading the URDF dominates short curriculum episodes. Keep one world
        # alive for the lifetime of the environment and reset only its state.
        self._initialize_world()

        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(thesis_observation_dim(
                self.include_orientation_error_vector,
                keypoint_jacobian_pose=self.keypoint_jacobian_pose,
                hybrid_explicit_pose_error=self.hybrid_explicit_pose_error,
            ),),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

    def configure_episode(
        self, scene: str, *, xi: float, strict: bool, goal_scale: float = 1.0,
        orientation_scale: float = 1.0, position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
        lambda_self: float = 1.0, goal_scale_history_min: float | None = None,
        goal_scale_frontier_min: float | None = None,
        goal_scale_history_probability: float = 0.5,
        target_distance_min_m: float | None = None,
        target_distance_max_m: float | None = None,
        target_orientation_min_rad: float | None = None,
        target_orientation_max_rad: float | None = None,
        target_history_distance_max_m: float | None = None,
        target_history_orientation_max_rad: float | None = None,
        target_history_probability: float = 0.5,
        task_space_bin_probabilities: list[float] | np.ndarray | None = None,
        terminate_on_success: bool = True,
    ) -> None:
        if scene not in {"none", "static", "dynamic"}:
            raise ValueError(f"unknown thesis scene {scene!r}")
        if not 0.02 <= float(xi) <= 1.0:
            raise ValueError("xi must be in [0.02, 1]")
        if not 0.0 < float(goal_scale) <= 1.0:
            raise ValueError("goal_scale must be in (0, 1]")
        if (goal_scale_history_min is None) != (goal_scale_frontier_min is None):
            raise ValueError(
                "goal_scale_history_min and goal_scale_frontier_min must be set together"
            )
        if goal_scale_history_min is not None:
            history_min = float(goal_scale_history_min)
            frontier_min = float(goal_scale_frontier_min)
            if not 0.0 < history_min <= frontier_min <= float(goal_scale):
                raise ValueError(
                    "goal scale sampling requires 0 < history_min <= frontier_min <= goal_scale"
                )
        if not 0.0 <= float(goal_scale_history_probability) <= 1.0:
            raise ValueError("goal_scale_history_probability must be in [0, 1]")
        task_bounds = (
            target_distance_min_m, target_distance_max_m,
            target_orientation_min_rad, target_orientation_max_rad,
        )
        if any(value is not None for value in task_bounds):
            if any(value is None for value in task_bounds):
                raise ValueError("all task-space target bounds must be set together")
            if not 0.0 < float(target_distance_min_m) < float(target_distance_max_m):
                raise ValueError("task-space distance bounds must satisfy 0 < min < max")
            if not 0.0 <= float(target_orientation_min_rad) < float(target_orientation_max_rad) <= np.pi:
                raise ValueError("task-space orientation bounds must satisfy 0 <= min < max <= pi")
            if target_history_distance_max_m is not None and not (
                float(target_distance_min_m) <= float(target_history_distance_max_m)
                <= float(target_distance_max_m)
            ):
                raise ValueError("history distance maximum must lie inside current bounds")
            if target_history_orientation_max_rad is not None and not (
                float(target_orientation_min_rad) <= float(target_history_orientation_max_rad)
                <= float(target_orientation_max_rad)
            ):
                raise ValueError("history orientation maximum must lie inside current bounds")
        if not 0.0 <= float(target_history_probability) <= 1.0:
            raise ValueError("target_history_probability must be in [0, 1]")
        expected_bins = self.task_space_position_bins * self.task_space_orientation_bins
        if task_space_bin_probabilities is None:
            self.task_space_bin_probabilities = None
        else:
            probabilities = np.asarray(task_space_bin_probabilities, dtype=np.float64)
            if probabilities.shape != (expected_bins,):
                raise ValueError(
                    f"task_space_bin_probabilities must contain {expected_bins} values"
                )
            if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
                raise ValueError("task-space bin probabilities must be finite and positive")
            self.task_space_bin_probabilities = probabilities / probabilities.sum()
        if not 0.0 <= float(orientation_scale) <= 1.0:
            raise ValueError("orientation_scale must be in [0, 1]")
        if not 0.0 <= float(lambda_self) <= 1.0:
            raise ValueError("lambda_self must be in [0, 1]")
        effective_position_tolerance = (
            self.position_tolerance if position_tolerance is None else float(position_tolerance)
        )
        if not 0.0 < effective_position_tolerance <= 0.20:
            raise ValueError("position_tolerance must be in (0, 0.20]")
        effective_orientation_tolerance = (
            self.orientation_tolerance if orientation_tolerance is None else float(orientation_tolerance)
        )
        if not 0.0 < effective_orientation_tolerance <= np.pi:
            raise ValueError("orientation_tolerance must be in (0, pi]")
        self.contract = EpisodeContract(
            scene=scene,
            xi=float(xi),
            strict_obstacle_collision=bool(strict),
            goal_scale=float(goal_scale),
            goal_scale_history_min=(
                None if goal_scale_history_min is None else float(goal_scale_history_min)
            ),
            goal_scale_frontier_min=(
                None if goal_scale_frontier_min is None else float(goal_scale_frontier_min)
            ),
            goal_scale_history_probability=float(goal_scale_history_probability),
            target_distance_min_m=(None if target_distance_min_m is None else float(target_distance_min_m)),
            target_distance_max_m=(None if target_distance_max_m is None else float(target_distance_max_m)),
            target_orientation_min_rad=(None if target_orientation_min_rad is None else float(target_orientation_min_rad)),
            target_orientation_max_rad=(None if target_orientation_max_rad is None else float(target_orientation_max_rad)),
            target_history_distance_max_m=(None if target_history_distance_max_m is None else float(target_history_distance_max_m)),
            target_history_orientation_max_rad=(None if target_history_orientation_max_rad is None else float(target_history_orientation_max_rad)),
            target_history_probability=float(target_history_probability),
            orientation_scale=float(orientation_scale),
            position_tolerance=effective_position_tolerance,
            orientation_tolerance=effective_orientation_tolerance,
            lambda_self=float(lambda_self),
            terminate_on_success=bool(terminate_on_success),
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._task_space_bin_order = np.empty(0, dtype=np.int64)
            self._task_space_bin_cursor = 0
        self._disable_obstacle()
        balanced_task_space = bool(
            self.task_space_balanced_bins
            and self.contract.target_distance_min_m is not None
        )
        if balanced_task_space:
            self._select_balanced_task_space_bin()
            last_error: RuntimeError | None = None
            for _ in range(self.task_space_reset_max_attempts):
                self._sample_valid_reset()
                try:
                    self._sample_goal_pose()
                except RuntimeError as error:
                    last_error = error
                    continue
                break
            else:
                raise RuntimeError(
                    "failed to sample the selected balanced task-space cell "
                    f"({self._selected_task_space_position_bin}, "
                    f"{self._selected_task_space_orientation_bin}) across "
                    f"{self.task_space_reset_max_attempts} valid robot resets"
                ) from last_error
        else:
            self._sample_valid_reset()
            self._sample_goal_pose()
        self._create_obstacle()
        self.step_count = 0
        self.success_hold_count = 0
        self.obstacle_contact_seen = False
        observation, info, geometry, self_geometry, position_error, orientation_error = self._observe()
        self.previous_rho_position = float(np.linalg.norm(position_error))
        self.previous_rho_orientation = float(np.linalg.norm(orientation_error))
        self.previous_keypoint_distance = float(np.mean(info["keypoint_distances"]))
        info.update(self._contract_info())
        info.update(self._geometry_info(geometry))
        info.update(self._self_geometry_info(self_geometry))
        return observation, info

    def _select_balanced_task_space_bin(self) -> None:
        total_bins = self.task_space_position_bins * self.task_space_orientation_bins
        if self.task_space_bin_probabilities is not None:
            flat_bin = int(self.rng.choice(
                total_bins, p=self.task_space_bin_probabilities,
            ))
        elif self._task_space_bin_cursor >= self._task_space_bin_order.size:
            self._task_space_bin_order = self.rng.permutation(total_bins)
            self._task_space_bin_cursor = 0
            flat_bin = int(self._task_space_bin_order[self._task_space_bin_cursor])
            self._task_space_bin_cursor += 1
        else:
            flat_bin = int(self._task_space_bin_order[self._task_space_bin_cursor])
            self._task_space_bin_cursor += 1
        (
            self._selected_task_space_position_bin,
            self._selected_task_space_orientation_bin,
        ) = divmod(flat_bin, self.task_space_orientation_bins)

    def _initialize_world(self) -> None:
        p.resetSimulation(physicsClientId=self.client_id)
        p.setTimeStep(self.physics_dt, physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, 0.0, physicsClientId=self.client_id)
        # Robot.load currently initializes its joints. A private generator
        # prevents that one-time initialization from consuming episode RNG.
        self.robot_id = self.robot.load(
            np.random.default_rng(0), enable_self_collision=True,
        )
        self._resolve_collision_links()
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.obstacle_radius,
            physicsClientId=self.client_id,
        )
        self._obstacle_body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            basePosition=[10.0, 10.0, 10.0],
            physicsClientId=self.client_id,
        )
        self._disable_obstacle()

    def _disable_obstacle(self) -> None:
        self.obstacle_id = None
        self.obstacle_position.fill(0.0)
        self.obstacle_velocity.fill(0.0)
        p.resetBasePositionAndOrientation(
            self._obstacle_body_id,
            [10.0, 10.0, 10.0],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        p.resetBaseVelocity(
            self._obstacle_body_id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.client_id,
        )

    def _advance_success_hold(self, strict_pose_reached: bool) -> bool:
        """Advance the consecutive strict-pose counter and report success."""
        self.success_hold_count = (
            self.success_hold_count + 1 if strict_pose_reached else 0
        )
        return bool(
            strict_pose_reached and self.success_hold_count >= self.success_hold_steps
        )

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # SAC directly controls all six joint velocities.  The action is the
        # policy command, not a residual around an analytic controller.
        policy_command = action * self.action_scale
        precision_action_scale = 1.0
        if self.precision_control_enabled:
            z = max(
                self.previous_rho_position / self.contract.position_tolerance,
                self.previous_rho_orientation / self.contract.orientation_tolerance,
            )
            if z <= 1.0:
                precision_action_scale = self.precision_min_action_scale + (
                    self.precision_strict_action_scale - self.precision_min_action_scale
                ) * max(z, 0.0)
            else:
                precision_action_scale = self.precision_strict_action_scale + (
                    1.0 - self.precision_strict_action_scale
                ) * min(z - 1.0, 1.0) ** 2
        scaled_policy_command = policy_command * precision_action_scale
        unprojected_command = np.clip(
            scaled_policy_command, -self.action_scale, self.action_scale
        )
        command, projection_info = self._project_self_safe_command(unprojected_command)
        q_before, qdot_before = self.robot.joint_state()
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

        for _ in range(self.substeps):
            self._advance_obstacle()
            p.setJointMotorControlArray(
                self.robot_id,
                self.robot.joint_ids,
                p.VELOCITY_CONTROL,
                targetVelocities=command.tolist(),
                forces=[self.motor_force] * 6,
                physicsClientId=self.client_id,
            )
            p.stepSimulation(physicsClientId=self.client_id)
            executed_substeps += 1
            geometry = self._geometry()
            if geometry is not None:
                max_risk = max(max_risk, geometry.risk_max)
                min_distance = min(min_distance, geometry.distance_min)
            self_geometry = self._self_geometry()
            self_max_risk = max(self_max_risk, self_geometry.risk_max)
            self_min_distance = min(self_min_distance, self_geometry.distance_min)
            self_min_ttc = min(self_min_ttc, self_geometry.ttc_min)
            self_max_approach = max(self_max_approach, self_geometry.approach_max)
            events = self._collision_events()
            obstacle_collision = obstacle_collision or events["obstacle_collision"]
            self_collision = self_collision or events["self_collision"]
            environment_collision = environment_collision or events["environment_collision"]
            q_now, _ = self.robot.joint_state()
            lower, upper = self.robot.joint_lower_limits, self.robot.joint_upper_limits
            joint_limit = joint_limit or bool(np.any(q_now < lower - 1e-6) or np.any(q_now > upper + 1e-6))
            hard_failure = self_collision or environment_collision or joint_limit
            strict_obstacle_failure = obstacle_collision and self.contract.strict_obstacle_collision
            if hard_failure or strict_obstacle_failure:
                break

        self.obstacle_contact_seen = self.obstacle_contact_seen or obstacle_collision
        self.step_count += 1
        observation, info, endpoint_geometry, endpoint_self_geometry, position_error, orientation_error = self._observe()
        next_rho_position = float(np.linalg.norm(position_error))
        next_rho_orientation = float(np.linalg.norm(orientation_error))
        next_keypoint_distances = np.asarray(
            info["keypoint_distances"], dtype=np.float32
        )
        next_keypoint_distance = float(np.mean(next_keypoint_distances))
        keypoint_progress = float(
            self.previous_keypoint_distance - next_keypoint_distance
        )
        keypoint_tracking_quality = float(np.mean(np.exp(
            -next_keypoint_distances
            / float(self.reward_parameters.get("keypoint_tracking_sigma", 0.05))
        )))
        hard_failure = bool(self_collision or environment_collision or joint_limit)
        obstacle_failure = bool(obstacle_collision and self.contract.strict_obstacle_collision and not hard_failure)
        position_reached = bool(
            not hard_failure
            and not obstacle_failure
            and next_rho_position <= self.contract.position_tolerance
        )
        strict_pose_reached = bool(
            position_reached
            and next_rho_orientation <= self.contract.orientation_tolerance
        )
        task_reached = self._advance_success_hold(strict_pose_reached)
        terminated = bool(
            hard_failure or obstacle_failure
            or (task_reached and self.contract.terminate_on_success)
        )
        truncated = bool(self.step_count >= self.horizon and not terminated)
        _, qdot_after = self.robot.joint_state()
        # Average across joints before the bounded reward mapping. Summing six
        # normalized squared changes made the smoothness term saturate on
        # nearly every transition.
        smooth_velocity = float(
            np.mean(np.square((qdot_after - qdot_before) / self.action_scale))
        )
        velocity_magnitude = float(np.mean(np.square(qdot_after / self.action_scale)))
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
        episode_reward_parameters = dict(self.reward_parameters)
        if bool(self.config["thesis"]["reward"].get(
            "use_episode_tolerance_for_joint_precision", False
        )):
            episode_reward_parameters["joint_position_tolerance"] = (
                self.contract.position_tolerance
            )
            episode_reward_parameters["joint_orientation_tolerance"] = (
                self.contract.orientation_tolerance
            )
        reward, reward_fields = homotopy_reward(
            rho_position=self.previous_rho_position,
            next_rho_position=next_rho_position,
            rho_orientation=self.previous_rho_orientation,
            next_rho_orientation=next_rho_orientation,
            orientation_tolerance=self.contract.orientation_tolerance,
            smooth_velocity=smooth_velocity,
            velocity_magnitude=velocity_magnitude,
            orientation_scale=self.contract.orientation_scale,
            task_reached=task_reached,
            hard_failure=hard_failure,
            timeout=truncated,
            obstacle_collision=obstacle_collision,
            terminal_obstacle_collision=obstacle_failure,
            risk_max=max_risk,
            distance_min=min_distance,
            xi=self.contract.xi if self.contract.scene != "none" else 1.0,
            self_risk_max=self_max_risk,
            self_distance_min=self_min_distance,
            lambda_self=self.contract.lambda_self,
            gamma=float(self.config["sac"]["gamma"]),
            horizon=self.horizon,
            keypoint_tracking_quality=keypoint_tracking_quality,
            keypoint_distance=self.previous_keypoint_distance,
            next_keypoint_distance=next_keypoint_distance,
            **episode_reward_parameters,
        )
        collision_assisted_reach = bool(task_reached and self.obstacle_contact_seen)
        safe_success = bool(task_reached and not self.obstacle_contact_seen and not hard_failure)
        cost = float(
            max_risk
            + self_max_risk
            + 3.0 * (min_distance < self.d_safe)
            + 3.0 * (self_min_distance < self.d_self_safe)
            + 8.0 * (obstacle_collision or hard_failure)
        )
        info.update(
            {
                **self._contract_info(),
                **self._geometry_info(endpoint_geometry),
                **self._self_geometry_info(endpoint_self_geometry),
                **reward_fields,
                "reward": reward,
                "hard_penalty": reward_fields["hard_penalty"],
                "safety_penalty": reward_fields["safety_penalty"],
                "terminal_guard_penalty": reward_fields["terminal_guard_penalty"],
                "cost": cost,
                "policy_joint_velocity": policy_command.copy(),
                "precision_action_scale": precision_action_scale,
                "scaled_policy_joint_velocity": scaled_policy_command.copy(),
                "commanded_joint_velocity": command.copy(),
                "joint_velocity": qdot_after.copy(),
                **projection_info,
                "task_reached": task_reached,
                "strict_pose_reached": strict_pose_reached,
                "success_hold_count": self.success_hold_count,
                "success_hold_steps_required": self.success_hold_steps,
                "position_reached": position_reached,
                "timeout": truncated,
                "collision_assisted_reach": collision_assisted_reach,
                "safe_success": safe_success,
                "success": safe_success,
                "collision": bool(obstacle_collision or self_collision or environment_collision),
                "obstacle_collision": obstacle_collision,
                "self_collision": self_collision,
                "environment_collision": environment_collision,
                "joint_limit": joint_limit,
                "hard_failure": hard_failure,
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
                "qdot_cmd": command.copy(),
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
        self.previous_keypoint_distance = next_keypoint_distance
        return observation, reward, cost, terminated, truncated, info

    def _sample_valid_reset(self) -> None:
        for _ in range(100):
            self.robot.reset_joints(self.rng)
            p.setJointMotorControlArray(
                self.robot_id,
                self.robot.joint_ids,
                p.VELOCITY_CONTROL,
                targetVelocities=[0.0] * 6,
                forces=[self.motor_force] * 6,
                physicsClientId=self.client_id,
            )
            p.performCollisionDetection(physicsClientId=self.client_id)
            events = self._collision_events()
            self_geometry = self._self_geometry()
            q, _ = self.robot.joint_state()
            if (
                not events["self_collision"]
                and self_geometry.distance_min >= self.d_self_safe
                and np.all(q >= self.robot.joint_lower_limits)
                and np.all(q <= self.robot.joint_upper_limits)
            ):
                return
        raise RuntimeError("failed to sample a valid initial joint configuration in 100 attempts")

    def _sample_goal_pose(self) -> None:
        if self.contract.target_distance_min_m is not None:
            self._sample_task_space_goal_pose()
            return
        initial_q, initial_qdot = self.robot.joint_state()
        lower = self.robot.joint_lower_limits + 0.10
        upper = self.robot.joint_upper_limits - 0.10
        sampled_goal_scale = self.contract.goal_scale
        if self.contract.goal_scale_history_min is not None:
            history_min = self.contract.goal_scale_history_min
            frontier_min = self.contract.goal_scale_frontier_min
            if self.rng.random() < self.contract.goal_scale_history_probability:
                sampled_goal_scale = self.rng.uniform(history_min, frontier_min)
            else:
                sampled_goal_scale = self.rng.uniform(frontier_min, self.contract.goal_scale)
        self.sampled_goal_scale = float(sampled_goal_scale)
        for attempt in range(1, self.goal_sample_max_attempts + 1):
            full_candidate = self.rng.uniform(lower, upper).astype(np.float32)
            candidate = initial_q + self.sampled_goal_scale * (full_candidate - initial_q)
            self._set_joint_state(candidate)
            p.performCollisionDetection(physicsClientId=self.client_id)
            position, quaternion = self.robot.end_effector_pose()
            within_workspace = bool(
                0.25 <= position[0] <= 0.78
                and -0.45 <= position[1] <= 0.45
                and 0.18 <= position[2] <= 0.78
            )
            target_self_collision = self._collision_events()["self_collision"]
            target_self_clearance = self._self_geometry().distance_min
            ik_validation = self.robot.check_pose_ik(
                position,
                quaternion,
                position_tolerance=0.01,
                orientation_tolerance=0.05,
                rest_pose=candidate,
            )
            self._set_joint_state(initial_q, initial_qdot)
            current_position, current_quaternion = self.robot.end_effector_pose()
            initial_position_error, initial_orientation_error = pose_error(position, quaternion, current_position, current_quaternion)
            already_reached = (
                np.linalg.norm(initial_position_error) <= self.contract.position_tolerance
                and np.linalg.norm(initial_orientation_error) <= self.contract.orientation_tolerance
            )
            if (
                within_workspace
                and not target_self_collision
                and target_self_clearance >= self.d_self_safe
                and bool(ik_validation["reachable"])
                and not already_reached
                and np.linalg.norm(initial_orientation_error) < np.pi - 1e-3
            ):
                self.goal_position = position.copy()
                self.goal_quaternion = quaternion.copy()
                self.goal_joint_positions = candidate.copy()
                self.goal_ik_validation = {
                    **ik_validation,
                    "q_ik": np.asarray(ik_validation["q_ik"], dtype=np.float32),
                }
                self.goal_sample_attempts_last = attempt
                return
        self._set_joint_state(initial_q, initial_qdot)
        raise RuntimeError(
            "failed to sample a valid pose goal in "
            f"{self.goal_sample_max_attempts} attempts"
        )

    def _sample_task_space_goal_pose(self) -> None:
        """Sample a Cartesian pose shell, then retain only an IK-safe solution."""
        initial_q, initial_qdot = self.robot.joint_state()
        initial_position, initial_quaternion = self.robot.end_effector_pose()
        distance_min = float(self.contract.target_distance_min_m)
        distance_max = float(self.contract.target_distance_max_m)
        orientation_min = float(self.contract.target_orientation_min_rad)
        orientation_max = float(self.contract.target_orientation_max_rad)
        history_distance_max = self.contract.target_history_distance_max_m
        history_orientation_max = self.contract.target_history_orientation_max_rad

        # Select the semantic cell once, before any workspace/IK rejection.
        # Re-sampling the cell after every rejection systematically favored the
        # easy near-distance cells.  Keeping the selected cell fixed makes the
        # distribution of accepted goals match the intended uniform 2-D grid.
        position_bin = -1
        orientation_bin = -1
        if self.task_space_balanced_bins:
            if self._selected_task_space_position_bin < 0:
                self._select_balanced_task_space_bin()
            position_bin = self._selected_task_space_position_bin
            orientation_bin = self._selected_task_space_orientation_bin
            distance_width = (
                distance_max - distance_min
            ) / self.task_space_position_bins
            orientation_width = (
                orientation_max - orientation_min
            ) / self.task_space_orientation_bins
            distance_low = distance_min + position_bin * distance_width
            distance_high = distance_low + distance_width
            orientation_low = orientation_min + orientation_bin * orientation_width
            orientation_high = orientation_low + orientation_width

        for attempt in range(1, self.goal_sample_max_attempts + 1):
            if position_bin >= 0 and orientation_bin >= 0:
                flat_bin = position_bin * self.task_space_orientation_bins + orientation_bin
                self.goal_proposal_counts[flat_bin] += 1
            if self.task_space_balanced_bins:
                distance = float(self.rng.uniform(distance_low, distance_high))
                angle = float(self.rng.uniform(orientation_low, orientation_high))
            else:
                use_history = bool(
                    history_distance_max is not None
                    and history_orientation_max is not None
                    and self.rng.random() < self.contract.target_history_probability
                )
            if not self.task_space_balanced_bins and use_history:
                distance_high = float(history_distance_max)
                orientation_high = float(history_orientation_max)
                distance = float(self.rng.uniform(distance_min, distance_high))
                angle = float(self.rng.uniform(orientation_min, orientation_high))
            elif (
                not self.task_space_balanced_bins
                and history_distance_max is not None
                and history_orientation_max is not None
            ):
                # The frontier is the new part of the cumulative rectangle.
                # Alternate its position and orientation strips so that far poses
                # with easy attitudes and near poses with hard attitudes both occur.
                if self.rng.random() < 0.5:
                    distance = float(self.rng.uniform(float(history_distance_max), distance_max))
                    angle = float(self.rng.uniform(orientation_min, orientation_max))
                else:
                    distance = float(self.rng.uniform(distance_min, distance_max))
                    angle = float(self.rng.uniform(float(history_orientation_max), orientation_max))
            elif not self.task_space_balanced_bins:
                distance = float(self.rng.uniform(distance_min, distance_max))
                angle = float(self.rng.uniform(orientation_min, orientation_max))

            direction = self.rng.normal(size=3)
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            axis = self.rng.normal(size=3)
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            target_position = initial_position + distance * direction
            delta_quaternion = np.asarray(
                p.getQuaternionFromAxisAngle(axis.tolist(), angle), dtype=np.float32
            )
            _, composed = p.multiplyTransforms(
                [0.0, 0.0, 0.0], initial_quaternion.tolist(),
                [0.0, 0.0, 0.0], delta_quaternion.tolist(),
            )
            target_quaternion = np.asarray(composed, dtype=np.float32)
            within_workspace = bool(
                0.25 <= target_position[0] <= 0.78
                and -0.45 <= target_position[1] <= 0.45
                and 0.18 <= target_position[2] <= 0.78
            )
            if not within_workspace:
                continue
            ik_validation = self.robot.check_pose_ik(
                target_position, target_quaternion,
                position_tolerance=self.position_tolerance,
                orientation_tolerance=self.orientation_tolerance,
                rest_pose=initial_q,
            )
            if not bool(ik_validation["reachable"]):
                continue
            candidate = np.asarray(ik_validation["q_ik"], dtype=np.float32)
            self._set_joint_state(candidate)
            p.performCollisionDetection(physicsClientId=self.client_id)
            target_self_collision = self._collision_events()["self_collision"]
            target_self_clearance = self._self_geometry().distance_min
            self._set_joint_state(initial_q, initial_qdot)
            if target_self_collision or target_self_clearance < self.d_self_safe:
                continue
            self.goal_position = target_position.astype(np.float32)
            self.goal_quaternion = target_quaternion
            self.goal_joint_positions = candidate
            self.goal_ik_validation = {
                **ik_validation, "q_ik": candidate.copy(),
            }
            self.goal_sample_attempts_last = attempt
            self.sampled_goal_scale = self.contract.goal_scale
            self.sampled_target_distance_m = distance
            self.sampled_target_orientation_rad = angle
            self.sampled_target_position_bin = position_bin
            self.sampled_target_orientation_bin = orientation_bin
            if position_bin >= 0 and orientation_bin >= 0:
                self.goal_accept_counts[flat_bin] += 1
                self.goal_resample_counts[flat_bin] += attempt - 1
            return
        self._set_joint_state(initial_q, initial_qdot)
        raise RuntimeError(
            "failed to sample a valid task-space goal pose in "
            f"{self.goal_sample_max_attempts} attempts"
        )

    def _create_obstacle(self) -> None:
        self._disable_obstacle()
        if self.contract.scene == "none":
            return
        for _ in range(100):
            if self.contract.scene == "static":
                position = np.asarray(
                    [
                        self.rng.uniform(0.22, 0.72),
                        self.rng.uniform(-0.48, 0.48),
                        self.rng.uniform(0.18, 0.62),
                    ],
                    dtype=np.float32,
                )
                velocity = np.zeros(3, dtype=np.float32)
                if np.linalg.norm(position - self.goal_position) < 0.15:
                    continue
            else:
                side = -1.0 if self.rng.random() < 0.5 else 1.0
                position = np.asarray(
                    [self.rng.uniform(0.22, 0.72), side * self.rng.uniform(0.42, 0.62), self.rng.uniform(0.18, 0.62)],
                    dtype=np.float32,
                )
                waypoint = np.asarray(
                    [self.rng.uniform(0.22, 0.72), -side * self.rng.uniform(0.24, 0.48), self.rng.uniform(0.18, 0.62)],
                    dtype=np.float32,
                )
                direction = waypoint - position
                velocity = (self.obstacle_speed * direction / np.linalg.norm(direction)).astype(np.float32)
            geometry = compute_thesis_geometry(
                self.robot.capsules(), position, velocity, self._closest_point_velocities(position),
                obstacle_radius=self.obstacle_radius, d_safe=self.d_safe,
            )
            if geometry.distance_min >= self.d_safe:
                self.obstacle_position = position
                self.obstacle_velocity = velocity
                self._create_obstacle_body()
                p.performCollisionDetection(physicsClientId=self.client_id)
                if not self._collision_events()["obstacle_collision"]:
                    return
                self._disable_obstacle()
        else:
            raise RuntimeError("failed to sample a valid obstacle in 100 attempts")

    def _create_obstacle_body(self) -> None:
        self.obstacle_id = self._obstacle_body_id
        p.resetBasePositionAndOrientation(
            self.obstacle_id,
            self.obstacle_position.tolist(),
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        p.resetBaseVelocity(
            self.obstacle_id,
            linearVelocity=self.obstacle_velocity.tolist(),
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.client_id,
        )

    def episode_state_dict(self) -> dict[str, Any]:
        q, qdot = self.robot.joint_state_exact()
        return {
            "contract": self.contract, "q": q, "qdot": qdot,
            "goal_position": self.goal_position.copy(), "goal_quaternion": self.goal_quaternion.copy(),
            "goal_joint_positions": self.goal_joint_positions.copy(),
            "goal_ik_validation": self.goal_ik_validation.copy(),
            "goal_sample_attempts_last": self.goal_sample_attempts_last,
            "sampled_goal_scale": self.sampled_goal_scale,
            "sampled_target_distance_m": self.sampled_target_distance_m,
            "sampled_target_orientation_rad": self.sampled_target_orientation_rad,
            "sampled_target_position_bin": self.sampled_target_position_bin,
            "sampled_target_orientation_bin": self.sampled_target_orientation_bin,
            "task_space_bin_order": self._task_space_bin_order.copy(),
            "task_space_bin_cursor": self._task_space_bin_cursor,
            "selected_task_space_position_bin": self._selected_task_space_position_bin,
            "selected_task_space_orientation_bin": self._selected_task_space_orientation_bin,
            "goal_proposal_counts": self.goal_proposal_counts.copy(),
            "goal_accept_counts": self.goal_accept_counts.copy(),
            "goal_resample_counts": self.goal_resample_counts.copy(),
            "task_space_bin_probabilities": (
                None if self.task_space_bin_probabilities is None
                else self.task_space_bin_probabilities.copy()
            ),
            "obstacle_position": self.obstacle_position.copy(), "obstacle_velocity": self.obstacle_velocity.copy(),
            "step_count": self.step_count, "obstacle_contact_seen": self.obstacle_contact_seen,
            "success_hold_count": self.success_hold_count,
            "previous_rho_position": self.previous_rho_position,
            "previous_rho_orientation": self.previous_rho_orientation,
            "previous_keypoint_distance": self.previous_keypoint_distance,
            "rng_state": self.rng.bit_generator.state,
        }

    def restore_episode_state(self, state: dict[str, Any]) -> np.ndarray:
        """Rebuild a boundary state saved after a complete control transition."""
        self.contract = state["contract"]
        self._disable_obstacle()
        self._set_joint_state(state["q"], state["qdot"])
        self.goal_position = np.asarray(state["goal_position"], dtype=np.float32)
        self.goal_quaternion = np.asarray(state["goal_quaternion"], dtype=np.float32)
        self.goal_joint_positions = np.asarray(
            state.get("goal_joint_positions", np.zeros(6)), dtype=np.float32
        )
        self.goal_ik_validation = state.get("goal_ik_validation", {}).copy()
        self.goal_sample_attempts_last = int(state.get("goal_sample_attempts_last", 0))
        self.sampled_goal_scale = float(
            state.get("sampled_goal_scale", self.contract.goal_scale)
        )
        self.sampled_target_distance_m = float(state.get("sampled_target_distance_m", 0.0))
        self.sampled_target_orientation_rad = float(state.get("sampled_target_orientation_rad", 0.0))
        self.sampled_target_position_bin = int(state.get("sampled_target_position_bin", -1))
        self.sampled_target_orientation_bin = int(
            state.get("sampled_target_orientation_bin", -1)
        )
        self._task_space_bin_order = np.asarray(
            state.get("task_space_bin_order", np.empty(0)), dtype=np.int64
        ).copy()
        self._task_space_bin_cursor = int(state.get("task_space_bin_cursor", 0))
        self._selected_task_space_position_bin = int(
            state.get("selected_task_space_position_bin", -1)
        )
        self._selected_task_space_orientation_bin = int(
            state.get("selected_task_space_orientation_bin", -1)
        )
        total_bins = self.task_space_position_bins * self.task_space_orientation_bins
        self.goal_proposal_counts = np.asarray(
            state.get("goal_proposal_counts", np.zeros(total_bins)), dtype=np.int64
        ).copy()
        self.goal_accept_counts = np.asarray(
            state.get("goal_accept_counts", np.zeros(total_bins)), dtype=np.int64
        ).copy()
        self.goal_resample_counts = np.asarray(
            state.get("goal_resample_counts", np.zeros(total_bins)), dtype=np.int64
        ).copy()
        probabilities = state.get("task_space_bin_probabilities")
        self.task_space_bin_probabilities = (
            None if probabilities is None
            else np.asarray(probabilities, dtype=np.float64).copy()
        )
        self.obstacle_position = np.asarray(state["obstacle_position"], dtype=np.float32)
        self.obstacle_velocity = np.asarray(state["obstacle_velocity"], dtype=np.float32)
        if self.contract.scene != "none": self._create_obstacle_body()
        self.step_count = int(state["step_count"])
        self.success_hold_count = int(state.get("success_hold_count", 0))
        self.obstacle_contact_seen = bool(state["obstacle_contact_seen"])
        self.previous_rho_position = float(state["previous_rho_position"])
        self.previous_rho_orientation = float(state["previous_rho_orientation"])
        saved_keypoint_distance = state.get("previous_keypoint_distance")
        self.previous_keypoint_distance = float(saved_keypoint_distance or 0.0)
        self.rng.bit_generator.state = state["rng_state"]
        p.performCollisionDetection(physicsClientId=self.client_id)
        observation, info, *_ = self._observe()
        if saved_keypoint_distance is None:
            self.previous_keypoint_distance = float(
                np.mean(info["keypoint_distances"])
            )
        return observation

    def reset_bad_state(self, center: dict[str, Any], contract: dict[str, Any]):
        """Restart a perturbed, reachable local state without resetting its clock."""
        self.configure_episode(**contract)
        if self.contract.scene != "none":
            raise ValueError("bad-state starts support obstacle-free S0 only")
        current = self.episode_state_dict()
        snapshot = dict(center["state"])
        # Keep the training RNG and balanced sampler stream, not the evaluation's.
        for key in current:
            if key.startswith(("task_space_", "selected_task_space_", "goal_proposal_",
                               "goal_accept_", "goal_resample_")) or key in ("contract", "rng_state"):
                snapshot[key] = current[key]
        if not 0 <= int(snapshot["step_count"]) < self.horizon:
            raise ValueError("bad-state start has no remaining episode budget")
        reference = np.asarray(center["observation"])
        for _ in range(64):
            snapshot["rng_state"] = self.rng.bit_generator.state
            self.restore_episode_state(snapshot)
            q = np.asarray(snapshot["q"]) + self.rng.uniform(-.015, .015, 6)
            qdot = np.asarray(snapshot["qdot"]) + self.rng.uniform(-.02, .02, 6)
            lower, upper = self.robot.joint_lower_limits, self.robot.joint_upper_limits
            if (np.any(q < lower) or np.any(q > upper) or
                    np.any(np.abs(qdot) > self.velocity_limits)):
                continue
            self.goal_position = (np.asarray(snapshot["goal_position"]) +
                                  self.rng.uniform(-.004, .004, 3)).astype(np.float32)
            rotation = p.getQuaternionFromEuler(self.rng.uniform(-.012, .012, 3).tolist())
            self.goal_quaternion = np.asarray(p.multiplyTransforms(
                [0., 0., 0.], snapshot["goal_quaternion"],
                [0., 0., 0.], rotation,
            )[1], dtype=np.float32)
            self._set_joint_state(q, qdot)
            p.performCollisionDetection(physicsClientId=self.client_id)
            if self._collision_events()["self_collision"]:
                continue
            observation, info, geometry, self_geometry, *_ = self._observe()
            if (abs(info["orientation_error_norm"] - center["rho_R"]) > .10 or
                    abs(info["goal_error_norm"] - center["rho_p"]) > .12 or
                    any(np.sqrt(np.mean((observation[sl] - reference[sl]) ** 2)) > bound
                        for sl, bound in ((slice(0, 6), .25), (slice(6, 12), .35),
                                          (slice(21, 75), .18)))):
                continue
            self.success_hold_count = 0
            self.previous_rho_position = float(info["goal_error_norm"])
            self.previous_rho_orientation = float(info["orientation_error_norm"])
            self.previous_keypoint_distance = float(np.mean(info["keypoint_distances"]))
            info.update(self._contract_info())
            info.update(self._geometry_info(geometry))
            info.update(self._self_geometry_info(self_geometry))
            return observation, info
        raise RuntimeError("could not sample a valid perturbed bad-state neighborhood")

    def _advance_obstacle(self) -> None:
        if self.obstacle_id is None or self.contract.scene != "dynamic":
            return
        position = (self.obstacle_position + self.obstacle_velocity * self.physics_dt).astype(np.float32)
        velocity = self.obstacle_velocity.copy()
        lows = np.asarray([0.22, -0.62, 0.18], dtype=np.float32)
        highs = np.asarray([0.72, 0.62, 0.62], dtype=np.float32)
        for axis in range(3):
            if position[axis] < lows[axis]:
                position[axis] = 2.0 * lows[axis] - position[axis]
                velocity[axis] = abs(velocity[axis])
            elif position[axis] > highs[axis]:
                position[axis] = 2.0 * highs[axis] - position[axis]
                velocity[axis] = -abs(velocity[axis])
        self.obstacle_position, self.obstacle_velocity = position, velocity
        p.resetBasePositionAndOrientation(
            self.obstacle_id,
            position.tolist(),
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        p.resetBaseVelocity(
            self.obstacle_id,
            linearVelocity=velocity.tolist(),
            physicsClientId=self.client_id,
        )

    def _observe(self):
        q, qdot = self.robot.joint_state()
        ee_position, ee_quaternion, ee_linear_velocity, ee_angular_velocity = (
            self.robot.end_effector_kinematics()
        )
        position_error, orientation_error = pose_error(
            self.goal_position, self.goal_quaternion, ee_position, ee_quaternion
        )
        keypoint_errors = pose_keypoint_errors(
            self.goal_position, self.goal_quaternion,
            ee_position, ee_quaternion,
            cube_side_m=self.keypoint_cube_side_m,
        )
        keypoint_distances = np.linalg.norm(keypoint_errors, axis=1).astype(np.float32)
        ee_jacobian = None
        keypoint_jacobian = None
        jacobian_clip_ratio = 0.0
        if self.keypoint_jacobian_pose:
            ee_jacobian = self.robot.end_effector_jacobian()
            keypoint_jacobian = keypoint_position_jacobian(
                ee_jacobian, ee_quaternion,
                cube_side_m=self.keypoint_cube_side_m,
            )
            jacobian_clip_ratio = float(np.mean(
                np.abs(keypoint_jacobian / self.keypoint_jacobian_scale) > 1.0
            ))
        geometry = self._geometry()
        self_geometry = self._self_geometry()
        observation = build_thesis_observation(
            q=q,
            qdot=qdot,
            joint_lower=self.robot.joint_lower_limits,
            joint_upper=self.robot.joint_upper_limits,
            joint_velocity_scale=self.qdot_observation_scale,
            ee_position=ee_position,
            ee_quaternion=ee_quaternion,
            ee_linear_velocity=ee_linear_velocity,
            ee_angular_velocity=ee_angular_velocity,
            ee_linear_velocity_scale=self.ee_linear_velocity_scale,
            ee_angular_velocity_scale=self.ee_angular_velocity_scale,
            goal_position=self.goal_position,
            goal_quaternion=self.goal_quaternion,
            position_error=position_error,
            orientation_error=orientation_error,
            obstacle_present=self.obstacle_id is not None,
            obstacle_position=self.obstacle_position,
            obstacle_velocity=self.obstacle_velocity,
            geometry=geometry,
            self_geometry=self_geometry,
            include_orientation_error_vector=self.include_orientation_error_vector,
            keypoint_jacobian_pose=self.keypoint_jacobian_pose,
            hybrid_explicit_pose_error=self.hybrid_explicit_pose_error,
            keypoint_cube_side_m=self.keypoint_cube_side_m,
            keypoint_error_scale_m=self.keypoint_error_scale_m,
            keypoint_jacobian=keypoint_jacobian,
            keypoint_jacobian_scale=self.keypoint_jacobian_scale,
        )
        info = {
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
            "orientation_error_vector": orientation_error.copy(),
            "keypoint_errors": keypoint_errors.copy(),
            "keypoint_distances": keypoint_distances.copy(),
            "jacobian_clip_ratio": jacobian_clip_ratio,
            # The hybrid/keypoint observation already computes this matrix.
            # Exposing it in ``info`` lets offline evaluators inspect how the
            # executed joint velocity is used without paying for a second
            # PyBullet Jacobian query.  Parallel evaluators opt in explicitly
            # through ``step_info_keys``, so normal training IPC is unchanged.
            "ee_jacobian": None if ee_jacobian is None else ee_jacobian.copy(),
        }
        return observation, info, geometry, self_geometry, position_error, orientation_error

    def pose_crossing_diagnostics(self) -> dict[str, object]:
        """Return expensive kinematic diagnostics for the current state.

        This is intentionally an on-demand evaluation hook: computing SVDs and
        self geometry on every training transition would slow the normal SAC
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
        keypoint_errors = pose_keypoint_errors(
            self.goal_position, self.goal_quaternion,
            ee_position, ee_quaternion,
            cube_side_m=self.keypoint_cube_side_m,
        )
        keypoint_distance = float(np.mean(np.linalg.norm(keypoint_errors, axis=1)))
        ee_jacobian = np.asarray(
            self.robot.end_effector_jacobian(), dtype=np.float64,
        )
        keypoint_jacobian = np.asarray(
            keypoint_position_jacobian(
                ee_jacobian, ee_quaternion,
                cube_side_m=self.keypoint_cube_side_m,
            ),
            dtype=np.float64,
        )
        ee_singular_values = np.linalg.svd(
            ee_jacobian, compute_uv=False,
        )
        keypoint_singular_values = np.linalg.svd(
            keypoint_jacobian, compute_uv=False,
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
            "keypoint_distance": keypoint_distance,
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
            "keypoint_jacobian_sigma_min": float(
                np.min(keypoint_singular_values)
            ),
            "keypoint_jacobian_sigma_max": float(
                np.max(keypoint_singular_values)
            ),
            "keypoint_jacobian_condition_number": float(
                np.max(keypoint_singular_values)
                / max(float(np.min(keypoint_singular_values)), 1.0e-12)
            ),
            "keypoint_manipulability": float(
                np.prod(keypoint_singular_values)
            ),
            "keypoint_log_manipulability": float(np.sum(np.log(
                np.maximum(keypoint_singular_values, 1.0e-12)
            ))),
            "self_distance": self_distance,
        }

    def _geometry(self) -> ThesisGeometry | None:
        if self.obstacle_id is None:
            return None
        return compute_thesis_geometry(
            self.robot.capsules(),
            self.obstacle_position,
            self.obstacle_velocity,
            self._closest_point_velocities(self.obstacle_position),
            obstacle_radius=self.obstacle_radius,
            d_safe=self.d_safe,
        )

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

    def _project_self_safe_command(
        self, target_command: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        default = {
            "self_projection_enabled": self.self_projection_enabled,
            "self_projection_intervened": False,
            "self_projection_infeasible": False,
            "self_projection_correction_norm": 0.0,
            "self_projection_max_slack": 0.0,
            "self_projection_active_constraints": 0,
            "self_projection_solve_time_ms": 0.0,
        }
        target = np.asarray(target_command, dtype=np.float32)
        if not self.self_projection_enabled:
            return target, default
        base_distances, _ = self._self_pair_geometry()
        active = np.flatnonzero(
            base_distances <= self.self_projection_activation_distance
        )
        if len(active) == 0:
            return target, default

        q, qdot = self.robot.joint_state()
        jacobians = np.zeros((len(active), 6), dtype=np.float32)
        try:
            for joint_offset, joint_id in enumerate(self.robot.joint_ids):
                p.resetJointState(
                    self.robot_id, joint_id,
                    float(q[joint_offset] + self.self_projection_fd_epsilon),
                    targetVelocity=float(qdot[joint_offset]),
                    physicsClientId=self.client_id,
                )
                perturbed, _ = self._self_pair_geometry()
                jacobians[:, joint_offset] = (
                    perturbed[active] - base_distances[active]
                ) / self.self_projection_fd_epsilon
                p.resetJointState(
                    self.robot_id, joint_id, float(q[joint_offset]),
                    targetVelocity=float(qdot[joint_offset]),
                    physicsClientId=self.client_id,
                )
        finally:
            self._set_joint_state(q, qdot)

        constraints = [
            distance_rate_constraint(
                jacobian, float(distance), self.d_self_safe, self.self_projection_gain,
            )
            for jacobian, distance in zip(jacobians, base_distances[active], strict=True)
            if float(np.linalg.norm(jacobian)) > 1.0e-7
        ]
        if not constraints:
            return target, default
        matrix = np.stack([row for row, _ in constraints])
        bound = np.asarray([value for _, value in constraints], dtype=np.float32)
        result = self.self_safety_qp.solve(
            target, matrix, bound, -self.action_scale, self.action_scale,
        )
        return result.command, {
            **default,
            "self_projection_intervened": result.intervened,
            "self_projection_infeasible": result.infeasible,
            "self_projection_correction_norm": result.correction_norm,
            "self_projection_max_slack": result.max_slack,
            "self_projection_active_constraints": result.active_constraints,
            "self_projection_solve_time_ms": result.solve_time_ms,
        }

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

    def _closest_point_velocities(self, obstacle_position: np.ndarray) -> np.ndarray:
        capsules = self.robot.capsules()
        velocities = []
        for capsule, spec in zip(capsules, self.robot.capsule_model.specs):
            _, fraction = closest_point_on_segment(obstacle_position, capsule.start, capsule.end)
            start_velocity = self._link_point_velocity(spec.parent_link_name, spec.parent_offset)
            end_velocity = self._link_point_velocity(spec.child_link_name, spec.child_offset)
            velocities.append((1.0 - fraction) * start_velocity + fraction * end_velocity)
        return np.asarray(velocities, dtype=np.float32)

    def _link_point_velocity(self, link_name: str, local_offset: np.ndarray) -> np.ndarray:
        link_id = self.robot.capsule_model.link_name_to_id[link_name]
        if link_id < 0:
            return np.zeros(3, dtype=np.float32)
        state = p.getLinkState(
            self.robot_id,
            link_id,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        rotation = np.asarray(p.getMatrixFromQuaternion(state[5]), dtype=np.float32).reshape(3, 3)
        radius = rotation @ np.asarray(local_offset, dtype=np.float32)
        linear = np.asarray(state[6], dtype=np.float32)
        angular = np.asarray(state[7], dtype=np.float32)
        return linear + np.cross(angular, radius)

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
        names = [
            "base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link",
            "wrist_1_link", "wrist_2_link", "wrist_3_link",
        ]
        self._self_link_order = [self._link_name_to_id[name] for name in names]
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
        if self.obstacle_id is not None:
            obstacle_collision = any(
                point[8] <= 0.0 and point[3] in self._collision_link_ids
                for point in p.getContactPoints(
                    bodyA=self.robot_id,
                    bodyB=self.obstacle_id,
                    physicsClientId=self.client_id,
                )
            )
        self_collision = False
        order = {link_id: index for index, link_id in enumerate(self._self_link_order)}
        for point in p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id, physicsClientId=self.client_id):
            first, second = int(point[3]), int(point[4])
            if point[8] > 0.0 or first not in order or second not in order:
                continue
            if abs(order[first] - order[second]) > 1:
                self_collision = True
                break
        return {
            "obstacle_collision": bool(obstacle_collision),
            "self_collision": bool(self_collision),
            "environment_collision": False,
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
        selected_bin = -1
        if self.sampled_target_position_bin >= 0 and self.sampled_target_orientation_bin >= 0:
            selected_bin = (
                self.sampled_target_position_bin * self.task_space_orientation_bins
                + self.sampled_target_orientation_bin
            )
        selected_acceptance = 0.0
        selected_mean_resamples = 0.0
        if selected_bin >= 0:
            selected_acceptance = float(
                self.goal_accept_counts[selected_bin]
                / max(self.goal_proposal_counts[selected_bin], 1)
            )
            selected_mean_resamples = float(
                self.goal_resample_counts[selected_bin]
                / max(self.goal_accept_counts[selected_bin], 1)
            )
        return {
            "scene": self.contract.scene,
            "xi_scene": self.contract.xi,
            "strict_obstacle_collision": self.contract.strict_obstacle_collision,
            "goal_scale": self.contract.goal_scale,
            "sampled_goal_scale": self.sampled_goal_scale,
            "sampled_target_distance_m": self.sampled_target_distance_m,
            "sampled_target_orientation_rad": self.sampled_target_orientation_rad,
            "sampled_target_position_bin": self.sampled_target_position_bin,
            "sampled_target_orientation_bin": self.sampled_target_orientation_bin,
            "orientation_scale": self.contract.orientation_scale,
            "position_tolerance": self.contract.position_tolerance,
            "orientation_tolerance": self.contract.orientation_tolerance,
            "lambda_self": self.contract.lambda_self,
            "terminate_on_success": self.contract.terminate_on_success,
            "obstacle_enabled": self.obstacle_id is not None,
            "goal_proposal_counts": self.goal_proposal_counts.copy(),
            "goal_accept_counts": self.goal_accept_counts.copy(),
            "goal_resample_counts": self.goal_resample_counts.copy(),
            "goal_acceptance_rates": np.divide(
                self.goal_accept_counts,
                np.maximum(self.goal_proposal_counts, 1),
            ).astype(np.float64),
            "goal_acceptance_rate": selected_acceptance,
            "goal_mean_resamples": np.divide(
                self.goal_resample_counts,
                np.maximum(self.goal_accept_counts, 1),
            ).astype(np.float64),
            "goal_mean_resamples": selected_mean_resamples,
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
