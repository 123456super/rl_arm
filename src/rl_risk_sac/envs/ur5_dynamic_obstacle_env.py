from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from rl_risk_sac.robots.ur5_capsules import CapsuleState, UR5CapsuleModel
from rl_risk_sac.utils.predictive_risk import (
    ObstacleStateEstimate,
    PredictiveLinkRisk,
    PredictiveRiskConfig,
    compute_predictive_link_risk,
)
from rl_risk_sac.utils.risk import LinkRisk, RiskConfig, compute_link_risk
from rl_risk_sac.utils.safety_filter import (
    LinearVelocityConstraints,
    SafetyFilterConfig,
    SafetyFilterInput,
    SafetyFilterResult,
    SafetyFilterStatus,
    filter_joint_velocity,
)


METHODS = {"ee_fixed", "link_fixed", "ldrc_fixed", "ldrc_adaptive"}
RISK_REPRESENTATIONS = {"current", "predictive", "robust_predictive"}


class UR5DynamicObstacleEnv(gym.Env):
    """UR5-like joint-velocity reaching task with one dynamic spherical obstacle."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: dict[str, Any], method: str = "ldrc_adaptive", render_mode: str | None = None):
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {sorted(METHODS)}")
        self.config = config
        self.method = method
        self.render_mode = render_mode

        env_cfg = config["env"]
        risk_cfg = config["risk"]
        smoothing_cfg = config["smoothing"]
        self.reward_cfg = config["reward"]

        self.time_step = float(env_cfg["time_step"])
        self.control_dt = float(env_cfg["control_dt"])
        self.sim_substeps = max(1, int(round(self.control_dt / self.time_step)))
        self.max_episode_steps = int(env_cfg["max_episode_steps"])
        self.action_scale = float(env_cfg["action_scale"])
        self.fixed_beta = float(env_cfg["fixed_beta"])
        self.success_tolerance = float(env_cfg["success_tolerance"])
        self.workspace = env_cfg["workspace"]
        self.obstacle_cfg = env_cfg["obstacle"]
        self.goal_cfg = env_cfg["goal"]
        self.robot_cfg = config.get("robot", {})
        self.observation_cfg = env_cfg.get("observation", {})
        self.visual_cfg = env_cfg.get("visual", {})
        self.execution_cfg = env_cfg.get("execution", {})
        self.safety_filter_cfg = env_cfg.get("safety_filter", {})
        self.safety_filter_enabled = bool(self.safety_filter_cfg.get("enabled", False))
        self.obstacle_enabled = bool(self.obstacle_cfg.get("enabled", True))
        self.obstacle_scenario = str(self.obstacle_cfg.get("scenario", "random"))

        self.beta_min = float(smoothing_cfg["beta_min"])
        self.beta_max = float(smoothing_cfg["beta_max"])
        self.risk_high = float(smoothing_cfg["risk_high"])
        self.lambda_beta = float(smoothing_cfg["lambda_beta"])

        weights = risk_cfg["weights"]
        self.risk_config = RiskConfig(
            d_safe=float(risk_cfg["d_safe"]),
            sigma_d=float(risk_cfg["sigma_d"]),
            v_max=float(risk_cfg["v_max"]),
            ttc_max=float(risk_cfg["ttc_max"]),
            tau=float(risk_cfg["tau"]),
            eps=float(risk_cfg["eps"]),
            eps_v=float(risk_cfg["eps_v"]),
            w_distance=float(weights["distance"]),
            w_velocity=float(weights["velocity"]),
            w_ttc=float(weights["ttc"]),
        )
        self.cost_cfg = risk_cfg["cost"]
        self.risk_representation = str(risk_cfg.get("representation", "current"))
        if self.risk_representation not in RISK_REPRESENTATIONS:
            raise ValueError(
                f"Unknown risk.representation {self.risk_representation!r}; "
                f"expected one of {sorted(RISK_REPRESENTATIONS)}"
            )

        self.rng = np.random.default_rng(int(config.get("seed", 42)))
        self.physics_client_id = p.connect(p.GUI if render_mode == "human" or env_cfg.get("gui") else p.DIRECT)
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client_id)
        p.setGravity(*env_cfg["gravity"], physicsClientId=self.physics_client_id)

        self.repo_root = Path(__file__).resolve().parents[3]
        robot_urdf = Path(self.robot_cfg["urdf"])
        self.robot_urdf = robot_urdf if robot_urdf.is_absolute() else self.repo_root / robot_urdf
        self.capsule_model = UR5CapsuleModel(self.robot_cfg.get("capsules"))
        self.joint_ids: list[int] = []
        self.jacobian_joint_ids: list[int] = []
        self.control_jacobian_columns: list[int] = []
        self.tool_link_id = -1
        self.joint_count = len(self.robot_cfg["joint_names"])

        self.robot_id: int | None = None
        self.obstacle_id: int | None = None
        self.goal_marker_id: int | None = None
        self.prev_capsules: list[CapsuleState] | None = None
        self.prev_goal_error_norm = 0.0
        self.prev_qdot_cmd = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_joint_acc = np.zeros(self.joint_count, dtype=np.float32)
        self.beta = self.fixed_beta
        self.step_count = 0
        self.goal = np.zeros(3, dtype=np.float32)
        self.obstacle_center = np.zeros(3, dtype=np.float32)
        self.obstacle_velocity = np.zeros(3, dtype=np.float32)
        self.last_risk = None
        self.last_policy_risk = None
        self.last_info: dict[str, Any] = {}
        self.sim_time_s = 0.0
        self.predictive_risk_config: PredictiveRiskConfig | None = None
        self.safety_filter_config: SafetyFilterConfig | None = None
        self.last_filter_phase_times_s: dict[str, float] = {}
        self.obstacle_state_estimate_override: ObstacleStateEstimate | None = None
        self.recovery_active = False
        self.recovery_triggered = False
        self.recovery_steps = 0
        self.recovery_success = False
        self.recovery_initially_unsafe = False
        self.recovery_initial_h_min_m = float("nan")

        obs_dim = self.joint_count * 3 + self.capsule_model.count * 7 + 7
        obs_bound = float(self.observation_cfg["space_bound"])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.joint_count,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_bound, high=obs_bound, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        p.resetSimulation(physicsClientId=self.physics_client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client_id)
        p.setGravity(*self.config["env"]["gravity"], physicsClientId=self.physics_client_id)
        self._create_floor()
        self.robot_id = p.loadURDF(
            str(self.robot_urdf),
            basePosition=self.robot_cfg["base_position"],
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        self._resolve_robot_references()
        self._configure_safety_filter()
        self._reset_robot()
        self.goal = self._sample_goal()
        self.obstacle_center, self.obstacle_velocity = self._sample_obstacle()
        self.obstacle_id = self._create_obstacle(self.obstacle_center) if self.obstacle_enabled else None
        self.goal_marker_id = self._create_goal_marker(self.goal)

        self.prev_qdot_cmd = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_joint_acc = np.zeros(self.joint_count, dtype=np.float32)
        self.beta = self.fixed_beta if self.method.endswith("fixed") else self.beta_min
        self.step_count = 0
        self.sim_time_s = 0.0
        self.obstacle_state_estimate_override = None
        self.recovery_active = False
        self.recovery_triggered = False
        self.recovery_steps = 0
        self.recovery_success = False
        self.recovery_initially_unsafe = False
        self.recovery_initial_h_min_m = float("nan")
        if self.safety_filter_enabled:
            initial_predictive_risk = self._compute_predictive_risk()
            if initial_predictive_risk.usable:
                self.recovery_initial_h_min_m = float(np.min(initial_predictive_risk.safety_functions_m))
                self.recovery_initially_unsafe = self.recovery_initial_h_min_m <= 0.0
        self.prev_capsules = self._capsules()
        self.prev_goal_error_norm = float(np.linalg.norm(self._goal_error()))
        obs, info = self._get_obs_and_info()
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        pre_risk = self._compute_policy_risk(self._compute_risk())
        predictive_risk_for_scaling = None
        precomputed_risk_time_s = 0.0
        risk_speed_scale = 1.0
        filter_cycle_started_at = perf_counter() if self.safety_filter_enabled else None
        if self.safety_filter_enabled and bool(self.safety_filter_cfg.get("risk_speed_scaling_enabled", False)):
            phase_started_at = perf_counter()
            predictive_risk_for_scaling = self._compute_predictive_risk()
            precomputed_risk_time_s += perf_counter() - phase_started_at
            risk_speed_scale = self._risk_speed_scale(predictive_risk_for_scaling)
        if self.safety_filter_enabled and bool(self.safety_filter_cfg.get("recovery_mode_enabled", False)):
            if predictive_risk_for_scaling is None:
                phase_started_at = perf_counter()
                predictive_risk_for_scaling = self._compute_predictive_risk()
                precomputed_risk_time_s += perf_counter() - phase_started_at
            self._update_recovery_state(predictive_risk_for_scaling)
        qdot_policy = self.action_scale * action
        if self.method.endswith("adaptive"):
            beta = self._adaptive_beta(pre_risk.risk_global)
        else:
            beta = self.fixed_beta
        qdot_cmd = beta * qdot_policy + (1.0 - beta) * self.prev_qdot_cmd
        if predictive_risk_for_scaling is not None:
            qdot_cmd *= risk_speed_scale
        recovery_command = None
        if self.recovery_active and predictive_risk_for_scaling is not None:
            recovery_command = self._recovery_command(predictive_risk_for_scaling)
            qdot_cmd = recovery_command
        qdot_cmd = np.clip(qdot_cmd, -self.action_scale, self.action_scale).astype(np.float32)
        qdot_requested = qdot_cmd.copy()
        filter_result = None
        predictive_risk = None
        filter_solve_time_s = None
        filter_link_diagnostics: dict[str, Any] = {}
        if self.safety_filter_enabled:
            if predictive_risk_for_scaling is None:
                filter_result, predictive_risk, filter_solve_time_s = self._filter_command(
                    qdot_requested,
                    cycle_started_at=filter_cycle_started_at,
                )
            else:
                filter_result, predictive_risk, filter_solve_time_s = self._filter_command(
                    qdot_requested,
                    predictive_risk_for_scaling,
                    cycle_started_at=filter_cycle_started_at,
                    precomputed_risk_time_s=precomputed_risk_time_s,
                )
        if filter_result is not None:
            qdot_cmd = filter_result.command_joint_velocity_radps.astype(np.float32)
            if bool(self.safety_filter_cfg.get("diagnostic_logging", False)):
                filter_link_diagnostics = self._filter_link_diagnostics(predictive_risk, qdot_cmd)

        for _ in range(self.sim_substeps):
            self._move_obstacle(self.time_step)
            p.setJointMotorControlArray(
                self.robot_id,
                self.joint_ids,
                p.VELOCITY_CONTROL,
                targetVelocities=qdot_cmd.tolist(),
                forces=[float(self.execution_cfg["joint_motor_force"])] * self.joint_count,
                physicsClientId=self.physics_client_id,
            )
            p.stepSimulation(physicsClientId=self.physics_client_id)

        self.sim_time_s += self.sim_substeps * self.time_step

        self.step_count += 1
        obs, info = self._get_obs_and_info()
        current_risk = self.last_risk
        policy_risk = self.last_policy_risk
        (
            collision_capsule_overlap,
            collision_pybullet_contact,
            collision_contact_link_indices,
            collision_contact_link_names,
            collision_min_contact_distance,
        ) = self._collision_sources(current_risk.d_min)
        collision = collision_capsule_overlap or collision_pybullet_contact
        success = info["goal_error_norm"] < self.success_tolerance and current_risk.d_min > self.risk_config.d_safe
        terminated = bool(collision or success)
        truncated = self.step_count >= self.max_episode_steps

        reward = self._reward(qdot_cmd, success, collision)
        cost = self._cost(policy_risk, collision, bool(info["safety_violation"]))
        if self.method in {"ee_fixed", "link_fixed"}:
            reward -= float(self.config["sac"]["fixed_risk_penalty"]) * cost

        joint_acc = (qdot_cmd - self.prev_qdot_cmd) / self.control_dt
        jerk = (joint_acc - self.prev_joint_acc) / self.control_dt
        info.update(
            {
                "reward": float(reward),
                "cost": float(cost),
                "collision": bool(collision),
                "collision_capsule_overlap": bool(collision_capsule_overlap),
                "collision_pybullet_contact": bool(collision_pybullet_contact),
                "collision_contact_link_indices": collision_contact_link_indices,
                "collision_contact_link_names": collision_contact_link_names,
                "collision_min_contact_distance": collision_min_contact_distance,
                "success": bool(success),
                "qdot_cmd": qdot_cmd.copy(),
                "joint_acc": joint_acc.copy(),
                "joint_jerk": jerk.copy(),
                "beta": float(beta),
                "qdot_requested": qdot_requested.copy(),
                "risk_speed_scale": float(risk_speed_scale),
                "risk_speed_h_min_m": (
                    float(np.min(predictive_risk_for_scaling.safety_functions_m))
                    if predictive_risk_for_scaling is not None and predictive_risk_for_scaling.usable
                    else float("nan")
                ),
                "recovery_active": bool(self.recovery_active),
                "recovery_triggered": bool(self.recovery_triggered),
                "recovery_success": bool(self.recovery_success),
                "recovery_initially_unsafe": bool(self.recovery_initially_unsafe),
                "recovery_initial_h_min_m": float(self.recovery_initial_h_min_m),
                "recovery_command_norm": (
                    float(np.linalg.norm(recovery_command)) if recovery_command is not None else 0.0
                ),
            }
        )
        if filter_result is not None:
            info.update(
                self._filter_info(
                    filter_result,
                    predictive_risk,
                    filter_solve_time_s,
                    self.last_filter_phase_times_s,
                )
            )
            info.update(filter_link_diagnostics)
        self.prev_qdot_cmd = qdot_cmd
        self.prev_joint_acc = joint_acc
        self.prev_capsules = self._capsules()
        self.prev_goal_error_norm = info["goal_error_norm"]
        self.beta = float(beta)
        self.last_info = info
        return obs, float(reward), float(cost), terminated, truncated, info

    def close(self) -> None:
        if p.isConnected(self.physics_client_id):
            p.disconnect(self.physics_client_id)

    def get_metrics(self) -> dict[str, Any]:
        return dict(self.last_info)

    def set_obstacle_state_estimate_override(self, estimate: ObstacleStateEstimate | None) -> None:
        """Set a perception estimate for the next safety-filtered control steps.

        ``None`` restores the timestamped PyBullet ground-truth estimate.  The
        override exists for P2 fault injection and will also be the hand-off
        point for a real perception adapter; it does not affect stage-one risk
        observations or collision labels.
        """

        self.obstacle_state_estimate_override = estimate

    def _get_obs_and_info(self) -> tuple[np.ndarray, dict[str, Any]]:
        q, q_dot = self._joint_state()
        ee_pos, ee_vel = self._end_effector_state()
        goal_error = self.goal - ee_pos
        goal_velocity_error = -ee_vel
        current_risk = self._compute_risk()
        policy_risk = self._compute_policy_risk(current_risk)
        self.last_risk = current_risk
        self.last_policy_risk = policy_risk

        obs = np.concatenate(
            [
                q / np.pi,
                q_dot / max(self.action_scale, 1e-6),
                goal_error,
                goal_velocity_error,
                np.clip(
                    policy_risk.distances,
                    float(self.observation_cfg["distance_clip"][0]),
                    float(self.observation_cfg["distance_clip"][1]),
                ),
                policy_risk.directions.reshape(-1),
                np.clip(policy_risk.approach_velocities / max(self.risk_config.v_max, 1e-6), 0.0, 1.0),
                policy_risk.ttc / max(self.risk_config.ttc_max, 1e-6),
                policy_risk.risks,
                self.prev_qdot_cmd / max(self.action_scale, 1e-6),
                np.array([self.beta], dtype=np.float32),
            ]
        ).astype(np.float32)
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(f"Observation shape {obs.shape} does not match {self.observation_space.shape}")

        info = {
            "goal": self.goal.copy(),
            "ee_pos": ee_pos.copy(),
            "goal_error_norm": float(np.linalg.norm(goal_error)),
            "obstacle_enabled": bool(self.obstacle_enabled),
            "obstacle_center": self.obstacle_center.copy(),
            "obstacle_velocity": self.obstacle_velocity.copy(),
            "risk_global": float(current_risk.risk_global),
            "d_min": float(current_risk.d_min),
            "closest_link": int(current_risk.closest_link),
            "safety_violation": bool(current_risk.d_min < self.risk_config.d_safe),
            "risk_body": current_risk.risks.copy(),
            "policy_risk_representation": self.risk_representation,
            "policy_risk_global": float(policy_risk.risk_global),
            "policy_risk_d_min": float(policy_risk.d_min),
            "recovery_initially_unsafe": bool(self.recovery_initially_unsafe),
            "recovery_initial_h_min_m": float(self.recovery_initial_h_min_m),
        }
        return obs, info

    def _compute_risk(self):
        if not self.obstacle_enabled:
            return self._zero_risk()
        use_ee_only = self.method == "ee_fixed"
        return compute_link_risk(
            capsules=self._capsules(),
            prev_capsules=self.prev_capsules,
            obstacle_center=self.obstacle_center,
            obstacle_velocity=self.obstacle_velocity,
            obstacle_radius=float(self.obstacle_cfg["radius"]),
            dt=self.control_dt,
            config=self.risk_config,
            use_end_effector_only=use_ee_only,
        )

    def _compute_policy_risk(self, current_risk: LinkRisk) -> LinkRisk:
        if self.risk_representation == "current" or not self.obstacle_enabled:
            return current_risk

        predictive_risk = self._compute_predictive_risk()
        if predictive_risk.requires_safe_stop:
            return LinkRisk(
                closest_points=current_risk.closest_points,
                distances=np.full(self.capsule_model.count, -np.inf, dtype=np.float32),
                directions=current_risk.directions,
                link_velocities=current_risk.link_velocities,
                approach_velocities=current_risk.approach_velocities,
                ttc=np.zeros(self.capsule_model.count, dtype=np.float32),
                risks=np.ones(self.capsule_model.count, dtype=np.float32),
                risk_global=1.0,
                d_min=float("-inf"),
                closest_link=0,
            )

        distances = (
            predictive_risk.robust_distances_m
            if self.risk_representation == "robust_predictive"
            else predictive_risk.predicted_distances_m
        ).astype(np.float32)
        distance_risks = np.clip(
            np.exp(-(distances - self.risk_config.d_safe) / self.risk_config.sigma_d),
            0.0,
            1.0,
        )
        velocity_risks = np.clip(current_risk.approach_velocities / self.risk_config.v_max, 0.0, 1.0)
        ttc_risks = np.exp(-current_risk.ttc / self.risk_config.tau)
        risks = np.clip(
            self.risk_config.w_distance * distance_risks
            + self.risk_config.w_velocity * velocity_risks
            + self.risk_config.w_ttc * ttc_risks,
            0.0,
            1.0,
        ).astype(np.float32)
        if self.method == "ee_fixed":
            risks[:-1] = 0.0
        return LinkRisk(
            closest_points=current_risk.closest_points,
            distances=distances,
            directions=current_risk.directions,
            link_velocities=current_risk.link_velocities,
            approach_velocities=current_risk.approach_velocities,
            ttc=current_risk.ttc,
            risks=risks,
            risk_global=float(np.max(risks)),
            d_min=float(np.min(distances)),
            closest_link=int(np.argmin(distances)),
        )

    def _zero_risk(self) -> LinkRisk:
        count = self.capsule_model.count
        return LinkRisk(
            closest_points=np.zeros((count, 3), dtype=np.float32),
            distances=np.full(count, float(self.observation_cfg["no_obstacle_distance"]), dtype=np.float32),
            directions=np.zeros((count, 3), dtype=np.float32),
            link_velocities=np.zeros((count, 3), dtype=np.float32),
            approach_velocities=np.zeros(count, dtype=np.float32),
            ttc=np.full(count, self.risk_config.ttc_max, dtype=np.float32),
            risks=np.zeros(count, dtype=np.float32),
            risk_global=0.0,
            d_min=float(self.observation_cfg["no_obstacle_distance"]),
            closest_link=0,
        )

    def _configure_safety_filter(self) -> None:
        if not self.safety_filter_enabled and self.risk_representation == "current":
            return

        self.predictive_risk_config = PredictiveRiskConfig(
            d_safe_m=self.risk_config.d_safe,
            prediction_horizon_s=float(self.safety_filter_cfg["prediction_horizon_s"]),
            max_observation_age_s=float(self.safety_filter_cfg["max_observation_age_s"]),
            control_delay_s=float(self.safety_filter_cfg["control_delay_s"]),
            max_link_speed_mps=self.action_scale,
            tracking_error_bound_m=float(self.safety_filter_cfg["tracking_error_bound_m"]),
            geometry_margin_m=float(self.safety_filter_cfg["geometry_margin_m"]),
        )
        if not self.safety_filter_enabled:
            return

        joint_lower, joint_upper = self._joint_position_limits()
        acceleration_limit = float(self.safety_filter_cfg["joint_acceleration_limit_radps2"])
        self.safety_filter_config = SafetyFilterConfig(
            joint_velocity_limits_radps=np.full(self.joint_count, self.action_scale, dtype=np.float64),
            joint_acceleration_limits_radps2=np.full(self.joint_count, acceleration_limit, dtype=np.float64),
            joint_position_lower_rad=joint_lower,
            joint_position_upper_rad=joint_upper,
            control_dt_s=self.control_dt,
            safety_gain=float(self.safety_filter_cfg["safety_gain"]),
            preemptive_margin_m=float(self.safety_filter_cfg.get("preemptive_margin_m", 0.0)),
            max_projection_iterations=int(self.safety_filter_cfg["max_projection_iterations"]),
            constraint_tolerance=float(self.safety_filter_cfg["constraint_tolerance"]),
            projection_failure_tolerance=float(
                self.safety_filter_cfg.get(
                    "projection_failure_tolerance",
                    max(float(self.safety_filter_cfg["constraint_tolerance"]) * 100.0, 1.0e-6),
                )
            ),
            allow_iterative_fallback=bool(self.safety_filter_cfg.get("allow_iterative_fallback", True)),
            active_set_fallback_enabled=bool(self.safety_filter_cfg.get("active_set_fallback_enabled", True)),
            active_set_max_candidate_constraints=int(
                self.safety_filter_cfg.get("active_set_max_candidate_constraints", 12)
            ),
            fallback_projection_iterations=int(self.safety_filter_cfg.get("fallback_projection_iterations", 320)),
            use_qp_solver=bool(self.safety_filter_cfg.get("use_qp_solver", False)),
            qp_time_limit_s=(
                None
                if self.safety_filter_cfg.get("qp_time_limit_s") is None
                else float(self.safety_filter_cfg["qp_time_limit_s"])
            ),
        )

    def _filter_command(
        self,
        qdot_requested: np.ndarray,
        predictive_risk_override: PredictiveLinkRisk | None = None,
        cycle_started_at: float | None = None,
        precomputed_risk_time_s: float = 0.0,
    ) -> tuple[SafetyFilterResult, PredictiveLinkRisk | None, float]:
        if self.predictive_risk_config is None or self.safety_filter_config is None:
            raise RuntimeError("Safety filter was enabled but not configured")

        started_at = perf_counter() if cycle_started_at is None else cycle_started_at
        phase_times_s = {
            "predictive_risk": float(precomputed_risk_time_s),
            "jacobian_workspace": 0.0,
            "projection": 0.0,
        }
        try:
            if predictive_risk_override is None:
                phase_started_at = perf_counter()
                predictive_risk = self._compute_predictive_risk()
                phase_times_s["predictive_risk"] += perf_counter() - phase_started_at
            else:
                predictive_risk = predictive_risk_override
            if predictive_risk.requires_safe_stop:
                phase_started_at = perf_counter()
                result = filter_joint_velocity(
                    SafetyFilterInput(
                        requested_joint_velocity_radps=qdot_requested,
                        joint_positions_rad=self._joint_state()[0],
                        previous_command_radps=self.prev_qdot_cmd,
                        predictive_risk=predictive_risk,
                        safety_jacobian_m_per_rad=None,
                    ),
                    self.safety_filter_config,
                )
                phase_times_s["projection"] = perf_counter() - phase_started_at
                return self._finalize_filter_command(result, predictive_risk, qdot_requested, started_at, phase_times_s)

            phase_started_at = perf_counter()
            safety_jacobian, ee_position, ee_jacobian = self._analytic_constraint_jacobians(predictive_risk)
            workspace_constraints = self._workspace_velocity_constraints(ee_position, ee_jacobian)
            safety_drift = self._safety_drift_mps(predictive_risk)
            phase_times_s["jacobian_workspace"] = perf_counter() - phase_started_at
            phase_started_at = perf_counter()
            result = filter_joint_velocity(
                SafetyFilterInput(
                    requested_joint_velocity_radps=qdot_requested,
                    joint_positions_rad=self._joint_state()[0],
                    previous_command_radps=self.prev_qdot_cmd,
                    predictive_risk=predictive_risk,
                    safety_jacobian_m_per_rad=safety_jacobian,
                    safety_drift_mps=safety_drift,
                    allow_infeasible_recovery=(
                        self.recovery_active
                        and bool(self.safety_filter_cfg.get("recovery_allow_constraint_relaxation", False))
                    ),
                    recovery_target_mask=(
                        np.asarray(predictive_risk.safety_functions_m, dtype=np.float64)
                        < float(self.safety_filter_cfg.get("recovery_exit_margin_m", 0.02))
                    ),
                    maximize_min_clearance_recovery=(
                        self.recovery_active
                        and bool(self.safety_filter_cfg.get("recovery_maximize_min_clearance", False))
                    ),
                    workspace_constraints=workspace_constraints,
                ),
                self.safety_filter_config,
            )
            phase_times_s["projection"] = perf_counter() - phase_started_at
            return self._finalize_filter_command(result, predictive_risk, qdot_requested, started_at, phase_times_s)
        except (ValueError, RuntimeError, p.error) as error:
            result = SafetyFilterResult(
                command_joint_velocity_radps=np.zeros(self.joint_count, dtype=np.float64),
                status=SafetyFilterStatus.SAFE_STOP_INVALID_INPUT,
                reason=f"safety-filter integration error: {error}",
                intervention_norm_radps=float(np.linalg.norm(qdot_requested)),
                active_constraint_count=0,
                max_constraint_violation=0.0,
            )
            return self._finalize_filter_command(result, None, qdot_requested, started_at, phase_times_s)

    def _finalize_filter_command(
        self,
        result: SafetyFilterResult,
        predictive_risk: PredictiveLinkRisk | None,
        qdot_requested: np.ndarray,
        started_at: float,
        phase_times_s: dict[str, float],
    ) -> tuple[SafetyFilterResult, PredictiveLinkRisk | None, float]:
        elapsed_s = perf_counter() - started_at
        self.last_filter_phase_times_s = phase_times_s
        compute_budget_s = self.safety_filter_cfg.get("max_filter_compute_time_s")
        if compute_budget_s is None:
            return result, predictive_risk, elapsed_s
        compute_budget_s = float(compute_budget_s)
        if not np.isfinite(compute_budget_s) or compute_budget_s <= 0.0:
            raise ValueError("max_filter_compute_time_s must be finite and positive when set")
        if elapsed_s <= compute_budget_s:
            return result, predictive_risk, elapsed_s
        return (
            SafetyFilterResult(
                command_joint_velocity_radps=np.zeros(self.joint_count, dtype=np.float64),
                status=SafetyFilterStatus.SAFE_STOP_COMPUTE_BUDGET,
                reason=(
                    f"safety-filter computation took {elapsed_s:.6f}s, "
                    f"exceeding the {compute_budget_s:.6f}s diagnostic budget"
                ),
                intervention_norm_radps=float(np.linalg.norm(qdot_requested)),
                active_constraint_count=result.active_constraint_count,
                max_constraint_violation=result.max_constraint_violation,
                constraint_count=result.constraint_count,
                active_constraint_categories=result.active_constraint_categories,
                max_constraint_category=result.max_constraint_category,
                projection_iterations=result.projection_iterations,
                fallback_used=result.fallback_used,
                qp_solver_used=result.qp_solver_used,
                qp_solver_status=result.qp_solver_status,
                fallback_stage=result.fallback_stage,
            ),
            predictive_risk,
            elapsed_s,
        )

    def _compute_predictive_risk(self) -> PredictiveLinkRisk:
        if self.predictive_risk_config is None:
            raise RuntimeError("Predictive risk requested while safety filter is disabled")
        return self._compute_predictive_risk_for_capsules(self._capsules())

    def _compute_predictive_risk_for_capsules(self, capsules: list[CapsuleState]) -> PredictiveLinkRisk:
        if self.predictive_risk_config is None:
            raise RuntimeError("Predictive risk requested while safety filter is disabled")
        return compute_predictive_link_risk(
            capsules=capsules,
            # The velocity effect of the next command enters through J_h qdot.
            link_velocities_mps=np.zeros((self.capsule_model.count, 3), dtype=np.float64),
            obstacle=self._obstacle_state_estimate(),
            now_s=self.sim_time_s,
            config=self.predictive_risk_config,
        )

    def _obstacle_state_estimate(self) -> ObstacleStateEstimate:
        if self.obstacle_state_estimate_override is not None:
            return self.obstacle_state_estimate_override
        return ObstacleStateEstimate(
            position=self.obstacle_center,
            velocity=self.obstacle_velocity,
            radius_m=float(self.obstacle_cfg["radius"]),
            timestamp_s=self.sim_time_s,
            position_error_bound_m=0.0,
            velocity_error_bound_mps=0.0,
            valid=True,
        )

    def _analytic_constraint_jacobians(
        self,
        predictive_risk: PredictiveLinkRisk,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return safety and TCP Jacobians without perturbing the simulator state.

        The prediction minimises distance over capsule position and time.  Away
        from ties, the envelope theorem lets us hold those optimum parameters
        fixed while differentiating the distance with respect to the capsule
        endpoints.  PyBullet then supplies the endpoint and TCP point
        Jacobians in a single kinematic state.
        """

        if not predictive_risk.usable:
            raise ValueError("analytic Jacobians require a usable predictive risk")
        obstacle = self._obstacle_state_estimate()
        if not obstacle.valid:
            raise ValueError("analytic Jacobians require a valid obstacle estimate")

        capsules = self._capsules()
        if len(capsules) != self.capsule_model.count:
            raise RuntimeError("capsule state count does not match the configured capsule model")
        obstacle_position = np.asarray(obstacle.position, dtype=np.float64)
        obstacle_velocity = np.asarray(obstacle.velocity, dtype=np.float64)
        safety_jacobian = np.zeros((self.capsule_model.count, self.joint_count), dtype=np.float64)
        for index, (capsule, spec, prediction_time_s) in enumerate(
            zip(capsules, self.capsule_model.specs, predictive_risk.closest_prediction_times_s, strict=True)
        ):
            segment = np.asarray(capsule.end, dtype=np.float64) - np.asarray(capsule.start, dtype=np.float64)
            segment_norm_sq = float(np.dot(segment, segment))
            if segment_norm_sq <= 1e-14:
                continue
            predicted_obstacle_position = obstacle_position + obstacle_velocity * float(prediction_time_s)
            segment_fraction = float(
                np.clip(
                    np.dot(predicted_obstacle_position - capsule.start, segment) / segment_norm_sq,
                    0.0,
                    1.0,
                )
            )
            closest_capsule_point = capsule.start + segment_fraction * segment
            separation = predicted_obstacle_position - closest_capsule_point
            separation_norm = float(np.linalg.norm(separation))
            if separation_norm <= 1e-10:
                continue
            start_link_id = self.capsule_model.link_name_to_id[spec.parent_link_name]
            end_link_id = self.capsule_model.link_name_to_id[spec.child_link_name]
            start_local = (
                spec.start_local_position
                if spec.start_local_position is not None
                else np.zeros(3, dtype=np.float64)
            )
            end_local = (
                spec.end_local_position
                if spec.end_local_position is not None
                else np.zeros(3, dtype=np.float64)
            )
            point_jacobian = (
                (1.0 - segment_fraction) * self._link_point_jacobian(start_link_id, start_local)
                + segment_fraction * self._link_point_jacobian(end_link_id, end_local)
            )
            safety_jacobian[index] = -(separation / separation_norm) @ point_jacobian

        ee_position, _ = self._end_effector_state()
        return safety_jacobian, ee_position.astype(np.float64), self._link_origin_jacobian(self.tool_link_id)

    def _safety_drift_mps(self, predictive_risk: PredictiveLinkRisk) -> np.ndarray:
        """Return the obstacle-induced part of each safety-function derivative."""
        obstacle = self._obstacle_state_estimate()
        obstacle_position = np.asarray(obstacle.position, dtype=np.float64)
        obstacle_velocity = np.asarray(obstacle.velocity, dtype=np.float64)
        drift = np.zeros(self.capsule_model.count, dtype=np.float64)
        for index, (capsule, prediction_time_s) in enumerate(
            zip(self._capsules(), predictive_risk.closest_prediction_times_s, strict=True)
        ):
            segment = np.asarray(capsule.end, dtype=np.float64) - np.asarray(capsule.start, dtype=np.float64)
            segment_norm_sq = float(np.dot(segment, segment))
            if segment_norm_sq <= 1e-14:
                continue
            predicted_obstacle_position = obstacle_position + obstacle_velocity * float(prediction_time_s)
            segment_fraction = float(
                np.clip(
                    np.dot(predicted_obstacle_position - capsule.start, segment) / segment_norm_sq,
                    0.0,
                    1.0,
                )
            )
            closest_capsule_point = capsule.start + segment_fraction * segment
            separation = predicted_obstacle_position - closest_capsule_point
            separation_norm = float(np.linalg.norm(separation))
            if separation_norm > 1e-10:
                drift[index] = float(np.dot(separation / separation_norm, obstacle_velocity))
        return drift

    def _workspace_velocity_constraints(
        self,
        ee_position: np.ndarray,
        ee_jacobian: np.ndarray,
    ) -> LinearVelocityConstraints:
        if self.safety_filter_config is None:
            raise RuntimeError("Safety filter config is unavailable")
        rows = []
        bounds = []
        for axis, limits in enumerate((self.workspace["x"], self.workspace["y"], self.workspace["z"])):
            low, high = (float(value) for value in limits)
            rows.extend((ee_jacobian[axis], -ee_jacobian[axis]))
            bounds.extend(
                (
                    -self.safety_filter_config.safety_gain * (float(ee_position[axis]) - low),
                    -self.safety_filter_config.safety_gain * (high - float(ee_position[axis])),
                )
            )
        labels = tuple(
            label
            for axis_name in ("x", "y", "z")
            for label in (f"workspace_{axis_name}_lower", f"workspace_{axis_name}_upper")
        )
        return LinearVelocityConstraints(matrix=np.asarray(rows), lower_bound=np.asarray(bounds), labels=labels)

    def _link_origin_jacobian(self, link_id: int) -> np.ndarray:
        return self._link_point_jacobian(link_id, np.zeros(3, dtype=np.float64))

    def _link_point_jacobian(self, link_id: int, local_position: np.ndarray) -> np.ndarray:
        if link_id < 0:
            return np.zeros((3, self.joint_count), dtype=np.float64)
        joint_states = p.getJointStates(self.robot_id, self.jacobian_joint_ids, physicsClientId=self.physics_client_id)
        positions = [float(state[0]) for state in joint_states]
        zeros = [0.0] * len(positions)
        linear, _ = p.calculateJacobian(
            self.robot_id,
            link_id,
            [0.0, 0.0, 0.0],
            positions,
            zeros,
            zeros,
            physicsClientId=self.physics_client_id,
        )
        return np.asarray(linear, dtype=np.float64)[:, self.control_jacobian_columns]

    def _joint_position_limits(self) -> tuple[np.ndarray, np.ndarray]:
        limits = [p.getJointInfo(self.robot_id, joint_id, physicsClientId=self.physics_client_id) for joint_id in self.joint_ids]
        lower = np.asarray([info[8] for info in limits], dtype=np.float64)
        upper = np.asarray([info[9] for info in limits], dtype=np.float64)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower >= upper).any():
            raise ValueError("URDF must provide finite lower and upper limits for every controlled joint")
        return lower, upper

    @staticmethod
    def _filter_info(
        result: SafetyFilterResult,
        predictive_risk: PredictiveLinkRisk | None,
        solve_time_s: float | None,
        phase_times_s: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        predictive_info = {
            "predictive_risk_status": "integration_error",
            "predictive_risk_reason": "predictive risk was not produced",
            "predictive_risk_age_s": float("nan"),
            "predictive_h_min_m": float("-inf"),
        }
        if predictive_risk is not None:
            predictive_info = {
                "predictive_risk_status": predictive_risk.status.value,
                "predictive_risk_reason": predictive_risk.status_reason,
                "predictive_risk_age_s": float(predictive_risk.observation_age_s),
                "predictive_h_min_m": float(np.min(predictive_risk.safety_functions_m)),
            }
        return {
            "safety_filter_status": result.status.value,
            "safety_filter_reason": result.reason,
            "safety_filter_intervention_norm": float(result.intervention_norm_radps),
            "safety_filter_active_constraints": int(result.active_constraint_count),
            "safety_filter_constraint_count": int(result.constraint_count),
            "safety_filter_active_constraint_categories": "|".join(result.active_constraint_categories),
            "safety_filter_max_constraint_category": result.max_constraint_category,
            "safety_filter_projection_iterations": int(result.projection_iterations),
            "safety_filter_fallback_used": bool(result.fallback_used),
            "safety_filter_qp_solver_used": bool(result.qp_solver_used),
            "safety_filter_qp_solver_status": result.qp_solver_status,
            "safety_filter_fallback_stage": result.fallback_stage,
            "safety_filter_max_constraint_violation": float(result.max_constraint_violation),
            "safety_filter_safe_stop": bool(result.requires_safe_stop),
            "safety_filter_solve_time_s": float(solve_time_s) if solve_time_s is not None else float("nan"),
            "safety_filter_predictive_risk_time_s": float((phase_times_s or {}).get("predictive_risk", 0.0)),
            "safety_filter_jacobian_workspace_time_s": float((phase_times_s or {}).get("jacobian_workspace", 0.0)),
            "safety_filter_projection_time_s": float((phase_times_s or {}).get("projection", 0.0)),
            **predictive_info,
        }

    def _filter_link_diagnostics(
        self,
        predictive_risk: PredictiveLinkRisk | None,
        command_joint_velocity_radps: np.ndarray,
    ) -> dict[str, Any]:
        """Expose pre-command per-link values needed to diagnose filter misses."""
        if predictive_risk is None or not predictive_risk.usable:
            return {}
        safety_jacobian, _, _ = self._analytic_constraint_jacobians(predictive_risk)
        command = np.asarray(command_joint_velocity_radps, dtype=np.float64)
        h = np.asarray(predictive_risk.safety_functions_m, dtype=np.float64)
        jacobian_command = safety_jacobian @ command
        drift = self._safety_drift_mps(predictive_risk)
        safety_gain = float(self.safety_filter_cfg["safety_gain"])
        preemptive_margin = float(self.safety_filter_cfg.get("preemptive_margin_m", 0.0))
        residual = jacobian_command + drift + safety_gain * (h - preemptive_margin)
        obstacle = self._obstacle_state_estimate()
        return {
            "predictive_h_by_link_m": self._format_diagnostic_vector(h),
            "predictive_distance_by_link_m": self._format_diagnostic_vector(predictive_risk.predicted_distances_m),
            "predictive_robust_distance_by_link_m": self._format_diagnostic_vector(predictive_risk.robust_distances_m),
            "predictive_time_by_link_s": self._format_diagnostic_vector(predictive_risk.closest_prediction_times_s),
            "safety_jacobian_command_by_link_mps": self._format_diagnostic_vector(jacobian_command),
            "safety_drift_by_link_mps": self._format_diagnostic_vector(drift),
            "safety_constraint_residual_by_link_mps": self._format_diagnostic_vector(residual),
            "filter_obstacle_position_m": self._format_diagnostic_vector(obstacle.position),
            "filter_obstacle_velocity_mps": self._format_diagnostic_vector(obstacle.velocity),
        }

    @staticmethod
    def _format_diagnostic_vector(values: np.ndarray) -> str:
        return "|".join(f"{float(value):.9g}" for value in np.asarray(values, dtype=np.float64))

    def _reward(self, qdot_cmd: np.ndarray, success: bool, collision: bool) -> float:
        goal_error_norm = float(np.linalg.norm(self._goal_error()))
        progress = self.prev_goal_error_norm - goal_error_norm
        smooth = float(np.sum(np.square(qdot_cmd - self.prev_qdot_cmd)))
        reward = (
            -float(self.reward_cfg["w_position"]) * goal_error_norm**2
            + float(self.reward_cfg["w_progress"]) * progress
            - float(self.reward_cfg["w_smooth"]) * smooth
        )
        if success:
            reward += float(self.reward_cfg["success_bonus"])
        if collision:
            reward -= float(self.reward_cfg["collision_penalty"])
        return float(reward)

    def _cost(self, policy_risk: LinkRisk, collision: bool, safety_violation: bool) -> float:
        violation = 1.0 if safety_violation else 0.0
        collision_f = 1.0 if collision else 0.0
        return float(
            float(self.cost_cfg["k_risk"]) * policy_risk.risk_global
            + float(self.cost_cfg["k_violation"]) * violation
            + float(self.cost_cfg["k_collision"]) * collision_f
        )

    def _adaptive_beta(self, risk_global: float) -> float:
        ratio = np.clip(risk_global / max(self.risk_high, 1e-6), 0.0, 1.0)
        beta_raw = self.beta_min + (self.beta_max - self.beta_min) * ratio
        return float(self.lambda_beta * beta_raw + (1.0 - self.lambda_beta) * self.beta)

    def _risk_speed_scale(self, predictive_risk: PredictiveLinkRisk) -> float:
        """Continuously reduce policy speed as the predictive margin closes."""
        if predictive_risk.requires_safe_stop or not predictive_risk.usable:
            return 0.0
        h_min = float(np.min(predictive_risk.safety_functions_m))
        start = float(self.safety_filter_cfg.get("risk_speed_scaling_start_m", 0.08))
        stop = float(self.safety_filter_cfg.get("risk_speed_scaling_stop_m", 0.0))
        if not np.isfinite(h_min) or not np.isfinite(start) or not np.isfinite(stop) or start <= stop or stop < 0.0:
            raise ValueError("risk speed scaling requires finite start > stop >= 0")
        return float(np.clip((h_min - stop) / (start - stop), 0.0, 1.0))

    def _update_recovery_state(self, predictive_risk: PredictiveLinkRisk) -> None:
        """Enter recovery at non-positive margin and exit after clearance."""
        if predictive_risk.requires_safe_stop or not predictive_risk.usable:
            return
        h_min = float(np.min(predictive_risk.safety_functions_m))
        enter_margin = float(self.safety_filter_cfg.get("recovery_enter_margin_m", 0.0))
        exit_margin = float(self.safety_filter_cfg.get("recovery_exit_margin_m", 0.02))
        if (
            not np.isfinite(enter_margin)
            or not np.isfinite(exit_margin)
            or enter_margin < 0.0
            or exit_margin <= enter_margin
        ):
            raise ValueError("recovery requires finite 0 <= enter margin < exit margin")
        if not self.recovery_active and h_min <= enter_margin:
            self.recovery_active = True
            self.recovery_triggered = True
            self.recovery_success = False
        elif self.recovery_active:
            self.recovery_steps += 1
            if h_min >= exit_margin:
                self.recovery_active = False
                self.recovery_success = True

    def _recovery_command(self, predictive_risk: PredictiveLinkRisk) -> np.ndarray:
        """Generate a bounded velocity that separates every endangered link.

        Negative margins receive weights proportional to their clearance
        deficits, so the command is driven by all endangered links rather
        than only the currently worst one.  While recovering, links below the
        exit margin remain in the objective to avoid an immediate re-trigger.
        """
        safety_jacobian, _, _ = self._analytic_constraint_jacobians(predictive_risk)
        margins = np.asarray(predictive_risk.safety_functions_m, dtype=np.float64)
        if safety_jacobian.shape != (margins.size, self.joint_count):
            raise ValueError("safety Jacobian shape does not match predictive margins")
        if not np.isfinite(margins).all() or not np.isfinite(safety_jacobian).all():
            return np.zeros(self.joint_count, dtype=np.float32)

        exit_margin = float(self.safety_filter_cfg.get("recovery_exit_margin_m", 0.02))
        # Include every negative-margin link; during recovery also retain links
        # that have not yet reached the exit clearance.
        deficits = np.maximum(-margins, 0.0)
        if not np.any(deficits > 0.0):
            deficits = np.maximum(exit_margin - margins, 0.0)
        if not np.any(deficits > 0.0):
            return np.zeros(self.joint_count, dtype=np.float32)
        weights = deficits / float(np.sum(deficits))
        direction = np.sum(weights[:, None] * safety_jacobian, axis=0)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-10:
            return np.zeros(self.joint_count, dtype=np.float32)
        speed = float(self.safety_filter_cfg.get("recovery_speed_radps", 0.25))
        if not np.isfinite(speed) or speed < 0.0:
            raise ValueError("recovery_speed_radps must be finite and non-negative")
        return np.clip(speed * direction / norm, -self.action_scale, self.action_scale).astype(np.float32)

    def _joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        states = p.getJointStates(self.robot_id, self.joint_ids, physicsClientId=self.physics_client_id)
        q = np.asarray([s[0] for s in states], dtype=np.float32)
        q_dot = np.asarray([s[1] for s in states], dtype=np.float32)
        return q, q_dot

    def _end_effector_state(self) -> tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(
            self.robot_id,
            self.tool_link_id,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
            physicsClientId=self.physics_client_id,
        )
        return np.asarray(state[4], dtype=np.float32), np.asarray(state[6], dtype=np.float32)

    def _goal_error(self) -> np.ndarray:
        ee_pos, _ = self._end_effector_state()
        return self.goal - ee_pos

    def _capsules(self) -> list[CapsuleState]:
        return self.capsule_model.states(self.robot_id, self.physics_client_id)

    def _collision_sources(self, d_min: float) -> tuple[bool, bool, str, str, float]:
        if not self.obstacle_enabled or self.obstacle_id is None:
            return False, False, "", "", float("nan")
        capsule_overlap = d_min <= 0.0
        contacts = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.obstacle_id,
            physicsClientId=self.physics_client_id,
        )
        link_indices = sorted({int(contact[3]) for contact in contacts})
        link_names = []
        for link_index in link_indices:
            if link_index < 0:
                link_names.append("base")
            else:
                link_names.append(
                    p.getJointInfo(self.robot_id, link_index, physicsClientId=self.physics_client_id)[12].decode(
                        "utf-8"
                    )
                )
        contact_distances = [float(contact[8]) for contact in contacts]
        return (
            capsule_overlap,
            bool(contacts),
            "|".join(str(index) for index in link_indices),
            "|".join(link_names),
            min(contact_distances, default=float("nan")),
        )

    def _reset_robot(self) -> None:
        reset_cfg = self.robot_cfg["reset"]
        default = np.asarray(reset_cfg["default_joint_positions"], dtype=np.float32)
        noise_range = float(reset_cfg["joint_noise_range"])
        noise = self.rng.uniform(-noise_range, noise_range, size=self.joint_count).astype(np.float32)
        q0 = default + noise
        for joint_id, joint_value in zip(self.joint_ids, q0):
            p.resetJointState(
                self.robot_id,
                joint_id,
                float(joint_value),
                targetVelocity=0.0,
                physicsClientId=self.physics_client_id,
            )

    def _resolve_robot_references(self) -> None:
        joint_name_to_id, link_name_to_id = self._robot_name_maps()
        self.joint_ids = [self._resolve_name(name, joint_name_to_id, "joint") for name in self.robot_cfg["joint_names"]]
        self.jacobian_joint_ids = [
            joint_id
            for joint_id in range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client_id))
            if p.getJointInfo(self.robot_id, joint_id, physicsClientId=self.physics_client_id)[3] >= 0
        ]
        jacobian_columns = {joint_id: index for index, joint_id in enumerate(self.jacobian_joint_ids)}
        self.control_jacobian_columns = [jacobian_columns[joint_id] for joint_id in self.joint_ids]
        self.tool_link_id = self._resolve_name(self.robot_cfg["tool_link_name"], link_name_to_id, "link")
        self.joint_count = len(self.joint_ids)
        self.capsule_model.resolve_link_names(link_name_to_id)

    def _robot_name_maps(self) -> tuple[dict[str, int], dict[str, int]]:
        joint_name_to_id: dict[str, int] = {}
        link_name_to_id = {"base": -1}
        for joint_id in range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client_id)):
            info = p.getJointInfo(self.robot_id, joint_id, physicsClientId=self.physics_client_id)
            joint_name_to_id[info[1].decode("utf-8")] = joint_id
            link_name_to_id[info[12].decode("utf-8")] = joint_id
        return joint_name_to_id, link_name_to_id

    @staticmethod
    def _resolve_name(name: str, name_to_id: dict[str, int], kind: str) -> int:
        if name not in name_to_id:
            available = ", ".join(sorted(name_to_id))
            raise KeyError(f"Unknown {kind} name {name!r}; available {kind}s: {available}")
        return name_to_id[name]

    def _sample_goal(self) -> np.ndarray:
        if self.goal_cfg.get("fixed", False):
            return np.asarray(self.goal_cfg["position"], dtype=np.float32)
        return np.array(
            [
                self.rng.uniform(*self.workspace["x"]),
                self.rng.uniform(*self.workspace["y"]),
                self.rng.uniform(*self.workspace["z"]),
            ],
            dtype=np.float32,
        )

    def _sample_obstacle(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.obstacle_enabled:
            return np.asarray(self.obstacle_cfg["disabled_position"], dtype=np.float32), np.zeros(3, dtype=np.float32)

        if self.obstacle_scenario != "random":
            return self._sample_named_obstacle(self.obstacle_scenario)

        random_cfg = self.obstacle_cfg["random"]
        target_band_z = self.rng.uniform(*random_cfg["z_range"])
        side = -1.0 if self.rng.random() < 0.5 else 1.0
        center = np.array(
            [
                self.rng.uniform(*random_cfg["x_range"]),
                side * self.rng.uniform(*random_cfg["start_y_abs_range"]),
                target_band_z,
            ],
            dtype=np.float32,
        )
        target = np.array(
            [
                self.rng.uniform(*random_cfg["x_range"]),
                -side * self.rng.uniform(*random_cfg["target_y_abs_range"]),
                self.rng.uniform(*random_cfg["z_range"]),
            ],
            dtype=np.float32,
        )
        direction = target - center
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        speed = self.rng.uniform(*self.obstacle_cfg["speed_range"])
        return center, (direction * speed).astype(np.float32)

    def _sample_named_obstacle(self, scenario: str) -> tuple[np.ndarray, np.ndarray]:
        scenarios = self.obstacle_cfg["scenarios"]
        if scenario not in scenarios:
            raise ValueError(f"Unknown obstacle scenario {scenario!r}; expected random or one of {sorted(scenarios)}")

        scenario_cfg = scenarios[scenario]
        side = -1.0 if self.rng.random() < 0.5 else 1.0
        x_center = self.rng.uniform(*scenario_cfg["x_range"])
        z_range = scenario_cfg["z_range"]
        center = np.array(
            [x_center, side * float(scenario_cfg["start_y_abs"]), self.rng.uniform(*z_range)],
            dtype=np.float32,
        )
        target = np.array(
            [x_center, -side * float(scenario_cfg["target_y_abs"]), self.rng.uniform(*z_range)],
            dtype=np.float32,
        )
        direction = target - center
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        speed = self.rng.uniform(*self.obstacle_cfg["speed_range"])
        return center, (direction * speed).astype(np.float32)

    def _move_obstacle(self, dt: float) -> None:
        if not self.obstacle_enabled or self.obstacle_id is None:
            return
        self.obstacle_center = (self.obstacle_center + self.obstacle_velocity * dt).astype(np.float32)
        bounds = self.obstacle_cfg["bounds"]
        for axis, (low, high) in enumerate([bounds["x"], bounds["y"], bounds["z"]]):
            if self.obstacle_center[axis] < low or self.obstacle_center[axis] > high:
                self.obstacle_velocity[axis] *= -1.0
                self.obstacle_center[axis] = np.clip(self.obstacle_center[axis], low, high)
        p.resetBasePositionAndOrientation(
            self.obstacle_id,
            self.obstacle_center.tolist(),
            [0, 0, 0, 1],
            physicsClientId=self.physics_client_id,
        )

    def _create_floor(self) -> None:
        collision = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.physics_client_id)
        visual = p.createVisualShape(
            p.GEOM_PLANE,
            rgbaColor=self.visual_cfg["floor_rgba"],
            physicsClientId=self.physics_client_id,
        )
        p.createMultiBody(
            0,
            collision,
            visual,
            self.visual_cfg["floor_position"],
            physicsClientId=self.physics_client_id,
        )

    def _create_obstacle(self, center: np.ndarray) -> int:
        radius = float(self.obstacle_cfg["radius"])
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.physics_client_id)
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=self.visual_cfg["obstacle_rgba"],
            physicsClientId=self.physics_client_id,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=center.tolist(),
            physicsClientId=self.physics_client_id,
        )

    def _create_goal_marker(self, goal: np.ndarray) -> int:
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=float(self.visual_cfg["goal_marker_radius"]),
            rgbaColor=self.visual_cfg["goal_rgba"],
            physicsClientId=self.physics_client_id,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=goal.tolist(),
            physicsClientId=self.physics_client_id,
        )
