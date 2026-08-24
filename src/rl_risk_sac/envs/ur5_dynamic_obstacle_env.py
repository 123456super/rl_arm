from __future__ import annotations
# 十分丰富
from dataclasses import replace
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from rl_risk_sac.robots.ur5_capsules import CapsuleSpec, CapsuleState, UR5CapsuleModel
from rl_risk_sac.utils.collision import COLLISION_TERMINATION_MODES, classify_collision_events
from rl_risk_sac.utils.predictive_risk import (
    ObstacleStateEstimate,
    PredictiveLinkRisk,
    PredictiveRiskConfig,
    compute_predictive_link_risk,
)
from rl_risk_sac.utils.motion_planning import RRTConnectConfig, RRTConnectPlanner
from rl_risk_sac.utils.risk import LinkRisk, RiskConfig, closest_point_on_segment, compute_link_risk
from rl_risk_sac.utils.safety_filter import (
    LinearVelocityConstraints,
    SafetyFilterConfig,
    SafetyFilterInput,
    SafetyFilterResult,
    SafetyFilterStatus,
    assess_strict_viability,
    filter_joint_velocity,
    maximize_linear_velocity,
)


METHODS = {"ee_fixed", "link_fixed", "ldrc_fixed", "ldrc_adaptive", "hierarchical_residual"}
RISK_REPRESENTATIONS = {"current", "predictive", "robust_predictive"}
HIERARCHICAL_STATES = ("TRACK", "AVOID_HOLD", "REPLAN", "SERVO", "PLAN_FAILED")


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
        self.residual_control_cfg = env_cfg.get("residual_control", {})
        self.residual_control_enabled = bool(self.residual_control_cfg.get("enabled", True))
        self.hierarchical_cfg = env_cfg.get("hierarchical_control", {})
        self.hierarchical_enabled = bool(self.hierarchical_cfg.get("enabled", False))
        self.hierarchical_planner_cfg = self.hierarchical_cfg.get("planner", {})
        self.hierarchical_tracker_cfg = self.hierarchical_cfg.get("tracker", {})
        self.hierarchical_residual_cfg = self.hierarchical_cfg.get("residual", {})
        self.hierarchical_state = "TRACK"
        self.hierarchical_path: list[np.ndarray] = []
        self.hierarchical_waypoint_index = 0
        self.hierarchical_plan_reason = "disabled"
        self.hierarchical_plan_iterations = 0
        self.hierarchical_planning_time_s = 0.0
        self.hierarchical_planning_time_total_s = 0.0
        self.hierarchical_ik_candidate_count = 0
        self.hierarchical_plan_count = 0
        self.hierarchical_replan_count = 0
        self.hierarchical_last_plan_step = 0
        self.hierarchical_hold_steps = 0
        self.hierarchical_consecutive_hold_steps = 0
        configured_joint_count = len(self.robot_cfg["joint_names"])
        self.hierarchical_last_nominal_qdot = np.zeros(configured_joint_count, dtype=np.float32)
        self.hierarchical_last_residual_budget = 1.0
        self.hierarchical_last_progress = 0.0
        self.hierarchical_last_waypoint_error = np.zeros(configured_joint_count, dtype=np.float32)
        self.hierarchical_selected_candidate_index = -1
        self.hierarchical_selected_path_length_rad = float("nan")
        self.hierarchical_selected_path_min_clearance_m = float("nan")
        self.hierarchical_selected_path_predictive_h_m = float("nan")
        self.hierarchical_selected_path_filter_intervention = float("nan")
        self.hierarchical_ik_candidate_errors_m: list[float] = []
        self.hierarchical_ik_candidate_clearances_m: list[float] = []
        self.hierarchical_ik_candidate_kinds: list[str] = []
        # A goal-region IK solution is a valid terminal configuration for the
        # original task (its TCP remains within the normal success tolerance).
        # Keep it across obstacle holds so a replan does not discard the only
        # known collision-free terminal configuration and retry the blocked
        # exact target from scratch.
        self.hierarchical_goal_region_terminal_q: np.ndarray | None = None
        self.hierarchical_selected_terminal_is_goal_region = False
        self.hierarchical_ik_attempts_used = 0
        self.hierarchical_ik_fallback_used = False
        self.hierarchical_ik_fallback_attempted = False
        self.hierarchical_ik_target_shell_attempted = False
        self.hierarchical_ik_target_shell_candidate_count = 0
        self.hierarchical_ik_goal_region_attempted = False
        self.hierarchical_ik_goal_region_candidate_count = 0
        self.hierarchical_ik_goal_region_sample_count = 0
        self.hierarchical_ik_obstacle_aware_attempted = False
        self.hierarchical_ik_obstacle_aware_candidate_count = 0
        self.hierarchical_ik_rejection_counts: dict[str, int] = {}
        self.hierarchical_ik_min_goal_error_m = float("nan")
        self.hierarchical_ik_goal_reachable_count = 0
        self.hierarchical_ik_obstacle_free_goal_reachable_count = 0
        self._hierarchical_planning_start_q: np.ndarray | None = None
        self.hierarchical_last_filter_intervention_ratio = 0.0
        self.hierarchical_last_filter_status = "not_enabled"
        self.hierarchical_last_filtered_qdot = np.zeros(configured_joint_count, dtype=np.float32)
        self.hierarchical_servo_stall_steps = 0
        self.hierarchical_servo_stall_replans = 0
        self.safety_filter_cfg = env_cfg.get("safety_filter", {})
        self.safety_filter_enabled = bool(self.safety_filter_cfg.get("enabled", False))
        self.obstacle_enabled = bool(self.obstacle_cfg.get("enabled", True))
        self.obstacle_scenario = str(self.obstacle_cfg.get("scenario", "random"))
        self.collision_termination_mode = str(env_cfg.get("collision", {}).get("termination", "any"))
        if self.collision_termination_mode not in COLLISION_TERMINATION_MODES:
            raise ValueError(
                f"Unknown env.collision.termination {self.collision_termination_mode!r}; "
                f"expected one of {sorted(COLLISION_TERMINATION_MODES)}"
            )

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
        self.last_residual_observation = np.zeros(0, dtype=np.float32)
        self.sim_time_s = 0.0
        self.predictive_risk_config: PredictiveRiskConfig | None = None
        self.safety_filter_config: SafetyFilterConfig | None = None
        self.joint_velocity_limit_corners = np.asarray(
            list(product((-self.action_scale, self.action_scale), repeat=self.joint_count)),
            dtype=np.float64,
        )
        self.last_filter_phase_times_s: dict[str, float] = {}
        self.obstacle_state_estimate_override: ObstacleStateEstimate | None = None
        self.recovery_active = False
        self.recovery_triggered = False
        self.recovery_steps = 0
        self.recovery_success = False
        self.recovery_initially_unsafe = False
        self.recovery_initial_h_min_m = float("nan")
        self.recovery_trigger_reason = ""
        self.recovery_min_ttc_s = float("inf")
        self.unavoidable_collision = False
        self.avoidable_collision = False
        self.safe_stop_infeasible_first_step: int | None = None
        self.safe_stop_infeasible_last_step: int | None = None
        self.safe_stop_infeasible_first_h_m = float("nan")
        self.safe_stop_infeasible_last_h_m = float("nan")

        self.residual_mode_names = ("disabled", "goal_clf", "terminal_clf", "waypoint", "hold_for_clearance")
        self.residual_observation_dim = self.joint_count * 3 + len(self.residual_mode_names)
        obs_dim = self.joint_count * 3 + self.capsule_model.count * 7 + 7
        if self.residual_control_enabled:
            obs_dim += self.residual_observation_dim
        self.hierarchical_state_names = HIERARCHICAL_STATES
        self.hierarchical_observation_dim = self.joint_count * 2 + 1 + 1 + len(HIERARCHICAL_STATES)
        if self.hierarchical_enabled:
            obs_dim += self.hierarchical_observation_dim
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
            flags=(p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT if self.hierarchical_enabled else 0),
            physicsClientId=self.physics_client_id,
        )
        self._resolve_robot_references()
        self._configure_safety_filter()
        self._reset_robot()
        self.goal = self._sample_goal()
        self.obstacle_center, self.obstacle_velocity = self._sample_obstacle()
        self._apply_reset_jitter(options)
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
        self.recovery_trigger_reason = ""
        self.recovery_min_ttc_s = float("inf")
        self.unavoidable_collision = False
        self.avoidable_collision = False
        self.safe_stop_infeasible_first_step = None
        self.safe_stop_infeasible_last_step = None
        self.safe_stop_infeasible_first_h_m = float("nan")
        self.safe_stop_infeasible_last_h_m = float("nan")
        self.hierarchical_state = "TRACK"
        self.hierarchical_path = []
        self.hierarchical_waypoint_index = 0
        self.hierarchical_plan_reason = "disabled"
        self.hierarchical_plan_iterations = 0
        self.hierarchical_planning_time_s = 0.0
        self.hierarchical_planning_time_total_s = 0.0
        self.hierarchical_ik_candidate_count = 0
        self.hierarchical_plan_count = 0
        self.hierarchical_replan_count = 0
        self.hierarchical_last_plan_step = 0
        self.hierarchical_hold_steps = 0
        self.hierarchical_consecutive_hold_steps = 0
        self.hierarchical_last_nominal_qdot = np.zeros(self.joint_count, dtype=np.float32)
        self.hierarchical_last_residual_budget = 1.0
        self.hierarchical_last_progress = 0.0
        self.hierarchical_last_waypoint_error = np.zeros(self.joint_count, dtype=np.float32)
        self.hierarchical_selected_candidate_index = -1
        self.hierarchical_selected_path_length_rad = float("nan")
        self.hierarchical_selected_path_min_clearance_m = float("nan")
        self.hierarchical_ik_candidate_errors_m = []
        self.hierarchical_ik_candidate_clearances_m = []
        self.hierarchical_ik_candidate_kinds = []
        self.hierarchical_goal_region_terminal_q = None
        self.hierarchical_selected_terminal_is_goal_region = False
        self.hierarchical_ik_attempts_used = 0
        self.hierarchical_ik_fallback_used = False
        self.hierarchical_ik_fallback_attempted = False
        self.hierarchical_ik_target_shell_attempted = False
        self.hierarchical_ik_target_shell_candidate_count = 0
        self.hierarchical_ik_goal_region_attempted = False
        self.hierarchical_ik_goal_region_candidate_count = 0
        self.hierarchical_ik_goal_region_sample_count = 0
        self.hierarchical_ik_obstacle_aware_attempted = False
        self.hierarchical_ik_obstacle_aware_candidate_count = 0
        self.hierarchical_ik_rejection_counts = {
            "solver_error": 0,
            "short_solution": 0,
            "duplicate": 0,
            "goal_error": 0,
            "joint_limit": 0,
            "workspace": 0,
            "obstacle_clearance": 0,
            "obstacle_contact": 0,
            "self_collision": 0,
        }
        self.hierarchical_ik_min_goal_error_m = float("nan")
        self.hierarchical_ik_goal_reachable_count = 0
        self.hierarchical_ik_obstacle_free_goal_reachable_count = 0
        self.hierarchical_selected_path_predictive_h_m = float("nan")
        self.hierarchical_selected_path_filter_intervention = float("nan")
        self._hierarchical_planning_start_q = None
        self.hierarchical_last_filter_intervention_ratio = 0.0
        self.hierarchical_last_filter_status = "not_enabled"
        self.hierarchical_last_filtered_qdot = np.zeros(self.joint_count, dtype=np.float32)
        self.hierarchical_servo_stall_steps = 0
        self.hierarchical_servo_stall_replans = 0
        if self.hierarchical_enabled:
            self._initialize_hierarchical_plan()
        if self.safety_filter_enabled:
            initial_predictive_risk = self._compute_predictive_risk()
            if initial_predictive_risk.usable:
                self.recovery_initial_h_min_m = float(np.min(initial_predictive_risk.safety_functions_m))
                self.recovery_initially_unsafe = self.recovery_initial_h_min_m <= 0.0
        self.prev_capsules = self._capsules()
        self.prev_goal_error_norm = float(np.linalg.norm(self._goal_error()))
        obs, info = self._get_obs_and_info()
        return obs, info

    def _initialize_hierarchical_plan(self) -> None:
        planning_started_at = perf_counter()
        self.hierarchical_plan_count += 1
        self.hierarchical_selected_candidate_index = -1
        self.hierarchical_selected_path_length_rad = float("nan")
        self.hierarchical_selected_path_min_clearance_m = float("nan")
        self.hierarchical_selected_terminal_is_goal_region = False
        q, _ = self._joint_state()
        self._hierarchical_planning_start_q = np.asarray(q, dtype=np.float64).copy()
        lower, upper = self._joint_position_limits()
        ik_candidates = self._ik_goal_candidates(
            q,
            lower,
            upper,
            preferred_goal_q=self.hierarchical_goal_region_terminal_q,
        )
        self.hierarchical_ik_candidate_count = len(ik_candidates)
        if not ik_candidates:
            self.hierarchical_state = "PLAN_FAILED"
            self.hierarchical_path = []
            self.hierarchical_plan_reason = "ik_not_found"
            self.hierarchical_planning_time_s = perf_counter() - planning_started_at
            self.hierarchical_planning_time_total_s += self.hierarchical_planning_time_s
            self._hierarchical_planning_start_q = None
            return

        planner_cfg = RRTConnectConfig(
            step_size_rad=float(self.hierarchical_planner_cfg.get("step_size_rad", 0.25)),
            edge_resolution_rad=float(self.hierarchical_planner_cfg.get("edge_resolution_rad", 0.06)),
            max_iterations=int(self.hierarchical_planner_cfg.get("max_iterations", 1200)),
            goal_sample_probability=float(self.hierarchical_planner_cfg.get("goal_sample_probability", 0.10)),
        )
        successful_plans: list[dict[str, Any]] = []
        best_result = None
        for candidate_index, candidate in enumerate(ik_candidates):
            planner = RRTConnectPlanner(
                lower,
                upper,
                self._planning_state_is_valid,
                config=planner_cfg,
                edge_is_valid=self._planning_edge_is_valid,
                rng=self.rng,
            )
            result = planner.plan(q, candidate)
            if result.success:
                path_length, min_clearance = self._planning_path_metrics(result.path)
                successful_plans.append(
                    {
                        "candidate_index": candidate_index,
                        "result": result,
                        "path_length": path_length,
                        "min_clearance": min_clearance,
                        "goal_error": float(self.hierarchical_ik_candidate_errors_m[candidate_index])
                        if candidate_index < len(self.hierarchical_ik_candidate_errors_m)
                        else float("nan"),
                        "predictive_h": float("nan"),
                        "filter_intervention": float("nan"),
                        "predictive_scored": False,
                        "preferred_terminal": bool(
                            candidate_index < len(self.hierarchical_ik_candidate_kinds)
                            and self.hierarchical_ik_candidate_kinds[candidate_index] == "retained_goal_region"
                        ),
                    }
                )
                if str(self.hierarchical_planner_cfg.get("candidate_selection", "first_success")) == "first_success":
                    break
            if best_result is None or result.iterations < best_result.iterations:
                best_result = result
        if successful_plans:
            selection = str(self.hierarchical_planner_cfg.get("candidate_selection", "clearance_then_length"))
            goal_region_selection = str(
                self.hierarchical_planner_cfg.get("goal_region_candidate_selection", "clearance_then_length")
            )
            if self.hierarchical_ik_goal_region_candidate_count > 0 and goal_region_selection == "goal_error_then_clearance":
                successful_plans.sort(
                    key=lambda item: (
                        0 if item["preferred_terminal"] else 1,
                        float(item["goal_error"]),
                        -float(item["min_clearance"]),
                        float(item["path_length"]),
                        int(item["candidate_index"]),
                    )
                )
            elif selection == "predictive_clearance_then_length":
                # Predictive filter scoring is more expensive than geometric
                # metrics; retain only the safest geometric shortlist first.
                shortlist_limit = max(
                    1,
                    int(self.hierarchical_planner_cfg.get("predictive_candidate_limit", 8)),
                )
                predictive_candidates = sorted(
                    successful_plans,
                    key=lambda item: (
                        -float(item["min_clearance"]),
                        float(item["path_length"]),
                        int(item["candidate_index"]),
                    ),
                )[:shortlist_limit]
                self._score_predictive_path_candidates(predictive_candidates)
                scored_candidates = [item for item in predictive_candidates if item["predictive_scored"]]
                if scored_candidates:
                    successful_plans = sorted(
                        scored_candidates,
                        key=lambda item: (
                            0 if item["preferred_terminal"] else 1,
                            -float(item["predictive_h"]),
                            float(item["filter_intervention"]),
                            float(item["path_length"]),
                            int(item["candidate_index"]),
                        ),
                    )
                else:
                    # A path that was not actually evaluated by the strict
                    # filter must never enter predictive ranking. Fall back
                    # explicitly to the established geometric selector.
                    successful_plans.sort(
                        key=lambda item: (
                            0 if item["preferred_terminal"] else 1,
                            -float(item["min_clearance"]),
                            float(item["path_length"]),
                            int(item["candidate_index"]),
                        )
                    )
            elif selection == "length_then_clearance":
                successful_plans.sort(
                    key=lambda item: (
                        0 if item["preferred_terminal"] else 1,
                        float(item["path_length"]),
                        -float(item["min_clearance"]),
                        int(item["candidate_index"]),
                    )
                )
            else:
                successful_plans.sort(
                    key=lambda item: (
                        0 if item["preferred_terminal"] else 1,
                        -float(item["min_clearance"]),
                        float(item["path_length"]),
                        int(item["candidate_index"]),
                    )
                )
            selected = successful_plans[0]
            candidate_index = int(selected["candidate_index"])
            best_result = selected["result"]
            path_length = float(selected["path_length"])
            min_clearance = float(selected["min_clearance"])
            self.hierarchical_selected_candidate_index = candidate_index
            self.hierarchical_selected_path_length_rad = path_length
            self.hierarchical_selected_path_min_clearance_m = min_clearance
            self.hierarchical_selected_path_predictive_h_m = float(selected["predictive_h"])
            self.hierarchical_selected_path_filter_intervention = float(selected["filter_intervention"])
            self.hierarchical_selected_terminal_is_goal_region = bool(
                selected["preferred_terminal"]
                or (
                    candidate_index < len(self.hierarchical_ik_candidate_kinds)
                    and self.hierarchical_ik_candidate_kinds[candidate_index] == "goal_region"
                )
            )
            if (
                candidate_index < len(self.hierarchical_ik_candidate_kinds)
                and self.hierarchical_ik_candidate_kinds[candidate_index] == "goal_region"
            ):
                self.hierarchical_goal_region_terminal_q = np.asarray(ik_candidates[candidate_index], dtype=np.float64).copy()
        if not successful_plans:
            self.hierarchical_state = "PLAN_FAILED"
            self.hierarchical_path = []
            self.hierarchical_plan_reason = "not_found"
            self.hierarchical_plan_iterations = int(best_result.iterations if best_result else 0)
            self.hierarchical_planning_time_s = perf_counter() - planning_started_at
            self.hierarchical_planning_time_total_s += self.hierarchical_planning_time_s
            self._hierarchical_planning_start_q = None
            return
        self.hierarchical_path = [np.asarray(point, dtype=np.float32) for point in best_result.path]
        self.hierarchical_waypoint_index = 1 if len(self.hierarchical_path) > 1 else 0
        self.hierarchical_state = "TRACK"
        self.hierarchical_plan_reason = best_result.reason
        self.hierarchical_plan_iterations = int(best_result.iterations)
        self.hierarchical_planning_time_s = perf_counter() - planning_started_at
        self.hierarchical_planning_time_total_s += self.hierarchical_planning_time_s
        self._hierarchical_planning_start_q = None

    def _planning_path_metrics(self, path: tuple[np.ndarray, ...] | list[np.ndarray]) -> tuple[float, float]:
        """Return joint-space path length and minimum capsule clearance.

        This diagnostic evaluation restores the live simulator state after every
        path, so selecting a safer candidate cannot perturb the episode.
        """
        if len(path) < 2:
            return 0.0, float("inf")
        length = float(sum(np.linalg.norm(np.asarray(end) - np.asarray(start)) for start, end in zip(path[:-1], path[1:])))
        saved_q, saved_qdot = self._joint_state()
        clearances: list[float] = []
        try:
            for point in path:
                for joint_id, value in zip(self.joint_ids, point, strict=True):
                    p.resetJointState(self.robot_id, joint_id, float(value), targetVelocity=0.0, physicsClientId=self.physics_client_id)
                p.performCollisionDetection(physicsClientId=self.physics_client_id)
                clearances.append(float(self._compute_risk().d_min))
        finally:
            for joint_id, value, velocity in zip(self.joint_ids, saved_q, saved_qdot, strict=True):
                p.resetJointState(self.robot_id, joint_id, float(value), targetVelocity=float(velocity), physicsClientId=self.physics_client_id)
        return length, min(clearances, default=float("inf"))

    def _score_predictive_path_candidates(self, candidates: list[dict[str, Any]]) -> None:
        """Estimate path-level predictive feasibility without changing live state.

        The score is deliberately diagnostic/planning-only: it never replaces
        the strict safety filter. A candidate is evaluated at each waypoint
        with a clipped finite-difference path velocity and the same filter
        constraints used online; this ranks executable paths without claiming
        that an unexecuted path is certified.
        """
        if not self.safety_filter_enabled or not self.obstacle_enabled:
            for item in candidates:
                item["predictive_h"] = float(item["min_clearance"] - self.risk_config.d_safe)
                item["filter_intervention"] = 0.0
                item["predictive_scored"] = True
            return
        saved_q, saved_qdot = self._joint_state()
        try:
            for item in candidates:
                path = item["result"].path
                h_values: list[float] = []
                intervention_values: list[float] = []
                scoring_failed = False
                for point_index, point in enumerate(path):
                    for joint_id, value in zip(self.joint_ids, point, strict=True):
                        p.resetJointState(
                            self.robot_id,
                            joint_id,
                            float(value),
                            targetVelocity=0.0,
                            physicsClientId=self.physics_client_id,
                        )
                    p.performCollisionDetection(physicsClientId=self.physics_client_id)
                    predictive = self._compute_predictive_risk()
                    if not predictive.usable:
                        scoring_failed = True
                        break
                    h_values.append(float(np.min(predictive.safety_functions_m)))
                    jacobian, ee_position, ee_jacobian = self._analytic_constraint_jacobians(predictive)
                    workspace = self._workspace_velocity_constraints(ee_position, ee_jacobian)
                    if point_index < len(path) - 1:
                        requested_qdot = (
                            np.asarray(path[point_index + 1], dtype=np.float64)
                            - np.asarray(point, dtype=np.float64)
                        ) / max(self.control_dt, 1.0e-6)
                    elif point_index > 0:
                        requested_qdot = (
                            np.asarray(point, dtype=np.float64)
                            - np.asarray(path[point_index - 1], dtype=np.float64)
                        ) / max(self.control_dt, 1.0e-6)
                    else:
                        requested_qdot = np.zeros(self.joint_count, dtype=np.float64)
                    requested_qdot = np.clip(requested_qdot, -self.action_scale, self.action_scale)
                    try:
                        result = filter_joint_velocity(
                            SafetyFilterInput(
                                requested_joint_velocity_radps=requested_qdot,
                                joint_positions_rad=np.asarray(point, dtype=np.float64),
                                previous_command_radps=requested_qdot,
                                predictive_risk=predictive,
                                safety_jacobian_m_per_rad=jacobian,
                                safety_drift_mps=self._safety_drift_mps(predictive),
                                workspace_constraints=workspace,
                            ),
                            self.safety_filter_config,
                        )
                    except (ValueError, RuntimeError, p.error):
                        scoring_failed = True
                        break
                    if result.requires_safe_stop or not np.isfinite(result.intervention_norm_radps):
                        scoring_failed = True
                        break
                    intervention_values.append(float(result.intervention_norm_radps))
                if (
                    not scoring_failed
                    and h_values
                    and intervention_values
                    and len(h_values) == len(path)
                    and np.isfinite(h_values).all()
                    and np.isfinite(intervention_values).all()
                ):
                    item["predictive_h"] = min(h_values)
                    item["filter_intervention"] = float(np.mean(intervention_values))
                    item["predictive_scored"] = True
                else:
                    item["predictive_h"] = float("nan")
                    item["filter_intervention"] = float("nan")
                    item["predictive_scored"] = False
        finally:
            for joint_id, value, velocity in zip(self.joint_ids, saved_q, saved_qdot, strict=True):
                p.resetJointState(
                    self.robot_id,
                    joint_id,
                    float(value),
                    targetVelocity=float(velocity),
                    physicsClientId=self.physics_client_id,
                )

    def _ik_goal_candidates(
        self,
        current_q: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        preferred_goal_q: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        position = np.asarray(self.goal, dtype=np.float64)
        attempts = max(1, int(self.hierarchical_planner_cfg.get("ik_attempts", 16)))
        fallback_attempts = max(attempts, int(self.hierarchical_planner_cfg.get("ik_fallback_attempts", attempts)))
        candidates: list[np.ndarray] = []
        rest_poses = [np.asarray(current_q, dtype=np.float64)]
        default_pose = np.asarray(self.robot_cfg["reset"]["default_joint_positions"], dtype=np.float64)
        if bool(self.hierarchical_planner_cfg.get("use_structured_rest_poses", False)):
            midpoint = 0.5 * (lower + upper)
            structured = [default_pose, midpoint, np.clip(0.5 * (current_q + midpoint), lower, upper)]
            structured.extend([np.clip(default_pose + sign * 0.35 * (upper - lower), lower, upper) for sign in (-1.0, 1.0)])
            rest_poses.extend(structured)
        else:
            rest_poses.append(default_pose)
        random_count = max(0, attempts - len(rest_poses))
        if random_count:
            rest_poses.extend(self.rng.uniform(lower, upper, size=(random_count, self.joint_count)))
        goal_tolerance = float(self.hierarchical_planner_cfg.get("ik_goal_tolerance_m", 0.02))
        self.hierarchical_ik_candidate_errors_m = []
        self.hierarchical_ik_candidate_clearances_m = []
        self.hierarchical_ik_candidate_kinds = []

        # Preserve a previously selected goal-region terminal across
        # AVOID_HOLD/replan.  Validate it against the current obstacle state;
        # if it is still safe, it is the first planning target and therefore
        # avoids throwing away the only known reachable terminal.
        if preferred_goal_q is not None:
            retained = np.clip(np.asarray(preferred_goal_q, dtype=np.float64), lower, upper)
            retained_error = self._planning_goal_error(retained)
            if retained_error <= goal_tolerance and self._planning_state_validity_reason(retained) is None:
                candidates.append(retained)
                self.hierarchical_ik_candidate_errors_m.append(float(retained_error))
                self.hierarchical_ik_candidate_clearances_m.append(float(self._planning_clearance(retained)))
                self.hierarchical_ik_candidate_kinds.append("retained_goal_region")

        def run_ik_attempts(
            poses: list[np.ndarray],
            target_position: np.ndarray = position,
            candidate_kind: str = "exact",
        ) -> None:
            for rest in poses:
                self.hierarchical_ik_attempts_used += 1
                try:
                    solution = p.calculateInverseKinematics(
                        self.robot_id,
                        self.tool_link_id,
                        np.asarray(target_position, dtype=np.float64).tolist(),
                        lowerLimits=lower.tolist(),
                        upperLimits=upper.tolist(),
                        jointRanges=(upper - lower).tolist(),
                        restPoses=np.asarray(rest, dtype=np.float64).tolist(),
                        maxNumIterations=int(self.hierarchical_planner_cfg.get("ik_max_iterations", 100)),
                        residualThreshold=float(self.hierarchical_planner_cfg.get("ik_residual_threshold", 1.0e-4)),
                        physicsClientId=self.physics_client_id,
                    )
                except (TypeError, ValueError, p.error):
                    self.hierarchical_ik_rejection_counts["solver_error"] += 1
                    continue
                solution = np.asarray(solution, dtype=np.float64)
                if solution.size <= max(self.control_jacobian_columns):
                    self.hierarchical_ik_rejection_counts["short_solution"] += 1
                    continue
                candidate = np.clip(solution[self.control_jacobian_columns], lower, upper)
                if any(np.linalg.norm(candidate - previous) < 1.0e-3 for previous in candidates):
                    self.hierarchical_ik_rejection_counts["duplicate"] += 1
                    continue
                error = self._planning_goal_error(candidate)
                if not np.isfinite(self.hierarchical_ik_min_goal_error_m) or error < self.hierarchical_ik_min_goal_error_m:
                    self.hierarchical_ik_min_goal_error_m = error
                if error <= goal_tolerance:
                    self.hierarchical_ik_goal_reachable_count += 1
                    if self._planning_state_validity_reason(candidate, ignore_obstacle=True) is None:
                        self.hierarchical_ik_obstacle_free_goal_reachable_count += 1
                if error > goal_tolerance:
                    self.hierarchical_ik_rejection_counts["goal_error"] += 1
                    continue
                state_reason = self._planning_state_validity_reason(candidate)
                if state_reason is not None:
                    self.hierarchical_ik_rejection_counts[state_reason] += 1
                    continue
                candidates.append(candidate)
                self.hierarchical_ik_candidate_kinds.append(candidate_kind)
                if candidate_kind == "shell":
                    self.hierarchical_ik_target_shell_candidate_count += 1
                elif candidate_kind == "goal_region":
                    self.hierarchical_ik_goal_region_candidate_count += 1
                self.hierarchical_ik_candidate_errors_m.append(error)
                self.hierarchical_ik_candidate_clearances_m.append(float(self._planning_clearance(candidate)))

        run_ik_attempts(rest_poses[:attempts])
        # When raw IK can reach the goal but every exact-target solution is
        # rejected by the obstacle, locally explore the redundant joint
        # nullspace around those raw solutions.  This keeps the exact target
        # and the existing strict clearance/contact checks unchanged; it only
        # searches for a different configuration of the same TCP pose.
        if (
            not candidates
            and bool(self.hierarchical_planner_cfg.get("ik_obstacle_aware_nullspace_enabled", False))
            and self.hierarchical_ik_goal_reachable_count > 0
            and not self.hierarchical_ik_obstacle_aware_attempted
        ):
            self.hierarchical_ik_obstacle_aware_attempted = True
            null_attempts = max(
                0,
                int(self.hierarchical_planner_cfg.get("ik_obstacle_aware_nullspace_attempts", 0)),
            )
            null_step = float(
                self.hierarchical_planner_cfg.get("ik_obstacle_aware_nullspace_step_rad", 0.12)
            )
            if null_attempts > 0 and null_step > 0.0:
                raw_seeds = []
                # Re-query deterministic IK seeds so this branch remains
                # independent of the accepted-candidate list, which is empty
                # precisely in the blocked case.
                for rest in rest_poses[:attempts]:
                    try:
                        solution = p.calculateInverseKinematics(
                            self.robot_id,
                            self.tool_link_id,
                            position.tolist(),
                            lowerLimits=lower.tolist(),
                            upperLimits=upper.tolist(),
                            jointRanges=(upper - lower).tolist(),
                            restPoses=np.asarray(rest, dtype=np.float64).tolist(),
                            maxNumIterations=int(self.hierarchical_planner_cfg.get("ik_max_iterations", 100)),
                            residualThreshold=float(self.hierarchical_planner_cfg.get("ik_residual_threshold", 1.0e-4)),
                            physicsClientId=self.physics_client_id,
                        )
                    except (TypeError, ValueError, p.error):
                        continue
                    solution = np.asarray(solution, dtype=np.float64)
                    if solution.size > max(self.control_jacobian_columns):
                        raw = np.clip(solution[self.control_jacobian_columns], lower, upper)
                        if (
                            self._planning_goal_error(raw) <= goal_tolerance
                            and not any(np.linalg.norm(raw - previous) < 1.0e-3 for previous in raw_seeds)
                        ):
                            raw_seeds.append(raw)
                for raw in raw_seeds[:null_attempts]:
                    self.hierarchical_ik_obstacle_aware_candidate_count += 1
                    trial = self._obstacle_aware_nullspace_ik(raw, lower, upper)
                    if trial is None:
                        continue
                    error = self._planning_goal_error(trial)
                    if not np.isfinite(self.hierarchical_ik_min_goal_error_m) or error < self.hierarchical_ik_min_goal_error_m:
                        self.hierarchical_ik_min_goal_error_m = error
                    if error > goal_tolerance:
                        self.hierarchical_ik_rejection_counts["goal_error"] += 1
                        continue
                    state_reason = self._planning_state_validity_reason(trial)
                    if state_reason is not None:
                        self.hierarchical_ik_rejection_counts[state_reason] += 1
                        continue
                    if any(np.linalg.norm(trial - previous) < 1.0e-3 for previous in candidates):
                        self.hierarchical_ik_rejection_counts["duplicate"] += 1
                        continue
                    candidates.append(trial)
                    self.hierarchical_ik_candidate_kinds.append("obstacle_aware")
                    self.hierarchical_ik_candidate_errors_m.append(error)
                    self.hierarchical_ik_candidate_clearances_m.append(float(self._planning_clearance(trial)))
        # Search a bounded, obstacle-aware goal region only after exact-target
        # IK (and the optional exact-target nullspace branch) failed.  Every
        # returned solution is still checked against the original goal error,
        # strict d_safe and collision rules by run_ik_attempts above.
        if not candidates and bool(self.hierarchical_planner_cfg.get("ik_goal_region_enabled", False)):
            region_attempts = max(
                0,
                int(self.hierarchical_planner_cfg.get("ik_goal_region_attempts", 0)),
            )
            region_radius = min(
                float(self.hierarchical_planner_cfg.get("ik_goal_region_radius_m", 0.0)),
                goal_tolerance,
            )
            if region_attempts > 0 and region_radius > 0.0 and not self.hierarchical_ik_goal_region_attempted:
                self.hierarchical_ik_goal_region_attempted = True
                self.hierarchical_ik_goal_region_sample_count = region_attempts
                away = position - np.asarray(self.obstacle_center, dtype=np.float64)
                away_norm = float(np.linalg.norm(away))
                away = away / max(away_norm, 1.0e-8)
                # Fibonacci directions provide deterministic, approximately
                # uniform coverage of the positional tolerance ball.  Samples
                # pointing away from the obstacle are tried first.
                directions: list[np.ndarray] = []
                golden = np.pi * (3.0 - np.sqrt(5.0))
                for index in range(max(region_attempts * 2, 16)):
                    z = 1.0 - 2.0 * (index + 0.5) / max(region_attempts * 2, 16)
                    radial = np.sqrt(max(0.0, 1.0 - z * z))
                    angle = golden * index
                    directions.append(
                        np.array([radial * np.cos(angle), radial * np.sin(angle), z], dtype=np.float64)
                    )
                directions.sort(key=lambda direction: -float(np.dot(direction, away)))
                radii = (region_radius, 0.75 * region_radius, 0.5 * region_radius)
                region_targets: list[np.ndarray] = []
                # Interleave radii so the bounded budget covers both the
                # boundary and the interior of the accepted goal ball.
                for index in range(region_attempts):
                    radius = radii[index % len(radii)]
                    direction = directions[index // len(radii) % len(directions)]
                    region_targets.append(position + radius * direction)
                region_rest_poses = rest_poses[: min(len(rest_poses), len(region_targets))]
                if len(region_rest_poses) < len(region_targets):
                    region_rest_poses.extend(
                        self.rng.uniform(
                            lower,
                            upper,
                            size=(len(region_targets) - len(region_rest_poses), self.joint_count),
                        )
                    )
                for target_position, rest in zip(region_targets, region_rest_poses, strict=True):
                    run_ik_attempts([rest], target_position=target_position, candidate_kind="goal_region")

        if not candidates and bool(self.hierarchical_planner_cfg.get("ik_target_shell_enabled", False)):
            shell_attempts = max(0, int(self.hierarchical_planner_cfg.get("ik_target_shell_attempts", 0)))
            shell_radius = float(self.hierarchical_planner_cfg.get("ik_target_shell_radius_m", 0.0))
            if shell_attempts > 0 and shell_radius > 0.0 and not self.hierarchical_ik_target_shell_attempted:
                self.hierarchical_ik_target_shell_attempted = True
                shell_directions = [
                    np.array([1.0, 0.0, 0.0]),
                    np.array([-1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, -1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                    np.array([0.0, 0.0, -1.0]),
                    np.array([1.0, 1.0, 0.0]),
                    np.array([1.0, -1.0, 0.0]),
                    np.array([-1.0, 1.0, 0.0]),
                    np.array([-1.0, -1.0, 0.0]),
                    np.array([1.0, 0.0, 1.0]),
                    np.array([1.0, 0.0, -1.0]),
                    np.array([-1.0, 0.0, 1.0]),
                    np.array([-1.0, 0.0, -1.0]),
                    np.array([0.0, 1.0, 1.0]),
                    np.array([0.0, 1.0, -1.0]),
                    np.array([0.0, -1.0, 1.0]),
                    np.array([0.0, -1.0, -1.0]),
                ]
                if self.obstacle_enabled:
                    away = position - np.asarray(self.obstacle_center, dtype=np.float64)
                    away_norm = float(np.linalg.norm(away))
                    if away_norm > 1.0e-8:
                        shell_directions.insert(0, away / away_norm)
                shell_directions = [direction / max(float(np.linalg.norm(direction)), 1.0e-8) for direction in shell_directions]
                shell_targets: list[np.ndarray] = []
                radii = (shell_radius, 0.5 * shell_radius)
                for radius in radii:
                    for direction in shell_directions:
                        shell_targets.append(position + radius * direction)
                shell_targets = shell_targets[:shell_attempts]
                shell_rest_poses = rest_poses[: min(len(rest_poses), shell_attempts)]
                if len(shell_rest_poses) < len(shell_targets):
                    shell_rest_poses.extend(
                        self.rng.uniform(lower, upper, size=(len(shell_targets) - len(shell_rest_poses), self.joint_count))
                    )
                for target_position, rest in zip(shell_targets, shell_rest_poses, strict=True):
                    run_ik_attempts([rest], target_position=target_position, candidate_kind="shell")
        if not candidates and fallback_attempts > attempts and not self.hierarchical_ik_fallback_attempted:
            self.hierarchical_ik_fallback_used = True
            self.hierarchical_ik_fallback_attempted = True
            fallback_count = fallback_attempts - attempts
            fallback_poses = [
                np.clip(0.5 * (current_q + lower), lower, upper),
                np.clip(0.5 * (current_q + upper), lower, upper),
                np.clip(0.5 * (default_pose + lower), lower, upper),
                np.clip(0.5 * (default_pose + upper), lower, upper),
            ]
            fallback_random_count = max(0, fallback_count - len(fallback_poses))
            if fallback_random_count:
                fallback_poses.extend(
                    self.rng.uniform(lower, upper, size=(fallback_random_count, self.joint_count))
                )
            run_ik_attempts(fallback_poses[:fallback_count])

        dls_seeds = [np.asarray(current_q, dtype=np.float64)]
        if self.hierarchical_ik_fallback_used:
            dls_seed_count = max(1, int(self.hierarchical_planner_cfg.get("ik_fallback_dls_seeds", 4)))
            dls_seeds.extend(
                [
                    np.clip(0.5 * (current_q + lower), lower, upper),
                    np.clip(0.5 * (current_q + upper), lower, upper),
                    np.clip(default_pose, lower, upper),
                ][: max(0, dls_seed_count - 1)]
            )
        if bool(self.hierarchical_planner_cfg.get("use_dls_fallback", False)):
            for dls_seed in dls_seeds:
                dls_candidate = self._dls_position_ik(dls_seed, lower, upper)
                if dls_candidate is None:
                    continue
                if any(np.linalg.norm(dls_candidate - previous) < 1.0e-3 for previous in candidates):
                    self.hierarchical_ik_rejection_counts["duplicate"] += 1
                    continue
                error = self._planning_goal_error(dls_candidate)
                if not np.isfinite(self.hierarchical_ik_min_goal_error_m) or error < self.hierarchical_ik_min_goal_error_m:
                    self.hierarchical_ik_min_goal_error_m = error
                if error <= goal_tolerance:
                    self.hierarchical_ik_goal_reachable_count += 1
                    if self._planning_state_validity_reason(dls_candidate, ignore_obstacle=True) is None:
                        self.hierarchical_ik_obstacle_free_goal_reachable_count += 1
                state_reason = self._planning_state_validity_reason(dls_candidate)
                if state_reason is None and error <= goal_tolerance:
                    candidates.append(dls_candidate)
                    self.hierarchical_ik_candidate_kinds.append("dls")
                    self.hierarchical_ik_candidate_errors_m.append(error)
                    self.hierarchical_ik_candidate_clearances_m.append(float(self._planning_clearance(dls_candidate)))
                else:
                    if error > goal_tolerance:
                        self.hierarchical_ik_rejection_counts["goal_error"] += 1
                    elif state_reason is not None:
                        self.hierarchical_ik_rejection_counts[state_reason] += 1
        return candidates

    def _obstacle_aware_nullspace_ik(
        self, seed: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray | None:
        """Search a redundant exact-target IK neighbourhood for clearance.

        The target correction is solved in task space while a finite-difference
        clearance gradient is projected into the Jacobian nullspace.  This is
        deliberately bounded and only used after raw exact-target IK has been
        shown reachable but all returned configurations failed obstacle checks.
        """
        q = np.clip(np.asarray(seed, dtype=np.float64).copy(), lower, upper)
        iterations = max(1, int(self.hierarchical_planner_cfg.get("ik_obstacle_aware_nullspace_iterations", 60)))
        step = float(self.hierarchical_planner_cfg.get("ik_obstacle_aware_nullspace_step_rad", 0.08))
        gradient_gain = float(self.hierarchical_planner_cfg.get("ik_obstacle_aware_nullspace_gradient_gain", 1.0))
        finite_step = float(self.hierarchical_planner_cfg.get("ik_obstacle_aware_nullspace_finite_difference_rad", 0.015))
        damping = float(self.hierarchical_planner_cfg.get("ik_obstacle_aware_nullspace_damping", 0.08))
        goal_tolerance = float(self.hierarchical_planner_cfg.get("ik_goal_tolerance_m", 0.055))
        saved_q, saved_qdot = self._joint_state()
        try:
            for _ in range(iterations):
                for joint_id, value in zip(self.joint_ids, q, strict=True):
                    p.resetJointState(self.robot_id, joint_id, float(value), targetVelocity=0.0, physicsClientId=self.physics_client_id)
                ee_position, _ = self._end_effector_state()
                error = np.asarray(self.goal, dtype=np.float64) - np.asarray(ee_position, dtype=np.float64)
                jacobian = np.asarray(self._link_origin_jacobian(self.tool_link_id), dtype=np.float64)
                if np.linalg.norm(error) <= goal_tolerance and self._planning_clearance(q) >= float(self.risk_config.d_safe):
                    return q.copy()
                clearance_gradient = np.zeros(self.joint_count, dtype=np.float64)
                for index in range(self.joint_count):
                    q_plus = q.copy(); q_plus[index] = min(upper[index], q_plus[index] + finite_step)
                    q_minus = q.copy(); q_minus[index] = max(lower[index], q_minus[index] - finite_step)
                    denominator = max(q_plus[index] - q_minus[index], 1.0e-6)
                    clearance_gradient[index] = (self._planning_clearance(q_plus) - self._planning_clearance(q_minus)) / denominator
                projector = np.eye(self.joint_count) - np.linalg.pinv(jacobian) @ jacobian
                task_update = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping**2 * np.eye(3), error
                )
                null_update = projector @ clearance_gradient
                q = np.clip(q + step * (task_update + gradient_gain * null_update), lower, upper)
            return q if self._planning_goal_error(q) <= goal_tolerance else None
        finally:
            for joint_id, value, velocity in zip(self.joint_ids, saved_q, saved_qdot, strict=True):
                p.resetJointState(self.robot_id, joint_id, float(value), targetVelocity=float(velocity), physicsClientId=self.physics_client_id)

    def _planning_clearance(self, q: np.ndarray) -> float:
        saved_q, saved_qdot = self._joint_state()
        try:
            for joint_id, value in zip(self.joint_ids, q, strict=True):
                p.resetJointState(self.robot_id, joint_id, float(value), targetVelocity=0.0, physicsClientId=self.physics_client_id)
            p.performCollisionDetection(physicsClientId=self.physics_client_id)
            return float(self._compute_risk().d_min)
        finally:
            for joint_id, value, velocity in zip(self.joint_ids, saved_q, saved_qdot, strict=True):
                p.resetJointState(self.robot_id, joint_id, float(value), targetVelocity=float(velocity), physicsClientId=self.physics_client_id)

    def _dls_position_ik(self, seed: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray | None:
        """Small position-only damped-least-squares fallback for difficult IK seeds."""
        q = np.clip(np.asarray(seed, dtype=np.float64).copy(), lower, upper)
        iterations = int(self.hierarchical_planner_cfg.get("dls_max_iterations", 240))
        damping = float(self.hierarchical_planner_cfg.get("dls_damping", 0.08))
        step = float(self.hierarchical_planner_cfg.get("dls_step_size", 0.7))
        saved_q, saved_qdot = self._joint_state()
        try:
            for _ in range(max(1, iterations)):
                for joint_id, value in zip(self.joint_ids, q, strict=True):
                    p.resetJointState(self.robot_id, joint_id, float(value), targetVelocity=0.0, physicsClientId=self.physics_client_id)
                error = np.asarray(self.goal, dtype=np.float64) - np.asarray(self._end_effector_state()[0], dtype=np.float64)
                if np.linalg.norm(error) <= float(self.hierarchical_planner_cfg.get("dls_goal_tolerance_m", 0.01)):
                    return q.copy()
                jacobian = self._link_origin_jacobian(self.tool_link_id)
                update = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping**2 * np.eye(3), error)
                q = np.clip(q + step * update, lower, upper)
            return q if self._planning_goal_error(q) <= float(self.hierarchical_planner_cfg.get("dls_goal_tolerance_m", 0.01)) else None
        finally:
            for joint_id, value, velocity in zip(self.joint_ids, saved_q, saved_qdot, strict=True):
                p.resetJointState(self.robot_id, joint_id, float(value), targetVelocity=float(velocity), physicsClientId=self.physics_client_id)

    def _planning_goal_error(self, q: np.ndarray) -> float:
        saved_q, saved_qdot = self._joint_state()
        try:
            for joint_id, value in zip(self.joint_ids, q, strict=True):
                p.resetJointState(self.robot_id, joint_id, float(value), targetVelocity=0.0, physicsClientId=self.physics_client_id)
            ee_position, _ = self._end_effector_state()
            return float(np.linalg.norm(np.asarray(self.goal, dtype=np.float64) - ee_position))
        finally:
            for joint_id, value, velocity in zip(self.joint_ids, saved_q, saved_qdot, strict=True):
                p.resetJointState(self.robot_id, joint_id, float(value), targetVelocity=float(velocity), physicsClientId=self.physics_client_id)

    def _planning_state_validity_reason(self, q: np.ndarray, *, ignore_obstacle: bool = False) -> str | None:
        q = np.asarray(q, dtype=np.float64)
        lower, upper = self._joint_position_limits()
        if q.shape != lower.shape or not np.isfinite(q).all() or np.any(q < lower) or np.any(q > upper):
            return "joint_limit"
        saved_q, saved_qdot = self._joint_state()
        try:
            for joint_id, value in zip(self.joint_ids, q, strict=True):
                p.resetJointState(self.robot_id, joint_id, float(value), targetVelocity=0.0, physicsClientId=self.physics_client_id)
            ee_position, _ = self._end_effector_state()
            boundary_tolerance = float(self.hierarchical_planner_cfg.get("boundary_start_tolerance_m", 0.0))
            is_planning_start = self._hierarchical_planning_start_q is not None and np.linalg.norm(q - self._hierarchical_planning_start_q) <= 1.0e-8
            if any(
                not (float(limits[0]) - (boundary_tolerance if is_planning_start else 0.0) <= float(ee_position[index]) <= float(limits[1]) + (boundary_tolerance if is_planning_start else 0.0))
                for index, limits in enumerate((self.workspace["x"], self.workspace["y"], self.workspace["z"]))
            ):
                return "workspace"
            p.performCollisionDetection(physicsClientId=self.physics_client_id)
            if self.obstacle_enabled and not ignore_obstacle:
                if self.obstacle_id is not None and p.getContactPoints(
                    bodyA=self.robot_id, bodyB=self.obstacle_id, physicsClientId=self.physics_client_id
                ):
                    return "obstacle_contact"
                clearance = float(self.hierarchical_planner_cfg.get("clearance_m", self.risk_config.d_safe))
                if self._compute_risk().d_min < clearance:
                    return "obstacle_clearance"
            self_collision = p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id, physicsClientId=self.physics_client_id)
            if self_collision:
                return "self_collision"
            return None
        finally:
            for joint_id, value, velocity in zip(self.joint_ids, saved_q, saved_qdot, strict=True):
                p.resetJointState(self.robot_id, joint_id, float(value), targetVelocity=float(velocity), physicsClientId=self.physics_client_id)

    def _planning_state_is_valid(self, q: np.ndarray) -> bool:
        """Boolean adapter used by the RRT planner and edge checker."""
        return self._planning_state_validity_reason(q) is None

    def _planning_edge_is_valid(self, start: np.ndarray, end: np.ndarray) -> bool:
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        resolution = float(self.hierarchical_planner_cfg.get("edge_resolution_rad", 0.06))
        segments = max(1, int(np.ceil(np.max(np.abs(end - start), initial=0.0) / resolution)))
        return all(
            self._planning_state_is_valid(start + fraction * (end - start))
            for fraction in np.linspace(0.0, 1.0, segments + 1)[1:]
        )

    def _hierarchical_nominal_command(
        self,
        current_risk: LinkRisk,
        *,
        update_state: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        q, _ = self._joint_state()
        retry_interval = int(self.hierarchical_cfg.get("plan_retry_interval_steps", 20))
        if (
            update_state
            and self.hierarchical_state == "PLAN_FAILED"
            and self.step_count - self.hierarchical_last_plan_step >= retry_interval
        ):
            self._initialize_hierarchical_plan()
            self.hierarchical_last_plan_step = self.step_count
        if update_state and self.hierarchical_state == "REPLAN":
            self.hierarchical_replan_count += 1
            self.hierarchical_consecutive_hold_steps = 0
            self._initialize_hierarchical_plan()
            self.hierarchical_last_plan_step = self.step_count
        predictive_h = float("inf")
        if self.obstacle_enabled and self.safety_filter_enabled:
            predictive = self._compute_predictive_risk()
            if predictive.usable:
                predictive_h = float(np.min(predictive.safety_functions_m))
        hold_margin = float(self.hierarchical_cfg.get("hold_margin_m", 0.04))
        release_margin = float(self.hierarchical_cfg.get("release_margin_m", 0.08))
        if update_state:
            if self.hierarchical_state in {"TRACK", "SERVO"} and predictive_h <= hold_margin:
                self.hierarchical_state = "AVOID_HOLD"
                self.hierarchical_consecutive_hold_steps = 0
            if self.hierarchical_state == "AVOID_HOLD":
                self.hierarchical_hold_steps += 1
                self.hierarchical_consecutive_hold_steps += 1
                replan_after = int(self.hierarchical_cfg.get("replan_after_hold_steps", 8))
                if predictive_h >= release_margin or self.hierarchical_consecutive_hold_steps >= replan_after:
                    self.hierarchical_state = "REPLAN"
        if not self.hierarchical_path:
            self.hierarchical_last_nominal_qdot = np.zeros(self.joint_count, dtype=np.float32)
            self.hierarchical_last_waypoint_error = np.zeros(self.joint_count, dtype=np.float32)
            self.hierarchical_last_progress = 0.0
            self.hierarchical_last_residual_budget = 0.0
            return np.zeros(self.joint_count, dtype=np.float32), self._hierarchical_info(current_risk, predictive_h)
        index = min(self.hierarchical_waypoint_index, len(self.hierarchical_path) - 1)
        waypoint = self.hierarchical_path[index]
        waypoint_error = waypoint - q
        tolerance = float(self.hierarchical_tracker_cfg.get("waypoint_tolerance_rad", 0.08))
        while update_state and index < len(self.hierarchical_path) - 1 and np.linalg.norm(waypoint_error) <= tolerance:
            index += 1
            waypoint = self.hierarchical_path[index]
            waypoint_error = waypoint - q
        if update_state:
            self.hierarchical_waypoint_index = index
        if update_state and index == len(self.hierarchical_path) - 1 and np.linalg.norm(self._goal_error()) <= float(self.hierarchical_tracker_cfg.get("servo_trigger_m", 0.10)):
            self.hierarchical_state = "SERVO"
        if update_state and self.hierarchical_state == "SERVO":
            goal_error_norm = float(np.linalg.norm(self._goal_error()))
            improvement = float(self.prev_goal_error_norm - goal_error_norm)
            improvement_threshold = float(self.hierarchical_tracker_cfg.get("servo_stall_improvement_m", 0.002))
            if improvement <= improvement_threshold:
                self.hierarchical_servo_stall_steps += 1
            else:
                self.hierarchical_servo_stall_steps = 0
            stall_steps = int(self.hierarchical_tracker_cfg.get("servo_stall_steps", 20))
            filter_threshold = float(self.hierarchical_tracker_cfg.get("servo_stall_filter_ratio", 0.20))
            replan_limit = int(self.hierarchical_tracker_cfg.get("servo_stall_replan_limit", 2))
            if (
                self.hierarchical_servo_stall_steps >= stall_steps
                and self.hierarchical_last_filter_intervention_ratio >= filter_threshold
                and self.hierarchical_servo_stall_replans < replan_limit
            ):
                self.hierarchical_state = "REPLAN"
        if update_state and self.hierarchical_state == "REPLAN":
            self.hierarchical_replan_count += 1
            if self.hierarchical_servo_stall_steps > 0:
                self.hierarchical_servo_stall_replans += 1
            self.hierarchical_servo_stall_steps = 0
            self.hierarchical_consecutive_hold_steps = 0
            self._initialize_hierarchical_plan()
            self.hierarchical_last_plan_step = self.step_count
        if self.hierarchical_state in {"AVOID_HOLD", "REPLAN", "PLAN_FAILED"}:
            # A retained goal-region terminal already has a collision-checked
            # RRT path.  Continue tracking that path during AVOID_HOLD and let
            # the strict filter project the command, instead of replacing it
            # with the recovery command that previously drove this case away
            # from the success ball.
            if self.hierarchical_state == "AVOID_HOLD" and self.hierarchical_selected_terminal_is_goal_region:
                hold_gain = float(self.hierarchical_tracker_cfg.get("goal_region_hold_waypoint_gain", 1.0))
                nominal = hold_gain * waypoint_error
            else:
                nominal = np.zeros(self.joint_count, dtype=np.float32)
        elif self.hierarchical_state == "SERVO":
            gain = float(self.hierarchical_tracker_cfg.get("servo_gain", 2.5))
            damping = float(self.hierarchical_tracker_cfg.get("servo_damping", 0.05))
            goal_error = self._goal_error().astype(np.float64)
            ee_velocity = np.asarray(self._end_effector_state()[1], dtype=np.float64)
            velocity_damping = float(self.hierarchical_tracker_cfg.get("servo_velocity_damping", 0.0))
            target_velocity = gain * goal_error - velocity_damping * ee_velocity
            near_goal_radius = float(self.hierarchical_tracker_cfg.get("servo_near_goal_radius_m", 0.08))
            if np.linalg.norm(goal_error) <= near_goal_radius:
                gain *= float(self.hierarchical_tracker_cfg.get("servo_near_goal_gain_scale", 0.65))
                target_velocity = gain * goal_error - velocity_damping * ee_velocity
            jacobian = self._link_origin_jacobian(self.tool_link_id)
            if bool(self.hierarchical_tracker_cfg.get("adaptive_damping", False)):
                damping_near = float(self.hierarchical_tracker_cfg.get("servo_damping_near", max(damping * 0.35, 1.0e-3)))
                damping_far = float(self.hierarchical_tracker_cfg.get("servo_damping_far", max(damping * 2.0, damping)))
                error_scale = float(self.hierarchical_tracker_cfg.get("adaptive_damping_error_m", 0.10))
                blend = float(np.clip(np.linalg.norm(goal_error) / max(error_scale, 1.0e-6), 0.0, 1.0))
                damping = damping_near + blend * (damping_far - damping_near)
            nominal = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping**2 * np.eye(3), target_velocity)
            if bool(self.hierarchical_tracker_cfg.get("nullspace_limit_avoidance", False)):
                lower, upper = self._joint_position_limits()
                midpoint = 0.5 * (lower + upper)
                half_range = np.maximum(0.5 * (upper - lower), 1.0e-3)
                null_gain = float(self.hierarchical_tracker_cfg.get("nullspace_gain", 0.08))
                null_velocity = null_gain * (midpoint - q.astype(np.float64)) / half_range
                projector = np.eye(self.joint_count) - np.linalg.pinv(jacobian) @ jacobian
                nominal = nominal + projector @ null_velocity
            if bool(self.hierarchical_tracker_cfg.get("filter_aware_servo", False)):
                intervention = float(np.clip(self.hierarchical_last_filter_intervention_ratio, 0.0, 1.0))
                threshold = float(self.hierarchical_tracker_cfg.get("filter_intervention_threshold", 0.15))
                gain_floor = float(self.hierarchical_tracker_cfg.get("filter_aware_gain_floor", 0.35))
                blend_strength = float(self.hierarchical_tracker_cfg.get("filter_aware_blend_strength", 0.75))
                if intervention > threshold:
                    excess = (intervention - threshold) / max(1.0 - threshold, 1.0e-6)
                    scale = 1.0 - excess * (1.0 - gain_floor)
                    nominal *= float(np.clip(scale, gain_floor, 1.0))
                    previous = self.hierarchical_last_filtered_qdot.astype(np.float64)
                    if np.linalg.norm(previous) > 1.0e-8 and np.linalg.norm(nominal) > 1.0e-8:
                        alignment = float(np.dot(previous, nominal) / (np.linalg.norm(previous) * np.linalg.norm(nominal)))
                        if alignment >= float(self.hierarchical_tracker_cfg.get("filter_aware_min_alignment", 0.0)):
                            blend = float(np.clip(blend_strength * excess, 0.0, 0.85))
                            nominal = (1.0 - blend) * nominal + blend * previous
        else:
            gain = float(self.hierarchical_tracker_cfg.get("waypoint_gain", 1.8))
            if bool(self.hierarchical_tracker_cfg.get("clearance_aware_gain", False)) and np.isfinite(predictive_h):
                low = float(self.hierarchical_tracker_cfg.get("clearance_gain_low_m", 0.04))
                high = float(self.hierarchical_tracker_cfg.get("clearance_gain_high_m", 0.16))
                gain *= float(np.clip((predictive_h - low) / max(high - low, 1.0e-6), 0.35, 1.0))
            nominal = gain * waypoint_error
        nominal = np.clip(nominal, -self.action_scale, self.action_scale).astype(np.float32)
        self.hierarchical_last_nominal_qdot = nominal.copy()
        self.hierarchical_last_waypoint_error = waypoint_error.astype(np.float32)
        self.hierarchical_last_progress = float(index / max(len(self.hierarchical_path) - 1, 1))
        budget = self._hierarchical_residual_budget(predictive_h)
        if self.hierarchical_state in {"AVOID_HOLD", "REPLAN", "PLAN_FAILED"}:
            budget = 0.0
        self.hierarchical_last_residual_budget = budget
        return nominal, self._hierarchical_info(current_risk, predictive_h)

    def _hierarchical_residual_budget(self, predictive_h: float) -> float:
        if not np.isfinite(predictive_h):
            return 1.0
        start = float(self.hierarchical_residual_cfg.get("risk_start_m", 0.20))
        stop = float(self.hierarchical_residual_cfg.get("risk_stop_m", 0.04))
        return float(np.clip((predictive_h - stop) / max(start - stop, 1.0e-6), 0.0, 1.0))

    def _hierarchical_info(self, current_risk: LinkRisk, predictive_h: float) -> dict[str, Any]:
        return {
            "hierarchical_control_enabled": True,
            "hierarchical_state": self.hierarchical_state,
            "hierarchical_plan_reason": self.hierarchical_plan_reason,
            "hierarchical_plan_iterations": self.hierarchical_plan_iterations,
            "hierarchical_planning_time_s": self.hierarchical_planning_time_s,
            "hierarchical_planning_time_total_s": self.hierarchical_planning_time_total_s,
            "hierarchical_ik_candidate_count": self.hierarchical_ik_candidate_count,
            "hierarchical_ik_found": self.hierarchical_ik_candidate_count > 0,
            "hierarchical_plan_found": bool(self.hierarchical_path),
            "hierarchical_direct_path": self.hierarchical_plan_reason == "direct_path",
            "hierarchical_plan_count": self.hierarchical_plan_count,
            "hierarchical_replan_count": self.hierarchical_replan_count,
            "hierarchical_hold_steps": self.hierarchical_hold_steps,
            "hierarchical_consecutive_hold_steps": self.hierarchical_consecutive_hold_steps,
            "hierarchical_waypoint_index": self.hierarchical_waypoint_index,
            "hierarchical_waypoint_count": len(self.hierarchical_path),
            "hierarchical_path_progress": self.hierarchical_last_progress,
            "hierarchical_nominal_qdot": self.hierarchical_last_nominal_qdot.copy(),
            "hierarchical_residual_budget": self.hierarchical_last_residual_budget,
            "hierarchical_waypoint_error": self.hierarchical_last_waypoint_error.copy(),
            "hierarchical_predictive_h_min_m": predictive_h,
            "hierarchical_current_d_min_m": float(current_risk.d_min),
            "hierarchical_selected_candidate_index": self.hierarchical_selected_candidate_index,
            "hierarchical_selected_terminal_is_goal_region": self.hierarchical_selected_terminal_is_goal_region,
            "hierarchical_selected_path_length_rad": self.hierarchical_selected_path_length_rad,
            "hierarchical_selected_path_min_clearance_m": self.hierarchical_selected_path_min_clearance_m,
            "hierarchical_ik_candidate_errors_m": tuple(self.hierarchical_ik_candidate_errors_m),
            "hierarchical_ik_candidate_clearances_m": tuple(self.hierarchical_ik_candidate_clearances_m),
            "hierarchical_ik_attempts_used": self.hierarchical_ik_attempts_used,
            "hierarchical_ik_fallback_used": self.hierarchical_ik_fallback_used,
            "hierarchical_ik_fallback_attempted": self.hierarchical_ik_fallback_attempted,
            "hierarchical_ik_target_shell_attempted": self.hierarchical_ik_target_shell_attempted,
            "hierarchical_ik_target_shell_candidate_count": self.hierarchical_ik_target_shell_candidate_count,
            "hierarchical_ik_goal_region_attempted": self.hierarchical_ik_goal_region_attempted,
            "hierarchical_ik_goal_region_candidate_count": self.hierarchical_ik_goal_region_candidate_count,
            "hierarchical_ik_goal_region_sample_count": self.hierarchical_ik_goal_region_sample_count,
            "hierarchical_ik_obstacle_aware_attempted": self.hierarchical_ik_obstacle_aware_attempted,
            "hierarchical_ik_obstacle_aware_candidate_count": self.hierarchical_ik_obstacle_aware_candidate_count,
            "hierarchical_ik_rejection_counts": dict(self.hierarchical_ik_rejection_counts),
            "hierarchical_ik_min_goal_error_m": self.hierarchical_ik_min_goal_error_m,
            "hierarchical_ik_goal_reachable_count": self.hierarchical_ik_goal_reachable_count,
            "hierarchical_ik_obstacle_free_goal_reachable_count": self.hierarchical_ik_obstacle_free_goal_reachable_count,
            "hierarchical_selected_path_predictive_h_m": self.hierarchical_selected_path_predictive_h_m,
            "hierarchical_selected_path_filter_intervention": self.hierarchical_selected_path_filter_intervention,
            "hierarchical_filter_intervention_ratio": self.hierarchical_last_filter_intervention_ratio,
            "hierarchical_filter_status": self.hierarchical_last_filter_status,
            "hierarchical_servo_stall_steps": self.hierarchical_servo_stall_steps,
            "hierarchical_servo_stall_replans": self.hierarchical_servo_stall_replans,
        }

    def _apply_reset_jitter(self, options: dict[str, Any] | None) -> None:
        if not options:
            return
        jitter_cfg = options.get("reset_jitter")
        if jitter_cfg is None:
            return
        if not isinstance(jitter_cfg, dict):
            raise TypeError("reset_jitter option must be a mapping")
        rng = np.random.default_rng(jitter_cfg.get("seed"))

        joint_range = float(jitter_cfg.get("joint_noise_range_rad", 0.0))
        if not np.isfinite(joint_range) or joint_range < 0.0:
            raise ValueError("reset_jitter.joint_noise_range_rad must be finite and non-negative")
        if joint_range > 0.0:
            lower, upper = self._joint_position_limits()
            q, _ = self._joint_state()
            q = np.clip(q + rng.uniform(-joint_range, joint_range, size=self.joint_count), lower, upper)
            for joint_id, joint_value in zip(self.joint_ids, q, strict=True):
                p.resetJointState(
                    self.robot_id,
                    joint_id,
                    float(joint_value),
                    targetVelocity=0.0,
                    physicsClientId=self.physics_client_id,
                )

        goal_radius = float(jitter_cfg.get("goal_radius_m", 0.0))
        if not np.isfinite(goal_radius) or goal_radius < 0.0:
            raise ValueError("reset_jitter.goal_radius_m must be finite and non-negative")
        if goal_radius > 0.0:
            self.goal = self._jitter_vector(
                self.goal,
                goal_radius,
                rng,
                limits=(self.workspace["x"], self.workspace["y"], self.workspace["z"]),
            ).astype(np.float32)

        obstacle_radius = float(jitter_cfg.get("obstacle_radius_m", 0.0))
        if not np.isfinite(obstacle_radius) or obstacle_radius < 0.0:
            raise ValueError("reset_jitter.obstacle_radius_m must be finite and non-negative")
        if obstacle_radius > 0.0 and self.obstacle_enabled:
            bounds = self.obstacle_cfg["bounds"]
            self.obstacle_center = self._jitter_vector(
                self.obstacle_center,
                obstacle_radius,
                rng,
                limits=(bounds["x"], bounds["y"], bounds["z"]),
            ).astype(np.float32)

    @staticmethod
    def _jitter_vector(
        value: np.ndarray,
        radius: float,
        rng: np.random.Generator,
        *,
        limits: tuple[list[float], list[float], list[float]],
    ) -> np.ndarray:
        jitter = rng.uniform(-radius, radius, size=3)
        jittered = np.asarray(value, dtype=np.float64) + jitter
        lower = np.asarray([axis[0] for axis in limits], dtype=np.float64)
        upper = np.asarray([axis[1] for axis in limits], dtype=np.float64)
        return np.clip(jittered, lower, upper)

    def _residual_base_command(self, current_risk: LinkRisk) -> tuple[np.ndarray, dict[str, Any]]:
        """Build a deterministic waypoint/terminal velocity command.

        The planner only chooses a Cartesian intermediate target.  A damped
        least-squares Jacobian controller then converts that target into joint
        velocity, leaving the actor responsible for the residual.
        """
        ee_position, _ = self._end_effector_state()
        goal = np.asarray(self.goal, dtype=np.float64)
        goal_error = goal - np.asarray(ee_position, dtype=np.float64)
        goal_error_norm = float(np.linalg.norm(goal_error))
        terminal_radius = float(self.residual_control_cfg.get("terminal_goal_radius_m", 0.12))
        clearance_margin = float(self.residual_control_cfg.get("clearance_margin_m", 0.03))
        safe_margin = float(self.risk_config.d_safe) + clearance_margin
        safe_to_use_terminal = (
            not self.obstacle_enabled
            or (np.isfinite(current_risk.d_min) and current_risk.d_min > safe_margin)
        )

        waypoint, waypoint_active, waypoint_reason = self._planner_waypoint(ee_position, goal)
        if goal_error_norm <= terminal_radius and safe_to_use_terminal:
            target = goal
            mode = "terminal_clf"
            target_reason = "terminal_clearance_ok"
        elif waypoint_active:
            target = waypoint
            mode = "waypoint"
            target_reason = waypoint_reason
        elif not safe_to_use_terminal:
            target = np.asarray(ee_position, dtype=np.float64)
            mode = "hold_for_clearance"
            target_reason = "insufficient_clearance_for_goal_clf"
        else:
            target = goal
            mode = "goal_clf"
            target_reason = "direct_path_clear"

        target_error = target - np.asarray(ee_position, dtype=np.float64)
        gain = (
            float(self.residual_control_cfg.get("terminal_gain", 3.0))
            if mode == "terminal_clf"
            else float(self.residual_control_cfg.get("waypoint_gain", 2.0))
        )
        damping = float(self.residual_control_cfg.get("damping", 0.05))
        try:
            jacobian = self._link_origin_jacobian(self.tool_link_id)
            regularizer = damping**2
            target_qdot = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + regularizer * np.eye(3),
                gain * target_error,
            )
            controller_reason = "active"
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as error:
            target_qdot = np.zeros(self.joint_count, dtype=np.float64)
            controller_reason = f"jacobian_failure:{error}"

        avoidance_qdot, avoidance_info = self._link_avoidance_command(current_risk)
        base_qdot = target_qdot + avoidance_qdot
        base_speed_scale = float(self.residual_control_cfg.get("base_speed_scale", 1.0))
        base_qdot = np.clip(
            base_speed_scale * base_qdot,
            -self.action_scale,
            self.action_scale,
        ).astype(np.float32)
        return base_qdot, {
            "residual_control_mode": mode,
            "residual_control_reason": target_reason,
            "residual_control_controller": controller_reason,
            "residual_control_terminal_safe": bool(safe_to_use_terminal),
            "residual_control_goal_error_norm": goal_error_norm,
            "residual_control_target": np.asarray(target, dtype=np.float32),
            "residual_control_waypoint_active": bool(waypoint_active),
            "residual_control_waypoint": np.asarray(waypoint, dtype=np.float32),
            "residual_control_target_qdot": np.asarray(target_qdot, dtype=np.float32),
            **avoidance_info,
            "residual_control_base_qdot": base_qdot.copy(),
        }

    def _link_avoidance_command(self, current_risk: LinkRisk) -> tuple[np.ndarray, dict[str, Any]]:
        enabled = bool(self.residual_control_cfg.get("link_avoidance_enabled", True))
        inactive_info: dict[str, Any] = {
            "residual_control_link_avoidance_active": False,
            "residual_control_link_avoidance_reason": "disabled" if not enabled else "inactive",
            "residual_control_link_avoidance_qdot": np.zeros(self.joint_count, dtype=np.float32),
            "residual_control_link_avoidance_closest_link": int(current_risk.closest_link),
            "residual_control_link_avoidance_d_min": float(current_risk.d_min),
            "residual_control_link_avoidance_weight": 0.0,
        }
        if not enabled or not self.obstacle_enabled:
            if not self.obstacle_enabled:
                inactive_info["residual_control_link_avoidance_reason"] = "obstacle_disabled"
            return np.zeros(self.joint_count, dtype=np.float32), inactive_info
        closest_link = int(current_risk.closest_link)
        if closest_link < 0 or closest_link >= self.capsule_model.count:
            inactive_info["residual_control_link_avoidance_reason"] = "invalid_closest_link"
            return np.zeros(self.joint_count, dtype=np.float32), inactive_info
        d_min = float(current_risk.d_min)
        if not np.isfinite(d_min):
            inactive_info["residual_control_link_avoidance_reason"] = "nonfinite_distance"
            return np.zeros(self.joint_count, dtype=np.float32), inactive_info

        clearance_margin = float(self.residual_control_cfg.get("clearance_margin_m", 0.03))
        activation_margin = float(self.residual_control_cfg.get("link_avoidance_activation_margin_m", 0.08))
        activation_distance = float(self.risk_config.d_safe) + clearance_margin + activation_margin
        if d_min >= activation_distance:
            inactive_info["residual_control_link_avoidance_reason"] = "clearance_ok"
            return np.zeros(self.joint_count, dtype=np.float32), inactive_info

        capsules = self._capsules()
        capsule = capsules[closest_link]
        _, segment_fraction = closest_point_on_segment(
            np.asarray(self.obstacle_center, dtype=np.float64),
            np.asarray(capsule.start, dtype=np.float64),
            np.asarray(capsule.end, dtype=np.float64),
        )
        start_jacobian, end_jacobian = self._capsule_endpoint_jacobians(self.capsule_model.specs[closest_link])
        point_jacobian = (1.0 - segment_fraction) * start_jacobian + segment_fraction * end_jacobian

        escape_direction = -np.asarray(current_risk.directions[closest_link], dtype=np.float64)
        escape_norm = float(np.linalg.norm(escape_direction))
        if escape_norm <= 1.0e-10:
            inactive_info["residual_control_link_avoidance_reason"] = "zero_escape_direction"
            return np.zeros(self.joint_count, dtype=np.float32), inactive_info
        escape_direction /= escape_norm

        weight = float(np.clip((activation_distance - d_min) / max(activation_margin, 1.0e-6), 0.0, 1.0))
        max_speed_mps = float(self.residual_control_cfg.get("link_avoidance_max_speed_mps", 0.20))
        desired_velocity = weight * max_speed_mps * escape_direction
        damping = float(self.residual_control_cfg.get("damping", 0.05))
        try:
            regularizer = damping**2
            qdot = point_jacobian.T @ np.linalg.solve(
                point_jacobian @ point_jacobian.T + regularizer * np.eye(3),
                desired_velocity,
            )
            reason = "active"
            active = True
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as error:
            qdot = np.zeros(self.joint_count, dtype=np.float64)
            reason = f"jacobian_failure:{error}"
            active = False
        qdot = np.clip(qdot, -self.action_scale, self.action_scale).astype(np.float32)
        return qdot, {
            "residual_control_link_avoidance_active": bool(active),
            "residual_control_link_avoidance_reason": reason,
            "residual_control_link_avoidance_qdot": qdot.copy(),
            "residual_control_link_avoidance_closest_link": closest_link,
            "residual_control_link_avoidance_d_min": d_min,
            "residual_control_link_avoidance_weight": weight if active else 0.0,
        }

    def _planner_waypoint(
        self,
        ee_position: np.ndarray,
        goal: np.ndarray,
    ) -> tuple[np.ndarray, bool, str]:
        if not self.obstacle_enabled:
            return np.asarray(goal, dtype=np.float64), False, "obstacle_disabled"

        path = np.asarray(goal, dtype=np.float64) - np.asarray(ee_position, dtype=np.float64)
        path_norm_sq = float(np.dot(path, path))
        if path_norm_sq <= 1.0e-12:
            return np.asarray(goal, dtype=np.float64), False, "zero_goal_error"

        obstacle = np.asarray(self.obstacle_center, dtype=np.float64)
        projection = float(np.clip(np.dot(obstacle - ee_position, path) / path_norm_sq, 0.0, 1.0))
        closest = np.asarray(ee_position, dtype=np.float64) + projection * path
        offset = closest - obstacle
        offset_norm = float(np.linalg.norm(offset))
        obstacle_radius = float(self.obstacle_cfg["radius"])
        clearance_margin = float(self.residual_control_cfg.get("clearance_margin_m", 0.03))
        lateral_margin = float(self.residual_control_cfg.get("waypoint_lateral_margin_m", 0.08))
        influence_radius = obstacle_radius + clearance_margin + lateral_margin
        if projection <= 0.02 or projection >= 0.98 or offset_norm >= influence_radius:
            return np.asarray(goal, dtype=np.float64), False, "direct_path_clear"

        if offset_norm <= 1.0e-10:
            path_direction = path / np.sqrt(path_norm_sq)
            lateral = np.cross(path_direction, np.asarray([0.0, 0.0, 1.0]))
            if np.linalg.norm(lateral) <= 1.0e-10:
                lateral = np.cross(path_direction, np.asarray([0.0, 1.0, 0.0]))
            offset_direction = lateral / max(float(np.linalg.norm(lateral)), 1.0e-10)
        else:
            offset_direction = offset / offset_norm

        waypoint = closest + offset_direction * influence_radius
        waypoint[2] += float(self.residual_control_cfg.get("waypoint_height_offset_m", 0.0))
        for axis, name in enumerate(("x", "y", "z")):
            low, high = (float(value) for value in self.workspace[name])
            waypoint[axis] = np.clip(waypoint[axis], low, high)
        return waypoint, True, "direct_path_blocked"

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        current_risk_before_action = self._compute_risk()
        predictive_diagnostics_enabled = bool(self.safety_filter_cfg.get("diagnostic_logging", False))
        policy_predictive_risk = None
        if (self.risk_representation != "current" or predictive_diagnostics_enabled) and self.obstacle_enabled:
            policy_predictive_risk = self._compute_predictive_risk()
        pre_risk = self._compute_policy_risk(current_risk_before_action, policy_predictive_risk)
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
        if self.hierarchical_enabled:
            hierarchical_nominal, hierarchical_info = self._hierarchical_nominal_command(current_risk_before_action)
            residual_scale = float(self.hierarchical_residual_cfg.get("residual_scale", 0.25))
            residual_budget = float(hierarchical_info["hierarchical_residual_budget"])
            residual_qdot = residual_budget * residual_scale * self.action_scale * action
            qdot_policy = np.clip(
                hierarchical_nominal + residual_qdot,
                -self.action_scale,
                self.action_scale,
            ).astype(np.float32)
            residual_info = self._disabled_residual_info()
            residual_info.update({"residual_qdot": residual_qdot.copy(), **hierarchical_info})
            self.last_residual_observation = np.zeros(self.residual_observation_dim, dtype=np.float32)
        elif self.residual_control_enabled:
            residual_base_qdot, residual_info = self._residual_base_command(current_risk_before_action)
            residual_scale = float(self.residual_control_cfg.get("residual_scale", 0.35))
            residual_qdot = residual_scale * self.action_scale * action
            qdot_policy = np.clip(
                residual_base_qdot + residual_qdot,
                -self.action_scale,
                self.action_scale,
            ).astype(np.float32)
            residual_info.update(
                {
                    "residual_control_enabled": True,
                    "residual_qdot": residual_qdot.copy(),
                }
            )
            self.last_residual_observation = self._residual_observation_features(residual_info)
        else:
            residual_info = self._disabled_residual_info()
            qdot_policy = self.action_scale * action
            self.last_residual_observation = np.zeros(self.residual_observation_dim, dtype=np.float32)
        if self.hierarchical_enabled:
            beta = float(self.hierarchical_cfg.get("command_smoothing_beta", 1.0))
        elif self.method.endswith("adaptive"):
            beta = self._adaptive_beta(pre_risk.risk_global)
        else:
            beta = self.fixed_beta
        qdot_cmd = beta * qdot_policy + (1.0 - beta) * self.prev_qdot_cmd
        nominal_ee_velocity = np.zeros(3, dtype=np.float64)
        nominal_toward_goal = 0.0
        if self.hierarchical_enabled:
            nominal_ee_velocity = self._link_origin_jacobian(self.tool_link_id) @ np.asarray(
                hierarchical_nominal, dtype=np.float64
            )
            nominal_goal_norm = float(np.linalg.norm(self._goal_error()))
            if nominal_goal_norm > 1.0e-9:
                nominal_toward_goal = float(
                    np.dot(nominal_ee_velocity, np.asarray(self._goal_error(), dtype=np.float64) / nominal_goal_norm)
                )
        if predictive_risk_for_scaling is not None:
            qdot_cmd *= risk_speed_scale
        recovery_in_servo_enabled = bool(
            self.safety_filter_cfg.get("recovery_in_servo_enabled", True)
        )
        recovery_allowed = not (
            self.hierarchical_enabled
            and self.hierarchical_state == "SERVO"
            and not recovery_in_servo_enabled
        )
        if self.hierarchical_enabled and self.hierarchical_selected_terminal_is_goal_region:
            # The retained RRT terminal is already obstacle-checked.  Do not
            # replace its path request with the relaxed recovery command; the
            # strict filter remains the sole command gate for this terminal.
            recovery_allowed = False
            self.recovery_active = False
        if bool(self.safety_filter_cfg.get("recovery_mode_enabled", False)):
            if recovery_allowed:
                self._update_recovery_state(predictive_risk_for_scaling, qdot_cmd)
            else:
                # A gated SERVO must also use the strict filter path: do not
                # leave the previous AVOID_HOLD relaxation active in the
                # filter merely because the state changed this cycle.
                self.recovery_active = False
        recovery_command = None
        if self.recovery_active and predictive_risk_for_scaling is not None and recovery_allowed:
            recovery_command = self._recovery_command(predictive_risk_for_scaling)
            qdot_cmd = recovery_command
        qdot_cmd = np.clip(qdot_cmd, -self.action_scale, self.action_scale).astype(np.float32)
        qdot_requested = qdot_cmd.copy()
        goal_error_before = self._goal_error().astype(np.float64)
        goal_error_before_norm = float(np.linalg.norm(goal_error_before))
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
            requested_norm = float(np.linalg.norm(qdot_requested))
            self.hierarchical_last_filter_intervention_ratio = float(
                np.clip(filter_result.intervention_norm_radps / max(requested_norm, 1.0e-6), 0.0, 1.0)
            )
            self.hierarchical_last_filter_status = str(filter_result.status.value)
            self.hierarchical_last_filtered_qdot = qdot_cmd.copy()
            previous_norm = float(np.linalg.norm(self.prev_qdot_cmd))
            command_norm = float(np.linalg.norm(qdot_cmd))
            command_alignment = float(
                np.dot(qdot_cmd, self.prev_qdot_cmd) / max(command_norm * previous_norm, 1.0e-9)
            )
            command_sign_change = bool(previous_norm > 1.0e-6 and command_norm > 1.0e-6 and command_alignment < 0.0)
            if bool(self.safety_filter_cfg.get("diagnostic_logging", False)):
                filter_link_diagnostics = self._filter_link_diagnostics(
                    predictive_risk,
                    qdot_cmd,
                    qdot_requested,
                )
        else:
            self.hierarchical_last_filter_intervention_ratio = 0.0
            self.hierarchical_last_filter_status = "not_enabled"
            self.hierarchical_last_filtered_qdot = qdot_cmd.copy()
            command_alignment = float("nan")
            command_sign_change = False

        # Task-space diagnostics are read-only: they quantify whether the
        # safety-filtered command still points toward the goal.  They do not
        # alter the command or the safety constraints.
        ee_jacobian = self._link_origin_jacobian(self.tool_link_id)
        requested_ee_velocity = ee_jacobian @ np.asarray(qdot_requested, dtype=np.float64)
        executed_ee_velocity = ee_jacobian @ np.asarray(qdot_cmd, dtype=np.float64)
        if goal_error_before_norm > 1.0e-9:
            goal_direction = goal_error_before / goal_error_before_norm
            requested_toward_goal = float(np.dot(requested_ee_velocity, goal_direction))
            executed_toward_goal = float(np.dot(executed_ee_velocity, goal_direction))
        else:
            requested_toward_goal = 0.0
            executed_toward_goal = 0.0
        requested_ee_speed = float(np.linalg.norm(requested_ee_velocity))
        executed_ee_speed = float(np.linalg.norm(executed_ee_velocity))

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
        collision_events = classify_collision_events(
            collision_capsule_overlap,
            collision_pybullet_contact,
            self.collision_termination_mode,
        )
        collision = collision_events.collision_any
        termination_collision = collision_events.termination_collision
        success = info["goal_error_norm"] < self.success_tolerance and current_risk.d_min > self.risk_config.d_safe
        terminated = bool(termination_collision or success)
        truncated = self.step_count >= self.max_episode_steps

        reward = self._reward(qdot_cmd, success, termination_collision)
        cost = self._cost(policy_risk, termination_collision, bool(info["safety_violation"]))
        if self.method in {"ee_fixed", "link_fixed", "hierarchical_residual"}:
            reward -= float(self.config["sac"]["fixed_risk_penalty"]) * cost

        joint_acc = (qdot_cmd - self.prev_qdot_cmd) / self.control_dt
        jerk = (joint_acc - self.prev_joint_acc) / self.control_dt
        info.update(
            {
                "reward": float(reward),
                "cost": float(cost),
                "collision": bool(collision),
                "collision_any": bool(collision),
                "termination_collision": bool(termination_collision),
                "termination_reason": collision_events.termination_reason,
                "collision_termination_mode": self.collision_termination_mode,
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
                "qdot_policy": qdot_policy.copy(),
                "task_space_goal_error_before_m": goal_error_before_norm,
                "task_space_goal_error_after_m": float(info["goal_error_norm"]),
                "task_space_goal_error_delta_m": goal_error_before_norm - float(info["goal_error_norm"]),
                "task_space_requested_velocity_mps": requested_ee_velocity.copy(),
                "task_space_executed_velocity_mps": executed_ee_velocity.copy(),
                "task_space_requested_speed_mps": requested_ee_speed,
                "task_space_executed_speed_mps": executed_ee_speed,
                "task_space_requested_toward_goal_mps": requested_toward_goal,
                "task_space_executed_toward_goal_mps": executed_toward_goal,
                "task_space_filter_velocity_loss_mps": requested_toward_goal - executed_toward_goal,
                "task_space_nominal_velocity_mps": nominal_ee_velocity.copy(),
                "task_space_nominal_toward_goal_mps": nominal_toward_goal,
                "task_space_recovery_active_for_command": bool(recovery_command is not None),
                "hierarchical_filter_intervention_ratio": self.hierarchical_last_filter_intervention_ratio,
                "hierarchical_filter_status": self.hierarchical_last_filter_status,
                "safety_filter_command_alignment_previous": command_alignment,
                "safety_filter_command_sign_change": command_sign_change,
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
                "recovery_trigger_reason": self.recovery_trigger_reason,
                "recovery_min_ttc_s": float(self.recovery_min_ttc_s),
                "recovery_command_norm": (
                    float(np.linalg.norm(recovery_command)) if recovery_command is not None else 0.0
                ),
            }
        )
        info.update(residual_info)
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
            self._record_infeasible_safe_stop(filter_result, predictive_risk)
            safe_stop_drift_mps, safe_stop_drift_class = self._safe_stop_drift()
            unavoidable_collision = bool(termination_collision and safe_stop_drift_class == "dynamic_drift")
            self.unavoidable_collision = self.unavoidable_collision or unavoidable_collision
            self.avoidable_collision = self.avoidable_collision or bool(
                termination_collision and not unavoidable_collision
            )
            info.update(
                {
                    "unavoidable_collision": unavoidable_collision,
                    "avoidable_collision": bool(termination_collision and not unavoidable_collision),
                    "collision_avoidability_reason": (
                        "dynamic_drift_after_safe_stop" if unavoidable_collision else "avoidable_or_static_failure"
                    )
                    if termination_collision
                    else "",
                    "safe_stop_infeasible_first_step": (
                        self.safe_stop_infeasible_first_step
                        if self.safe_stop_infeasible_first_step is not None
                        else -1
                    ),
                    "safe_stop_infeasible_last_step": (
                        self.safe_stop_infeasible_last_step
                        if self.safe_stop_infeasible_last_step is not None
                        else -1
                    ),
                    "safe_stop_h_drift_mps": safe_stop_drift_mps,
                    "safe_stop_drift_class": safe_stop_drift_class,
                }
            )
            if bool(self.safety_filter_cfg.get("viability_monitor_enabled", False)):
                info.update(self._viability_info(filter_result, predictive_risk, safe_stop_drift_class))
        elif policy_predictive_risk is not None:
            info.update(self._predictive_risk_info(policy_predictive_risk))
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

    def _record_infeasible_safe_stop(
        self,
        result: SafetyFilterResult,
        predictive_risk: PredictiveLinkRisk | None,
    ) -> None:
        if result.status is not SafetyFilterStatus.SAFE_STOP_INFEASIBLE:
            return
        if predictive_risk is None or not predictive_risk.usable:
            return
        h_min = float(np.min(predictive_risk.safety_functions_m))
        step = self.step_count - 1
        if self.safe_stop_infeasible_first_step is None:
            self.safe_stop_infeasible_first_step = step
            self.safe_stop_infeasible_first_h_m = h_min
        self.safe_stop_infeasible_last_step = step
        self.safe_stop_infeasible_last_h_m = h_min

    def _safe_stop_drift(self) -> tuple[float, str]:
        first_step = self.safe_stop_infeasible_first_step
        last_step = self.safe_stop_infeasible_last_step
        if first_step is None or last_step is None:
            return float("nan"), "not_applicable"
        if last_step <= first_step:
            return 0.0, "static_or_slow"
        duration_s = (last_step - first_step) * self.control_dt
        drift_mps = (self.safe_stop_infeasible_last_h_m - self.safe_stop_infeasible_first_h_m) / duration_s
        threshold_mps = float(self.safety_filter_cfg.get("safe_stop_dynamic_drift_threshold_mps", 0.05))
        if not np.isfinite(threshold_mps) or threshold_mps < 0.0:
            raise ValueError("safe_stop_dynamic_drift_threshold_mps must be finite and non-negative")
        drift_class = "dynamic_drift" if drift_mps < -threshold_mps else "static_or_slow"
        return float(drift_mps), drift_class

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
        hierarchical_info: dict[str, Any] = {}
        hierarchical_features = np.zeros(self.hierarchical_observation_dim, dtype=np.float32)
        if self.hierarchical_enabled:
            _, hierarchical_info = self._hierarchical_nominal_command(current_risk, update_state=False)
            hierarchical_features = np.concatenate(
                [
                    self.hierarchical_last_nominal_qdot / max(self.action_scale, 1.0e-6),
                    self.hierarchical_last_waypoint_error / max(self.action_scale, 1.0e-6),
                    np.asarray([self.hierarchical_last_progress, self.hierarchical_last_residual_budget], dtype=np.float32),
                    np.asarray(
                        [1.0 if state == self.hierarchical_state else 0.0 for state in self.hierarchical_state_names],
                        dtype=np.float32,
                    ),
                ]
            ).astype(np.float32)
        if self.residual_control_enabled:
            _, residual_info = self._residual_base_command(current_risk)
            self.last_residual_observation = self._residual_observation_features(residual_info)
        else:
            residual_info = self._disabled_residual_info()
            self.last_residual_observation = np.zeros(self.residual_observation_dim, dtype=np.float32)

        observation_parts = [
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
        if self.residual_control_enabled:
            observation_parts.append(self.last_residual_observation)
        if self.hierarchical_enabled:
            observation_parts.append(hierarchical_features.astype(np.float32))
        obs = np.concatenate(observation_parts).astype(np.float32)
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
        info.update(residual_info)
        info.update(hierarchical_info)
        return obs, info

    def _disabled_residual_info(self) -> dict[str, Any]:
        zeros = np.zeros(self.joint_count, dtype=np.float32)
        return {
            "residual_control_enabled": False,
            "residual_control_mode": "disabled",
            "residual_control_reason": "disabled",
            "residual_control_controller": "disabled",
            "residual_control_terminal_safe": False,
            "residual_control_goal_error_norm": float("nan"),
            "residual_control_target": np.zeros(3, dtype=np.float32),
            "residual_control_waypoint_active": False,
            "residual_control_waypoint": np.zeros(3, dtype=np.float32),
            "residual_control_target_qdot": zeros.copy(),
            "residual_control_link_avoidance_active": False,
            "residual_control_link_avoidance_reason": "disabled",
            "residual_control_link_avoidance_qdot": zeros.copy(),
            "residual_control_link_avoidance_closest_link": -1,
            "residual_control_link_avoidance_d_min": float("nan"),
            "residual_control_link_avoidance_weight": 0.0,
            "residual_control_base_qdot": zeros.copy(),
            "residual_qdot": zeros.copy(),
        }

    def _residual_observation_features(self, residual_info: dict[str, Any]) -> np.ndarray:
        mode = str(residual_info.get("residual_control_mode", "disabled"))
        mode_one_hot = np.zeros(len(self.residual_mode_names), dtype=np.float32)
        try:
            mode_one_hot[self.residual_mode_names.index(mode)] = 1.0
        except ValueError:
            mode_one_hot[0] = 1.0
        scale = max(self.action_scale, 1.0e-6)
        return np.concatenate(
            [
                np.asarray(
                    residual_info.get("residual_control_base_qdot", np.zeros(self.joint_count)),
                    dtype=np.float32,
                )
                / scale,
                np.asarray(
                    residual_info.get("residual_control_target_qdot", np.zeros(self.joint_count)),
                    dtype=np.float32,
                )
                / scale,
                np.asarray(
                    residual_info.get("residual_control_link_avoidance_qdot", np.zeros(self.joint_count)),
                    dtype=np.float32,
                )
                / scale,
                mode_one_hot,
            ]
        ).astype(np.float32)

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

    def _compute_policy_risk(
        self,
        current_risk: LinkRisk,
        predictive_risk: PredictiveLinkRisk | None = None,
    ) -> LinkRisk:
        if self.risk_representation == "current" or not self.obstacle_enabled:
            return current_risk

        if predictive_risk is None:
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
        if (
            not self.safety_filter_enabled
            and self.risk_representation == "current"
            and not bool(self.safety_filter_cfg.get("diagnostic_logging", False))
        ):
            return

        self.predictive_risk_config = PredictiveRiskConfig(
            d_safe_m=self.risk_config.d_safe,
            prediction_horizon_s=float(self.safety_filter_cfg["prediction_horizon_s"]),
            max_observation_age_s=float(self.safety_filter_cfg["max_observation_age_s"]),
            control_delay_s=float(self.safety_filter_cfg["control_delay_s"]),
            max_link_speed_mps=0.0,
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
            infeasibility_diagnostics_enabled=bool(
                self.safety_filter_cfg.get("infeasibility_diagnostics_enabled", False)
            ),
            goal_velocity_priority_enabled=bool(
                self.safety_filter_cfg.get("goal_velocity_priority_enabled", False)
            ),
            goal_velocity_priority_require_nonzero_request=bool(
                self.safety_filter_cfg.get("goal_velocity_priority_require_nonzero_request", False)
            ),
            goal_velocity_priority_solver=str(
                self.safety_filter_cfg.get("goal_velocity_priority_solver", "linprog")
            ),
            goal_velocity_priority_weight=float(
                self.safety_filter_cfg.get("goal_velocity_priority_weight", 1.0)
            ),
            goal_velocity_intervention_weight=float(
                self.safety_filter_cfg.get("goal_velocity_intervention_weight", 1.0)
            ),
            goal_velocity_objective_scale=float(
                self.safety_filter_cfg.get("goal_velocity_objective_scale", 1.0)
            ),
            goal_velocity_priority_for_goal_region=bool(
                self.safety_filter_cfg.get("goal_velocity_priority_for_goal_region", False)
            ),
            goal_velocity_temporal_consistency_enabled=bool(
                self.safety_filter_cfg.get("goal_velocity_temporal_consistency_enabled", False)
            ),
            goal_velocity_near_optimal_tolerance_mps=float(
                self.safety_filter_cfg.get("goal_velocity_near_optimal_tolerance_mps", 0.0)
            ),
            goal_velocity_continuity_requested_weight=float(
                self.safety_filter_cfg.get("goal_velocity_continuity_requested_weight", 0.25)
            ),
            goal_velocity_continuity_previous_weight=float(
                self.safety_filter_cfg.get("goal_velocity_continuity_previous_weight", 1.0)
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
            goal_velocity_objective = None
            if (
                self.safety_filter_config.goal_velocity_priority_enabled
                and (
                    not (
                        self.hierarchical_enabled
                        and self.hierarchical_selected_terminal_is_goal_region
                    )
                    or self.safety_filter_config.goal_velocity_priority_for_goal_region
                )
            ):
                goal_error = np.asarray(self._goal_error(), dtype=np.float64)
                goal_error_norm = float(np.linalg.norm(goal_error))
                if goal_error_norm > 1.0e-9:
                    goal_velocity_objective = (goal_error / goal_error_norm) @ ee_jacobian
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
                    goal_velocity_objective=goal_velocity_objective,
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
                infeasible_constraint_categories=result.infeasible_constraint_categories,
                infeasibility_diagnostic_status=result.infeasibility_diagnostic_status,
                goal_velocity_primary_mps=result.goal_velocity_primary_mps,
                goal_velocity_secondary_used=result.goal_velocity_secondary_used,
                goal_velocity_secondary_status=result.goal_velocity_secondary_status,
            ),
            predictive_risk,
            elapsed_s,
        )

    def _compute_predictive_risk(
        self,
        max_link_speed_bound_mps: float | None = None,
    ) -> PredictiveLinkRisk:
        if self.predictive_risk_config is None:
            raise RuntimeError("Predictive risk requested while safety filter is disabled")
        return self._compute_predictive_risk_for_capsules(
            self._capsules(),
            max_link_speed_bound_mps=max_link_speed_bound_mps,
        )

    def _compute_predictive_risk_for_capsules(
        self,
        capsules: list[CapsuleState],
        max_link_speed_bound_mps: float | None = None,
    ) -> PredictiveLinkRisk:
        if self.predictive_risk_config is None:
            raise RuntimeError("Predictive risk requested while safety filter is disabled")
        speed_bound_mps = (
            self._max_capsule_point_speed_bound_mps()
            if max_link_speed_bound_mps is None
            else float(max_link_speed_bound_mps)
        )
        predictive_config = replace(
            self.predictive_risk_config,
            max_link_speed_mps=speed_bound_mps,
        )
        obstacle_estimate = self._obstacle_state_estimate()
        return compute_predictive_link_risk(
            capsules=capsules,
            link_velocities_mps=self._capsule_point_velocities_mps(
                capsules,
                np.asarray(obstacle_estimate.position, dtype=np.float64),
            ),
            obstacle=obstacle_estimate,
            now_s=self.sim_time_s,
            config=predictive_config,
        )

    def _capsule_point_velocities_mps(
        self,
        capsules: list[CapsuleState],
        obstacle_position_m: np.ndarray,
    ) -> np.ndarray:
        joint_velocities_radps = self._joint_state()[1].astype(np.float64)
        velocities = np.zeros((self.capsule_model.count, 3), dtype=np.float64)
        for index, (capsule, spec) in enumerate(zip(capsules, self.capsule_model.specs, strict=True)):
            _, segment_fraction = closest_point_on_segment(
                obstacle_position_m,
                np.asarray(capsule.start, dtype=np.float64),
                np.asarray(capsule.end, dtype=np.float64),
            )
            start_jacobian, end_jacobian = self._capsule_endpoint_jacobians(spec)
            point_jacobian_m_per_rad = (
                (1.0 - segment_fraction) * start_jacobian + segment_fraction * end_jacobian
            )
            velocities[index] = point_jacobian_m_per_rad @ joint_velocities_radps
        return velocities

    def _max_capsule_point_speed_bound_mps(self) -> float:
        max_speed_mps = 0.0
        for spec in self.capsule_model.specs:
            for point_jacobian_m_per_rad in self._capsule_endpoint_jacobians(spec):
                endpoint_velocities_mps = self.joint_velocity_limit_corners @ point_jacobian_m_per_rad.T
                max_speed_mps = max(
                    max_speed_mps,
                    float(np.max(np.linalg.norm(endpoint_velocities_mps, axis=1), initial=0.0)),
                )
        return max_speed_mps

    def _capsule_endpoint_jacobians(self, spec: CapsuleSpec) -> tuple[np.ndarray, np.ndarray]:
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
            else spec.child_offset
        )
        return (
            self._link_point_jacobian(start_link_id, start_local),
            self._link_point_jacobian(end_link_id, end_local),
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
            predicted_translation = (
                predictive_risk.link_velocities_mps[index] * float(prediction_time_s)
            )
            predicted_start = np.asarray(capsule.start, dtype=np.float64) + predicted_translation
            predicted_end = np.asarray(capsule.end, dtype=np.float64) + predicted_translation
            segment = predicted_end - predicted_start
            segment_norm_sq = float(np.dot(segment, segment))
            if segment_norm_sq <= 1e-14:
                predicted_obstacle_position = obstacle_position + obstacle_velocity * float(prediction_time_s)
                separation = predicted_obstacle_position - predicted_start
                separation_norm = float(np.linalg.norm(separation))
                if separation_norm > 1e-10:
                    point_link_id = self.capsule_model.link_name_to_id[spec.parent_link_name]
                    point_local = (
                        spec.start_local_position
                        if spec.start_local_position is not None
                        else np.zeros(3, dtype=np.float64)
                    )
                    safety_jacobian[index] = -(
                        separation / separation_norm
                    ) @ self._link_point_jacobian(point_link_id, point_local)
                continue
            predicted_obstacle_position = obstacle_position + obstacle_velocity * float(prediction_time_s)
            segment_fraction = float(
                np.clip(
                    np.dot(predicted_obstacle_position - predicted_start, segment) / segment_norm_sq,
                    0.0,
                    1.0,
                )
            )
            closest_capsule_point = predicted_start + segment_fraction * segment
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
        jacobian_link_id, jacobian_local_position = self._movable_ancestor_point(link_id, local_position)
        if jacobian_link_id < 0:
            return np.zeros((3, self.joint_count), dtype=np.float64)
        joint_states = p.getJointStates(self.robot_id, self.jacobian_joint_ids, physicsClientId=self.physics_client_id)
        positions = [float(state[0]) for state in joint_states]
        zeros = [0.0] * len(positions)
        linear, _ = p.calculateJacobian(
            self.robot_id,
            jacobian_link_id,
            jacobian_local_position.tolist(),
            positions,
            zeros,
            zeros,
            physicsClientId=self.physics_client_id,
        )
        return np.asarray(linear, dtype=np.float64)[:, self.control_jacobian_columns]

    def _movable_ancestor_point(
        self,
        link_id: int,
        local_position: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """Express a point on a fixed-link chain in its nearest movable frame."""
        world_position, _ = p.multiplyTransforms(
            *p.getLinkState(self.robot_id, link_id, computeForwardKinematics=True, physicsClientId=self.physics_client_id)[4:6],
            np.asarray(local_position, dtype=np.float64).tolist(),
            [0.0, 0.0, 0.0, 1.0],
        )
        current = link_id
        while current >= 0:
            joint_info = p.getJointInfo(self.robot_id, current, physicsClientId=self.physics_client_id)
            if joint_info[2] != p.JOINT_FIXED:
                link_state = p.getLinkState(
                    self.robot_id,
                    current,
                    computeForwardKinematics=True,
                    physicsClientId=self.physics_client_id,
                )
                inverse_position, inverse_orientation = p.invertTransform(link_state[4], link_state[5])
                parent_position, _ = p.multiplyTransforms(
                    inverse_position,
                    inverse_orientation,
                    world_position,
                    [0.0, 0.0, 0.0, 1.0],
                )
                return current, np.asarray(parent_position, dtype=np.float64)
            current = int(joint_info[16])
        return -1, np.zeros(3, dtype=np.float64)

    def _joint_position_limits(self) -> tuple[np.ndarray, np.ndarray]:
        limits = [p.getJointInfo(self.robot_id, joint_id, physicsClientId=self.physics_client_id) for joint_id in self.joint_ids]
        lower = np.asarray([info[8] for info in limits], dtype=np.float64)
        upper = np.asarray([info[9] for info in limits], dtype=np.float64)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower >= upper).any():
            raise ValueError("URDF must provide finite lower and upper limits for every controlled joint")
        return lower, upper

    @staticmethod
    def _predictive_risk_info(predictive_risk: PredictiveLinkRisk | None) -> dict[str, Any]:
        predictive_info = {
            "predictive_risk_status": "integration_error",
            "predictive_risk_reason": "predictive risk was not produced",
            "predictive_risk_age_s": float("nan"),
            "predictive_h_min_m": float("-inf"),
            "predictive_max_link_speed_bound_mps": float("nan"),
            "predictive_link_velocity_norms_mps": "",
        }
        if predictive_risk is not None:
            predictive_info = {
                "predictive_risk_status": predictive_risk.status.value,
                "predictive_risk_reason": predictive_risk.status_reason,
                "predictive_risk_age_s": float(predictive_risk.observation_age_s),
                "predictive_h_min_m": float(np.min(predictive_risk.safety_functions_m)),
                "predictive_max_link_speed_bound_mps": float(predictive_risk.max_link_speed_bound_mps),
                "predictive_link_velocity_norms_mps": UR5DynamicObstacleEnv._format_diagnostic_vector(
                    np.linalg.norm(predictive_risk.link_velocities_mps, axis=1)
                ),
            }
        return predictive_info

    @staticmethod
    def _filter_info(
        result: SafetyFilterResult,
        predictive_risk: PredictiveLinkRisk | None,
        solve_time_s: float | None,
        phase_times_s: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        predictive_info = UR5DynamicObstacleEnv._predictive_risk_info(predictive_risk)
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
            "safety_filter_goal_velocity_primary_mps": float(result.goal_velocity_primary_mps),
            "safety_filter_goal_velocity_secondary_used": bool(result.goal_velocity_secondary_used),
            "safety_filter_goal_velocity_secondary_status": result.goal_velocity_secondary_status,
            "safety_filter_fallback_stage": result.fallback_stage,
            "safety_filter_infeasible_constraint_categories": "|".join(result.infeasible_constraint_categories),
            "safety_filter_infeasibility_diagnostic_status": result.infeasibility_diagnostic_status,
            "safety_filter_max_constraint_violation": float(result.max_constraint_violation),
            "safety_filter_safe_stop": bool(result.requires_safe_stop),
            "safety_filter_solve_time_s": float(solve_time_s) if solve_time_s is not None else float("nan"),
            "safety_filter_predictive_risk_time_s": float((phase_times_s or {}).get("predictive_risk", 0.0)),
            "safety_filter_jacobian_workspace_time_s": float((phase_times_s or {}).get("jacobian_workspace", 0.0)),
            "safety_filter_projection_time_s": float((phase_times_s or {}).get("projection", 0.0)),
            **predictive_info,
        }

    def _viability_info(
        self,
        result: SafetyFilterResult,
        predictive_risk: PredictiveLinkRisk | None,
        safe_stop_drift_class: str,
    ) -> dict[str, Any]:
        """Return V1 diagnostic fields after the strict command is finalized."""

        assessment = assess_strict_viability(result, predictive_risk, safe_stop_drift_class)
        viable_risk = predictive_risk is not None and predictive_risk.usable
        return {
            "viability_status": assessment.status.value,
            "viability_horizon_s": float(self.safety_filter_cfg["prediction_horizon_s"]),
            "viability_min_h_m": (
                float(np.min(predictive_risk.safety_functions_m)) if viable_risk else float("-inf")
            ),
            "viability_strict_feasible": assessment.strict_feasible,
            "viability_solver_status": assessment.solver_status,
            "viability_model_assumptions_valid": assessment.model_assumptions_valid,
        }

    def _filter_link_diagnostics(
        self,
        predictive_risk: PredictiveLinkRisk | None,
        command_joint_velocity_radps: np.ndarray,
        requested_joint_velocity_radps: np.ndarray,
    ) -> dict[str, Any]:
        """Expose per-link values and a read-only goal-direction feasibility audit."""
        if predictive_risk is None or not predictive_risk.usable:
            return {}
        safety_jacobian, ee_position, _ = self._analytic_constraint_jacobians(predictive_risk)
        command = np.asarray(command_joint_velocity_radps, dtype=np.float64)
        h = np.asarray(predictive_risk.safety_functions_m, dtype=np.float64)
        jacobian_command = safety_jacobian @ command
        drift = self._safety_drift_mps(predictive_risk)
        safety_gain = float(self.safety_filter_cfg["safety_gain"])
        preemptive_margin = float(self.safety_filter_cfg.get("preemptive_margin_m", 0.0))
        residual = jacobian_command + drift + safety_gain * (h - preemptive_margin)
        obstacle = self._obstacle_state_estimate()
        diagnostics = {
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
        goal_error = np.asarray(self._goal_error(), dtype=np.float64)
        goal_error_norm = float(np.linalg.norm(goal_error))
        if goal_error_norm <= 1.0e-9:
            diagnostics.update(
                {
                    "safety_filter_goal_velocity_audit_status": "goal_already_reached",
                    "safety_filter_max_feasible_goal_velocity_mps": 0.0,
                    "safety_filter_projected_goal_velocity_mps": 0.0,
                    "safety_filter_goal_velocity_feasibility_gap_mps": 0.0,
                }
            )
            return diagnostics
        # Use the same TCP Jacobian as the executed task-space diagnostic in
        # step(); the workspace rows are rebuilt identically for the audit.
        ee_jacobian = self._link_origin_jacobian(self.tool_link_id)
        goal_direction = goal_error / goal_error_norm
        objective = goal_direction @ ee_jacobian
        joint_positions, _ = self._joint_state()
        workspace = self._workspace_velocity_constraints(ee_position, ee_jacobian)
        audit_input = SafetyFilterInput(
            requested_joint_velocity_radps=np.asarray(requested_joint_velocity_radps, dtype=np.float64),
            joint_positions_rad=joint_positions,
            previous_command_radps=self.prev_qdot_cmd,
            predictive_risk=predictive_risk,
            safety_jacobian_m_per_rad=safety_jacobian,
            safety_drift_mps=drift,
            workspace_constraints=workspace,
        )
        max_goal_velocity, _, audit_status = maximize_linear_velocity(
            audit_input,
            self.safety_filter_config,
            objective,
        )
        projected_goal_velocity = float(np.dot(objective, np.asarray(command_joint_velocity_radps, dtype=np.float64)))
        diagnostics.update(
            {
                "safety_filter_goal_velocity_audit_status": audit_status,
                "safety_filter_max_feasible_goal_velocity_mps": max_goal_velocity,
                "safety_filter_projected_goal_velocity_mps": projected_goal_velocity,
                "safety_filter_goal_velocity_feasibility_gap_mps": (
                    float(max_goal_velocity - projected_goal_velocity)
                    if np.isfinite(max_goal_velocity)
                    else float("nan")
                ),
            }
        )
        return diagnostics

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
        reward += self._terminal_progress_reward(goal_error_norm, progress)
        if success:
            reward += float(self.reward_cfg["success_bonus"])
        if collision:
            reward -= float(self.reward_cfg["collision_penalty"])
        return float(reward)

    def _terminal_progress_reward(self, goal_error_norm: float, progress: float) -> float:
        radius = float(self.reward_cfg.get("terminal_goal_radius_m", 0.0))
        weight = float(self.reward_cfg.get("w_terminal_progress", 0.0))
        if radius <= 0.0 or weight == 0.0:
            return 0.0
        if self.prev_goal_error_norm <= radius or goal_error_norm <= radius:
            # Signed progress rewards the final approach and penalizes moving
            # back out of the terminal region using the same potential.
            return weight * progress
        return 0.0

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

    def _update_recovery_state(
        self,
        predictive_risk: PredictiveLinkRisk | None,
        candidate_command: np.ndarray | None = None,
    ) -> None:
        """Trigger bounded escape before a closing obstacle makes recovery infeasible."""
        if predictive_risk is None:
            return
        if predictive_risk.requires_safe_stop or not predictive_risk.usable:
            return
        h_min = float(np.min(predictive_risk.safety_functions_m))
        if candidate_command is None:
            candidate_command = getattr(self, "prev_qdot_cmd", np.zeros(self.joint_count, dtype=np.float64))
        min_ttc = float("inf")
        if (
            hasattr(self, "_analytic_constraint_jacobians")
            and hasattr(self, "_safety_drift_mps")
            and hasattr(self, "obstacle_state_estimate_override")
        ):
            safety_jacobian, _, _ = self._analytic_constraint_jacobians(predictive_risk)
            drift = self._safety_drift_mps(predictive_risk)
            derivative = safety_jacobian @ np.asarray(candidate_command, dtype=np.float64) + drift
            closing = np.maximum(-derivative, 0.0)
            ttc = np.full_like(closing, np.inf, dtype=np.float64)
            positive_closing = closing > 1.0e-8
            ttc[positive_closing] = np.maximum(
                np.asarray(predictive_risk.safety_functions_m, dtype=np.float64)[positive_closing], 0.0
            ) / closing[positive_closing]
            min_ttc = float(np.min(ttc, initial=np.inf))
        self.recovery_min_ttc_s = min(getattr(self, "recovery_min_ttc_s", float("inf")), min_ttc)
        enter_margin = float(self.safety_filter_cfg.get("recovery_enter_margin_m", 0.0))
        exit_margin = float(self.safety_filter_cfg.get("recovery_exit_margin_m", 0.02))
        ttc_threshold = float(self.safety_filter_cfg.get("recovery_ttc_threshold_s", getattr(self, "control_dt", 0.05)))
        if (
            not np.isfinite(enter_margin)
            or not np.isfinite(exit_margin)
            or enter_margin < 0.0
            or exit_margin <= enter_margin
            or not np.isfinite(ttc_threshold)
            or ttc_threshold <= 0.0
        ):
            raise ValueError("recovery requires finite margins and a positive TTC threshold")
        should_enter = h_min <= enter_margin or min_ttc <= ttc_threshold
        if not self.recovery_active and should_enter:
            self.recovery_active = True
            self.recovery_triggered = True
            self.recovery_success = False
            self.recovery_trigger_reason = "margin" if h_min <= enter_margin else "ttc"
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
        direction_mode = str(self.safety_filter_cfg.get("recovery_direction_mode", "weighted_all_links"))
        if direction_mode == "worst_link":
            direction = safety_jacobian[int(np.argmin(margins))].copy()
        elif direction_mode == "weighted_all_links":
            weights = deficits / float(np.sum(deficits))
            direction = np.sum(weights[:, None] * safety_jacobian, axis=0)
        else:
            raise ValueError(
                "recovery_direction_mode must be 'weighted_all_links' or 'worst_link'"
            )
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

        if self.obstacle_scenario == "mixed_static":
            # Keep half of the original random distribution while covering
            # each controlled link-crossing band equally in the other half.
            choices = ("random", "upper_arm_crossing", "elbow_crossing", "forearm_crossing", "wrist_crossing")
            probabilities = (0.50, 0.125, 0.125, 0.125, 0.125)
            selected = str(self.rng.choice(choices, p=probabilities))
            if selected == "random":
                return self._sample_random_obstacle()
            return self._sample_named_obstacle(selected)

        if self.obstacle_scenario != "random":
            return self._sample_named_obstacle(self.obstacle_scenario)

        return self._sample_random_obstacle()

    def _sample_random_obstacle(self) -> tuple[np.ndarray, np.ndarray]:
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
