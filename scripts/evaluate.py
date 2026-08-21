from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from rl_risk_sac.algorithms import SACAgent
from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.device import resolve_device
from rl_risk_sac.utils.seeding import set_seed

try:
    from scripts.seed_manifest import load_seed_manifest
except ModuleNotFoundError:
    from seed_manifest import load_seed_manifest


def mean_finite(rows: list[dict[str, float | int]], field: str) -> float:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def max_finite(rows: list[dict[str, float | int]], field: str) -> float:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if len(values) else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--method",
        default=None,
        choices=["ee_fixed", "link_fixed", "ldrc_fixed", "ldrc_adaptive", "hierarchical_residual"],
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-manifest", default=None)
    parser.add_argument(
        "--manifest-shard-index",
        type=int,
        default=None,
        help="zero-based shard index for parallel evaluation of a seed manifest",
    )
    parser.add_argument(
        "--manifest-shard-count",
        type=int,
        default=None,
        help="number of disjoint shards used to partition the requested manifest episodes",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--trace-output", default=None)
    parser.add_argument(
        "--nominal-only",
        action="store_true",
        help="run the hierarchical planner/tracker with a zero residual and no actor checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    eval_cfg = config.get("eval", {})
    method = args.method or eval_cfg["method"]
    nominal_only = bool(args.nominal_only or eval_cfg.get("nominal_only", False))
    checkpoint = args.checkpoint or eval_cfg.get("checkpoint")
    if not checkpoint and not nominal_only:
        raise ValueError("checkpoint must be provided by --checkpoint or eval.checkpoint")
    episodes = int(args.episodes if args.episodes is not None else eval_cfg.get("episodes", 20))
    seed = int(args.seed if args.seed is not None else eval_cfg.get("seed", 123))
    seed_manifest_arg = args.seed_manifest if args.seed_manifest is not None else eval_cfg.get("seed_manifest")
    manifest_seeds = load_seed_manifest(seed_manifest_arg) if seed_manifest_arg else None
    if episodes <= 0:
        raise ValueError("Evaluation episodes must be positive")
    if manifest_seeds is not None and episodes > len(manifest_seeds):
        raise ValueError(
            f"Requested {episodes} episodes but seed manifest only contains {len(manifest_seeds)} seeds"
        )
    shard_index = args.manifest_shard_index
    shard_count = args.manifest_shard_count
    if (shard_index is None) != (shard_count is None):
        raise ValueError("manifest shard index and count must be provided together")
    if shard_count is not None:
        if manifest_seeds is None:
            raise ValueError("manifest sharding requires --seed-manifest or eval.seed_manifest")
        if shard_count <= 0 or shard_count > episodes:
            raise ValueError("manifest shard count must be in [1, episodes]")
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError("manifest shard index must be in [0, shard_count)")
    indexed_episode_seeds = list(
        enumerate(
            manifest_seeds[:episodes]
            if manifest_seeds is not None
            else [seed + episode for episode in range(episodes)]
        )
    )
    if shard_count is not None:
        indexed_episode_seeds = indexed_episode_seeds[shard_index::shard_count]
    output_arg = args.output if args.output is not None else eval_cfg.get("output")
    trace_output_arg = args.trace_output if args.trace_output is not None else eval_cfg.get("trace_output")
    config["seed"] = seed
    config["device"] = resolve_device(config)
    set_seed(seed)
    safety_filter_enabled = bool(config["env"].get("safety_filter", {}).get("enabled", False))

    env = UR5DynamicObstacleEnv(config, method=method)
    if nominal_only and not env.hierarchical_enabled:
        raise ValueError("nominal-only evaluation requires env.hierarchical_control.enabled=true")
    agent = None
    if not nominal_only:
        agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], config, method=method)
        agent.load_actor(checkpoint)

    rows = []
    trace_dir = Path(trace_output_arg) if trace_output_arg else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)

    for episode, episode_seed in indexed_episode_seeds:
        observation, reset_info = env.reset(seed=episode_seed)
        total_reward = 0.0
        total_cost = 0.0
        risks = []
        distances = []
        violations = 0
        accelerations = []
        jerks = []
        action_variations = []
        nominal_qdot_norms = []
        residual_qdot_norms = []
        task_requested_toward_goal = []
        task_executed_toward_goal = []
        task_velocity_losses = []
        task_error_deltas = []
        filter_interventions = 0
        filter_intervention_norms = []
        filter_safe_stops = 0
        filter_infeasible = 0
        filter_projection_failures = 0
        filter_recovery_relaxed = 0
        filter_fallbacks = 0
        filter_qp_used = 0
        filter_qp_timeouts = 0
        filter_compute_budget_stops = 0
        predictive_near_misses = 0
        predictive_h_mins = []
        filter_solve_times_s = []
        filter_predictive_risk_times_s = []
        filter_jacobian_workspace_times_s = []
        filter_projection_times_s = []
        recovery_steps = 0
        recovery_triggered = False
        recovery_success = False
        hierarchical_state_counts: dict[str, int] = {}
        initially_unsafe = bool(reset_info.get("recovery_initially_unsafe", False))
        initial_predictive_h_m = float(reset_info.get("recovery_initial_h_min_m", float("nan")))
        initial_ik_found = bool(reset_info.get("hierarchical_ik_found", False))
        initial_plan_found = bool(reset_info.get("hierarchical_plan_found", False))
        initial_direct_path = bool(reset_info.get("hierarchical_direct_path", False))
        initial_plan_reason = str(reset_info.get("hierarchical_plan_reason", ""))
        initial_planning_time_s = float(reset_info.get("hierarchical_planning_time_s", float("nan")))
        initial_plan_iterations = int(reset_info.get("hierarchical_plan_iterations", 0))
        initial_ik_candidate_count = int(reset_info.get("hierarchical_ik_candidate_count", 0))
        success = False
        collision = False
        termination_collision = False
        termination_reason = ""
        collision_capsule_overlap = False
        collision_pybullet_contact = False
        unavoidable_collision = False
        avoidable_collision = False
        final_position_error = 0.0
        closest_link = -1
        non_end_link_collision = False
        prev_qdot_cmd = np.zeros(env.action_space.shape[0], dtype=np.float32)
        trace_rows = []
        for step in range(int(config["env"]["max_episode_steps"])):
            action = (
                np.zeros(env.action_space.shape, dtype=np.float32)
                if nominal_only
                else agent.select_action(observation, deterministic=True)
            )
            observation, reward, cost, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_cost += cost
            risks.append(float(info["risk_global"]))
            distances.append(float(info["d_min"]))
            violations += int(info["safety_violation"])
            accelerations.append(info["joint_acc"])
            jerks.append(info["joint_jerk"])
            action_variations.append(float(np.linalg.norm(info["qdot_cmd"] - prev_qdot_cmd)))
            nominal_qdot_norms.append(
                float(np.linalg.norm(info.get("hierarchical_nominal_qdot", np.zeros(env.action_space.shape[0]))))
            )
            residual_qdot_norms.append(
                float(np.linalg.norm(info.get("residual_qdot", np.zeros(env.action_space.shape[0]))))
            )
            for values, key in (
                (task_requested_toward_goal, "task_space_requested_toward_goal_mps"),
                (task_executed_toward_goal, "task_space_executed_toward_goal_mps"),
                (task_velocity_losses, "task_space_filter_velocity_loss_mps"),
                (task_error_deltas, "task_space_goal_error_delta_m"),
            ):
                value = float(info.get(key, float("nan")))
                if np.isfinite(value):
                    values.append(value)
            filter_status = str(info.get("safety_filter_status", "not_enabled"))
            intervention_norm = float(info.get("safety_filter_intervention_norm", float("nan")))
            predictive_h_min = float(info.get("predictive_h_min_m", float("nan")))
            solve_time_s = float(info.get("safety_filter_solve_time_s", float("nan")))
            predictive_risk_time_s = float(info.get("safety_filter_predictive_risk_time_s", float("nan")))
            jacobian_workspace_time_s = float(info.get("safety_filter_jacobian_workspace_time_s", float("nan")))
            projection_time_s = float(info.get("safety_filter_projection_time_s", float("nan")))
            recovery_steps += int(bool(info.get("recovery_active", False)))
            recovery_triggered = recovery_triggered or bool(info.get("recovery_triggered", False))
            recovery_success = recovery_success or bool(info.get("recovery_success", False))
            hierarchical_state = str(info.get("hierarchical_state", "disabled"))
            hierarchical_state_counts[hierarchical_state] = hierarchical_state_counts.get(hierarchical_state, 0) + 1
            filter_interventions += int(filter_status not in {"passthrough", "not_enabled"})
            filter_safe_stops += int(bool(info.get("safety_filter_safe_stop", False)))
            filter_infeasible += int(filter_status == "safe_stop_infeasible")
            filter_projection_failures += int(filter_status == "safe_stop_projection_failed")
            filter_recovery_relaxed += int(filter_status == "recovery_relaxed")
            filter_fallbacks += int(bool(info.get("safety_filter_fallback_used", False)))
            filter_qp_used += int(bool(info.get("safety_filter_qp_solver_used", False)))
            filter_qp_timeouts += int("time limit" in str(info.get("safety_filter_qp_solver_status", "")).lower())
            filter_compute_budget_stops += int(filter_status == "safe_stop_compute_budget")
            if np.isfinite(intervention_norm):
                filter_intervention_norms.append(intervention_norm)
            if np.isfinite(predictive_h_min):
                predictive_h_mins.append(predictive_h_min)
                predictive_near_misses += int(predictive_h_min < 0.0)
            if np.isfinite(solve_time_s):
                filter_solve_times_s.append(solve_time_s)
            if np.isfinite(predictive_risk_time_s):
                filter_predictive_risk_times_s.append(predictive_risk_time_s)
            if np.isfinite(jacobian_workspace_time_s):
                filter_jacobian_workspace_times_s.append(jacobian_workspace_time_s)
            if np.isfinite(projection_time_s):
                filter_projection_times_s.append(projection_time_s)
            success = bool(info["success"])
            collision = collision or bool(info["collision_any"])
            collision_capsule_overlap = collision_capsule_overlap or bool(
                info.get("collision_capsule_overlap", False)
            )
            collision_pybullet_contact = collision_pybullet_contact or bool(
                info.get("collision_pybullet_contact", False)
            )
            if bool(info.get("termination_collision", False)):
                termination_collision = True
                termination_reason = str(info.get("termination_reason", ""))
            unavoidable_collision = unavoidable_collision or bool(info.get("unavoidable_collision", False))
            avoidable_collision = avoidable_collision or bool(info.get("avoidable_collision", False))
            final_position_error = float(info["goal_error_norm"])
            closest_link = int(info["closest_link"])
            non_end_link_collision = non_end_link_collision or bool(
                info["collision_any"] and 0 <= closest_link < env.capsule_model.count - 1
            )
            if trace_dir is not None:
                trace_rows.append(
                    {
                        "step": step,
                        "time": (step + 1) * float(config["env"]["control_dt"]),
                        "reward": float(reward),
                        "cost": float(cost),
                        "goal_error_norm": final_position_error,
                        "d_min": float(info["d_min"]),
                        "risk_global": float(info["risk_global"]),
                        "beta": float(info["beta"]),
                        "closest_link": closest_link,
                        "safety_violation": int(info["safety_violation"]),
                        "collision": int(info["collision"]),
                        "collision_any": int(info["collision_any"]),
                        "collision_capsule_overlap": int(bool(info.get("collision_capsule_overlap", False))),
                        "collision_pybullet_contact": int(bool(info.get("collision_pybullet_contact", False))),
                        "termination_collision": int(bool(info.get("termination_collision", False))),
                        "termination_reason": info.get("termination_reason", ""),
                        "unavoidable_collision": int(bool(info.get("unavoidable_collision", False))),
                        "avoidable_collision": int(bool(info.get("avoidable_collision", False))),
                        "collision_avoidability_reason": info.get("collision_avoidability_reason", ""),
                        "safe_stop_infeasible_first_step": info.get("safe_stop_infeasible_first_step", -1),
                        "safe_stop_infeasible_last_step": info.get("safe_stop_infeasible_last_step", -1),
                        "safe_stop_h_drift_mps": float(info.get("safe_stop_h_drift_mps", float("nan"))),
                        "safe_stop_drift_class": info.get("safe_stop_drift_class", "not_applicable"),
                        "collision_contact_link_indices": info.get("collision_contact_link_indices", ""),
                        "collision_contact_link_names": info.get("collision_contact_link_names", ""),
                        "collision_min_contact_distance": float(
                            info.get("collision_min_contact_distance", float("nan"))
                        ),
                        "success": int(info["success"]),
                        "qdot_norm": float(np.linalg.norm(info["qdot_cmd"])),
                        "acc_norm": float(np.linalg.norm(info["joint_acc"])),
                        "jerk_norm": float(np.linalg.norm(info["joint_jerk"])),
                        "qdot_requested_norm": float(np.linalg.norm(info.get("qdot_requested", info["qdot_cmd"]))),
                        "qdot_policy_norm": float(np.linalg.norm(info.get("qdot_policy", info["qdot_cmd"]))),
                        "task_space_goal_error_before_m": info.get("task_space_goal_error_before_m", float("nan")),
                        "task_space_goal_error_after_m": info.get("task_space_goal_error_after_m", float("nan")),
                        "task_space_goal_error_delta_m": info.get("task_space_goal_error_delta_m", float("nan")),
                        "task_space_requested_speed_mps": info.get("task_space_requested_speed_mps", float("nan")),
                        "task_space_executed_speed_mps": info.get("task_space_executed_speed_mps", float("nan")),
                        "task_space_requested_toward_goal_mps": info.get("task_space_requested_toward_goal_mps", float("nan")),
                        "task_space_executed_toward_goal_mps": info.get("task_space_executed_toward_goal_mps", float("nan")),
                        "task_space_filter_velocity_loss_mps": info.get("task_space_filter_velocity_loss_mps", float("nan")),
                        "task_space_nominal_toward_goal_mps": info.get("task_space_nominal_toward_goal_mps", float("nan")),
                        "task_space_recovery_active_for_command": int(bool(info.get("task_space_recovery_active_for_command", False))),
                        "residual_control_enabled": int(bool(info.get("residual_control_enabled", False))),
                        "residual_control_mode": info.get("residual_control_mode", "disabled"),
                        "residual_control_base_qdot_norm": float(
                            np.linalg.norm(info.get("residual_control_base_qdot", np.zeros(env.action_space.shape[0])))
                        ),
                        "hierarchical_control_enabled": int(bool(info.get("hierarchical_control_enabled", False))),
                        "hierarchical_state": info.get("hierarchical_state", "disabled"),
                        "hierarchical_plan_reason": info.get("hierarchical_plan_reason", ""),
                        "hierarchical_plan_iterations": info.get("hierarchical_plan_iterations", 0),
                        "hierarchical_planning_time_s": info.get("hierarchical_planning_time_s", float("nan")),
                        "hierarchical_planning_time_total_s": info.get(
                            "hierarchical_planning_time_total_s", float("nan")
                        ),
                        "hierarchical_ik_candidate_count": info.get("hierarchical_ik_candidate_count", 0),
                        "hierarchical_ik_found": int(bool(info.get("hierarchical_ik_found", False))),
                        "hierarchical_plan_found": int(bool(info.get("hierarchical_plan_found", False))),
                        "hierarchical_direct_path": int(bool(info.get("hierarchical_direct_path", False))),
                        "hierarchical_selected_candidate_index": info.get("hierarchical_selected_candidate_index", -1),
                        "hierarchical_selected_path_length_rad": info.get("hierarchical_selected_path_length_rad", float("nan")),
                        "hierarchical_selected_path_min_clearance_m": info.get("hierarchical_selected_path_min_clearance_m", float("nan")),
                        "hierarchical_ik_candidate_errors_m": ";".join(str(value) for value in info.get("hierarchical_ik_candidate_errors_m", ())),
                        "hierarchical_ik_candidate_clearances_m": ";".join(str(value) for value in info.get("hierarchical_ik_candidate_clearances_m", ())),
                        "hierarchical_filter_intervention_ratio": info.get("hierarchical_filter_intervention_ratio", 0.0),
                        "hierarchical_filter_status": info.get("hierarchical_filter_status", "not_enabled"),
                        "hierarchical_servo_stall_steps": info.get("hierarchical_servo_stall_steps", 0),
                        "hierarchical_servo_stall_replans": info.get("hierarchical_servo_stall_replans", 0),
                        "hierarchical_plan_count": info.get("hierarchical_plan_count", 0),
                        "hierarchical_replan_count": info.get("hierarchical_replan_count", 0),
                        "hierarchical_hold_steps": info.get("hierarchical_hold_steps", 0),
                        "hierarchical_consecutive_hold_steps": info.get("hierarchical_consecutive_hold_steps", 0),
                        "hierarchical_waypoint_index": info.get("hierarchical_waypoint_index", 0),
                        "hierarchical_waypoint_count": info.get("hierarchical_waypoint_count", 0),
                        "hierarchical_path_progress": info.get("hierarchical_path_progress", 0.0),
                        "hierarchical_residual_budget": info.get("hierarchical_residual_budget", 0.0),
                        "hierarchical_nominal_qdot_norm": float(
                            np.linalg.norm(info.get("hierarchical_nominal_qdot", np.zeros(env.action_space.shape[0])))
                        ),
                        "residual_qdot_norm": float(
                            np.linalg.norm(info.get("residual_qdot", np.zeros(env.action_space.shape[0])))
                        ),
                        "risk_speed_scale": float(info.get("risk_speed_scale", 1.0)),
                        "risk_speed_h_min_m": float(info.get("risk_speed_h_min_m", float("nan"))),
                        "recovery_active": int(bool(info.get("recovery_active", False))),
                        "recovery_triggered": int(bool(info.get("recovery_triggered", False))),
                        "recovery_success": int(bool(info.get("recovery_success", False))),
                        "recovery_initially_unsafe": int(bool(info.get("recovery_initially_unsafe", False))),
                        "recovery_initial_h_min_m": float(info.get("recovery_initial_h_min_m", float("nan"))),
                        "recovery_command_norm": float(info.get("recovery_command_norm", 0.0)),
                        "safety_filter_status": filter_status,
                        "safety_filter_reason": info.get("safety_filter_reason", ""),
                        "safety_filter_intervention_norm": intervention_norm,
                        "safety_filter_safe_stop": int(bool(info.get("safety_filter_safe_stop", False))),
                        "safety_filter_active_constraints": info.get("safety_filter_active_constraints", 0),
                        "safety_filter_constraint_count": info.get("safety_filter_constraint_count", 0),
                        "safety_filter_active_constraint_categories": info.get(
                            "safety_filter_active_constraint_categories", ""
                        ),
                        "safety_filter_max_constraint_category": info.get(
                            "safety_filter_max_constraint_category", ""
                        ),
                        "safety_filter_infeasible_constraint_categories": info.get(
                            "safety_filter_infeasible_constraint_categories", ""
                        ),
                        "safety_filter_infeasibility_diagnostic_status": info.get(
                            "safety_filter_infeasibility_diagnostic_status", ""
                        ),
                        "safety_filter_projection_iterations": info.get(
                            "safety_filter_projection_iterations", 0
                        ),
                        "safety_filter_fallback_used": int(info.get("safety_filter_fallback_used", False)),
                        "safety_filter_qp_solver_used": int(info.get("safety_filter_qp_solver_used", False)),
                        "safety_filter_qp_solver_status": info.get("safety_filter_qp_solver_status", ""),
                        "safety_filter_fallback_stage": info.get("safety_filter_fallback_stage", ""),
                        "safety_filter_max_constraint_violation": info.get(
                            "safety_filter_max_constraint_violation", float("nan")
                        ),
                        "predictive_risk_status": info.get("predictive_risk_status", "not_enabled"),
                        "predictive_h_min_m": predictive_h_min,
                        "predictive_max_link_speed_bound_mps": float(
                            info.get("predictive_max_link_speed_bound_mps", float("nan"))
                        ),
                        "predictive_link_velocity_norms_mps": info.get(
                            "predictive_link_velocity_norms_mps", ""
                        ),
                        "predictive_h_by_link_m": info.get("predictive_h_by_link_m", ""),
                        "predictive_distance_by_link_m": info.get("predictive_distance_by_link_m", ""),
                        "predictive_robust_distance_by_link_m": info.get("predictive_robust_distance_by_link_m", ""),
                        "predictive_time_by_link_s": info.get("predictive_time_by_link_s", ""),
                        "safety_jacobian_command_by_link_mps": info.get(
                            "safety_jacobian_command_by_link_mps", ""
                        ),
                        "safety_drift_by_link_mps": info.get("safety_drift_by_link_mps", ""),
                        "safety_constraint_residual_by_link_mps": info.get(
                            "safety_constraint_residual_by_link_mps", ""
                        ),
                        "filter_obstacle_position_m": info.get("filter_obstacle_position_m", ""),
                        "filter_obstacle_velocity_mps": info.get("filter_obstacle_velocity_mps", ""),
                        "safety_filter_solve_time_s": solve_time_s,
                        "safety_filter_predictive_risk_time_s": predictive_risk_time_s,
                        "safety_filter_jacobian_workspace_time_s": jacobian_workspace_time_s,
                        "safety_filter_projection_time_s": projection_time_s,
                    }
                )
            prev_qdot_cmd = info["qdot_cmd"].copy()
            if terminated or truncated:
                break

        acc = np.asarray(accelerations, dtype=np.float32)
        jerk = np.asarray(jerks, dtype=np.float32)
        if success:
            hierarchical_failure_mode = "SUCCESS"
        elif termination_collision:
            hierarchical_failure_mode = "COLLISION_TERMINATION"
        elif env.hierarchical_enabled and not initial_ik_found:
            hierarchical_failure_mode = "IK_NOT_FOUND"
        elif env.hierarchical_enabled and not initial_plan_found:
            hierarchical_failure_mode = "PLAN_NOT_FOUND"
        elif filter_safe_stops > 0 or int(info.get("hierarchical_hold_steps", 0)) > 0:
            hierarchical_failure_mode = "FILTER_STOP_TIMEOUT"
        elif info.get("hierarchical_state") == "SERVO":
            hierarchical_failure_mode = "SERVO_TIMEOUT"
        else:
            hierarchical_failure_mode = "TRACK_TIMEOUT"
        rows.append(
            {
                "episode": episode,
                "seed": episode_seed,
                "seed_manifest": str(seed_manifest_arg) if seed_manifest_arg else "",
                "architecture_version": config.get("env", {}).get("hierarchical_control", {}).get("architecture_version", "legacy"),
                "manifest_shard_index": shard_index if shard_index is not None else "",
                "manifest_shard_count": shard_count if shard_count is not None else "",
                "reward": total_reward,
                "cost": total_cost,
                "length": step + 1,
                "success": int(success),
                "collision": int(collision),
                "collision_any": int(collision),
                "collision_capsule_overlap": int(collision_capsule_overlap),
                "collision_pybullet_contact": int(collision_pybullet_contact),
                "termination_collision": int(termination_collision),
                "termination_reason": termination_reason,
                "collision_termination_mode": env.collision_termination_mode,
                "unavoidable_collision": int(unavoidable_collision),
                "avoidable_collision": int(avoidable_collision),
                "collision_contact_link_indices": info.get("collision_contact_link_indices", ""),
                "collision_contact_link_names": info.get("collision_contact_link_names", ""),
                "collision_min_contact_distance": float(
                    info.get("collision_min_contact_distance", float("nan"))
                ),
                "non_end_link_collision": int(non_end_link_collision),
                "final_position_error": final_position_error,
                "completion_time": (step + 1) * float(config["env"]["control_dt"]),
                "min_distance": min(distances) if distances else 0.0,
                "mean_risk": float(np.mean(risks)) if risks else 0.0,
                "max_risk": float(np.max(risks)) if risks else 0.0,
                "safety_violation_count": violations,
                "safety_violation_rate": violations / max(step + 1, 1),
                "safety_filter_intervention_rate": (
                    filter_interventions / max(step + 1, 1) if safety_filter_enabled else float("nan")
                ),
                "mean_safety_filter_intervention_norm": (
                    float(np.mean(filter_intervention_norms)) if filter_intervention_norms else float("nan")
                ),
                "safety_filter_safe_stop_rate": (
                    filter_safe_stops / max(step + 1, 1) if safety_filter_enabled else float("nan")
                ),
                "safety_filter_infeasible_rate": (
                    filter_infeasible / max(step + 1, 1) if safety_filter_enabled else float("nan")
                ),
                "safety_filter_projection_failure_rate": (
                    filter_projection_failures / max(step + 1, 1) if safety_filter_enabled else float("nan")
                ),
                "safety_filter_recovery_relaxed_rate": (
                    filter_recovery_relaxed / max(step + 1, 1) if safety_filter_enabled else float("nan")
                ),
                "safety_filter_fallback_rate": (
                    filter_fallbacks / max(step + 1, 1) if safety_filter_enabled else float("nan")
                ),
                "safety_filter_qp_used_rate": (
                    filter_qp_used / max(step + 1, 1) if safety_filter_enabled else float("nan")
                ),
                "safety_filter_qp_timeout_rate": (
                    filter_qp_timeouts / max(step + 1, 1) if safety_filter_enabled else float("nan")
                ),
                "safety_filter_compute_budget_stop_rate": (
                    filter_compute_budget_stops / max(step + 1, 1) if safety_filter_enabled else float("nan")
                ),
                "predictive_near_miss_rate": (
                    predictive_near_misses / max(step + 1, 1) if safety_filter_enabled else float("nan")
                ),
                "min_predictive_h_m": min(predictive_h_mins) if predictive_h_mins else float("nan"),
                "mean_safety_filter_solve_time_s": (
                    float(np.mean(filter_solve_times_s)) if filter_solve_times_s else float("nan")
                ),
                "max_safety_filter_solve_time_s": max(filter_solve_times_s, default=float("nan")),
                "mean_safety_filter_predictive_risk_time_s": (
                    float(np.mean(filter_predictive_risk_times_s)) if filter_predictive_risk_times_s else float("nan")
                ),
                "mean_safety_filter_jacobian_workspace_time_s": (
                    float(np.mean(filter_jacobian_workspace_times_s)) if filter_jacobian_workspace_times_s else float("nan")
                ),
                "mean_safety_filter_projection_time_s": (
                    float(np.mean(filter_projection_times_s)) if filter_projection_times_s else float("nan")
                ),
                "mean_action_variation": float(np.mean(action_variations)) if action_variations else 0.0,
                "mean_hierarchical_nominal_qdot_norm": (
                    float(np.mean(nominal_qdot_norms)) if nominal_qdot_norms else 0.0
                ),
                "mean_residual_qdot_norm": (
                    float(np.mean(residual_qdot_norms)) if residual_qdot_norms else 0.0
                ),
                "task_space_mean_requested_toward_goal_mps": float(np.mean(task_requested_toward_goal)) if task_requested_toward_goal else float("nan"),
                "task_space_mean_executed_toward_goal_mps": float(np.mean(task_executed_toward_goal)) if task_executed_toward_goal else float("nan"),
                "task_space_mean_filter_velocity_loss_mps": float(np.mean(task_velocity_losses)) if task_velocity_losses else float("nan"),
                "task_space_positive_error_delta_rate": float(np.mean(np.asarray(task_error_deltas) > 0.0)) if task_error_deltas else float("nan"),
                "rms_acceleration": float(np.sqrt(np.mean(np.square(acc)))) if len(acc) else 0.0,
                "rms_jerk": float(np.sqrt(np.mean(np.square(jerk)))) if len(jerk) else 0.0,
                "recovery_triggered": int(recovery_triggered),
                "recovery_success": int(recovery_success),
                "recovery_steps": recovery_steps,
                "recovery_duration_s": recovery_steps * float(config["env"]["control_dt"]),
                "initially_unsafe": int(initially_unsafe),
                "initial_predictive_h_m": initial_predictive_h_m,
                "nominal_only": int(nominal_only),
                "hierarchical_final_state": info.get("hierarchical_state", "disabled"),
                "hierarchical_plan_reason": info.get("hierarchical_plan_reason", ""),
                "hierarchical_planning_time_s": info.get("hierarchical_planning_time_s", float("nan")),
                "hierarchical_planning_time_total_s": info.get(
                    "hierarchical_planning_time_total_s", float("nan")
                ),
                "hierarchical_ik_candidate_count": info.get("hierarchical_ik_candidate_count", 0),
                "hierarchical_ik_found": int(bool(info.get("hierarchical_ik_found", False))),
                "hierarchical_plan_found": int(bool(info.get("hierarchical_plan_found", False))),
                "hierarchical_direct_path": int(bool(info.get("hierarchical_direct_path", False))),
                "hierarchical_selected_candidate_index": info.get("hierarchical_selected_candidate_index", -1),
                "hierarchical_selected_path_length_rad": info.get("hierarchical_selected_path_length_rad", float("nan")),
                "hierarchical_selected_path_min_clearance_m": info.get("hierarchical_selected_path_min_clearance_m", float("nan")),
                "hierarchical_ik_candidate_errors_m": ";".join(str(value) for value in info.get("hierarchical_ik_candidate_errors_m", ())),
                "hierarchical_ik_candidate_clearances_m": ";".join(str(value) for value in info.get("hierarchical_ik_candidate_clearances_m", ())),
                "hierarchical_filter_intervention_ratio": info.get("hierarchical_filter_intervention_ratio", 0.0),
                "hierarchical_filter_status": info.get("hierarchical_filter_status", "not_enabled"),
                "hierarchical_servo_stall_steps": info.get("hierarchical_servo_stall_steps", 0),
                "hierarchical_servo_stall_replans": info.get("hierarchical_servo_stall_replans", 0),
                "task_space_mean_requested_toward_goal_mps": float("nan"),
                "hierarchical_plan_count": info.get("hierarchical_plan_count", 0),
                "hierarchical_replan_count": info.get("hierarchical_replan_count", 0),
                "hierarchical_hold_steps": info.get("hierarchical_hold_steps", 0),
                "hierarchical_initial_ik_found": int(initial_ik_found),
                "hierarchical_initial_plan_found": int(initial_plan_found),
                "hierarchical_initial_direct_path": int(initial_direct_path),
                "hierarchical_initial_plan_reason": initial_plan_reason,
                "hierarchical_initial_planning_time_s": initial_planning_time_s,
                "hierarchical_initial_plan_iterations": initial_plan_iterations,
                "hierarchical_initial_ik_candidate_count": initial_ik_candidate_count,
                "hierarchical_final_path_progress": info.get("hierarchical_path_progress", 0.0),
                "hierarchical_failure_mode": hierarchical_failure_mode,
                **{
                    f"hierarchical_{state.lower()}_rate": hierarchical_state_counts.get(state, 0) / max(step + 1, 1)
                    for state in env.hierarchical_state_names
                },
            }
        )
        if trace_dir is not None and trace_rows:
            trace_path = trace_dir / f"episode_{episode:04d}.csv"
            with open(trace_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(trace_rows[0].keys()))
                writer.writeheader()
                writer.writerows(trace_rows)

    env.close()
    if output_arg:
        output = Path(output_arg)
    elif checkpoint:
        output = Path(checkpoint).resolve().parent / "eval_metrics.csv"
    else:
        output = Path("outputs/hierarchical/nominal_only_eval.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "collision_rate": float(np.mean([r["collision"] for r in rows])),
        "collision_any_rate": float(np.mean([r["collision_any"] for r in rows])),
        "collision_capsule_overlap_rate": float(np.mean([r["collision_capsule_overlap"] for r in rows])),
        "collision_pybullet_contact_rate": float(np.mean([r["collision_pybullet_contact"] for r in rows])),
        "termination_collision_rate": float(np.mean([r["termination_collision"] for r in rows])),
        "unavoidable_collision_rate": float(np.mean([r["unavoidable_collision"] for r in rows])),
        "avoidable_collision_rate": float(np.mean([r["avoidable_collision"] for r in rows])),
        "non_end_link_collision_rate": float(np.mean([r["non_end_link_collision"] for r in rows])),
        "mean_final_position_error": float(np.mean([r["final_position_error"] for r in rows])),
        "mean_min_distance": float(np.mean([r["min_distance"] for r in rows])),
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "mean_cost": float(np.mean([r["cost"] for r in rows])),
        "mean_safety_violation_count": float(np.mean([r["safety_violation_count"] for r in rows])),
        "mean_safety_filter_intervention_rate": mean_finite(rows, "safety_filter_intervention_rate"),
        "mean_safety_filter_safe_stop_rate": mean_finite(rows, "safety_filter_safe_stop_rate"),
        "mean_safety_filter_infeasible_rate": mean_finite(rows, "safety_filter_infeasible_rate"),
        "mean_safety_filter_projection_failure_rate": mean_finite(rows, "safety_filter_projection_failure_rate"),
        "mean_safety_filter_recovery_relaxed_rate": mean_finite(rows, "safety_filter_recovery_relaxed_rate"),
        "mean_safety_filter_fallback_rate": mean_finite(rows, "safety_filter_fallback_rate"),
        "mean_safety_filter_qp_used_rate": mean_finite(rows, "safety_filter_qp_used_rate"),
        "mean_safety_filter_qp_timeout_rate": mean_finite(rows, "safety_filter_qp_timeout_rate"),
        "mean_safety_filter_compute_budget_stop_rate": mean_finite(rows, "safety_filter_compute_budget_stop_rate"),
        "mean_predictive_near_miss_rate": mean_finite(rows, "predictive_near_miss_rate"),
        "mean_min_predictive_h_m": mean_finite(rows, "min_predictive_h_m"),
        "mean_safety_filter_solve_time_s": mean_finite(rows, "mean_safety_filter_solve_time_s"),
        "max_safety_filter_solve_time_s": max_finite(rows, "max_safety_filter_solve_time_s"),
        "mean_safety_filter_predictive_risk_time_s": mean_finite(rows, "mean_safety_filter_predictive_risk_time_s"),
        "mean_safety_filter_jacobian_workspace_time_s": mean_finite(rows, "mean_safety_filter_jacobian_workspace_time_s"),
        "mean_safety_filter_projection_time_s": mean_finite(rows, "mean_safety_filter_projection_time_s"),
        "mean_action_variation": float(np.mean([r["mean_action_variation"] for r in rows])),
        "mean_rms_acceleration": float(np.mean([r["rms_acceleration"] for r in rows])),
        "mean_rms_jerk": float(np.mean([r["rms_jerk"] for r in rows])),
        "hierarchical_initial_ik_found_rate": float(
            np.mean([r["hierarchical_initial_ik_found"] for r in rows])
        ),
        "hierarchical_initial_plan_found_rate": float(
            np.mean([r["hierarchical_initial_plan_found"] for r in rows])
        ),
        "hierarchical_initial_direct_path_rate": float(
            np.mean([r["hierarchical_initial_direct_path"] for r in rows])
        ),
        "hierarchical_mean_initial_planning_time_s": mean_finite(
            rows, "hierarchical_initial_planning_time_s"
        ),
        "hierarchical_mean_planning_time_total_s": mean_finite(
            rows, "hierarchical_planning_time_total_s"
        ),
        "hierarchical_mean_replan_count": float(
            np.mean([r["hierarchical_replan_count"] for r in rows])
        ),
        "hierarchical_mean_hold_steps": float(
            np.mean([r["hierarchical_hold_steps"] for r in rows])
        ),
        "hierarchical_plan_conditioned_success_rate": (
            float(
                np.mean(
                    [r["success"] for r in rows if r["hierarchical_initial_plan_found"]]
                )
            )
            if any(r["hierarchical_initial_plan_found"] for r in rows)
            else float("nan")
        ),
        "hierarchical_failure_mode_counts": {
            mode: sum(r["hierarchical_failure_mode"] == mode for r in rows)
            for mode in (
                "SUCCESS",
                "IK_NOT_FOUND",
                "PLAN_NOT_FOUND",
                "TRACK_TIMEOUT",
                "SERVO_TIMEOUT",
                "FILTER_STOP_TIMEOUT",
                "COLLISION_TERMINATION",
            )
        },
        **{
            f"hierarchical_mean_{state.lower()}_rate": mean_finite(
                rows, f"hierarchical_{state.lower()}_rate"
            )
            for state in env.hierarchical_state_names
        },
    }
    print(summary)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
