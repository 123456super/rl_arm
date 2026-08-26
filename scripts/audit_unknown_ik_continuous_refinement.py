"""Deterministic continuous endpoint-IK refinement for unknown nominal seeds.

This is a constructive follow-up, not an infeasibility proof.  Raw-unreachable
seeds minimize endpoint position error from deterministic grid IK starts;
raw-reachable/obstacle-blocked seeds maximize strict capsule clearance subject
to the original success-ball constraint.  Failure remains ``unknown``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
from scipy.optimize import minimize

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config

try:
    from scripts.audit_unknown_ik_grid import deterministic_rest_poses
    from scripts.seed_manifest import load_seed_manifest
except ModuleNotFoundError:
    from audit_unknown_ik_grid import deterministic_rest_poses
    from seed_manifest import load_seed_manifest


FIELDNAMES = [
    "episode",
    "seed",
    "classification_after_refinement",
    "refinement_mode",
    "grid_levels",
    "grid_rest_pose_count",
    "grid_unique_solution_count",
    "grid_raw_goal_reachable_count",
    "grid_obstacle_free_goal_reachable_count",
    "refinement_start_count",
    "refinement_attempt_count",
    "refinement_success_count",
    "min_goal_error_before_m",
    "min_goal_error_after_m",
    "max_clearance_before_m",
    "max_clearance_after_m",
    "strict_candidate_count_after",
    "best_q",
    "best_error_m",
    "best_clearance_m",
    "rejection_counts",
    "goal_m",
    "obstacle_center_m",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--manifest-shard-index", type=int, required=True)
    parser.add_argument("--manifest-shard-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _unique_qs(qs: list[np.ndarray], tolerance: float = 1.0e-3) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for q in qs:
        q = np.asarray(q, dtype=np.float64)
        if not any(np.linalg.norm(q - previous) < tolerance for previous in unique):
            unique.append(q.copy())
    return unique


def _solve_grid_ik(
    env: UR5DynamicObstacleEnv,
    rest_poses: list[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    goal: np.ndarray,
    max_iterations: int,
    residual_threshold: float,
) -> list[np.ndarray]:
    solutions: list[np.ndarray] = []
    for rest in rest_poses:
        try:
            solution = p.calculateInverseKinematics(
                env.robot_id,
                env.tool_link_id,
                goal.tolist(),
                lowerLimits=lower.tolist(),
                upperLimits=upper.tolist(),
                jointRanges=(upper - lower).tolist(),
                restPoses=np.asarray(rest, dtype=np.float64).tolist(),
                maxNumIterations=max_iterations,
                residualThreshold=residual_threshold,
                physicsClientId=env.physics_client_id,
            )
        except (TypeError, ValueError, p.error):
            continue
        solution = np.asarray(solution, dtype=np.float64)
        if solution.size <= max(env.control_jacobian_columns):
            continue
        solutions.append(np.clip(solution[env.control_jacobian_columns], lower, upper))
    return _unique_qs(solutions)


def _sort_by_error(env: UR5DynamicObstacleEnv, qs: list[np.ndarray]) -> list[tuple[float, np.ndarray]]:
    values = [(float(env._planning_goal_error(q)), q.copy()) for q in qs]
    return sorted(values, key=lambda item: item[0])


def _sort_by_clearance(env: UR5DynamicObstacleEnv, qs: list[np.ndarray]) -> list[tuple[float, float, np.ndarray]]:
    values = [
        (float(env._planning_clearance(q)), float(env._planning_goal_error(q)), q.copy())
        for q in qs
    ]
    return sorted(values, key=lambda item: (-item[0], item[1]))


def audit_seed(
    env: UR5DynamicObstacleEnv,
    seed: int,
    episode: int,
    planner_cfg: dict[str, Any],
) -> dict[str, Any]:
    _, reset_info = env.reset(seed=seed)
    lower, upper = env._joint_position_limits()
    current_q, _ = env._joint_state()
    default_q = np.asarray(env.robot_cfg["reset"]["default_joint_positions"], dtype=np.float64)
    grid_levels = int(planner_cfg.get("ik_refinement_grid_levels", 3))
    rest_poses = deterministic_rest_poses(lower, upper, current_q, default_q, grid_levels)
    goal = np.asarray(env.goal, dtype=np.float64)
    tolerance = float(planner_cfg.get("ik_goal_tolerance_m", env.success_tolerance))
    solutions = _solve_grid_ik(
        env,
        rest_poses,
        lower,
        upper,
        goal,
        int(planner_cfg.get("ik_refinement_seed_max_iterations", 1000)),
        float(planner_cfg.get("ik_refinement_seed_residual_threshold", 1.0e-8)),
    )
    error_values = _sort_by_error(env, solutions)
    min_before = error_values[0][0] if error_values else float("inf")
    raw = [q for error, q in error_values if error <= tolerance]
    obstacle_free = [
        q for q in raw if env._planning_state_validity_reason(q, ignore_obstacle=True) is None
    ]
    clearance_values = _sort_by_clearance(env, raw) if raw else []
    max_clearance_before = clearance_values[0][0] if clearance_values else float("-inf")

    # Use only a bounded deterministic shortlist.  For raw-error failures,
    # include both IK outputs and coarse rest poses, then keep lowest errors.
    if raw:
        mode = "clearance_under_goal_ball"
        starts = [q for _, _, q in clearance_values]
        starts = starts[: int(planner_cfg.get("ik_refinement_starts", 16))]
    else:
        mode = "goal_error_minimization"
        coarse = _sort_by_error(env, rest_poses)
        starts = [q for _, q in error_values[: int(planner_cfg.get("ik_refinement_starts", 16))]]
        starts.extend(q for _, q in coarse[: int(planner_cfg.get("ik_refinement_coarse_starts", 16))])
        starts = _unique_qs(starts)[: int(planner_cfg.get("ik_refinement_starts", 16))]

    best_error = min_before
    best_clearance = max_clearance_before
    best_q: np.ndarray | None = error_values[0][1].copy() if error_values else None
    if clearance_values and clearance_values[0][0] > best_clearance:
        best_clearance = clearance_values[0][0]
        best_error = clearance_values[0][1]
        best_q = clearance_values[0][2].copy()
    strict_qs: list[np.ndarray] = []
    attempts = 0
    solver_successes = 0
    max_iterations = int(planner_cfg.get("ik_refinement_max_iterations", 160))
    if mode == "goal_error_minimization":
        def objective(q: np.ndarray) -> float:
            error = float(env._planning_goal_error(q))
            return error * error

        for start in starts:
            attempts += 1
            result = minimize(
                objective,
                np.clip(start, lower, upper),
                method="L-BFGS-B",
                bounds=list(zip(lower, upper, strict=True)),
                options={"maxiter": max_iterations, "ftol": 1.0e-14, "gtol": 1.0e-10, "maxls": 30},
            )
            if bool(result.success):
                solver_successes += 1
            q = np.clip(np.asarray(result.x, dtype=np.float64), lower, upper)
            error = float(env._planning_goal_error(q))
            clearance = float(env._planning_clearance(q))
            if error < best_error or (abs(error - best_error) <= 1.0e-10 and clearance > best_clearance):
                best_error, best_clearance, best_q = error, clearance, q.copy()
            if error <= tolerance and env._planning_state_validity_reason(q) is None:
                strict_qs.append(q.copy())
    else:
        def objective(q: np.ndarray) -> float:
            return -float(env._planning_clearance(q))

        def goal_constraint(q: np.ndarray) -> float:
            return tolerance - float(env._planning_goal_error(q))

        for start in starts:
            attempts += 1
            result = minimize(
                objective,
                np.clip(start, lower, upper),
                method="SLSQP",
                bounds=list(zip(lower, upper, strict=True)),
                constraints=[{"type": "ineq", "fun": goal_constraint}],
                options={"maxiter": max_iterations, "ftol": 1.0e-10, "disp": False},
            )
            if bool(result.success):
                solver_successes += 1
            q = np.clip(np.asarray(result.x, dtype=np.float64), lower, upper)
            error = float(env._planning_goal_error(q))
            clearance = float(env._planning_clearance(q))
            # Never report an optimizer point outside the original success
            # ball as an improved clearance certificate.  SLSQP may terminate
            # unsuccessfully on the piecewise collision-distance objective.
            if error <= tolerance and (
                clearance > best_clearance
                or (abs(clearance - best_clearance) <= 1.0e-10 and error < best_error)
            ):
                best_error, best_clearance, best_q = error, clearance, q.copy()
            if error <= tolerance and env._planning_state_validity_reason(q) is None:
                strict_qs.append(q.copy())

    strict_qs = _unique_qs(strict_qs)
    classification = "certified_feasible" if strict_qs else "unknown"
    rejection_counts: dict[str, int] = {}
    for q in starts:
        reason = env._planning_state_validity_reason(q)
        if reason is not None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    return {
        "episode": episode,
        "seed": seed,
        "classification_after_refinement": classification,
        "refinement_mode": mode,
        "grid_levels": grid_levels,
        "grid_rest_pose_count": len(rest_poses),
        "grid_unique_solution_count": len(solutions),
        "grid_raw_goal_reachable_count": len(raw),
        "grid_obstacle_free_goal_reachable_count": len(obstacle_free),
        "refinement_start_count": len(starts),
        "refinement_attempt_count": attempts,
        "refinement_success_count": solver_successes,
        "min_goal_error_before_m": min_before,
        "min_goal_error_after_m": best_error,
        "max_clearance_before_m": max_clearance_before,
        "max_clearance_after_m": best_clearance,
        "strict_candidate_count_after": len(strict_qs),
        "best_q": _json(best_q.tolist() if best_q is not None else []),
        "best_error_m": best_error if np.isfinite(best_error) else None,
        "best_clearance_m": best_clearance if np.isfinite(best_clearance) else None,
        "rejection_counts": _json(rejection_counts),
        "goal_m": _json(goal.tolist()),
        "obstacle_center_m": _json(np.asarray(env.obstacle_center, dtype=np.float64).tolist()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.manifest_shard_count <= 0 or not 0 <= args.manifest_shard_index < args.manifest_shard_count:
        raise ValueError("invalid manifest shard index/count")
    config = load_config(args.config)
    planner_cfg = config["env"]["hierarchical_control"]["planner"]
    if not bool(planner_cfg.get("ik_feasibility_audit_only", False)):
        raise ValueError("continuous refinement requires planner.ik_feasibility_audit_only=true")
    seeds = load_seed_manifest(args.seed_manifest)
    env = UR5DynamicObstacleEnv(config, method=str(config["eval"]["method"]))
    rows: list[dict[str, Any]] = []
    try:
        for episode, seed in enumerate(seeds):
            if episode % args.manifest_shard_count != args.manifest_shard_index:
                continue
            rows.append(audit_seed(env, int(seed), episode, planner_cfg))
    finally:
        env.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    counts = {
        name: sum(row["classification_after_refinement"] == name for row in rows)
        for name in ("certified_feasible", "unknown")
    }
    print(json.dumps({"episodes": len(rows), "classification_counts": counts}, ensure_ascii=True))


if __name__ == "__main__":
    main()
