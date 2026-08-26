"""Run a deterministic rest-pose grid IK audit on the unknown nominal seeds."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config

try:
    from scripts.seed_manifest import load_seed_manifest
except ModuleNotFoundError:
    from seed_manifest import load_seed_manifest


FIELDNAMES = [
    "episode",
    "seed",
    "classification_after_grid",
    "grid_levels",
    "rest_pose_count",
    "ik_attempts",
    "unique_solution_count",
    "strict_candidate_count",
    "raw_goal_reachable_count",
    "obstacle_free_goal_reachable_count",
    "min_goal_error_m",
    "best_strict_error_m",
    "best_strict_clearance_m",
    "rejection_counts",
    "best_strict_q",
    "best_raw_q",
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


def deterministic_rest_poses(
    lower: np.ndarray,
    upper: np.ndarray,
    current_q: np.ndarray,
    default_q: np.ndarray,
    levels: int,
) -> list[np.ndarray]:
    if levels < 2:
        raise ValueError("grid levels must be at least 2")
    fractions = np.linspace(0.0, 1.0, levels, dtype=np.float64)
    poses = [np.asarray(current_q, dtype=np.float64), np.asarray(default_q, dtype=np.float64)]
    for fraction_tuple in itertools.product(fractions, repeat=len(lower)):
        fraction = np.asarray(fraction_tuple, dtype=np.float64)
        poses.append(lower + fraction * (upper - lower))
    unique: list[np.ndarray] = []
    for pose in poses:
        pose = np.clip(np.asarray(pose, dtype=np.float64), lower, upper)
        if not any(np.linalg.norm(pose - previous) < 1.0e-10 for previous in unique):
            unique.append(pose)
    return unique


def audit_seed(env: UR5DynamicObstacleEnv, seed: int, episode: int, grid_levels: int) -> dict[str, Any]:
    _, reset_info = env.reset(seed=seed)
    lower, upper = env._joint_position_limits()
    current_q, _ = env._joint_state()
    default_q = np.asarray(env.robot_cfg["reset"]["default_joint_positions"], dtype=np.float64)
    rest_poses = deterministic_rest_poses(lower, upper, current_q, default_q, grid_levels)
    goal = np.asarray(env.goal, dtype=np.float64)
    tolerance = float(env.hierarchical_planner_cfg.get("ik_goal_tolerance_m", env.success_tolerance))
    max_iterations = int(env.hierarchical_planner_cfg.get("ik_audit_max_iterations", 500))
    residual_threshold = float(env.hierarchical_planner_cfg.get("ik_audit_residual_threshold", 1.0e-6))
    rejection_counts = {
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
    unique_solutions: list[np.ndarray] = []
    strict_candidates: list[tuple[float, float, np.ndarray]] = []
    raw_candidates: list[tuple[float, np.ndarray]] = []
    min_goal_error = float("inf")
    attempts = 0
    for rest in rest_poses:
        attempts += 1
        try:
            solution = p.calculateInverseKinematics(
                env.robot_id,
                env.tool_link_id,
                goal.tolist(),
                lowerLimits=lower.tolist(),
                upperLimits=upper.tolist(),
                jointRanges=(upper - lower).tolist(),
                restPoses=rest.tolist(),
                maxNumIterations=max_iterations,
                residualThreshold=residual_threshold,
                physicsClientId=env.physics_client_id,
            )
        except (TypeError, ValueError, p.error):
            rejection_counts["solver_error"] += 1
            continue
        solution = np.asarray(solution, dtype=np.float64)
        if solution.size <= max(env.control_jacobian_columns):
            rejection_counts["short_solution"] += 1
            continue
        candidate = np.clip(solution[env.control_jacobian_columns], lower, upper)
        if any(np.linalg.norm(candidate - previous) < 1.0e-3 for previous in unique_solutions):
            rejection_counts["duplicate"] += 1
            continue
        unique_solutions.append(candidate.copy())
        error = float(env._planning_goal_error(candidate))
        min_goal_error = min(min_goal_error, error)
        if error > tolerance:
            rejection_counts["goal_error"] += 1
            continue
        raw_candidates.append((error, candidate.copy()))
        if env._planning_state_validity_reason(candidate, ignore_obstacle=True) is None:
            raw_goal_reachable_count = True
        else:
            raw_goal_reachable_count = False
        state_reason = env._planning_state_validity_reason(candidate)
        if state_reason is not None:
            rejection_counts[state_reason] += 1
            continue
        clearance = float(env._planning_clearance(candidate))
        strict_candidates.append((error, clearance, candidate.copy()))
        if not raw_goal_reachable_count:
            raise RuntimeError("strict candidate cannot be invalid under ignore_obstacle checks")

    strict_candidates.sort(key=lambda item: (item[0], -item[1]))
    raw_candidates.sort(key=lambda item: item[0])
    strict_count = len(strict_candidates)
    raw_count = len(raw_candidates)
    obstacle_free_count = sum(
        env._planning_state_validity_reason(candidate, ignore_obstacle=True) is None
        for _, candidate in raw_candidates
    )
    if strict_count > 0:
        classification = "certified_feasible"
    else:
        classification = "unknown"
    return {
        "episode": episode,
        "seed": seed,
        "classification_after_grid": classification,
        "grid_levels": grid_levels,
        "rest_pose_count": len(rest_poses),
        "ik_attempts": attempts,
        "unique_solution_count": len(unique_solutions),
        "strict_candidate_count": strict_count,
        "raw_goal_reachable_count": raw_count,
        "obstacle_free_goal_reachable_count": obstacle_free_count,
        "min_goal_error_m": min_goal_error,
        "best_strict_error_m": strict_candidates[0][0] if strict_candidates else None,
        "best_strict_clearance_m": strict_candidates[0][1] if strict_candidates else None,
        "rejection_counts": _json(rejection_counts),
        "best_strict_q": _json(strict_candidates[0][2].tolist()) if strict_candidates else "[]",
        "best_raw_q": _json(raw_candidates[0][1].tolist()) if raw_candidates else "[]",
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
        raise ValueError("grid audit requires planner.ik_feasibility_audit_only=true")
    seeds = load_seed_manifest(args.seed_manifest)
    env = UR5DynamicObstacleEnv(config, method=str(config["eval"]["method"]))
    rows: list[dict[str, Any]] = []
    grid_levels = int(planner_cfg.get("ik_audit_grid_levels", 3))
    try:
        for episode, seed in enumerate(seeds):
            if episode % args.manifest_shard_count != args.manifest_shard_index:
                continue
            rows.append(audit_seed(env, int(seed), episode, grid_levels))
    finally:
        env.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    counts = {name: sum(row["classification_after_grid"] == name for row in rows) for name in ("certified_feasible", "unknown")}
    print(json.dumps({"episodes": len(rows), "classification_counts": counts}, ensure_ascii=True))


if __name__ == "__main__":
    main()
