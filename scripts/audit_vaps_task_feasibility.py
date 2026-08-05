"""Offline three-layer task-feasibility labels for the VAPS protocol.

The audit keeps every reset in its output. A successful label means that the
deterministic candidate search found a witness, while a ``not_found`` label is
not a proof that no trajectory exists. No policy action, training, recovery,
or runtime safety-filter setting is changed by this script.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config


PROTOCOL = "vaps_task_feasibility_precheck_v1"
DEFAULT_CONFIG = "configs/experiments/vaps/v3_task_feasibility_precheck.yaml"
DEFAULT_SEED_MANIFEST = "configs/experiments/vaps_manifests/v2_g2_validation.json"
DEFAULT_SHARD_COUNT = 4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit IK and candidate path feasibility for fixed reset seeds.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seed-manifest", default=DEFAULT_SEED_MANIFEST)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _load_seed_manifest(path: str | Path) -> list[int]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or not isinstance(payload.get("seeds"), list):
        raise ValueError("seed manifest must be a JSON object with a seeds list")
    seeds = [int(seed) for seed in payload["seeds"]]
    if not seeds or len(seeds) != len(set(seeds)) or any(seed <= 0 for seed in seeds):
        raise ValueError("seed manifest must contain unique positive seeds")
    return seeds


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    env_cfg = config["env"]
    precheck = config.get("precheck")
    if not isinstance(precheck, dict) or precheck.get("protocol") != PROTOCOL:
        raise ValueError("config must use the v3 task-feasibility precheck protocol")
    fixed_values = (
        (env_cfg["control_dt"], 0.05, "env.control_dt"),
        (env_cfg["time_step"], 0.0041666667, "env.time_step"),
        (env_cfg["obstacle"]["radius"], 0.07, "env.obstacle.radius"),
        (env_cfg["safety_filter"]["geometry_margin_m"], 0.03, "geometry_margin_m"),
        (env_cfg["safety_filter"]["tracking_error_bound_m"], 0.01, "tracking_error_bound_m"),
        (env_cfg["safety_filter"]["control_delay_s"], 0.05, "control_delay_s"),
        (config["risk"]["d_safe"], 0.12, "risk.d_safe"),
    )
    for value, expected, name in fixed_values:
        if not np.isclose(float(value), expected, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"{name} must be {expected}")
    if env_cfg["obstacle"]["speed_range"] != [0.05, 0.05]:
        raise ValueError("precheck requires fixed obstacle speed 0.05 m/s")
    if not bool(env_cfg["obstacle"]["enabled"]):
        raise ValueError("precheck requires the dynamic obstacle to remain enabled")
    if int(precheck["ik_attempts"]) <= 0 or float(precheck["ik_position_tolerance_m"]) <= 0.0:
        raise ValueError("precheck IK settings must be positive")
    if float(precheck["path_dt_s"]) != float(env_cfg["control_dt"]):
        raise ValueError("precheck.path_dt_s must equal env.control_dt")
    if float(precheck["path_max_duration_s"]) <= 0.0:
        raise ValueError("precheck.path_max_duration_s must be positive")
    if float(precheck["self_collision_distance_m"]) < 0.0:
        raise ValueError("precheck.self_collision_distance_m must be non-negative")
    for key in ("recovery_mode_enabled", "recovery_allow_constraint_relaxation", "recovery_maximize_min_clearance"):
        if bool(env_cfg["safety_filter"].get(key, False)):
            raise ValueError(f"precheck requires {key}=false")
    return precheck


def _set_joint_state(env: UR5DynamicObstacleEnv, q: np.ndarray) -> None:
    for joint_id, value in zip(env.joint_ids, q, strict=True):
        p.resetJointState(env.robot_id, joint_id, float(value), targetVelocity=0.0, physicsClientId=env.physics_client_id)
    p.performCollisionDetection(physicsClientId=env.physics_client_id)


def _joint_limits(env: UR5DynamicObstacleEnv) -> tuple[np.ndarray, np.ndarray]:
    return env._joint_position_limits()


def _self_collision(env: UR5DynamicObstacleEnv, distance_m: float) -> bool:
    points = p.getClosestPoints(
        env.robot_id,
        env.robot_id,
        distance=float(distance_m),
        physicsClientId=env.physics_client_id,
    )
    for point in points:
        link_a, link_b = int(point[3]), int(point[4])
        if link_a == link_b or (link_a >= 0 and link_b >= 0 and abs(link_a - link_b) <= 1):
            continue
        if float(point[8]) <= float(distance_m):
            return True
    return False


def _ik_candidates(
    env: UR5DynamicObstacleEnv,
    goal: np.ndarray,
    q_start: np.ndarray,
    attempts: int,
    tolerance_m: float,
    self_collision_distance_m: float,
    seed: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    lower, upper = _joint_limits(env)
    rng = np.random.default_rng(seed + 17001)
    rest_poses = [q_start] + [rng.uniform(lower, upper) for _ in range(attempts - 1)]
    candidates: list[np.ndarray] = []
    rejected: Counter[str] = Counter()
    for rest in rest_poses:
        solution = p.calculateInverseKinematics(
            env.robot_id,
            env.tool_link_id,
            targetPosition=np.asarray(goal, dtype=np.float64).tolist(),
            lowerLimits=lower.tolist(),
            upperLimits=upper.tolist(),
            jointRanges=(upper - lower).tolist(),
            restPoses=np.asarray(rest, dtype=np.float64).tolist(),
            maxNumIterations=200,
            residualThreshold=tolerance_m,
            physicsClientId=env.physics_client_id,
        )
        raw = np.asarray(solution, dtype=np.float64)
        if raw.size != len(env.jacobian_joint_ids):
            rejected["ik_dimension"] += 1
            continue
        q = raw[np.asarray(env.control_jacobian_columns, dtype=int)]
        if not np.isfinite(q).all() or np.any(q < lower - 1.0e-7) or np.any(q > upper + 1.0e-7):
            rejected["joint_limits"] += 1
            continue
        _set_joint_state(env, q)
        ee_position, _ = env._end_effector_state()
        error_m = float(np.linalg.norm(np.asarray(ee_position, dtype=np.float64) - goal))
        if error_m > tolerance_m:
            rejected["position_residual"] += 1
            continue
        if _self_collision(env, self_collision_distance_m):
            rejected["self_collision"] += 1
            continue
        if not any(np.max(np.abs(q - previous), initial=0.0) <= 1.0e-6 for previous in candidates):
            candidates.append(q.copy())
    _set_joint_state(env, q_start)
    return candidates, {
        "attempts": attempts,
        "accepted_candidate_count": len(candidates),
        "rejected_reason_counts": dict(rejected),
        "status": "reachable" if candidates else "not_reachable",
    }


def _smooth_path(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    dt_s: float,
    max_duration_s: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    delta = np.abs(q_goal - q_start)
    min_duration = max(
        float(np.max(1.5 * delta / velocity_limits, initial=0.0)),
        float(np.max(np.sqrt(6.0 * delta / acceleration_limits), initial=0.0)),
        dt_s,
    )
    if min_duration > max_duration_s + 1.0e-12:
        return None, {"status": "not_found", "reason": "minimum_duration_exceeds_budget", "duration_s": min_duration}
    steps = max(1, int(math.ceil(min_duration / dt_s)))
    duration = steps * dt_s
    tau = np.linspace(0.0, 1.0, steps + 1)
    smooth = 3.0 * tau**2 - 2.0 * tau**3
    path = q_start[None, :] + smooth[:, None] * (q_goal - q_start)[None, :]
    return path, {"status": "candidate", "reason": "", "duration_s": duration}


def _path_limits_ok(
    path: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    dt_s: float,
    tolerance: float,
) -> tuple[bool, dict[str, Any]]:
    qdot = np.diff(path, axis=0) / dt_s
    qddot = np.diff(qdot, axis=0) / dt_s
    position_violation = float(
        max(np.max(q_lower[None, :] - path), np.max(path - q_upper[None, :]), 0.0)
    )
    velocity_violation = float(max(np.max(np.abs(qdot) - velocity_limits[None, :]), 0.0))
    acceleration_violation = float(max(np.max(np.abs(qddot) - acceleration_limits[None, :]), 0.0))
    return (
        max(position_violation, velocity_violation, acceleration_violation) <= tolerance,
        {
            "max_position_violation_rad": position_violation,
            "max_velocity_violation_radps": velocity_violation,
            "max_acceleration_violation_radps2": acceleration_violation,
        },
    )


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    fraction = 0.0 if denominator <= 1.0e-14 else float(np.dot(point - start, segment) / denominator)
    closest = start + np.clip(fraction, 0.0, 1.0) * segment
    return float(np.linalg.norm(point - closest))


def _capsule_clearance(env: UR5DynamicObstacleEnv, obstacle_center: np.ndarray) -> float:
    radius = float(env.obstacle_cfg["radius"])
    clearances = [
        _point_segment_distance(obstacle_center, capsule.start, capsule.end) - capsule.radius - radius
        for capsule in env._capsules()
    ]
    return float(min(clearances))


def _workspace_ok(env: UR5DynamicObstacleEnv, tolerance_m: float) -> tuple[bool, float]:
    position, _ = env._end_effector_state()
    violations = []
    for axis, name in enumerate(("x", "y", "z")):
        low, high = (float(value) for value in env.workspace[name])
        violations.extend((low - float(position[axis]), float(position[axis]) - high))
    maximum = float(max(max(violations), 0.0))
    return maximum <= tolerance_m, maximum


def _advance_obstacle(
    center: np.ndarray,
    velocity: np.ndarray,
    bounds: dict[str, Sequence[float]],
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    next_center = center + velocity * dt_s
    next_velocity = velocity.copy()
    for axis, name in enumerate(("x", "y", "z")):
        low, high = (float(value) for value in bounds[name])
        if next_center[axis] < low or next_center[axis] > high:
            next_velocity[axis] *= -1.0
            next_center[axis] = np.clip(next_center[axis], low, high)
    return next_center, next_velocity


def _evaluate_candidate(
    env: UR5DynamicObstacleEnv,
    path: np.ndarray,
    *,
    dynamic: bool,
    initial_obstacle_center: np.ndarray,
    initial_obstacle_velocity: np.ndarray,
    clearance_m: float,
    path_tolerance_rad: float,
    workspace_tolerance_m: float,
    clearance_tolerance_m: float,
    self_collision_distance_m: float,
) -> tuple[bool, dict[str, Any]]:
    q_lower, q_upper = _joint_limits(env)
    limits_ok, limit_detail = _path_limits_ok(
        path,
        q_lower,
        q_upper,
        np.full(env.joint_count, env.action_scale, dtype=np.float64),
        np.full(env.joint_count, float(env.safety_filter_cfg.get("joint_acceleration_limit_radps2", 4.0)), dtype=np.float64),
        env.control_dt,
        path_tolerance_rad,
    )
    if not limits_ok:
        return False, {"status": "not_found", "reason": "joint_limit_violation", **limit_detail}
    center = initial_obstacle_center.copy()
    velocity = initial_obstacle_velocity.copy()
    min_clearance = float("inf")
    max_workspace_violation = 0.0
    for index, q in enumerate(path):
        _set_joint_state(env, q)
        if _self_collision(env, self_collision_distance_m):
            return False, {"status": "not_found", "reason": "self_collision", **limit_detail}
        workspace_ok, workspace_violation = _workspace_ok(env, workspace_tolerance_m)
        max_workspace_violation = max(max_workspace_violation, workspace_violation)
        if not workspace_ok:
            return False, {
                "status": "not_found",
                "reason": "workspace_violation",
                "max_workspace_violation_m": max_workspace_violation,
                **limit_detail,
            }
        if dynamic:
            min_clearance = min(min_clearance, _capsule_clearance(env, center))
            if min_clearance < clearance_m - clearance_tolerance_m:
                return False, {
                    "status": "not_found",
                    "reason": "candidate_capsule_clearance_violation",
                    "min_capsule_clearance_m": min_clearance,
                    "required_clearance_m": clearance_m,
                    **limit_detail,
                }
            if index + 1 < len(path):
                center, velocity = _advance_obstacle(center, velocity, env.obstacle_cfg["bounds"], env.control_dt)
    _set_joint_state(env, path[0])
    return True, {
        "status": "candidate_path_found",
        "reason": "candidate_path_found",
        "duration_s": (len(path) - 1) * env.control_dt,
        "min_capsule_clearance_m": None if not dynamic else min_clearance,
        "required_clearance_m": None if not dynamic else clearance_m,
        "max_workspace_violation_m": max_workspace_violation,
        **limit_detail,
    }


def precheck_episode(env: UR5DynamicObstacleEnv, seed: int, precheck: dict[str, Any]) -> dict[str, Any]:
    _, info = env.reset(seed=seed)
    q_start, _ = env._joint_state()
    goal = np.asarray(env.goal, dtype=np.float64)
    ik_candidates, ik_detail = _ik_candidates(
        env,
        goal,
        np.asarray(q_start, dtype=np.float64),
        int(precheck["ik_attempts"]),
        float(precheck["ik_position_tolerance_m"]),
        float(precheck["self_collision_distance_m"]),
        seed,
    )
    q_lower, q_upper = _joint_limits(env)
    velocity_limits = np.full(env.joint_count, env.action_scale, dtype=np.float64)
    acceleration_limits = np.full(
        env.joint_count,
        float(env.safety_filter_cfg.get("joint_acceleration_limit_radps2", 4.0)),
        dtype=np.float64,
    )
    no_obstacle_candidates = []
    dynamic_candidates = []
    no_obstacle_detail = {"status": "not_checked_due_to_ik", "reason": "no_ik_candidate"}
    dynamic_detail = {"status": "not_checked_due_to_ik", "reason": "no_ik_candidate"}
    required_clearance = float(env.risk_config.d_safe)
    required_clearance += float(env.safety_filter_cfg.get("geometry_margin_m", 0.03))
    required_clearance += float(env.safety_filter_cfg.get("tracking_error_bound_m", 0.01))
    required_clearance += float(np.linalg.norm(env.obstacle_velocity)) * float(
        env.safety_filter_cfg.get("control_delay_s", 0.05)
    )
    if ik_candidates:
        no_obstacle_detail = {"status": "not_found", "reason": "all_candidate_paths_rejected"}
        dynamic_detail = {"status": "not_found", "reason": "all_candidate_paths_rejected"}
        for q_goal in ik_candidates:
            path, path_detail = _smooth_path(
                np.asarray(q_start, dtype=np.float64),
                q_goal,
                velocity_limits,
                acceleration_limits,
                float(precheck["path_dt_s"]),
                float(precheck["path_max_duration_s"]),
            )
            if path is None:
                no_obstacle_detail = path_detail
                dynamic_detail = path_detail
                continue
            no_ok, candidate_no_detail = _evaluate_candidate(
                env,
                path,
                dynamic=False,
                initial_obstacle_center=np.asarray(env.obstacle_center, dtype=np.float64),
                initial_obstacle_velocity=np.asarray(env.obstacle_velocity, dtype=np.float64),
                clearance_m=required_clearance,
                path_tolerance_rad=float(precheck["path_joint_tolerance_rad"]),
                workspace_tolerance_m=float(precheck["path_workspace_tolerance_m"]),
                clearance_tolerance_m=float(precheck["path_clearance_tolerance_m"]),
                self_collision_distance_m=float(precheck["self_collision_distance_m"]),
            )
            if no_ok:
                no_obstacle_candidates.append({"goal_q_rad": q_goal.tolist(), **candidate_no_detail})
                if no_obstacle_detail.get("status") != "candidate_path_found":
                    no_obstacle_detail = {"status": "candidate_path_found", **candidate_no_detail}
                dynamic_ok, candidate_dynamic_detail = _evaluate_candidate(
                    env,
                    path,
                    dynamic=True,
                    initial_obstacle_center=np.asarray(env.obstacle_center, dtype=np.float64),
                    initial_obstacle_velocity=np.asarray(env.obstacle_velocity, dtype=np.float64),
                    clearance_m=required_clearance,
                    path_tolerance_rad=float(precheck["path_joint_tolerance_rad"]),
                    workspace_tolerance_m=float(precheck["path_workspace_tolerance_m"]),
                    clearance_tolerance_m=float(precheck["path_clearance_tolerance_m"]),
                    self_collision_distance_m=float(precheck["self_collision_distance_m"]),
                )
                if dynamic_ok:
                    dynamic_candidates.append({"goal_q_rad": q_goal.tolist(), **candidate_dynamic_detail})
                    if dynamic_detail.get("status") != "candidate_path_found":
                        dynamic_detail = {"status": "candidate_path_found", **candidate_dynamic_detail}
                elif dynamic_detail.get("status") != "candidate_path_found":
                    dynamic_detail = candidate_dynamic_detail
            elif no_obstacle_detail.get("status") != "candidate_path_found":
                no_obstacle_detail = candidate_no_detail
    _set_joint_state(env, np.asarray(q_start, dtype=np.float64))
    return {
        "seed": int(seed),
        "goal_m": goal.tolist(),
        "initial_joint_positions_rad": np.asarray(q_start, dtype=np.float64).tolist(),
        "obstacle_position_m": np.asarray(env.obstacle_center, dtype=np.float64).tolist(),
        "obstacle_velocity_mps": np.asarray(env.obstacle_velocity, dtype=np.float64).tolist(),
        "obstacle_speed_mps": float(np.linalg.norm(env.obstacle_velocity)),
        "ik": ik_detail,
        "no_obstacle_task": {
            **no_obstacle_detail,
            "candidate_count": len(no_obstacle_candidates),
            "candidate_paths": no_obstacle_candidates[:1],
        },
        "dynamic_obstacle_path": {
            **dynamic_detail,
            "candidate_count": len(dynamic_candidates),
            "candidate_paths": dynamic_candidates[:1],
            "required_clearance_m": required_clearance,
        },
        "precheck_interpretation": {
            "ik_reachable": bool(ik_candidates),
            "no_obstacle_candidate_found": bool(no_obstacle_candidates),
            "dynamic_obstacle_candidate_found": bool(dynamic_candidates),
            "candidate_search_is_not_completeness_proof": True,
        },
    }


def _shard(seeds: Sequence[int], index: int, count: int) -> list[int]:
    if count <= 0 or index < 0 or index >= count:
        raise ValueError("invalid shard index/count")
    return [int(seed) for seed in seeds[index::count]]


def audit_task_feasibility(config_path: str | Path, seed_manifest_path: str | Path, shard_index: int, shard_count: int) -> dict[str, Any]:
    config = load_config(config_path)
    precheck = _validate_config(config)
    seeds = _load_seed_manifest(seed_manifest_path)
    executed = _shard(seeds, shard_index, shard_count)
    env = UR5DynamicObstacleEnv(config, method=str(config["eval"]["method"]))
    try:
        rows = [precheck_episode(env, seed, precheck) for seed in executed]
    finally:
        env.close()
    status_fields = (
        ("ik_status", "ik"),
        ("no_obstacle_task_status", "no_obstacle_task"),
        ("dynamic_obstacle_path_status", "dynamic_obstacle_path"),
    )
    counts = {
        field: dict(Counter(str(row[section]["status"]) for row in rows))
        for field, section in status_fields
    }
    return {
        "protocol": PROTOCOL,
        "config": str(config_path),
        "seed_manifest": str(seed_manifest_path),
        "seed_count": len(seeds),
        "executed_seeds": executed,
        "shard": {"index": shard_index, "count": shard_count},
        "actor_loaded": False,
        "training_performed": False,
        "runtime_control_modified": False,
        "candidate_search_is_not_completeness_proof": True,
        "status_counts": counts,
        "episodes_detail": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit_task_feasibility(args.config, args.seed_manifest, args.shard_index, args.shard_count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(result["episodes_detail"]), "status_counts": result["status_counts"], "output": str(output)}))


if __name__ == "__main__":
    main()
