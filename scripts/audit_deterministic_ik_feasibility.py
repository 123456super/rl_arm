"""Classify nominal reset targets with conservative IK feasibility certificates.

The audit is deliberately narrower than episode success.  A constructive
certificate is a concrete terminal joint configuration accepted by the same
goal-error, workspace, collision, and clearance checks used by the planner.
An infeasibility certificate is emitted only for a sound geometric bound.  A
failed finite IK search is therefore reported as ``unknown``.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config

try:
    from scripts.seed_manifest import load_seed_manifest
except ModuleNotFoundError:
    from seed_manifest import load_seed_manifest


CLASSIFICATIONS = ("certified_feasible", "certified_infeasible", "unknown")
_EPS = 1.0e-9


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--manifest-shard-index", type=int, required=True)
    parser.add_argument("--manifest-shard-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _origin_xyz(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    if origin is None:
        return np.zeros(3, dtype=np.float64)
    values = [float(value) for value in origin.attrib.get("xyz", "0 0 0").split()]
    if len(values) != 3:
        raise ValueError(f"URDF origin must have three coordinates: {joint.attrib.get('name')!r}")
    return np.asarray(values, dtype=np.float64)


def urdf_absolute_reach_bound(urdf_path: str | Path, *, base_link: str, tool_link: str) -> float:
    """Return a triangle-inequality upper bound for base-to-tool reach.

    Each fixed/revolute joint origin on the serial chain contributes its
    translation norm.  Rotations and joint limits can only reduce this bound,
    so a target farther than this bound plus the allowed position error is
    formally unreachable.
    """

    root = ET.parse(urdf_path).getroot()
    parent_by_child: dict[str, tuple[str, np.ndarray]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        child_name = child.attrib.get("link")
        parent_name = parent.attrib.get("link")
        if not child_name or not parent_name:
            continue
        if child_name in parent_by_child:
            raise ValueError(f"URDF has multiple parents for link {child_name!r}")
        parent_by_child[child_name] = (parent_name, _origin_xyz(joint))

    total = 0.0
    current = tool_link
    visited: set[str] = set()
    while current != base_link:
        if current in visited:
            raise ValueError(f"URDF link cycle while tracing {tool_link!r}")
        visited.add(current)
        try:
            parent, origin = parent_by_child[current]
        except KeyError as error:
            raise ValueError(
                f"Cannot trace URDF chain from {tool_link!r} to {base_link!r}; stopped at {current!r}"
            ) from error
        total += float(np.linalg.norm(origin))
        current = parent
    return total


def _tool_capsule_radius(config: dict[str, Any]) -> float | None:
    tool_link = str(config["robot"]["tool_link_name"])
    radii = [
        float(spec["radius"])
        for spec in config["robot"].get("capsules", [])
        if str(spec.get("child_link_name")) == tool_link
    ]
    return max(radii) if radii else None


def classify_reset(
    *,
    config: dict[str, Any],
    goal: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_enabled: bool,
    candidate_count: int,
    reach_bound_m: float,
    tool_capsule_radius_m: float | None,
) -> tuple[str, list[str]]:
    """Classify one reset without treating finite-search failure as proof."""

    if candidate_count > 0:
        return "certified_feasible", ["constructive_strict_valid_terminal"]

    env_cfg = config["env"]
    planner_cfg = env_cfg["hierarchical_control"]["planner"]
    tolerance_m = min(
        float(env_cfg["success_tolerance"]),
        float(planner_cfg.get("ik_goal_tolerance_m", env_cfg["success_tolerance"])),
    )
    base = np.asarray(config["robot"]["base_position"], dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    reasons: list[str] = []

    goal_distance_m = float(np.linalg.norm(goal - base))
    if goal_distance_m > reach_bound_m + tolerance_m + _EPS:
        reasons.append("outside_urdf_triangle_inequality_reach_bound")

    for axis, name in enumerate(("x", "y", "z")):
        low, high = (float(value) for value in env_cfg["workspace"][name])
        if goal[axis] + tolerance_m < low - _EPS or goal[axis] - tolerance_m > high + _EPS:
            reasons.append(f"success_ball_outside_workspace_{name}")

    # The final capsule contains tool0 as an endpoint.  If the whole allowed
    # goal ball lies inside its strict forbidden shell, no terminal IK can be
    # valid regardless of the other links' configurations.
    if obstacle_enabled and tool_capsule_radius_m is not None:
        obstacle_radius_m = float(env_cfg["obstacle"]["radius"])
        d_safe_m = float(config["risk"]["d_safe"])
        goal_obstacle_distance_m = float(np.linalg.norm(goal - obstacle_center))
        forbidden_radius_m = obstacle_radius_m + tool_capsule_radius_m + d_safe_m
        if goal_obstacle_distance_m + tolerance_m <= forbidden_radius_m + _EPS:
            reasons.append("success_ball_inside_tool_obstacle_forbidden_shell")

    return ("certified_infeasible", reasons) if reasons else ("unknown", ["finite_search_no_certificate"])


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _row_from_reset(
    *,
    episode: int,
    seed: int,
    info: dict[str, Any],
    config: dict[str, Any],
    reach_bound_m: float,
    tool_capsule_radius_m: float | None,
) -> dict[str, Any]:
    goal = np.asarray(info["goal"], dtype=np.float64)
    obstacle_center = np.asarray(info["obstacle_center"], dtype=np.float64)
    candidate_count = int(info.get("hierarchical_ik_candidate_count", 0))
    classification, certificate_reasons = classify_reset(
        config=config,
        goal=goal,
        obstacle_center=obstacle_center,
        obstacle_enabled=bool(info.get("obstacle_enabled", False)),
        candidate_count=candidate_count,
        reach_bound_m=reach_bound_m,
        tool_capsule_radius_m=tool_capsule_radius_m,
    )
    candidate_qs = [np.asarray(q, dtype=np.float64).tolist() for q in info.get("hierarchical_ik_candidate_qs", ())]
    rejection_counts = dict(info.get("hierarchical_ik_rejection_counts", {}))
    return {
        "episode": episode,
        "seed": seed,
        "classification": classification,
        "certificate_reasons": _json_value(certificate_reasons),
        "goal_m": _json_value(goal.tolist()),
        "obstacle_center_m": _json_value(obstacle_center.tolist()),
        "obstacle_enabled": int(bool(info.get("obstacle_enabled", False))),
        "goal_obstacle_distance_m": float(np.linalg.norm(goal - obstacle_center)),
        "urdf_reach_bound_m": reach_bound_m,
        "tool_capsule_radius_m": "" if tool_capsule_radius_m is None else tool_capsule_radius_m,
        "candidate_count": candidate_count,
        "plan_found": int(bool(info.get("hierarchical_plan_found", False))),
        "plan_reason": str(info.get("hierarchical_plan_reason", "")),
        "min_goal_error_m": float(info.get("hierarchical_ik_min_goal_error_m", float("nan"))),
        "goal_reachable_count": int(info.get("hierarchical_ik_goal_reachable_count", 0)),
        "obstacle_free_goal_reachable_count": int(
            info.get("hierarchical_ik_obstacle_free_goal_reachable_count", 0)
        ),
        "rejection_counts": _json_value(rejection_counts),
        "candidate_kinds": _json_value(list(info.get("hierarchical_ik_candidate_kinds", ()))),
        "candidate_errors_m": _json_value(
            [float(value) for value in info.get("hierarchical_ik_candidate_errors_m", ())]
        ),
        "candidate_clearances_m": _json_value(
            [float(value) for value in info.get("hierarchical_ik_candidate_clearances_m", ())]
        ),
        "candidate_qs": _json_value(candidate_qs),
        "ik_attempts_used": int(info.get("hierarchical_ik_attempts_used", 0)),
        "ik_fallback_attempted": int(bool(info.get("hierarchical_ik_fallback_attempted", False))),
        "obstacle_aware_attempted": int(bool(info.get("hierarchical_ik_obstacle_aware_attempted", False))),
        "goal_region_attempted": int(bool(info.get("hierarchical_ik_goal_region_attempted", False))),
        "target_shell_attempted": int(bool(info.get("hierarchical_ik_target_shell_attempted", False))),
    }


def audit_manifest_shard(
    config_path: str | Path,
    seed_manifest_path: str | Path,
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("manifest shard index/count are invalid")
    config = load_config(config_path)
    if not bool(config["env"].get("hierarchical_control", {}).get("enabled", False)):
        raise ValueError("deterministic IK audit requires hierarchical_control.enabled=true")
    planner_cfg = config["env"]["hierarchical_control"]["planner"]
    if not bool(planner_cfg.get("ik_feasibility_audit_only", False)):
        raise ValueError("audit config must set planner.ik_feasibility_audit_only=true")
    forbidden_branches = (
        "ik_goal_region_enabled",
        "ik_target_shell_enabled",
        "ik_obstacle_aware_nullspace_enabled",
        "ik_obstacle_aware_beam_enabled",
    )
    enabled_branches = [name for name in forbidden_branches if bool(planner_cfg.get(name, False))]
    if enabled_branches:
        raise ValueError(f"audit must use the deterministic base IK only; enabled branches: {enabled_branches}")

    seeds = load_seed_manifest(seed_manifest_path)
    config["seed"] = int(seeds[0]) if seeds else 0
    env = UR5DynamicObstacleEnv(config, method=str(config["eval"]["method"]))
    reach_bound_m = urdf_absolute_reach_bound(
        env.robot_urdf,
        base_link="base_link",
        tool_link=str(config["robot"]["tool_link_name"]),
    )
    tool_capsule_radius_m = _tool_capsule_radius(config)
    rows: list[dict[str, Any]] = []
    try:
        for episode, seed in enumerate(seeds):
            if episode % shard_count != shard_index:
                continue
            _, info = env.reset(seed=int(seed))
            rows.append(
                _row_from_reset(
                    episode=episode,
                    seed=int(seed),
                    info=info,
                    config=config,
                    reach_bound_m=reach_bound_m,
                    tool_capsule_radius_m=tool_capsule_radius_m,
                )
            )
    finally:
        env.close()
    return rows


FIELDNAMES = [
    "episode",
    "seed",
    "classification",
    "certificate_reasons",
    "goal_m",
    "obstacle_center_m",
    "obstacle_enabled",
    "goal_obstacle_distance_m",
    "urdf_reach_bound_m",
    "tool_capsule_radius_m",
    "candidate_count",
    "plan_found",
    "plan_reason",
    "min_goal_error_m",
    "goal_reachable_count",
    "obstacle_free_goal_reachable_count",
    "rejection_counts",
    "candidate_kinds",
    "candidate_errors_m",
    "candidate_clearances_m",
    "candidate_qs",
    "ik_attempts_used",
    "ik_fallback_attempted",
    "obstacle_aware_attempted",
    "goal_region_attempted",
    "target_shell_attempted",
]


def write_rows(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = audit_manifest_shard(
        args.config,
        args.seed_manifest,
        args.manifest_shard_index,
        args.manifest_shard_count,
    )
    write_rows(rows, args.output)
    counts = {name: sum(row["classification"] == name for row in rows) for name in CLASSIFICATIONS}
    print(json.dumps({"episodes": len(rows), "classification_counts": counts}, ensure_ascii=True))


if __name__ == "__main__":
    main()
