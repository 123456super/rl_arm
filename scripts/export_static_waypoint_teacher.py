from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export static-obstacle waypoint teacher data from feasibility labels.")
    parser.add_argument("--feasibility", required=True, help="Static feasibility JSON with candidate paths.")
    parser.add_argument("--output-json", required=True, help="Summary JSON output path.")
    parser.add_argument("--output-jsonl", required=True, help="Waypoint JSONL output path.")
    parser.add_argument("--dt", type=float, default=0.05, help="Waypoint spacing in seconds.")
    parser.add_argument(
        "--candidate-kind",
        default="dynamic_obstacle_path",
        choices=["dynamic_obstacle_path", "no_obstacle_task"],
        help="Which candidate path field to export.",
    )
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _cubic_waypoint_path(q_start: np.ndarray, q_goal: np.ndarray, dt_s: float, duration_s: float) -> dict[str, np.ndarray]:
    if dt_s <= 0.0:
        raise ValueError("dt must be positive")
    steps = max(1, int(math.ceil(duration_s / dt_s)))
    if steps <= 0:
        steps = 1
    tau = np.linspace(0.0, 1.0, steps + 1)
    smooth = 3.0 * tau**2 - 2.0 * tau**3
    delta = q_goal - q_start
    q = q_start[None, :] + smooth[:, None] * delta[None, :]
    d_smooth = 6.0 * tau - 6.0 * tau**2
    dd_smooth = 6.0 - 12.0 * tau
    duration = steps * dt_s
    qdot = (d_smooth[:, None] / duration) * delta[None, :]
    qddot = (dd_smooth[:, None] / (duration**2)) * delta[None, :]
    return {"q": q, "qdot": qdot, "qddot": qddot, "tau": tau}


def export_teacher(
    feasibility_path: str | Path,
    *,
    output_json: str | Path,
    output_jsonl: str | Path,
    dt_s: float = 0.05,
    candidate_kind: str = "dynamic_obstacle_path",
) -> dict[str, Any]:
    payload = _load_json(feasibility_path)
    episodes = payload.get("episodes_detail", [])
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("feasibility JSON has no episodes_detail")
    rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    status_counts = Counter()
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("feasibility episode must be an object")
        candidate = episode.get(candidate_kind)
        if not isinstance(candidate, dict):
            raise ValueError(f"missing {candidate_kind} entry in feasibility episode")
        status = str(candidate.get("status", ""))
        status_counts[status] += 1
        if status != "candidate_path_found":
            continue
        candidates = candidate.get("candidate_paths", [])
        if not candidates:
            continue
        path = candidates[0]
        goal_q_rad = np.asarray(path["goal_q_rad"], dtype=np.float64)
        q_start = np.asarray(episode["initial_joint_positions_rad"], dtype=np.float64)
        duration_s = float(path["duration_s"])
        required_clearance_m = path.get("required_clearance_m", candidate.get("required_clearance_m"))
        min_capsule_clearance_m = path.get("min_capsule_clearance_m", candidate.get("min_capsule_clearance_m"))
        max_workspace_violation_m = path.get(
            "max_workspace_violation_m",
            candidate.get("max_workspace_violation_m"),
        )
        waypoints = _cubic_waypoint_path(q_start, goal_q_rad, dt_s, duration_s)
        path_rows = []
        for index, (q, qdot, qddot, tau) in enumerate(
            zip(waypoints["q"], waypoints["qdot"], waypoints["qddot"], waypoints["tau"], strict=True)
        ):
            row = {
                "seed": int(episode["seed"]),
                "candidate_kind": candidate_kind,
                "path_status": status,
                "path_reason": str(candidate.get("reason", "")),
                "waypoint_index": index,
                "waypoint_count": len(waypoints["q"]),
                "time_s": float(index * dt_s),
                "tau": float(tau),
                "goal_q_rad": goal_q_rad.tolist(),
                "q_start_rad": q_start.tolist(),
                "q_rad": np.asarray(q, dtype=np.float64).tolist(),
                "qdot_radps": np.asarray(qdot, dtype=np.float64).tolist(),
                "qddot_radps2": np.asarray(qddot, dtype=np.float64).tolist(),
                "goal_m": episode.get("goal_m", []),
                "obstacle_position_m": episode.get("obstacle_position_m", []),
                "obstacle_velocity_mps": episode.get("obstacle_velocity_mps", []),
                "obstacle_speed_mps": episode.get("obstacle_speed_mps"),
                "path_duration_s": duration_s,
                "required_clearance_m": required_clearance_m,
                "min_capsule_clearance_m": min_capsule_clearance_m,
                "max_workspace_violation_m": max_workspace_violation_m,
            }
            rows.append(row)
            path_rows.append(row)
        episode_summaries.append(
            {
                "seed": int(episode["seed"]),
                "candidate_kind": candidate_kind,
                "path_status": status,
                "path_reason": str(candidate.get("reason", "")),
                "waypoint_count": len(path_rows),
                "path_duration_s": duration_s,
                "goal_q_rad": goal_q_rad.tolist(),
                "q_start_rad": q_start.tolist(),
                "goal_m": episode.get("goal_m", []),
                "obstacle_position_m": episode.get("obstacle_position_m", []),
                "obstacle_velocity_mps": episode.get("obstacle_velocity_mps", []),
                "obstacle_speed_mps": episode.get("obstacle_speed_mps"),
                "required_clearance_m": required_clearance_m,
                "min_capsule_clearance_m": min_capsule_clearance_m,
                "max_workspace_violation_m": max_workspace_violation_m,
            }
        )

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "feasibility": str(Path(feasibility_path)),
        "candidate_kind": candidate_kind,
        "dt_s": dt_s,
        "episodes": len(episodes),
        "candidate_episode_count": len(episode_summaries),
        "waypoint_count": len(rows),
        "path_status_counts": dict(sorted(status_counts.items())),
        "episodes_detail": episode_summaries,
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")

    return summary


def main() -> None:
    args = parse_args()
    summary = export_teacher(
        args.feasibility,
        output_json=args.output_json,
        output_jsonl=args.output_jsonl,
        dt_s=float(args.dt),
        candidate_kind=str(args.candidate_kind),
    )
    print(
        json.dumps(
            {
                "output_json": args.output_json,
                "output_jsonl": args.output_jsonl,
                "candidate_episode_count": summary["candidate_episode_count"],
                "waypoint_count": summary["waypoint_count"],
                "path_status_counts": summary["path_status_counts"],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
