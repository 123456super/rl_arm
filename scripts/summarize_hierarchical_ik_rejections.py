"""Aggregate per-episode IK rejection diagnostics from hierarchical evaluation CSVs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, dest="inputs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for input_path in args.inputs:
        with Path(input_path).open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))

    totals: Counter[str] = Counter()
    by_failure_mode: dict[str, Counter[str]] = {}
    fallback_episodes = 0
    raw_goal_reachable_episodes = 0
    workspace_self_collision_free_goal_reachable_episodes = 0
    accepted_obstacle_valid_goal_episodes = 0
    obstacle_blocked_goal_reachable_episodes = 0
    raw_goal_unreachable_episodes = 0
    min_goal_errors: list[float] = []
    for row in rows:
        raw = row.get("hierarchical_ik_rejection_counts", "{}")
        parsed = ast.literal_eval(raw) if raw else {}
        if not isinstance(parsed, dict):
            raise ValueError(f"invalid rejection-count field: {raw!r}")
        counts = {str(key): int(value) for key, value in parsed.items()}
        totals.update(counts)
        mode = row.get("hierarchical_failure_mode", "UNKNOWN")
        by_failure_mode.setdefault(mode, Counter()).update(counts)
        if int(float(row.get("hierarchical_ik_fallback_attempted", 0))) != 0:
            fallback_episodes += 1
        raw_goal_reachable = int(float(row.get("hierarchical_ik_goal_reachable_count", 0))) > 0
        workspace_self_collision_free_goal_reachable = (
            int(float(row.get("hierarchical_ik_obstacle_free_goal_reachable_count", 0))) > 0
        )
        accepted_obstacle_valid_goal = int(float(row.get("hierarchical_ik_candidate_count", 0))) > 0
        if raw_goal_reachable:
            raw_goal_reachable_episodes += 1
        else:
            raw_goal_unreachable_episodes += 1
        if workspace_self_collision_free_goal_reachable:
            workspace_self_collision_free_goal_reachable_episodes += 1
        if accepted_obstacle_valid_goal:
            accepted_obstacle_valid_goal_episodes += 1
        if raw_goal_reachable and not accepted_obstacle_valid_goal:
            obstacle_blocked_goal_reachable_episodes += 1
        try:
            min_goal_error = float(row.get("hierarchical_ik_min_goal_error_m", "nan"))
            if min_goal_error == min_goal_error:
                min_goal_errors.append(min_goal_error)
        except ValueError:
            pass

    result = {
        "episodes": len(rows),
        "fallback_attempted_episodes": fallback_episodes,
        "raw_goal_reachable_episodes": raw_goal_reachable_episodes,
        "raw_goal_unreachable_episodes": raw_goal_unreachable_episodes,
        "workspace_self_collision_free_goal_reachable_episodes": workspace_self_collision_free_goal_reachable_episodes,
        "accepted_obstacle_valid_goal_episodes": accepted_obstacle_valid_goal_episodes,
        "obstacle_blocked_goal_reachable_episodes": obstacle_blocked_goal_reachable_episodes,
        "goal_reachable_but_obstacle_blocked_episodes": obstacle_blocked_goal_reachable_episodes,
        "mean_min_goal_error_m": sum(min_goal_errors) / len(min_goal_errors) if min_goal_errors else None,
        "total_rejections": dict(sorted(totals.items())),
        "rejections_by_failure_mode": {
            mode: dict(sorted(counter.items())) for mode, counter in sorted(by_failure_mode.items())
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
