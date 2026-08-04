from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize strict-QP infeasibility attribution traces.")
    parser.add_argument("--trace-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def categories(value: str) -> list[str]:
    return [category for category in value.split("|") if category]


def summarize_trace_dirs(trace_dirs: list[Path]) -> dict[str, Any]:
    step_categories: Counter[str] = Counter()
    diagnostic_statuses: Counter[str] = Counter()
    combination_counts: Counter[str] = Counter()
    episodes_with_infeasible = 0
    infeasible_steps = 0
    physical_contact_episodes = 0
    capsule_overlap_episodes = 0

    for trace_dir in trace_dirs:
        for trace_path in sorted(trace_dir.glob("episode_*.csv")):
            with open(trace_path, newline="", encoding="utf-8") as file:
                trace = list(csv.DictReader(file))
            infeasible = [
                row for row in trace if row.get("safety_filter_status") == "safe_stop_infeasible"
            ]
            if not infeasible:
                continue
            episodes_with_infeasible += 1
            infeasible_steps += len(infeasible)
            physical_contact_episodes += int(
                any(row.get("collision_pybullet_contact") == "1" for row in trace)
            )
            capsule_overlap_episodes += int(
                any(row.get("collision_capsule_overlap") == "1" for row in trace)
            )
            for row in infeasible:
                row_categories = sorted(categories(row.get("safety_filter_infeasible_constraint_categories", "")))
                if row_categories:
                    step_categories.update(row_categories)
                    combination_counts["|".join(row_categories)] += 1
                diagnostic_statuses[row.get("safety_filter_infeasibility_diagnostic_status", "")] += 1

    return {
        "trace_dirs": [str(path) for path in trace_dirs],
        "episodes_with_infeasible_stop": episodes_with_infeasible,
        "infeasible_steps": infeasible_steps,
        "physical_contact_episodes_with_infeasible_stop": physical_contact_episodes,
        "capsule_overlap_episodes_with_infeasible_stop": capsule_overlap_episodes,
        "single_category_relaxation_step_counts": dict(sorted(step_categories.items())),
        "diagnostic_status_counts": dict(sorted(diagnostic_statuses.items())),
        "category_combination_step_counts": dict(sorted(combination_counts.items())),
    }


def main() -> None:
    args = parse_args()
    trace_dirs = [Path(path) for path in args.trace_dir]
    result = summarize_trace_dirs(trace_dirs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
