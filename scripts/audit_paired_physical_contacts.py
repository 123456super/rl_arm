from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CONTEXT_STEPS = 10
FIELDS = (
    "step",
    "time",
    "goal_error_norm",
    "d_min",
    "closest_link",
    "safety_violation",
    "collision_capsule_overlap",
    "collision_pybullet_contact",
    "termination_collision",
    "collision_contact_link_names",
    "collision_min_contact_distance",
    "qdot_norm",
    "qdot_requested_norm",
    "safety_filter_status",
    "safety_filter_reason",
    "safety_filter_intervention_norm",
    "safety_filter_safe_stop",
    "safety_filter_active_constraint_categories",
    "safety_filter_infeasible_constraint_categories",
    "safety_filter_qp_solver_status",
    "predictive_h_min_m",
    "predictive_h_by_link_m",
    "safety_jacobian_command_by_link_mps",
    "safety_drift_by_link_mps",
    "safety_constraint_residual_by_link_mps",
    "filter_obstacle_position_m",
    "filter_obstacle_velocity_mps",
    "safety_filter_solve_time_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit physical-contact episodes shared by baseline and an ablation."
    )
    parser.add_argument("--baseline-trace-dir", action="append", required=True)
    parser.add_argument("--ablation-trace-dir", action="append", required=True)
    parser.add_argument("--context-steps", type=int, default=CONTEXT_STEPS)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_traces(trace_dir: Path) -> dict[str, list[dict[str, str]]]:
    traces: dict[str, list[dict[str, str]]] = {}
    for path in sorted(trace_dir.glob("episode_*.csv")):
        with open(path, newline="", encoding="utf-8") as file:
            traces[path.stem] = list(csv.DictReader(file))
    return traces


def contact_step(rows: list[dict[str, str]]) -> int | None:
    for index, row in enumerate(rows):
        if row.get("collision_pybullet_contact") == "1":
            return index
    return None


def selected_rows(rows: list[dict[str, str]], collision_index: int, context_steps: int) -> list[dict[str, str]]:
    start = max(0, collision_index - context_steps)
    return [{field: row.get(field, "") for field in FIELDS} for row in rows[start : collision_index + 1]]


def paired_contact_audit(
    baseline_dirs: list[Path], ablation_dirs: list[Path], context_steps: int
) -> dict[str, Any]:
    if len(baseline_dirs) != len(ablation_dirs):
        raise ValueError("Baseline and ablation trace directory counts must match")
    if context_steps < 0:
        raise ValueError("--context-steps must be non-negative")

    cases: list[dict[str, Any]] = []
    baseline_contact_episodes = 0
    ablation_contact_episodes = 0
    for baseline_dir, ablation_dir in zip(baseline_dirs, ablation_dirs, strict=True):
        baseline_traces = load_traces(baseline_dir)
        ablation_traces = load_traces(ablation_dir)
        if baseline_traces.keys() != ablation_traces.keys():
            raise ValueError(f"Trace episode sets differ: {baseline_dir} vs {ablation_dir}")
        for episode in sorted(baseline_traces):
            baseline_rows = baseline_traces[episode]
            ablation_rows = ablation_traces[episode]
            baseline_step = contact_step(baseline_rows)
            ablation_step = contact_step(ablation_rows)
            baseline_contact_episodes += int(baseline_step is not None)
            ablation_contact_episodes += int(ablation_step is not None)
            if baseline_step is None or ablation_step is None:
                continue
            cases.append(
                {
                    "baseline_trace_dir": str(baseline_dir),
                    "ablation_trace_dir": str(ablation_dir),
                    "episode": episode,
                    "baseline_contact_step": baseline_step,
                    "ablation_contact_step": ablation_step,
                    "baseline_context": selected_rows(baseline_rows, baseline_step, context_steps),
                    "ablation_context": selected_rows(ablation_rows, ablation_step, context_steps),
                }
            )
    return {
        "context_steps": context_steps,
        "baseline_physical_contact_episodes": baseline_contact_episodes,
        "ablation_physical_contact_episodes": ablation_contact_episodes,
        "shared_physical_contact_episodes": len(cases),
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    result = paired_contact_audit(
        [Path(path) for path in args.baseline_trace_dir],
        [Path(path) for path in args.ablation_trace_dir],
        args.context_steps,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(
        f"baseline contacts={result['baseline_physical_contact_episodes']}; "
        f"ablation contacts={result['ablation_physical_contact_episodes']}; "
        f"shared={result['shared_physical_contact_episodes']}"
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
