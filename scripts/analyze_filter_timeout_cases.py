"""Report recovery/filter behavior for selected hard-case seeds."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


def finite_mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = np.asarray([float(row.get(key, "nan")) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else None


def finite_values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = np.asarray([float(row.get(key, "nan")) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def finite_mean_alias(rows: list[dict[str, str]], *keys: str) -> float | None:
    for key in keys:
        values = finite_values(rows, key)
        if values.size:
            return float(np.mean(values))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[9093, 9143])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.evaluation).open(newline="", encoding="utf-8") as file:
        episodes = list(csv.DictReader(file))
    selected = [row for row in episodes if int(row["seed"]) in set(args.seeds)]
    reports = []
    trace_root = Path(args.trace_dir)
    for episode in selected:
        index = int(episode["episode"])
        matches = sorted(
            set(
                glob.glob(str(trace_root / f"episode_{index:04d}.csv"))
                + glob.glob(str(trace_root / "**" / f"episode_{index:04d}.csv"), recursive=True)
            ),
            key=lambda path: Path(path).stat().st_mtime,
            reverse=True,
        )
        if not matches:
            reports.append({"seed": int(episode["seed"]), "episode": index, "trace_available": False})
            continue
        with open(matches[0], newline="", encoding="utf-8") as file:
            trace = list(csv.DictReader(file))
        hold = [row for row in trace if row.get("hierarchical_state") == "AVOID_HOLD"]
        servo = [row for row in trace if row.get("hierarchical_state") == "SERVO"]
        recovery = [row for row in trace if int(float(row.get("recovery_active", 0))) == 1]
        filter_rows = [row for row in trace if row.get("safety_filter_status")]
        audit_rows = [
            row
            for row in filter_rows
            if row.get("safety_filter_goal_velocity_audit_status")
        ]
        max_feasible = finite_values(audit_rows, "safety_filter_max_feasible_goal_velocity_mps")
        projected = finite_values(audit_rows, "safety_filter_projected_goal_velocity_mps")
        gap = finite_values(audit_rows, "safety_filter_goal_velocity_feasibility_gap_mps")
        audit_status_counts = {
            status: sum(
                row.get("safety_filter_goal_velocity_audit_status") == status
                for row in audit_rows
            )
            for status in sorted(
                {
                    row.get("safety_filter_goal_velocity_audit_status", "")
                    for row in audit_rows
                }
            )
        }
        reports.append(
            {
                "seed": int(episode["seed"]),
                "episode": index,
                "trace_available": True,
                "failure_mode": episode.get("hierarchical_failure_mode", ""),
                "success": int(float(episode["success"])),
                "final_position_error_m": float(episode["final_position_error"]),
                "hold_steps": len(hold),
                "servo_steps": len(servo),
                "recovery_steps": len(recovery),
                "recovery_rate": float(len(recovery) / len(trace)) if trace else None,
                "mean_predictive_h_m": finite_mean_alias(
                    trace, "predictive_h_min_m", "min_predictive_h_m"
                ),
                "mean_filter_intervention_norm": finite_mean(filter_rows, "safety_filter_intervention_norm"),
                "mean_executed_toward_goal_mps": finite_mean(servo, "task_space_executed_toward_goal_mps"),
                "mean_filter_velocity_loss_mps": finite_mean(servo, "task_space_filter_velocity_loss_mps"),
                "goal_velocity_audit_status_counts": audit_status_counts,
                "goal_velocity_audit_steps": len(audit_rows),
                "mean_max_feasible_goal_velocity_mps": (
                    float(np.mean(max_feasible)) if max_feasible.size else None
                ),
                "min_max_feasible_goal_velocity_mps": (
                    float(np.min(max_feasible)) if max_feasible.size else None
                ),
                "mean_projected_goal_velocity_mps": (
                    float(np.mean(projected)) if projected.size else None
                ),
                "mean_goal_velocity_feasibility_gap_mps": (
                    float(np.mean(gap)) if gap.size else None
                ),
                "mean_command_alignment_previous": finite_mean(
                    filter_rows, "safety_filter_command_alignment_previous"
                ),
                "command_sign_change_steps": sum(
                    int(float(row.get("safety_filter_command_sign_change", 0))) for row in filter_rows
                ),
                "positive_max_feasible_goal_velocity_rate": (
                    float(np.mean(max_feasible > 1.0e-3)) if max_feasible.size else None
                ),
                "safe_stop_steps": sum(int(float(row.get("safety_filter_safe_stop", 0))) for row in filter_rows),
                "filter_status_counts": {
                    status: sum(row.get("safety_filter_status") == status for row in filter_rows)
                    for status in sorted({row.get("safety_filter_status", "") for row in filter_rows})
                },
                "trace_path": matches[0],
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"seeds": args.seeds, "episodes": reports}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"seeds": args.seeds, "episodes": reports}, indent=2))


if __name__ == "__main__":
    main()
