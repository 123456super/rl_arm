"""Summarize task-space terminal-servo diagnostics from evaluation traces."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np


def finite(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = finite(rows, key)
    return float(np.mean(values)) if len(values) else None


def rate(rows: list[dict[str, str]], predicate) -> float | None:
    if not rows:
        return None
    return float(np.mean([bool(predicate(row)) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.evaluation, newline="", encoding="utf-8") as file:
        episodes = list(csv.DictReader(file))
    report_rows = []
    trace_root = Path(args.trace_dir)
    for episode in episodes:
        index = int(episode["episode"])
        matches = sorted(glob.glob(str(trace_root / f"episode_{index:04d}.csv")))
        if not matches:
            matches = sorted(glob.glob(str(trace_root / "**" / f"episode_{index:04d}.csv"), recursive=True))
        if not matches:
            raise FileNotFoundError(f"missing trace for episode {index}: {trace_root}")
        with open(matches[0], newline="", encoding="utf-8") as file:
            trace = list(csv.DictReader(file))
        servo = [row for row in trace if row.get("hierarchical_state") == "SERVO"]
        toward = [row for row in servo if np.isfinite(float(row["task_space_executed_toward_goal_mps"]))]
        positive_progress = [row for row in toward if float(row["task_space_goal_error_delta_m"]) > 0.0]
        requested_positive = [row for row in toward if float(row["task_space_requested_toward_goal_mps"]) > 0.0]
        executed_nonpositive = [row for row in toward if float(row["task_space_executed_toward_goal_mps"]) <= 0.0]
        requested_positive_executed_nonpositive = [
            row for row in toward
            if float(row["task_space_requested_toward_goal_mps"]) > 0.0
            and float(row["task_space_executed_toward_goal_mps"]) <= 0.0
        ]
        nominal = [row for row in servo if np.isfinite(float(row.get("task_space_nominal_toward_goal_mps", "nan")))]
        nominal_positive = [row for row in nominal if float(row["task_space_nominal_toward_goal_mps"]) > 0.0]
        recovery_rows = [row for row in servo if int(float(row.get("task_space_recovery_active_for_command", 0))) == 1]
        report_rows.append(
            {
                "episode": index,
                "seed": int(episode["seed"]),
                "success": int(float(episode["success"])),
                "failure_mode": episode.get("hierarchical_failure_mode", ""),
                "servo_steps": len(servo),
                "servo_mean_requested_toward_goal_mps": mean(servo, "task_space_requested_toward_goal_mps"),
                "servo_mean_executed_toward_goal_mps": mean(servo, "task_space_executed_toward_goal_mps"),
                "servo_mean_filter_velocity_loss_mps": mean(servo, "task_space_filter_velocity_loss_mps"),
                "servo_positive_progress_rate": rate(toward, lambda row: float(row["task_space_goal_error_delta_m"]) > 0.0),
                "servo_requested_positive_rate": rate(toward, lambda row: float(row["task_space_requested_toward_goal_mps"]) > 0.0),
                "servo_executed_nonpositive_rate": rate(toward, lambda row: float(row["task_space_executed_toward_goal_mps"]) <= 0.0),
                "filter_reverses_positive_request_rate": rate(
                    toward,
                    lambda row: float(row["task_space_requested_toward_goal_mps"]) > 0.0
                    and float(row["task_space_executed_toward_goal_mps"]) <= 0.0,
                ),
                "nominal_toward_goal_rate": rate(nominal, lambda row: float(row["task_space_nominal_toward_goal_mps"]) > 0.0),
                "recovery_command_rate": float(len(recovery_rows) / len(servo)) if servo else None,
                "nominal_mean_toward_goal_mps": mean(nominal, "task_space_nominal_toward_goal_mps"),
                "servo_min_error_m": float(np.min(finite(servo, "task_space_goal_error_after_m"))) if finite(servo, "task_space_goal_error_after_m").size else None,
                "trace_path": matches[0],
                "_counts": {"toward": len(toward), "positive_progress": len(positive_progress), "requested_positive": len(requested_positive), "executed_nonpositive": len(executed_nonpositive), "reversed": len(requested_positive_executed_nonpositive)},
            }
        )

    servo_rows = [row for row in report_rows if row["servo_steps"] > 0]
    report = {
        "evaluation": args.evaluation,
        "trace_dir": args.trace_dir,
        "episodes": len(report_rows),
        "successes": sum(row["success"] for row in report_rows),
        "failure_modes": dict(Counter(row["failure_mode"] for row in report_rows)),
        "servo_episodes": len(servo_rows),
        "aggregate": {
            "mean_requested_toward_goal_mps": float(np.mean([row["servo_mean_requested_toward_goal_mps"] for row in servo_rows if row["servo_mean_requested_toward_goal_mps"] is not None])) if servo_rows else None,
            "mean_executed_toward_goal_mps": float(np.mean([row["servo_mean_executed_toward_goal_mps"] for row in servo_rows if row["servo_mean_executed_toward_goal_mps"] is not None])) if servo_rows else None,
            "mean_filter_velocity_loss_mps": float(np.mean([row["servo_mean_filter_velocity_loss_mps"] for row in servo_rows if row["servo_mean_filter_velocity_loss_mps"] is not None])) if servo_rows else None,
            "mean_positive_progress_rate": float(np.mean([row["servo_positive_progress_rate"] for row in servo_rows if row["servo_positive_progress_rate"] is not None])) if servo_rows else None,
            "mean_requested_positive_rate": float(np.mean([row["servo_requested_positive_rate"] for row in servo_rows if row["servo_requested_positive_rate"] is not None])) if servo_rows else None,
            "mean_executed_nonpositive_rate": float(np.mean([row["servo_executed_nonpositive_rate"] for row in servo_rows if row["servo_executed_nonpositive_rate"] is not None])) if servo_rows else None,
            "mean_filter_reverses_positive_request_rate": float(np.mean([row["filter_reverses_positive_request_rate"] for row in servo_rows if row["filter_reverses_positive_request_rate"] is not None])) if servo_rows else None,
            "mean_nominal_toward_goal_rate": float(np.mean([row["nominal_toward_goal_rate"] for row in servo_rows if row["nominal_toward_goal_rate"] is not None])) if servo_rows else None,
            "mean_recovery_command_rate": float(np.mean([row["recovery_command_rate"] for row in servo_rows if row["recovery_command_rate"] is not None])) if servo_rows else None,
            "mean_nominal_toward_goal_mps": float(np.mean([row["nominal_mean_toward_goal_mps"] for row in servo_rows if row["nominal_mean_toward_goal_mps"] is not None])) if servo_rows else None,
        },
        "episodes_detail": [{key: value for key, value in row.items() if key != "_counts"} for row in report_rows],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes_detail"}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
