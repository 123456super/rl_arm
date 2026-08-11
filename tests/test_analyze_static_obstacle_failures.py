from __future__ import annotations

import csv
import json

from scripts.analyze_static_obstacle_failures import analyze


def _write_trace(path, errors: list[float], qdot: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["goal_error_norm", "d_min", "risk_global", "qdot_norm"])
        writer.writeheader()
        for error, velocity in zip(errors, qdot, strict=True):
            writer.writerow({"goal_error_norm": error, "d_min": 0.2, "risk_global": 0.1, "qdot_norm": velocity})


def test_static_failure_diagnostic_keeps_candidate_timeout(tmp_path) -> None:
    feasibility = tmp_path / "feasibility.json"
    feasibility.write_text(
        json.dumps(
            {
                "episodes_detail": [
                    {
                        "seed": 9001,
                        "goal_m": [0.4, -0.2, 0.4],
                        "obstacle_position_m": [0.5, 0.3, 0.4],
                        "ik": {"status": "reachable"},
                        "dynamic_obstacle_path": {"status": "candidate_path_found"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    evaluation = tmp_path / "final.csv"
    with evaluation.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "seed", "success", "collision_any"])
        writer.writeheader()
        writer.writerow({"episode": 0, "seed": 9001, "success": 0, "collision_any": 0})
    traces = tmp_path / "traces"
    traces.mkdir()
    _write_trace(traces / "episode_0000.csv", [0.5, 0.3, 0.3, 0.3], [0.1, 0.1, 0.1, 0.1])

    report, rows, reset_rows = analyze(
        feasibility,
        [f"4301={evaluation}"],
        [f"4301={traces}"],
        tail_steps=2,
    )

    assert report["pooled_summary"]["timeouts"] == 1
    assert report["by_static_candidate_path_status"]["candidate_path_found"]["timeouts"] == 1
    assert rows[0]["failure_mode"] == "low_motion_stall"
    assert reset_rows[0]["success_count"] == 0
