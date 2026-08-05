from __future__ import annotations

import csv
import json

from scripts.summarize_no_obstacle_reachability import summarize


def test_no_obstacle_summary_joins_points_and_policy_success(tmp_path) -> None:
    feasibility = tmp_path / "feasibility.json"
    feasibility.write_text(
        json.dumps(
            {
                "candidate_search_is_not_completeness_proof": True,
                "episodes_detail": [
                    {
                        "seed": 9001,
                        "goal_m": [0.4, 0.1, 0.5],
                        "ik": {"status": "reachable"},
                        "no_obstacle_task": {"status": "candidate_path_found"},
                        "dynamic_obstacle_path": {"status": "candidate_path_found"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evaluation = tmp_path / "eval.csv"
    with evaluation.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["seed", "success", "final_position_error", "completion_time", "collision_any", "termination_collision", "termination_reason"],
        )
        writer.writeheader()
        writer.writerow({"seed": 9001, "success": 1, "final_position_error": 0.01, "completion_time": 1.0, "collision_any": 0, "termination_collision": 0, "termination_reason": ""})
    report, rows = summarize(feasibility, [f"4108={evaluation}"])
    assert report["success_rate_is_policy_result"] is True
    assert report["pooled_summary"]["success_rate"] == 1.0
    assert rows[0]["goal_z_m"] == 0.5
