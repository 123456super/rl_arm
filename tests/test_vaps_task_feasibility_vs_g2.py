from __future__ import annotations

import csv
import json

from scripts.analyze_vaps_task_feasibility_vs_g2 import analyze


def test_analysis_reports_events_without_inventing_success(tmp_path) -> None:
    feasibility = tmp_path / "feasibility.json"
    feasibility.write_text(
        json.dumps(
            {
                "candidate_search_is_not_completeness_proof": True,
                "episodes_detail": [
                    {
                        "seed": 9001,
                        "ik": {"status": "reachable"},
                        "no_obstacle_task": {"status": "candidate_path_found"},
                        "dynamic_obstacle_path": {"status": "candidate_path_found"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    trace = tmp_path / "train_seed_4108_final_trace.csv"
    with trace.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed",
                "step",
                "termination_reason",
                "collision_pybullet_contact",
                "collision_capsule_overlap",
                "collision_any",
                "termination_collision",
                "safety_filter_status",
                "safety_filter_safe_stop",
                "viability_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "seed": 9001,
                "step": 0,
                "termination_reason": "",
                "collision_pybullet_contact": "0",
                "collision_capsule_overlap": "0",
                "collision_any": "0",
                "termination_collision": "0",
                "safety_filter_status": "filtered",
                "safety_filter_safe_stop": "0",
                "viability_status": "certified_viable",
            }
        )
    result = analyze(feasibility, [f"4108={trace}"], max_episode_steps=1)
    assert result["success_rate_available"] is False
    assert result["groups_pooled"]["ik_status=reachable"]["episodes"] == 1
    assert result["groups_pooled"][
        "ik_status=reachable|no_obstacle_task_status=candidate_path_found|dynamic_obstacle_path_status=candidate_path_found"
    ]["physical_contact_episodes"] == 0
