import csv
from pathlib import Path

from scripts.summarize_infeasibility_attribution import summarize_trace_dirs


def test_summarize_infeasibility_attribution_counts_categories(tmp_path: Path) -> None:
    trace = tmp_path / "episode_0000.csv"
    with open(trace, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "safety_filter_status",
                "safety_filter_infeasible_constraint_categories",
                "safety_filter_infeasibility_diagnostic_status",
                "collision_pybullet_contact",
                "collision_capsule_overlap",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "safety_filter_status": "safe_stop_infeasible",
                    "safety_filter_infeasible_constraint_categories": "joint_acceleration|predictive_barrier",
                    "safety_filter_infeasibility_diagnostic_status": "single_category_relaxation",
                    "collision_pybullet_contact": "1",
                    "collision_capsule_overlap": "1",
                },
                {
                    "safety_filter_status": "safe_stop_infeasible",
                    "safety_filter_infeasible_constraint_categories": "predictive_barrier",
                    "safety_filter_infeasibility_diagnostic_status": "single_category_relaxation",
                    "collision_pybullet_contact": "0",
                    "collision_capsule_overlap": "0",
                },
            ]
        )

    result = summarize_trace_dirs([tmp_path])

    assert result["episodes_with_infeasible_stop"] == 1
    assert result["infeasible_steps"] == 2
    assert result["physical_contact_episodes_with_infeasible_stop"] == 1
    assert result["single_category_relaxation_step_counts"] == {
        "joint_acceleration": 1,
        "predictive_barrier": 2,
    }
