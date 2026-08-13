from __future__ import annotations

import csv

from scripts.summarize_s1_static_chain import summarize


def _write_eval(path, rows) -> None:
    fieldnames = [
        "episode",
        "seed",
        "seed_manifest",
        "success",
        "collision",
        "collision_any",
        "collision_capsule_overlap",
        "collision_pybullet_contact",
        "termination_collision",
        "termination_reason",
        "final_position_error",
        "completion_time",
        "min_distance",
        "mean_risk",
        "max_risk",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_s1_static_summary_reports_pooled_collisions_and_failures(tmp_path) -> None:
    eval_4301 = tmp_path / "seed_4301.csv"
    eval_4302 = tmp_path / "seed_4302.csv"
    rows = [
        {
            "episode": 0,
            "seed": 9001,
            "seed_manifest": "manifest.json",
            "success": 1,
            "collision": 0,
            "collision_any": 0,
            "collision_capsule_overlap": 0,
            "collision_pybullet_contact": 0,
            "termination_collision": 0,
            "termination_reason": "",
            "final_position_error": 0.01,
            "completion_time": 0.5,
            "min_distance": 0.2,
            "mean_risk": 0.1,
            "max_risk": 0.2,
        },
        {
            "episode": 1,
            "seed": 9002,
            "seed_manifest": "manifest.json",
            "success": 0,
            "collision": 1,
            "collision_any": 1,
            "collision_capsule_overlap": 1,
            "collision_pybullet_contact": 0,
            "termination_collision": 1,
            "termination_reason": "capsule_overlap",
            "final_position_error": 0.2,
            "completion_time": 12.0,
            "min_distance": -0.01,
            "mean_risk": 0.7,
            "max_risk": 1.0,
        },
    ]
    _write_eval(eval_4301, rows)
    _write_eval(eval_4302, rows)

    report, joined = summarize([f"4301={eval_4301}", f"4302={eval_4302}"], "test_protocol")

    assert report["protocol"] == "test_protocol"
    assert report["pooled_summary"]["episodes"] == 4
    assert report["pooled_summary"]["successes"] == 2
    assert report["pooled_summary"]["collision_any"] == 2
    assert report["pooled_summary"]["timeouts"] == 0
    assert report["failed_reset_seeds_by_train_seed"]["4301"] == [9002]
    assert len(joined) == 4
