from __future__ import annotations

import csv
import json

import pytest

from scripts.merge_hierarchical_evaluations import load_and_validate_shards, summarize


def _row(episode: int, seed: int, manifest: str, success: int, plan_found: int) -> dict[str, object]:
    return {
        "episode": episode,
        "seed": seed,
        "seed_manifest": manifest,
        "manifest_shard_index": episode,
        "manifest_shard_count": 2,
        "nominal_only": 1,
        "success": success,
        "length": 10,
        "collision_any": 0,
        "final_position_error": 0.01 if success else 0.1,
        "completion_time": 0.5 if success else 12.0,
        "hierarchical_initial_ik_found": 1,
        "hierarchical_initial_plan_found": plan_found,
        "hierarchical_initial_direct_path": plan_found,
        "hierarchical_initial_plan_iterations": 0,
        "hierarchical_initial_planning_time_s": 0.01,
        "hierarchical_planning_time_total_s": 0.01,
        "hierarchical_replan_count": 0,
        "hierarchical_track_rate": 0.8,
        "hierarchical_avoid_hold_rate": 0.0,
        "hierarchical_replan_rate": 0.0,
        "hierarchical_servo_rate": 0.2,
        "hierarchical_plan_failed_rate": 0.0,
        "safety_filter_intervention_rate": 0.1,
        "safety_filter_safe_stop_rate": 0.0,
        "safety_filter_recovery_relaxed_rate": 0.0,
        "mean_hierarchical_nominal_qdot_norm": 0.2,
        "mean_residual_qdot_norm": 0.0,
        "min_predictive_h_m": 0.03,
        "hierarchical_failure_mode": "SUCCESS" if success else "PLAN_NOT_FOUND",
    }


def _write(path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_merge_validates_manifest_coverage_and_reports_gate(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"seeds": [101, 102]}), encoding="utf-8")
    shard_0 = tmp_path / "shard_0.csv"
    shard_1 = tmp_path / "shard_1.csv"
    _write(shard_0, [_row(0, 101, str(manifest), 1, 1)])
    _write(shard_1, [_row(1, 102, str(manifest), 0, 0)])

    rows = load_and_validate_shards([shard_0, shard_1], manifest, 2)
    report = summarize(rows, manifest)

    assert [int(row["seed"]) for row in rows] == [101, 102]
    assert report["main"]["full_success"]["rate"] == 0.5
    assert report["main"]["plan_conditioned_success"]["rate"] == 1.0
    assert report["gate"]["plan_conditioned_success_pass"] is True
    assert report["mechanism"]["mean_residual_qdot_norm"] == 0.0


def test_merge_rejects_duplicate_and_missing_seed(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"seeds": [101, 102]}), encoding="utf-8")
    shard_0 = tmp_path / "shard_0.csv"
    shard_1 = tmp_path / "shard_1.csv"
    _write(shard_0, [_row(0, 101, str(manifest), 1, 1)])
    _write(shard_1, [_row(0, 101, str(manifest), 1, 1)])

    with pytest.raises(ValueError, match="duplicates=.*missing"):
        load_and_validate_shards([shard_0, shard_1], manifest, 2)
