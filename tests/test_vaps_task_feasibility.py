from __future__ import annotations

import json

import numpy as np

from scripts.audit_vaps_task_feasibility import (
    _path_limits_ok,
    _point_segment_distance,
    _smooth_path,
)
from scripts.merge_vaps_task_feasibility_audits import merge_audits


def test_smooth_path_respects_discrete_joint_limits() -> None:
    path, detail = _smooth_path(
        np.asarray([0.0]),
        np.asarray([0.5]),
        np.asarray([1.0]),
        np.asarray([4.0]),
        0.05,
        12.0,
    )

    assert path is not None
    assert detail["status"] == "candidate"
    valid, violations = _path_limits_ok(
        path,
        np.asarray([-2.0]),
        np.asarray([2.0]),
        np.asarray([1.0]),
        np.asarray([4.0]),
        0.05,
        1.0e-7,
    )
    assert valid
    assert max(violations.values()) <= 1.0e-7


def test_smooth_path_reports_duration_budget() -> None:
    path, detail = _smooth_path(
        np.asarray([0.0]),
        np.asarray([2.0]),
        np.asarray([1.0]),
        np.asarray([1.0]),
        0.05,
        0.1,
    )

    assert path is None
    assert detail["reason"] == "minimum_duration_exceeds_budget"


def test_point_segment_distance_handles_degenerate_segment() -> None:
    assert _point_segment_distance(np.asarray([1.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)) == 1.0


def test_merge_requires_complete_non_overlapping_precheck_shards(tmp_path) -> None:
    manifest = tmp_path / "seeds.json"
    manifest.write_text('{"seeds": [8201, 8202]}', encoding="utf-8")
    rows = []
    paths = []
    for index, seed in enumerate((8201, 8202)):
        row = {
            "seed": seed,
            "ik": {"status": "reachable"},
            "no_obstacle_task": {"status": "candidate_path_found"},
            "dynamic_obstacle_path": {"status": "not_found"},
        }
        rows.append(row)
        payload = {
            "protocol": "vaps_task_feasibility_precheck_v1",
            "config": "config.yaml",
            "seed_manifest": str(manifest),
            "seed_count": 2,
            "actor_loaded": False,
            "training_performed": False,
            "runtime_control_modified": False,
            "candidate_search_is_not_completeness_proof": True,
            "episodes_detail": [row],
        }
        path = tmp_path / f"shard_{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    merged = merge_audits(paths, manifest)

    assert merged["seed_count"] == 2
    assert merged["status_counts"]["dynamic_obstacle_path_status"] == {"not_found": 2}
