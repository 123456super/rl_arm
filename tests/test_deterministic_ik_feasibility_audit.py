import numpy as np
import json
import csv

from scripts.audit_deterministic_ik_feasibility import classify_reset, urdf_absolute_reach_bound
from scripts.audit_unknown_ik_grid import deterministic_rest_poses
from scripts.merge_unknown_ik_continuous_refinement import merge as merge_continuous_refinement
from scripts.merge_deterministic_ik_feasibility_audit import merge_audit
from rl_risk_sac.utils.config import load_config


def test_reach_bound_is_finite_and_positive() -> None:
    bound = urdf_absolute_reach_bound(
        "assets/robots/universal_robots/ur_models/ur5.urdf",
        base_link="base_link",
        tool_link="tool0",
    )
    assert 1.0 < bound < 1.3


def test_constructive_candidate_is_feasible_certificate() -> None:
    config = load_config("configs/experiments/hierarchical/s1_static_deterministic_ik_feasibility_audit_v1.yaml")
    classification, reasons = classify_reset(
        config=config,
        goal=np.asarray([0.45, 0.0, 0.42]),
        obstacle_center=np.asarray([2.0, 2.0, 2.0]),
        obstacle_enabled=True,
        candidate_count=1,
        reach_bound_m=1.1,
        tool_capsule_radius_m=0.04,
    )
    assert classification == "certified_feasible"
    assert reasons == ["constructive_strict_valid_terminal"]


def test_finite_search_failure_is_unknown() -> None:
    config = load_config("configs/experiments/hierarchical/s1_static_deterministic_ik_feasibility_audit_v1.yaml")
    classification, reasons = classify_reset(
        config=config,
        goal=np.asarray([0.45, 0.0, 0.42]),
        obstacle_center=np.asarray([2.0, 2.0, 2.0]),
        obstacle_enabled=True,
        candidate_count=0,
        reach_bound_m=1.1,
        tool_capsule_radius_m=0.04,
    )
    assert classification == "unknown"
    assert reasons == ["finite_search_no_certificate"]


def test_geometric_reach_bound_can_certify_infeasibility() -> None:
    config = load_config("configs/experiments/hierarchical/s1_static_deterministic_ik_feasibility_audit_v1.yaml")
    classification, reasons = classify_reset(
        config=config,
        goal=np.asarray([2.0, 0.0, 0.0]),
        obstacle_center=np.asarray([2.0, 2.0, 2.0]),
        obstacle_enabled=True,
        candidate_count=0,
        reach_bound_m=1.1,
        tool_capsule_radius_m=0.04,
    )
    assert classification == "certified_infeasible"
    assert "outside_urdf_triangle_inequality_reach_bound" in reasons


def test_merge_emits_only_unknown_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"seeds": [11, 12, 13]}), encoding="utf-8")
    from scripts.audit_deterministic_ik_feasibility import FIELDNAMES

    rows = []
    for episode, (seed, classification) in enumerate(
        ((11, "certified_feasible"), (12, "unknown"), (13, "certified_infeasible"))
    ):
        row = {field: "" for field in FIELDNAMES}
        row.update({"episode": str(episode), "seed": str(seed), "classification": classification})
        rows.append(row)
    shard_path = tmp_path / "shard.csv"
    with shard_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    ordered, summary, unknown = merge_audit(
        inputs=[shard_path], manifest_path=manifest_path, config_path="audit.yaml"
    )
    assert [row["seed"] for row in ordered] == ["11", "12", "13"]
    assert summary["classification_counts"] == {
        "certified_feasible": 1,
        "certified_infeasible": 1,
        "unknown": 1,
    }
    assert unknown["seeds"] == [12]


def test_deterministic_rest_pose_grid_has_expected_coverage() -> None:
    poses = deterministic_rest_poses(
        np.zeros(6), np.ones(6), np.full(6, 0.2), np.full(6, 0.3), levels=3
    )
    assert len(poses) == 731
    assert all(np.isfinite(pose).all() for pose in poses)


def test_continuous_refinement_merge_preserves_unknowns(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"seeds": [21, 22]}), encoding="utf-8")
    from scripts.audit_unknown_ik_continuous_refinement import FIELDNAMES

    rows = []
    for episode, (seed, classification) in enumerate(((21, "unknown"), (22, "certified_feasible"))):
        row = {field: "" for field in FIELDNAMES}
        row.update(
            {
                "episode": str(episode),
                "seed": str(seed),
                "classification_after_refinement": classification,
            }
        )
        rows.append(row)
    shard_path = tmp_path / "shard.csv"
    with shard_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    ordered, summary, remaining = merge_continuous_refinement(
        config="config.yaml", manifest=manifest_path, inputs=[shard_path]
    )
    assert [row["seed"] for row in ordered] == ["21", "22"]
    assert summary["upgraded_to_certified_feasible"] == [22]
    assert remaining["seeds"] == [21]
