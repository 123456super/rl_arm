import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_vaps_g2_projection_feasibility import (
    ITERATION_BUDGETS,
    _validate_config,
    classify_attempt,
    constraint_residual_audit,
)
from scripts.merge_vaps_g2_projection_feasibility_audits import merge_audits


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_trace(path: Path, events: list[tuple[int, int]] | None = None) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "seed",
                "step",
                "safety_filter_status",
                "v1_safety_filter_status",
                "passed",
                "field_mismatches",
            ),
        )
        writer.writeheader()
        for seed, step in events or [(8201, 17)]:
            writer.writerow(
                {
                    "seed": seed,
                    "step": step,
                    "safety_filter_status": "safe_stop_projection_failed",
                    "v1_safety_filter_status": "safe_stop_projection_failed",
                    "passed": "True",
                    "field_mismatches": "[]",
                }
            )


def test_constraint_residual_audit_includes_box_constraints() -> None:
    residual = constraint_residual_audit(
        np.array([1.5]),
        np.array([-1.0]),
        np.array([1.0]),
        np.array([[1.0]]),
        np.array([0.0]),
        ("predictive_link_0",),
        ("joint_0_acceleration_lower",),
        ("joint_0_acceleration_upper",),
        1.0e-6,
    )

    assert residual["satisfies_all_constraints"] is False
    assert residual["max_constraint_violation"] == pytest.approx(0.5)
    assert residual["max_constraint_label"] == "joint_0_acceleration_upper"


def test_classify_attempt_requires_a_strict_witness() -> None:
    residual = {
        "candidate_available": True,
        "candidate_is_finite": True,
        "max_constraint_violation": 1.0e-3,
        "max_constraint_label": "predictive_link_0",
        "max_constraint_category": "predictive_link",
        "satisfies_all_constraints": False,
    }

    assert classify_attempt("maximum iterations reached", residual) == "indeterminate_max_iterations"
    assert classify_attempt("primal infeasible", residual) == "primal_infeasible"


def test_baseline_replay_allows_the_frozen_qp_time_limit() -> None:
    _validate_config(
        {
            "eval": {"method": "link_fixed"},
            "env": {
                "safety_filter": {
                    "enabled": True,
                    "use_qp_solver": True,
                    "max_filter_compute_time_s": None,
                    "qp_time_limit_s": 0.02,
                    "recovery_mode_enabled": False,
                    "recovery_allow_constraint_relaxation": False,
                    "recovery_maximize_min_clearance": False,
                    "risk_speed_scaling_enabled": False,
                }
            },
        }
    )


def _attempt(maximum_iterations: int) -> dict[str, object]:
    candidate = [0.25]
    residual = constraint_residual_audit(
        np.asarray(candidate),
        np.array([-1.0]),
        np.array([1.0]),
        np.array([[1.0]]),
        np.array([0.0]),
        ("predictive_link_0",),
        ("joint_0_acceleration_lower",),
        ("joint_0_acceleration_upper",),
        1.0e-6,
    )
    return {
        "maximum_iterations": maximum_iterations,
        "osqp_status": "solved",
        "osqp_iterations_used": 25,
        "candidate_joint_velocity_radps": candidate,
        "residual_audit": residual,
        "classification": "strict_feasible",
    }


def _valid_shard(target_manifest: Path, trace: Path) -> dict[str, object]:
    snapshot = {
        "requested_joint_velocity_radps": [0.0],
        "joint_lower_radps": [-1.0],
        "joint_upper_radps": [1.0],
        "constraint_rows": [[1.0]],
        "constraint_lower_bounds": [0.0],
        "constraint_labels": ["predictive_link_0"],
        "joint_lower_labels": ["joint_0_acceleration_lower"],
        "joint_upper_labels": ["joint_0_acceleration_upper"],
    }
    event = {
        "train_seed": 4108,
        "seed": 8201,
        "step": 17,
        "source_trace": str(trace),
        "baseline_status": "safe_stop_projection_failed",
        "baseline_command_joint_velocity_radps": [0.0],
        "baseline_command_is_zero": True,
        "source_requested_max_abs_diff": 0.0,
        "source_command_max_abs_diff": 0.0,
        "constraint_snapshot": snapshot,
        "attempts": [_attempt(budget) for budget in ITERATION_BUDGETS],
    }
    return {
        "protocol": "vaps_g2_projection_feasibility_audit_v1",
        "source_protocol": "vaps_g2_strict_execution_v2",
        "config": "config.yaml",
        "config_sha256": "a" * 64,
        "resolved_config_sha256": "b" * 64,
        "checkpoint_manifest": "actors.json",
        "checkpoint_manifest_sha256": "c" * 64,
        "target_manifest": str(target_manifest),
        "target_manifest_sha256": _sha256(target_manifest),
        "source_trace_sha256": {str(trace): _sha256(trace)},
        "atol": 1.0e-7,
        "projection_failure_tolerance": 1.0e-6,
        "iteration_budgets": list(ITERATION_BUDGETS),
        "training_performed": False,
        "actor_checkpoint_selection_performed": False,
        "recovery_or_relaxation_performed": False,
        "target_count": 1,
        "total_frozen_target_count": 1,
        "target_index": 0,
        "source_projection_failure_count": 1,
        "baseline_failure_reproduced": True,
        "baseline_zero_command_reproduced": True,
        "source_requested_reproduced": True,
        "source_command_reproduced": True,
        "strict_feasible_attempt_count": len(ITERATION_BUDGETS),
        "g3_authorization": "not_granted_by_this_audit",
        "event": event,
    }


def test_merge_rejects_tampered_candidate_residual(tmp_path: Path) -> None:
    trace = tmp_path / "train_seed_4108_validation_trace.csv"
    _write_trace(trace)
    target_manifest = tmp_path / "targets.json"
    target_manifest.write_text(
        json.dumps(
            {
                "protocol": "vaps_g2_projection_failure_audit_v1",
                "source_protocol": "vaps_g2_strict_execution_v2",
                "source_trace_sha256": {str(trace): _sha256(trace)},
                "targets": [{"train_seed": 4108, "seed": 8201, "step": 17}],
            }
        ),
        encoding="utf-8",
    )
    shard = _valid_shard(target_manifest, trace)
    shard["event"]["attempts"][0]["candidate_joint_velocity_radps"] = [-2.0]
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(shard), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate residual"):
        merge_audits([path], target_manifest)


def test_merge_rejects_an_incomplete_isolated_shard_set(tmp_path: Path) -> None:
    trace = tmp_path / "train_seed_4108_validation_trace.csv"
    _write_trace(trace, events=[(8201, 17), (8202, 18)])
    target_manifest = tmp_path / "targets.json"
    target_manifest.write_text(
        json.dumps(
            {
                "protocol": "vaps_g2_projection_failure_audit_v1",
                "source_protocol": "vaps_g2_strict_execution_v2",
                "source_trace_sha256": {str(trace): _sha256(trace)},
                "targets": [
                    {"train_seed": 4108, "seed": 8201, "step": 17},
                    {"train_seed": 4108, "seed": 8202, "step": 18},
                ],
            }
        ),
        encoding="utf-8",
    )
    shard = _valid_shard(target_manifest, trace)
    shard["total_frozen_target_count"] = 2
    shard["source_projection_failure_count"] = 2
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(shard), encoding="utf-8")

    with pytest.raises(ValueError, match="cover every frozen target index"):
        merge_audits([path], target_manifest)
