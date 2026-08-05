"""Merge all isolated G2 projection-feasibility audit shards strictly."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.audit_vaps_g2_projection_failures import _sha256, _target_key, load_target_manifest, source_projection_failures
    from scripts.audit_vaps_g2_projection_feasibility import (
        ITERATION_BUDGETS,
        PROTOCOL,
        classify_attempt,
        constraint_residual_audit,
    )
except ModuleNotFoundError:
    from audit_vaps_g2_projection_failures import _sha256, _target_key, load_target_manifest, source_projection_failures
    from audit_vaps_g2_projection_feasibility import (
        ITERATION_BUDGETS,
        PROTOCOL,
        classify_attempt,
        constraint_residual_audit,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge all isolated frozen G2 projection-feasibility audit shards.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument(
        "--target-manifest",
        default="configs/experiments/vaps_manifests/v1_g2_projection_failure_targets.json",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _load(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"audit shard must be a JSON object: {path}")
    return payload


def _residual_from_snapshot(event: dict[str, Any], attempt: dict[str, Any], tolerance: float) -> dict[str, Any]:
    snapshot = event.get("constraint_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("audit event is missing its original constraint snapshot")
    try:
        candidate_raw = attempt["candidate_joint_velocity_radps"]
        candidate = None if candidate_raw is None else np.asarray(candidate_raw, dtype=np.float64)
        return constraint_residual_audit(
            candidate,
            np.asarray(snapshot["joint_lower_radps"], dtype=np.float64),
            np.asarray(snapshot["joint_upper_radps"], dtype=np.float64),
            np.asarray(snapshot["constraint_rows"], dtype=np.float64),
            np.asarray(snapshot["constraint_lower_bounds"], dtype=np.float64),
            tuple(snapshot["constraint_labels"]),
            tuple(snapshot["joint_lower_labels"]),
            tuple(snapshot["joint_upper_labels"]),
            tolerance,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("audit event has an invalid constraint snapshot or candidate") from error


def _validate_attempts(event: dict[str, Any], tolerance: float) -> list[dict[str, Any]]:
    attempts = event.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != len(ITERATION_BUDGETS):
        raise ValueError("audit event must contain every fixed iteration budget exactly once")
    if [attempt.get("maximum_iterations") for attempt in attempts] != list(ITERATION_BUDGETS):
        raise ValueError("audit attempts do not use the fixed iteration budgets")
    verified: list[dict[str, Any]] = []
    allowed = {
        "strict_feasible",
        "primal_infeasible",
        "indeterminate_max_iterations",
        "indeterminate_solver_status",
    }
    for attempt in attempts:
        if not isinstance(attempt, dict) or attempt.get("classification") not in allowed:
            raise ValueError("audit attempt has an invalid classification")
        status = attempt.get("osqp_status")
        if not isinstance(status, str):
            raise ValueError("audit attempt is missing its OSQP status")
        reported = attempt.get("residual_audit")
        if not isinstance(reported, dict):
            raise ValueError("audit attempt is missing its residual audit")
        recomputed = _residual_from_snapshot(event, attempt, tolerance)
        if recomputed != reported:
            raise ValueError("audit candidate residual does not match the original frozen constraints")
        if attempt["classification"] != classify_attempt(status, recomputed):
            raise ValueError("audit classification does not match the solver status and residual audit")
        verified.append(attempt)
    return verified


def merge_audits(input_paths: Sequence[str | Path], target_manifest: str | Path) -> dict[str, Any]:
    if not input_paths:
        raise ValueError("at least one audit shard is required")
    targets, trace_hashes = load_target_manifest(target_manifest)
    source_events = source_projection_failures(targets, trace_hashes)
    manifest_hash = _sha256(target_manifest)
    shards = [(Path(path), _load(path)) for path in input_paths]
    _, first = shards[0]
    common_fields = (
        "source_protocol",
        "config",
        "config_sha256",
        "resolved_config_sha256",
        "checkpoint_manifest",
        "checkpoint_manifest_sha256",
        "target_manifest",
        "target_manifest_sha256",
        "source_trace_sha256",
        "atol",
        "projection_failure_tolerance",
        "iteration_budgets",
    )
    events: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    for path, payload in shards:
        if payload.get("protocol") != PROTOCOL:
            raise ValueError(f"wrong audit protocol: {path}")
        if payload.get("target_manifest") != str(target_manifest) or payload.get("target_manifest_sha256") != manifest_hash:
            raise ValueError(f"target manifest mismatch: {path}")
        if payload.get("source_trace_sha256") != trace_hashes:
            raise ValueError(f"source trace hashes mismatch: {path}")
        if payload.get("iteration_budgets") != list(ITERATION_BUDGETS):
            raise ValueError(f"fixed iteration budgets mismatch: {path}")
        if (
            payload.get("training_performed") is not False
            or payload.get("actor_checkpoint_selection_performed") is not False
            or payload.get("recovery_or_relaxation_performed") is not False
            or payload.get("g3_authorization") != "not_granted_by_this_audit"
        ):
            raise ValueError(f"audit shard records a forbidden action: {path}")
        if payload.get("total_frozen_target_count") != len(targets) or payload.get("target_count") != 1:
            raise ValueError(f"audit shard must cover exactly one frozen target: {path}")
        target_index = payload.get("target_index")
        if isinstance(target_index, bool) or not isinstance(target_index, int) or target_index not in range(len(targets)):
            raise ValueError(f"audit shard target_index is invalid: {path}")
        if target_index in seen_indexes:
            raise ValueError(f"duplicate audit shard target_index: {target_index}")
        seen_indexes.add(target_index)
        if any(payload.get(field) != first.get(field) for field in common_fields):
            raise ValueError(f"audit shards disagree on frozen inputs: {path}")
        if (
            payload.get("baseline_failure_reproduced") is not True
            or payload.get("baseline_zero_command_reproduced") is not True
            or payload.get("source_requested_reproduced") is not True
            or payload.get("source_command_reproduced") is not True
        ):
            raise ValueError(f"audit shard does not reproduce the frozen fail-safe behavior: {path}")
        event = payload.get("event")
        if not isinstance(event, dict) or _target_key(event) != _target_key(targets[target_index]):
            raise ValueError(f"audit shard event does not match target_index: {path}")
        event_key = _target_key(event)
        if event_key not in source_events:
            raise ValueError(f"audit event is not in the frozen source set: {path}")
        if event.get("source_trace") != source_events[event_key]["source_trace"]:
            raise ValueError(f"audit event source trace does not match the frozen source: {path}")
        snapshot = event.get("constraint_snapshot")
        if not isinstance(snapshot, dict) or "joint_lower_radps" not in snapshot:
            raise ValueError(f"audit event is missing its original constraint snapshot: {path}")
        baseline_command = np.asarray(event.get("baseline_command_joint_velocity_radps"), dtype=np.float64)
        source_differences = np.asarray(
            [event.get("source_requested_max_abs_diff"), event.get("source_command_max_abs_diff")],
            dtype=np.float64,
        )
        if (
            event.get("baseline_status") != "safe_stop_projection_failed"
            or event.get("baseline_command_is_zero") is not True
            or baseline_command.shape != np.asarray(snapshot["joint_lower_radps"]).shape
            or not np.isfinite(baseline_command).all()
            or np.max(np.abs(baseline_command), initial=0.0) > float(payload["atol"])
            or not np.isfinite(source_differences).all()
            or (source_differences < 0.0).any()
            or np.max(source_differences) > float(payload["atol"])
        ):
            raise ValueError(f"audit event does not preserve the original strict fail-safe status: {path}")
        attempts = _validate_attempts(event, float(payload["projection_failure_tolerance"]))
        if payload.get("strict_feasible_attempt_count") != sum(
            attempt["classification"] == "strict_feasible" for attempt in attempts
        ):
            raise ValueError(f"strict_feasible_attempt_count disagrees with audit attempts: {path}")
        events.append(event)
    if seen_indexes != set(range(len(targets))):
        missing_indexes = sorted(set(range(len(targets))) - seen_indexes)
        raise ValueError(f"audit shards do not cover every frozen target index; missing={missing_indexes}")
    events.sort(key=_target_key)
    classifications = [attempt["classification"] for event in events for attempt in event["attempts"]]
    return {
        "protocol": PROTOCOL,
        **{field: first[field] for field in common_fields},
        "training_performed": False,
        "actor_checkpoint_selection_performed": False,
        "recovery_or_relaxation_performed": False,
        "target_count": len(events),
        "source_projection_failure_count": len(source_events),
        "all_baseline_failures_reproduced": True,
        "all_baseline_commands_zero": True,
        "strict_feasible_attempt_count": sum(item == "strict_feasible" for item in classifications),
        "primal_infeasible_attempt_count": sum(item == "primal_infeasible" for item in classifications),
        "indeterminate_attempt_count": sum(item.startswith("indeterminate_") for item in classifications),
        "g3_authorization": "not_granted_by_this_audit",
        "merged_shards": [str(path) for path, _ in shards],
        "events": events,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = merge_audits(args.input, args.target_manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "targets": payload["target_count"],
                "strict_feasible_attempts": payload["strict_feasible_attempt_count"],
                "primal_infeasible_attempts": payload["primal_infeasible_attempt_count"],
                "indeterminate_attempts": payload["indeterminate_attempt_count"],
                "g3_authorization": payload["g3_authorization"],
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
