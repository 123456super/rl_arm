"""Merge isolated G2 strict-projection-failure audit shards without relaxing evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from scripts.audit_vaps_g2_projection_failures import (
        PROTOCOL,
        _sha256,
        _target_key,
        load_target_manifest,
        source_projection_failures,
    )
except ModuleNotFoundError:
    from audit_vaps_g2_projection_failures import (
        PROTOCOL,
        _sha256,
        _target_key,
        load_target_manifest,
        source_projection_failures,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge all frozen G2 projection-failure audit shards.")
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


def merge_audits(input_paths: Sequence[str | Path], target_manifest: str | Path) -> dict[str, Any]:
    if not input_paths:
        raise ValueError("at least one audit shard is required")
    targets, trace_hashes = load_target_manifest(target_manifest)
    source_events = source_projection_failures(targets, trace_hashes)
    expected_keys = {_target_key(target) for target in targets}
    manifest_hash = _sha256(target_manifest)
    shards: list[tuple[Path, dict[str, Any]]] = [(Path(path), _load(path)) for path in input_paths]
    _, first = shards[0]
    common_fields = (
        "config",
        "config_sha256",
        "resolved_config_sha256",
        "checkpoint_manifest",
        "checkpoint_manifest_sha256",
        "target_manifest",
        "target_manifest_sha256",
        "source_trace_sha256",
        "atol",
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
        if payload.get("training_performed") is not False or payload.get("actor_checkpoint_selection_performed") is not False:
            raise ValueError(f"audit shard records forbidden training or selection: {path}")
        if payload.get("total_frozen_target_count") != len(targets) or payload.get("target_count") != 1:
            raise ValueError(f"audit shard must cover exactly one frozen target: {path}")
        target_index = payload.get("target_index")
        if isinstance(target_index, bool) or not isinstance(target_index, int) or target_index not in range(len(targets)):
            raise ValueError(f"audit shard target_index is invalid: {path}")
        if target_index in seen_indexes:
            raise ValueError(f"duplicate audit shard target_index: {target_index}")
        seen_indexes.add(target_index)
        if any(payload.get(field) != first.get(field) for field in common_fields):
            raise ValueError(f"audit shards disagree on frozen input: {path}")
        shard_events = payload.get("events")
        if not isinstance(shard_events, list) or len(shard_events) != 1:
            raise ValueError(f"audit shard must contain exactly one event: {path}")
        event = shard_events[0]
        if not isinstance(event, dict) or _target_key(event) != _target_key(targets[target_index]):
            raise ValueError(f"audit shard event does not match target_index: {path}")
        key = _target_key(event)
        if key not in source_events:
            raise ValueError(f"audit shard event is not frozen: {path}")
        if not (
            event.get("status_reproduced") is True
            and event.get("command_is_zero") is True
            and event.get("source_command_reproduced") is True
            and event.get("v0_v1_execution_reproduced") is True
        ):
            raise ValueError(f"audit shard does not verify strict fail-safe semantics: {path}")
        events.append(event)
    if seen_indexes != set(range(len(targets))):
        missing_indexes = sorted(set(range(len(targets))) - seen_indexes)
        raise ValueError(f"audit shards do not cover every frozen target index; missing={missing_indexes}")
    events.sort(key=lambda event: _target_key(event))
    zero_feasible_count = sum(bool(event["constraints"]["zero_command_satisfies_all_constraints"]) for event in events)
    return {
        "protocol": PROTOCOL,
        "source_protocol": first["source_protocol"],
        **{field: first[field] for field in common_fields},
        "training_performed": False,
        "actor_checkpoint_selection_performed": False,
        "target_count": len(events),
        "source_projection_failure_count": len(source_events),
        "all_statuses_reproduced": True,
        "all_commands_zero": True,
        "all_source_commands_reproduced": True,
        "all_v0_v1_execution_reproduced": True,
        "source_requested_exact_reproduction_count": sum(
            bool(event["source_requested_exactly_reproduced"]) for event in events
        ),
        "max_source_requested_abs_diff": max(event["source_requested_max_abs_diff"] for event in events),
        "zero_command_constraint_feasible_count": zero_feasible_count,
        "strict_fail_safe_semantics_verified": True,
        "projection_failures_resolved": False,
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
    print(json.dumps({"targets": payload["target_count"], "strict_fail_safe_semantics_verified": True, "output": str(output)}))


if __name__ == "__main__":
    main()
