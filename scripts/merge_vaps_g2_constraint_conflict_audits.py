"""Strictly merge all isolated G2 v2 constraint-conflict audit shards."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from scripts.audit_vaps_g2_constraint_conflicts import (
        DEFAULT_TARGET_MANIFEST,
        PROTOCOL,
        _sha256,
        load_target_manifest,
    )
    from scripts.audit_vaps_g2_projection_failures import _target_key
except ModuleNotFoundError:
    from audit_vaps_g2_constraint_conflicts import DEFAULT_TARGET_MANIFEST, PROTOCOL, _sha256, load_target_manifest
    from audit_vaps_g2_projection_failures import _target_key


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge all frozen G2 v2 constraint-conflict audit shards.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--target-manifest", default=DEFAULT_TARGET_MANIFEST)
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
    manifest, targets = load_target_manifest(target_manifest)
    manifest_hash = _sha256(target_manifest)
    shards = [(Path(path), _load(path)) for path in input_paths]
    _, first = shards[0]
    common_fields = (
        "source_protocol",
        "target_manifest",
        "target_manifest_sha256",
        "source_audit",
        "source_audit_sha256",
        "source_audit_protocol",
        "required_iteration_budget",
        "required_classification",
        "projection_failure_tolerance",
    )
    events: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path, payload in shards:
        if payload.get("protocol") != PROTOCOL:
            raise ValueError(f"wrong audit protocol: {path}")
        if payload.get("target_manifest") != str(target_manifest) or payload.get("target_manifest_sha256") != manifest_hash:
            raise ValueError(f"audit shard uses a different frozen target manifest: {path}")
        if any(payload.get(field) != first.get(field) for field in common_fields):
            raise ValueError(f"audit shards disagree on frozen inputs: {path}")
        if (
            payload.get("training_performed") is not False
            or payload.get("actor_checkpoint_selection_performed") is not False
            or payload.get("recovery_or_relaxation_performed") is not False
            or payload.get("runtime_control_modified") is not False
            or payload.get("g3_authorization") != "not_granted_by_this_audit"
        ):
            raise ValueError(f"audit shard exceeds the offline diagnostic authorization: {path}")
        target_index = payload.get("target_index")
        if isinstance(target_index, bool) or not isinstance(target_index, int) or target_index not in range(len(targets)):
            raise ValueError(f"audit shard target_index is invalid: {path}")
        if target_index in seen:
            raise ValueError(f"duplicate audit shard target_index: {target_index}")
        seen.add(target_index)
        target = payload.get("target")
        if not isinstance(target, dict) or _target_key(target) != _target_key(targets[target_index]):
            raise ValueError(f"audit shard target does not match target_index: {path}")
        conflicts = payload.get("minimal_category_conflicts")
        if not isinstance(conflicts, list) or not conflicts:
            raise ValueError(f"audit shard does not contain a minimal category conflict: {path}")
        categories = payload.get("available_constraint_categories")
        if not isinstance(categories, list) or not all(isinstance(category, str) for category in categories):
            raise ValueError(f"audit shard has invalid available constraint categories: {path}")
        events.append(
            {
                "target_index": target_index,
                "target": target,
                "available_constraint_categories": categories,
                "minimal_category_conflicts": conflicts,
            }
        )
    expected = set(range(len(targets)))
    if seen != expected:
        raise ValueError(f"audit shards do not cover every frozen target index; missing={sorted(expected - seen)}")
    events.sort(key=lambda item: _target_key(item["target"]))
    all_categories = sorted({category for event in events for category in event["available_constraint_categories"]})
    return {
        "protocol": PROTOCOL,
        **{field: first[field] for field in common_fields},
        "training_performed": False,
        "actor_checkpoint_selection_performed": False,
        "recovery_or_relaxation_performed": False,
        "runtime_control_modified": False,
        "target_count": len(events),
        "all_frozen_primal_infeasible_targets_covered": True,
        "available_constraint_categories": all_categories,
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
                "g3_authorization": payload["g3_authorization"],
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
