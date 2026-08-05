"""Merge complete, non-overlapping task-feasibility precheck shards."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from scripts.audit_vaps_task_feasibility import PROTOCOL, _load_seed_manifest
except ModuleNotFoundError:
    from audit_vaps_task_feasibility import PROTOCOL, _load_seed_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge VAPS task-feasibility precheck shards.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--seed-manifest", default="configs/experiments/vaps_manifests/v2_g2_validation.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _load(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"shard must be a JSON object: {path}")
    return payload


def merge_audits(input_paths: Sequence[str | Path], seed_manifest_path: str | Path) -> dict[str, Any]:
    expected_seeds = _load_seed_manifest(seed_manifest_path)
    if not input_paths:
        raise ValueError("at least one shard is required")
    shards = [(Path(path), _load(path)) for path in input_paths]
    _, first = shards[0]
    common = ("config", "seed_manifest", "seed_count", "candidate_search_is_not_completeness_proof")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path, payload in shards:
        if payload.get("protocol") != PROTOCOL:
            raise ValueError(f"wrong precheck protocol: {path}")
        if payload.get("seed_manifest") != str(seed_manifest_path):
            raise ValueError(f"shard uses a different seed manifest: {path}")
        if any(payload.get(field) != first.get(field) for field in common):
            raise ValueError(f"shards disagree on fixed inputs: {path}")
        if (
            payload.get("actor_loaded") is not False
            or payload.get("training_performed") is not False
            or payload.get("runtime_control_modified") is not False
            or payload.get("candidate_search_is_not_completeness_proof") is not True
        ):
            raise ValueError(f"shard is not offline-only: {path}")
        for row in payload.get("episodes_detail", []):
            seed = int(row["seed"])
            if seed in seen:
                raise ValueError(f"duplicate seed in shards: {seed}")
            seen.add(seed)
            rows.append(row)
    if seen != set(expected_seeds):
        raise ValueError(f"shards do not cover the fixed manifest; missing={sorted(set(expected_seeds) - seen)}")
    rows.sort(key=lambda row: int(row["seed"]))
    status_counts = {
        field: {
            status: sum(row[section]["status"] == status for row in rows)
            for status in sorted({row[section]["status"] for row in rows})
        }
        for field, section in (
            ("ik_status", "ik"),
            ("no_obstacle_task_status", "no_obstacle_task"),
            ("dynamic_obstacle_path_status", "dynamic_obstacle_path"),
        )
    }
    return {
        "protocol": PROTOCOL,
        "config": first["config"],
        "seed_manifest": str(seed_manifest_path),
        "seed_count": len(rows),
        "actor_loaded": False,
        "training_performed": False,
        "runtime_control_modified": False,
        "candidate_search_is_not_completeness_proof": True,
        "status_counts": status_counts,
        "merged_shards": [str(path) for path, _ in shards],
        "episodes_detail": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = merge_audits(args.input, args.seed_manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": payload["seed_count"], "status_counts": payload["status_counts"], "output": str(output)}))


if __name__ == "__main__":
    main()
