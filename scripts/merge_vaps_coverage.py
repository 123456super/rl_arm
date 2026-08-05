"""Merge and validate independently executed VAPS G1 coverage-audit shards."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__:
    from .audit_vaps_coverage import _summarize_rows
else:
    from audit_vaps_coverage import _summarize_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge complete VAPS G1 coverage-audit shards.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def merge_vaps_coverage(paths: Sequence[str | Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one input shard is required")
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"audit shard must be an object: {path}")
        payloads.append((path, payload))

    _, first = payloads[0]
    required = ("protocol", "config", "method", "seed_manifest", "actor_loaded", "training_performed", "shard")
    for key in required:
        if key not in first:
            raise ValueError(f"missing {key} in first audit shard")
    if first["actor_loaded"] or first["training_performed"]:
        raise ValueError("G1 shards must not load an actor or train")
    seed_manifest = first["seed_manifest"]
    if not isinstance(seed_manifest, dict) or not isinstance(seed_manifest.get("seeds"), list):
        raise ValueError("seed_manifest must contain seeds")
    expected_seeds = seed_manifest["seeds"]
    rows: list[dict[str, Any]] = []
    shard_indexes: set[int] = set()
    shard_count: int | None = None
    for path, payload in payloads:
        for key in required:
            if key not in payload:
                raise ValueError(f"missing {key} in audit shard: {path}")
        for key in ("protocol", "config", "method", "seed_manifest"):
            if payload[key] != first[key]:
                raise ValueError(f"inconsistent {key} in audit shard: {path}")
        if payload["actor_loaded"] or payload["training_performed"]:
            raise ValueError(f"actor or training recorded in audit shard: {path}")
        shard = payload["shard"]
        if not isinstance(shard, dict) or not isinstance(shard.get("index"), int) or not isinstance(shard.get("count"), int):
            raise ValueError(f"invalid shard metadata: {path}")
        if shard_count is None:
            shard_count = shard["count"]
        if shard["count"] != shard_count or shard["index"] in shard_indexes:
            raise ValueError(f"inconsistent or duplicate shard index: {path}")
        shard_indexes.add(shard["index"])
        detail = payload.get("episodes_detail")
        if not isinstance(detail, list):
            raise ValueError(f"missing episodes_detail in audit shard: {path}")
        rows.extend(detail)

    assert shard_count is not None
    if shard_indexes != set(range(shard_count)):
        raise ValueError("input shards do not cover every shard index")
    rows.sort(key=lambda row: int(row["seed"]))
    observed_seeds = [row.get("seed") for row in rows]
    if observed_seeds != expected_seeds:
        raise ValueError("merged audit rows do not exactly match the fixed seed manifest")
    return {
        "protocol": first["protocol"],
        "config": first["config"],
        "method": first["method"],
        "actor_loaded": False,
        "training_performed": False,
        "seed_manifest": seed_manifest,
        "merged_shards": [str(path) for path, _ in payloads],
        "shard_count": shard_count,
        **_summarize_rows(rows),
        "episodes_detail": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = merge_vaps_coverage(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "episodes": result["episodes"],
                "certified_viable_coverage_rate": result["certified_viable_coverage_rate"],
                "initially_unsafe_rate": result["initially_unsafe_rate"],
                "output": str(output),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
