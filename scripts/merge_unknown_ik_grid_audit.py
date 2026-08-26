"""Merge unknown-seed deterministic grid IK audit shards."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.audit_unknown_ik_grid import FIELDNAMES
    from scripts.seed_manifest import load_seed_manifest
except ModuleNotFoundError:
    from audit_unknown_ik_grid import FIELDNAMES
    from seed_manifest import load_seed_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--remaining-unknown-manifest-output", required=True)
    return parser.parse_args(argv)


def merge(
    *, config: str | Path, manifest: str | Path, inputs: Sequence[str | Path]
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    expected = load_seed_manifest(manifest)
    rows: list[dict[str, str]] = []
    for path in inputs:
        with Path(path).open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != FIELDNAMES:
                raise ValueError(f"unexpected grid audit columns: {path}")
            shard_rows = list(reader)
        if not shard_rows:
            raise ValueError(f"empty grid audit shard: {path}")
        rows.extend(shard_rows)
    by_seed: dict[int, dict[str, str]] = {}
    for row in rows:
        seed = int(row["seed"])
        episode = int(row["episode"])
        if seed in by_seed:
            raise ValueError(f"duplicate seed: {seed}")
        if episode < 0 or episode >= len(expected) or expected[episode] != seed:
            raise ValueError(f"manifest mismatch for seed {seed}, episode {episode}")
        by_seed[seed] = row
    missing = [seed for seed in expected if seed not in by_seed]
    if missing or len(rows) != len(expected):
        raise ValueError(f"incomplete unknown manifest coverage: missing={missing}")
    ordered = [by_seed[seed] for seed in expected]
    counts = Counter(row["classification_after_grid"] for row in ordered)
    remaining = [int(row["seed"]) for row in ordered if row["classification_after_grid"] == "unknown"]
    summary = {
        "protocol": "hierarchical_s1_deterministic_ik_unknown_grid_audit_v1",
        "config": str(config),
        "seed_manifest": str(manifest),
        "episodes": len(ordered),
        "classification_counts": {
            "certified_feasible": int(counts["certified_feasible"]),
            "unknown": int(counts["unknown"]),
        },
        "upgraded_to_certified_feasible": [
            int(row["seed"]) for row in ordered if row["classification_after_grid"] == "certified_feasible"
        ],
        "remaining_unknown_seeds": remaining,
        "endpoint_ik_only": True,
        "negative_result_is_not_infeasibility_proof": True,
    }
    remaining_manifest = {
        "manifest_version": 1,
        "protocol": "hierarchical_s1_deterministic_ik_unknown_after_grid_v1",
        "source_manifest": str(manifest),
        "description": "Only unknown seeds remaining after deterministic 3^6 rest-pose grid IK audit.",
        "seeds": remaining,
    }
    return ordered, summary, remaining_manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows, summary, remaining = merge(config=args.config, manifest=args.seed_manifest, inputs=args.input)
    output = Path(args.output)
    summary_output = Path(args.summary_output)
    manifest_output = Path(args.remaining_unknown_manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    summary["remaining_unknown_manifest_output"] = str(manifest_output)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_output.write_text(json.dumps(remaining, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
