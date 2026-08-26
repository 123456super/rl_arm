"""Merge deterministic nominal IK audit shards and emit the unknown manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.audit_deterministic_ik_feasibility import CLASSIFICATIONS, FIELDNAMES
    from scripts.seed_manifest import load_seed_manifest
except ModuleNotFoundError:
    from audit_deterministic_ik_feasibility import CLASSIFICATIONS, FIELDNAMES
    from seed_manifest import load_seed_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--unknown-manifest-output", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def _read_rows(paths: Sequence[str | Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != FIELDNAMES:
                raise ValueError(f"audit shard columns differ from the protocol: {path}")
            shard_rows = list(reader)
        if not shard_rows:
            raise ValueError(f"audit shard is empty: {path}")
        rows.extend(shard_rows)
    return rows


def merge_audit(
    *,
    inputs: Sequence[str | Path],
    manifest_path: str | Path,
    config_path: str | Path,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    expected = load_seed_manifest(manifest_path)
    rows = _read_rows(inputs)
    by_seed: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            seed = int(row["seed"])
            episode = int(row["episode"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"invalid seed/episode in audit row: {row}") from error
        if seed in by_seed:
            raise ValueError(f"duplicate seed in audit shards: {seed}")
        if episode < 0 or episode >= len(expected) or expected[episode] != seed:
            raise ValueError(f"seed/episode does not match manifest order: seed={seed}, episode={episode}")
        if row["classification"] not in CLASSIFICATIONS:
            raise ValueError(f"unknown classification for seed {seed}: {row['classification']!r}")
        by_seed[seed] = row

    missing = [seed for seed in expected if seed not in by_seed]
    unexpected = [seed for seed in by_seed if seed not in set(expected)]
    if missing or unexpected or len(rows) != len(expected):
        raise ValueError(f"audit shards do not exactly cover manifest: missing={missing}, unexpected={unexpected}")
    ordered = [by_seed[seed] for seed in expected]
    counts = Counter(row["classification"] for row in ordered)
    unknown_seeds = [int(row["seed"]) for row in ordered if row["classification"] == "unknown"]
    feasible_seeds = [int(row["seed"]) for row in ordered if row["classification"] == "certified_feasible"]
    infeasible_seeds = [int(row["seed"]) for row in ordered if row["classification"] == "certified_infeasible"]
    summary: dict[str, Any] = {
        "protocol": "hierarchical_s1_deterministic_ik_feasibility_audit_v1",
        "config": str(config_path),
        "seed_manifest": str(manifest_path),
        "episodes": len(ordered),
        "classification_counts": {name: int(counts[name]) for name in CLASSIFICATIONS},
        "certified_feasible_seeds": feasible_seeds,
        "certified_infeasible_seeds": infeasible_seeds,
        "unknown_seeds": unknown_seeds,
        "unknown_manifest_output": "",
        "classification_is_endpoint_ik_only": True,
        "infeasible_certificate_policy": (
            "Only workspace/reach-bound/terminal-obstacle-shell proofs are certified; "
            "finite IK-search failure remains unknown."
        ),
    }
    unknown_manifest = {
        "manifest_version": 1,
        "protocol": "hierarchical_s1_deterministic_ik_unknown_v1",
        "source_protocol": summary["protocol"],
        "source_seed_manifest": str(manifest_path),
        "description": "Only resets without a deterministic endpoint IK certificate; use for follow-up research.",
        "seeds": unknown_seeds,
    }
    return ordered, summary, unknown_manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows, summary, unknown_manifest = merge_audit(
        inputs=args.input,
        manifest_path=args.seed_manifest,
        config_path=args.config,
    )
    output = Path(args.output)
    summary_output = Path(args.summary_output)
    unknown_output = Path(args.unknown_manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    unknown_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    summary["unknown_manifest_output"] = str(unknown_output)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    unknown_output.write_text(json.dumps(unknown_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
