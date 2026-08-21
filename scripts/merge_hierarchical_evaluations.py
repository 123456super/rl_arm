"""Validate, merge, and summarize parallel hierarchical evaluation shards."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from scripts.seed_manifest import load_seed_manifest
except ModuleNotFoundError:
    from seed_manifest import load_seed_manifest


STATE_NAMES = ("TRACK", "AVOID_HOLD", "REPLAN", "SERVO", "PLAN_FAILED")
FAILURE_MODES = (
    "SUCCESS",
    "IK_NOT_FOUND",
    "PLAN_NOT_FOUND",
    "TRACK_TIMEOUT",
    "SERVO_TIMEOUT",
    "FILTER_STOP_TIMEOUT",
    "COLLISION_TERMINATION",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--input", action="append", required=True, dest="inputs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--architecture-version", default="hierarchical_s1_nominal_final_v1")
    return parser.parse_args(argv)


def _int(row: dict[str, str], field: str) -> int:
    return int(float(row[field]))


def _float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def _rate(count: int, total: int) -> float | None:
    return float(count / total) if total else None


def _mean(rows: Sequence[dict[str, str]], field: str) -> float | None:
    values = np.asarray([_float(row, field) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else None


def _weighted_step_mean(rows: Sequence[dict[str, str]], field: str) -> float | None:
    values = np.asarray([_float(row, field) for row in rows], dtype=np.float64)
    weights = np.asarray([_int(row, "length") for row in rows], dtype=np.float64)
    usable = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[usable], weights=weights[usable])) if np.any(usable) else None


def load_and_validate_shards(
    input_paths: Sequence[str | Path], manifest_path: str | Path, episodes: int | None
) -> list[dict[str, str]]:
    manifest = load_seed_manifest(manifest_path)
    episode_count = len(manifest) if episodes is None else int(episodes)
    if episode_count <= 0 or episode_count > len(manifest):
        raise ValueError("episodes must be in [1, manifest length]")
    expected_seeds = manifest[:episode_count]
    expected_by_seed = {seed: episode for episode, seed in enumerate(expected_seeds)}

    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for input_path in input_paths:
        path = Path(input_path)
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"evaluation shard has no header: {path}")
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"evaluation shard columns differ: {path}")
            shard_rows = list(reader)
        if not shard_rows:
            raise ValueError(f"evaluation shard is empty: {path}")
        rows.extend(shard_rows)

    seen_seeds = [_int(row, "seed") for row in rows]
    duplicates = sorted(seed for seed, count in Counter(seen_seeds).items() if count > 1)
    missing = sorted(set(expected_seeds) - set(seen_seeds))
    unexpected = sorted(set(seen_seeds) - set(expected_seeds))
    if duplicates or missing or unexpected:
        raise ValueError(
            f"shards do not exactly cover manifest: duplicates={duplicates}, missing={missing}, "
            f"unexpected={unexpected}"
        )
    if len(rows) != episode_count:
        raise ValueError(f"expected {episode_count} rows, got {len(rows)}")

    manifest_label = str(manifest_path)
    for row in rows:
        seed = _int(row, "seed")
        if _int(row, "episode") != expected_by_seed[seed]:
            raise ValueError(f"episode index does not match manifest order for seed {seed}")
        if row.get("seed_manifest") != manifest_label:
            raise ValueError(f"seed {seed} records a different manifest: {row.get('seed_manifest')!r}")
        if _int(row, "nominal_only") != 1:
            raise ValueError(f"seed {seed} is not a nominal-only evaluation")
    return sorted(rows, key=lambda row: _int(row, "episode"))


def summarize(rows: Sequence[dict[str, str]], manifest_path: str | Path, architecture_version: str = "hierarchical_s1_nominal_final_v1") -> dict[str, Any]:
    total = len(rows)
    successes = sum(_int(row, "success") for row in rows)
    ik_found = sum(_int(row, "hierarchical_initial_ik_found") for row in rows)
    plan_found = sum(_int(row, "hierarchical_initial_plan_found") for row in rows)
    direct_paths = sum(_int(row, "hierarchical_initial_direct_path") for row in rows)
    collisions = sum(_int(row, "collision_any") for row in rows)
    failure_counts = Counter(row["hierarchical_failure_mode"] for row in rows)
    unknown_modes = sorted(set(failure_counts) - set(FAILURE_MODES))
    if unknown_modes:
        raise ValueError(f"unknown hierarchical failure modes: {unknown_modes}")
    timeouts = sum(failure_counts[mode] for mode in FAILURE_MODES if mode.endswith("_TIMEOUT"))
    success_rows = [row for row in rows if _int(row, "success") == 1]
    plan_conditioned_successes = sum(
        _int(row, "success") for row in rows if _int(row, "hierarchical_initial_plan_found") == 1
    )

    plan_conditioned_rate = _rate(plan_conditioned_successes, plan_found)
    full_success_rate = _rate(successes, total)
    return {
        "protocol": architecture_version,
        "seed_manifest": str(manifest_path),
        "episodes": total,
        "main": {
            "full_success": {"count": successes, "total": total, "rate": full_success_rate},
            "plan_conditioned_success": {
                "count": plan_conditioned_successes,
                "total": plan_found,
                "rate": plan_conditioned_rate,
            },
            "ik_found": {"count": ik_found, "total": total, "rate": _rate(ik_found, total)},
            "plan_found": {"count": plan_found, "total": total, "rate": _rate(plan_found, total)},
            "timeouts": {"count": timeouts, "total": total, "rate": _rate(timeouts, total)},
            "collision_any": {"count": collisions, "total": total, "rate": _rate(collisions, total)},
            "mean_final_position_error_m": _mean(rows, "final_position_error"),
            "mean_success_completion_time_s": _mean(success_rows, "completion_time"),
        },
        "mechanism": {
            "direct_path": {"count": direct_paths, "total": plan_found, "rate": _rate(direct_paths, plan_found)},
            "rrt_path": {
                "count": plan_found - direct_paths,
                "total": plan_found,
                "rate": _rate(plan_found - direct_paths, plan_found),
            },
            "mean_initial_plan_iterations": _mean(rows, "hierarchical_initial_plan_iterations"),
            "mean_initial_planning_time_s": _mean(rows, "hierarchical_initial_planning_time_s"),
            "mean_total_planning_time_s": _mean(rows, "hierarchical_planning_time_total_s"),
            "mean_replans_per_episode": _mean(rows, "hierarchical_replan_count"),
            "state_step_rates": {
                state: _weighted_step_mean(rows, f"hierarchical_{state.lower()}_rate")
                for state in STATE_NAMES
            },
            "filter_intervention_step_rate": _weighted_step_mean(rows, "safety_filter_intervention_rate"),
            "filter_safe_stop_step_rate": _weighted_step_mean(rows, "safety_filter_safe_stop_rate"),
            "filter_recovery_relaxed_step_rate": _weighted_step_mean(
                rows, "safety_filter_recovery_relaxed_rate"
            ),
            "mean_nominal_qdot_norm": _weighted_step_mean(
                rows, "mean_hierarchical_nominal_qdot_norm"
            ),
            "mean_residual_qdot_norm": _weighted_step_mean(rows, "mean_residual_qdot_norm"),
            "mean_episode_min_predictive_h_m": _mean(rows, "min_predictive_h_m"),
        },
        "failure_mode_counts": {mode: failure_counts[mode] for mode in FAILURE_MODES},
        "gate": {
            "plan_conditioned_success_threshold": 0.95,
            "plan_conditioned_success_pass": bool(
                plan_conditioned_rate is not None and plan_conditioned_rate >= 0.95
            ),
            "recommended_full_success_threshold": 0.90,
            "recommended_full_success_pass": bool(
                full_success_rate is not None and full_success_rate >= 0.90
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_and_validate_shards(args.inputs, args.seed_manifest, args.episodes)
    report = summarize(rows, args.seed_manifest, args.architecture_version)

    output = Path(args.output)
    summary_output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    print(f"saved: {output}")
    print(f"saved: {summary_output}")


if __name__ == "__main__":
    main()
