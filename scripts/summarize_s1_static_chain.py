"""Summarize S1 static-obstacle evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one CSV per S1 actor.")
    parser.add_argument(
        "--eval",
        action="append",
        required=True,
        metavar="TRAIN_SEED=CSV",
        help="One evaluate.py CSV per actor, for example 4301=outputs/.../seed_4301_final.csv.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--protocol", default="s1_static_obstacle")
    return parser.parse_args(argv)


def _parse_spec(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise ValueError(f"spec must use TRAIN_SEED=CSV: {value}")
    seed_text, path_text = value.split("=", 1)
    return int(seed_text), Path(path_text)


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _mean(values: Sequence[float]) -> float | None:
    finite = _finite(values)
    return float(np.mean(finite)) if len(finite) else None


def _percentile(values: Sequence[float], q: float) -> float | None:
    finite = _finite(values)
    return float(np.percentile(finite, q)) if len(finite) else None


def _max(values: Sequence[float]) -> float | None:
    finite = _finite(values)
    return float(np.max(finite)) if len(finite) else None


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successes = sum(int(row["success"]) for row in rows)
    failures = total - successes
    collision_any = sum(int(row["collision_any"]) for row in rows)
    capsule = sum(int(row["collision_capsule_overlap"]) for row in rows)
    physical = sum(int(row["collision_pybullet_contact"]) for row in rows)
    termination_collision = sum(int(row["termination_collision"]) for row in rows)
    timeouts = sum(
        int(int(row["success"]) == 0 and int(row["termination_collision"]) == 0)
        for row in rows
    )
    success_completion_times = [
        float(row["completion_time_s"]) for row in rows if int(row["success"]) == 1
    ]
    errors = [float(row["final_position_error_m"]) for row in rows]
    return {
        "episodes": total,
        "successes": successes,
        "failures": failures,
        "success_rate": _rate(successes, total),
        "timeouts": timeouts,
        "timeout_rate": _rate(timeouts, total),
        "collision_any": collision_any,
        "collision_any_rate": _rate(collision_any, total),
        "collision_capsule_overlap": capsule,
        "collision_capsule_overlap_rate": _rate(capsule, total),
        "collision_pybullet_contact": physical,
        "collision_pybullet_contact_rate": _rate(physical, total),
        "termination_collision": termination_collision,
        "termination_collision_rate": _rate(termination_collision, total),
        "mean_final_position_error_m": _mean(errors),
        "p95_final_position_error_m": _percentile(errors, 95),
        "max_final_position_error_m": _max(errors),
        "mean_success_completion_time_s": _mean(success_completion_times),
    }


def summarize(eval_specs: Sequence[str], protocol: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    specs = [_parse_spec(spec) for spec in eval_specs]
    if len({train_seed for train_seed, _ in specs}) != len(specs):
        raise ValueError("duplicate train seed in --eval")

    joined: list[dict[str, Any]] = []
    per_train_seed: dict[str, dict[str, Any]] = {}
    reference_episode_seeds: set[int] | None = None
    for train_seed, path in specs:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError(f"empty evaluation CSV: {path}")

        episode_seeds = {int(row["seed"]) for row in rows}
        if reference_episode_seeds is None:
            reference_episode_seeds = episode_seeds
        elif episode_seeds != reference_episode_seeds:
            raise ValueError(f"{path} does not cover the same episode seeds as prior eval CSVs")

        actor_rows: list[dict[str, Any]] = []
        for row in rows:
            joined_row = {
                "train_seed": int(train_seed),
                "episode": int(row["episode"]),
                "seed": int(row["seed"]),
                "seed_manifest": row.get("seed_manifest", ""),
                "success": int(float(row["success"])),
                "collision_any": int(float(row.get("collision_any", row.get("collision", 0)))),
                "collision_capsule_overlap": int(float(row.get("collision_capsule_overlap", 0))),
                "collision_pybullet_contact": int(float(row.get("collision_pybullet_contact", 0))),
                "termination_collision": int(float(row.get("termination_collision", 0))),
                "termination_reason": row.get("termination_reason", ""),
                "final_position_error_m": float(row["final_position_error"]),
                "completion_time_s": float(row["completion_time"]),
                "min_distance_m": float(row["min_distance"]),
                "mean_risk": float(row["mean_risk"]),
                "max_risk": float(row["max_risk"]),
            }
            joined.append(joined_row)
            actor_rows.append(joined_row)
        per_train_seed[str(train_seed)] = {"eval_csv": str(path), "summary": _summary(actor_rows)}

    report = {
        "protocol": protocol,
        "episodes_per_train_seed": len(reference_episode_seeds or []),
        "eval_train_seeds": [train_seed for train_seed, _ in specs],
        "full_final_not_conditioned_on_feasibility_labels": True,
        "success_threshold_m": 0.055,
        "pooled_summary": _summary(joined),
        "per_train_seed": per_train_seed,
        "failed_reset_seeds_by_train_seed": {
            str(train_seed): [
                row["seed"] for row in joined if row["train_seed"] == train_seed and int(row["success"]) == 0
            ]
            for train_seed, _ in specs
        },
    }
    return report, sorted(joined, key=lambda row: (row["train_seed"], row["episode"]))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report, rows = summarize(args.eval, args.protocol)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_json": str(output_json), "output_csv": str(output_csv), **report["pooled_summary"]}))


if __name__ == "__main__":
    main()
