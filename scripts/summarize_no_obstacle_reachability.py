"""Summarize no-obstacle policy reachability against fixed task labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join no-obstacle policy eval CSVs with task-feasibility labels.")
    parser.add_argument("--feasibility", required=True)
    parser.add_argument(
        "--eval",
        action="append",
        required=True,
        metavar="TRAIN_SEED=CSV",
        help="One evaluate.py CSV per actor, for example 4108=outputs/...csv.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args(argv)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _parse_spec(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise ValueError(f"spec must use TRAIN_SEED=CSV: {value}")
    seed_text, path_text = value.split("=", 1)
    return int(seed_text), Path(path_text)


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successes = sum(int(row["success"]) for row in rows)
    errors = np.asarray([float(row["final_position_error_m"]) for row in rows], dtype=np.float64)
    durations = np.asarray([float(row["completion_time_s"]) for row in rows], dtype=np.float64)
    return {
        "episodes": total,
        "successes": successes,
        "success_rate": _rate(successes, total),
        "mean_final_position_error_m": float(np.mean(errors)) if total else 0.0,
        "p95_final_position_error_m": float(np.percentile(errors, 95)) if total else 0.0,
        "max_final_position_error_m": float(np.max(errors)) if total else 0.0,
        "mean_completion_time_s": float(np.mean(durations)) if total else 0.0,
        "collision_episodes": sum(int(row.get("collision_any", 0)) for row in rows),
        "termination_collision_episodes": sum(int(row.get("termination_collision", 0)) for row in rows),
    }


def summarize(feasibility_path: str | Path, eval_specs: Sequence[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feasibility = _load_json(feasibility_path)
    labels: dict[int, dict[str, Any]] = {}
    for item in feasibility.get("episodes_detail", []):
        seed = int(item["seed"])
        if seed in labels:
            raise ValueError(f"duplicate feasibility seed: {seed}")
        labels[seed] = item
    if not labels:
        raise ValueError("feasibility JSON has no episodes_detail")

    specs = [_parse_spec(spec) for spec in eval_specs]
    if len({train_seed for train_seed, _ in specs}) != len(specs):
        raise ValueError("duplicate train seed in --eval")
    joined: list[dict[str, Any]] = []
    per_train_seed: dict[str, dict[str, Any]] = {}
    for train_seed, path in specs:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as file:
            eval_rows = list(csv.DictReader(file))
        eval_seeds = {int(row["seed"]) for row in eval_rows}
        if eval_seeds != set(labels):
            raise ValueError(
                f"{path} does not cover the fixed labels; missing={sorted(set(labels) - eval_seeds)}, "
                f"extra={sorted(eval_seeds - set(labels))}"
            )
        actor_rows: list[dict[str, Any]] = []
        for row in eval_rows:
            seed = int(row["seed"])
            label = labels[seed]
            joined_row = {
                "train_seed": train_seed,
                "seed": seed,
                "goal_x_m": float(label["goal_m"][0]),
                "goal_y_m": float(label["goal_m"][1]),
                "goal_z_m": float(label["goal_m"][2]),
                "ik_status": label["ik"]["status"],
                "no_obstacle_task_status": label["no_obstacle_task"]["status"],
                "dynamic_obstacle_path_status": label["dynamic_obstacle_path"]["status"],
                "success": int(row["success"]),
                "final_position_error_m": float(row["final_position_error"]),
                "completion_time_s": float(row["completion_time"]),
                "collision_any": int(row.get("collision_any", 0)),
                "collision_pybullet_contact": int(row.get("collision_pybullet_contact", 0)),
                "termination_collision": int(row.get("termination_collision", 0)),
                "termination_reason": row.get("termination_reason", ""),
            }
            joined.append(joined_row)
            actor_rows.append(joined_row)
        per_train_seed[str(train_seed)] = {"eval_csv": str(path), "summary": _summary(actor_rows)}

    dimensions = ("ik_status", "no_obstacle_task_status", "dynamic_obstacle_path_status")
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        for dimension in dimensions:
            group_rows[f"{dimension}={row[dimension]}"].append(row)
        exact = "|".join(f"{dimension}={row[dimension]}" for dimension in dimensions)
        group_rows[exact].append(row)

    report = {
        "protocol": "vaps_no_obstacle_policy_reachability_audit_v1",
        "feasibility": str(feasibility_path),
        "eval_train_seeds": [train_seed for train_seed, _ in specs],
        "episodes_per_train_seed": len(labels),
        "success_rate_is_policy_result": True,
        "candidate_search_is_not_completeness_proof": bool(
            feasibility.get("candidate_search_is_not_completeness_proof", False)
        ),
        "pooled_summary": _summary(joined),
        "per_train_seed": per_train_seed,
        "groups": {key: _summary(value) for key, value in sorted(group_rows.items())},
    }
    return report, sorted(joined, key=lambda row: (row["train_seed"], row["seed"]))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report, rows = summarize(args.feasibility, args.eval)
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
