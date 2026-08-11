"""Join static-obstacle final traces with fixed feasibility labels.

This is an offline diagnostic.  It keeps every final-manifest reset and
reports policy outcomes separately from finite candidate-path search labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


REQUIRED_TRACE_COLUMNS = {
    "goal_error_norm",
    "d_min",
    "risk_global",
    "qdot_norm",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group static-obstacle policy failures using final traces and fixed feasibility labels."
    )
    parser.add_argument("--feasibility", required=True, help="Static zero-speed candidate-path precheck JSON.")
    parser.add_argument(
        "--eval",
        action="append",
        required=True,
        metavar="TRAIN_SEED=CSV",
        help="One final evaluate.py CSV per actor.",
    )
    parser.add_argument(
        "--trace-dir",
        action="append",
        required=True,
        metavar="TRAIN_SEED=DIR",
        help="Trace directory matching each --eval actor.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--reset-output-csv", required=True)
    parser.add_argument("--success-tolerance-m", type=float, default=0.055)
    parser.add_argument("--near-goal-margin-m", type=float, default=0.025)
    parser.add_argument("--tail-steps", type=int, default=60)
    parser.add_argument("--stall-qdot-threshold-radps", type=float, default=0.2)
    parser.add_argument("--stall-progress-threshold-m", type=float, default=0.01)
    return parser.parse_args(argv)


def _parse_spec(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise ValueError(f"spec must use TRAIN_SEED=PATH: {value}")
    seed_text, path_text = value.split("=", 1)
    return int(seed_text), Path(path_text)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _mean(rows: Sequence[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows])) if rows else 0.0


def _median(rows: Sequence[dict[str, Any]], field: str) -> float:
    return float(median(float(row[field]) for row in rows)) if rows else 0.0


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successes = sum(int(row["success"]) for row in rows)
    outcomes = Counter(str(row["outcome"]) for row in rows)
    modes = Counter(str(row["failure_mode"]) for row in rows)
    return {
        "episodes": total,
        "successes": successes,
        "success_rate": _rate(successes, total),
        "timeouts": outcomes["timeout"],
        "collisions": outcomes["collision"],
        "failure_mode_counts": dict(sorted(modes.items())),
        "mean_initial_goal_error_m": _mean(rows, "initial_goal_error_m"),
        "mean_best_goal_error_m": _mean(rows, "best_goal_error_m"),
        "mean_final_goal_error_m": _mean(rows, "final_goal_error_m"),
        "median_final_goal_error_m": _median(rows, "final_goal_error_m"),
        "mean_initial_to_best_progress_m": _mean(rows, "initial_to_best_progress_m"),
        "mean_regression_after_best_m": _mean(rows, "regression_after_best_m"),
        "mean_tail_goal_progress_m": _mean(rows, "tail_goal_progress_m"),
        "mean_tail_qdot_norm_radps": _mean(rows, "tail_qdot_norm_radps"),
        "mean_tail_risk": _mean(rows, "tail_risk"),
        "mean_min_distance_m": _mean(rows, "min_distance_m"),
        "mean_obstacle_goal_distance_m": _mean(rows, "obstacle_goal_distance_m"),
    }


def _failure_mode(
    outcome: str,
    best_goal_error_m: float,
    final_goal_error_m: float,
    tail_qdot_norm_radps: float,
    tail_goal_progress_m: float,
    success_tolerance_m: float,
    near_goal_margin_m: float,
    stall_qdot_threshold_radps: float,
    stall_progress_threshold_m: float,
) -> str:
    if outcome == "success":
        return "success"
    if outcome == "collision":
        return "collision"
    near_goal_threshold_m = success_tolerance_m + near_goal_margin_m
    if best_goal_error_m <= near_goal_threshold_m:
        if final_goal_error_m - best_goal_error_m >= success_tolerance_m:
            return "near_goal_regression"
        return "near_goal_timeout"
    if (
        tail_qdot_norm_radps <= stall_qdot_threshold_radps
        and abs(tail_goal_progress_m) <= stall_progress_threshold_m
    ):
        return "low_motion_stall"
    return "nonconvergent_timeout"


def _load_labels(path: str | Path) -> dict[int, dict[str, Any]]:
    payload = _load_json(path)
    labels: dict[int, dict[str, Any]] = {}
    for item in payload.get("episodes_detail", []):
        if not isinstance(item, dict):
            raise ValueError("feasibility episodes_detail entries must be objects")
        seed = int(item["seed"])
        if seed in labels:
            raise ValueError(f"duplicate feasibility seed: {seed}")
        labels[seed] = item
    if not labels:
        raise ValueError("feasibility JSON has no episodes_detail")
    return labels


def _trace_metrics(path: Path, tail_steps: int) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"trace contains no steps: {path}")
    missing = REQUIRED_TRACE_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"trace is missing columns {sorted(missing)}: {path}")
    errors = [float(row["goal_error_norm"]) for row in rows]
    qdot_norms = [float(row["qdot_norm"]) for row in rows]
    risks = [float(row["risk_global"]) for row in rows]
    distances = [float(row["d_min"]) for row in rows]
    tail_start = max(0, len(rows) - tail_steps)
    return {
        "trace_steps": len(rows),
        "initial_goal_error_m": errors[0],
        "best_goal_error_m": min(errors),
        "final_goal_error_m": errors[-1],
        "initial_to_best_progress_m": errors[0] - min(errors),
        "regression_after_best_m": errors[-1] - min(errors),
        "tail_goal_progress_m": errors[tail_start] - errors[-1],
        "tail_qdot_norm_radps": float(np.mean(qdot_norms[tail_start:])),
        "tail_risk": float(np.mean(risks[tail_start:])),
        "min_distance_m": min(distances),
    }


def analyze(
    feasibility_path: str | Path,
    eval_specs: Sequence[str],
    trace_specs: Sequence[str],
    *,
    success_tolerance_m: float = 0.055,
    near_goal_margin_m: float = 0.025,
    tail_steps: int = 60,
    stall_qdot_threshold_radps: float = 0.2,
    stall_progress_threshold_m: float = 0.01,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if success_tolerance_m <= 0 or near_goal_margin_m < 0:
        raise ValueError("success tolerance must be positive and near-goal margin non-negative")
    if tail_steps <= 0 or stall_qdot_threshold_radps < 0 or stall_progress_threshold_m < 0:
        raise ValueError("tail and stall thresholds must be non-negative, with tail steps positive")

    labels = _load_labels(feasibility_path)
    eval_by_seed = dict(_parse_spec(value) for value in eval_specs)
    trace_by_seed = dict(_parse_spec(value) for value in trace_specs)
    if len(eval_by_seed) != len(eval_specs) or len(trace_by_seed) != len(trace_specs):
        raise ValueError("duplicate train seed in --eval or --trace-dir")
    if set(eval_by_seed) != set(trace_by_seed):
        raise ValueError("--eval and --trace-dir must specify the same train seeds")

    joined: list[dict[str, Any]] = []
    for train_seed, eval_path in sorted(eval_by_seed.items()):
        if not eval_path.is_file():
            raise FileNotFoundError(eval_path)
        with eval_path.open(newline="", encoding="utf-8") as file:
            eval_rows = list(csv.DictReader(file))
        seeds = [int(row["seed"]) for row in eval_rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate reset seed in evaluation: {eval_path}")
        if set(seeds) != set(labels):
            raise ValueError(
                f"{eval_path} does not cover feasibility labels; "
                f"missing={sorted(set(labels) - set(seeds))}, extra={sorted(set(seeds) - set(labels))}"
            )
        trace_dir = trace_by_seed[train_seed]
        for row in eval_rows:
            episode = int(row["episode"])
            reset_seed = int(row["seed"])
            label = labels[reset_seed]
            metrics = _trace_metrics(trace_dir / f"episode_{episode:04d}.csv", tail_steps)
            success = int(row["success"])
            collision_any = int(row.get("collision_any", row.get("collision", 0)))
            outcome = "success" if success else ("collision" if collision_any else "timeout")
            obstacle = [float(value) for value in label["obstacle_position_m"]]
            goal = [float(value) for value in label["goal_m"]]
            static_path = label["dynamic_obstacle_path"]
            static_candidate_status = str(static_path["status"])
            joined_row: dict[str, Any] = {
                "train_seed": train_seed,
                "episode": episode,
                "reset_seed": reset_seed,
                "success": success,
                "outcome": outcome,
                "collision_any": collision_any,
                "collision_capsule_overlap": int(row.get("collision_capsule_overlap", 0)),
                "collision_pybullet_contact": int(row.get("collision_pybullet_contact", 0)),
                "termination_collision": int(row.get("termination_collision", 0)),
                "termination_reason": str(row.get("termination_reason", "")),
                "static_candidate_path_status": static_candidate_status,
                "ik_status": str(label["ik"]["status"]),
                "goal_x_m": goal[0],
                "goal_y_m": goal[1],
                "goal_z_m": goal[2],
                "obstacle_x_m": obstacle[0],
                "obstacle_y_m": obstacle[1],
                "obstacle_z_m": obstacle[2],
                "obstacle_goal_distance_m": float(np.linalg.norm(np.asarray(obstacle) - np.asarray(goal))),
                "obstacle_goal_y_opposite_side": int(obstacle[1] * goal[1] < 0.0),
                **metrics,
            }
            joined_row["failure_mode"] = _failure_mode(
                outcome=outcome,
                best_goal_error_m=float(joined_row["best_goal_error_m"]),
                final_goal_error_m=float(joined_row["final_goal_error_m"]),
                tail_qdot_norm_radps=float(joined_row["tail_qdot_norm_radps"]),
                tail_goal_progress_m=float(joined_row["tail_goal_progress_m"]),
                success_tolerance_m=success_tolerance_m,
                near_goal_margin_m=near_goal_margin_m,
                stall_qdot_threshold_radps=stall_qdot_threshold_radps,
                stall_progress_threshold_m=stall_progress_threshold_m,
            )
            joined.append(joined_row)

    by_train_seed: dict[str, dict[str, Any]] = {}
    by_candidate_status: dict[str, dict[str, Any]] = {}
    by_failure_mode: dict[str, dict[str, Any]] = {}
    for train_seed in sorted(eval_by_seed):
        by_train_seed[str(train_seed)] = _summary([row for row in joined if row["train_seed"] == train_seed])
    for status in sorted({str(row["static_candidate_path_status"]) for row in joined}):
        by_candidate_status[status] = _summary(
            [row for row in joined if row["static_candidate_path_status"] == status]
        )
    for mode in sorted({str(row["failure_mode"]) for row in joined}):
        by_failure_mode[mode] = _summary([row for row in joined if row["failure_mode"] == mode])

    reset_rows: list[dict[str, Any]] = []
    rows_by_reset: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        rows_by_reset[int(row["reset_seed"])].append(row)
    for reset_seed, rows in sorted(rows_by_reset.items()):
        if len(rows) != len(eval_by_seed):
            raise ValueError(f"reset {reset_seed} lacks actor results")
        base = rows[0]
        mode_counts = Counter(str(row["failure_mode"]) for row in rows)
        outcomes = Counter(str(row["outcome"]) for row in rows)
        reset_rows.append(
            {
                "reset_seed": reset_seed,
                "static_candidate_path_status": base["static_candidate_path_status"],
                "ik_status": base["ik_status"],
                "goal_x_m": base["goal_x_m"],
                "goal_y_m": base["goal_y_m"],
                "goal_z_m": base["goal_z_m"],
                "obstacle_x_m": base["obstacle_x_m"],
                "obstacle_y_m": base["obstacle_y_m"],
                "obstacle_z_m": base["obstacle_z_m"],
                "obstacle_goal_distance_m": base["obstacle_goal_distance_m"],
                "obstacle_goal_y_opposite_side": base["obstacle_goal_y_opposite_side"],
                "actor_count": len(rows),
                "success_count": sum(int(row["success"]) for row in rows),
                "timeout_count": outcomes["timeout"],
                "collision_count": outcomes["collision"],
                "failure_mode_counts": json.dumps(dict(sorted(mode_counts.items())), sort_keys=True),
                "mean_final_goal_error_m": _mean(rows, "final_goal_error_m"),
                "mean_best_goal_error_m": _mean(rows, "best_goal_error_m"),
                "mean_tail_qdot_norm_radps": _mean(rows, "tail_qdot_norm_radps"),
                "mean_tail_goal_progress_m": _mean(rows, "tail_goal_progress_m"),
                "mean_min_distance_m": _mean(rows, "min_distance_m"),
            }
        )

    consistency_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actor_count = len(eval_by_seed)
    for row in reset_rows:
        if int(row["success_count"]) == actor_count:
            consistency_groups["all_success"].append(row)
        elif int(row["timeout_count"]) == actor_count:
            consistency_groups["all_timeout"].append(row)
        elif int(row["collision_count"]) > 0:
            consistency_groups["mixed_or_collision"].append(row)
        else:
            consistency_groups["mixed_success_timeout"].append(row)

    report = {
        "protocol": "s1_static_obstacle_failure_diagnostics_v1",
        "offline_diagnostic_only": True,
        "feasibility": str(feasibility_path),
        "eval_train_seeds": sorted(eval_by_seed),
        "episodes_per_train_seed": len(labels),
        "candidate_search_is_not_completeness_proof": True,
        "parameters": {
            "success_tolerance_m": success_tolerance_m,
            "near_goal_threshold_m": success_tolerance_m + near_goal_margin_m,
            "tail_steps": tail_steps,
            "stall_qdot_threshold_radps": stall_qdot_threshold_radps,
            "stall_progress_threshold_m": stall_progress_threshold_m,
        },
        "pooled_summary": _summary(joined),
        "per_train_seed": by_train_seed,
        "by_static_candidate_path_status": by_candidate_status,
        "by_failure_mode": by_failure_mode,
        "reset_consistency_counts": {key: len(value) for key, value in sorted(consistency_groups.items())},
        "reset_consistency_candidate_path_found": {
            key: sum(row["static_candidate_path_status"] == "candidate_path_found" for row in value)
            for key, value in sorted(consistency_groups.items())
        },
    }
    return report, sorted(joined, key=lambda row: (int(row["train_seed"]), int(row["reset_seed"]))), reset_rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report, rows, reset_rows = analyze(
        args.feasibility,
        args.eval,
        args.trace_dir,
        success_tolerance_m=args.success_tolerance_m,
        near_goal_margin_m=args.near_goal_margin_m,
        tail_steps=args.tail_steps,
        stall_qdot_threshold_radps=args.stall_qdot_threshold_radps,
        stall_progress_threshold_m=args.stall_progress_threshold_m,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(Path(args.output_csv), rows)
    _write_csv(Path(args.reset_output_csv), reset_rows)
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_csv": args.output_csv,
                "reset_output_csv": args.reset_output_csv,
                **report["pooled_summary"],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
