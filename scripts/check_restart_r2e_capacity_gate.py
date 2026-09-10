from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the preregistered restarted-R2e no-obstacle capacity gate."
    )
    parser.add_argument(
        "--matrix",
        default="configs/restart_2026-09-09/r2e_no_obstacle_validation.yaml",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def build_capacity_table(episodes: pd.DataFrame, matrix: dict[str, Any]) -> pd.DataFrame:
    required = {
        "variant",
        "scenario",
        "success",
        "collision",
        "final_position_error",
        "min_position_error",
        "completion_time",
    }
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise ValueError(f"Missing evaluation columns: {missing}")

    records: list[dict[str, Any]] = []
    expected_variants = set(matrix["variants"])
    unexpected_scenarios = sorted(set(episodes["scenario"]) - {"no_obstacle"})
    if unexpected_scenarios:
        raise ValueError(f"R2e capacity gate only accepts no_obstacle: {unexpected_scenarios}")

    for variant, cfg in matrix["variants"].items():
        frame = episodes[
            (episodes["variant"] == variant) & (episodes["scenario"] == "no_obstacle")
        ]
        if frame.empty:
            raise ValueError(f"Candidate {variant!r} is missing no_obstacle episodes")
        success_rate = float(frame["success"].mean())
        collision_rate = float(frame["collision"].mean())
        timeout = frame[(frame["success"] == 0) & (frame["collision"] == 0)]
        successful = frame[frame["success"] == 1]
        records.append(
            {
                "variant": variant,
                "checkpoint_step": int(cfg["checkpoint_step"]),
                "checkpoint": str(next(iter(cfg["checkpoints"].values()))),
                "episodes": int(len(frame)),
                "no_obstacle_success_rate": success_rate,
                "no_obstacle_collision_rate": collision_rate,
                "timeout_rate": float(1.0 - success_rate - collision_rate),
                "mean_final_position_error": float(frame["final_position_error"].mean()),
                "mean_min_position_error": float(frame["min_position_error"].mean()),
                "timeout_mean_final_position_error": float(timeout["final_position_error"].mean())
                if not timeout.empty
                else None,
                "timeout_mean_min_position_error": float(timeout["min_position_error"].mean())
                if not timeout.empty
                else None,
                "success_mean_completion_time": float(successful["completion_time"].mean())
                if not successful.empty
                else None,
            }
        )

    observed_variants = set(episodes["variant"])
    extras = sorted(observed_variants - expected_variants)
    if extras:
        raise ValueError(f"Evaluation contains variants not registered by the matrix: {extras}")

    table = pd.DataFrame.from_records(records)
    threshold = float(matrix["diagnostic_gate"]["minimum"])
    table["capacity_pass"] = table["no_obstacle_success_rate"] >= threshold
    return table.sort_values("checkpoint_step", kind="stable").reset_index(drop=True)


def best_diagnostic(table: pd.DataFrame) -> pd.Series:
    return table.sort_values(
        by=["no_obstacle_success_rate", "mean_final_position_error", "checkpoint_step"],
        ascending=[False, True, True],
        kind="stable",
    ).iloc[0]


def json_record(row: pd.Series) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            record[str(key)] = None
        elif hasattr(value, "item"):
            record[str(key)] = value.item()
        else:
            record[str(key)] = value
    return record


def main() -> None:
    matrix_path = Path(parse_args().matrix)
    matrix = _load_yaml(matrix_path)
    root = Path(matrix["evaluation"]["output_root"])
    summary = root / "summary"
    manifest_path = summary / "manifest.json"
    episodes_path = summary / "all_eval_episodes.csv"
    if not manifest_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError("Run summarize_fixed_actor_eval.py before the R2e capacity gate")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete", False):
        raise RuntimeError(f"Refusing an incomplete evaluation matrix: {manifest_path}")

    table = build_capacity_table(pd.read_csv(episodes_path), matrix)
    table.to_csv(summary / "capacity_metrics.csv", index=False)
    best = best_diagnostic(table)
    threshold = float(matrix["diagnostic_gate"]["minimum"])
    passed = bool(table["capacity_pass"].any())
    result = {
        "status": "capacity_passed" if passed else "capacity_failed",
        "gate": f"any checkpoint no_obstacle_success_rate >= {threshold:.2f}",
        "passing_candidates": int(table["capacity_pass"].sum()),
        "best_diagnostic": json_record(best),
        "actor_frozen": False,
        "next_stage": "R2-B/R2-C preregistration" if passed else "repair SAC/task learning before R2-C",
    }
    (summary / "capacity_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
