from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


OBSTACLE_SCENARIOS = (
    "random_crossing",
    "upper_arm_crossing",
    "elbow_crossing",
    "forearm_crossing",
    "wrist_crossing",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the preregistered restarted-R2 checkpoint rule.")
    parser.add_argument(
        "--matrix",
        default="configs/restart_2026-09-09/r2a_checkpoint_validation.yaml",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def build_candidate_table(episodes: pd.DataFrame, matrix: dict[str, Any]) -> pd.DataFrame:
    required = {"variant", "scenario", "success", "collision", "safety_violation_rate", "final_position_error"}
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise ValueError(f"Missing evaluation columns: {missing}")

    variants = matrix["variants"]
    records = []
    for variant, cfg in variants.items():
        frame = episodes[episodes["variant"] == variant]
        no_obstacle = frame[frame["scenario"] == "no_obstacle"]
        obstacle = frame[frame["scenario"].isin(OBSTACLE_SCENARIOS)]
        present = set(obstacle["scenario"].unique())
        if no_obstacle.empty or present != set(OBSTACLE_SCENARIOS):
            raise ValueError(f"Candidate {variant!r} is missing required validation scenarios")

        # Average within each equally sized scenario first, then macro-average
        # the five obstacle scenarios exactly as preregistered.
        by_scenario = obstacle.groupby("scenario", sort=True)[
            ["success", "collision", "safety_violation_rate", "final_position_error"]
        ].mean()
        records.append(
            {
                "variant": variant,
                "penalty": float(cfg["penalty"]),
                "checkpoint_step": int(cfg["checkpoint_step"]),
                "checkpoint": str(next(iter(cfg["checkpoints"].values()))),
                "no_obstacle_success_rate": float(no_obstacle["success"].mean()),
                "macro_success_rate": float(by_scenario["success"].mean()),
                "macro_collision_rate": float(by_scenario["collision"].mean()),
                "macro_safety_violation_rate": float(by_scenario["safety_violation_rate"].mean()),
                "macro_final_position_error": float(by_scenario["final_position_error"].mean()),
            }
        )

    table = pd.DataFrame.from_records(records)
    table["eligible"] = (
        (table["no_obstacle_success_rate"] >= 0.80)
        & (table["macro_success_rate"] >= 0.60)
    )
    return table


def select_candidate(table: pd.DataFrame) -> pd.Series:
    eligible = table[table["eligible"]].copy()
    if eligible.empty:
        raise RuntimeError("Checkpoint gate failed: no candidate meets both preregistered success thresholds")
    ranked = eligible.sort_values(
        by=[
            "macro_collision_rate",
            "macro_safety_violation_rate",
            "macro_success_rate",
            "macro_final_position_error",
            "penalty",
            "checkpoint_step",
        ],
        ascending=[True, True, False, True, True, True],
        kind="stable",
    )
    return ranked.iloc[0]


def main() -> None:
    args = parse_args()
    matrix = _load_yaml(Path(args.matrix))
    root = Path(matrix["evaluation"]["output_root"])
    summary = root / "summary"
    manifest_path = summary / "manifest.json"
    episodes_path = summary / "all_eval_episodes.csv"
    if not manifest_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError("Run summarize_fixed_actor_eval.py before checkpoint selection")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete", False):
        raise RuntimeError(f"Refusing selection from an incomplete evaluation matrix: {manifest_path}")

    table = build_candidate_table(pd.read_csv(episodes_path), matrix)
    table = table.sort_values(["penalty", "checkpoint_step"], kind="stable")
    table.to_csv(summary / "candidate_metrics.csv", index=False)
    try:
        selected = select_candidate(table)
    except RuntimeError as error:
        result = {"status": "gate_failed", "reason": str(error), "eligible_candidates": 0}
        (summary / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        raise SystemExit(2) from None

    result = {
        "status": "selected",
        "selection_rule": [
            "no_obstacle_success_rate >= 0.80",
            "macro_success_rate >= 0.60",
            "macro_collision_rate ascending",
            "macro_safety_violation_rate ascending",
            "macro_success_rate descending",
            "macro_final_position_error ascending",
            "penalty ascending",
            "checkpoint_step ascending (same-penalty deterministic tie-break)",
        ],
        "selected": selected.to_dict(),
        "eligible_candidates": int(table["eligible"].sum()),
    }
    (summary / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
