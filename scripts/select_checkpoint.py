from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from rl_risk_sac.utils.config import load_config


STEP_PATTERN = re.compile(r"actor_step_(\d+)\.pt$")
METRICS = {"mean_reward", "success_rate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved actor checkpoints on a validation seed and select one reproducibly."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--metric", choices=sorted(METRICS), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true", help="Re-run evaluations even when their CSV files exist.")
    return parser.parse_args()


def actor_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in run_dir.glob("actor_step_*.pt"):
        match = STEP_PATTERN.search(path.name)
        if match is not None:
            checkpoints.append((int(match.group(1)), path))
    return sorted(checkpoints)


def read_summary(path: Path) -> dict[str, float]:
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Validation output contains no episodes: {path}")

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in rows]))

    def mean_optional(field: str, fallback: str) -> float:
        return mean(field if field in rows[0] else fallback)

    return {
        "episodes": float(len(rows)),
        "mean_reward": mean("reward"),
        "success_rate": mean("success"),
        "collision_rate": mean_optional("collision_any", "collision"),
        "physical_contact_rate": mean_optional("collision_pybullet_contact", "collision"),
        "capsule_overlap_rate": mean_optional("collision_capsule_overlap", "collision"),
        "safety_violation_rate": mean("safety_violation_rate"),
        "mean_final_position_error": mean("final_position_error"),
        "mean_min_distance": mean("min_distance"),
    }


def selection_key(summary: dict[str, float], metric: str) -> tuple[float, float, float, float, float, int]:
    # The primary validation metric is configured explicitly. Safety metrics make ties deterministic.
    return (
        -summary[metric],
        summary["physical_contact_rate"],
        summary["capsule_overlap_rate"],
        summary["safety_violation_rate"],
        summary["mean_final_position_error"],
        int(summary["step"]),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty checkpoint summary")
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    selection_config = config.get("checkpoint_selection", {})
    seed = int(args.seed if args.seed is not None else selection_config["seed"])
    episodes = int(args.episodes if args.episodes is not None else selection_config["episodes"])
    metric = str(args.metric if args.metric is not None else selection_config["metric"])
    if episodes <= 0:
        raise ValueError("Validation episodes must be positive")
    if metric not in METRICS:
        raise ValueError(f"Unsupported selection metric: {metric}")

    run_dir = Path(args.run_dir)
    checkpoints = actor_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No actor_step_*.pt checkpoints found in {run_dir}")
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "checkpoint_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    evaluate_script = Path(__file__).with_name("evaluate.py")
    method = str(config["eval"]["method"])
    for step, checkpoint in checkpoints:
        output = output_dir / f"step_{step}_seed_{seed}_episodes_{episodes}.csv"
        if args.force or not output.exists():
            command = [
                sys.executable,
                str(evaluate_script),
                "--config",
                str(args.config),
                "--method",
                method,
                "--checkpoint",
                str(checkpoint),
                "--episodes",
                str(episodes),
                "--seed",
                str(seed),
                "--output",
                str(output),
            ]
            subprocess.run(command, check=True)
        summary: dict[str, Any] = read_summary(output)
        summary.update({"step": step, "checkpoint": str(checkpoint), "validation_csv": str(output)})
        summaries.append(summary)

    selected = min(summaries, key=lambda row: selection_key(row, metric))
    for row in summaries:
        row["selected"] = int(row is selected)
        row["selection_metric"] = metric
        row["validation_seed"] = seed
    write_csv(output_dir / "checkpoint_summary.csv", summaries)
    write_csv(output_dir / "selected_checkpoint.csv", [selected])
    print(f"selected checkpoint: {selected['checkpoint']}")
    print(f"saved: {output_dir / 'checkpoint_summary.csv'}")


if __name__ == "__main__":
    main()
