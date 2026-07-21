from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

import yaml


SCENARIO_CONFIGS = {
    "upper_arm_crossing": "configs/experiments/upper_arm_crossing.yaml",
    "elbow_crossing": "configs/experiments/elbow_crossing.yaml",
    "forearm_crossing": "configs/experiments/forearm_crossing.yaml",
    "wrist_crossing": "configs/experiments/wrist_crossing.yaml",
    "random_crossing": "configs/experiments/random_crossing.yaml",
    "no_obstacle": "configs/experiments/no_obstacle.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/experiments/formal_matrix.yaml")
    parser.add_argument("--output", default="outputs/formal/commands.sh")
    return parser.parse_args()


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def train_command(config: str, method: str, seed: int, total_steps: int, output_dir: Path) -> str:
    return shell_join(
        [
            "conda",
            "run",
            "-n",
            "rl",
            "python",
            "scripts/train.py",
            "--config",
            config,
            "--method",
            method,
            "--seed",
            str(seed),
            "--total-steps",
            str(total_steps),
            "--output-dir",
            str(output_dir),
        ]
    )


def eval_command(config: str, method: str, seed: int, episodes: int, checkpoint: Path, output: Path) -> str:
    return shell_join(
        [
            "conda",
            "run",
            "-n",
            "rl",
            "python",
            "scripts/evaluate.py",
            "--config",
            config,
            "--method",
            method,
            "--seed",
            str(seed),
            "--episodes",
            str(episodes),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ]
    )


def main() -> None:
    args = parse_args()
    matrix = load_yaml(args.matrix)
    methods = [str(method) for method in matrix["methods"]]
    training = matrix["training"]
    evaluation = matrix["evaluation"]
    train_root = Path(training["output_root"])
    eval_root = Path(evaluation["output_root"])

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated commands only. Review and run selected lines when ready.",
        "",
        "# Training commands",
    ]
    for method in methods:
        for seed in training["seeds"]:
            output_dir = train_root / method / f"seed_{int(seed)}"
            lines.append(train_command(str(training["config"]), method, int(seed), int(training["total_steps"]), output_dir))

    lines.extend(["", "# Evaluation commands"])
    for method in methods:
        for train_seed in training["seeds"]:
            checkpoint = train_root / method / f"seed_{int(train_seed)}" / method / "actor.pt"
            for scenario in evaluation["scenarios"]:
                config = SCENARIO_CONFIGS[str(scenario)]
                for eval_seed in evaluation["seeds"]:
                    output = (
                        eval_root
                        / str(scenario)
                        / method
                        / f"train_seed_{int(train_seed)}"
                        / f"eval_seed_{int(eval_seed)}.csv"
                    )
                    lines.append(
                        eval_command(
                            config,
                            method,
                            int(eval_seed),
                            int(evaluation["episodes_per_seed"]),
                            checkpoint,
                            output,
                        )
                    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
