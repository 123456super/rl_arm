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


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def write_train_config(base_config: str, path: Path, overrides: dict[str, Any]) -> None:
    base_path = Path(base_config)
    config: dict[str, Any] = {"includes": [str(base_path if base_path.is_absolute() else base_path.resolve())]}
    for key, value in overrides.items():
        set_dotted(config, key, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)


def train_command(config: Path) -> str:
    return shell_join(
        [
            "conda",
            "run",
            "-n",
            "rl",
            "python",
            "scripts/train.py",
            "--config",
            str(config),
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
    config_root = train_root / "configs"
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
            seed_int = int(seed)
            total_steps = int(training["total_steps"])
            progress_interval = int(training.get("progress_interval", 100))
            log_interval = int(training.get("log_interval", 10))
            save_interval = int(training.get("save_interval", 10000))
            run_name = f"{method}_seed{seed_int}_steps{total_steps}"
            output_dir = train_root / method / f"seed_{seed_int}"
            config_path = config_root / f"{run_name}.yaml"
            write_train_config(
                str(training["config"]),
                config_path,
                {
                    "seed": seed_int,
                    "train.method": method,
                    "train.total_steps": total_steps,
                    "train.progress_interval": progress_interval,
                    "train.log_interval": log_interval,
                    "train.save_interval": save_interval,
                    "train.output_dir": str(output_dir),
                    "train.run_name": run_name,
                },
            )
            lines.append(train_command(config_path))

    lines.extend(["", "# Evaluation commands"])
    for method in methods:
        for train_seed in training["seeds"]:
            train_seed_int = int(train_seed)
            run_name = f"{method}_seed{train_seed_int}_steps{int(training['total_steps'])}"
            checkpoint = train_root / method / f"seed_{train_seed_int}" / run_name / "actor.pt"
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
