from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def value_label(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p").replace("/", "_")


def variant_name(sweep: dict[str, Any], variant: dict[str, Any]) -> str:
    if "parameter" in sweep:
        key = str(sweep["parameter"]).split(".")[-1]
        return f"{key}_{value_label(variant[str(sweep['parameter'])])}"
    return "_".join(f"{key.split('.')[-1]}_{value_label(value)}" for key, value in variant.items())


def variants_from_sweep(sweep: dict[str, Any]) -> list[dict[str, Any]]:
    if "parameter" in sweep:
        parameter = str(sweep["parameter"])
        return [{parameter: value} for value in sweep["values"]]
    if "parameter_pairs" in sweep:
        return [{str(key): value for key, value in pair.items()} for pair in sweep["parameter_pairs"]]
    raise ValueError("sweep must define either parameter/values or parameter_pairs")


def write_variant_config(base_config: str, variant: dict[str, Any], path: Path) -> None:
    base_path = Path(base_config)
    config: dict[str, Any] = {"includes": [str(base_path if base_path.is_absolute() else base_path.resolve())]}
    for key, value in variant.items():
        set_dotted(config, key, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)


def train_command(config: Path, method: str, seed: int, total_steps: int, output_dir: Path) -> str:
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
            seed,
            "--total-steps",
            total_steps,
            "--output-dir",
            output_dir,
        ]
    )


def eval_command(config: Path, method: str, seed: int, episodes: int, checkpoint: Path, output: Path) -> str:
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
            seed,
            "--episodes",
            episodes,
            "--checkpoint",
            checkpoint,
            "--output",
            output,
        ]
    )


def main() -> None:
    args = parse_args()
    sweep_path = Path(args.sweep)
    sweep = load_yaml(sweep_path)
    method = str(sweep["method"])
    base_config = str(sweep["base_config"])
    output_root = Path(sweep["output_root"])
    config_root = output_root / "configs"
    train_root = output_root / "train"
    eval_root = output_root / "eval"
    total_steps = int(sweep["total_steps"])
    eval_episodes = int(sweep["eval_episodes"])
    eval_seed = int(sweep.get("eval_seed", 1001))

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated sweep commands only. Review and run selected lines when ready.",
        "",
        "# Training commands",
    ]
    eval_lines = ["", "# Evaluation commands"]

    for variant in variants_from_sweep(sweep):
        name = variant_name(sweep, variant)
        config_path = config_root / f"{name}.yaml"
        write_variant_config(base_config, variant, config_path)
        for seed in sweep["seeds"]:
            seed_int = int(seed)
            output_dir = train_root / name / f"seed_{seed_int}"
            lines.append(train_command(config_path, method, seed_int, total_steps, output_dir))
            checkpoint = output_dir / method / "actor.pt"
            output = eval_root / name / f"seed_{seed_int}" / "eval_metrics.csv"
            eval_lines.append(eval_command(config_path, method, eval_seed, eval_episodes, checkpoint, output))

    output_path = Path(args.output) if args.output else output_root / "commands.sh"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines + eval_lines) + "\n", encoding="utf-8")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
