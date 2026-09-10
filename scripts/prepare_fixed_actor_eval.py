from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import yaml

from rl_risk_sac.utils.provenance import _source_hashes, sha256_file


ENV_SCENARIOS = {
    "no_obstacle": None,
    "random_crossing": "random",
    "upper_arm_crossing": "upper_arm_crossing",
    "elbow_crossing": "elbow_crossing",
    "forearm_crossing": "forearm_crossing",
    "wrist_crossing": "wrist_crossing",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired fixed-actor evaluation configs and commands."
    )
    parser.add_argument(
        "--matrix",
        default="configs/experiments/minimal_qp_heldout_counterfactual.yaml",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Command file path (default: <matrix evaluation.output_root>/commands.sh)",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def write_scenario_config(base_config: Path, scenario: str, output: Path) -> None:
    if scenario not in ENV_SCENARIOS:
        raise ValueError(f"Unsupported scenario {scenario!r}; expected one of {sorted(ENV_SCENARIOS)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "includes": [str(base_config.resolve())],
        "env": {
            "obstacle": {
                "enabled": ENV_SCENARIOS[scenario] is not None,
                "scenario": ENV_SCENARIOS[scenario] or "random",
            }
        },
    }
    with open(output, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()
    matrix_path = Path(args.matrix)
    matrix = load_yaml(matrix_path)
    variants = matrix["variants"]
    evaluation = matrix["evaluation"]
    output_root = Path(evaluation["output_root"])
    command_output = Path(args.output) if args.output is not None else output_root / "commands.sh"
    config_root = output_root / "configs"
    command_prefix = [str(part) for part in evaluation.get(
        "command_prefix",
        ["conda", "run", "--no-capture-output", "-n", "rl", "python"],
    )]
    if not command_prefix:
        raise ValueError("evaluation.command_prefix must not be empty")

    variant_settings: dict[str, tuple[str, dict[int, Path], Path]] = {}
    for variant, variant_cfg in variants.items():
        method_value = variant_cfg.get("method", matrix.get("method"))
        checkpoints_value = variant_cfg.get("checkpoints", matrix.get("checkpoints"))
        if method_value is None or checkpoints_value is None:
            raise KeyError(
                f"Variant {variant!r} must define method/checkpoints or inherit top-level values"
            )
        base_config = Path(str(variant_cfg["config"]))
        variant_settings[str(variant)] = (
            str(method_value),
            {int(seed): Path(path) for seed, path in checkpoints_value.items()},
            base_config,
        )

    missing = [
        str(path)
        for _, checkpoints, _ in variant_settings.values()
        for path in checkpoints.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing selected checkpoint(s): " + ", ".join(missing))

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Fixed-actor evaluation with independent episode seeds.",
        "# No training commands. Completed non-empty CSV files are skipped on resume.",
    ]
    evaluation_command_count = 0
    for variant, (method, checkpoints, base_config) in variant_settings.items():
        if not base_config.is_file():
            raise FileNotFoundError(f"Missing variant config: {base_config}")
        for scenario in evaluation["scenarios"]:
            scenario_name = str(scenario)
            generated_config = config_root / f"{variant}_{scenario_name}.yaml"
            write_scenario_config(base_config, scenario_name, generated_config)
            for train_seed, checkpoint in checkpoints.items():
                for eval_seed in evaluation["seeds"]:
                    output = (
                        output_root
                        / scenario_name
                        / str(variant)
                        / f"train_seed_{train_seed}"
                        / f"eval_seed_{int(eval_seed)}.csv"
                    )
                    command = shell_join(
                        command_prefix
                        + [
                            "scripts/evaluate.py",
                            "--config",
                            str(generated_config),
                            "--method",
                            method,
                            "--checkpoint",
                            str(checkpoint),
                            "--episodes",
                            str(int(evaluation["episodes_per_seed"])),
                            "--seed",
                            str(int(eval_seed)),
                            "--output",
                            str(output),
                        ]
                    )
                    quoted_output = shlex.quote(str(output))
                    expected_lines = int(evaluation["episodes_per_seed"]) + 1
                    lines.append(
                        f"if [[ -s {quoted_output} ]] && "
                        f"[[ $(wc -l < {quoted_output}) -eq {expected_lines} ]]; "
                        f"then echo 'skip: {output}'; else {command}; fi"
                    )
                    evaluation_command_count += 1

    lines.extend(
        [
            "",
            shell_join(
                command_prefix
                + [
                    "scripts/summarize_fixed_actor_eval.py",
                    "--matrix",
                    str(matrix_path),
                ]
            ),
        ]
    )
    output_path = command_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_path.chmod(0o755)
    repo_root = Path(__file__).resolve().parents[1]
    generated_configs = sorted(config_root.glob("*.yaml"))
    checkpoint_paths = sorted(
        {path for _, checkpoints, _ in variant_settings.values() for path in checkpoints.values()}
    )
    plan_manifest = {
        "schema_version": "restart_eval_plan_v1",
        "matrix": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "commands": str(output_path),
        "commands_sha256": sha256_file(output_path),
        "expected_commands": evaluation_command_count,
        "source_files_sha256": _source_hashes(repo_root),
        "generated_config_sha256": {
            str(path): sha256_file(path) for path in generated_configs
        },
        "checkpoint_sha256": {
            str(path): sha256_file(path) for path in checkpoint_paths
        },
    }
    plan_path = output_root / "evaluation_plan_manifest.json"
    plan_path.write_text(json.dumps(plan_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {evaluation_command_count} evaluation commands: {output_path}")
    print(f"wrote evaluation plan manifest: {plan_path}")


if __name__ == "__main__":
    main()
