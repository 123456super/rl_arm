from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rl_risk_sac.algorithms import SACAgent
from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.seeding import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run no-hardware checks for deployment candidates.")
    parser.add_argument("--config", default="configs/experiments/random_crossing_link_fixed_penalty1.yaml")
    parser.add_argument(
        "--selected-checkpoints",
        default="outputs/rechecks/link_fixed_penalty1/eval/checkpoint_selection/selected_checkpoints.csv",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", default="outputs/deployment_preflight/offline_report.json")
    return parser.parse_args()


def check_config(config: dict[str, Any]) -> list[str]:
    failures = []
    if float(config["sac"]["fixed_risk_penalty"]) != 1.0:
        failures.append("sac.fixed_risk_penalty must be 1.0 for the deployment candidate")
    if not 0.0 < float(config["env"]["action_scale"]):
        failures.append("env.action_scale must be positive")
    smoothing_mode = str(config["env"]["execution"]["fixed_smoothing_mode"])
    if smoothing_mode == "ema" and not 0.0 <= float(config["env"]["fixed_beta"]) <= 1.0:
        failures.append("env.fixed_beta must be in [0, 1] when fixed_smoothing_mode=ema")
    if float(config["env"]["execution"]["rtb"]["cutoff_angular_frequency"]) <= 0.0:
        failures.append("env.execution.rtb.cutoff_angular_frequency must be positive")
    if not 0.0 < float(config["risk"]["d_safe"]):
        failures.append("risk.d_safe must be positive")
    for axis, limits in config["env"]["workspace"].items():
        if len(limits) != 2 or not float(limits[0]) < float(limits[1]):
            failures.append(f"env.workspace.{axis} must be an increasing [min, max] pair")
    return failures


def load_selected_checkpoints(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    expected_seeds = {"101", "202", "303"}
    actual_seeds = {row["train_seed"] for row in rows}
    if actual_seeds != expected_seeds:
        raise ValueError(f"Expected train seeds {sorted(expected_seeds)}, got {sorted(actual_seeds)}")
    for row in rows:
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    return rows


def run_candidate(config: dict[str, Any], checkpoint: Path, seed: int, steps: int) -> dict[str, float | int | bool]:
    set_seed(seed)
    env = UR5DynamicObstacleEnv(config, method="link_fixed")
    try:
        agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], config, method="link_fixed")
        agent.load_actor(checkpoint)
        observation, _ = env.reset(seed=seed)
        max_action = 0.0
        max_command = 0.0
        finite = bool(np.isfinite(observation).all())
        d_min = float("inf")
        completed_steps = 0
        for _ in range(steps):
            action = agent.select_action(observation, deterministic=True)
            max_action = max(max_action, float(np.max(np.abs(action))))
            observation, _, _, terminated, truncated, info = env.step(action)
            finite = finite and bool(np.isfinite(observation).all()) and bool(np.isfinite(info["qdot_cmd"]).all())
            max_command = max(max_command, float(np.max(np.abs(info["qdot_cmd"]))))
            d_min = min(d_min, float(info["d_min"]))
            completed_steps += 1
            if terminated or truncated:
                break
        if max_action > 1.0 + 1e-6:
            raise RuntimeError(f"Actor action exceeds normalized range: {max_action}")
        if max_command > float(config["env"]["action_scale"]) + 1e-6:
            raise RuntimeError(f"Joint command exceeds action_scale: {max_command}")
        if not finite:
            raise RuntimeError("Non-finite inference or command value")
        return {
            "steps": completed_steps,
            "max_normalized_action": max_action,
            "max_abs_qdot_cmd_rad_s": max_command,
            "minimum_observed_distance_m": d_min,
            "finite_values": finite,
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["device"] = "cpu"
    config_failures = check_config(config)
    selected = load_selected_checkpoints(Path(args.selected_checkpoints))
    candidates = []
    failures = list(config_failures)
    for row in selected:
        try:
            result = run_candidate(config, Path(row["checkpoint"]), int(row["train_seed"]), args.steps)
            candidates.append({**row, "status": "pass", **result})
        except Exception as error:  # Keep an actionable report for every candidate.
            candidates.append({**row, "status": "fail", "error": str(error)})
            failures.append(f"seed {row['train_seed']}: {error}")

    report = {
        "offline_status": "pass" if not failures else "fail",
        "scope": "PyBullet-only inference, normalized-action and joint-command-limit checks; no real robot is connected.",
        "config": {
            "method": "link_fixed_penalty1",
            "fixed_risk_penalty": config["sac"]["fixed_risk_penalty"],
            "action_scale_rad_s": config["env"]["action_scale"],
            "fixed_smoothing_mode": config["env"]["execution"]["fixed_smoothing_mode"],
            "rtb_cutoff_angular_frequency_rad_s": config["env"]["execution"]["rtb"][
                "cutoff_angular_frequency"
            ],
            "fixed_beta": config["env"]["fixed_beta"],
            "d_safe_m": config["risk"]["d_safe"],
            "workspace_m": config["env"]["workspace"],
        },
        "candidates": candidates,
        "failures": failures,
        "required_on_site_signoff": [
            "Robot-specific joint-speed, acceleration and workspace limits are configured in the robot controller, not inferred from the simulation action scale.",
            "The physical emergency-stop and protective-stop path has been tested while the robot is in a safe state.",
            "RGB-D detection loss, invalid depth, stale obstacle state and excessive risk all command a safe stop before commands reach the robot.",
            "Camera-to-robot extrinsics and the conservative collision envelope have been verified on the physical cell.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"offline preflight {report['offline_status']}: {output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
