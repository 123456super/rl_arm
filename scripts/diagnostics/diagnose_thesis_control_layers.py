#!/usr/bin/env python3
"""Evaluate the observable DLS nominal controller independently of SAC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config


def run(env: ThesisHomotopyEnv, contract: dict[str, float], episodes: int, seed: int,
        residual: str) -> dict[str, float]:
    successes = collisions = timeouts = joint_limits = 0
    projected_steps = infeasible_steps = total_steps = 0
    lengths: list[int] = []
    final_position: list[float] = []
    final_orientation: list[float] = []
    minimum_self_distance: list[float] = []
    for episode in range(episodes):
        env.configure_episode("none", xi=1.0, strict=True, lambda_self=0.2, **contract)
        env.reset(seed=seed + episode)
        done = False
        length = 0
        min_self = float("inf")
        while not done:
            if residual == "zero":
                action = np.zeros(6, dtype=np.float32)
            elif residual == "random":
                rng = np.random.default_rng(seed + 100000 + episode * 1000 + length)
                action = rng.uniform(-1.0, 1.0, size=6).astype(np.float32)
            else:
                raise ValueError(f"unknown residual mode {residual!r}")
            _, _, _, terminated, truncated, info = env.step(action)
            length += 1
            min_self = min(min_self, float(info["control_self_min_distance"]))
            projected_steps += int(info["self_projection_intervened"])
            infeasible_steps += int(info["self_projection_infeasible"])
            total_steps += 1
            done = bool(terminated or truncated)
        successes += int(info["safe_success"])
        collisions += int(info["collision"])
        timeouts += int(truncated)
        joint_limits += int(info["joint_limit"])
        lengths.append(length)
        final_position.append(float(info["next_rho_position"]))
        final_orientation.append(float(info["next_rho_orientation"]))
        minimum_self_distance.append(min_self)
    count = float(episodes)
    return {
        "episodes": episodes,
        "safe_success_rate": successes / count,
        "collision_rate": collisions / count,
        "timeout_rate": timeouts / count,
        "joint_limit_rate": joint_limits / count,
        "projection_intervention_step_rate": projected_steps / max(total_steps, 1),
        "projection_infeasible_step_rate": infeasible_steps / max(total_steps, 1),
        "mean_length": float(np.mean(lengths)),
        "mean_final_position_error_m": float(np.mean(final_position)),
        "mean_final_orientation_error_rad": float(np.mean(final_orientation)),
        "mean_minimum_self_distance_m": float(np.mean(minimum_self_distance)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    parser.add_argument("--levels", nargs="+", type=int, default=[15, 20, 24])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=91001)
    parser.add_argument("--residuals", nargs="+", choices=("zero", "random"),
                        default=["zero", "random"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    levels = config["thesis"]["joint_pose_curriculum"]["levels"]
    result: dict[str, object] = {
        "protocol": config["thesis"]["protocol"],
        "episodes_per_level": args.episodes,
        "seed": args.seed,
        "results": {},
    }
    env = ThesisHomotopyEnv(config)
    try:
        for level_index in args.levels:
            level = levels[level_index]
            contract = {
                "goal_scale": float(level["goal_scale"]),
                "orientation_scale": 1.0,
                "position_tolerance": float(level["position_tolerance_m"]),
                "orientation_tolerance": float(level["orientation_tolerance_rad"]),
            }
            result["results"][f"L{level_index}"] = {
                mode: run(env, contract, args.episodes, args.seed, mode)
                for mode in args.residuals
            }
    finally:
        env.close()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
