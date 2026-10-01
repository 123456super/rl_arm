from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config
from scripts.core.train_thesis_homotopy import fixed_task_space_bin_probabilities


def statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(np.quantile(values, .01)),
        "p99": float(np.quantile(values, .99)),
        "max_abs": float(np.max(np.abs(values))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate 9D keypoint error on the real S0 goal sampler."
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml",
    )
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=81003)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    config = load_config(ROOT / args.config)
    final_level = config["thesis"]["joint_pose_curriculum"]["levels"][-1]
    probabilities = fixed_task_space_bin_probabilities(config)
    env = ThesisHomotopyEnv(config)
    values = np.empty((args.samples, 9), dtype=np.float32)
    distances = np.empty((args.samples, 3), dtype=np.float32)
    try:
        env.configure_episode(
            "none", xi=1.0, strict=False, goal_scale=float(final_level["goal_scale"]),
            orientation_scale=1.0,
            position_tolerance=float(final_level["position_tolerance_m"]),
            orientation_tolerance=float(final_level["orientation_tolerance_rad"]),
            target_distance_min_m=float(final_level["target_distance_min_m"]),
            target_distance_max_m=float(final_level["target_distance_max_m"]),
            target_orientation_min_rad=float(final_level["target_orientation_min_rad"]),
            target_orientation_max_rad=float(final_level["target_orientation_max_rad"]),
            task_space_bin_probabilities=probabilities,
        )
        for index in range(args.samples):
            _, info = env.reset(seed=args.seed if index == 0 else None)
            error = np.asarray(info["keypoint_errors"], dtype=np.float32)
            values[index] = error.reshape(-1)
            distances[index] = np.linalg.norm(error, axis=1)
    finally:
        env.close()

    scale = float(config["thesis"]["observation"]["keypoint_error_scale_m"])
    report = {
        "config": args.config,
        "samples": args.samples,
        "seed": args.seed,
        "sampler": "S0 final-level real reset/goal sampler with fixed orientation mixture",
        "keypoint_error_scale_m": scale,
        "overall_signed_components_m": statistics(values),
        "overall_point_distances_m": statistics(distances),
        "fraction_components_clipped": float(np.mean(np.abs(values / scale) > 1.0)),
        "per_feature": [
            {"index": index, "keypoint": index // 3, "axis": index % 3,
             **statistics(values[:, index])}
            for index in range(values.shape[1])
        ],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "overall_signed_components_m": report["overall_signed_components_m"],
        "overall_point_distances_m": report["overall_point_distances_m"],
        "fraction_components_clipped": report["fraction_components_clipped"],
    }, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
