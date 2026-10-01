#!/usr/bin/env python3
"""Audit pre-clip observation features on frozen, seeded pose evaluations."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.algorithms.networks import GaussianActor
from rl_risk_sac.envs import thesis_homotopy_env as env_module
from rl_risk_sac.tasks.thesis_reaching import pose_keypoint_errors, keypoint_position_jacobian
from rl_risk_sac.utils.config import load_config


def preclip_blocks(k: dict) -> dict[str, np.ndarray]:
    q = np.asarray(k["q"])
    low, high = np.asarray(k["joint_lower"]), np.asarray(k["joint_upper"])
    geometry, self_geometry = k["geometry"], k["self_geometry"]
    blocks = {
        "q": 2 * (q - low) / (high - low) - 1,
        "qdot": np.asarray(k["qdot"]) / k["joint_velocity_scale"],
        "keypoint_error": pose_keypoint_errors(
            k["goal_position"], k["goal_quaternion"],
            k["ee_position"], k["ee_quaternion"],
            cube_side_m=k["keypoint_cube_side_m"],
        ).reshape(-1) / k["keypoint_error_scale_m"],
        "J_KP": np.asarray(k["keypoint_jacobian"] if k["keypoint_jacobian"] is not None
                           else keypoint_position_jacobian(k["ee_jacobian"], k["ee_quaternion"],
                                                           cube_side_m=k["keypoint_cube_side_m"]))
                .reshape(-1) / k["keypoint_jacobian_scale"],
        "position_error": np.asarray(k["position_error"]),
        "orientation_error": np.asarray(k["orientation_error"]) / np.pi,
        "rho_position": np.asarray([np.linalg.norm(k["position_error"])]),
        "rho_orientation": np.asarray([np.linalg.norm(k["orientation_error"]) / np.pi]),
        "EE_linear_velocity": np.asarray(k["ee_linear_velocity"]) / k["ee_linear_velocity_scale"],
        "EE_angular_velocity": np.asarray(k["ee_angular_velocity"]) / k["ee_angular_velocity_scale"],
        "goal_scale": np.asarray([2 * k["goal_scale"] - 1]),
        "position_tolerance": np.asarray([2 * k["position_tolerance"] / .20 - 1]),
        "orientation_tolerance": np.asarray([2 * k["orientation_tolerance"] / np.pi - 1]),
        "remaining_time": np.asarray([2 * k["remaining_time_fraction"] - 1]),
    }
    if k["obstacle_present"]:
        blocks.update({
            "obstacle_relative": np.asarray(geometry.relative_vectors).reshape(-1),
            "obstacle_position": 2 * (np.asarray(k["obstacle_position"]) - [0.22, -0.62, 0.18])
                                 / ([0.72, 0.62, 0.62] - np.asarray([0.22, -0.62, 0.18])) - 1,
            "obstacle_velocity": np.asarray(k["obstacle_velocity"]) / .1,
            "obstacle_distance": 2 * (np.asarray(geometry.distances) + .20) - 1,
            "obstacle_TTC": np.asarray(geometry.ttc) / 3,
            "obstacle_closing_velocity": np.asarray(geometry.approach_velocities),
            "obstacle_risk": np.asarray(geometry.risk_per_link),
            "obstacle_present": np.asarray([1.]),
        })
    else:
        blocks.update({
            "obstacle_relative": np.zeros(18), "obstacle_position": np.zeros(3),
            "obstacle_velocity": np.zeros(3), "obstacle_distance": np.ones(6),
            "obstacle_TTC": np.ones(6), "obstacle_closing_velocity": np.zeros(6),
            "obstacle_risk": np.zeros(6), "obstacle_present": np.zeros(1),
        })
    if self_geometry is not None:
        blocks.update({
            "self_distance": 2 * (np.asarray(self_geometry.distances) + .02) / .27 - 1,
            "self_TTC": np.asarray(self_geometry.ttc) / 3,
            "self_closing_velocity": np.asarray(self_geometry.approach_velocities),
            "self_risk": np.asarray(self_geometry.risk_per_link),
        })
    else:
        blocks.update({"self_distance": np.ones(6), "self_TTC": np.ones(6),
                       "self_closing_velocity": np.zeros(6), "self_risk": np.zeros(6)})
    return {name: np.asarray(value, dtype=np.float32).reshape(-1) for name, value in blocks.items()}


def run_worker(args: tuple) -> dict:
    checkpoint, config_path, output, seed, episodes, level_index, scene = args
    torch.set_num_threads(1)
    config = load_config(Path(config_path))
    config["device"] = "cpu"
    level = config["thesis"]["joint_pose_curriculum"]["levels"][level_index]
    eta = config["thesis"]["orientation_curriculum"]["levels"][level_index]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = (state["agent"]["actor"] if "agent" in state else
               state["actor"] if "actor" in state else state)
    actor = GaussianActor(162, 6, list(config["sac"]["hidden_dims"]))
    actor.load_state_dict(weights)
    actor.eval()
    measurements: dict[str, list[np.ndarray]] = {}
    original = env_module.build_thesis_observation

    def intercept(**kwargs):
        blocks = preclip_blocks(kwargs)
        for name, values in blocks.items():
            measurements.setdefault(name, []).append(values)
        return original(**kwargs)

    env_module.build_thesis_observation = intercept
    env = env_module.ThesisHomotopyEnv(config)
    position_bins = int(config["thesis"]["orientation_curriculum"]["deterministic_probe_position_bins"])
    orientation_bins = int(config["thesis"]["orientation_curriculum"]["deterministic_probe_orientation_bins"])
    position_edges = np.linspace(level["target_distance_min_m"], level["target_distance_max_m"], position_bins + 1)
    orientation_edges = np.linspace(level["target_orientation_min_rad"], level["target_orientation_max_rad"], orientation_bins + 1)
    outcomes = []
    try:
        for index in episodes:
            p_bin = (index // orientation_bins) % position_bins
            o_bin = index % orientation_bins
            env.configure_episode(
                scene, xi=1., strict=True, goal_scale=float(level["goal_scale"]),
                orientation_scale=float(eta), position_tolerance=float(level["position_tolerance_m"]),
                orientation_tolerance=float(level["orientation_tolerance_rad"]),
                target_distance_min_m=float(position_edges[p_bin]),
                target_distance_max_m=float(position_edges[p_bin + 1]),
                target_orientation_min_rad=float(orientation_edges[o_bin]),
                target_orientation_max_rad=float(orientation_edges[o_bin + 1]),
            )
            obs, _ = env.reset(seed=int(seed) + index)
            terminated = truncated = False
            while not (terminated or truncated):
                with torch.no_grad():
                    action = actor.deterministic(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
                obs, _, _, terminated, truncated, info = env.step(action)
            outcomes.append([index, int(info["task_reached"]), int(info["timeout"]),
                             int(info["obstacle_collision"] or info["self_collision"] or info["environment_collision"]),
                             int(info["joint_limit"]), env.step_count])
    finally:
        env.close()
        env_module.build_thesis_observation = original
    np.savez_compressed(output, **{key: np.stack(rows) for key, rows in measurements.items()},
                        outcomes=np.asarray(outcomes, dtype=np.int32))
    return {"file": str(output), "episodes": len(outcomes), "observations": len(next(iter(measurements.values())))}


def summarize(values: np.ndarray) -> dict:
    flat = values.reshape(-1).astype(np.float64)
    abs_values = np.abs(flat)
    return {
        "shape_per_observation": list(values.shape[1:]), "count": len(flat),
        "min": float(flat.min()), "max": float(flat.max()),
        "mean": float(flat.mean()), "std": float(flat.std()),
        "p95": float(np.percentile(flat, 95)), "p99": float(np.percentile(flat, 99)),
        "p99_5": float(np.percentile(flat, 99.5)), "p99_9": float(np.percentile(flat, 99.9)),
        "p99_abs": float(np.percentile(abs_values, 99)),
        "p99_9_abs": float(np.percentile(abs_values, 99.9)),
        "low_clip_ratio": float(np.mean(flat < -1)),
        "high_clip_ratio": float(np.mean(flat > 1)),
        "total_clip_ratio": float(np.mean(abs_values > 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument("--level-index", type=int, default=4)
    parser.add_argument("--scene", choices=("none", "static", "dynamic"), default="none")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1 or not 0 <= args.level_index <= 4:
        parser.error("episodes/workers must be positive and level-index must be in [0,4]")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(str(args.checkpoint.resolve()), str(args.config.resolve()),
              args.output / f"preclip_worker_{worker:02}.npz", args.seed,
              list(range(worker, args.episodes, args.workers)), args.level_index, args.scene)
             for worker in range(min(args.workers, args.episodes))]
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        results = list(executor.map(run_worker, tasks))
    files = [np.load(row["file"]) for row in results]
    try:
        blocks = {name: summarize(np.concatenate([item[name] for item in files]))
                  for name in files[0].files if name != "outcomes"}
        outcomes = np.concatenate([item["outcomes"] for item in files])
        outcomes = outcomes[np.argsort(outcomes[:, 0])]
    finally:
        for item in files:
            item.close()
    report = {"checkpoint": str(args.checkpoint), "config": str(args.config),
              "scene": args.scene, "level_index": args.level_index, "seed": args.seed,
              "episodes": args.episodes, "observations": sum(row["observations"] for row in results),
              "success_rate": float(outcomes[:, 1].mean()),
              "timeout_rate": float(outcomes[:, 2].mean()),
              "collision_rate": float(outcomes[:, 3].mean()),
              "joint_limit_rate": float(outcomes[:, 4].mean()),
              "mean_episode_steps": float(outcomes[:, 5].mean()), "blocks": blocks,
              "data_files": [row["file"] for row in results]}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("blocks", "data_files")}, indent=2), flush=True)
    for name, stats in blocks.items():
        print(f"{name:30} clip={stats['total_clip_ratio']:.3%} "
              f"low={stats['low_clip_ratio']:.3%} high={stats['high_clip_ratio']:.3%} "
              f"p99abs={stats['p99_abs']:.3f}", flush=True)


if __name__ == "__main__":
    main()
