#!/usr/bin/env python3
"""Compare thesis actors on identical deterministic random pose tasks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    path = Path(raw_path)
    return label, path if path.is_absolute() else ROOT / path


def parse_contract(value: str) -> tuple[str, dict[str, float]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "contract must be LABEL=GOAL_SCALE,ETA,POSITION_TOLERANCE,ORIENTATION_TOLERANCE"
        )
    label, raw_values = value.split("=", 1)
    try:
        values = [float(item) for item in raw_values.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("contract values must be numeric") from error
    if not label or len(values) != 4:
        raise argparse.ArgumentTypeError(
            "contract must be LABEL=GOAL_SCALE,ETA,POSITION_TOLERANCE,ORIENTATION_TOLERANCE"
        )
    goal_scale, orientation_scale, position_tolerance, orientation_tolerance = values
    if not 0.0 < goal_scale <= 1.0 or not 0.0 <= orientation_scale <= 1.0:
        raise argparse.ArgumentTypeError("contract goal scale or eta is out of range")
    if position_tolerance <= 0.0 or orientation_tolerance <= 0.0:
        raise argparse.ArgumentTypeError("contract tolerances must be positive")
    return label, {
        "goal_scale": goal_scale,
        "orientation_scale": orientation_scale,
        "position_tolerance": position_tolerance,
        "orientation_tolerance": orientation_tolerance,
    }


def wilson_interval(successes: int, count: int) -> tuple[float, float]:
    if count < 1:
        raise ValueError("Wilson interval requires at least one sample")
    z = 1.959963984540054
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    radius = z * np.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count)) / denominator
    return float(center - radius), float(center + radius)


def evaluate(
    pool: ParallelThesisEnvPool,
    agent: ThesisSACAgent,
    contract: dict[str, Any],
    *,
    episodes: int,
    seed: int,
    label: str,
    position_bins: int = 10,
    orientation_bins: int = 10,
    trajectory_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    num_envs = len(pool)
    active: dict[int, dict[str, Any]] = {}
    rows: dict[int, dict[str, Any]] = {}
    next_episode = 0
    next_report = 50
    started = time.perf_counter()
    position_edges = (
        np.linspace(
            contract["target_distance_min_m"],
            contract["target_distance_max_m"],
            position_bins + 1,
        )
        if "target_distance_max_m" in contract else None
    )
    orientation_edges = (
        np.linspace(
            contract["target_orientation_min_rad"],
            contract["target_orientation_max_rad"],
            orientation_bins + 1,
        )
        if "target_orientation_max_rad" in contract else None
    )

    def launch_idle() -> None:
        nonlocal next_episode
        assignments: dict[int, int] = {}
        for worker_index in range(num_envs):
            if worker_index in active or next_episode >= episodes:
                continue
            episode = next_episode
            next_episode += 1
            assignments[worker_index] = episode
            active[worker_index] = {
                "episode": episode,
                "length": 0,
                "minimum_self_distance_m": float("inf"),
            }
        if not assignments:
            return
        pool.set_rng_states({
            worker_index: np.random.default_rng(seed + episode).bit_generator.state
            for worker_index, episode in assignments.items()
        })
        episode_contracts = {}
        for worker_index, episode in assignments.items():
            episode_contract = dict(contract)
            if position_edges is not None and orientation_edges is not None:
                position_bin = (episode // orientation_bins) % position_bins
                orientation_bin = episode % orientation_bins
                episode_contract.update({
                    "target_distance_min_m": float(position_edges[position_bin]),
                    "target_distance_max_m": float(position_edges[position_bin + 1]),
                    "target_orientation_min_rad": float(orientation_edges[orientation_bin]),
                    "target_orientation_max_rad": float(orientation_edges[orientation_bin + 1]),
                })
            episode_contracts[worker_index] = episode_contract
        resets = pool.reset_many(episode_contracts)
        for worker_index, (observation, _) in resets.items():
            active[worker_index]["observation"] = observation

    while len(rows) < episodes:
        launch_idle()
        indices = sorted(active)
        observations = np.stack([active[index]["observation"] for index in indices])
        actions = agent.select_actions(observations, deterministic=True)
        results = pool.step_many(dict(zip(indices, actions)))
        finished = []
        for batch_index, index in enumerate(indices):
            next_observation, _, _, terminated, truncated, info = results[index]
            state = active[index]
            state["observation"] = next_observation
            state["length"] += 1
            if trajectory_rows is not None:
                trajectory_rows.append({
                    "checkpoint": label.split("/", 1)[0],
                    "contract_name": label.split("/", 1)[1],
                    "episode": int(state["episode"]),
                    "episode_seed": seed + int(state["episode"]),
                    "step": int(state["length"]),
                    **{
                        f"action_{dimension}": float(actions[batch_index, dimension])
                        for dimension in range(actions.shape[1])
                    },
                    "position_error_m": float(info["next_rho_position"]),
                    "orientation_error_rad": float(info["next_rho_orientation"]),
                    "velocity_magnitude": float(info["velocity_magnitude"]),
                    "minimum_self_distance_m": float(info["control_self_min_distance"]),
                    "task_reached": int(info["task_reached"]),
                    "collision": int(
                        info["obstacle_collision"]
                        or info["self_collision"]
                        or info["environment_collision"]
                    ),
                    "timeout": int(truncated),
                })
            state["minimum_self_distance_m"] = min(
                state["minimum_self_distance_m"],
                float(info["control_self_min_distance"]),
            )
            if not (terminated or truncated):
                continue
            episode = int(state["episode"])
            rows[episode] = {
                "episode": episode,
                "episode_seed": seed + episode,
                "safe_success": int(info["safe_success"]),
                "collision": int(
                    info["obstacle_collision"]
                    or info["self_collision"]
                    or info["environment_collision"]
                ),
                "self_collision": int(info["self_collision"]),
                "joint_limit": int(info["joint_limit"]),
                "timeout": int(truncated),
                "length": int(state["length"]),
                "position_error_m": float(info["next_rho_position"]),
                "orientation_error_rad": float(info["next_rho_orientation"]),
                "minimum_self_distance_m": float(state["minimum_self_distance_m"]),
            }
            finished.append(index)
        for index in finished:
            del active[index]
        if len(rows) >= next_report:
            print(
                f"{label}: {len(rows)}/{episodes} episodes "
                f"in {time.perf_counter() - started:.1f}s",
                flush=True,
            )
            next_report += 50
    return [rows[index] for index in range(episodes)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    successes = sum(int(row["safe_success"]) for row in rows)
    low, high = wilson_interval(successes, count)
    summary: dict[str, Any] = {
        "episodes": count,
        "successes": successes,
        "safe_success_rate": successes / count,
        "safe_success_wilson95_low": low,
        "safe_success_wilson95_high": high,
    }
    for key in ("collision", "self_collision", "joint_limit", "timeout"):
        summary[f"{key}_rate"] = float(np.mean([row[key] for row in rows]))
    for key in (
        "length",
        "position_error_m",
        "orientation_error_rad",
        "minimum_self_distance_m",
    ):
        summary[f"mean_{key}"] = float(np.mean([row[key] for row in rows]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple thesis checkpoints on identical pose tasks"
    )
    parser.add_argument("--config", default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--levels", nargs="+", type=int)
    parser.add_argument(
        "--contract",
        action="append",
        type=parse_contract,
        help=(
            "custom LABEL=GOAL_SCALE,ETA,POSITION_TOLERANCE,ORIENTATION_TOLERANCE; "
            "may be repeated"
        ),
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=51001)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--save-trajectories", action="store_true",
        help="write every control step to trajectories.csv for paired drift diagnosis",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.num_envs < 1:
        parser.error("episodes and num-envs must be positive")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    probe_position_bins = int(config["thesis"]["orientation_curriculum"].get(
        "deterministic_probe_position_bins", 10
    ))
    probe_orientation_bins = int(config["thesis"]["orientation_curriculum"].get(
        "deterministic_probe_orientation_bins", 10
    ))
    pose_levels = config["thesis"]["joint_pose_curriculum"]["levels"]
    orientation_levels = config["thesis"]["orientation_curriculum"]["levels"]
    levels = (
        [len(pose_levels) - 2, len(pose_levels) - 1]
        if args.levels is None and not args.contract else (args.levels or [])
    )
    if any(level < 0 or level >= len(pose_levels) for level in levels):
        parser.error(f"levels must be in [0, {len(pose_levels) - 1}]")
    labels = [label for label, _ in args.checkpoint]
    if len(labels) != len(set(labels)):
        parser.error("checkpoint labels must be unique")
    contracts: list[tuple[str, int | None, dict[str, Any]]] = []
    for level_index in levels:
        level = pose_levels[level_index]
        sampling_contract = {
            key: float(level[key])
            for key in (
                "target_distance_min_m",
                "target_distance_max_m",
                "target_orientation_min_rad",
                "target_orientation_max_rad",
            )
            if key in level
        }
        contracts.append((
            f"L{level_index}",
            level_index,
            {
                "goal_scale": float(level["goal_scale"]),
                "orientation_scale": float(orientation_levels[level_index]),
                "position_tolerance": float(level["position_tolerance_m"]),
                "orientation_tolerance": float(level["orientation_tolerance_rad"]),
                **sampling_contract,
            },
        ))
    contracts.extend(
        (label, None, values) for label, values in (args.contract or [])
    )
    contract_labels = [label for label, _, _ in contracts]
    if not contracts:
        parser.error("at least one level or custom contract is required")
    if len(contract_labels) != len(set(contract_labels)):
        parser.error("level and custom contract labels must be unique")

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=False)

    template = ThesisHomotopyEnv(config)
    observation_dim = int(template.observation_space.shape[0])
    action_dim = int(template.action_space.shape[0])
    template.close()
    all_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] | None = [] if args.save_trajectories else None
    summaries: dict[str, Any] = {}
    worker_seeds = [args.seed + 1_000_000 + index for index in range(args.num_envs)]

    with ParallelThesisEnvPool(config, worker_seeds) as pool:
        for checkpoint_label, checkpoint_path in args.checkpoint:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            agent = ThesisSACAgent(observation_dim, action_dim, config)
            if "agent" in checkpoint:
                if checkpoint.get("protocol") != config["thesis"]["protocol"]:
                    raise ValueError(f"protocol mismatch for {checkpoint_path}")
                agent.load_state_dict(checkpoint["agent"])
            else:
                # Actor-only snapshots are intentionally protocol-agnostic.
                # Shape validation in load_state_dict still rejects incompatible
                # observation, action, or network contracts.
                agent.actor.load_state_dict(checkpoint)
            del checkpoint
            summaries[checkpoint_label] = {}
            for contract_label, level_index, values in contracts:
                contract = {
                    "scene": "none",
                    "xi": 1.0,
                    "strict": True,
                    "lambda_self": 1.0,
                    **values,
                }
                label = f"{checkpoint_label}/{contract_label}"
                rows = evaluate(
                    pool,
                    agent,
                    contract,
                    episodes=args.episodes,
                    seed=args.seed,
                    label=label,
                    position_bins=probe_position_bins,
                    orientation_bins=probe_orientation_bins,
                    trajectory_rows=trajectory_rows,
                )
                for row in rows:
                    all_rows.append({
                        "checkpoint": checkpoint_label,
                        "contract_name": contract_label,
                        "level": "" if level_index is None else level_index,
                        **contract,
                        **row,
                    })
                summaries[checkpoint_label][contract_label] = {
                    "contract": contract,
                    **summarize(rows),
                }
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    if trajectory_rows:
        with (output / "trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
            writer.writeheader()
            writer.writerows(trajectory_rows)
    result = {
        "evaluation_seed": args.seed,
        "episodes_per_checkpoint_level": args.episodes,
        "num_envs": args.num_envs,
        "deterministic_actions": True,
        "checkpoints": {label: str(path) for label, path in args.checkpoint},
        "results": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
