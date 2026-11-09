#!/usr/bin/env python3
"""Deterministically evaluate an S0, S1, or S2 thesis PPO checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from rl_risk_sac.algorithms.thesis_ppo import ThesisPPOAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.tasks.reference_reaching import residual_action_penalty
from rl_risk_sac.utils.config import deep_update, load_config


ROOT = Path(__file__).resolve().parents[2]


def episode_contract(scene: str) -> dict[str, object]:
    return {"scene": scene, "terminate_on_success": False}


def scene_schedule(
    config: dict, stage: str, selection: str, episodes: int, seed: int,
) -> list[str]:
    if selection != "mixture":
        return [selection] * episodes
    probabilities = config["thesis"]["scene_mixture"][stage]
    scenes = tuple(probabilities)
    weights = np.asarray([probabilities[item] for item in scenes], dtype=np.float64)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    return [str(item) for item in rng.choice(scenes, size=episodes, p=weights)]


def new_stats(info: dict) -> dict[str, object]:
    reference_error = float(
        0.5 * np.asarray(info["reference_point_distances"], dtype=np.float64).sum()
        + 0.25 * float(info["orientation_error_norm"])
    )
    return {
        "return": 0.0,
        "length": 0,
        "obstacle_count": int(info.get("obstacle_count", 0)),
        "obstacle_layout": str(info.get("obstacle_layout", "none")),
        "success": False,
        "collision": False,
        "obstacle_collision": False,
        "self_collision": False,
        "environment_collision": False,
        "joint_limit": False,
        "initial_position_m": float(info["goal_error_norm"]),
        "initial_orientation_rad": float(info["orientation_error_norm"]),
        "best_position_m": float(info["goal_error_norm"]),
        "best_orientation_rad": float(info["orientation_error_norm"]),
        "best_reference_error": reference_error,
        "position_at_best_reference_m": float(info["goal_error_norm"]),
        "orientation_at_best_reference_rad": float(info["orientation_error_norm"]),
        "action_abs_sum": 0.0,
        "action_abs_max": 0.0,
        "residual_penalty_sum": 0.0,
        "executed_substeps_min": 10**9,
    }


def metric(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "percentiles": {
            str(q): float(np.percentile(array, q))
            for q in (0, 5, 10, 25, 50, 75, 90, 95, 100)
        },
    }


def rates(rows: list[dict[str, object]]) -> dict[str, float]:
    collision = np.asarray([bool(row["collision"]) for row in rows], dtype=bool)
    pose10 = np.asarray([
        float(row["position_at_best_reference_m"]) <= 0.01
        and float(row["orientation_at_best_reference_rad"]) <= 0.10
        for row in rows
    ])
    pose5 = np.asarray([
        float(row["position_at_best_reference_m"]) <= 0.005
        and float(row["orientation_at_best_reference_rad"]) <= 0.05
        for row in rows
    ])
    return {
        "success": float(np.mean([bool(row["success"]) for row in rows])),
        "collision": float(collision.mean()),
        "obstacle_collision": float(np.mean([
            bool(row["obstacle_collision"]) for row in rows
        ])),
        "self_collision": float(np.mean([
            bool(row["self_collision"]) for row in rows
        ])),
        "environment_collision": float(np.mean([
            bool(row["environment_collision"]) for row in rows
        ])),
        "joint_limit": float(np.mean([bool(row["joint_limit"]) for row in rows])),
        "collision_free": float((~collision).mean()),
        "best_pose_10mm_0.1rad": float(pose10.mean()),
        "best_pose_5mm_0.05rad": float(pose5.mean()),
        "safe_pose_10mm_0.1rad": float((pose10 & ~collision).mean()),
        "safe_pose_5mm_0.05rad": float((pose5 & ~collision).mean()),
    }


def grouped_report(rows: list[dict[str, object]], key: str) -> dict[str, object]:
    result = {}
    for value in sorted({str(row[key]) for row in rows}):
        selected = [row for row in rows if str(row[key]) == value]
        result[value] = {
            "episodes": len(selected),
            "return": metric([float(row["return"]) for row in selected]),
            "best_position_m": metric([
                float(row["best_position_m"]) for row in selected
            ]),
            "best_orientation_rad": metric([
                float(row["best_orientation_rad"]) for row in selected
            ]),
            "rates": rates(selected),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=41001)
    parser.add_argument("--stage", choices=("s0", "s1", "s2"))
    parser.add_argument(
        "--scene", choices=("mixture", "none", "static", "dynamic"),
        default="mixture",
    )
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.num_envs < 1:
        parser.error("episodes and num-envs must be positive")

    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "config" not in payload:
        raise ValueError("checkpoint must contain its training config")
    saved_config = deepcopy(payload["config"])
    checkpoint_stage = str(
        payload.get("stage", saved_config["thesis"].get("stage", ""))
    ).lower()
    stage = args.stage or checkpoint_stage
    if stage not in {"s0", "s1", "s2"}:
        raise ValueError(f"invalid checkpoint stage {stage!r}")
    # Historical checkpoints predate centralized S2/default fields. Merge
    # their authoritative experiment values over today's complete schema.
    base_path = (
        "configs/experiments/paper_s2_lstm_sweep_single.yaml"
        if stage == "s2"
        else "configs/experiments/paper_full.yaml"
    )
    config = load_config(ROOT / base_path)
    deep_update(config, saved_config)
    if args.device:
        config["device"] = args.device
    config["seed"] = args.seed
    config["thesis"]["stage"] = stage
    config["thesis"]["simple_reaching_mode"] = stage == "s0"
    config["thesis"]["blocking_obstacle_mode"] = stage != "s0"
    scheduled_scenes = scene_schedule(
        config, stage, args.scene, args.episodes, args.seed
    )
    if stage == "s0" and any(scene != "none" for scene in scheduled_scenes):
        raise ValueError("S0 evaluation supports only obstacle-free scenes")
    if any(scene == "dynamic" for scene in scheduled_scenes):
        prediction = config["thesis"]["dynamic_prediction"]
        if not prediction["enabled"] or not prediction["checkpoint"]:
            raise ValueError("dynamic evaluation requires an enabled LSTM checkpoint")

    from rl_risk_sac.envs.thesis_homotopy_env_reduced_observation import (
        ThesisHomotopyEnv,
    )

    template = ThesisHomotopyEnv(config)
    obs_dim = int(template.observation_space.shape[0])
    action_dim = int(template.action_space.shape[0])
    template.close()
    agent = ThesisPPOAgent(obs_dim, action_dim, config)
    agent.load_state_dict(payload["agent"] if "agent" in payload else payload)

    affinity = bool(config.get("train", {}).get("cpu_affinity", False))
    configured_cpu_ids = config.get("train", {}).get("cpu_ids")
    cpu_ids = None
    if affinity and configured_cpu_ids is not None:
        cpu_ids = [int(value) for value in configured_cpu_ids[: args.num_envs]]
        if len(cpu_ids) != args.num_envs:
            cpu_ids = None
    worker_seeds = [args.seed + index * 9973 for index in range(args.num_envs)]
    info_keys = (
        "collision", "obstacle_collision", "self_collision",
        "environment_collision", "joint_limit", "success",
        "obstacle_count", "obstacle_layout", "reference_error",
        "next_rho_position", "next_rho_orientation", "timeout",
        "termination_reason", "d_min", "goal_error_norm", "executed_substeps",
    )
    rows: list[dict[str, object]] = []
    next_episode_id = 0
    active_episode_ids: dict[int, int] = {}
    observations = np.zeros((args.num_envs, obs_dim), dtype=np.float32)
    stats: dict[int, dict[str, object]] = {}
    reward_config = config["thesis"]["reward"]
    started = perf_counter()
    with ParallelThesisEnvPool(
        config, worker_seeds, cpu_ids=cpu_ids, step_info_keys=info_keys
    ) as pool:
        requests = {}
        reset_seeds = {}
        initial_workers = min(args.num_envs, args.episodes)
        for worker in range(initial_workers):
            active_episode_ids[worker] = next_episode_id
            requests[worker] = episode_contract(scheduled_scenes[next_episode_id])
            reset_seeds[worker] = args.seed + next_episode_id
            next_episode_id += 1
        resets = pool.reset_many(requests, seeds=reset_seeds)
        for worker, (observation, info) in resets.items():
            observations[worker] = observation
            stats[worker] = new_stats(info)

        while len(rows) < args.episodes:
            workers = tuple(active_episode_ids)
            batch_observations = observations[list(workers)]
            actions, _logp, _values, policy = agent.act(
                batch_observations, deterministic=True
            )
            results = pool.step_many({
                worker: actions[index] for index, worker in enumerate(workers)
            })
            reset_requests = {}
            reset_seeds = {}
            for index, worker in enumerate(workers):
                observation, reward, _cost, terminated, truncated, info = results[worker]
                observations[worker] = observation
                residual_penalty, _ = residual_action_penalty(
                    policy["residual_actions"][index],
                    float(info["goal_error_norm"]),
                    float(info["d_min"]),
                    penalty_weight=float(reward_config["reference_residual_penalty_weight"]),
                    goal_start_m=float(reward_config["reference_residual_penalty_goal_start_m"]),
                    goal_full_m=float(reward_config["reference_residual_penalty_goal_full_m"]),
                    clearance_start_m=float(reward_config["reference_residual_penalty_clearance_start_m"]),
                    clearance_full_m=float(reward_config["reference_residual_penalty_clearance_full_m"]),
                )
                adjusted_reward = float(reward) - residual_penalty
                stat = stats[worker]
                stat["return"] = float(stat["return"]) + adjusted_reward
                stat["length"] = int(stat["length"]) + 1
                stat["residual_penalty_sum"] = (
                    float(stat["residual_penalty_sum"]) + residual_penalty
                )
                stat["executed_substeps_min"] = min(
                    int(stat["executed_substeps_min"]), int(info["executed_substeps"])
                )
                for key in (
                    "collision", "obstacle_collision", "self_collision",
                    "environment_collision", "joint_limit", "success",
                ):
                    stat[key] = bool(stat[key]) or bool(info[key])
                position = float(info["next_rho_position"])
                orientation = float(info["next_rho_orientation"])
                reference = float(info["reference_error"])
                stat["best_position_m"] = min(float(stat["best_position_m"]), position)
                stat["best_orientation_rad"] = min(
                    float(stat["best_orientation_rad"]), orientation
                )
                if reference < float(stat["best_reference_error"]):
                    stat["best_reference_error"] = reference
                    stat["position_at_best_reference_m"] = position
                    stat["orientation_at_best_reference_rad"] = orientation
                stat["action_abs_sum"] = float(stat["action_abs_sum"]) + float(
                    np.mean(np.abs(actions[index]))
                )
                stat["action_abs_max"] = max(
                    float(stat["action_abs_max"]),
                    float(np.max(np.abs(actions[index]))),
                )
                if terminated or truncated:
                    episode_id = active_episode_ids.pop(worker)
                    rows.append({
                        "episode": episode_id + 1,
                        "seed": args.seed + episode_id,
                        "scene": scheduled_scenes[episode_id],
                        **stat,
                        "mean_abs_action": float(stat["action_abs_sum"]) / int(stat["length"]),
                        "final_position_m": position,
                        "final_orientation_rad": orientation,
                        "final_reference_error": reference,
                        "termination_reason": info["termination_reason"],
                    })
                    if next_episode_id < args.episodes:
                        active_episode_ids[worker] = next_episode_id
                        reset_requests[worker] = episode_contract(
                            scheduled_scenes[next_episode_id]
                        )
                        reset_seeds[worker] = args.seed + next_episode_id
                        next_episode_id += 1
            if reset_requests:
                resets = pool.reset_many(reset_requests, seeds=reset_seeds)
                for worker, (observation, info) in resets.items():
                    observations[worker] = observation
                    stats[worker] = new_stats(info)

    rows.sort(key=lambda row: int(row["episode"]))
    numeric_columns = (
        "return", "initial_position_m", "initial_orientation_rad",
        "best_position_m", "best_orientation_rad", "best_reference_error",
        "position_at_best_reference_m", "orientation_at_best_reference_rad",
        "final_position_m", "final_orientation_rad", "final_reference_error",
        "mean_abs_action", "action_abs_max", "residual_penalty_sum",
        "executed_substeps_min",
    )
    report = {
        "kind": "deterministic_thesis_ppo_evaluation",
        "stage": stage,
        "scene_selection": args.scene,
        "episodes": args.episodes,
        "elapsed_s": perf_counter() - started,
        "seed_base": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload.get("global_step", payload.get("step", 0))),
        "scene_mixture": config["thesis"]["scene_mixture"][stage],
        "metrics": {
            name: metric([float(row[name]) for row in rows])
            for name in numeric_columns
        },
        "rates": rates(rows),
        "by_scene": grouped_report(rows, "scene"),
        "by_obstacle_count": grouped_report(rows, "obstacle_count"),
        "by_obstacle_layout": grouped_report(rows, "obstacle_layout"),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".csv").exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with output.with_suffix(".csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"saved summary: {output}", flush=True)
    print(f"saved episodes: {output.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()
