from copy import deepcopy
import json
from time import perf_counter

import numpy as np
import torch

from rl_risk_sac.algorithms.thesis_ppo import ThesisPPOAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.utils.config import deep_update, load_config


CHECKPOINT = (
    "outputs/base_models/"
    "s0_base_cb31_10deg_125hz_horizon150_pose170_30_95/s0_base.pt"
)
OUTPUT = "tmp/s0_base_scale_paired_1000.json"
SEED_START = 41001
EPISODES = 1000
HORIZON = 150
SCALES = (1.0, 0.75)
BATCH_SIZE = 16


def evaluate_scale(payload, source_config, scale):
    config = deepcopy(source_config)
    config["device"] = "cuda:0"
    config["thesis"]["stage"] = "s0"
    config["thesis"]["simple_reaching_mode"] = True
    config["thesis"]["blocking_obstacle_mode"] = False
    agent = ThesisPPOAgent(int(payload["agent"]["obs_dim"]), 6, config)
    agent.load_state_dict(payload["agent"])
    rows = []
    started = perf_counter()
    info_keys = (
        "collision", "obstacle_collision", "self_collision",
        "environment_collision", "success", "goal_error_norm",
        "orientation_error_norm", "next_rho_position",
        "next_rho_orientation", "reference_error",
        "joint_speed_peak_rad_s", "joint_acceleration_peak_rad_s2",
        "joint_velocity_limiter_fraction",
        "joint_acceleration_limiter_fraction",
    )
    def initial_stat(seed, info):
        return {
                    "seed": seed,
                    "step": 0,
                    "return": 0.0,
                    "collision": False,
                    "self_collision": False,
                    "environment_collision": False,
                    "success": False,
                    "pose5": False,
                    "pose10": False,
                    "first_pose5_step": None,
                    "best_reference_error": float("inf"),
                    "position_at_best_reference_m": float(info["goal_error_norm"]),
                    "orientation_at_best_reference_rad": float(
                        info["orientation_error_norm"]
                    ),
                    "best_position_m": float(info["goal_error_norm"]),
                    "best_orientation_rad": float(info["orientation_error_norm"]),
                    "final_position_m": float(info["goal_error_norm"]),
                    "final_orientation_rad": float(info["orientation_error_norm"]),
                    "joint_speed_peak_rad_s": 0.0,
                    "joint_acceleration_peak_rad_s2": 0.0,
                    "velocity_limiter_sum": 0.0,
                    "acceleration_limiter_sum": 0.0,
                }

    worker_count = min(BATCH_SIZE, EPISODES)
    cpu_ids = config["train"].get("cpu_ids", [])[:worker_count] or None
    worker_seeds = [SEED_START + index * 9973 for index in range(worker_count)]
    with ParallelThesisEnvPool(
        config, worker_seeds, cpu_ids=cpu_ids, step_info_keys=info_keys
    ) as pool:
        active = {worker: worker for worker in range(worker_count)}
        next_episode = worker_count
        observations = np.zeros(
            (worker_count, int(payload["agent"]["obs_dim"])), dtype=np.float32
        )
        stats = {}
        resets = pool.reset_many(
            {
                worker: {"scene": "none", "terminate_on_success": False}
                for worker in active
            },
            seeds={
                worker: SEED_START + episode
                for worker, episode in active.items()
            },
        )
        for worker, (observation, info) in resets.items():
            observations[worker] = observation
            stats[worker] = initial_stat(
                SEED_START + active[worker], info
            )

        while active:
                workers = tuple(active)
                batch_observations = observations[list(workers)]
                actions = float(scale) * agent.predict_base_actions(
                    batch_observations
                )
                results = pool.step_many(
                    {
                        worker: actions[index]
                        for index, worker in enumerate(workers)
                    }
                )
                reset_requests = {}
                reset_seeds = {}
                for worker, (
                    next_observation, reward, _cost, _terminated, _truncated, info
                ) in results.items():
                    observations[worker] = next_observation
                    stat = stats[worker]
                    stat["step"] += 1
                    position = float(info["next_rho_position"])
                    orientation = float(info["next_rho_orientation"])
                    reference = float(info["reference_error"])
                    stat["return"] += float(reward)
                    stat["collision"] |= bool(info["collision"])
                    stat["self_collision"] |= bool(info["self_collision"])
                    stat["environment_collision"] |= bool(
                        info["environment_collision"]
                    )
                    stat["success"] |= bool(info["success"])
                    stat["best_position_m"] = min(stat["best_position_m"], position)
                    stat["best_orientation_rad"] = min(
                        stat["best_orientation_rad"], orientation
                    )
                    if position <= 0.005 and orientation <= 0.05:
                        stat["pose5"] = True
                        if stat["first_pose5_step"] is None:
                            stat["first_pose5_step"] = stat["step"]
                    if position <= 0.01 and orientation <= 0.10:
                        stat["pose10"] = True
                    if reference < stat["best_reference_error"]:
                        stat["best_reference_error"] = reference
                        stat["position_at_best_reference_m"] = position
                        stat["orientation_at_best_reference_rad"] = orientation
                    stat["final_position_m"] = position
                    stat["final_orientation_rad"] = orientation
                    stat["joint_speed_peak_rad_s"] = max(
                        stat["joint_speed_peak_rad_s"],
                        float(info["joint_speed_peak_rad_s"]),
                    )
                    stat["joint_acceleration_peak_rad_s2"] = max(
                        stat["joint_acceleration_peak_rad_s2"],
                        float(info["joint_acceleration_peak_rad_s2"]),
                    )
                    stat["velocity_limiter_sum"] += float(
                        info["joint_velocity_limiter_fraction"]
                    )
                    stat["acceleration_limiter_sum"] += float(
                        info["joint_acceleration_limiter_fraction"]
                    )
                    if stat["step"] >= HORIZON:
                        stat["velocity_limiter_mean"] = stat.pop(
                            "velocity_limiter_sum"
                        ) / HORIZON
                        stat["acceleration_limiter_mean"] = stat.pop(
                            "acceleration_limiter_sum"
                        ) / HORIZON
                        rows.append(stat)
                        if next_episode < EPISODES:
                            active[worker] = next_episode
                            reset_requests[worker] = {
                                "scene": "none", "terminate_on_success": False
                            }
                            reset_seeds[worker] = SEED_START + next_episode
                            next_episode += 1
                        else:
                            del active[worker]
                            del stats[worker]
                if reset_requests:
                    resets = pool.reset_many(reset_requests, seeds=reset_seeds)
                    for worker, (observation, info) in resets.items():
                        observations[worker] = observation
                        stats[worker] = initial_stat(
                            SEED_START + active[worker], info
                        )
                if len(rows) % 100 < len(results) and len(rows) > 0:
                    print(
                        f"scale={scale:.2f} episodes={len(rows)}/{EPISODES} "
                        f"elapsed={perf_counter()-started:.1f}s",
                        flush=True,
                    )

    pose5_steps = [
        row["first_pose5_step"] for row in rows
        if row["first_pose5_step"] is not None
    ]
    summary = {
        "base_scale": scale,
        "episodes": len(rows),
        "pose5_count": sum(row["pose5"] for row in rows),
        "pose10_count": sum(row["pose10"] for row in rows),
        "environment_success_count": sum(row["success"] for row in rows),
        "collision_count": sum(row["collision"] for row in rows),
        "self_collision_count": sum(row["self_collision"] for row in rows),
        "plane_collision_count": sum(
            row["environment_collision"] for row in rows
        ),
        "safe_pose5_count": sum(
            row["pose5"] and not row["collision"] for row in rows
        ),
        "best_position_mean_m": float(
            np.mean([row["best_position_m"] for row in rows])
        ),
        "best_position_median_m": float(
            np.median([row["best_position_m"] for row in rows])
        ),
        "best_orientation_mean_rad": float(
            np.mean([row["best_orientation_rad"] for row in rows])
        ),
        "position_at_best_reference_mean_m": float(
            np.mean([row["position_at_best_reference_m"] for row in rows])
        ),
        "orientation_at_best_reference_mean_rad": float(
            np.mean([row["orientation_at_best_reference_rad"] for row in rows])
        ),
        "first_pose5_step_mean": (
            float(np.mean(pose5_steps)) if pose5_steps else None
        ),
        "return_mean": float(np.mean([row["return"] for row in rows])),
        "velocity_limiter_mean": float(
            np.mean([row["velocity_limiter_mean"] for row in rows])
        ),
        "acceleration_limiter_mean": float(
            np.mean([row["acceleration_limiter_mean"] for row in rows])
        ),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return rows, summary


def main():
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    # Match the production evaluator: old checkpoints retain their
    # authoritative experiment values while today's schema supplies newly
    # centralized fields that did not yet exist when the checkpoint was saved.
    source_config = load_config("configs/experiments/paper_full.yaml")
    deep_update(source_config, deepcopy(payload["config"]))
    results = {}
    for scale in SCALES:
        rows, summary = evaluate_scale(payload, source_config, scale)
        results[str(scale)] = {"summary": summary, "rows": rows}
    baseline = {row["seed"]: row for row in results["1.0"]["rows"]}
    scaled = {row["seed"]: row for row in results["0.75"]["rows"]}
    paired = {
        "pose5_retained": sum(
            baseline[seed]["pose5"] and scaled[seed]["pose5"]
            for seed in baseline
        ),
        "pose5_lost": [
            seed for seed in baseline
            if baseline[seed]["pose5"] and not scaled[seed]["pose5"]
        ],
        "pose5_gained": [
            seed for seed in baseline
            if not baseline[seed]["pose5"] and scaled[seed]["pose5"]
        ],
        "collision_gained": [
            seed for seed in baseline
            if not baseline[seed]["collision"] and scaled[seed]["collision"]
        ],
        "collision_removed": [
            seed for seed in baseline
            if baseline[seed]["collision"] and not scaled[seed]["collision"]
        ],
    }
    results["paired"] = paired
    with open(OUTPUT, "w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    print(json.dumps(paired, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
