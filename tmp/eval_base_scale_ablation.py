from copy import deepcopy
import json

import numpy as np
import torch

from rl_risk_sac.algorithms.thesis_ppo import ThesisPPOAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool


CHECKPOINT = (
    "outputs/ppo/s1_single_obb_compact9_fresh5m_seed11001/"
    "checkpoints/step_5000000.pt"
)
SEEDS = [
    41006, 41026, 41042, 41138, 41143, 41164, 41186, 41213, 41247,
    41287, 41327, 41333, 41343, 41387, 41412, 41434, 41511, 41543,
    41560, 41565, 41587, 41596, 41613, 41628, 41643, 41666, 41679,
    41708, 41731, 41785, 41803, 41833, 41869, 41921, 41928, 41940,
    41961, 41974, 41978,
]
BASE_SCALES = (1.0, 0.75, 0.5, 0.25)


def evaluate_scale(payload, base_config, base_scale):
    config = deepcopy(base_config)
    agent = ThesisPPOAgent(int(payload["agent"]["obs_dim"]), 6, config)
    agent.load_state_dict(payload["agent"])
    rows = []
    batch_size = 16
    for start in range(0, len(SEEDS), batch_size):
        batch = SEEDS[start : start + batch_size]
        count = len(batch)
        cpu_ids = config["train"].get("cpu_ids", [])[:count] or None
        with ParallelThesisEnvPool(config, batch, cpu_ids=cpu_ids) as pool:
            resets = pool.reset_many(
                {
                    index: {"scene": "static", "terminate_on_success": False}
                    for index in range(count)
                },
                seeds={index: seed for index, seed in enumerate(batch)},
            )
            observations = np.stack(
                [resets[index][0] for index in range(count)]
            ).astype(np.float32)
            position_errors = np.asarray(
                [float(resets[index][1]["goal_error_norm"]) for index in range(count)],
                dtype=np.float32,
            )
            stats = {
                index: {
                    "seed": batch[index],
                    "collision": False,
                    "obstacle_collision": False,
                    "self_collision": False,
                    "environment_collision": False,
                    "best_position_m": float(position_errors[index]),
                    "best_orientation_rad": float(
                        resets[index][1]["orientation_error_norm"]
                    ),
                    "return": 0.0,
                    "peak_speed_rad_s": 0.0,
                    "velocity_limiter_sum": 0.0,
                    "acceleration_limiter_sum": 0.0,
                }
                for index in range(count)
            }
            for _step in range(150):
                original_base = agent.predict_base_actions(observations)
                context_results = pool.residual_safety_context_many(
                    {index: original_base[index] for index in range(count)}
                )
                safety_context = np.stack(
                    [context_results[index] for index in range(count)]
                ).astype(np.float32)
                _actions, _logp, _values, policy = agent.act_for_deployment(
                    observations,
                    position_errors,
                    residual_cutoff_position_m=0.025,
                    residual_safety_context=safety_context,
                )
                # Keep the learned Residual and its inputs exactly unchanged.
                # Only attenuate Base at the final action-composition boundary.
                actions = (
                    float(base_scale) * policy["base_actions"]
                    + policy["residual_authority"][:, None]
                    * policy["residual_actions"]
                ).astype(np.float32)
                results = pool.step_many(
                    {index: actions[index] for index in range(count)}
                )
                for index, (
                    next_observation,
                    reward,
                    _cost,
                    _terminated,
                    _truncated,
                    info,
                ) in results.items():
                    observations[index] = next_observation
                    position_errors[index] = float(info["goal_error_norm"])
                    stat = stats[index]
                    stat["return"] += float(reward)
                    stat["best_position_m"] = min(
                        stat["best_position_m"], float(info["next_rho_position"])
                    )
                    stat["best_orientation_rad"] = min(
                        stat["best_orientation_rad"],
                        float(info["next_rho_orientation"]),
                    )
                    for key in (
                        "collision",
                        "obstacle_collision",
                        "self_collision",
                        "environment_collision",
                    ):
                        stat[key] = stat[key] or bool(info[key])
                    stat["peak_speed_rad_s"] = max(
                        stat["peak_speed_rad_s"],
                        float(info["joint_speed_peak_rad_s"]),
                    )
                    stat["velocity_limiter_sum"] += float(
                        info["joint_velocity_limiter_fraction"]
                    )
                    stat["acceleration_limiter_sum"] += float(
                        info["joint_acceleration_limiter_fraction"]
                    )
            for stat in stats.values():
                stat["velocity_limiter_mean"] = stat.pop(
                    "velocity_limiter_sum"
                ) / 150
                stat["acceleration_limiter_mean"] = stat.pop(
                    "acceleration_limiter_sum"
                ) / 150
                rows.append(stat)

    pose5 = [
        row["best_position_m"] <= 0.005
        and row["best_orientation_rad"] <= 0.05
        for row in rows
    ]
    safe_pose5 = [
        ok and not row["collision"] for ok, row in zip(pose5, rows, strict=True)
    ]
    summary = {
        "base_scale": base_scale,
        "episodes": len(rows),
        "collision_count": sum(row["collision"] for row in rows),
        "obstacle_collision_count": sum(
            row["obstacle_collision"] for row in rows
        ),
        "self_collision_count": sum(row["self_collision"] for row in rows),
        "plane_collision_count": sum(
            row["environment_collision"] for row in rows
        ),
        "pose5_count": sum(pose5),
        "safe_pose5_count": sum(safe_pose5),
        "best_position_mean_m": float(
            np.mean([row["best_position_m"] for row in rows])
        ),
        "best_position_median_m": float(
            np.median([row["best_position_m"] for row in rows])
        ),
        "return_mean": float(np.mean([row["return"] for row in rows])),
        "peak_speed_mean_deg_s": float(
            np.rad2deg(np.mean([row["peak_speed_rad_s"] for row in rows]))
        ),
        "velocity_limiter_mean": float(
            np.mean([row["velocity_limiter_mean"] for row in rows])
        ),
        "acceleration_limiter_mean": float(
            np.mean([row["acceleration_limiter_mean"] for row in rows])
        ),
        "collision_seeds": [row["seed"] for row in rows if row["collision"]],
        "pose5_failure_seeds": [
            row["seed"] for row, ok in zip(rows, pose5, strict=True) if not ok
        ],
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return rows, summary


def main():
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = deepcopy(payload["config"])
    config["device"] = "cuda:0"
    config["thesis"]["stage"] = "s1"
    config["thesis"]["simple_reaching_mode"] = False
    config["thesis"]["blocking_obstacle_mode"] = True
    results = {}
    for scale in BASE_SCALES:
        rows, summary = evaluate_scale(payload, config, scale)
        results[str(scale)] = {"rows": rows, "summary": summary}
    with open(
        "tmp/base_scale_ablation_results.json", "w", encoding="utf-8"
    ) as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
