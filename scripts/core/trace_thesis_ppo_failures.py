#!/usr/bin/env python3
"""Replay selected deterministic S1 episodes and record every policy decision."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from rl_risk_sac.algorithms.thesis_ppo import ThesisPPOAgent
from rl_risk_sac.envs.thesis_homotopy_env_reduced_observation import ThesisHomotopyEnv


def contract() -> dict[str, object]:
    return {
        "scene": "static", "terminate_on_success": False,
    }


def vector_fields(prefix: str, values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {f"{prefix}_{index + 1}": float(value) for index, value in enumerate(flat)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")

    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = deepcopy(payload["config"])
    if args.device:
        config["device"] = args.device
    config["thesis"]["stage"] = "s1"
    config["thesis"]["simple_reaching_mode"] = False
    config["thesis"]["blocking_obstacle_mode"] = True
    config["thesis"].setdefault("scene_mixture", {})["s1"] = {"static": 1.0}
    env = ThesisHomotopyEnv(config)
    agent = ThesisPPOAgent(
        int(env.observation_space.shape[0]), int(env.action_space.shape[0]), config
    )
    agent.load_state_dict(payload["agent"])
    rows: list[dict[str, object]] = []
    try:
        for seed in args.seeds:
            env.configure_episode(**contract())
            observation, reset_info = env.reset(seed=seed)
            obstacle_position = env.obstacle_position.copy()
            goal_position = env.goal_position.copy()
            cumulative_return = 0.0
            contact_seen = False
            for decision in range(1, env.horizon + 1):
                q_before, qdot_before = env.robot.joint_state_exact()
                obs_tensor = torch.as_tensor(
                    observation[None, :], dtype=torch.float32,
                    device=agent.device,
                )
                with torch.no_grad():
                    total, base, residual = agent._policy_components(obs_tensor)
                action = total[0].cpu().numpy().astype(np.float32)
                base_action = base[0].cpu().numpy()
                residual_action = residual[0].cpu().numpy()
                next_observation, reward, _cost, terminated, truncated, info = env.step(action)
                q_after, qdot_after = env.robot.joint_state_exact()
                cumulative_return += float(reward)
                contact_seen |= bool(info["collision"])
                row: dict[str, object] = {
                    "seed": seed,
                    "decision": decision,
                    "reward": float(reward),
                    "cumulative_return": cumulative_return,
                    "position_error_m": float(info["next_rho_position"]),
                    "orientation_error_rad": float(info["next_rho_orientation"]),
                    "reference_error": float(info["reference_error"]),
                    "r_goal": float(info["r_goal"]),
                    "proximity_penalty": float(info["reference_obstacle_penalty"]),
                    "collision_penalty": float(info["reference_collision_penalty"]),
                    "collision": bool(info["collision"]),
                    "obstacle_collision": bool(info["obstacle_collision"]),
                    "self_collision": bool(info["self_collision"]),
                    "environment_collision": bool(info["environment_collision"]),
                    "contact_seen": contact_seen,
                    "min_obstacle_clearance_m": float(info["d_min"]),
                    "min_self_clearance_m": float(info["self_d_min"]),
                    "action_abs_mean": float(np.mean(np.abs(action))),
                    "base_action_abs_mean": float(np.mean(np.abs(base_action))),
                    "residual_action_abs_mean": float(np.mean(np.abs(residual_action))),
                    "residual_to_base_ratio": float(
                        np.linalg.norm(residual_action)
                        / max(np.linalg.norm(base_action), 1.0e-12)
                    ),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    **vector_fields("goal", goal_position),
                    **vector_fields("obstacle", obstacle_position),
                    **vector_fields("q_before", q_before),
                    **vector_fields("q_after", q_after),
                    **vector_fields("qdot_before", qdot_before),
                    **vector_fields("qdot_after", qdot_after),
                    **vector_fields("base_action", base_action),
                    **vector_fields("residual_action", residual_action),
                    **vector_fields("total_action", action),
                    **vector_fields("commanded_delta", info["commanded_joint_delta"]),
                    **vector_fields("link_clearance", info["distance_body"]),
                }
                rows.append(row)
                observation = next_observation
                if terminated or truncated:
                    break
            print(
                f"seed={seed} decisions={decision} return={cumulative_return:.3f} "
                f"best_position={min(row['position_error_m'] for row in rows if row['seed'] == seed):.6f} "
                f"collision={int(contact_seen)}",
                flush=True,
            )
    finally:
        env.close()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved trace: {output}", flush=True)


if __name__ == "__main__":
    main()
