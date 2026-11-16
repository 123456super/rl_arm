#!/usr/bin/env python3
"""Replay selected deterministic static-scene episodes and record every decision."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
import pybullet as p
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
    parser.add_argument(
        "--joint-velocity-limit-deg-s", type=float,
        help="Override the checkpoint joint-speed limit for this trace only.",
    )
    parser.add_argument(
        "--base-only", action="store_true",
        help="Execute the frozen Base action while still recording the residual proposal.",
    )
    parser.add_argument(
        "--residual-cutoff-position-m", type=float, default=0.025,
        help="Use the deployment Base handoff at or below this position error.",
    )
    parser.add_argument(
        "--no-residual-cutoff", action="store_const", const=None,
        dest="residual_cutoff_position_m",
        help="Trace the full-Residual ablation instead of deployment control.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if (
        args.joint_velocity_limit_deg_s is not None
        and args.joint_velocity_limit_deg_s <= 0.0
    ):
        parser.error("--joint-velocity-limit-deg-s must be positive")

    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = deepcopy(payload["config"])
    if args.device:
        config["device"] = args.device
    if args.joint_velocity_limit_deg_s is not None:
        motion = config["thesis"]["joint_motion_controller"]
        if not bool(motion.get("enabled", True)):
            parser.error("joint velocity override requires the bounded controller")
        velocity_limit = float(np.deg2rad(args.joint_velocity_limit_deg_s))
        motion["joint_velocity_limits_rad_s"] = [velocity_limit] * 6
    stage = str(payload.get("stage", config["thesis"].get("stage", ""))).lower()
    if stage not in {"s1", "s2"}:
        raise ValueError(f"static obstacle traces require an S1/S2 checkpoint, got {stage!r}")
    config["thesis"]["stage"] = stage
    config["thesis"]["simple_reaching_mode"] = False
    config["thesis"]["blocking_obstacle_mode"] = True
    config["thesis"].setdefault("scene_mixture", {})[stage] = {"static": 1.0}
    env = ThesisHomotopyEnv(config)
    agent = ThesisPPOAgent(
        int(env.observation_space.shape[0]), int(env.action_space.shape[0]), config
    )
    agent.load_state_dict(payload["agent"])
    rows: list[dict[str, object]] = []
    try:
        original_collision_events = env._collision_events
        for seed in args.seeds:
            env.configure_episode(**contract())
            observation, reset_info = env.reset(seed=seed)
            obstacle_position = env.obstacle_position.copy()
            goal_position = env.goal_position.copy()
            current_position_error = float(reset_info["goal_error_norm"])
            cumulative_return = 0.0
            contact_seen = False
            id_to_link_name = {
                link_id: name for name, link_id in env._link_name_to_id.items()
            }
            contact_capture: dict[str, object] = {}
            def traced_collision_events():
                result = original_collision_events()
                for point in p.getContactPoints(
                    bodyA=env.robot_id, physicsClientId=env.client_id
                ):
                    if point[8] > 0.0:
                        continue
                    other_body = int(point[2])
                    robot_link = int(point[3])
                    if other_body in env._obstacle_body_ids:
                        contact_capture["obstacle_links"].add(robot_link)
                    elif (
                        env.mounting_plane_id is not None
                        and other_body == env.mounting_plane_id
                    ):
                        contact_capture["plane_links"].add(robot_link)
                        contact_capture["plane_z"].append(float(point[5][2]))
                return result

            env._collision_events = traced_collision_events
            for decision in range(1, env.horizon + 1):
                contact_capture.clear()
                contact_capture.update(
                    obstacle_links=set(), plane_links=set(), plane_z=[]
                )
                q_before, qdot_before = env.robot.joint_state_exact()
                base_action_query = agent.predict_base_actions(
                    observation[None, :]
                )[0]
                residual_safety_context = env.residual_safety_context(
                    base_action_query
                )[None, :]
                actions, _logp, _values, policy = agent.act_for_deployment(
                    observation[None, :],
                    np.asarray([current_position_error], dtype=np.float32),
                    residual_cutoff_position_m=args.residual_cutoff_position_m,
                    residual_safety_context=residual_safety_context,
                )
                policy_action = policy["policy_actions"][0].copy()
                base_action = policy["base_actions"][0].copy()
                residual_action = policy["residual_actions"][0].copy()
                residual_authority = float(policy["residual_authority"][0])
                if args.base_only:
                    action = base_action.astype(np.float32, copy=True)
                else:
                    action = actions[0].copy()
                next_observation, reward, _cost, terminated, truncated, info = env.step(action)
                q_after, qdot_after = env.robot.joint_state_exact()
                capsules = env.robot.capsule_model.states(
                    env.robot_id, env.client_id
                )
                capsule_bottoms = np.asarray([
                    min(float(capsule.start[2]), float(capsule.end[2]))
                    - float(capsule.radius)
                    - float(env.mounting_plane_height)
                    for capsule in capsules
                ], dtype=np.float64)
                lowest_capsule_index = int(np.argmin(capsule_bottoms))
                cumulative_return += float(reward)
                contact_seen |= bool(info["collision"])
                current_position_error = float(info["next_rho_position"])
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
                    "residual_authority": residual_authority,
                    "lowest_capsule": capsules[lowest_capsule_index].name,
                    "lowest_capsule_bottom_m": float(
                        capsule_bottoms[lowest_capsule_index]
                    ),
                    "obstacle_contact_links": "|".join(
                        id_to_link_name.get(link_id, str(link_id))
                        for link_id in sorted(contact_capture["obstacle_links"])
                    ),
                    "plane_contact_links": "|".join(
                        id_to_link_name.get(link_id, str(link_id))
                        for link_id in sorted(contact_capture["plane_links"])
                    ),
                    "plane_contact_z_min_m": (
                        min(contact_capture["plane_z"])
                        if contact_capture["plane_z"] else float("nan")
                    ),
                    **vector_fields("ee_position", info["ee_position"]),
                    "joint_speed_peak_rad_s": float(info["joint_speed_peak_rad_s"]),
                    "joint_velocity_limiter_fraction": float(
                        info["joint_velocity_limiter_fraction"]
                    ),
                    "joint_acceleration_limiter_fraction": float(
                        info["joint_acceleration_limiter_fraction"]
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
                    **vector_fields("policy_total_action", policy_action),
                    **vector_fields("executed_action", action),
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
