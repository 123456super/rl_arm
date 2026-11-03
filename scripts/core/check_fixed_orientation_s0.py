#!/usr/bin/env python3
"""Audit reset targets, endpoint holding, and a diagnostic joint-space driver.

This is not a trained-policy evaluation. IK is used only by this offline check;
the trainer never feeds IK joint targets to the policy or uses this driver.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_risk_sac.envs.thesis_homotopy_env_reduced_observation import ThesisHomotopyEnv
from rl_risk_sac.tasks.thesis_reaching_reduced_observation import pose_error
from rl_risk_sac.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]


def audit(config: dict, *, samples: int, control_trials: int, seed: int) -> dict:
    if samples < 1 or control_trials < 0:
        raise ValueError("samples must be positive and control_trials nonnegative")
    config["seed"] = seed
    config["thesis"]["simple_reaching_mode"] = True
    env = ThesisHomotopyEnv(config)
    try:
        if not env.fixed_orientation_goal_enabled:
            raise ValueError("enable thesis.fixed_orientation_goal before running this audit")
        env.configure_episode("none", xi=1.0, strict=False)
        env._reset_fixed_initial_state()
        initial_position, initial_quaternion = env.robot.end_effector_pose()
        initial_singular_values = np.linalg.svd(
            env.robot.end_effector_jacobian(), compute_uv=False
        )
        initial_clearance = env._self_geometry().distance_min
        _, initial_orientation_error = pose_error(
            initial_position, env.fixed_goal_quaternion,
            initial_position, initial_quaternion,
        )
        positions, proposals, position_errors, orientation_errors = [], [], [], []
        for index in range(samples):
            obs, info = env.reset()
            np.testing.assert_allclose(env.robot.joint_state_exact()[0], env.fixed_initial_joint_positions, atol=1e-7)
            np.testing.assert_allclose(info["goal_quaternion"], env.fixed_goal_quaternion, atol=1e-7)
            if not env.observation_space.contains(obs):
                raise AssertionError("reset observation is outside its declared space")
            validation = info["goal_ik_validation"]
            if not validation["reachable"] or not validation["collision_free"]:
                raise AssertionError("invalid goal accepted")
            positions.append(info["goal_position"])
            proposals.append(info["goal_sample_attempts"])
            position_errors.append(validation["position_error_m"])
            orientation_errors.append(validation["orientation_error_rad"])
            if (index + 1) % 100 == 0:
                print(f"checked_targets={index + 1}/{samples}", flush=True)

        hold_successes = 0
        tracking_successes = 0
        collision_free_tracking_successes = 0
        lengths = []
        for _ in range(control_trials):
            env.reset()
            # Validate the actual physics controller can hold a known IK endpoint.
            env._set_joint_state(env.goal_joint_positions)
            for _ in range(env.success_hold_steps):
                _, _, _, terminated, truncated, info = env.step(env.goal_joint_positions)
                if terminated or truncated:
                    break
            hold_successes += int(info["success"])

            # A direct joint-space path need not be collision-free, even in S0.
            # Report this separately; it is neither a planning proof nor an RL score.
            env.reset()
            had_collision = False
            for step in range(env.horizon):
                q, _ = env.robot.joint_state_exact()
                action = q + np.clip(env.goal_joint_positions - q, -0.25, 0.25)
                _, _, _, terminated, truncated, info = env.step(action)
                had_collision |= bool(info["collision"])
                if terminated or truncated:
                    break
            tracking_successes += int(info["success"])
            collision_free_tracking_successes += int(info["success"] and not had_collision)
            lengths.append(step + 1)
        positions = np.asarray(positions)
        return {
            "kind": "fixed_orientation_s0_preflight_not_policy_evaluation",
            "reward_mode": env.reward_mode,
            "seed": seed,
            "samples": samples,
            "observation_dim": env.observation_space.shape[0],
            "target_euler_rad": env.fixed_goal_euler.tolist(),
            "target_quaternion_xyzw": env.fixed_goal_quaternion.tolist(),
            "initial_joint_positions_rad": env.fixed_initial_joint_positions.tolist(),
            "initial_ee_position_m": initial_position.tolist(),
            "initial_self_clearance_m": float(initial_clearance),
            "initial_jacobian_sigma_min": float(initial_singular_values[-1]),
            "initial_jacobian_condition_number": float(
                initial_singular_values[0] / max(initial_singular_values[-1], 1e-16)
            ),
            "initial_orientation_error_rad": float(np.linalg.norm(initial_orientation_error)),
            "position_min_m": positions.min(axis=0).tolist(),
            "position_max_m": positions.max(axis=0).tolist(),
            "max_ik_position_error_m": float(max(position_errors)),
            "max_ik_orientation_error_rad": float(max(orientation_errors)),
            "proposal_mean": float(np.mean(proposals)),
            "proposal_max": int(max(proposals)),
            "control_trials": control_trials,
            "ik_endpoint_hold_successes": hold_successes,
            "direct_joint_tracking_successes": tracking_successes,
            "collision_free_direct_tracking_successes": collision_free_tracking_successes,
            "direct_tracking_mean_steps": float(np.mean(lengths)) if lengths else None,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/paper_full.yaml")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--control-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument("--reward-mode", choices=("keypoint", "reference_ppo"), default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    if args.reward_mode is not None:
        config["thesis"]["reward"]["reward_mode"] = args.reward_mode
    report = audit(config, samples=args.samples,
                   control_trials=args.control_trials, seed=args.seed)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(output + "\n")
    if report["ik_endpoint_hold_successes"] != report["control_trials"]:
        raise SystemExit("some verified endpoints failed the physical hold check")


if __name__ == "__main__":
    main()
