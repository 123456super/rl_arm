#!/usr/bin/env python3
"""Preflight audits for the task-first safety-homotopy training protocol."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import pybullet as p
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.algorithms.homotopy_replay import HomotopyReplayBuffer
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.thesis_reaching import build_thesis_observation, homotopy_reward
from rl_risk_sac.utils.config import load_config


def discounted(values: list[float], gamma: float) -> float:
    return float(sum((gamma**index) * value for index, value in enumerate(values)))


def reward_audit(config: dict, samples: int) -> tuple[dict, list[dict]]:
    env = ThesisHomotopyEnv(config)
    rows: list[dict] = []
    gamma = float(config["sac"]["gamma"])
    horizon = int(config["thesis"]["horizon"])
    try:
        for index in range(samples):
            env.configure_episode("none", xi=1.0, strict=True)
            _, info = env.reset()
            rho_p = float(info["goal_error_norm"])
            rho_r = float(info["orientation_error_norm"])
            strict_collision, strict_collision_fields = homotopy_reward(
                rho_position=rho_p, next_rho_position=rho_p,
                rho_orientation=rho_r, next_rho_orientation=rho_r,
                smooth_velocity=0.0, task_reached=False, hard_failure=False,
                obstacle_collision=True, terminal_obstacle_collision=True,
                risk_max=1.0, distance_min=0.0, xi=1.0,
                gamma=gamma, horizon=horizon,
            )
            tolerant_collision, tolerant_collision_fields = homotopy_reward(
                rho_position=rho_p, next_rho_position=rho_p,
                rho_orientation=rho_r, next_rho_orientation=rho_r,
                smooth_velocity=0.0, task_reached=False, hard_failure=False,
                obstacle_collision=True, terminal_obstacle_collision=False,
                risk_max=1.0, distance_min=0.0, xi=0.02,
                gamma=gamma, horizon=horizon,
            )
            hard_failure, hard_failure_fields = homotopy_reward(
                rho_position=rho_p, next_rho_position=rho_p,
                rho_orientation=rho_r, next_rho_orientation=rho_r,
                smooth_velocity=0.0, task_reached=False, hard_failure=True,
                obstacle_collision=False, risk_max=0.0, distance_min=0.80, xi=1.0,
                gamma=gamma, horizon=horizon,
            )
            wait_step, _ = homotopy_reward(
                rho_position=rho_p, next_rho_position=rho_p,
                rho_orientation=rho_r, next_rho_orientation=rho_r,
                smooth_velocity=0.0, task_reached=False, hard_failure=False,
                obstacle_collision=False, risk_max=0.0, distance_min=0.80, xi=1.0,
                gamma=gamma, horizon=horizon,
            )
            timeout = discounted([wait_step] * horizon, gamma)
            ideal_values: list[float] = []
            steps = 60
            previous_p, previous_r = rho_p, rho_r
            for step in range(1, steps + 1):
                next_p = rho_p * (1.0 - step / steps)
                next_r = rho_r * (1.0 - step / steps)
                reward, _ = homotopy_reward(
                    rho_position=previous_p, next_rho_position=next_p,
                    rho_orientation=previous_r, next_rho_orientation=next_r,
                    smooth_velocity=0.0, task_reached=step == steps,
                    hard_failure=False, obstacle_collision=False,
                    risk_max=0.0, distance_min=0.80, xi=1.0,
                    gamma=gamma, horizon=horizon,
                )
                ideal_values.append(reward)
                previous_p, previous_r = next_p, next_r
            ideal_success = discounted(ideal_values, gamma)
            rows.append({
                "sample": index, "rho_position": rho_p, "rho_orientation": rho_r,
                "strict_collision_now": strict_collision,
                "tolerant_collision_step": tolerant_collision,
                "strict_collision_guard": strict_collision_fields["terminal_guard_penalty"],
                "hard_failure_now": hard_failure,
                "hard_failure_guard": hard_failure_fields["terminal_guard_penalty"],
                "stationary_timeout": timeout, "ideal_60_step_success": ideal_success,
                "strict_collision_better_than_timeout": strict_collision > timeout,
                "hard_failure_better_than_timeout": hard_failure > timeout,
                "desired_order": ideal_success > timeout > max(strict_collision, hard_failure),
            })
    finally:
        env.close()
    summary = {
        "samples": samples,
        "strict_collision_better_than_timeout_rate": float(np.mean([r["strict_collision_better_than_timeout"] for r in rows])),
        "hard_failure_better_than_timeout_rate": float(np.mean([r["hard_failure_better_than_timeout"] for r in rows])),
        "desired_order_rate": float(np.mean([r["desired_order"] for r in rows])),
        "median_returns": {
            key: float(np.median([r[key] for r in rows]))
            for key in (
                "strict_collision_now", "tolerant_collision_step", "hard_failure_now",
                "stationary_timeout", "ideal_60_step_success",
            )
        },
        "median_terminal_guard": {
            "strict_collision": float(np.median([r["strict_collision_guard"] for r in rows])),
            "hard_failure": float(np.median([r["hard_failure_guard"] for r in rows])),
        },
    }
    return summary, rows


def _set_q(env: ThesisHomotopyEnv, q: np.ndarray) -> None:
    env._set_joint_state(q)  # The audit intentionally checks the generator's exact joint-space witness.
    p.performCollisionDetection(physicsClientId=env.client_id)


def feasibility_audit(config: dict, samples_per_scene: int) -> tuple[dict, list[dict]]:
    env = ThesisHomotopyEnv(config)
    rows: list[dict] = []
    try:
        for scene in ("static", "dynamic"):
            for sample in range(samples_per_scene):
                env.configure_episode(scene, xi=1.0, strict=True)
                env.reset()
                q0, _ = env.robot.joint_state()
                q_goal = env.goal_joint_positions.copy()
                delta = q_goal - q0
                min_steps = max(1, int(math.ceil(float(np.max(np.abs(delta))) / (env.action_scale[0] * env.control_dt))))
                steps = min(env.horizon, min_steps)
                self_collision = obstacle_collision = False
                minimum_distance = float("inf")
                for step in range(1, steps + 1):
                    q = q0 + delta * (step / steps)
                    _set_q(env, q)
                    if scene == "dynamic":
                        for _ in range(env.substeps):
                            env._advance_obstacle()
                    events = env._collision_events()
                    geometry = env._geometry()
                    self_collision |= events["self_collision"]
                    obstacle_collision |= events["obstacle_collision"]
                    if geometry is not None:
                        minimum_distance = min(minimum_distance, geometry.distance_min)
                position, quaternion = env.robot.end_effector_pose()
                from rl_risk_sac.tasks.thesis_reaching import pose_error
                error_p, error_r = pose_error(env.goal_position, env.goal_quaternion, position, quaternion)
                witness = (
                    not self_collision and not obstacle_collision
                    and float(np.linalg.norm(error_p)) <= env.position_tolerance
                    and float(np.linalg.norm(error_r)) <= env.orientation_tolerance
                    and min_steps <= env.horizon
                )
                rows.append({
                    "scene": scene, "sample": sample, "min_steps": min_steps,
                    "within_horizon": min_steps <= env.horizon,
                    "self_collision": self_collision, "obstacle_collision": obstacle_collision,
                    "minimum_distance": minimum_distance,
                    "position_error": float(np.linalg.norm(error_p)),
                    "orientation_error": float(np.linalg.norm(error_r)),
                    "direct_joint_witness": witness,
                })
    finally:
        env.close()
    summary = {"samples_per_scene": samples_per_scene, "scenes": {}}
    for scene in ("static", "dynamic"):
        subset = [r for r in rows if r["scene"] == scene]
        summary["scenes"][scene] = {
            "direct_joint_witness_rate": float(np.mean([r["direct_joint_witness"] for r in subset])),
            "obstacle_collision_rate": float(np.mean([r["obstacle_collision"] for r in subset])),
            "self_collision_rate": float(np.mean([r["self_collision"] for r in subset])),
            "within_horizon_rate": float(np.mean([r["within_horizon"] for r in subset])),
        }
    summary["interpretation"] = (
        "A failed direct joint interpolation is not proof of infeasibility; a successful witness proves feasibility."
    )
    return summary, rows


def goal_kinematics_audit(config: dict, samples: int) -> tuple[dict, list[dict]]:
    """Audit the exact FK -> pose IK -> FK acceptance contract used by reset()."""
    env = ThesisHomotopyEnv(config)
    rows: list[dict] = []
    failures = 0
    try:
        for sample in range(samples):
            try:
                env.configure_episode("none", xi=1.0, strict=True)
                _, info = env.reset(seed=int(config["seed"]) + sample)
                validation = info["goal_ik_validation"]
                rows.append({
                    "sample": sample,
                    "accepted": bool(validation["reachable"]),
                    "attempts": int(info["goal_sample_attempts"]),
                    "ik_finite": bool(validation["finite"]),
                    "ik_within_limits": bool(validation["within_limits"]),
                    "position_error_m": float(validation["position_error_m"]),
                    "orientation_error_rad": float(validation["orientation_error_rad"]),
                })
            except RuntimeError:
                failures += 1
                rows.append({
                    "sample": sample, "accepted": False,
                    "attempts": int(config["thesis"].get("goal_sample_max_attempts", 10000)),
                    "ik_finite": False, "ik_within_limits": False,
                    "position_error_m": float("inf"), "orientation_error_rad": float("inf"),
                })
    finally:
        env.close()
    accepted = [row for row in rows if row["accepted"]]
    summary = {
        "samples": samples,
        "reset_failures": failures,
        "acceptance_rate": float(len(accepted) / samples),
        "thresholds": {"position_error_m": 0.01, "orientation_error_rad": 0.05},
        "attempts": {
            "median": float(np.median([row["attempts"] for row in rows])),
            "p95": float(np.quantile([row["attempts"] for row in rows], 0.95)),
            "max": int(max(row["attempts"] for row in rows)),
        },
        "accepted_error_max": {
            "position_error_m": float(max(row["position_error_m"] for row in accepted)) if accepted else None,
            "orientation_error_rad": float(max(row["orientation_error_rad"] for row in accepted)) if accepted else None,
        },
    }
    return summary, rows


def checkpoint_audit(config: dict, checkpoint: Path, samples: int) -> tuple[dict, list[dict]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent = ThesisSACAgent(55, 6, config)
    agent.load_state_dict(state["agent"])
    replay = HomotopyReplayBuffer(
        55, 6, str(agent.device), seed=int(config["seed"]),
        reward_gamma=float(config["sac"]["gamma"]),
        reward_horizon=int(config["thesis"]["horizon"]),
    )
    replay.load_state_dict(state["replay"])
    env = ThesisHomotopyEnv(config)
    rows: list[dict] = []
    action_samples = 256
    try:
        for index in range(samples):
            env.configure_episode("static", xi=0.02, strict=False)
            obs_obstacle, info = env.reset()
            q, qdot = env.robot.joint_state()
            error_p = np.asarray(info["goal_position"]) - np.asarray(info["ee_position"])
            from rl_risk_sac.tasks.thesis_reaching import pose_error
            error_p, error_r = pose_error(
                info["goal_position"], info["goal_quaternion"], info["ee_position"], info["ee_quaternion"]
            )
            obs_none = build_thesis_observation(
                q=q, qdot=qdot, joint_lower=env.robot.joint_lower_limits,
                joint_upper=env.robot.joint_upper_limits, joint_velocity_limits=env.velocity_limits,
                position_error=error_p, orientation_error=error_r, obstacle_present=False,
                obstacle_position=np.zeros(3), obstacle_velocity=np.zeros(3), geometry=None,
            )
            action_none = agent.select_action(obs_none, deterministic=True)
            action_obstacle = agent.select_action(obs_obstacle, deterministic=True)
            obs_tensor = torch.as_tensor(obs_obstacle, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                mean, log_std = agent.actor(obs_tensor)
                sampled = torch.stack([agent.actor.sample(obs_tensor)[0][0] for _ in range(action_samples)])
            rows.append({
                "sample": index,
                "action_shift_l2": float(np.linalg.norm(action_obstacle - action_none)),
                "action_shift_max_abs": float(np.max(np.abs(action_obstacle - action_none))),
                "mean_abs_action_none": float(np.mean(np.abs(action_none))),
                "mean_abs_action_obstacle": float(np.mean(np.abs(action_obstacle))),
                "policy_std_mean": float(log_std.exp().mean().cpu()),
                "sampled_action_std_mean": float(sampled.std(dim=0).mean().cpu()),
                "sampled_action_std_min": float(sampled.std(dim=0).min().cpu()),
            })
    finally:
        env.close()
    summary = {
        "checkpoint": str(checkpoint), "samples": samples,
        "alpha": float(agent.alpha.detach().cpu()),
        "action_shift_l2_median": float(np.median([r["action_shift_l2"] for r in rows])),
        "action_shift_l2_p95": float(np.quantile([r["action_shift_l2"] for r in rows], .95)),
        "action_shift_max_abs_p95": float(np.quantile([r["action_shift_max_abs"] for r in rows], .95)),
        "policy_std_mean": float(np.mean([r["policy_std_mean"] for r in rows])),
        "sampled_action_std_mean": float(np.mean([r["sampled_action_std_mean"] for r in rows])),
        "sampled_action_std_min": float(np.min([r["sampled_action_std_min"] for r in rows])),
    }
    if replay.parts["none"].size >= 256:
        batch = replay.sample("s0", {"static": .02, "dynamic": .02}, 256)
        with torch.no_grad():
            next_action, next_log_prob = agent.actor.sample(batch.next_observations)
            next_q = torch.min(
                agent.target_q1(batch.next_observations, next_action),
                agent.target_q2(batch.next_observations, next_action),
            )
            target = batch.rewards + agent.gamma * (1.0 - batch.dones) * (
                next_q - agent.alpha * next_log_prob
            )
            q1 = agent.q1(batch.observations, batch.actions)
            q2 = agent.q2(batch.observations, batch.actions)
            td = torch.maximum((q1 - target).abs(), (q2 - target).abs()).flatten()
            _, log_std = agent.actor(batch.observations)
        summary["critic"] = {
            "q1_mean": float(q1.mean().cpu()), "q2_mean": float(q2.mean().cpu()),
            "target_mean": float(target.mean().cpu()),
            "td_abs_p50": float(torch.quantile(td, .5).cpu()),
            "td_abs_p95": float(torch.quantile(td, .95).cpu()),
            "td_abs_max": float(td.max().cpu()),
            "all_finite": bool(torch.isfinite(torch.cat((q1.flatten(), q2.flatten(), target.flatten(), td))).all()),
        }
        summary["policy_std_by_action"] = [float(value) for value in log_std.exp().mean(dim=0).cpu()]
    return summary, rows


def write_results(output: Path, name: str, summary: dict, rows: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if rows:
        with (output / f"{name}_samples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("reward", "goal", "feasibility", "checkpoint"))
    parser.add_argument("--config", default="configs/experiments/thesis_homotopy.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    output = ROOT / args.output
    if args.mode == "reward":
        summary, rows = reward_audit(config, args.samples)
    elif args.mode == "goal":
        summary, rows = goal_kinematics_audit(config, args.samples)
    elif args.mode == "feasibility":
        summary, rows = feasibility_audit(config, args.samples)
    else:
        if not args.checkpoint:
            parser.error("checkpoint mode requires --checkpoint")
        summary, rows = checkpoint_audit(config, ROOT / args.checkpoint, args.samples)
    write_results(output, args.mode, summary, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
