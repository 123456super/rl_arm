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

ROOT = Path(__file__).resolve().parents[2]
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
                smooth_velocity=0.0, velocity_magnitude=0.0, orientation_scale=1.0,
                task_reached=False, hard_failure=False,
                obstacle_collision=True, terminal_obstacle_collision=True,
                risk_max=1.0, distance_min=0.0, xi=1.0,
                gamma=gamma, horizon=horizon,
                **env.reward_parameters,
            )
            tolerant_collision, tolerant_collision_fields = homotopy_reward(
                rho_position=rho_p, next_rho_position=rho_p,
                rho_orientation=rho_r, next_rho_orientation=rho_r,
                smooth_velocity=0.0, velocity_magnitude=0.0, orientation_scale=1.0,
                task_reached=False, hard_failure=False,
                obstacle_collision=True, terminal_obstacle_collision=False,
                risk_max=1.0, distance_min=0.0, xi=0.02,
                gamma=gamma, horizon=horizon,
                **env.reward_parameters,
            )
            hard_failure, hard_failure_fields = homotopy_reward(
                rho_position=rho_p, next_rho_position=rho_p,
                rho_orientation=rho_r, next_rho_orientation=rho_r,
                smooth_velocity=0.0, velocity_magnitude=0.0, orientation_scale=1.0,
                task_reached=False, hard_failure=True,
                obstacle_collision=False, risk_max=0.0, distance_min=0.80, xi=1.0,
                gamma=gamma, horizon=horizon,
                **env.reward_parameters,
            )
            wait_step, _ = homotopy_reward(
                rho_position=rho_p, next_rho_position=rho_p,
                rho_orientation=rho_r, next_rho_orientation=rho_r,
                smooth_velocity=0.0, velocity_magnitude=0.0, orientation_scale=1.0,
                task_reached=False, hard_failure=False,
                obstacle_collision=False, risk_max=0.0, distance_min=0.80, xi=1.0,
                gamma=gamma, horizon=horizon,
                **env.reward_parameters,
            )
            timeout_step, _ = homotopy_reward(
                rho_position=rho_p, next_rho_position=rho_p,
                rho_orientation=rho_r, next_rho_orientation=rho_r,
                smooth_velocity=0.0, velocity_magnitude=0.0, orientation_scale=1.0,
                task_reached=False, hard_failure=False, timeout=True,
                obstacle_collision=False, risk_max=0.0, distance_min=0.80, xi=1.0,
                gamma=gamma, horizon=horizon,
                **env.reward_parameters,
            )
            timeout = discounted([wait_step] * (horizon - 1) + [timeout_step], gamma)
            def trajectory_return(
                positions: list[float], orientations: list[float], *, success: bool,
            ) -> float:
                values: list[float] = []
                previous_p, previous_r = rho_p, rho_r
                for step, (next_p, next_r) in enumerate(
                    zip(positions, orientations), start=1
                ):
                    final = step == len(positions)
                    reward, _ = homotopy_reward(
                        rho_position=previous_p, next_rho_position=next_p,
                        rho_orientation=previous_r, next_rho_orientation=next_r,
                        smooth_velocity=0.0, velocity_magnitude=0.0,
                        orientation_scale=1.0,
                        task_reached=bool(success and final),
                        hard_failure=False, timeout=bool(not success and final),
                        obstacle_collision=False, risk_max=0.0,
                        distance_min=0.80, xi=1.0, gamma=gamma, horizon=horizon,
                        **env.reward_parameters,
                    )
                    values.append(reward)
                    previous_p, previous_r = next_p, next_r
                return discounted(values, gamma)

            ideal_returns = {}
            for steps in (60, 120, 180, 239, 240):
                fractions = [1.0 - step / steps for step in range(1, steps + 1)]
                ideal_returns[steps] = trajectory_return(
                    [rho_p * value for value in fractions],
                    [rho_r * value for value in fractions], success=True,
                )
            half = horizon // 2
            retreat_fractions = (
                [1.0 - .5 * step / half for step in range(1, half + 1)]
                + [.5 + .5 * step / (horizon - half) for step in range(1, horizon - half + 1)]
            )
            approach_then_retreat = trajectory_return(
                [rho_p * value for value in retreat_fractions],
                [rho_r * value for value in retreat_fractions], success=False,
            )
            oscillation_fractions = [
                .75 if step % 2 else 1.0 for step in range(1, horizon + 1)
            ]
            periodic_oscillation = trajectory_return(
                [rho_p * value for value in oscillation_fractions],
                [rho_r * value for value in oscillation_fractions], success=False,
            )
            ideal_success = ideal_returns[60]
            rows.append({
                "sample": index, "rho_position": rho_p, "rho_orientation": rho_r,
                "strict_collision_now": strict_collision,
                "tolerant_collision_step": tolerant_collision,
                "strict_collision_guard": strict_collision_fields["terminal_guard_penalty"],
                "hard_failure_now": hard_failure,
                "hard_failure_guard": hard_failure_fields["terminal_guard_penalty"],
                "stationary_timeout": timeout, "ideal_60_step_success": ideal_success,
                **{
                    f"ideal_{steps}_step_success": value
                    for steps, value in ideal_returns.items() if steps != 60
                },
                "approach_then_retreat_timeout": approach_then_retreat,
                "periodic_oscillation_timeout": periodic_oscillation,
                "strict_collision_better_than_timeout": strict_collision > timeout,
                "hard_failure_better_than_timeout": hard_failure > timeout,
                "desired_order": ideal_success > timeout > max(strict_collision, hard_failure),
                "all_successes_beat_failure_loops": min(ideal_returns.values())
                > max(approach_then_retreat, periodic_oscillation),
            })
    finally:
        env.close()
    summary = {
        "samples": samples,
        "strict_collision_better_than_timeout_rate": float(np.mean([r["strict_collision_better_than_timeout"] for r in rows])),
        "hard_failure_better_than_timeout_rate": float(np.mean([r["hard_failure_better_than_timeout"] for r in rows])),
        "desired_order_rate": float(np.mean([r["desired_order"] for r in rows])),
        "all_successes_beat_failure_loops_rate": float(np.mean([
            r["all_successes_beat_failure_loops"] for r in rows
        ])),
        "median_returns": {
            key: float(np.median([r[key] for r in rows]))
            for key in (
                "strict_collision_now", "tolerant_collision_step", "hard_failure_now",
                "stationary_timeout", "ideal_60_step_success",
                "ideal_120_step_success", "ideal_180_step_success",
                "ideal_239_step_success", "ideal_240_step_success",
                "approach_then_retreat_timeout", "periodic_oscillation_timeout",
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
    """Audit the configured task-space pose -> IK -> FK acceptance contract."""
    env = ThesisHomotopyEnv(config)
    levels = config["thesis"].get("joint_pose_curriculum", {}).get("levels", [])
    rows: list[dict] = []
    failures = 0
    try:
        for sample in range(samples):
            try:
                level_index = sample % len(levels) if levels else 0
                level = levels[level_index] if levels else {}
                task_bounds = {
                    key: level[key] for key in (
                        "target_distance_min_m", "target_distance_max_m",
                        "target_orientation_min_rad", "target_orientation_max_rad",
                    ) if key in level
                }
                env.configure_episode(
                    "none", xi=1.0, strict=True,
                    goal_scale=float(level.get("goal_scale", 1.0)),
                    position_tolerance=float(level.get(
                        "position_tolerance_m", config["thesis"]["position_tolerance"]
                    )),
                    orientation_tolerance=float(level.get(
                        "orientation_tolerance_rad", config["thesis"]["orientation_tolerance"]
                    )),
                    **task_bounds,
                )
                _, info = env.reset(seed=int(config["seed"]) + sample)
                validation = info["goal_ik_validation"]
                rows.append({
                    "sample": sample,
                    "level": level_index,
                    "accepted": bool(validation["reachable"]),
                    "attempts": int(info["goal_sample_attempts"]),
                    "ik_finite": bool(validation["finite"]),
                    "ik_within_limits": bool(validation["within_limits"]),
                    "position_error_m": float(validation["position_error_m"]),
                    "orientation_error_rad": float(validation["orientation_error_rad"]),
                    "target_distance_m": float(info.get("sampled_target_distance_m", 0.0)),
                    "target_orientation_rad": float(info.get("sampled_target_orientation_rad", 0.0)),
                })
            except RuntimeError:
                failures += 1
                rows.append({
                    "sample": sample, "level": level_index, "accepted": False,
                    "attempts": int(config["thesis"].get("goal_sample_max_attempts", 10000)),
                    "ik_finite": False, "ik_within_limits": False,
                    "position_error_m": float("inf"), "orientation_error_rad": float("inf"),
                    "target_distance_m": float("nan"), "target_orientation_rad": float("nan"),
                })
    finally:
        env.close()
    accepted = [row for row in rows if row["accepted"]]
    summary = {
        "samples": samples,
        "reset_failures": failures,
        "acceptance_rate": float(len(accepted) / samples),
        "thresholds": {
            "position_error_m": float(config["thesis"]["position_tolerance"]),
            "orientation_error_rad": float(config["thesis"]["orientation_tolerance"]),
        },
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


def contact_geometry_audit(config: dict, samples: int) -> tuple[dict, list[dict]]:
    """Cross-check shallow real contacts against the capsule safety geometry."""
    env = ThesisHomotopyEnv(config)
    rows: list[dict] = []
    try:
        for sample in range(samples):
            env.configure_episode("static", xi=1.0, strict=True)
            env.reset(seed=int(config["seed"]) + sample)
            for capsule in env.robot.capsules():
                center = 0.5 * (capsule.start + capsule.end)
                axis = capsule.end - capsule.start
                if float(np.linalg.norm(axis)) <= 1e-8:
                    perpendicular = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
                else:
                    axis = axis / np.linalg.norm(axis)
                    basis = np.eye(3, dtype=np.float32)[int(np.argmin(np.abs(axis)))]
                    perpendicular = np.cross(axis, basis)
                    perpendicular = perpendicular / np.linalg.norm(perpendicular)

                def move_obstacle(position: np.ndarray) -> bool:
                    env.obstacle_position = position.astype(np.float32)
                    env.obstacle_velocity.fill(0.0)
                    p.resetBasePositionAndOrientation(
                        env.obstacle_id, position.tolist(), [0.0, 0.0, 0.0, 1.0],
                        physicsClientId=env.client_id,
                    )
                    p.performCollisionDetection(physicsClientId=env.client_id)
                    return bool(env._collision_events()["obstacle_collision"])

                candidates = [
                    perpendicular, -perpendicular,
                    np.asarray([1.0, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0]),
                    np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, -1.0, 0.0]),
                    np.asarray([0.0, 0.0, 1.0]), np.asarray([0.0, 0.0, -1.0]),
                ]
                directions = []
                for direction in candidates:
                    if any(np.allclose(direction, selected) for selected in directions):
                        continue
                    if not move_obstacle(center + 0.6 * direction):
                        directions.append(direction)
                    if len(directions) == 2:
                        break
                for direction_index, direction in enumerate(directions):
                    center_contact = move_obstacle(center)
                    low, high = 0.0, 0.6
                    if center_contact:
                        for _ in range(24):
                            middle = 0.5 * (low + high)
                            if move_obstacle(center + middle * direction):
                                low = middle
                            else:
                                high = middle
                        # Query just inside the contact boundary.  Re-querying
                        # the exact bisection endpoint can flip to non-contact
                        # because Bullet and the numpy pose use different
                        # floating-point precision at a zero-depth boundary.
                        contact_offset = max(0.0, low - 1e-4)
                        contact = move_obstacle(center + contact_offset * direction)
                    else:
                        contact = False
                    geometry = env._geometry()
                    if geometry is None:
                        raise RuntimeError("static contact audit requires obstacle geometry")
                    consistent = bool(
                        contact and geometry.distance_min < env.d_safe and geometry.risk_max >= 0.79
                    )
                    rows.append({
                        "sample": sample,
                        "capsule": capsule.name,
                        "direction": direction_index,
                        "pybullet_contact": contact,
                        "contact_generated": contact,
                        "distance_min_m": geometry.distance_min,
                        "risk_max": geometry.risk_max,
                        "consistent": consistent if contact else None,
                    })
                if len(directions) < 2:
                    rows.append({
                        "sample": sample, "capsule": capsule.name, "direction": -1,
                        "pybullet_contact": False, "contact_generated": False,
                        "distance_min_m": float("inf"), "risk_max": 0.0,
                        "consistent": None,
                    })
    finally:
        env.close()
    contact_rows = [row for row in rows if row["contact_generated"]]
    return {
        "samples": samples,
        "checks": len(rows),
        "contact_witnesses": len(contact_rows),
        "contact_generation_rate": float(len(contact_rows) / len(rows)),
        "consistency_rate": float(np.mean([row["consistent"] for row in contact_rows])),
        "maximum_distance_at_contact_m": float(
            max(row["distance_min_m"] for row in contact_rows)
        ),
        "minimum_risk_at_contact": float(min(row["risk_max"] for row in contact_rows)),
        "requirements": {"distance_min_m_max_exclusive": env.d_safe, "risk_min": 0.79},
    }, rows


def self_contact_geometry_audit(config: dict, samples: int) -> tuple[dict, list[dict]]:
    """Cross-check exact self-contact events against mesh clearance and risk."""
    env = ThesisHomotopyEnv(config)
    rows: list[dict] = []
    rng = np.random.default_rng(int(config["seed"]) + 917)
    try:
        env.configure_episode("none", xi=1.0, strict=True)
        env.reset(seed=int(config["seed"]) + 917)
        for sample in range(samples):
            q = rng.uniform(
                env.robot.joint_lower_limits + 0.1,
                env.robot.joint_upper_limits - 0.1,
            )
            env._set_joint_state(q)
            p.performCollisionDetection(physicsClientId=env.client_id)
            contact = env._collision_events()["self_collision"]
            geometry = env._self_geometry()
            mesh_contact = geometry.distance_min <= 0.0
            rows.append({
                "sample": sample,
                "pybullet_contact": contact,
                "mesh_contact": mesh_contact,
                "distance_min_m": geometry.distance_min,
                "risk_max": geometry.risk_max,
                "ttc_min_s": geometry.ttc_min,
                "approach_max_mps": geometry.approach_max,
                "closest_first_slot": geometry.closest_pair[0],
                "closest_second_slot": geometry.closest_pair[1],
                "consistent": bool(
                    contact == mesh_contact
                    and (not contact or np.isclose(geometry.risk_max, 1.0))
                ),
            })
    finally:
        env.close()
    contacts = [row for row in rows if row["pybullet_contact"]]
    safe = [row for row in rows if not row["pybullet_contact"]]
    return {
        "samples": samples,
        "contacts": len(contacts),
        "collision_free": len(safe),
        "consistency_rate": float(np.mean([row["consistent"] for row in rows])),
        "maximum_contact_distance_m": (
            float(max(row["distance_min_m"] for row in contacts)) if contacts else None
        ),
        "minimum_collision_free_distance_m": (
            float(min(row["distance_min_m"] for row in safe)) if safe else None
        ),
        "safe_distance_m": env.d_self_safe,
        "query_distance_m": env.self_query_distance,
    }, rows


def checkpoint_audit(config: dict, checkpoint: Path, samples: int) -> tuple[dict, list[dict]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("protocol") != config["thesis"]["protocol"]:
        raise ValueError(
            f"checkpoint protocol {state.get('protocol')!r} does not match "
            f"configured protocol {config['thesis']['protocol']!r}"
        )
    env = ThesisHomotopyEnv(config)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    agent = ThesisSACAgent(obs_dim, action_dim, config)
    agent.load_state_dict(state["agent"])
    replay = HomotopyReplayBuffer(
        obs_dim, action_dim, str(agent.device), seed=int(config["seed"]),
        reward_gamma=float(config["sac"]["gamma"]),
        reward_horizon=int(config["thesis"]["horizon"]),
        reward_parameters=env.reward_parameters,
    )
    replay.load_state_dict(state["replay"])
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
                joint_upper=env.robot.joint_upper_limits, joint_velocity_scale=env.action_scale,
                ee_position=info["ee_position"], ee_quaternion=info["ee_quaternion"],
                ee_linear_velocity=info["ee_linear_velocity"],
                ee_angular_velocity=info["ee_angular_velocity"],
                ee_linear_velocity_scale=env.ee_linear_velocity_scale,
                ee_angular_velocity_scale=env.ee_angular_velocity_scale,
                goal_position=info["goal_position"], goal_quaternion=info["goal_quaternion"],
                position_error=error_p, orientation_error=error_r,
                obstacle_present=False,
                obstacle_position=np.zeros(3), obstacle_velocity=np.zeros(3), geometry=None,
                self_geometry=env._self_geometry(),
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
    if replay.scene_size("none") >= 256:
        self_safety = state["curriculum"].get("self_safety")
        if self_safety is None:
            raise ValueError("checkpoint lacks the self-safety curriculum state")
        batch = replay.sample(
            "s0", {"static": .02, "dynamic": .02}, 256,
            lambda_self=float(self_safety.weight),
        )
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
    parser.add_argument(
        "mode",
        choices=("reward", "goal", "contact", "self-contact", "feasibility", "checkpoint"),
    )
    parser.add_argument("--config", default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
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
    elif args.mode == "contact":
        summary, rows = contact_geometry_audit(config, args.samples)
    elif args.mode == "self-contact":
        summary, rows = self_contact_geometry_audit(config, args.samples)
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
