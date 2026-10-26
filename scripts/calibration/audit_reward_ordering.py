#!/usr/bin/env python3
"""Audit the active scalar reward using controlled, explicit feature trajectories.

These are reward-function counterfactuals, not certified robot trajectories.
No policy, reward configuration, or running training process is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.thesis_reaching import homotopy_reward, pose_keypoint_errors
from rl_risk_sac.tasks.convex_sweep import swept_risk


def audit(config):
    env = ThesisHomotopyEnv(config)
    parameters = dict(env.reward_parameters)
    env.close()
    gamma = float(config["sac"]["gamma"])
    horizon = int(config["thesis"]["horizon"])
    hold_steps = int(config["thesis"]["success_hold_steps"])
    cube = float(config["thesis"]["observation"]["keypoint_cube_side_m"])
    all_results = {}
    components = ["keypoint_tracking_reward", "keypoint_progress_reward", "keypoint_precision_reward",
                  "joint_bottleneck_shaping_reward", "hold_reward", "leave_joint_tolerance_penalty",
                  "velocity_cost", "smooth_cost", "external_safety_penalty", "timeout_penalty",
                  "terminal_guard_penalty", "hard_penalty"]

    for level_index in (0, 6, 13):
        level = config["thesis"]["joint_pose_curriculum"]["levels"][level_index]
        pt, rt = level["position_tolerance_m"], level["orientation_tolerance_rad"]
        rp = dict(parameters, joint_position_tolerance=pt, joint_orientation_tolerance=rt)

        def features(p, angle):
            errors = pose_keypoint_errors(
                np.zeros(3), np.array([0., 0., 0., 1.]), np.array([p, 0., 0.]),
                np.array([0., 0., np.sin(angle / 2), np.cos(angle / 2)]), cube_side_m=cube)
            distances = np.linalg.norm(errors, axis=1)
            return float(distances.mean()), float(np.exp(-distances / rp["keypoint_tracking_sigma"]).mean())

        def rollout(states, *, distance=.2, velocity=.05, terminal=None):
            total = undiscounted = 0.
            sums = dict.fromkeys(components, 0.)
            holds = 0
            success = False
            for step, ((p, a), (pn, an)) in enumerate(zip(states, states[1:])):
                holds = holds + 1 if pn <= pt and an <= rt else 0
                success = holds >= hold_steps
                collision = terminal == "collision" and step == len(states) - 2
                hard = terminal == "self_collision" and step == len(states) - 2
                timeout = step + 1 >= horizon and not success and not collision and not hard
                d, _ = features(p, a)
                dn, quality = features(pn, an)
                clearance = distance if not collision else -.001
                risk = swept_risk(np.full((1, 37), clearance), **config["thesis"]["obstacles"]["risk"])
                reward, fields = homotopy_reward(
                    rho_position=p, next_rho_position=pn, rho_orientation=a, next_rho_orientation=an,
                    orientation_tolerance=rt, orientation_scale=1., smooth_velocity=0.,
                    velocity_magnitude=velocity, task_reached=success, hard_failure=hard,
                    timeout=timeout, obstacle_collision=collision, terminal_obstacle_collision=collision,
                    risk_max=risk, distance_min=clearance, xi=1., lambda_self=.2,
                    gamma=gamma, horizon=horizon, keypoint_tracking_quality=quality,
                    keypoint_distance=d, next_keypoint_distance=dn, **rp)
                total += gamma**step * reward
                undiscounted += reward
                for key in sums:
                    sums[key] += gamma**step * fields[key]
                if success or collision or hard or timeout:
                    break
            return dict(discounted_return=total, undiscounted_return=undiscounted,
                        steps=step + 1, success=success, components=sums)

        start = (max(.12, 2 * pt), max(.4, 2 * rt))
        near = (pt * 1.01, 0.)

        def interpolate(points, lengths):
            result = [points[0]]
            for a, b, count in zip(points, points[1:], lengths):
                result.extend(map(tuple, np.linspace(a, b, count + 1)[1:]))
            return result

        def success_route(steps, detour=False):
            moving = steps - hold_steps
            if detour:
                via = (start[0] + .15, start[1] + .4)
                states = interpolate([start, via, (0., 0.)], [moving // 2, moving - moving // 2])
            else:
                states = interpolate([start, (0., 0.)], [moving])
            return states + [(0., 0.)] * hold_steps

        results = {}
        for steps in (100, 250, 450):
            for detour in (False, True):
                label = f"{'detour' if detour else 'direct'}_success_{steps}"
                results[label] = rollout(success_route(steps, detour))
        results["stationary_timeout"] = rollout([start] * (horizon + 1), velocity=0.)
        approach = interpolate([start, near], [30])
        results["approach_then_stall_timeout"] = rollout(approach + [near] * (horizon - 30), velocity=0.)
        for position, angle, name in [(0., rt * 1.01, "position_only"), (pt * 1.01, 0., "orientation_only")]:
            end = (position, angle)
            route = interpolate([start, end], [30]) + [end] * (horizon - 30)
            results[f"{name}_timeout"] = rollout(route, velocity=0.)
        loop = [start]
        closer = (start[0] * .9, start[1])
        for _ in range(horizon // 20):
            loop.extend(interpolate([start, closer, start], [10, 10])[1:])
        results["approach_retreat_loop_timeout"] = rollout(loop)
        results["immediate_collision"] = rollout([start, start], terminal="collision", velocity=0.)
        results["late_collision_450"] = rollout([start] * 451, terminal="collision", velocity=0.)
        results["immediate_self_collision"] = rollout([start, start], terminal="self_collision", velocity=0.)
        same_route = success_route(100)
        results["safe_success"] = rollout(same_route, distance=.08)
        results["close_success"] = rollout(same_route, distance=.01)
        # Same starting near-goal state: finishing promptly versus stalling.
        results["near_finish_promptly"] = rollout(interpolate([near, (0., 0.)], [10]) + [(0., 0.)] * hold_steps, velocity=.01)
        results["near_stall_timeout"] = rollout([near] * (horizon + 1), velocity=0.)
        comparisons = []
        def compare(better, worse, requirement):
            delta = results[better]["discounted_return"] - results[worse]["discounted_return"]
            comparisons.append(dict(requirement=requirement, better=better, worse=worse,
                                    passed=bool(delta > 0), margin=delta))
        compare("direct_success_100", "detour_success_100", "unnecessary detour should not improve return")
        compare("safe_success", "close_success", "same motion: safer success preferred")
        compare("near_finish_promptly", "near_stall_timeout", "same near state: finish rather than stall")
        compare("stationary_timeout", "immediate_collision", "safe waiting preferred to immediate collision")
        compare("stationary_timeout", "approach_retreat_loop_timeout", "zero-net progress loop must not beat waiting")
        compare("approach_then_stall_timeout", "stationary_timeout", "partial progress preferred among safe timeouts")
        for steps in (100, 250, 450):
            for failure in ("stationary_timeout", "approach_then_stall_timeout", "position_only_timeout", "late_collision_450"):
                compare(f"detour_success_{steps}", failure, "successful detour preferred to failure from same start")
        all_results[f"L{level_index}"] = dict(scenarios=results, comparisons=comparisons)
    return dict(config="configs/experiments/paper_full.yaml", gamma=gamma, horizon=horizon,
                reward_scheme=parameters["reward_scheme"], reward_parameters=parameters,
                method="controlled reward-feature counterfactuals; not certified robot paths; no entropy bonus",
                levels=all_results)


def physical_audit(config):
    """Same real L13 state: hold, finish, or retreat then finish via actions.

    This no-obstacle diagnostic isolates the long-horizon ordering. The retreat
    is deliberate; it is not a certified necessary obstacle detour.
    """
    import copy
    config = copy.deepcopy(config)
    config["thesis"]["obstacles"]["enabled"] = False
    env = ThesisHomotopyEnv(config)
    try:
        level = config["thesis"]["joint_pose_curriculum"]["levels"][13]
        pt, rt = level["position_tolerance_m"], level["orientation_tolerance_rad"]
        env.configure_episode("none", xi=1., strict=True, position_tolerance=pt, orientation_tolerance=rt)
        env.reset(seed=110)
        goal = env.goal_joint_positions.copy()
        env._set_joint_state(goal)
        env.goal_position, env.goal_quaternion = env.robot.end_effector_pose()
        low, high = 0., .05
        for _ in range(30):
            delta = (low + high) / 2
            q = goal.copy()
            q[0] += delta
            env._set_joint_state(q)
            position, _ = env.robot.end_effector_pose()
            if np.linalg.norm(position - env.goal_position) < 1.2 * pt:
                low = delta
            else:
                high = delta
        q = goal.copy()
        q[0] += high
        env._set_joint_state(q)
        _, info, _, _, pe, oe = env._observe()
        env.previous_rho_position = float(np.linalg.norm(pe))
        env.previous_rho_orientation = float(np.linalg.norm(oe))
        env.previous_keypoint_distance = float(np.mean(info["keypoint_distances"]))
        env.step_count = env.physics_step_count = env.success_hold_count = 0
        state = env.episode_state_dict()
        results = {}
        for name in ("wait", "prompt_success", "retreat_then_success"):
            env.restore_episode_state(state)
            discounted = undiscounted = 0.
            for step in range(env.horizon):
                action = np.zeros(6)
                if name != "wait":
                    target = goal.copy()
                    if name == "retreat_then_success" and step < 160:
                        target[0] += .10
                    current, _ = env.robot.joint_state()
                    z = max(env.previous_rho_position / pt, env.previous_rho_orientation / rt)
                    scale = env.precision_min_action_scale + (
                        env.precision_strict_action_scale - env.precision_min_action_scale) * max(z, 0.) if z <= 1 else (
                        env.precision_strict_action_scale + (1 - env.precision_strict_action_scale) * min(z - 1, 1)**2)
                    action = np.clip(8 * (target - current) / (env.action_scale * scale), -1, 1)
                _, reward, _, terminated, truncated, info = env.step(action)
                discounted += config["sac"]["gamma"]**step * reward
                undiscounted += reward
                if terminated or truncated:
                    break
            results[name] = dict(discounted_return=discounted, undiscounted_return=undiscounted,
                                 steps=step + 1, success=bool(info["task_reached"]),
                                 collision=bool(info["collision"]), timeout=bool(info["timeout"]),
                                 final_position_error_m=info["next_rho_position"],
                                 final_orientation_error_rad=info["next_rho_orientation"])
        return dict(method="actual PyBullet steps; same initial snapshot; no obstacles; deterministic joint controllers",
                    start_position_error_m=state["previous_rho_position"], results=results)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--physical", action="store_true", help="also run same-start real robot controllers")
    parser.add_argument("--legacy-baseline", action="store_true", help="reproduce legacy_dense with gamma=0.99")
    args = parser.parse_args()
    config = load_config(ROOT / "configs/experiments/paper_full.yaml")
    if args.legacy_baseline:
        config["thesis"]["reward"]["scheme"] = "legacy_dense"
        config["sac"]["gamma"] = .99
    result = audit(config)
    if args.physical:
        result["physical"] = physical_audit(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for level, values in result["levels"].items():
        print(level)
        for name, row in values["scenarios"].items():
            print(name, round(row["discounted_return"], 5), row["steps"], row["success"])
        print("FAILED", [(c["better"], c["worse"], round(c["margin"], 5)) for c in values["comparisons"] if not c["passed"]])
