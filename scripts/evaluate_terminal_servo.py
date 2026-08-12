from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

from rl_risk_sac.algorithms import SACAgent
from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.device import resolve_device
from rl_risk_sac.utils.seeding import set_seed

try:
    from scripts.seed_manifest import load_seed_manifest
except ModuleNotFoundError:
    from seed_manifest import load_seed_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--trace-output", default=None)
    parser.add_argument("--servo-trigger-error-m", type=float, default=0.10)
    parser.add_argument("--servo-full-error-m", type=float, default=0.055)
    parser.add_argument("--servo-clearance-margin-m", type=float, default=0.03)
    parser.add_argument("--servo-gain", type=float, default=3.0)
    parser.add_argument("--servo-damping", type=float, default=0.05)
    parser.add_argument("--servo-max-speed-radps", type=float, default=None)
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _servo_blend(goal_error_norm: float, trigger_m: float, full_m: float) -> float:
    if not np.isfinite(goal_error_norm):
        return 0.0
    if trigger_m <= full_m:
        raise ValueError("servo trigger threshold must be greater than the full-blend threshold")
    if goal_error_norm >= trigger_m:
        return 0.0
    if goal_error_norm <= full_m:
        return 1.0
    return float((trigger_m - goal_error_norm) / (trigger_m - full_m))


def _terminal_servo_action(
    env: UR5DynamicObstacleEnv,
    observation: np.ndarray,
    policy_action: np.ndarray,
    latest_info: dict[str, object],
    *,
    trigger_error_m: float,
    full_error_m: float,
    clearance_margin_m: float,
    gain: float,
    damping: float,
    max_speed_radps: float | None,
) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    joint_count = env.joint_count
    goal_error = np.asarray(env._goal_error(), dtype=np.float64)
    goal_error_norm = float(np.linalg.norm(goal_error))
    d_min = float(latest_info.get("d_min", float("nan")))
    safe_margin = float(env.risk_config.d_safe) + float(clearance_margin_m)
    safe_to_use = np.isfinite(d_min) and d_min > safe_margin
    blend = _servo_blend(goal_error_norm, trigger_error_m, full_error_m) if safe_to_use else 0.0
    servo_reason = "inactive"
    servo_qdot = np.zeros(joint_count, dtype=np.float64)
    if not safe_to_use:
        servo_reason = "insufficient_clearance"
    elif blend > 0.0:
        try:
            ee_jacobian = env._link_origin_jacobian(env.tool_link_id)
            jj_t = ee_jacobian @ ee_jacobian.T
            regularizer = float(damping) ** 2
            desired_twist = float(gain) * goal_error
            servo_qdot = ee_jacobian.T @ np.linalg.solve(jj_t + regularizer * np.eye(3), desired_twist)
            servo_reason = "active"
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as error:
            servo_reason = f"jacobian_failure:{error}"
            blend = 0.0

    max_speed = float(env.action_scale if max_speed_radps is None else max_speed_radps)
    if not np.isfinite(max_speed) or max_speed <= 0.0:
        raise ValueError("servo max speed must be finite and positive")

    policy_qdot = np.asarray(policy_action, dtype=np.float64) * float(env.action_scale)
    qdot_cmd = (1.0 - blend) * policy_qdot + blend * servo_qdot
    qdot_cmd = np.clip(qdot_cmd, -max_speed, max_speed)
    action = np.clip(qdot_cmd / float(env.action_scale), -1.0, 1.0).astype(np.float32)
    return action, {
        "servo_active": bool(blend > 0.0 and servo_reason == "active"),
        "servo_blend": float(blend),
        "servo_safe_to_use": bool(safe_to_use),
        "servo_reason": servo_reason,
        "servo_goal_error_norm": goal_error_norm,
        "servo_d_min_m": d_min,
        "servo_safe_margin_m": safe_margin,
        "policy_qdot_norm": float(np.linalg.norm(policy_qdot)),
        "servo_qdot_norm": float(np.linalg.norm(servo_qdot)),
        "executed_qdot_norm": float(np.linalg.norm(qdot_cmd)),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = copy.deepcopy(config)
    if "residual_control" in config["env"]:
        config["env"]["residual_control"]["enabled"] = False
    eval_cfg = config.get("eval", {})
    method = str(eval_cfg.get("method", config.get("train", {}).get("method", "link_fixed")))
    checkpoint = args.checkpoint or eval_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("checkpoint must be provided by --checkpoint or eval.checkpoint")
    episodes = int(args.episodes if args.episodes is not None else eval_cfg.get("episodes", 20))
    seed = int(args.seed if args.seed is not None else eval_cfg.get("seed", 123))
    seed_manifest_arg = args.seed_manifest if args.seed_manifest is not None else eval_cfg.get("seed_manifest")
    manifest_seeds = load_seed_manifest(seed_manifest_arg) if seed_manifest_arg else None
    if episodes <= 0:
        raise ValueError("Evaluation episodes must be positive")
    if manifest_seeds is not None and episodes > len(manifest_seeds):
        raise ValueError(
            f"Requested {episodes} episodes but seed manifest only contains {len(manifest_seeds)} seeds"
        )
    episode_seeds = (
        manifest_seeds[:episodes]
        if manifest_seeds is not None
        else [seed + episode for episode in range(episodes)]
    )

    config["seed"] = seed
    config["device"] = resolve_device(config)
    set_seed(seed)

    env = UR5DynamicObstacleEnv(config, method=method)
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], config, method=method)
    agent.load_actor(checkpoint)

    rows = []
    trace_dir = Path(args.trace_output) if args.trace_output else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)

    for episode, episode_seed in enumerate(episode_seeds):
        observation, reset_info = env.reset(seed=episode_seed)
        latest_info = dict(reset_info)
        total_reward = 0.0
        total_cost = 0.0
        risks: list[float] = []
        distances: list[float] = []
        accelerations: list[np.ndarray] = []
        jerks: list[np.ndarray] = []
        action_variations: list[float] = []
        success = False
        collision = False
        collision_capsule_overlap = False
        collision_pybullet_contact = False
        termination_collision = False
        termination_reason = ""
        final_position_error = 0.0
        prev_qdot_cmd = np.zeros(env.action_space.shape[0], dtype=np.float32)
        servo_active_steps = 0
        servo_blends: list[float] = []
        servo_qdot_norms: list[float] = []
        policy_qdot_norms: list[float] = []
        servo_safe_steps = 0
        trace_rows = []

        for step in range(int(config["env"]["max_episode_steps"])):
            policy_action = agent.select_action(observation, deterministic=True)
            action, servo_meta = _terminal_servo_action(
                env,
                observation,
                policy_action,
                latest_info,
                trigger_error_m=float(args.servo_trigger_error_m),
                full_error_m=float(args.servo_full_error_m),
                clearance_margin_m=float(args.servo_clearance_margin_m),
                gain=float(args.servo_gain),
                damping=float(args.servo_damping),
                max_speed_radps=args.servo_max_speed_radps,
            )
            observation, reward, cost, terminated, truncated, info = env.step(action)
            latest_info = info

            total_reward += reward
            total_cost += cost
            risks.append(float(info["risk_global"]))
            distances.append(float(info["d_min"]))
            accelerations.append(np.asarray(info["joint_acc"], dtype=np.float64))
            jerks.append(np.asarray(info["joint_jerk"], dtype=np.float64))
            action_variations.append(float(np.linalg.norm(info["qdot_cmd"] - prev_qdot_cmd)))
            success = bool(info["success"])
            collision = collision or bool(info["collision_any"])
            collision_capsule_overlap = collision_capsule_overlap or bool(info["collision_capsule_overlap"])
            collision_pybullet_contact = collision_pybullet_contact or bool(info["collision_pybullet_contact"])
            if bool(info.get("termination_collision", False)):
                termination_collision = True
                termination_reason = str(info.get("termination_reason", ""))
            final_position_error = float(info["goal_error_norm"])
            if bool(servo_meta["servo_active"]):
                servo_active_steps += 1
            if bool(servo_meta["servo_safe_to_use"]):
                servo_safe_steps += 1
            servo_blends.append(float(servo_meta["servo_blend"]))
            servo_qdot_norms.append(float(servo_meta["servo_qdot_norm"]))
            policy_qdot_norms.append(float(servo_meta["policy_qdot_norm"]))

            if trace_dir is not None:
                trace_rows.append(
                    {
                        "step": step,
                        "time": (step + 1) * float(config["env"]["control_dt"]),
                        "goal_error_norm": final_position_error,
                        "d_min": float(info["d_min"]),
                        "risk_global": float(info["risk_global"]),
                        "qdot_norm": float(np.linalg.norm(info["qdot_cmd"])),
                        "policy_qdot_norm": float(servo_meta["policy_qdot_norm"]),
                        "servo_qdot_norm": float(servo_meta["servo_qdot_norm"]),
                        "servo_active": int(bool(servo_meta["servo_active"])),
                        "servo_blend": float(servo_meta["servo_blend"]),
                        "servo_safe_to_use": int(bool(servo_meta["servo_safe_to_use"])),
                        "servo_reason": servo_meta["servo_reason"],
                    }
                )
            prev_qdot_cmd = np.asarray(info["qdot_cmd"], dtype=np.float32)
            if terminated or truncated:
                break

        rows.append(
            {
                "episode": episode,
                "seed": episode_seed,
                "seed_manifest": str(seed_manifest_arg) if seed_manifest_arg else "",
                "reward": total_reward,
                "cost": total_cost,
                "length": step + 1,
                "success": int(success),
                "collision": int(collision),
                "collision_any": int(collision),
                "collision_capsule_overlap": int(collision_capsule_overlap),
                "collision_pybullet_contact": int(collision_pybullet_contact),
                "termination_collision": int(termination_collision),
                "termination_reason": termination_reason,
                "final_position_error": final_position_error,
                "completion_time": (step + 1) * float(config["env"]["control_dt"]),
                "min_distance": min(distances) if distances else float("nan"),
                "mean_risk": _mean(risks),
                "mean_action_variation": float(np.mean(action_variations)) if action_variations else float("nan"),
                "servo_active_rate": servo_active_steps / max(step + 1, 1),
                "servo_safe_rate": servo_safe_steps / max(step + 1, 1),
                "mean_servo_blend": _mean(servo_blends),
                "mean_servo_qdot_norm": _mean(servo_qdot_norms),
                "mean_policy_qdot_norm": _mean(policy_qdot_norms),
            }
        )
        if trace_dir is not None and trace_rows:
            trace_path = trace_dir / f"episode_{episode:04d}.csv"
            with trace_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(trace_rows[0].keys()))
                writer.writeheader()
                writer.writerows(trace_rows)

    env.close()
    output = Path(args.output) if args.output else Path(checkpoint).resolve().parent / "eval_terminal_servo.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "collision_rate": float(np.mean([row["collision"] for row in rows])),
        "collision_any_rate": float(np.mean([row["collision_any"] for row in rows])),
        "collision_capsule_overlap_rate": float(np.mean([row["collision_capsule_overlap"] for row in rows])),
        "collision_pybullet_contact_rate": float(np.mean([row["collision_pybullet_contact"] for row in rows])),
        "termination_collision_rate": float(np.mean([row["termination_collision"] for row in rows])),
        "mean_final_position_error": float(np.mean([row["final_position_error"] for row in rows])),
        "mean_min_distance": float(np.mean([row["min_distance"] for row in rows])),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "mean_cost": float(np.mean([row["cost"] for row in rows])),
        "mean_servo_active_rate": float(np.mean([row["servo_active_rate"] for row in rows])),
        "mean_servo_safe_rate": float(np.mean([row["servo_safe_rate"] for row in rows])),
        "mean_servo_blend": float(np.mean([row["mean_servo_blend"] for row in rows])),
        "mean_servo_qdot_norm": float(np.mean([row["mean_servo_qdot_norm"] for row in rows])),
        "mean_policy_qdot_norm": float(np.mean([row["mean_policy_qdot_norm"] for row in rows])),
    }
    print(json.dumps(summary, ensure_ascii=True))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
