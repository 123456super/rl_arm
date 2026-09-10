from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from rl_risk_sac.algorithms import SACAgent
from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.device import resolve_device
from rl_risk_sac.utils.seeding import derive_episode_seed, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--method",
        default=None,
        choices=["ee_fixed", "link_fixed", "predictive_link", "ldrc_fixed", "ldrc_adaptive"],
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--trace-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    eval_cfg = config.get("eval", {})
    method = args.method or eval_cfg["method"]
    checkpoint = args.checkpoint or eval_cfg["checkpoint"]
    if not checkpoint:
        raise ValueError("checkpoint must be provided by --checkpoint or eval.checkpoint")
    episodes = int(args.episodes if args.episodes is not None else eval_cfg.get("episodes", 20))
    seed = int(args.seed if args.seed is not None else eval_cfg.get("seed", 123))
    output_arg = args.output if args.output is not None else eval_cfg.get("output")
    trace_output_arg = args.trace_output if args.trace_output is not None else eval_cfg.get("trace_output")
    config["seed"] = seed
    config["device"] = resolve_device(config)
    set_seed(seed)

    env = UR5DynamicObstacleEnv(config, method=method)
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], config, method=method)
    agent.load_actor(checkpoint)

    rows = []
    trace_dir = Path(trace_output_arg) if trace_output_arg else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(episodes):
        episode_seed = derive_episode_seed(seed, episode)
        observation, _ = env.reset(seed=episode_seed)
        total_reward = 0.0
        total_cost = 0.0
        risks = []
        distances = []
        violations = 0
        accelerations = []
        jerks = []
        command_acceleration_substeps = []
        command_jerk_substeps = []
        measured_acceleration_substeps = []
        measured_jerk_substeps = []
        action_variations = []
        policy_rate_limit_events = 0
        safety_qp_interventions = 0
        safety_qp_infeasible = 0
        safety_qp_correction_norms = []
        safety_qp_solve_times = []
        success = False
        collision = False
        final_position_error = 0.0
        closest_link = -1
        prev_qdot_cmd = np.zeros(env.action_space.shape[0], dtype=np.float32)
        trace_rows = []
        physics_trace_rows = []
        for step in range(int(config["env"]["max_episode_steps"])):
            action = agent.select_action(observation, deterministic=True)
            observation, reward, cost, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_cost += cost
            risks.append(float(info["control_max_risk"]))
            distances.append(float(info["control_min_distance"]))
            violations += int(info["control_safety_violation"])
            accelerations.append(info["joint_acc"])
            jerks.append(info["joint_jerk"])
            command_acceleration_substeps.append(info["command_acc_substeps"])
            command_jerk_substeps.append(info["command_jerk_substeps"])
            measured_acceleration_substeps.append(info["measured_acc_substeps"])
            measured_jerk_substeps.append(info["measured_jerk_substeps"])
            action_variations.append(float(np.linalg.norm(info["qdot_cmd"] - prev_qdot_cmd)))
            policy_rate_limit_events += int(info["policy_rate_limited"])
            safety_qp_interventions += int(info.get("safety_qp_intervened", False))
            safety_qp_infeasible += int(info.get("safety_qp_infeasible", False))
            safety_qp_correction_norms.append(float(info.get("safety_qp_correction_norm", 0.0)))
            safety_qp_solve_times.append(float(info.get("safety_qp_solve_time_ms", 0.0)))
            success = bool(info["success"])
            collision = bool(info["collision"])
            final_position_error = float(info["goal_error_norm"])
            closest_link = int(info["closest_link"])
            if trace_dir is not None:
                trace_rows.append(
                    {
                        "step": step,
                        "time": (step + 1) * float(config["env"]["control_dt"]),
                        "reward": float(reward),
                        "cost": float(cost),
                        "goal_error_norm": final_position_error,
                        "d_min": float(info["d_min"]),
                        "risk_global": float(info["risk_global"]),
                        "control_min_distance": float(info["control_min_distance"]),
                        "control_max_risk": float(info["control_max_risk"]),
                        "beta": float(info["beta"]),
                        "closest_link": closest_link,
                        "safety_violation": int(info["control_safety_violation"]),
                        "collision": int(info["collision"]),
                        "success": int(info["success"]),
                        "qdot_norm": float(np.linalg.norm(info["qdot_cmd"])),
                        "qdot_policy_norm": float(np.linalg.norm(info["qdot_policy"])),
                        "qdot_policy_limited_norm": float(np.linalg.norm(info["qdot_policy_limited"])),
                        "qdot_policy_safe_target_norm": float(
                            np.linalg.norm(info.get("qdot_policy_safe_target", info["qdot_policy_limited"]))
                        ),
                        "policy_rate_limited": int(info["policy_rate_limited"]),
                        "safety_qp_intervened": int(info.get("safety_qp_intervened", False)),
                        "safety_qp_infeasible": int(info.get("safety_qp_infeasible", False)),
                        "safety_qp_correction_norm": float(info.get("safety_qp_correction_norm", 0.0)),
                        "safety_qp_solve_time_ms": float(info.get("safety_qp_solve_time_ms", 0.0)),
                        "acc_norm": float(np.linalg.norm(info["joint_acc"])),
                        "jerk_norm": float(np.linalg.norm(info["joint_jerk"])),
                        "command_rms_acceleration": float(info["command_rms_acceleration"]),
                        "command_rms_jerk": float(info["command_rms_jerk"]),
                        "measured_rms_acceleration": float(info["measured_rms_acceleration"]),
                        "measured_rms_jerk": float(info["measured_rms_jerk"]),
                    }
                )
                for substep, (
                    command_qdot,
                    command_acc,
                    command_jerk,
                    measured_qdot,
                    measured_acc,
                    measured_jerk,
                    substep_distance,
                    substep_risk,
                    substep_contact,
                ) in enumerate(
                    zip(
                        info["qdot_substeps"],
                        info["command_acc_substeps"],
                        info["command_jerk_substeps"],
                        info["measured_qdot_substeps"],
                        info["measured_acc_substeps"],
                        info["measured_jerk_substeps"],
                        info["substep_distance"],
                        info["substep_risk"],
                        info["substep_contact"],
                    )
                ):
                    physics_row = {
                        "physics_step": step * env.sim_substeps + substep,
                        "time": step * float(config["env"]["control_dt"])
                        + (substep + 1) * float(config["env"]["time_step"]),
                        "policy_step": step,
                        "command_qdot_norm": float(np.linalg.norm(command_qdot)),
                        "command_acc_norm": float(np.linalg.norm(command_acc)),
                        "command_jerk_norm": float(np.linalg.norm(command_jerk)),
                        "measured_qdot_norm": float(np.linalg.norm(measured_qdot)),
                        "measured_acc_norm": float(np.linalg.norm(measured_acc)),
                        "measured_jerk_norm": float(np.linalg.norm(measured_jerk)),
                        "distance_m": float(substep_distance),
                        "risk": float(substep_risk),
                        "contact": int(substep_contact),
                    }
                    for joint_index in range(len(command_qdot)):
                        suffix = joint_index + 1
                        physics_row[f"command_qdot_{suffix}"] = float(command_qdot[joint_index])
                        physics_row[f"command_acc_{suffix}"] = float(command_acc[joint_index])
                        physics_row[f"command_jerk_{suffix}"] = float(command_jerk[joint_index])
                        physics_row[f"measured_qdot_{suffix}"] = float(measured_qdot[joint_index])
                        physics_row[f"measured_acc_{suffix}"] = float(measured_acc[joint_index])
                        physics_row[f"measured_jerk_{suffix}"] = float(measured_jerk[joint_index])
                    physics_trace_rows.append(physics_row)
            prev_qdot_cmd = info["qdot_cmd"].copy()
            if terminated or truncated:
                break

        acc = np.asarray(accelerations, dtype=np.float32)
        jerk = np.asarray(jerks, dtype=np.float32)
        command_acc = np.concatenate(command_acceleration_substeps, axis=0)
        command_jerk = np.concatenate(command_jerk_substeps, axis=0)
        measured_acc = np.concatenate(measured_acceleration_substeps, axis=0)
        measured_jerk = np.concatenate(measured_jerk_substeps, axis=0)
        non_end_link_collision = bool(collision and 0 <= closest_link < env.capsule_model.count - 1)
        rows.append(
            {
                "episode": episode,
                "episode_seed": episode_seed,
                "reward": total_reward,
                "cost": total_cost,
                "length": step + 1,
                "success": int(success),
                "collision": int(collision),
                "non_end_link_collision": int(non_end_link_collision),
                "final_position_error": final_position_error,
                "completion_time": (step + 1) * float(config["env"]["control_dt"]),
                "min_distance": min(distances) if distances else 0.0,
                "mean_risk": float(np.mean(risks)) if risks else 0.0,
                "max_risk": float(np.max(risks)) if risks else 0.0,
                "safety_violation_count": violations,
                "safety_violation_rate": violations / max(step + 1, 1),
                "mean_action_variation": float(np.mean(action_variations)) if action_variations else 0.0,
                "policy_rate_limit_rate": policy_rate_limit_events / max(step + 1, 1),
                "safety_qp_intervention_rate": safety_qp_interventions / max(step + 1, 1),
                "safety_qp_infeasible_rate": safety_qp_infeasible / max(step + 1, 1),
                "mean_safety_qp_correction_norm": float(np.mean(safety_qp_correction_norms))
                if safety_qp_correction_norms
                else 0.0,
                "max_safety_qp_correction_norm": float(np.max(safety_qp_correction_norms))
                if safety_qp_correction_norms
                else 0.0,
                "mean_safety_qp_solve_time_ms": float(np.mean(safety_qp_solve_times))
                if safety_qp_solve_times
                else 0.0,
                "max_safety_qp_solve_time_ms": float(np.max(safety_qp_solve_times))
                if safety_qp_solve_times
                else 0.0,
                "rms_acceleration": float(np.sqrt(np.mean(np.square(acc)))) if len(acc) else 0.0,
                "rms_jerk": float(np.sqrt(np.mean(np.square(jerk)))) if len(jerk) else 0.0,
                "peak_acceleration": float(np.max(np.abs(acc))) if len(acc) else 0.0,
                "peak_jerk": float(np.max(np.abs(jerk))) if len(jerk) else 0.0,
                "command_rms_acceleration": float(np.sqrt(np.mean(np.square(command_acc)))),
                "command_rms_jerk": float(np.sqrt(np.mean(np.square(command_jerk)))),
                "command_peak_acceleration": float(np.max(np.abs(command_acc))),
                "command_peak_jerk": float(np.max(np.abs(command_jerk))),
                "measured_rms_acceleration": float(np.sqrt(np.mean(np.square(measured_acc)))),
                "measured_rms_jerk": float(np.sqrt(np.mean(np.square(measured_jerk)))),
                "measured_peak_acceleration": float(np.max(np.abs(measured_acc))),
                "measured_peak_jerk": float(np.max(np.abs(measured_jerk))),
            }
        )
        if trace_dir is not None and trace_rows:
            trace_path = trace_dir / f"episode_{episode:04d}.csv"
            with open(trace_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(trace_rows[0].keys()))
                writer.writeheader()
                writer.writerows(trace_rows)
            physics_trace_path = trace_dir / f"episode_{episode:04d}_physics.csv"
            with open(physics_trace_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(physics_trace_rows[0].keys()))
                writer.writeheader()
                writer.writerows(physics_trace_rows)

    env.close()
    output = Path(output_arg) if output_arg else Path(checkpoint).resolve().parent / "eval_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "collision_rate": float(np.mean([r["collision"] for r in rows])),
        "non_end_link_collision_rate": float(np.mean([r["non_end_link_collision"] for r in rows])),
        "mean_final_position_error": float(np.mean([r["final_position_error"] for r in rows])),
        "mean_min_distance": float(np.mean([r["min_distance"] for r in rows])),
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "mean_cost": float(np.mean([r["cost"] for r in rows])),
        "mean_safety_violation_count": float(np.mean([r["safety_violation_count"] for r in rows])),
        "mean_action_variation": float(np.mean([r["mean_action_variation"] for r in rows])),
        "mean_policy_rate_limit_rate": float(np.mean([r["policy_rate_limit_rate"] for r in rows])),
        "mean_safety_qp_intervention_rate": float(np.mean([r["safety_qp_intervention_rate"] for r in rows])),
        "mean_safety_qp_infeasible_rate": float(np.mean([r["safety_qp_infeasible_rate"] for r in rows])),
        "mean_safety_qp_correction_norm": float(np.mean([r["mean_safety_qp_correction_norm"] for r in rows])),
        "mean_safety_qp_solve_time_ms": float(np.mean([r["mean_safety_qp_solve_time_ms"] for r in rows])),
        "mean_rms_acceleration": float(np.mean([r["rms_acceleration"] for r in rows])),
        "mean_rms_jerk": float(np.mean([r["rms_jerk"] for r in rows])),
        "mean_peak_acceleration": float(np.mean([r["peak_acceleration"] for r in rows])),
        "mean_peak_jerk": float(np.mean([r["peak_jerk"] for r in rows])),
        "mean_command_rms_acceleration": float(np.mean([r["command_rms_acceleration"] for r in rows])),
        "mean_command_rms_jerk": float(np.mean([r["command_rms_jerk"] for r in rows])),
        "mean_command_peak_acceleration": float(np.mean([r["command_peak_acceleration"] for r in rows])),
        "mean_command_peak_jerk": float(np.mean([r["command_peak_jerk"] for r in rows])),
        "mean_measured_rms_acceleration": float(np.mean([r["measured_rms_acceleration"] for r in rows])),
        "mean_measured_rms_jerk": float(np.mean([r["measured_rms_jerk"] for r in rows])),
        "mean_measured_peak_acceleration": float(np.mean([r["measured_peak_acceleration"] for r in rows])),
        "mean_measured_peak_jerk": float(np.mean([r["measured_peak_jerk"] for r in rows])),
    }
    print(summary)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
