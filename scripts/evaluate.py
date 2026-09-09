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
        physics_rms_accelerations = []
        physics_rms_jerks = []
        physics_peak_accelerations = []
        physics_peak_jerks = []
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
            risks.append(float(info["risk_global"]))
            distances.append(float(info["d_min"]))
            violations += int(info["safety_violation"])
            accelerations.append(info["joint_acc"])
            jerks.append(info["joint_jerk"])
            physics_rms_accelerations.append(float(info["physics_rms_acceleration"]))
            physics_rms_jerks.append(float(info["physics_rms_jerk"]))
            physics_peak_accelerations.append(float(info["physics_peak_acceleration"]))
            physics_peak_jerks.append(float(info["physics_peak_jerk"]))
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
                        "beta": float(info["beta"]),
                        "closest_link": closest_link,
                        "safety_violation": int(info["safety_violation"]),
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
                        "physics_rms_acceleration": float(info["physics_rms_acceleration"]),
                        "physics_rms_jerk": float(info["physics_rms_jerk"]),
                        "physics_peak_acceleration": float(info["physics_peak_acceleration"]),
                        "physics_peak_jerk": float(info["physics_peak_jerk"]),
                    }
                )
                for substep, (substep_qdot, substep_acc, substep_jerk) in enumerate(
                    zip(
                        info["qdot_substeps"],
                        info["joint_acc_substeps"],
                        info["joint_jerk_substeps"],
                    )
                ):
                    physics_row = {
                        "physics_step": step * env.sim_substeps + substep,
                        "time": step * float(config["env"]["control_dt"])
                        + (substep + 1) * float(config["env"]["time_step"]),
                        "policy_step": step,
                        "qdot_norm": float(np.linalg.norm(substep_qdot)),
                        "acc_norm": float(np.linalg.norm(substep_acc)),
                        "jerk_norm": float(np.linalg.norm(substep_jerk)),
                    }
                    for joint_index in range(len(substep_qdot)):
                        physics_row[f"qdot_{joint_index + 1}"] = float(substep_qdot[joint_index])
                        physics_row[f"acc_{joint_index + 1}"] = float(substep_acc[joint_index])
                        physics_row[f"jerk_{joint_index + 1}"] = float(substep_jerk[joint_index])
                    physics_trace_rows.append(physics_row)
            prev_qdot_cmd = info["qdot_cmd"].copy()
            if terminated or truncated:
                break

        acc = np.asarray(accelerations, dtype=np.float32)
        jerk = np.asarray(jerks, dtype=np.float32)
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
                "physics_rms_acceleration": float(np.sqrt(np.mean(np.square(physics_rms_accelerations))))
                if physics_rms_accelerations
                else 0.0,
                "physics_rms_jerk": float(np.sqrt(np.mean(np.square(physics_rms_jerks))))
                if physics_rms_jerks
                else 0.0,
                "physics_peak_acceleration": float(np.max(physics_peak_accelerations))
                if physics_peak_accelerations
                else 0.0,
                "physics_peak_jerk": float(np.max(physics_peak_jerks)) if physics_peak_jerks else 0.0,
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
        "mean_physics_rms_acceleration": float(np.mean([r["physics_rms_acceleration"] for r in rows])),
        "mean_physics_rms_jerk": float(np.mean([r["physics_rms_jerk"] for r in rows])),
        "mean_physics_peak_acceleration": float(np.mean([r["physics_peak_acceleration"] for r in rows])),
        "mean_physics_peak_jerk": float(np.mean([r["physics_peak_jerk"] for r in rows])),
    }
    print(summary)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
