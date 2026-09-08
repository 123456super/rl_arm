from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rl_risk_sac.algorithms import SACAgent
from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.scene import ObstacleState
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.device import resolve_device
from rl_risk_sac.utils.predictive_risk import (
    PredictiveLinkRisk,
    PredictiveRiskConfig,
    compute_predictive_link_risk,
)
from rl_risk_sac.utils.seeding import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/random_crossing_link_fixed_penalty1.yaml")
    parser.add_argument(
        "--method",
        default="link_fixed",
        choices=["ee_fixed", "link_fixed", "predictive_link", "ldrc_fixed", "ldrc_adaptive"],
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1004)
    parser.add_argument("--horizon", type=float, default=1.0)
    parser.add_argument("--prediction-step", type=float, default=0.05)
    parser.add_argument("--output-dir", default="outputs/predictive_risk/offline_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["seed"] = int(args.seed)
    config["device"] = resolve_device(config)
    set_seed(int(args.seed))

    env = UR5DynamicObstacleEnv(config, method=args.method)
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], config, method=args.method)
    agent.load_actor(args.checkpoint)

    pred_config = PredictiveRiskConfig(
        horizon=float(args.horizon),
        step=float(args.prediction_step),
        d_safe=env.risk_config.d_safe,
        sigma_d=env.risk_config.sigma_d,
        tau_enter=env.risk_config.tau,
        v_max=env.risk_config.v_max,
        eps=env.risk_config.eps,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = []
    all_step_rows = []
    horizon_steps = max(1, int(np.ceil(pred_config.horizon / env.control_dt)))

    for episode in range(int(args.episodes)):
        observation, info = env.reset(seed=int(args.seed) + episode)
        step_records: list[dict[str, Any]] = []
        terminated = False
        truncated = False

        for step in range(env.max_episode_steps):
            predictive = _compute_scene_predictive_risk(env, pred_config)
            step_records.append(
                {
                    "episode": episode,
                    "step": step,
                    "time": step * env.control_dt,
                    "d_now": float(info["d_min"]),
                    "risk_now": float(info["risk_global"]),
                    "closest_link_now": int(info["closest_link"]),
                    "safety_violation_now": int(info["safety_violation"]),
                    "d_pred": float(np.min(predictive.d_pred)) if len(predictive.d_pred) else np.inf,
                    "t_enter": float(np.min(predictive.t_enter)) if np.any(np.isfinite(predictive.t_enter)) else np.inf,
                    "risk_pred_body": float(predictive.risk_pred_body),
                    "critical_link_pred": int(predictive.critical_link),
                }
            )

            action = agent.select_action(observation, deterministic=True)
            observation, _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        _annotate_future_metrics(step_records, horizon_steps, env.risk_config.d_safe)
        episode_rows.append(_episode_summary(episode, step_records, terminated, truncated))
        all_step_rows.extend(step_records)

    env.close()

    step_path = output_dir / "step_metrics.csv"
    episode_path = output_dir / "episode_metrics.csv"
    summary_path = output_dir / "summary.json"
    _write_csv(step_path, all_step_rows)
    _write_csv(episode_path, episode_rows)
    summary = _overall_summary(args, pred_config, episode_rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved: {output_dir}")


def _compute_scene_predictive_risk(env: UR5DynamicObstacleEnv, config: PredictiveRiskConfig) -> PredictiveLinkRisk:
    capsules = env._capsules()
    active_obstacles = [obstacle for obstacle in env.obstacle_states if obstacle.enabled]
    if not active_obstacles:
        count = len(capsules)
        return PredictiveLinkRisk(
            d_pred=np.full(count, float(env.observation_cfg["no_obstacle_distance"]), dtype=np.float32),
            t_enter=np.full(count, np.inf, dtype=np.float32),
            risk_pred_per_link=np.zeros(count, dtype=np.float32),
            risk_pred_body=0.0,
            critical_link=0 if count else -1,
            closest_points_pred=np.zeros((count, 3), dtype=np.float32),
            closest_times=np.zeros(count, dtype=np.float32),
            approach_velocities=np.zeros(count, dtype=np.float32),
        )

    risks = [
        compute_predictive_link_risk(
            capsules=capsules,
            prev_capsules=env.prev_capsules,
            obstacle_center=obstacle.center,
            obstacle_velocity=obstacle.velocity,
            obstacle_radius=float(env.obstacle_cfg["radius"]),
            dt=env.control_dt,
            config=config,
            use_end_effector_only=env.method == "ee_fixed",
        )
        for obstacle in active_obstacles
    ]
    return _aggregate_predictive_risks(risks)


def _aggregate_predictive_risks(risks: list[PredictiveLinkRisk]) -> PredictiveLinkRisk:
    if len(risks) == 1:
        return risks[0]

    d_stack = np.stack([risk.d_pred for risk in risks])
    nearest_obstacle_indices = np.argmin(d_stack, axis=0)
    link_indices = np.arange(d_stack.shape[1])
    risk_stack = np.stack([risk.risk_pred_per_link for risk in risks])
    per_link_risk = np.max(risk_stack, axis=0)
    return PredictiveLinkRisk(
        d_pred=d_stack[nearest_obstacle_indices, link_indices].astype(np.float32),
        t_enter=np.min(np.stack([risk.t_enter for risk in risks]), axis=0).astype(np.float32),
        risk_pred_per_link=per_link_risk.astype(np.float32),
        risk_pred_body=float(np.max(per_link_risk)),
        critical_link=int(np.argmax(per_link_risk)),
        closest_points_pred=np.stack([risk.closest_points_pred for risk in risks])[
            nearest_obstacle_indices, link_indices
        ].astype(np.float32),
        closest_times=np.stack([risk.closest_times for risk in risks])[nearest_obstacle_indices, link_indices].astype(
            np.float32
        ),
        approach_velocities=np.max(np.stack([risk.approach_velocities for risk in risks]), axis=0).astype(np.float32),
    )


def _annotate_future_metrics(rows: list[dict[str, Any]], horizon_steps: int, d_safe: float) -> None:
    for index, row in enumerate(rows):
        future = rows[index : index + horizon_steps + 1]
        future_distances = [float(item["d_now"]) for item in future]
        future_min_index = int(np.argmin(future_distances))
        future_min_row = future[future_min_index]
        actual_future_min = float(future_min_row["d_now"])
        actual_future_violation = actual_future_min < d_safe
        predicted_warning = np.isfinite(float(row["t_enter"]))
        row["actual_future_min_distance"] = actual_future_min
        row["future_min_distance_error"] = float(row["d_pred"]) - actual_future_min
        row["actual_future_closest_link"] = int(future_min_row["closest_link_now"])
        row["actual_future_violation"] = int(actual_future_violation)
        row["warning"] = int(predicted_warning)
        row["false_positive"] = int(predicted_warning and not actual_future_violation)
        row["false_negative"] = int((not predicted_warning) and actual_future_violation)
        row["critical_link_match"] = int(int(row["critical_link_pred"]) == int(future_min_row["closest_link_now"]))


def _episode_summary(
    episode: int,
    rows: list[dict[str, Any]],
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    any_warning_times = [float(row["time"]) for row in rows if int(row["warning"])]
    warning_times = [float(row["time"]) for row in rows if int(row["warning"]) and int(row["actual_future_violation"])]
    violation_times = [float(row["time"]) for row in rows if int(row["safety_violation_now"])]
    first_any_warning = min(any_warning_times) if any_warning_times else np.inf
    first_warning = min(warning_times) if warning_times else np.inf
    first_violation = min(violation_times) if violation_times else np.inf
    lead_time = (
        first_violation - first_warning
        if np.isfinite(first_warning) and np.isfinite(first_violation) and first_warning <= first_violation
        else np.nan
    )
    return {
        "episode": episode,
        "length": len(rows),
        "terminated": int(terminated),
        "truncated": int(truncated),
        "first_any_warning_time": first_any_warning,
        "first_warning_time": first_warning,
        "first_violation_time": first_violation,
        "warning_lead_time": lead_time,
        "min_current_distance": min(float(row["d_now"]) for row in rows) if rows else np.nan,
        "min_predicted_distance": min(float(row["d_pred"]) for row in rows) if rows else np.nan,
        "mean_future_min_distance_error": float(np.mean([row["future_min_distance_error"] for row in rows]))
        if rows
        else np.nan,
        "critical_link_identification_rate": float(np.mean([row["critical_link_match"] for row in rows])) if rows else np.nan,
        "false_positive_rate": float(np.mean([row["false_positive"] for row in rows])) if rows else np.nan,
        "false_negative_rate": float(np.mean([row["false_negative"] for row in rows])) if rows else np.nan,
    }


def _overall_summary(
    args: argparse.Namespace,
    pred_config: PredictiveRiskConfig,
    episode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    finite_leads = [row["warning_lead_time"] for row in episode_rows if np.isfinite(row["warning_lead_time"])]
    return {
        "config": args.config,
        "method": args.method,
        "checkpoint": args.checkpoint,
        "episodes": len(episode_rows),
        "seed": int(args.seed),
        "predictive_risk_config": asdict(pred_config),
        "mean_warning_lead_time": float(np.mean(finite_leads)) if finite_leads else None,
        "episodes_with_warning_and_violation": len(finite_leads),
        "mean_future_min_distance_error": float(np.mean([row["mean_future_min_distance_error"] for row in episode_rows])),
        "mean_critical_link_identification_rate": float(
            np.mean([row["critical_link_identification_rate"] for row in episode_rows])
        ),
        "mean_false_positive_rate": float(np.mean([row["false_positive_rate"] for row in episode_rows])),
        "mean_false_negative_rate": float(np.mean([row["false_negative_rate"] for row in episode_rows])),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
