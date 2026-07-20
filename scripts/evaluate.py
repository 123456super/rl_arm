from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from rl_risk_sac.algorithms import SACAgent
from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.seeding import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default="ldrc_adaptive", choices=["ee_fixed", "link_fixed", "ldrc_fixed", "ldrc_adaptive"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["seed"] = args.seed
    set_seed(args.seed)

    env = UR5DynamicObstacleEnv(config, method=args.method)
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], config, method=args.method)
    agent.load_actor(args.checkpoint)

    rows = []
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        total_cost = 0.0
        risks = []
        distances = []
        violations = 0
        accelerations = []
        jerks = []
        success = False
        collision = False
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
            success = bool(info["success"])
            collision = bool(info["collision"])
            if terminated or truncated:
                break

        acc = np.asarray(accelerations, dtype=np.float32)
        jerk = np.asarray(jerks, dtype=np.float32)
        rows.append(
            {
                "episode": episode,
                "reward": total_reward,
                "cost": total_cost,
                "length": step + 1,
                "success": int(success),
                "collision": int(collision),
                "min_distance": min(distances) if distances else 0.0,
                "mean_risk": float(np.mean(risks)) if risks else 0.0,
                "max_risk": float(np.max(risks)) if risks else 0.0,
                "safety_violation_rate": violations / max(step + 1, 1),
                "rms_acceleration": float(np.sqrt(np.mean(np.square(acc)))) if len(acc) else 0.0,
                "rms_jerk": float(np.sqrt(np.mean(np.square(jerk)))) if len(jerk) else 0.0,
            }
        )

    env.close()
    output = Path(args.output) if args.output else Path(args.checkpoint).resolve().parent / "eval_metrics.csv"
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "collision_rate": float(np.mean([r["collision"] for r in rows])),
        "mean_min_distance": float(np.mean([r["min_distance"] for r in rows])),
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "mean_cost": float(np.mean([r["cost"] for r in rows])),
        "mean_rms_acceleration": float(np.mean([r["rms_acceleration"] for r in rows])),
        "mean_rms_jerk": float(np.mean([r["rms_jerk"] for r in rows])),
    }
    print(summary)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
