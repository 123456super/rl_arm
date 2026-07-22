from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import trange

from rl_risk_sac.algorithms import SACAgent
from rl_risk_sac.algorithms.replay_buffer import ReplayBuffer
from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.device import resolve_device
from rl_risk_sac.utils.seeding import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def build_run_dir(config: dict[str, Any], method: str, run_name: str | None) -> Path:
    output_root = Path(config["train"]["output_dir"])
    if run_name:
        return output_root / run_name

    robot_name = Path(str(config["robot"]["urdf"])).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_steps = int(config["train"]["total_steps"])
    seed = int(config["seed"])
    name = f"{timestamp}_{robot_name}_{method}_seed{seed}_steps{total_steps}"
    return output_root / "runs" / name


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    method = config["train"]["method"]

    config["device"] = resolve_device(config)
    set_seed(int(config["seed"]))
    env = UR5DynamicObstacleEnv(config, method=method)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, action_dim, config, method=method)
    replay = ReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        capacity=int(config["sac"]["replay_size"]),
        device=config.get("device", "cpu"),
    )

    run_dir = build_run_dir(config, method, config["train"].get("run_name"))
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = Path(config["train"]["output_dir"]) / "latest_run.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(run_dir) + "\n", encoding="utf-8")
    with open(run_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    metrics_path = run_dir / "train_metrics.csv"
    fieldnames = [
        "episode",
        "step",
        "episode_reward",
        "episode_cost",
        "episode_length",
        "success",
        "collision",
        "safety_violation_rate",
        "min_distance",
        "mean_risk",
        "lambda",
        "alpha",
    ]
    metrics_file = open(metrics_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
    writer.writeheader()

    progress_path = run_dir / "progress.csv"
    progress_fieldnames = [
        "step",
        "total_steps",
        "episode",
        "episode_length",
        "latest_reward",
        "latest_cost",
        "episode_reward",
        "episode_cost",
        "risk_global",
        "d_min",
        "success",
        "collision",
        "lambda",
        "alpha",
        "replay_size",
    ]
    progress_file = open(progress_path, "w", newline="", encoding="utf-8")
    progress_writer = csv.DictWriter(progress_file, fieldnames=progress_fieldnames)
    progress_writer.writeheader()

    total_steps = int(config["train"]["total_steps"])
    warmup_steps = int(config["sac"]["warmup_steps"])
    update_after = int(config["sac"]["update_after"])
    update_every = int(config["sac"]["update_every"])
    save_interval = int(config["train"]["save_interval"])
    log_interval = int(config["train"]["log_interval"])
    progress_interval = max(1, int(config["train"].get("progress_interval", 100)))

    observation, _ = env.reset(seed=int(config["seed"]))
    episode = 0
    episode_reward = 0.0
    episode_cost = 0.0
    episode_length = 0
    episode_costs: list[float] = []
    episode_risks: list[float] = []
    episode_distances: list[float] = []
    episode_violations = 0
    recent_rewards: deque[float] = deque(maxlen=20)
    update_info: dict[str, float] = {"alpha": float(agent.alpha.detach().cpu()), "lambda": agent.lagrange_multiplier}

    print(
        f"start training: method={method} total_steps={total_steps} device={config['device']} "
        f"run_dir={run_dir} progress_interval={progress_interval}",
        flush=True,
    )
    progress = trange(1, total_steps + 1, desc=f"train:{method}")
    for step in progress:
        if step <= warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(observation, deterministic=False)

        next_observation, reward, cost, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay.add(observation, action, reward, cost, next_observation, done)
        observation = next_observation

        episode_reward += reward
        episode_cost += cost
        episode_length += 1
        episode_costs.append(cost)
        episode_risks.append(float(info["risk_global"]))
        episode_distances.append(float(info["d_min"]))
        episode_violations += int(info["safety_violation"])

        if step >= update_after and len(replay) >= agent.batch_size and step % update_every == 0:
            batch = replay.sample(agent.batch_size)
            update_info = agent.update(batch)

        if step % progress_interval == 0 or step == total_steps:
            progress_row = {
                "step": step,
                "total_steps": total_steps,
                "episode": episode,
                "episode_length": episode_length,
                "latest_reward": reward,
                "latest_cost": cost,
                "episode_reward": episode_reward,
                "episode_cost": episode_cost,
                "risk_global": float(info["risk_global"]),
                "d_min": float(info["d_min"]),
                "success": int(info["success"]),
                "collision": int(info["collision"]),
                "lambda": agent.lagrange_multiplier,
                "alpha": float(agent.alpha.detach().cpu()),
                "replay_size": len(replay),
            }
            progress_writer.writerow(progress_row)
            progress_file.flush()
            print(
                "progress "
                f"step={step}/{total_steps} "
                f"ep={episode} "
                f"ep_len={episode_length} "
                f"ep_reward={episode_reward:.3f} "
                f"ep_cost={episode_cost:.3f} "
                f"risk={float(info['risk_global']):.3f} "
                f"d_min={float(info['d_min']):.3f} "
                f"lambda={agent.lagrange_multiplier:.3f} "
                f"alpha={float(agent.alpha.detach().cpu()):.3f} "
                f"replay={len(replay)}",
                flush=True,
            )

        if done:
            episode += 1
            mean_cost = float(np.mean(episode_costs)) if episode_costs else 0.0
            agent.update_lagrange(mean_cost)
            recent_rewards.append(episode_reward)
            writer.writerow(
                {
                    "episode": episode,
                    "step": step,
                    "episode_reward": episode_reward,
                    "episode_cost": episode_cost,
                    "episode_length": episode_length,
                    "success": int(info["success"]),
                    "collision": int(info["collision"]),
                    "safety_violation_rate": episode_violations / max(episode_length, 1),
                    "min_distance": min(episode_distances) if episode_distances else 0.0,
                    "mean_risk": float(np.mean(episode_risks)) if episode_risks else 0.0,
                    "lambda": agent.lagrange_multiplier,
                    "alpha": float(agent.alpha.detach().cpu()),
                }
            )
            metrics_file.flush()

            if episode % log_interval == 0:
                progress.set_postfix(
                    {
                        "ep": episode,
                        "r20": f"{np.mean(recent_rewards):.2f}" if recent_rewards else "0.00",
                        "cost": f"{mean_cost:.3f}",
                        "lambda": f"{agent.lagrange_multiplier:.3f}",
                        "alpha": f"{update_info.get('alpha', 0.0):.3f}",
                    }
                )

            observation, _ = env.reset()
            episode_reward = 0.0
            episode_cost = 0.0
            episode_length = 0
            episode_costs = []
            episode_risks = []
            episode_distances = []
            episode_violations = 0

        if step % save_interval == 0:
            agent.save(run_dir)

    agent.save(run_dir)
    progress_file.close()
    metrics_file.close()
    env.close()
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
