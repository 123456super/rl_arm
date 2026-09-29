#!/usr/bin/env python3
"""Short, isolated fixed-xi training probe used before formal curriculum runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.algorithms.homotopy_replay import HomotopyReplayBuffer
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--xi", type=float, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=11001)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--strictify", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.02 <= args.xi <= 1.0:
        parser.error("xi must be in [0.02, 1]")
    if args.strictify and not args.strict:
        parser.error("--strictify requires --strict")

    config = load_config(ROOT / args.config)
    config["seed"] = args.seed
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    source = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    if source.get("protocol") != config["thesis"]["protocol"]:
        raise ValueError(
            f"checkpoint protocol {source.get('protocol')!r} does not match "
            f"configured protocol {config['thesis']['protocol']!r}"
        )
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    env = ThesisHomotopyEnv(config)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    agent = ThesisSACAgent(obs_dim, action_dim, config)
    agent.load_state_dict(source["agent"])
    replay = HomotopyReplayBuffer(
        obs_dim, action_dim, str(agent.device), seed=args.seed,
        reward_gamma=float(config["sac"]["gamma"]),
        reward_horizon=int(config["thesis"]["horizon"]),
        reward_parameters=env.reward_parameters,
    )
    replay.load_state_dict(source["replay"])
    strictification = replay.strictify("static") if args.strictify else None
    rng = np.random.default_rng(args.seed + 77123)
    episode_rows: list[dict] = []
    update_rows: list[dict] = []
    transition_rows: list[dict] = []
    step = updates = episode = 0
    try:
        while step < args.steps:
            scene = str(rng.choice(["none", "static"], p=[.25, .75]))
            env.configure_episode(scene, xi=1.0 if scene == "none" else args.xi,
                                  strict=args.strict if scene == "static" else True)
            observation, _ = env.reset()
            terminated = truncated = False
            total_return = 0.0
            length = 0
            final: dict = {}
            while not (terminated or truncated) and step < args.steps:
                action = agent.select_action(observation)
                next_observation, reward, _, terminated, truncated, info = env.step(action)
                replay.add(scene, observation, action, next_observation, terminated, info, episode, length)
                observation = next_observation
                total_return += reward
                length += 1
                step += 1
                final = info
                if len(replay) >= 256:
                    batch = replay.sample("s1", {"static": args.xi, "dynamic": .02}, 256)
                    metrics = agent.update(batch)
                    updates += 1
                    if updates == 1 or updates % 100 == 0:
                        update_rows.append({"step": step, "updates": updates, **metrics})
                if step == 1 or step % 100 == 0:
                    transition_rows.append({
                        "step": step, "scene": scene, "xi": args.xi,
                        "strict": int(args.strict), "reward": reward,
                        "r_goal": info["r_goal"], "safety_penalty": info["safety_penalty"],
                        "risk_max": info["control_max_risk"], "d_min": info["control_min_distance"],
                        "alpha": float(agent.alpha.detach().cpu()),
                    })
            if terminated or truncated:
                episode_rows.append({
                    "episode": episode, "step": step, "scene": scene, "length": length,
                    "return": total_return, "task_reached": int(final.get("task_reached", False)),
                    "collision_assisted_reach": int(final.get("collision_assisted_reach", False)),
                    "safe_success": int(final.get("safe_success", False)),
                    "obstacle_collision": int(final.get("obstacle_collision", False)),
                    "self_collision": int(final.get("self_collision", False)),
                    "joint_limit": int(final.get("joint_limit", False)),
                    "timeout": int(truncated),
                })
                episode += 1
    finally:
        env.close()

    def write_csv(name: str, rows: list[dict]) -> None:
        if not rows: return
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    write_csv("episodes.csv", episode_rows)
    write_csv("updates.csv", update_rows)
    write_csv("transitions_sampled.csv", transition_rows)
    static = [row for row in episode_rows if row["scene"] == "static"]
    rates = {}
    for key in ("task_reached", "collision_assisted_reach", "safe_success", "obstacle_collision",
                "self_collision", "joint_limit", "timeout"):
        rates[key + "_rate"] = None if not static else float(np.mean([row[key] for row in static]))
    summary = {
        "source_checkpoint": args.checkpoint, "steps": step, "updates": updates,
        "episodes": len(episode_rows), "static_episodes": len(static), "xi": args.xi,
        "strict": args.strict, "strictification": strictification,
        "final_alpha": float(agent.alpha.detach().cpu()), **rates,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"agent": agent.state_dict(), "replay": replay.state_dict(), "probe": summary}, output / "checkpoint.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
