#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict validation for a thesis homotopy checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/experiments/thesis_homotopy.yaml")
    parser.add_argument("--scenes", nargs="+", choices=("none", "static", "dynamic"), default=["none", "static", "dynamic"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=41001)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(ROOT / args.config); output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_stage = checkpoint["curriculum"]["stage"]
    if checkpoint_stage in {"s1", "s2"}:
        new_scene = "static" if checkpoint_stage == "s1" else "dynamic"
        state = checkpoint["curriculum"]["states"][new_scene]
        if not (state.strict and state.replay_strictified and state.strict_steps >= int(config["thesis"]["strict_min_transitions"])):
            raise ValueError(f"checkpoint is not Gate-eligible: {new_scene} strict_steps={state.strict_steps}")
    agent = ThesisSACAgent(55, 6, config); agent.load_state_dict(checkpoint["agent"])
    env = ThesisHomotopyEnv(config); rows = []
    for scene_index, scene in enumerate(args.scenes):
        for episode in range(args.episodes):
            env.configure_episode(scene, xi=1.0, strict=True)
            observation, _ = env.reset(seed=args.seed + scene_index * 100000 + episode)
            terminated = truncated = False; total = 0.0; length = 0; info = {}
            while not (terminated or truncated):
                observation, reward, _, terminated, truncated, info = env.step(agent.select_action(observation, deterministic=True))
                total += reward; length += 1
            rows.append({"scene": scene, "episode": episode, "return": total, "length": length,
                         "safe_success": int(info["safe_success"]), "obstacle_collision": int(info["obstacle_collision"]),
                         "self_collision": int(info["self_collision"]), "environment_collision": int(info["environment_collision"]),
                         "joint_limit": int(info["joint_limit"]), "timeout": int(truncated)})
    env.close()
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for scene in args.scenes:
        selected = [row for row in rows if row["scene"] == scene]; count = len(selected)
        summary[scene] = {key + "_rate": float(np.mean([row[key] for row in selected]))
                          for key in ("safe_success", "obstacle_collision", "self_collision", "environment_collision", "joint_limit", "timeout")}
        summary[scene]["episodes"] = count
    thresholds = {
        "none": {"safe_success_rate": .95, "obstacle_collision_rate": 0.0, "joint_limit_rate": 0.0, "timeout_rate": .05},
        "static": {"safe_success_rate": .90, "obstacle_collision_rate": .03, "joint_limit_rate": 0.0, "timeout_rate": .10},
        "dynamic": {"safe_success_rate": .80, "obstacle_collision_rate": .05, "joint_limit_rate": 0.0, "timeout_rate": .20},
    }
    for scene in args.scenes:
        actual, limit = summary[scene], thresholds[scene]
        actual["gate_pass"] = bool(actual["safe_success_rate"] >= limit["safe_success_rate"]
                                   and actual["obstacle_collision_rate"] <= limit["obstacle_collision_rate"]
                                   and actual["joint_limit_rate"] <= limit["joint_limit_rate"]
                                   and actual["timeout_rate"] <= limit["timeout_rate"])
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
