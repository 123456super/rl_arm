#!/usr/bin/env python3
"""Calibrate external-risk parameters from real policy reaching trajectories.

The script runs a frozen 241-D Full actor in S1/S2/S3 scenes, stores the raw
obstacle feature sets seen at each transition, and evaluates a small robust
grid of (lambda_ext, d_safe, d_cut, tau, phi_max).  The selected values are
written as a standalone frozen YAML/JSON artifact; the active experiment
configuration is intentionally left untouched until the result is reviewed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.convex_sweep import swept_risk
from rl_risk_sac.tasks.obstacle_observation import obstacle_set
from rl_risk_sac.utils.config import load_config
from scripts.core.train_thesis_homotopy import apply_stage_profile


def _percentile(values: list[float] | np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    return 0.0 if values.size == 0 else float(np.percentile(values, q))


def collect_trajectories(
    config: dict,
    checkpoint: Path,
    stages: tuple[str, ...],
    episodes_per_stage: int,
    max_steps: int | None,
    seed: int,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor_state = state.get("agent", {}).get("actor") if isinstance(state, dict) else None
    if actor_state is None and isinstance(state, dict):
        actor_state = state.get("actor", state)
    if not isinstance(actor_state, dict):
        raise ValueError("checkpoint does not contain an actor state")

    base = copy.deepcopy(config)
    base["device"] = "cpu"
    level_index = 13
    level = base["thesis"]["joint_pose_curriculum"]["levels"][level_index]
    orientation_levels = base["thesis"]["orientation_curriculum"]["levels"]
    rows: list[dict] = []
    feature_rows: list[np.ndarray] = []
    feature_masks: list[np.ndarray] = []
    for stage_index, stage in enumerate(stages):
        cfg = copy.deepcopy(base)
        apply_stage_profile(cfg, stage)
        cfg["seed"] = int(seed) + stage_index
        env = ThesisHomotopyEnv(cfg)
        agent = ThesisSACAgent(
            int(env.observation_space.shape[0]), int(env.action_space.shape[0]), cfg,
        )
        agent.actor.load_state_dict(actor_state)
        agent.actor.eval()
        try:
            horizon = int(cfg["thesis"]["horizon"] if max_steps is None else max_steps)
            for episode in range(episodes_per_stage):
                env.configure_episode(
                    "none", xi=1.0, strict=True,
                    goal_scale=float(level["goal_scale"]),
                    orientation_scale=float(orientation_levels[level_index]),
                    position_tolerance=float(level["position_tolerance_m"]),
                    orientation_tolerance=float(level["orientation_tolerance_rad"]),
                    target_distance_min_m=float(level["target_distance_min_m"]),
                    target_distance_max_m=float(level["target_distance_max_m"]),
                    target_orientation_min_rad=float(level["target_orientation_min_rad"]),
                    target_orientation_max_rad=float(level["target_orientation_max_rad"]),
                )
                observation, _ = env.reset(seed=int(seed) + stage_index * 10000 + episode)
                for step in range(horizon):
                    action = agent.select_action(observation, deterministic=True)
                    observation, reward, cost, terminated, truncated, info = env.step(action)
                    features = np.asarray(obstacle_set(observation), dtype=np.float32)
                    padded = np.zeros((3, 37), dtype=np.float32)
                    mask = np.zeros(3, dtype=bool)
                    count = min(len(features), 3)
                    if count:
                        padded[:count] = features[:count]
                        mask[:count] = True
                    feature_rows.append(padded)
                    feature_masks.append(mask)
                    rows.append({
                        "stage": stage,
                        "episode": episode,
                        "step": step,
                        "obstacle_count": int(info.get("obstacle_count", 0)),
                        "distance_min_m": float(info.get("control_min_distance", 0.8)),
                        "current_risk": float(info.get("control_max_risk", 0.0)),
                        "r_goal": float(info.get("r_goal", 0.0)),
                        "reward": float(reward),
                        "cost": float(cost),
                        "external_safety_penalty": float(info.get("external_safety_penalty", 0.0)),
                        "task_reached": int(bool(info.get("task_reached", False))),
                        "collision": int(bool(info.get("collision", False))),
                        "obstacle_collision": int(bool(info.get("obstacle_collision", False))),
                        "timeout": int(bool(truncated)),
                    })
                    if terminated or truncated:
                        break
        finally:
            env.close()
    return rows, np.asarray(feature_rows), np.asarray(feature_masks)


def evaluate_candidate(
    rows: list[dict], features: np.ndarray, masks: np.ndarray,
    *, lambda_ext: float, d_safe: float, d_cut: float, tau: float, phi_max: float,
    reward_scale: float,
) -> dict[str, float]:
    penalties = np.zeros(len(rows), dtype=np.float64)
    for index, (values, mask) in enumerate(zip(features, masks)):
        active = values[mask]
        risk = swept_risk(active, d_safe=d_safe, d_cut=d_cut, tau=tau, phi_max=phi_max)
        penalties[index] = lambda_ext * float(risk)
    counts = np.asarray([row["obstacle_count"] for row in rows])
    distances = np.asarray([row["distance_min_m"] for row in rows], dtype=np.float64)
    empty = counts == 0
    separated = (counts > 0) & (distances >= d_cut)
    approaching = (counts > 0) & (distances > 0.0) & (distances < d_cut)
    contact = (counts > 0) & (distances <= 0.0)
    safe_values = penalties[empty | separated]
    approach_values = penalties[approaching]
    contact_values = penalties[contact]
    approach_median = float(np.median(approach_values)) if approach_values.size else 0.0
    contact_median = float(np.median(contact_values)) if contact_values.size else 0.0
    safe_p95 = _percentile(safe_values, 95)
    # Targets are tied to the observed reaching reward scale rather than an
    # arbitrary absolute reward number.  Near-miss risk is noticeable but
    # should not dominate progress; contact risk is substantially stronger.
    target_approach = 0.05 * reward_scale
    target_contact = 0.15 * reward_scale
    score = (
        ((safe_p95 / max(0.01 * reward_scale, 1.0e-9)) ** 2)
        + (((approach_median - target_approach) / max(target_approach, 1.0e-9)) ** 2)
        + (((contact_median - target_contact) / max(target_contact, 1.0e-9)) ** 2)
        + (0.0 if contact_median >= approach_median else 10.0)
    )
    return {
        "lambda_ext": lambda_ext, "d_safe": d_safe, "d_cut": d_cut,
        "tau": tau, "phi_max": phi_max, "score": float(score),
        "safe_p95_penalty": safe_p95,
        "approaching_median_penalty": approach_median,
        "contact_median_penalty": contact_median,
        "approaching_count": int(approach_values.size),
        "contact_count": int(contact_values.size),
        "reward_scale": reward_scale,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/paper_full.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes-per-stage", type=int, default=8)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=52001)
    args = parser.parse_args()
    if args.episodes_per_stage < 1:
        parser.error("--episodes-per-stage must be positive")
    config = load_config(ROOT / args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    stages = ("s1", "s2", "s3")
    rows, features, masks = collect_trajectories(
        config, checkpoint, stages, args.episodes_per_stage, args.max_steps, args.seed,
    )
    positive_rewards = [abs(row["r_goal"]) for row in rows if row["r_goal"] > 0.0]
    reward_scale = float(np.median(positive_rewards)) if positive_rewards else 1.0
    grid = {
        "lambda_ext": (0.02, 0.05, 0.08, 0.12),
        "d_safe": (0.04, 0.05, 0.06),
        "d_cut": (0.15, 0.20, 0.25),
        "tau": (0.01, 0.02, 0.04),
        "phi_max": (3.0, 5.0, 8.0),
    }
    candidates = []
    for lambda_ext in grid["lambda_ext"]:
        for d_safe in grid["d_safe"]:
            for d_cut in grid["d_cut"]:
                if d_cut <= d_safe:
                    continue
                for tau in grid["tau"]:
                    for phi_max in grid["phi_max"]:
                        candidates.append(evaluate_candidate(
                            rows, features, masks, lambda_ext=lambda_ext,
                            d_safe=d_safe, d_cut=d_cut, tau=tau,
                            phi_max=phi_max, reward_scale=reward_scale,
                        ))
    candidates.sort(key=lambda item: item["score"])
    selected = candidates[0]
    with (output / "trajectory_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "trajectory_features.npz", features=features, masks=masks)
    (output / "candidates.json").write_text(json.dumps(candidates[:50], indent=2), encoding="utf-8")
    summary = {
        "checkpoint": str(checkpoint),
        "stages": list(stages),
        "episodes_per_stage": args.episodes_per_stage,
        "transitions": len(rows),
        "success_rate_by_stage": {
            stage: float(np.mean([r["task_reached"] for r in rows if r["stage"] == stage]))
            for stage in stages
        },
        "collision_rate_by_stage": {
            stage: float(np.mean([r["collision"] for r in rows if r["stage"] == stage]))
            for stage in stages
        },
        "reward_scale": reward_scale,
        "selected": selected,
        "top_candidates": candidates[:10],
        "calibration_method": "real frozen-policy reaching trajectories; robust grid matching 1%/5%/15% reward-scale targets",
    }
    (output / "calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    frozen = {
        "thesis": {
            "obstacles": {
                "lambda_ext": float(selected["lambda_ext"]),
                "risk": {
                    "d_safe": float(selected["d_safe"]),
                    "d_cut": float(selected["d_cut"]),
                    "tau": float(selected["tau"]),
                    "phi_max": float(selected["phi_max"]),
                },
            }
        },
        "metadata": {
            "calibrated": True,
            "source": "calibration_summary.json",
            "checkpoint": str(checkpoint),
            "stages": list(stages),
            "transitions": len(rows),
            "reward_scale": reward_scale,
        },
    }
    (output / "frozen_risk.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
