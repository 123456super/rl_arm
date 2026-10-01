#!/usr/bin/env python3
"""Diagnose where and how a frozen V13.5 actor times out."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config


def load_actor(agent: ThesisSACAgent, path: Path) -> None:
    state = torch.load(path, map_location=agent.device, weights_only=False)
    if isinstance(state, dict) and "agent" in state:
        state = state["agent"]["actor"]
    elif isinstance(state, dict) and "actor" in state:
        state = state["actor"]
    agent.actor.load_state_dict(state)
    agent.actor.eval()


def evaluate(
    agent: ThesisSACAgent, pool: ParallelThesisEnvPool, config: dict[str, Any],
    *, episodes: int, seed: int, level_index: int,
) -> list[dict[str, Any]]:
    level = config["thesis"]["joint_pose_curriculum"]["levels"][level_index]
    orientation_scale = config["thesis"]["orientation_curriculum"]["levels"][level_index]
    position_tolerance = float(level["position_tolerance_m"])
    orientation_tolerance = float(level["orientation_tolerance_rad"])
    position_edges = np.linspace(
        level["target_distance_min_m"], level["target_distance_max_m"], 11,
    )
    orientation_edges = np.linspace(
        level["target_orientation_min_rad"], level["target_orientation_max_rad"], 11,
    )
    base_contract = {
        "scene": "none", "xi": 1.0, "strict": True,
        "goal_scale": float(level["goal_scale"]),
        "orientation_scale": float(orientation_scale),
        "position_tolerance": position_tolerance,
        "orientation_tolerance": orientation_tolerance,
        "lambda_self": 1.0,
    }
    active: dict[int, dict[str, Any]] = {}
    outcomes: dict[int, dict[str, Any]] = {}
    next_episode = 0
    next_report = 100
    while len(outcomes) < episodes:
        assignments: dict[int, int] = {}
        contracts: dict[int, dict[str, Any]] = {}
        for worker in range(len(pool)):
            if worker in active or next_episode >= episodes:
                continue
            episode = next_episode
            next_episode += 1
            p_bin = (episode // 10) % 10
            o_bin = episode % 10
            assignments[worker] = episode
            contracts[worker] = {
                **base_contract,
                "target_distance_min_m": float(position_edges[p_bin]),
                "target_distance_max_m": float(position_edges[p_bin + 1]),
                "target_orientation_min_rad": float(orientation_edges[o_bin]),
                "target_orientation_max_rad": float(orientation_edges[o_bin + 1]),
            }
            active[worker] = {
                "episode": episode, "position_bin": p_bin, "orientation_bin": o_bin,
                "steps": 0, "position_within_steps": 0,
                "orientation_within_steps": 0, "joint_within_steps": 0,
                "joint_entries": 0, "joint_exits": 0, "joint_streak": 0,
                "max_joint_streak": 0, "previous_joint_within": False,
                "minimum_joint_ratio": float("inf"),
                "closest_position_ratio": float("inf"),
                "closest_orientation_ratio": float("inf"),
                "minimum_position_ratio": float("inf"),
                "minimum_orientation_ratio": float("inf"),
                "wrist_action_abs_sum": np.zeros(3, dtype=np.float64),
                "wrist_velocity_abs_sum": np.zeros(3, dtype=np.float64),
                "wrist_action_saturated_steps": 0,
                "wrist_action_max_abs": np.zeros(3, dtype=np.float64),
                "wrist_velocity_max_abs": np.zeros(3, dtype=np.float64),
                "orientation_error_trace": [],
                "orientation_error_vector_trace": [],
                "wrist_action_trace": [],
            }
        if assignments:
            resets = pool.reset_many(
                contracts,
                seeds={worker: seed + episode for worker, episode in assignments.items()},
            )
            for worker, (observation, info) in resets.items():
                active[worker]["observation"] = observation
                active[worker]["target_distance_m"] = float(
                    info["sampled_target_distance_m"]
                )
                active[worker]["target_orientation_rad"] = float(
                    info["sampled_target_orientation_rad"]
                )

        workers = sorted(active)
        observations = np.stack([active[worker]["observation"] for worker in workers])
        actions = agent.select_actions(observations, deterministic=True)
        action_by_worker = dict(zip(workers, actions))
        results = pool.step_many(action_by_worker)
        finished = []
        for worker in workers:
            observation, _, _, terminated, truncated, info = results[worker]
            state = active[worker]
            state["observation"] = observation
            state["steps"] += 1
            # UR5 joints 4-6 are the wrist. Comparing policy commands with
            # applied velocities separates weak orientation correction from
            # a kinematic or control bottleneck.
            wrist_action = np.abs(np.asarray(action_by_worker[worker], dtype=np.float64)[3:6])
            wrist_action_signed = np.asarray(action_by_worker[worker], dtype=np.float64)[3:6]
            wrist_velocity = np.abs(np.asarray(
                info.get("joint_velocity", info.get("commanded_joint_velocity", np.zeros(6))),
                dtype=np.float64,
            )[3:6])
            state["wrist_action_abs_sum"] += wrist_action
            state["wrist_velocity_abs_sum"] += wrist_velocity
            state["wrist_action_saturated_steps"] += int(np.any(wrist_action >= 0.95))
            state["wrist_action_max_abs"] = np.maximum(state["wrist_action_max_abs"], wrist_action)
            state["wrist_velocity_max_abs"] = np.maximum(state["wrist_velocity_max_abs"], wrist_velocity)
            state["orientation_error_vector_trace"].append(
                np.asarray(info.get("orientation_error_vector", np.zeros(3)), dtype=np.float64).tolist()
            )
            state["wrist_action_trace"].append(wrist_action_signed.tolist())
            p_ratio = float(info["next_rho_position"]) / position_tolerance
            o_ratio = float(info["next_rho_orientation"]) / orientation_tolerance
            state["orientation_error_trace"].append(float(info["next_rho_orientation"]))
            joint_ratio = max(p_ratio, o_ratio)
            p_within = p_ratio <= 1.0
            o_within = o_ratio <= 1.0
            joint_within = p_within and o_within
            state["position_within_steps"] += int(p_within)
            state["orientation_within_steps"] += int(o_within)
            state["joint_within_steps"] += int(joint_within)
            state["minimum_position_ratio"] = min(state["minimum_position_ratio"], p_ratio)
            state["minimum_orientation_ratio"] = min(state["minimum_orientation_ratio"], o_ratio)
            if joint_ratio < state["minimum_joint_ratio"]:
                state["minimum_joint_ratio"] = joint_ratio
                state["closest_position_ratio"] = p_ratio
                state["closest_orientation_ratio"] = o_ratio
            if joint_within and not state["previous_joint_within"]:
                state["joint_entries"] += 1
            if not joint_within and state["previous_joint_within"]:
                state["joint_exits"] += 1
            state["joint_streak"] = state["joint_streak"] + 1 if joint_within else 0
            state["max_joint_streak"] = max(
                state["max_joint_streak"], state["joint_streak"]
            )
            state["previous_joint_within"] = joint_within
            if not (terminated or truncated):
                continue
            collision = bool(
                info["obstacle_collision"] or info["self_collision"]
                or info["environment_collision"]
            )
            outcomes[int(state["episode"])] = {
                "episode": int(state["episode"]),
                "seed": seed + int(state["episode"]),
                "position_bin": int(state["position_bin"]),
                "orientation_bin": int(state["orientation_bin"]),
                "target_distance_m": state["target_distance_m"],
                "target_orientation_rad": state["target_orientation_rad"],
                "success": int(bool(info["task_reached"])),
                "timeout": int(bool(truncated)),
                "collision": int(collision),
                "joint_limit": int(bool(info["joint_limit"])),
                "steps": int(state["steps"]),
                "final_position_error_m": float(info["next_rho_position"]),
                "final_orientation_error_rad": float(info["next_rho_orientation"]),
                "final_position_ratio": p_ratio,
                "final_orientation_ratio": o_ratio,
                **{
                    key: state[key] for key in (
                        "position_within_steps", "orientation_within_steps",
                        "joint_within_steps", "joint_entries", "joint_exits",
                        "max_joint_streak", "minimum_joint_ratio",
                        "closest_position_ratio", "closest_orientation_ratio",
                        "minimum_position_ratio", "minimum_orientation_ratio",
                    )
                },
                "mean_wrist_action_abs": (
                    state["wrist_action_abs_sum"] / max(state["steps"], 1)
                ).tolist(),
                "mean_wrist_joint_velocity_abs": (
                    state["wrist_velocity_abs_sum"] / max(state["steps"], 1)
                ).tolist(),
                "max_wrist_action_abs": state["wrist_action_max_abs"].tolist(),
                "max_wrist_joint_velocity_abs": state["wrist_velocity_max_abs"].tolist(),
                "wrist_action_saturation_rate": (
                    state["wrist_action_saturated_steps"] / max(state["steps"], 1)
                ),
                "orientation_error_trace_rad": state["orientation_error_trace"],
                "orientation_error_vector_trace": state["orientation_error_vector_trace"],
                "wrist_action_trace": state["wrist_action_trace"],
            }
            finished.append(worker)
        for worker in finished:
            del active[worker]
        if len(outcomes) >= next_report:
            print(f"completed {len(outcomes)}/{episodes}", flush=True)
            next_report += 100
    return [outcomes[index] for index in range(episodes)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timeouts = [row for row in rows if row["timeout"]]

    def mean(field: str) -> float:
        return float(np.mean([row[field] for row in timeouts])) if timeouts else 0.0

    def mean_vector(field: str) -> list[float]:
        if not timeouts:
            return [0.0, 0.0, 0.0]
        return np.mean(np.asarray([row[field] for row in timeouts], dtype=np.float64), axis=0).tolist()

    bins = []
    for p_bin in range(10):
        for o_bin in range(10):
            selected = [
                row for row in rows
                if row["position_bin"] == p_bin and row["orientation_bin"] == o_bin
            ]
            failed = [row for row in selected if row["timeout"]]
            bins.append({
                "position_bin": p_bin, "orientation_bin": o_bin,
                "episodes": len(selected), "timeouts": len(failed),
                "timeout_rate": len(failed) / len(selected) if selected else 0.0,
                "mean_timeout_final_position_ratio": (
                    float(np.mean([row["final_position_ratio"] for row in failed]))
                    if failed else None
                ),
                "mean_timeout_final_orientation_ratio": (
                    float(np.mean([row["final_orientation_ratio"] for row in failed]))
                    if failed else None
                ),
            })
    top_bins = sorted(
        bins, key=lambda row: (row["timeouts"], row["timeout_rate"]), reverse=True,
    )[:20]
    p_marginal = [
        sum(row["timeout"] for row in rows if row["position_bin"] == index)
        for index in range(10)
    ]
    o_marginal = [
        sum(row["timeout"] for row in rows if row["orientation_bin"] == index)
        for index in range(10)
    ]
    entered = [row for row in timeouts if row["joint_entries"] > 0]
    near = [
        row for row in timeouts
        if row["joint_entries"] == 0 and row["minimum_joint_ratio"] <= 1.5
    ]
    far = [row for row in timeouts if row["minimum_joint_ratio"] > 1.5]
    return {
        "episodes": len(rows), "timeouts": len(timeouts),
        "timeout_rate": len(timeouts) / len(rows),
        "timeout_final_error": {
            "mean_position_ratio": mean("final_position_ratio"),
            "mean_orientation_ratio": mean("final_orientation_ratio"),
            "position_larger_count": sum(
                row["final_position_ratio"] > row["final_orientation_ratio"]
                for row in timeouts
            ),
            "orientation_larger_count": sum(
                row["final_orientation_ratio"] > row["final_position_ratio"]
                for row in timeouts
            ),
            "position_outside_count": sum(row["final_position_ratio"] > 1 for row in timeouts),
            "orientation_outside_count": sum(row["final_orientation_ratio"] > 1 for row in timeouts),
        },
        "timeout_behavior": {
            "never_near_joint_tolerance_count": len(far),
            "near_but_never_entered_count": len(near),
            "entered_joint_tolerance_count": len(entered),
            "entered_and_exited_count": sum(row["joint_exits"] > 0 for row in entered),
            "multiple_entries_count": sum(row["joint_entries"] > 1 for row in entered),
            "mean_minimum_joint_ratio": mean("minimum_joint_ratio"),
            "mean_max_joint_streak_steps": mean("max_joint_streak"),
            "mean_wrist_action_abs": mean_vector("mean_wrist_action_abs"),
            "mean_wrist_joint_velocity_abs": mean_vector("mean_wrist_joint_velocity_abs"),
            "mean_max_wrist_action_abs": mean_vector("max_wrist_action_abs"),
            "mean_max_wrist_joint_velocity_abs": mean_vector("max_wrist_joint_velocity_abs"),
            "mean_wrist_action_saturation_rate": mean("wrist_action_saturation_rate"),
        },
        "position_bin_timeout_counts": p_marginal,
        "orientation_bin_timeout_counts": o_marginal,
        "top_timeout_bins": top_bins,
        "bins": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", required=True)
    parser.add_argument("--config", default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    parser.add_argument("--level-index", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(ROOT / args.config)
    config["device"] = "cpu"
    env = ThesisHomotopyEnv(config)
    try:
        agent = ThesisSACAgent(env.observation_space.shape[0], env.action_space.shape[0], config)
    finally:
        env.close()
    actor = Path(args.actor)
    if not actor.is_absolute():
        actor = ROOT / actor
    load_actor(agent, actor)
    worker_seeds = [args.seed + 1000000 + index for index in range(args.num_envs)]
    with ParallelThesisEnvPool(config, worker_seeds) as pool:
        rows = evaluate(
            agent, pool, config, episodes=args.episodes, seed=args.seed,
            level_index=args.level_index,
        )
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=False)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
