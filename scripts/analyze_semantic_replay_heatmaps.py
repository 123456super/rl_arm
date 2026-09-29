#!/usr/bin/env python3
"""Paired 10x10 pose probes and semantic-replay diagnostics for two actors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.utils.config import load_config


def load_actor(agent: ThesisSACAgent, path: Path) -> None:
    state = torch.load(path, map_location=agent.device, weights_only=False)
    if isinstance(state, dict) and "agent" in state:
        actor_state = state["agent"]["actor"]
    elif isinstance(state, dict) and "actor" in state:
        actor_state = state["actor"]
    else:
        actor_state = state
    agent.actor.load_state_dict(actor_state)
    agent.actor.eval()


def evaluate_actor(
    agent: ThesisSACAgent,
    pool: ParallelThesisEnvPool,
    config: dict[str, Any],
    *,
    label: str,
    episodes: int,
    seed: int,
    level: dict[str, float],
    orientation_scale: float,
) -> list[dict[str, Any]]:
    position_bins = 10
    orientation_bins = 10
    position_edges = np.linspace(
        level["target_distance_min_m"], level["target_distance_max_m"],
        position_bins + 1,
    )
    orientation_edges = np.linspace(
        level["target_orientation_min_rad"],
        level["target_orientation_max_rad"], orientation_bins + 1,
    )
    base_contract = {
        "scene": "none", "xi": 1.0, "strict": True,
        "goal_scale": float(level["goal_scale"]),
        "orientation_scale": float(orientation_scale),
        "position_tolerance": float(level["position_tolerance_m"]),
        "orientation_tolerance": float(level["orientation_tolerance_rad"]),
        "lambda_self": 1.0,
    }
    active: dict[int, dict[str, Any]] = {}
    outcomes: dict[int, dict[str, Any]] = {}
    next_episode = 0
    next_report = 100
    while len(outcomes) < episodes:
        assignments: dict[int, int] = {}
        for worker_index in range(len(pool)):
            if worker_index in active or next_episode >= episodes:
                continue
            episode = next_episode
            next_episode += 1
            assignments[worker_index] = episode
            position_bin = (episode // orientation_bins) % position_bins
            orientation_bin = episode % orientation_bins
            active[worker_index] = {
                "episode": episode,
                "position_bin": position_bin,
                "orientation_bin": orientation_bin,
                "length": 0,
            }
        if assignments:
            pool.set_rng_states({
                worker: np.random.default_rng(seed + episode).bit_generator.state
                for worker, episode in assignments.items()
            })
            contracts = {}
            for worker, episode in assignments.items():
                state = active[worker]
                p_bin = int(state["position_bin"])
                o_bin = int(state["orientation_bin"])
                contracts[worker] = {
                    **base_contract,
                    "target_distance_min_m": float(position_edges[p_bin]),
                    "target_distance_max_m": float(position_edges[p_bin + 1]),
                    "target_orientation_min_rad": float(orientation_edges[o_bin]),
                    "target_orientation_max_rad": float(orientation_edges[o_bin + 1]),
                }
            for worker, (observation, _) in pool.reset_many(contracts).items():
                active[worker]["observation"] = observation
        workers = sorted(active)
        observations = np.stack([active[index]["observation"] for index in workers])
        actions = agent.select_actions(observations, deterministic=True)
        results = pool.step_many(dict(zip(workers, actions)))
        finished = []
        for worker in workers:
            observation, _, _, terminated, truncated, info = results[worker]
            state = active[worker]
            state["observation"] = observation
            state["length"] += 1
            if not (terminated or truncated):
                continue
            episode = int(state["episode"])
            collision = bool(
                info["obstacle_collision"] or info["self_collision"]
                or info["environment_collision"]
            )
            outcomes[episode] = {
                "actor": label,
                "episode": episode,
                "seed": seed + episode,
                "position_bin": int(state["position_bin"]),
                "orientation_bin": int(state["orientation_bin"]),
                "sampled_target_distance_m": float(info["sampled_target_distance_m"]),
                "sampled_target_orientation_rad": float(
                    info["sampled_target_orientation_rad"]
                ),
                "success": int(bool(info["task_reached"])),
                "timeout": int(bool(truncated)),
                "collision": int(collision),
                "joint_limit": int(bool(info["joint_limit"])),
                "final_position_error_m": float(info["next_rho_position"]),
                "final_orientation_error_rad": float(info["next_rho_orientation"]),
                "episode_length": int(state["length"]),
            }
            finished.append(worker)
        for worker in finished:
            del active[worker]
        if len(outcomes) >= next_report:
            print(f"{label}: {len(outcomes)}/{episodes} episodes", flush=True)
            next_report += 100
    return [outcomes[index] for index in range(episodes)]


def replay_bin_metrics(checkpoint: Path) -> list[dict[str, Any]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    semantic = state["replay"].get("semantic_long_term")
    if not semantic or not semantic.get("enabled"):
        raise ValueError(f"checkpoint has no semantic replay: {checkpoint}")
    slots = semantic["slots"]
    seen = np.asarray(semantic["seen"], dtype=np.int64)
    rows = []
    for bin_index in range(100):
        parts = slots.get(bin_index, [])
        successful = 0
        transitions = 0
        for part in parts:
            indices = part.chronological_indices()
            transitions += int(part.size)
            if len(indices):
                successful += int(bool(part.raw[indices[-1], 5]))
        rows.append({
            "position_bin": bin_index // 10,
            "orientation_bin": bin_index % 10,
            "replay_seen_episodes": int(seen[bin_index]),
            "replay_stored_episodes": len(parts),
            "replay_successful_episodes": successful,
            "replay_success_fraction": (
                successful / len(parts) if parts else float("nan")
            ),
            "replay_transitions": transitions,
        })
    del state
    return rows


def aggregate_bins(
    outcomes: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, float]]:
    result = {}
    for p_bin in range(10):
        for o_bin in range(10):
            rows = [
                row for row in outcomes
                if row["position_bin"] == p_bin
                and row["orientation_bin"] == o_bin
            ]
            result[(p_bin, o_bin)] = {
                "success_rate": float(np.mean([row["success"] for row in rows])),
                "timeout_rate": float(np.mean([row["timeout"] for row in rows])),
                "collision_rate": float(np.mean([row["collision"] for row in rows])),
                "mean_position_error_m": float(np.mean([
                    row["final_position_error_m"] for row in rows
                ])),
                "mean_orientation_error_rad": float(np.mean([
                    row["final_orientation_error_rad"] for row in rows
                ])),
            }
    return result


def draw_heatmap(
    axis, values: np.ndarray, title: str, *, vmin: float, vmax: float,
    cmap: str, value_format: str,
) -> None:
    image = axis.imshow(values, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    axis.set_title(title)
    axis.set_xlabel("Orientation-change bin (small → large)")
    axis.set_ylabel("Target-distance bin (near → far)")
    axis.set_xticks(range(10)); axis.set_yticks(range(10))
    threshold = (vmin + vmax) / 2
    for row in range(10):
        for column in range(10):
            value = values[row, column]
            color = "white" if value < threshold else "black"
            axis.text(
                column, row, format(value, value_format), ha="center", va="center",
                fontsize=7, color=color,
            )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    parser.add_argument("--actor-400k", required=True)
    parser.add_argument("--actor-425k", required=True)
    parser.add_argument("--checkpoint-400k", required=True)
    parser.add_argument("--checkpoint-425k", required=True)
    parser.add_argument("--level-index", type=int, default=6)
    parser.add_argument("--episodes-per-bin", type=int, default=10)
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda:auto"), default="cuda:auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.episodes_per_bin < 1 or args.num_envs < 1:
        parser.error("episodes-per-bin and num-envs must be positive")
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / args.config)
    config["device"] = args.device
    level = config["thesis"]["joint_pose_curriculum"]["levels"][args.level_index]
    orientation_scale = config["thesis"]["orientation_curriculum"]["levels"][args.level_index]
    episodes = 100 * args.episodes_per_bin
    pool = ParallelThesisEnvPool(
        config, [args.seed + 1_000_000 + index for index in range(args.num_envs)],
    )
    try:
        agent = ThesisSACAgent(pool.observation_dim, pool.action_dim, config)
        all_outcomes = []
        for label, actor_path in (
            ("400k", ROOT / args.actor_400k),
            ("425k", ROOT / args.actor_425k),
        ):
            load_actor(agent, actor_path)
            all_outcomes.extend(evaluate_actor(
                agent, pool, config, label=label, episodes=episodes,
                seed=args.seed, level=level, orientation_scale=orientation_scale,
            ))
    finally:
        pool.close()

    fields = list(all_outcomes[0])
    with (output / "episode_outcomes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(all_outcomes)
    by_actor = {
        label: aggregate_bins([row for row in all_outcomes if row["actor"] == label])
        for label in ("400k", "425k")
    }
    replay = {
        "400k": replay_bin_metrics(ROOT / args.checkpoint_400k),
        "425k": replay_bin_metrics(ROOT / args.checkpoint_425k),
    }
    replay_lookup = {
        label: {(row["position_bin"], row["orientation_bin"]): row for row in rows}
        for label, rows in replay.items()
    }
    actor_outcomes = {
        label: {
            int(row["episode"]): row
            for row in all_outcomes if row["actor"] == label
        }
        for label in ("400k", "425k")
    }
    bin_rows = []
    for p_bin in range(10):
        for o_bin in range(10):
            key = (p_bin, o_bin)
            episode_ids = [p_bin * 10 + o_bin + 100 * cycle for cycle in range(args.episodes_per_bin)]
            paired = [
                (actor_outcomes["400k"][episode], actor_outcomes["425k"][episode])
                for episode in episode_ids
            ]
            row: dict[str, Any] = {
                "position_bin": p_bin, "orientation_bin": o_bin,
                "distance_min_m": float(np.linspace(
                    level["target_distance_min_m"], level["target_distance_max_m"], 11,
                )[p_bin]),
                "distance_max_m": float(np.linspace(
                    level["target_distance_min_m"], level["target_distance_max_m"], 11,
                )[p_bin + 1]),
                "orientation_min_rad": float(np.linspace(
                    level["target_orientation_min_rad"], level["target_orientation_max_rad"], 11,
                )[o_bin]),
                "orientation_max_rad": float(np.linspace(
                    level["target_orientation_min_rad"], level["target_orientation_max_rad"], 11,
                )[o_bin + 1]),
            }
            for label in ("400k", "425k"):
                row.update({
                    f"{label}_{metric}": value
                    for metric, value in by_actor[label][key].items()
                })
                row.update({
                    f"{label}_{metric}": value
                    for metric, value in replay_lookup[label][key].items()
                    if metric not in {"position_bin", "orientation_bin"}
                })
            row.update({
                "success_delta_425k_minus_400k": (
                    row["425k_success_rate"] - row["400k_success_rate"]
                ),
                "paired_both_success": sum(a["success"] and b["success"] for a, b in paired),
                "paired_both_fail": sum(not a["success"] and not b["success"] for a, b in paired),
                "paired_recovered_at_425k": sum(not a["success"] and b["success"] for a, b in paired),
                "paired_regressed_at_425k": sum(a["success"] and not b["success"] for a, b in paired),
            })
            row["diagnosis"] = (
                "insufficient_samples"
                if row["425k_replay_stored_episodes"] < 8
                else "low_replay_success_quality"
                if row["425k_replay_success_fraction"] < .75
                and row["425k_success_rate"] < .8
                else "actor_or_critic_retention"
                if row["425k_success_rate"] < .8
                else "adequate"
            )
            bin_rows.append(row)
    with (output / "bin_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bin_rows[0]))
        writer.writeheader(); writer.writerows(bin_rows)

    def matrix(field: str) -> np.ndarray:
        result = np.zeros((10, 10), dtype=np.float64)
        for row in bin_rows:
            result[row["position_bin"], row["orientation_bin"]] = row[field]
        return result

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7), constrained_layout=True)
    image = draw_heatmap(
        axes[0], matrix("400k_success_rate"), "400k actor success rate",
        vmin=0.0, vmax=1.0, cmap="RdYlGn", value_format=".1f",
    )
    draw_heatmap(
        axes[1], matrix("425k_success_rate"), "425k actor success rate",
        vmin=0.0, vmax=1.0, cmap="RdYlGn", value_format=".1f",
    )
    delta_image = draw_heatmap(
        axes[2], matrix("success_delta_425k_minus_400k"), "425k − 400k",
        vmin=-1.0, vmax=1.0, cmap="RdBu", value_format="+.1f",
    )
    fig.colorbar(image, ax=axes[:2], shrink=.75, label="Success rate")
    fig.colorbar(delta_image, ax=axes[2], shrink=.75, label="Success-rate change")
    fig.savefig(output / "success_rate_heatmaps.png", dpi=180)
    fig.savefig(output / "success_rate_heatmaps.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7), constrained_layout=True)
    draw_heatmap(
        axes[0], matrix("paired_both_fail"), "Fixed failures (both actors)",
        vmin=0, vmax=args.episodes_per_bin, cmap="Reds", value_format=".0f",
    )
    draw_heatmap(
        axes[1], matrix("425k_replay_seen_episodes"), "Replay episodes ever seen",
        vmin=0, vmax=float(max(row["425k_replay_seen_episodes"] for row in bin_rows)),
        cmap="Blues", value_format=".0f",
    )
    draw_heatmap(
        axes[2], matrix("425k_replay_success_fraction"), "Stored replay success fraction",
        vmin=0, vmax=1, cmap="RdYlGn", value_format=".2f",
    )
    fig.savefig(output / "failure_diagnostics.png", dpi=180)
    fig.savefig(output / "failure_diagnostics.pdf")
    plt.close(fig)

    hard = sorted(
        [row for row in bin_rows if row["400k_success_rate"] < .8 and row["425k_success_rate"] < .8],
        key=lambda row: (row["425k_success_rate"], row["400k_success_rate"]),
    )
    regressed = sorted(
        [row for row in bin_rows if row["success_delta_425k_minus_400k"] <= -.2],
        key=lambda row: row["success_delta_425k_minus_400k"],
    )
    summary = {
        "episodes_per_actor": episodes,
        "seed": args.seed,
        "level_index": args.level_index,
        "overall": {
            label: {
                "success_rate": float(np.mean([
                    row["success"] for row in all_outcomes if row["actor"] == label
                ])),
                "timeout_rate": float(np.mean([
                    row["timeout"] for row in all_outcomes if row["actor"] == label
                ])),
                "collision_rate": float(np.mean([
                    row["collision"] for row in all_outcomes if row["actor"] == label
                ])),
                "minimum_bin_success_rate": min(
                    value["success_rate"] for value in by_actor[label].values()
                ),
            } for label in ("400k", "425k")
        },
        "persistent_hard_bin_count": len(hard),
        "regressed_bin_count": len(regressed),
        "diagnosis_counts": {
            diagnosis: sum(row["diagnosis"] == diagnosis for row in bin_rows)
            for diagnosis in sorted(set(row["diagnosis"] for row in bin_rows))
        },
        "persistent_hard_bins": hard,
        "regressed_bins": regressed,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary["overall"], ensure_ascii=False), flush=True)
    print(f"wrote analysis to {output}", flush=True)


if __name__ == "__main__":
    main()
