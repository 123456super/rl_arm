#!/usr/bin/env python3
"""Build immutable success anchors and perturbed hard-state starts from frozen 1M."""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.algorithms.homotopy_replay import HomotopyReplayBuffer, _Partition
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bad-state-starts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--num-envs", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing seed")
    torch.set_num_threads(1)
    config = load_config(ROOT / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    config["device"] = "cpu"
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    bank = torch.load(args.bad_state_starts, map_location="cpu", weights_only=False)
    if Path(bank["source_checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("bad-state bank and frozen policy must have the same 1M source")
    if len(bank["centers"]) != 85 or source["curriculum"]["orientation"].level_index != 0:
        raise ValueError("expected the audited 85 states and level-0 1M policy")
    if source["config"]["thesis"] != config["thesis"] or source["config"]["sac"] != config["sac"]:
        raise ValueError("frozen checkpoint does not match the four-pool training contract")
    level = config["thesis"]["joint_pose_curriculum"]["levels"][0]
    contract = {
        "scene": "none", "xi": 1.0, "strict": True, "lambda_self": 0.2,
        "orientation_scale": 1.0, "goal_scale": float(level["goal_scale"]),
        "position_tolerance": float(level["position_tolerance_m"]),
        "orientation_tolerance": float(level["orientation_tolerance_rad"]),
        **{key: float(level[key]) for key in (
            "target_distance_min_m", "target_distance_max_m",
            "target_orientation_min_rad", "target_orientation_max_rad",
        )},
    }
    d_edges = np.linspace(level["target_distance_min_m"], level["target_distance_max_m"], 11)
    o_edges = np.linspace(level["target_orientation_min_rad"], level["target_orientation_max_rad"], 11)
    with ParallelThesisEnvPool(
        config, [610001 + i for i in range(args.num_envs)],
        start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn",
    ) as pool:
        agent = ThesisSACAgent(pool.observation_dim, pool.action_dim, config)
        agent.actor.load_state_dict(source["agent"]["actor"])
        agent.actor.eval()
        replay = HomotopyReplayBuffer(
            pool.observation_dim, pool.action_dim, "cpu", s0_joint_pose=True,
            reward_parameters=source["replay"]["reward_parameters"],
        )
        replay.load_state_dict(source["replay"])
        replay.four_pool_enabled = True
        replay.s0_current = _Partition(pool.observation_dim, pool.action_dim, 30000, replay.raw_dim)
        anchor = {tier: _Partition(pool.observation_dim, pool.action_dim, capacity, replay.raw_dim)
                  for tier, capacity in (("o0_o4", 9600), ("o5_o6", 7200), ("o7_o9", 7200))}
        frontier = {i: _Partition(pool.observation_dim, pool.action_dim, 176, replay.raw_dim)
                    for i in range(85)}
        replay.frontier = frontier
        active = {}
        launched = completed = 0
        tier_bins = (("o0_o4", (0, 1, 2, 3, 4)),
                     ("o5_o6", (5, 6)), ("o7_o9", (7, 8, 9)))
        tier_launched = {tier: 0 for tier, _ in tier_bins}
        while completed < args.episodes:
            resets = {}
            seeds = {}
            for worker in range(len(pool)):
                if worker in active or launched >= args.episodes:
                    continue
                episode = launched
                launched += 1
                tier_index = episode % 10
                tier, bins = tier_bins[0 if tier_index < 4 else 1 if tier_index < 7 else 2]
                tier_episode = tier_launched[tier]
                tier_launched[tier] += 1
                orientation_bin = bins[(tier_episode // 10) % len(bins)]
                position_bin = tier_episode % 10
                task = dict(contract)
                task.update({
                    "target_distance_min_m": float(d_edges[position_bin]),
                    "target_distance_max_m": float(d_edges[position_bin + 1]),
                    "target_orientation_min_rad": float(o_edges[orientation_bin]),
                    "target_orientation_max_rad": float(o_edges[orientation_bin + 1]),
                })
                resets[worker] = task
                seeds[worker] = 61001 + episode
                active[worker] = {"episode": episode, "step": 0, "tier": tier}
            for worker, (obs, _) in pool.reset_many(resets, seeds=seeds).items():
                active[worker]["obs"] = obs
            indices = sorted(active)
            actions = agent.select_actions(np.stack([active[w]["obs"] for w in indices]), deterministic=True)
            for worker, (next_obs, _, _, terminated, truncated, info) in pool.step_many(dict(zip(indices, actions))).items():
                record = active[worker]
                replay.add("none", record["obs"], actions[indices.index(worker)], next_obs,
                           terminated or truncated, info, record["episode"], record["step"],
                           curriculum_level=0)
                record["step"] += 1
                record["obs"] = next_obs
                if terminated or truncated:
                    if info["task_reached"]:
                        part = anchor[record["tier"]]
                        indices_episode = replay.s0_current.episode_indices(record["episode"])
                        if part.size + len(indices_episode) <= part.capacity:
                            replay.s0_current.copy_episode_to(record["episode"], part)
                    completed += 1
                    del active[worker]
            if completed % 100 == 0 and completed:
                print(f"normal {completed}/{args.episodes}: " +
                      ", ".join(f"{key}={part.size}" for key, part in anchor.items()), flush=True)
        print("frozen normal collection complete; generating perturbed frontier", flush=True)
        active.clear()
        launched = completed = 0
        while completed < len(frontier):
            resets = {}
            for worker in range(len(pool)):
                if worker not in active and launched < len(frontier):
                    index = launched
                    launched += 1
                    resets[worker] = (bank["centers"][index], dict(contract))
                    active[worker] = {"episode": 1_000_000 + index,
                                      "center": index, "start": bank["centers"][index]["step"],
                                      "step": bank["centers"][index]["step"]}
            for worker, (obs, _) in pool.reset_bad_states_many(resets).items():
                active[worker]["obs"] = obs
            indices = sorted(active)
            actions = agent.select_actions(np.stack([active[w]["obs"] for w in indices]), deterministic=True)
            for worker, (next_obs, _, _, terminated, truncated, info) in pool.step_many(dict(zip(indices, actions))).items():
                record = active[worker]
                replay.add("none", record["obs"], actions[indices.index(worker)], next_obs,
                           terminated or truncated, info, record["episode"], record["step"],
                           curriculum_level=0)
                error = np.asarray(info["orientation_error_vector"], dtype=float)
                omega = np.asarray(info["ee_angular_velocity"], dtype=float)
                norm = np.linalg.norm(error) * np.linalg.norm(omega)
                replay.add_frontier_from_recent(record["episode"], record["center"],
                                                record["start"], recovered=False,
                                                alignment=(float(np.dot(error, omega) / norm)
                                                           if norm > 1e-8 else None))
                record["step"] += 1
                record["obs"] = next_obs
                if terminated or truncated:
                    if info["task_reached"]:
                        replay.add_frontier_from_recent(record["episode"], record["center"],
                                                        record["start"], recovered=True)
                    completed += 1
                    del active[worker]
        payload = {
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "bad_state_starts": str(args.bad_state_starts.resolve()),
            "anchor": anchor, "frontier": frontier,
            "episodes": args.episodes,
        }
        if any(part.size < 256 for part in anchor.values()) or any(part.size < 1 for part in frontier.values()):
            raise RuntimeError("frozen seed collection did not fill all replay strata")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.output)
        print(f"saved {args.output}: anchor " + ", ".join(
            f"{key}={part.size}" for key, part in anchor.items()) +
            f"; frontier={sum(p.size for p in frontier.values())}; recent starts empty", flush=True)


if __name__ == "__main__":
    main()
