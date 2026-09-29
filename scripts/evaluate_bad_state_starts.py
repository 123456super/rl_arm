#!/usr/bin/env python3
"""Evaluate frozen actors from the same recorded failed-event environment states."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.utils.config import load_config


def evaluate(agent, pool, centers):
    pending = iter(enumerate(centers))
    active = {}
    finished = {}
    while len(finished) < len(centers):
        resets = {}
        for worker in range(len(pool)):
            if worker not in active:
                item = next(pending, None)
                if item is not None:
                    index, center = item
                    resets[worker] = center["state"]
                    active[worker] = {
                        "index": index, "steps": 0, "first_10_progress": [],
                        "first_10_reversal": [], "minimum_rho_R": float(center["rho_R"]),
                    }
        if resets:
            for worker, observation in pool.restore_many(resets).items():
                if not np.allclose(observation, centers[active[worker]["index"]]["observation"], atol=1e-5):
                    raise ValueError("snapshot reconstruction differs from saved observation")
                active[worker]["observation"] = observation
        if not active:
            break
        actions = agent.select_actions(
            np.stack([active[w]["observation"] for w in sorted(active)]), deterministic=True,
        )
        results = pool.step_many(dict(zip(sorted(active), actions)))
        for worker, (observation, _, _, terminated, truncated, info) in results.items():
            episode = active[worker]
            episode["steps"] += 1
            episode["observation"] = observation
            episode["minimum_rho_R"] = min(episode["minimum_rho_R"], float(info["next_rho_orientation"]))
            if episode["steps"] <= 10:
                delta = float(info["rho_orientation"] - info["next_rho_orientation"])
                episode["first_10_progress"].append(delta)
                episode["first_10_reversal"].append(delta < 0)
            if terminated or truncated:
                episode["success"] = bool(info["task_reached"])
                episode["timeout"] = bool(truncated)
                episode["collision"] = bool(
                    info["obstacle_collision"] or info["self_collision"] or info["environment_collision"]
                )
                episode.pop("observation")
                finished[episode["index"]] = episode
                del active[worker]
    return [finished[index] for index in range(len(centers))]


def summary(records):
    return {
        "n": len(records), "success": sum(item["success"] for item in records),
        "timeout": sum(item["timeout"] for item in records),
        "collision": sum(item["collision"] for item in records),
        "success_rate": float(np.mean([item["success"] for item in records])) if records else None,
        "first_10_mean_orientation_progress_rad_per_step": float(np.mean([
            x for item in records for x in item["first_10_progress"]
        ])) if records else None,
        "first_10_orientation_reversal_fraction": float(np.mean([
            x for item in records for x in item["first_10_reversal"]
        ])) if records else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    torch.set_num_threads(1)
    config = load_config(ROOT / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    config["device"] = "cpu"
    bank = torch.load(args.starts, map_location="cpu", weights_only=False)
    if Path(bank["source_checkpoint"]).resolve() != args.baseline.resolve():
        parser.error("bad-state starts were collected from a different baseline")
    centers = bank["centers"]
    result = {"starts": str(args.starts.resolve()), "n_unique_centers": len(centers), "actors": {}}
    with ParallelThesisEnvPool(
        config, [410001 + i for i in range(args.num_envs)],
        step_info_keys=("task_reached", "rho_orientation", "next_rho_orientation",
                        "timeout", "obstacle_collision", "self_collision",
                        "environment_collision"),
        start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn",
    ) as pool:
        agent = ThesisSACAgent(pool.observation_dim, pool.action_dim, config)
        for name, path in (("baseline", args.baseline), ("target", args.target)):
            state = torch.load(path, map_location="cpu", weights_only=False)
            agent.actor.load_state_dict(state.get("agent", state).get("actor", state))
            agent.actor.eval()
            records = evaluate(agent, pool, centers)
            result["actors"][name] = {
                "checkpoint": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "overall": summary(records),
                "by_event": {
                    kind: summary([r for r, c in zip(records, centers) if kind in c["events"]])
                    for kind in ("low_alignment", "stagnation", "post_crossing_rebound")
                },
                "episodes": [{"center_index": i, "source_episode": centers[i]["episode"],
                              "source_step": centers[i]["step"], "events": centers[i]["events"],
                              **record} for i, record in enumerate(records)],
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, record in result["actors"].items():
        print(name, record["overall"], flush=True)


if __name__ == "__main__":
    main()
