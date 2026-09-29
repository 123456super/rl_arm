#!/usr/bin/env python3
"""Collect reproducible, complete O7--O9 failed-event reset states from a frozen actor."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import torch

from audit_large_angle_bad_state_replay import RUN, ROOT, select_centers
from analyze_orientation_state_provenance import STEP_INFO_KEYS, evaluate
from evaluate_thesis_homotopy import physical_cpu_ids
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=RUN / "checkpoints/step_1000000.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    config = load_config(ROOT / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    config["device"] = "cpu"
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state["config"]["thesis"] != config["thesis"]:
        raise ValueError("checkpoint environment contract does not match the collector")
    agent = ThesisSACAgent(166, 6, config)
    agent.load_state_dict(state["agent"])
    agent.actor.eval()
    ids = [index for index in range(1000) if index % 10 in (7, 8, 9)]
    with ParallelThesisEnvPool(
        config, [1051001 + index for index in range(args.num_envs)],
        cpu_ids=physical_cpu_ids(args.num_envs), step_info_keys=STEP_INFO_KEYS,
        start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn",
    ) as pool:
        episodes, rows, _ = evaluate(
            agent, pool, config, episodes=len(ids), seed=51001, level_index=0,
            band_low=.6, band_high=1., damping=.05, counterfactuals=False,
            episode_ids=ids, capture_all_large_states=True, capture_observations=True,
            capture_episode_states=True,
        )
    bad, _ = select_centers(episodes, rows)
    if (sum(e["success"] for e in episodes), sum(not e["success"] for e in episodes)) != (207, 93):
        raise RuntimeError("frozen replay did not reproduce the audited 207/93 outcomes")
    centers: dict[tuple[int, int], dict] = {}
    for kind, items in bad.items():
        for item in items:
            key = (item["episode"], item["step"])
            snapshot = item["episode_state"]
            record = centers.setdefault(key, {
                "episode": item["episode"], "step": item["step"],
                "rho_R": item["rho_orientation"], "rho_p": item["rho_position"],
                "observation": item["observation"], "state": snapshot, "events": [],
            })
            record["events"].append(kind)
    if not centers:
        raise RuntimeError("no failed-event reset states were collected")
    payload = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "seed": 51001, "episode_counts": {"success": 207, "failure": 93},
        "event_counts": {key: len(items) for key, items in bad.items()},
        "centers": list(centers.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved {len(centers)} unique bad-state starts to {args.output}", flush=True)


if __name__ == "__main__":
    main()
