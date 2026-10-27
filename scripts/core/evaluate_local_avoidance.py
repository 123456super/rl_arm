#!/usr/bin/env python3
"""Evaluate local passage on held-out S0-generated starts, not final reaching."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch

from rl_risk_sac.algorithms.dual_sac import build_thesis_sac
from rl_risk_sac.envs.local_avoidance_env import LocalAvoidanceEnv
from rl_risk_sac.tasks.local_avoidance import PROTOCOL, GEOMETRIC_PROTOCOL, local_protocol, LOCAL_OBSERVATION_DIM, actor_digest
from rl_risk_sac.tasks.local_samples import validate_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("protocol") not in (PROTOCOL, GEOMETRIC_PROTOCOL):
        parser.error("evaluation requires a local passage checkpoint")
    config = copy.deepcopy(state["config"])
    if state["protocol"] != local_protocol(config):
        raise ValueError("checkpoint reference mode/protocol mismatch")
    config["device"] = "cpu"
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    agent.load_state_dict(state["agent"])
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    identities = validate_bank(bank, config, agent.reaching, split=("validation", "test"))
    trace_seeds = {sample["provenance"]["seed"] for sample in bank["samples"]}
    if identities.intersection(state["train_sample_ids"]) or trace_seeds.intersection(state["train_trace_seeds"]):
        raise ValueError("evaluation starts/traces overlap with training")
    before = actor_digest(agent.reaching.actor)
    env = LocalAvoidanceEnv(config, agent.reaching)
    outcomes = []
    try:
        for sample in bank["samples"]:
            observation, _ = env.reset(options={"sample": sample})
            reward_sum, length = 0., 0
            minimum = float("inf")
            while True:
                action = agent.select_action(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                reward_sum += reward
                length += 1
                minimum = min(minimum, info["min_external_mesh_distance_m"])
                if terminated or truncated:
                    outcomes.append({**info, "length": length, "return": reward_sum,
                                     "episode_min_external_mesh_distance_m": minimum})
                    break
    finally:
        env.close()
    report = dict(protocol=local_protocol(config), samples=len(outcomes),
                  local_success_rate=float(np.mean([row["local_success"] for row in outcomes])),
                  failure_rate=float(np.mean([row["local_failure"] for row in outcomes])),
                  timeout_rate=float(np.mean([row["timeout"] for row in outcomes])),
                  frozen_reaching_actor_unchanged=before == actor_digest(agent.reaching.actor),
                  training_checkpoint_validation_mode=state["validation_mode"],
                  independent_of_training=True, outcomes=outcomes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "outcomes"}), flush=True)


if __name__ == "__main__":
    main()
