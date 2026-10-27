#!/usr/bin/env python3
"""Reward-only rescore of trusted recorded S1 episodes; no policy rollout."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from rl_risk_sac.algorithms.s1_expert_replay import file_sha256
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.goal_obstacle_coupling import coupling_parameters, coupling_potential
from rl_risk_sac.utils.config import load_config
from scripts.core.train_thesis_homotopy import apply_stage_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--config", required=True, help="explicit config containing the S1 coupling reward to audit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    torch.set_num_threads(1)
    config = load_config(args.config)
    apply_stage_profile(config, "s1")
    parameters = coupling_parameters(config)
    if not parameters:
        parser.error("audit requires enabled S1 coupling")
    states_hash = file_sha256(args.states)
    states = torch.load(args.states, map_location="cpu", weights_only=False)
    gamma = float(config["sac"]["gamma"])
    rows, initial_potentials, sources = [], {}, []
    env = ThesisHomotopyEnv(config)
    try:
        for path in args.source:
            source = json.loads(path.read_text())
            source_reward = copy.deepcopy(source["reward_config"])
            target_reward = copy.deepcopy(config["thesis"]["reward"])
            target_reward.pop("goal_obstacle_coupling", None)
            if source_reward != target_reward:
                raise ValueError("source must use the identical baseline reward without coupling")
            if (source["gamma"] != gamma or source["horizon"] != env.horizon
                    or source["source_sha256"].get(args.states.name) != states_hash):
                raise ValueError("source gamma/horizon/snapshot provenance mismatch")
            sources.append(dict(path=str(path.resolve()), sha256=file_sha256(path)))
            for row in source["trajectories"]:
                episode = row["episode"]
                trace = row["trace"]
                if (not trace or len(trace) != row["steps"] or row["cause"] not in
                        {"success", "timeout", "obstacle_collision", "self_collision",
                         "environment_collision", "joint_limit"}):
                    raise ValueError("audit requires complete physical episodes")
                if row["control_dt"] != env.control_dt:
                    raise ValueError("source control frequency mismatch")
                if episode not in initial_potentials:
                    initial = states[episode]["initial"]
                    if initial["step_count"] != 0:
                        raise ValueError("snapshot must be a true initial state")
                    env.restore_episode_state(initial)
                    distance = env._external_geometry.distance_min if env._external_geometry is not None else .8
                    initial_potentials[episode] = float(coupling_potential(
                        env.previous_rho_position, env.previous_rho_orientation,
                        distance, bool(env.obstacles), parameters))
                phi0 = initial_potentials[episode]
                present = bool(states[episode]["initial"]["obstacles"])
                phi = np.r_[phi0, coupling_potential(
                    [t["position_error_m"] for t in trace],
                    [t["orientation_error_rad"] for t in trace],
                    [t["encoded_external_distance_m"] for t in trace], present, parameters)]
                phi[-1] = 0.  # Absorbing terminal, including timeout.
                shaping = gamma * phi[1:] - phi[:-1]
                discount = gamma ** np.arange(len(trace))
                old = float(np.dot(discount, [t["reward"] for t in trace]))
                if abs(old - row["discounted_return"]) > 1e-7:
                    raise ValueError("source trace/return mismatch")
                shift = float(np.dot(discount, shaping))
                residual = abs(shift + phi0)
                if residual > 1e-10:
                    raise ValueError("coupling failed terminal-zero telescoping")
                rows.append(dict(episode=episode, mode=row["mode"], cause=row["cause"],
                    steps=len(trace), old_return=old, new_return=old + shift,
                    initial_potential=phi0, discounted_coupling=shift,
                    min_step_coupling=float(shaping.min()), max_step_coupling=float(shaping.max()),
                    telescoping_residual=residual))
    finally:
        env.close()
    comparisons = []
    for episode in sorted(initial_potentials):
        for a, b in itertools.combinations([r for r in rows if r["episode"] == episode], 2):
            old_gap = a["old_return"] - b["old_return"]
            new_gap = a["new_return"] - b["new_return"]
            comparisons.append(dict(episode=episode, a=a["mode"], b=b["mode"],
                old_gap=old_gap, new_gap=new_gap, gap_error=abs(new_gap - old_gap)))
    max_gap_error = max((r["gap_error"] for r in comparisons), default=0.)
    if max_gap_error > 1e-10:
        raise ValueError("coupling changed complete same-start return gaps")
    report = dict(method="reward-only rescoring of existing physical traces; restore initial scenes only",
        caveat="not new rollouts, a safety revalidation, a policy success-rate test, or a fix for baseline ranking counterexamples",
        config=str(Path(args.config).resolve()), parameters=parameters, gamma=gamma,
        sources=sources, states=str(args.states.resolve()), states_sha256=states_hash,
        episodes=len(initial_potentials), trajectories=len(rows),
        max_telescoping_residual=max(r["telescoping_residual"] for r in rows),
        max_same_start_gap_error=max_gap_error, comparisons=comparisons, trajectories_rescored=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {"comparisons", "trajectories_rescored"}}, indent=2))


if __name__ == "__main__":
    main()
