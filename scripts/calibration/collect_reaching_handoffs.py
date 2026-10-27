#!/usr/bin/env python3
"""Capture real joint-controller handoffs without using held-out scenes for training."""
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
from rl_risk_sac.control.joint_policy import JointPolicyController, JointControlFailure
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.local_avoidance import LOCAL_OBSERVATION_DIM, local_protocol, actor_digest
from rl_risk_sac.tasks.local_samples import simulation_config, validate_bank
from rl_risk_sac.tasks.reaching_handoff import BANK_PROTOCOL, capture_handoff, file_digest, validate_handoff_bank
from scripts.core.evaluate_joint_policy import restore_full_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--exclude-bank", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--max-scenes", type=int, default=2048)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=27101)
    args = parser.parse_args()
    if min(args.samples, args.max_scenes, args.max_episode_steps) < 1 or args.seed < 0:
        parser.error("positive budgets and nonnegative seed required")
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = copy.deepcopy(state["config"])
    if state.get("protocol") != local_protocol(config) or state["validation_mode"]:
        parser.error("collection requires a formal joint/local checkpoint")
    config["device"] = "cpu"
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    agent.load_state_dict(state["agent"], actor_only=True)
    for role in ("reaching", "avoidance"):
        agent.policy(role).actor.eval().requires_grad_(False)
    bank = torch.load(args.source_bank, map_location="cpu", weights_only=False)
    allowed = ("train",) if args.split == "train" else ("validation", "test")
    validate_bank(bank, config, agent.reaching, split=allowed)
    excluded = set()
    for path in args.exclude_bank:
        other = torch.load(path, map_location="cpu", weights_only=False)
        excluded.update(int(s["provenance"]["seed"]) for s in other["samples"])
    if args.split != "train":
        excluded.update(state["train_trace_seeds"])
    candidates = [s for s in bank["samples"] if int(s["provenance"]["seed"]) not in excluded]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(candidates))[:args.max_scenes]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    env = ThesisHomotopyEnv(simulation_config(config), cache_geometry_queries=True)
    samples, ids, scenes = [], set(), 0
    before = {role:actor_digest(agent.policy(role).actor) for role in ("reaching", "avoidance")}
    try:
        controller = JointPolicyController(config, agent)
        try:
            for index in order:
                source_scene = candidates[int(index)]
                observation = restore_full_scene(env, source_scene, config)
                env.horizon = args.max_episode_steps
                controller.reset()
                scenes += 1
                while not controller.finished:
                    previous = controller.passages
                    try:
                        observation, terminated, truncated, info = controller.step(env, observation)
                    except JointControlFailure:
                        break
                    if controller.passages > previous:
                        sample = capture_handoff(env, controller, source_scene)
                        if sample["sample_id"] not in ids:
                            ids.add(sample["sample_id"])
                            samples.append(sample)
                    if len(samples) >= args.samples or terminated or truncated:
                        break
                print(f"handoffs {len(samples)}/{args.samples}: scenes={scenes}", flush=True)
                if len(samples) >= args.samples:
                    break
        finally:
            controller.close()
    finally:
        env.close()
    if any(actor_digest(agent.policy(role).actor) != digest for role,digest in before.items()):
        raise RuntimeError("collection modified a source actor")
    result = dict(protocol=BANK_PROTOCOL, split=args.split, samples=samples,
                  source_checkpoint=str(Path(args.checkpoint).resolve()),
                  source_checkpoint_sha256=file_digest(args.checkpoint), source_config=state["config"],
                  source_bank=str(Path(args.source_bank).resolve()),
                  excluded_trace_seeds=sorted(excluded), source_actor_sha256=before,
                  seed=args.seed, scenes=scenes)
    (output / "collection_report.json").write_text(json.dumps(dict(
        requested=args.samples, captured=len(samples), scenes=scenes, split=args.split,
        actors_unchanged=True, source_checkpoint=result["source_checkpoint"]), indent=2)+"\n")
    if len(samples) != args.samples:
        raise RuntimeError("unique scene budget exhausted; report saved, no incomplete bank written")
    validate_handoff_bank(result, split=(args.split,))
    torch.save(result, output / "bank.pt")
    print(f"saved {output / 'bank.pt'}", flush=True)


if __name__ == "__main__":
    main()
