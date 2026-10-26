#!/usr/bin/env python3
"""Generate fresh training-only S1 demonstrations with physical admission.

No policy updates. Never read fixed evaluation scenes or failure snapshots.
The planner is a data generator only, never an inference-time controller.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import copy
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pybullet as p
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from rl_risk_sac.algorithms.homotopy_replay import HomotopyReplayBuffer, _Partition
from rl_risk_sac.algorithms.obstacle_encoder import SetGaussianActor
from rl_risk_sac.algorithms.s1_expert_replay import (
    candidate_rejection, contract_sha256, expert_contract, file_sha256, validate_expert_bank,
)
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv, StaticSceneSamplingError, ResetSamplingError
from rl_risk_sac.tasks.obstacle_observation import ObstacleObservation, obstacle_set, observation_tensors, stack_observations
from rl_risk_sac.utils.config import load_config
from scripts.core.train_thesis_homotopy import apply_stage_profile, checkpoint_actor_state


class PhysicalMargins:
    """Read-only full-mesh checks at the start and every executed 240 Hz step."""
    def __init__(self, env):
        self.env = env
        self.external = self.self_distance = float("inf")
        self.checks = 0
        self()

    def __call__(self):
        env = self.env
        self.external = min(self.external, *(env._static_clearance(o) for o in env.obstacles))
        for first, second in env._self_pair_link_ids:
            points = p.getClosestPoints(env.robot_id, env.robot_id, env.self_query_distance,
                                        linkIndexA=first, linkIndexB=second, physicsClientId=env.client_id)
            self.self_distance = min(self.self_distance, min((float(q[8]) for q in points),
                                                            default=env.self_query_distance))
        self.checks += 1

    @property
    def passed(self):
        return self.external >= self.env.d_safe and self.self_distance >= self.env.d_self_safe


def worker_init(config, reference_actor, gains, min_advantage):
    global ENV, ACTOR, GAINS, MIN_ADVANTAGE
    torch.set_num_threads(1)
    ENV = ThesisHomotopyEnv(copy.deepcopy(config))
    ACTOR = SetGaussianActor(241, 6, config["sac"]["hidden_dims"], np.ones(37, np.float32))
    ACTOR.load_state_dict(checkpoint_actor_state(Path(reference_actor), torch.device("cpu")))
    ACTOR.eval()
    GAINS, MIN_ADVANTAGE = gains, min_advantage


def actor_action(observation):
    obs, features, mask = observation_tensors(stack_observations([observation]), torch.device("cpu"))
    with torch.no_grad():
        return ACTOR.deterministic(obs, features, mask).numpy()[0]


def rollout(initial, level, episode_id, *, gain=None, wait=False):
    observation = ENV.restore_episode_state(initial)
    path = np.asarray(initial["obstacle_sampling"]["witness_waypoints"])
    target = 1
    transitions = []
    discounted = self_penalty = 0.
    margins = PhysicalMargins(ENV) if gain is not None else None
    cause = "horizon_exhausted"
    for step in range(ENV.horizon):
        if gain is not None:
            q, _ = ENV.robot.joint_state_exact()
            while target < len(path) - 1 and np.max(np.abs(q - path[target])) < .025:
                target += 1
            action = np.clip(gain * (path[target] - q) / ENV.action_scale, -1, 1).astype(np.float32)
        elif wait:
            action = np.zeros(6, np.float32)
        else:
            action = actor_action(observation)
        next_observation, reward, _, terminated, truncated, info = ENV.step(action, substep_observer=margins)
        discount = float(ENV.config["sac"]["gamma"]) ** step
        discounted += discount * reward
        self_penalty += discount * info["self_safety_penalty"]
        if gain is not None:
            transitions.append((
                ObstacleObservation(observation, obstacle_set(observation)), action.copy(),
                ObstacleObservation(next_observation, obstacle_set(next_observation)),
                bool(terminated or truncated),
                HomotopyReplayBuffer.transition_features(info, curriculum_level=level,
                                                         reward_parameters=ENV.reward_parameters),
            ))
        observation = next_observation
        if terminated or truncated:
            cause = ("success" if info["safe_success"] else "timeout" if truncated
                     else "collision" if info["collision"] else "joint_limit")
            break
        if margins is not None and not margins.passed:
            cause = "safety_margin"
            break
    result = dict(cause=cause, steps=step + 1, gain=gain,
                  discounted_return=float(discounted), discounted_self_penalty=float(self_penalty),
                  physical_substep_margin_passed=bool(margins is not None and margins.passed),
                  min_external_mesh_distance_m=None if margins is None else margins.external,
                  min_self_mesh_distance_m=None if margins is None else margins.self_distance,
                  physics_substep_checks=0 if margins is None else margins.checks,
                  final_position_error_m=float(info["next_rho_position"]),
                  final_orientation_error_rad=float(info["next_rho_orientation"]))
    part = None
    if gain is not None and cause == "success" and margins.passed:
        part = _Partition(241, 6, len(transitions))
        for index, (obs, action, nxt, done, raw) in enumerate(transitions):
            part.add(obs, action, nxt, done, raw, episode_id, index + 1)
    return result, part


def generate_scene(task):
    level, index, root_seed, max_scenes = task
    # Disjoint from the low-numbered fixed eval seeds used by this project.
    scene_seed = int(np.random.SeedSequence([root_seed, level, index]).generate_state(1)[0]) | 0x80000000
    episode_id = level * max_scenes + index + 1
    settings = ENV.config["thesis"]["joint_pose_curriculum"]["levels"][level]
    ENV.configure_episode(
        "none", xi=1., strict=True, goal_scale=settings["goal_scale"],
        orientation_scale=ENV.config["thesis"]["orientation_curriculum"]["levels"][level],
        position_tolerance=settings["position_tolerance_m"],
        orientation_tolerance=settings["orientation_tolerance_rad"], lambda_self=1.,
        **{key: settings[key] for key in ("target_distance_min_m", "target_distance_max_m",
                                         "target_orientation_min_rad", "target_orientation_max_rad")})
    try:
        ENV.reset(seed=scene_seed)
    except (StaticSceneSamplingError, ResetSamplingError) as error:
        return dict(level=level, seed=scene_seed, episode_id=episode_id,
                    rejected="sampling_budget", error=str(error)), None, None
    initial = ENV.episode_state_dict()
    if len(initial["obstacles"]) != 1:
        raise ValueError("expert generator requires exactly one obstacle per scene")
    actor, _ = rollout(initial, level, episode_id)
    waiting, _ = rollout(initial, level, episode_id, wait=True)
    candidates, admitted = [], []
    for gain in GAINS:
        candidate, part = rollout(initial, level, episode_id, gain=gain)
        rejection = candidate_rejection(candidate, [actor, waiting], MIN_ADVANTAGE)
        candidate["rejected"] = rejection
        candidates.append(candidate)
        if rejection is None:
            admitted.append((candidate, part))
    summary = dict(level=level, seed=scene_seed, episode_id=episode_id,
                   band=initial["obstacle_sampling"]["path_band"],
                   fresh_training_scene=True, reference=actor, waiting=waiting,
                   candidates=candidates, min_advantage=MIN_ADVANTAGE)
    if not admitted:
        summary["rejected"] = "no_qualified_candidate"
        return summary, None, None
    best, part = max(admitted, key=lambda item: item[0]["discounted_return"] + .8 * item[0]["discounted_self_penalty"])
    summary.update(best)
    return summary, part, initial


def combine_parts(parts):
    combined = _Partition(241, 6, sum(part.size for part in parts))
    for part in parts:
        for i in part.chronological_indices():
            combined.add(part.observation(i), part.action[i], part.observation(i, next_state=True),
                         part.done[i, 0], part.raw[i], part.episode[i], part.step[i])
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/paper_full.yaml")
    parser.add_argument("--reference-actor", required=True, help="frozen current 241-input S1 Actor")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[0])
    parser.add_argument("--episodes-per-level", type=int, default=128)
    parser.add_argument("--max-scenes-per-level", type=int, default=512)
    parser.add_argument("--gains", type=float, nargs="+", default=[3., 5., 8.])
    parser.add_argument("--min-return-advantage", type=float, default=2.)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if (args.episodes_per_level < 1 or args.max_scenes_per_level < args.episodes_per_level
            or args.workers < 1 or not np.isfinite(args.min_return_advantage)
            or args.min_return_advantage < 0 or not all(np.isfinite(g) and g > 0 for g in args.gains)
            or len(set(args.levels)) != len(args.levels)):
        parser.error("invalid generation quotas, gains, workers or return advantage")
    config = load_config(ROOT / args.config)
    apply_stage_profile(config, "s1")
    contract = expert_contract(config)
    if any(not 0 <= level < len(contract["levels"]) for level in args.levels):
        parser.error("precision level out of range")
    # Proposal distribution only: no zero-obstacle demonstrations; online mix is untouched.
    probabilities = np.asarray(config["thesis"]["obstacles"]["path_sampling"]["probabilities"], float)
    probabilities[0] = 0.
    probabilities /= probabilities.sum()
    config["thesis"]["obstacles"]["path_sampling"]["probabilities"] = probabilities.tolist()
    config["thesis"]["obstacles"]["count_probabilities"] = [0., 1., 0., 0.]
    # Fail in the parent for invalid assets/actors instead of respawning an
    # initializer that can never succeed.
    torch.set_num_threads(1)
    probe_env = ThesisHomotopyEnv(copy.deepcopy(config))
    reward_parameters = dict(probe_env.reward_parameters)
    probe_env.close()
    probe_actor = SetGaussianActor(241, 6, config["sac"]["hidden_dims"], np.ones(37, np.float32))
    probe_actor.load_state_dict(checkpoint_actor_state(Path(args.reference_actor), torch.device("cpu")))
    del probe_actor
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reports, accepted, initial_states, partitions = [], [], {}, {}
    with mp.get_context("spawn").Pool(args.workers, initializer=worker_init,
                                       initargs=(config, args.reference_actor, args.gains,
                                                 args.min_return_advantage)) as pool:
        for level in args.levels:
            parts = []
            pending = deque()
            next_index = 0

            def submit_one():
                nonlocal next_index
                pending.append(pool.apply_async(generate_scene, ((level, next_index, args.seed,
                                                                 args.max_scenes_per_level),)))
                next_index += 1

            for _ in range(min(args.workers, args.max_scenes_per_level)):
                submit_one()
            # Ordered bounded dispatch: at most workers-1 unused scenes, not
            # hundreds queued behind an already satisfied level quota.
            while pending:
                summary, part, initial = pending.popleft().get()
                reports.append(summary)
                if part is not None:
                    accepted.append(summary)
                    parts.append(part)
                    initial_states[summary["episode_id"]] = initial
                print(f"EXPERT L{level} proposal={len([r for r in reports if r['level']==level])} "
                      f"accepted={len(parts)}/{args.episodes_per_level} seed={summary['seed']} "
                      f"status={summary.get('rejected') or 'accepted'}", flush=True)
                if len(parts) >= args.episodes_per_level:
                    break
                if next_index < args.max_scenes_per_level:
                    submit_one()
            if level != args.levels[-1]:
                for future in pending:
                    future.get()
            if parts:
                partitions[level] = combine_parts(parts)
            if len(parts) < args.episodes_per_level:
                break
    complete = all(len([r for r in accepted if r["level"] == level]) >= args.episodes_per_level
                   for level in args.levels)
    bank = dict(format_version=1, stage="s1", complete=complete,
                contract=contract, contract_sha256=contract_sha256(contract),
                reference_actor=str(Path(args.reference_actor).resolve()),
                reference_actor_sha256=file_sha256(args.reference_actor),
                generator_seed=args.seed, levels=args.levels,
                episodes_per_level=args.episodes_per_level,
                min_return_advantage=args.min_return_advantage,
                reward_parameters=reward_parameters,
                partitions=partitions, episodes=accepted, initial_states=initial_states,
                safety_method="full mesh initial and every executed physical substep; not continuous-time certification")
    report = dict(complete=complete, config=args.config, reference_actor=bank["reference_actor"],
                  reference_actor_sha256=bank["reference_actor_sha256"], generator_seed=args.seed,
                  requested_episodes_per_level=args.episodes_per_level, levels=args.levels,
                  accepted=len(accepted), transitions=sum(part.size for part in partitions.values()),
                  accepted_bands=dict(Counter(r["band"] for r in accepted)), scenes=reports,
                  contract_sha256=bank["contract_sha256"], safety_method=bank["safety_method"])
    (args.output_dir / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not complete:
        raise RuntimeError("expert generation budget exhausted; report saved, no trainable bank produced")
    validate_expert_bank(bank, contract)
    torch.save(bank, args.output_dir / "bank.pt")
    print(f"DONE accepted={len(accepted)} transitions={report['transitions']} bank={args.output_dir/'bank.pt'}", flush=True)


if __name__ == "__main__":
    main()
