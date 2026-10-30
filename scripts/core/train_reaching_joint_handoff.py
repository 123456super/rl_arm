#!/usr/bin/env python3
"""Fine-tune obstacle-aware reaching under the real frozen-avoidance controller."""
from __future__ import annotations

import argparse
from collections import deque
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch

from rl_risk_sac.algorithms.dual_sac import ReachingSACAgent, checkpoint_policy_state
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_joint_handoff_env import ParallelJointHandoffEnvPool
from rl_risk_sac.tasks.local_avoidance import LOCAL_OBSERVATION_DIM, actor_digest, geometric_mode, local_protocol
from rl_risk_sac.tasks.local_samples import nominal_episode_contract
from rl_risk_sac.tasks.reaching_handoff import (
    HANDOFF_TRAIN_PROTOCOLS, OBSTACLE_AWARE_TRAIN_PROTOCOL, ObstacleAwareJointHandoffReplay, ReachingDecisionReturn,
    bank_fingerprint, file_digest, joint_handoff_state, obstacle_aware_reaching_config, validate_handoff_bank,
)
from rl_risk_sac.utils.training_logs import CsvLog, collect_sac_diagnostics, format_training_progress
from scripts.core.train_reaching_handoff import progress_fieldnames
from scripts.core.train_thesis_homotopy import accrue_update_credit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True, help="original complete two-SAC checkpoint")
    parser.add_argument("--bank", required=True, help="real training handoffs, with original obstacles")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default="outputs/reaching_handoff")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--initialize-reaching-from", help="handoff checkpoint weights; starts fresh replay")
    initialization.add_argument("--resume", help="complete real-obstacle handoff checkpoint, including replay")
    parser.add_argument("--steps", type=int, default=500000, help="additional live control steps, across both policies and workers")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=50000)
    parser.add_argument("--handoff-fraction", type=float, default=.5, help="fraction of episode starts")
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-5)
    parser.add_argument("--alpha-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=27104)
    parser.add_argument("--device", choices=("cpu", "cuda:auto"), default="cuda:auto")
    parser.add_argument("--validation-mode", action="store_true")
    args = parser.parse_args()
    if (min(args.steps, args.num_envs, args.max_episode_steps, args.save_interval) < 1
            or args.seed < 0 or not 0 < args.handoff_fraction < 1
            or min(args.actor_lr, args.critic_lr, args.alpha_lr) <= 0):
        parser.error("invalid budgets, mixture or learning rates")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    source_sha = file_digest(args.source_checkpoint)
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    if (source.get("protocol") != local_protocol(source["config"])
            or not geometric_mode(source["config"])
            or (source["validation_mode"] and not args.validation_mode)):
        parser.error("requires a matching geometric two-SAC checkpoint; formal runs require formal weights")
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    validate_handoff_bank(bank, split=("train",), source_sha=source_sha)
    config = obstacle_aware_reaching_config(source["config"], actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr)
    config["device"] = args.device
    # Retain the 241D schema, now accompanied by raw variable-size obstacle sets.
    agent = ReachingSACAgent(241, 6, config)
    avoidance_config = copy.deepcopy(source["config"])
    avoidance_config["device"] = "cpu"
    avoidance = ThesisSACAgent(LOCAL_OBSERVATION_DIM, 6, avoidance_config)
    avoidance.load_state_dict(checkpoint_policy_state(source, "avoidance"), actor_only=True)
    avoidance.actor.eval().requires_grad_(False)
    frozen_sha = actor_digest(avoidance.actor)
    if frozen_sha != bank["source_actor_sha256"]["avoidance"]:
        raise ValueError("handoff bank avoidance actor differs from source")
    settings = dict(num_envs=args.num_envs, max_episode_steps=args.max_episode_steps,
                    handoff_fraction=args.handoff_fraction, actor_lr=args.actor_lr,
                    critic_lr=args.critic_lr, alpha_lr=args.alpha_lr, seed=args.seed,
                    validation_mode=args.validation_mode, reaching_obstacle_aware=True)
    replay = ObstacleAwareJointHandoffReplay(512 if args.validation_mode else 50000, agent.device, args.seed + 1)
    batch_size = 8 if args.validation_mode else 1024
    rng = np.random.default_rng(args.seed)
    counters = dict(steps=0, episodes=0, updates=0, update_credit=0., reaching_steps=0, avoidance_steps=0,
                    reaching_transitions=0, task_decisions=0, adjusted_decisions=0)
    recent = {name: deque(maxlen=100) for name in ("all", "original", "handoff")}
    handoff_stats = {name: deque(maxlen=100) for name in ("collision", "timeout", "reentries")}
    workers, pending = [None] * args.num_envs, [None] * args.num_envs
    observations, normal_seeds, inherited_seeds = {}, set(), set()
    cursor, loaded = 0, None
    if args.resume:
        loaded = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (loaded.get("protocol") != OBSTACLE_AWARE_TRAIN_PROTOCOL
                or loaded["source_checkpoint_sha256"] != source_sha or loaded["settings"] != settings
                or loaded["bank_fingerprint"] != bank_fingerprint(bank)
                or loaded["frozen_avoidance_actor_sha256"] != frozen_sha):
            raise ValueError("resume protocol/source/bank/settings differ; old obstacle-blind replay is incompatible")
        agent.load_state_dict(loaded["agent"])
        if not agent.obstacle_aware:
            raise ValueError("obstacle-aware resume is missing the reaching input contract")
        replay.load_state_dict(loaded["replay"])
        counters.update(loaded["counters"])
        workers, pending, cursor = loaded["workers"], loaded["pending_decisions"], loaded["worker_cursor"]
        normal_seeds.update(loaded["normal_reset_seeds"])
        inherited_seeds.update(loaded["training_trace_seeds"])
        rng.bit_generator.state = loaded["rng"]
        torch.set_rng_state(loaded["torch_rng"].cpu())
        if torch.cuda.is_available() and loaded["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in loaded["cuda_rng"]])
        for name, values in loaded["recent"].items():
            recent[name].extend(values)
        for name, values in loaded["handoff_stats"].items():
            handoff_stats[name].extend(values)
    elif args.initialize_reaching_from:
        initial = torch.load(args.initialize_reaching_from, map_location="cpu", weights_only=False)
        if (initial.get("protocol") not in HANDOFF_TRAIN_PROTOCOLS
                or initial["source_checkpoint_sha256"] != source_sha
                or initial["frozen_avoidance_actor_sha256"] != frozen_sha
                or (initial["settings"]["validation_mode"] and not args.validation_mode)):
            raise ValueError("initial reaching weights must belong to this source and frozen avoidance actor")
        agent.load_state_dict(initial["agent"])
        if initial["protocol"] == OBSTACLE_AWARE_TRAIN_PROTOCOL and not agent.obstacle_aware:
            raise ValueError("obstacle-aware initialization is missing the reaching input contract")
        inherited_seeds.update(initial["training_trace_seeds"])
        agent.reset_chain_reference_actor()
    else:
        agent.load_state_dict(checkpoint_policy_state(source, "reaching"))
        agent.reset_chain_reference_actor()
    protected = set(bank["excluded_trace_seeds"])
    handoff_seeds = {s["source_seed"] for s in bank["samples"]}
    if inherited_seeds.intersection(protected):
        raise ValueError("initial or resumed reaching policy has trained on protected scenes")
    initialized_encoders = agent.initialize_obstacle_awareness(avoidance.actor.obstacle_encoder.scale)
    for optimizer, rate in ((agent.actor_optimizer, args.actor_lr), (agent.q_optimizer, args.critic_lr),
                            (agent.alpha_optimizer, args.alpha_lr)):
        for group in optimizer.param_groups:
            group["lr"] = rate
    output = Path(args.output_dir) / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    (output / "manifest.json").write_text(json.dumps(dict(
        protocol=OBSTACLE_AWARE_TRAIN_PROTOCOL, config=config, settings=settings,
        source_checkpoint=str(Path(args.source_checkpoint).resolve()), source_checkpoint_sha256=source_sha,
        bank=str(Path(args.bank).resolve()), bank_fingerprint=bank_fingerprint(bank),
        frozen_avoidance_actor_sha256=frozen_sha, obstacles_removed=False,
        reaching_obstacle_aware=True, newly_initialized_obstacle_encoders=initialized_encoders,
        obstacle_encoder_initialization="zero final projection; original reaching outputs preserved at transfer",
        initialize_reaching_from=args.initialize_reaching_from, resume=args.resume,
        live_control_step_budget=True, replay="task proposals and actual shielded outcomes, with discounted frozen-avoidance continuations",
        terminal_time_limits=True, reward="existing live S1 reaching/safety reward; no local-pass bonus"), indent=2) + "\n")
    pool = ParallelJointHandoffEnvPool(source["config"], list(range(args.seed + 10, args.seed + 10 + args.num_envs)),
                                     agent.actor, avoidance.actor, max_steps=args.max_episode_steps)
    start, end = counters["steps"], counters["steps"] + args.steps
    started, last_metrics, due = time.perf_counter(), {}, []

    def store(index, following, terminal):
        if pending[index] is not None:
            pending[index].store(replay, following, terminal, agent.gamma)
            pending[index] = None
            counters["reaching_transitions"] += 1
            if replay.size >= batch_size:
                due.extend([None] * accrue_update_credit(counters, .25))

    def update():
        nonlocal last_metrics
        for _ in due:
            batch = replay.sample(batch_size)
            counters["updates"] += 1
            collect = collect_sac_diagnostics(agent, counters["updates"])
            metrics = agent.update(batch, reference_observations=batch, collect_diagnostics=collect, check_finite=True)
            if not all(np.isfinite(value) for value in metrics.values()):
                raise RuntimeError("nonfinite joint handoff update")
            if collect:
                last_metrics = metrics
        due.clear()

    try:
        if loaded:
            observations.update(pool.restore_many(loaded["environment_states"]))
        episode_fields = ("steps", "worker", "kind", "length", "success", "collision", "joint_limit", "status",
                          "reentries", "verified_passages", "position_error_m", "orientation_error_rad")
        fields = (*progress_fieldnames(agent), "reaching_steps", "avoidance_steps", "reaching_transitions",
                  "task_decisions", "adjusted_decisions",
                  "handoff_collision_rate", "handoff_timeout_rate", "handoff_reentries_mean")
        with CsvLog(output / "episodes.csv", episode_fields) as episodes, CsvLog(output / "progress.csv", fields) as progress:
            def finish(index, info, status):
                worker = workers[index]
                worker["active"] = False
                counters["episodes"] += 1
                success = status == "success"
                recent["all"].append(success)
                recent[worker["kind"]].append(success)
                if worker["kind"] == "handoff":
                    handoff_stats["collision"].append(bool(info.get("collision", False)))
                    handoff_stats["timeout"].append(status in ("local_timeout", "global_timeout"))
                    handoff_stats["reentries"].append(worker["attempts"])
                episodes.writerow(dict(steps=counters["steps"], worker=index, kind=worker["kind"], length=worker["length"],
                    success=success, collision=info.get("collision", False), joint_limit=info.get("joint_limit", False),
                    status=status, reentries=worker["attempts"], verified_passages=worker["passages"],
                    position_error_m=info.get("next_rho_position", ""), orientation_error_rad=info.get("next_rho_orientation", "")))

            while counters["steps"] < end:
                boundary = min(end, (counters["steps"] // 1000 + 1) * 1000,
                               (counters["steps"] // args.save_interval + 1) * args.save_interval)
                count = min(args.num_envs, boundary - counters["steps"])
                indices = [(cursor + i) % args.num_envs for i in range(count)]
                cursor = (cursor + count) % args.num_envs
                normal, seeds, handoffs = {}, {}, {}
                for i in indices:
                    if workers[i] is None or not workers[i]["active"]:
                        if rng.random() < args.handoff_fraction:
                            sample = bank["samples"][int(rng.integers(len(bank["samples"])))]
                            handoffs[i] = joint_handoff_state(sample["state"])
                            kind = "handoff"
                        else:
                            seed = int(rng.integers(2**32))
                            while seed in protected or seed in handoff_seeds or seed in inherited_seeds:
                                seed = int(rng.integers(2**32))
                            normal_seeds.add(seed)
                            normal[i], seeds[i] = nominal_episode_contract(source["config"]), seed
                            kind = "original"
                        workers[i] = dict(active=True, kind=kind, length=0, attempts=0, passages=0)
                for i, (observation, _) in pool.reset_many(normal, seeds=seeds).items():
                    observations[i] = observation
                observations.update(pool.restore_many(handoffs))
                pool.sync_reaching(agent.actor)
                proposed = dict(zip(indices, agent.select_actions([observations[i] for i in indices])))
                prepared = pool.prepare_many(indices, reaching_actions=proposed)
                actions = {}
                for i in indices:
                    decision = prepared[i]
                    if decision["failure"]:
                        store(i, observations[i], True)
                        finish(i, {}, decision["failure"])
                    else:
                        if decision["task_decision"]:
                            # Learn the task's proposed action even when rejected.
                            # Its outcome includes execution by the safety controller,
                            # never relabel the backup action as a task action.
                            store(i, observations[i], False)
                            pending[i] = ReachingDecisionReturn(observations[i], proposed[i])
                            counters["task_decisions"] += 1
                            counters["adjusted_decisions"] += int(decision["role"] == "avoidance")
                        actions[i] = proposed[i] if decision["role"] == "reaching" else None
                if actions:
                    pool.send_steps(actions)
                    update()
                    results = pool.recv_steps()
                    for i, (following, reward, _, terminated, truncated, info) in results.items():
                        counters["steps"] += 1
                        counters[f"{info['action_policy']}_steps"] += 1
                        worker = workers[i]
                        worker["length"] += 1
                        worker["attempts"], worker["passages"] = info["avoidance_attempts"], info["verified_passages"]
                        if pending[i] is not None:
                            pending[i].advance(reward, agent.gamma)
                        observations[i] = following
                        if terminated or truncated:
                            # These are complete bounded joint tasks. No Q of a
                            # reaching action is evaluated at an avoidance-only endpoint.
                            store(i, following, True)
                            status = ("collision" if info["collision"] else "joint_limit" if info["joint_limit"] else
                                      "success" if info["safe_success"] else "local_timeout" if info["local_timeout"] else
                                      "global_timeout" if truncated else "terminated")
                            finish(i, info, status)
                else:
                    update()
                log = counters["steps"] % 1000 == 0 or counters["steps"] == end
                save = counters["steps"] % args.save_interval == 0 or counters["steps"] == end
                if log or save:
                    update()
                if log:
                    rates = {f"{name}_success_rate": float(np.mean(values)) if values else "" for name, values in recent.items()}
                    rates.update({f"handoff_{name}_{'mean' if name == 'reentries' else 'rate'}":
                                  float(np.mean(values)) if values else "" for name, values in handoff_stats.items()})
                    elapsed = time.perf_counter() - started
                    progress.writerow(dict(**counters, replay_size=replay.size, elapsed_s=elapsed, **rates, **last_metrics))
                    progress.flush()
                    episodes.flush()
                    print(format_training_progress(counters["steps"], end, elapsed, recent["handoff"],
                          label="带障碍交接微调", completed_steps=counters["steps"] - start)
                          + f" | 原始={rates['original_success_rate']} 混合任务={rates['all_success_rate']}"
                          + f" 交接碰撞={rates['handoff_collision_rate']} 再接管均值={rates['handoff_reentries_mean']}"
                          + f" 累计决策介入率={counters['adjusted_decisions'] / max(counters['task_decisions'], 1):.1%}"
                          + f" | 到达步={counters['reaching_steps']} 越障步={counters['avoidance_steps']}", flush=True)
                if save:
                    pool.sync_reaching(agent.actor)
                    policies = pool.policy_info()
                    if any(row["avoidance"] != frozen_sha or row["reaching"] != actor_digest(agent.actor)
                           or not row["reaching_obstacle_aware"] or row["cuda_initialized"] for row in policies.values()):
                        raise RuntimeError("worker policy snapshot mismatch or unexpected CUDA initialization")
                    torch.save(dict(protocol=OBSTACLE_AWARE_TRAIN_PROTOCOL, agent=agent.state_dict(), config=config, settings=settings,
                        source_checkpoint=str(Path(args.source_checkpoint).resolve()), source_checkpoint_sha256=source_sha,
                        frozen_avoidance_actor_sha256=frozen_sha, bank_fingerprint=bank_fingerprint(bank),
                        training_trace_seeds=sorted(handoff_seeds | normal_seeds | inherited_seeds), normal_reset_seeds=sorted(normal_seeds),
                        handoff_sample_ids=sorted(s["sample_id"] for s in bank["samples"]), replay=replay.state_dict(),
                        counters=dict(counters), workers=copy.deepcopy(workers), pending_decisions=copy.deepcopy(pending), worker_cursor=cursor,
                        environment_states=pool.states([i for i, worker in enumerate(workers) if worker is not None]),
                        recent={k: list(v) for k, v in recent.items()}, handoff_stats={k: list(v) for k, v in handoff_stats.items()},
                        rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                        checkpoints / f"step_{counters['steps']:07d}.pt")
    finally:
        pool.close()
    (output / "summary.json").write_text(json.dumps(dict(**counters, frozen_avoidance_actor_sha256=frozen_sha,
                                                       obstacles_removed=False, reaching_obstacle_aware=True), indent=2) + "\n")


if __name__ == "__main__":
    main()
