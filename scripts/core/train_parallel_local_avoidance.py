"""Parallel execution backend for train_local_avoidance.py (same local SAC task)."""
from __future__ import annotations

from collections import deque
import csv
import json
import time

import numpy as np
import torch

from rl_risk_sac.envs.parallel_local_avoidance_env import ParallelLocalAvoidanceEnvPool
from rl_risk_sac.tasks.local_avoidance import local_protocol, actor_digest, contract_digest
from rl_risk_sac.utils.training_logs import CsvLog, detailed_log_enabled, format_training_progress
from scripts.core.train_thesis_homotopy import accrue_update_credit, physical_cpu_ids


def run_parallel_local(*, args, config, agent, replay, bank, bank_ids, bank_fingerprint,
                       rng, counters, loaded, num_envs, output, checkpoints,
                       reaching_initialization, frozen_digest, batch_size, update_after, warmup,
                       schedule, migrations):
    """CPU simulation overlaps centralized learning; budgets count all workers together.

    Only updates earned by the preceding batch are overlapped with the next
    simulator step. Every log/checkpoint/final boundary drains those updates.
    Actor actions can consequently lag at most one batch (num_envs transitions).
    """
    progress_interval = int(config["train"].get("progress_interval", 1000))
    worker_seeds = [int(child.generate_state(1)[0]) for child in
                    np.random.SeedSequence(int(config["seed"])).spawn(num_envs)]
    cpu_ids = physical_cpu_ids(num_envs) if config["train"].get("cpu_affinity", False) else None
    pool = ParallelLocalAvoidanceEnvPool(config, worker_seeds, agent.reaching.actor, cpu_ids=cpu_ids)
    workers = [None] * num_envs
    observations = {}
    next_worker = int(loaded.get("parallel_next_worker", 0)) if loaded else 0
    updates_due = []
    last_update = {}
    recent = deque(loaded.get("recent_successes", []) if loaded else [], maxlen=100)
    start_steps = counters["steps"]
    budget_end = start_steps + args.steps
    fields = ("steps", "episode", "worker", "sample_id", "length", "return", "local_success",
              "local_failure", "timeout", "local_progress", "min_external_mesh_distance_m",
              "min_self_mesh_distance_m")
    progress_fields = ("steps", "episodes", "updates", "update_credit", "replay_size", "elapsed_s", "steps_per_second",
                       "remaining_steps", "eta_seconds", "recent_episodes", "recent_success_rate",
                       "critic_loss", *agent.avoidance.last_actor_metrics,
                       "alpha", "q1_mean", "q2_mean", "target_mean", "critic_gradient_norm",
                       "actor_gradient_norm", "actor_updated")
    try:
        policies = pool.policy_info()
        if any(p["actor_sha256"] != frozen_digest or p["trainable"] or p["cuda_initialized"]
               or p["device"] != "cpu" for p in policies.values()):
            raise RuntimeError("local workers must use unchanged, frozen, CPU-only reaching actors")
        if loaded:
            if "parallel_workers" in loaded:
                saved_workers = loaded["parallel_workers"]
                if len(saved_workers) != num_envs:
                    raise ValueError("incomplete parallel worker checkpoint")
                states = {}
                for index, state in enumerate(saved_workers):
                    if state is not None:
                        workers[index] = {key: value for key, value in state.items() if key != "environment"}
                        states[index] = state["environment"]
                observations.update(pool.restore_many(states))
            else:
                # Preserve the active serial episode and all its accumulators.
                state = loaded["environment"]
                workers[0] = dict(active=not state["finished"],
                                  episode_return=loaded["episode_return"],
                                  episode_steps=loaded["episode_steps"],
                                  episode_external=loaded["episode_external"],
                                  episode_self=loaded["episode_self"])
                observations.update(pool.restore_many({0: state}))
                print(f"migrated serial checkpoint: step={start_steps}, envs={num_envs}", flush=True)
        started = time.perf_counter()
        print(format_training_progress(start_steps, budget_end, 0., recent, label="并行越障训练",
                                       completed_steps=0) + f" | 环境={num_envs}", flush=True)
        with (output / "episodes.csv").open("w", newline="") as handle, \
                CsvLog(output / "updates.csv", enabled=detailed_log_enabled(config, "updates")) as updates_log, \
                CsvLog(output / "progress.csv", progress_fields) as progress_log:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            progress_log.writeheader()

            def perform_deferred_updates():
                nonlocal last_update
                for earned_step in updates_due:
                    batch = replay.sample(batch_size)
                    next_update = counters["updates"] + 1
                    collect = next_update == 1 or next_update % schedule["diagnostic_interval"] == 0
                    metrics = agent.update(batch, reference_observations=batch,
                                           collect_diagnostics=collect, check_finite=True)
                    counters["updates"] += 1
                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise RuntimeError("nonfinite local SAC update")
                    if collect:
                        last_update = metrics
                        updates_log.writerow({"steps": earned_step, **metrics})
                updates_due.clear()

            def save():
                if updates_due:
                    raise RuntimeError("cannot checkpoint with deferred SAC updates")
                if actor_digest(agent.reaching.actor) != frozen_digest:
                    raise RuntimeError("frozen reaching actor changed during avoidance training")
                states = pool.states([i for i, worker in enumerate(workers) if worker is not None])
                snapshots = [None if worker is None else {**worker, "environment": states[i]}
                             for i, worker in enumerate(workers)]
                torch.save(dict(
                    protocol=local_protocol(config), config=config, contract_sha256=contract_digest(config),
                    agent=agent.state_dict(), replay=replay.state_dict(), counters=dict(counters),
                    rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    parallel_num_envs=num_envs, parallel_workers=snapshots,
                    parallel_next_worker=next_worker, parallel_updates_due=0,
                    bank_fingerprint=bank_fingerprint, train_sample_ids=bank_ids,
                    train_trace_seeds=sorted({s["provenance"]["seed"] for s in bank["samples"]}),
                    reaching_initialization=reaching_initialization, validation_mode=args.validation_mode,
                    recent_successes=list(recent),
                    training_schedule=schedule, training_schedule_migrations=migrations,
                ), checkpoints / f"step_{counters['steps']:07d}.pt")
                agent.avoidance.save_actor(checkpoints / f"avoidance_actor_step_{counters['steps']:07d}.pt")

            while counters["steps"] < budget_end:
                # Partial batches stop exactly at log/save/budget boundaries.
                boundary = min(budget_end,
                               (counters["steps"] // progress_interval + 1) * progress_interval,
                               (counters["steps"] // args.save_interval + 1) * args.save_interval)
                count = min(num_envs, boundary - counters["steps"])
                indices = [(next_worker + offset) % num_envs for offset in range(count)]
                next_worker = (next_worker + count) % num_envs
                resets = {i: bank["samples"][int(rng.integers(len(bank["samples"])))]
                          for i in indices if workers[i] is None or not workers[i]["active"]}
                for i, (observation, _) in pool.reset_many(resets).items():
                    observations[i] = observation
                    workers[i] = dict(active=True, episode_return=0., episode_steps=0,
                                      episode_external=float("inf"), episode_self=float("inf"))
                actions = {}
                policy_indices = []
                for offset, i in enumerate(indices):
                    if counters["steps"] + offset < warmup:
                        actions[i] = rng.uniform(-1., 1., 6).astype(np.float32)
                    else:
                        policy_indices.append(i)
                if policy_indices:
                    policy_actions = agent.select_actions([observations[i] for i in policy_indices])
                    actions.update(zip(policy_indices, policy_actions))
                # Keep result/replay ordering stable even across the warmup boundary.
                pool.send_steps({i: actions[i] for i in indices})
                perform_deferred_updates()
                results = pool.recv_steps()
                for i in indices:
                    following, reward, terminated, truncated, info = results[i]
                    replay.add(observations[i], actions[i], reward, following, terminated or truncated)
                    observations[i] = following
                    counters["steps"] += 1
                    worker = workers[i]
                    worker["episode_steps"] += 1
                    worker["episode_return"] += reward
                    worker["episode_external"] = min(worker["episode_external"], info["min_external_mesh_distance_m"])
                    worker["episode_self"] = min(worker["episode_self"], info["min_self_mesh_distance_m"])
                    if replay.size >= max(batch_size, update_after):
                        due = accrue_update_credit(counters, schedule["updates_per_transition"])
                        updates_due.extend([counters["steps"]] * due)
                    if terminated or truncated:
                        worker["active"] = False
                        counters["episodes"] += 1
                        recent.append(bool(info["local_success"]))
                        writer.writerow(dict(steps=counters["steps"], episode=counters["episodes"], worker=i,
                                             sample_id=info["sample_id"], length=worker["episode_steps"],
                                             **{"return": worker["episode_return"]},
                                             **{key: info[key] for key in ("local_success", "local_failure", "timeout", "local_progress")},
                                             min_external_mesh_distance_m=worker["episode_external"],
                                             min_self_mesh_distance_m=worker["episode_self"]))
                        handle.flush()
                progress_due = counters["steps"] % progress_interval == 0 or counters["steps"] == budget_end
                save_due = counters["steps"] % args.save_interval == 0 or counters["steps"] == budget_end
                if progress_due or save_due:
                    perform_deferred_updates()
                if progress_due:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    speed = (counters["steps"] - start_steps) / elapsed
                    remaining = budget_end - counters["steps"]
                    progress_log.writerow(dict(**counters, replay_size=replay.size, elapsed_s=elapsed,
                                               steps_per_second=speed, remaining_steps=remaining,
                                               eta_seconds=remaining / speed, recent_episodes=len(recent),
                                               recent_success_rate=float(np.mean(recent)) if recent else "",
                                               **last_update))
                    progress_log.flush()
                    updates_log.flush()
                    print(format_training_progress(counters["steps"], budget_end, elapsed, recent,
                                                   label="并行越障训练", completed_steps=counters["steps"] - start_steps)
                          + f" | 环境={num_envs} 回合={counters['episodes']} 更新={counters['updates']}", flush=True)
                if save_due:
                    save()
                    print(f"saved local checkpoint step={counters['steps']} envs={num_envs}", flush=True)
    finally:
        pool.close()
    (output / "summary.json").write_text(json.dumps({**counters, "replay_size": replay.size,
                "frozen_reaching_actor_unchanged": actor_digest(agent.reaching.actor) == frozen_digest,
                "validation_mode": args.validation_mode, "num_envs": num_envs,
                "training_schedule": schedule}, indent=2) + "\n")
    print(f"saved {output}", flush=True)
