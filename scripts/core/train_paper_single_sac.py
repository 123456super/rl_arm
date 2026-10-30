#!/usr/bin/env python3
"""Train one obstacle-reaching SAC with eight asynchronous PyBullet workers."""
from __future__ import annotations

import argparse
from collections import deque
import copy
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch

from rl_risk_sac.algorithms.paper_sac import PaperReplayBuffer, PaperSACAgent
from rl_risk_sac.envs.parallel_paper_env import ParallelPaperEnvPool
from rl_risk_sac.utils.paper_config import PAPER_PROTOCOL, experiment_signature, load_paper_config
from rl_risk_sac.utils.training_logs import CsvLog, collect_sac_diagnostics, format_training_progress

METRIC_FIELDS = ("critic_loss", "alpha", "q1_mean", "q2_mean", "target_mean", "critic_gradient_norm",
                 "actor_loss", "alpha_loss", "policy_entropy", "actor_gradient_norm", "actor_updated")
EPISODE_FIELDS = ("steps", "worker", "seed", "length", "episode_return", "reached", "safe_success",
                  "report_success", "final_report_success", "strict_success", "collision_seen", "joint_limit",
                  "joint_limit_seen", "position_error_m", "orientation_error_rad", "min_position_error_m",
                  "min_orientation_error_rad", "min_combined_error")
STEP_INFO_FIELDS = tuple(key for key in EPISODE_FIELDS if key not in ("steps", "worker", "seed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="independent paper_single_sac YAML")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--steps", type=int, help="additional aggregate environment transitions; resume defaults to remaining budget")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="cpu, cuda:auto, or cuda:N")
    parser.add_argument("--resume", help="complete step checkpoint, including replay and all worker states")
    ensemble = parser.add_mutually_exclusive_group()
    ensemble.add_argument("--action-ensemble", dest="action_ensemble", action="store_true", default=None,
                          help="enable the optional SAC-AEW ablation")
    ensemble.add_argument("--no-action-ensemble", dest="action_ensemble", action="store_false", default=None,
                          help="disable AE and use standard SAC (the default config)")
    parser.add_argument("--validation-mode", action="store_true", help="small networks/replay for integration checks only")
    args = parser.parse_args()
    loaded = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if loaded is not None and loaded.get("protocol") != PAPER_PROTOCOL:
        parser.error("resume requires paper_single_sac_v2; v1 actions/networks differ, so start a new v2 run")
    config = (load_paper_config(args.config) if args.config else copy.deepcopy(loaded["config"]) if loaded else
              load_paper_config(ROOT / "configs/experiments/paper_single_sac.yaml"))
    if loaded and bool(loaded["validation_mode"]) != args.validation_mode:
        parser.error("validation/formal checkpoint mode differs; validation weights cannot initialize a formal run")
    for key in ("device", "seed"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.num_envs is not None:
        config["train"]["num_envs"] = args.num_envs
    if args.action_ensemble is not None:
        config["sac"]["action_ensemble"]["enabled"] = args.action_ensemble
    if args.validation_mode and loaded is None:
        config["sac"].update(hidden_dims=[32, 32], batch_size=16, replay_capacity=512,
                             warmup_steps=16, update_after=16)
    num_envs = int(config["train"]["num_envs"])
    if num_envs < 1 or config["seed"] < 0 or (args.steps is not None and args.steps <= 0):
        parser.error("steps/worker count must be positive and seed nonnegative")
    signature = experiment_signature(config)
    if loaded and (loaded["signature"] != signature or loaded["num_envs"] != num_envs):
        parser.error("resume task/optimizer/AE schedule/seed or parallel worker count differs")
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    agent = PaperSACAgent(config)
    replay = PaperReplayBuffer(config["sac"]["replay_capacity"], agent.device, config["seed"] + 1)
    rng = np.random.default_rng(config["seed"] + 2)
    counters = dict(steps=0, episodes=0, updates=0, update_credit=0.)
    recent = {name: deque(maxlen=100) for name in ("reached", "safe_success", "collision_seen", "report_success",
                                                "final_report_success", "strict_success")}
    recent_errors = {name: deque(maxlen=100) for name in (
        "position_error_m", "orientation_error_rad", "min_position_error_m")}
    workers, observations = [None] * num_envs, {}
    cursor, elapsed_before, last_metrics = 0, 0., {}
    if loaded:
        agent.load_state_dict(loaded["agent"])
        replay.load_state_dict(loaded["replay"])
        counters.update(loaded["counters"])
        workers, cursor = copy.deepcopy(loaded["workers"]), loaded["worker_cursor"]
        rng.bit_generator.state = loaded["rng"]
        torch.set_rng_state(loaded["torch_rng"].cpu())
        if agent.device.type == "cuda" and loaded["cuda_rng"] is not None:
            torch.cuda.set_rng_state(loaded["cuda_rng"].cpu(), agent.device)
        for key, values in loaded["recent"].items():
            recent[key].extend(values)
        for key, values in loaded["recent_errors"].items():
            recent_errors[key].extend(values)
        elapsed_before = loaded["elapsed_s"]
        last_metrics = dict(loaded["last_metrics"])
    additional = args.steps if args.steps is not None else max(0, config["train"]["total_steps"] - counters["steps"])
    if additional <= 0:
        parser.error("checkpoint already reached the configured budget; specify --steps for extra transitions")
    output = Path(args.output_dir or config["train"]["output_dir"]) / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    (output / "manifest.json").write_text(json.dumps(dict(
        protocol=PAPER_PROTOCOL, signature=signature, config=config, resume=args.resume,
        validation_mode=args.validation_mode, num_envs=num_envs,
        additional_steps=additional, batched_learner_inference=True, overlap_learning_and_simulation=True,
        training_scene_seed_domain="[0, 2**31); evaluation reserves [2**31, 2**32)",
        action="q_target = clip(fixed initial q + action_limit * normalized action, URDF limits), position servo",
        joint_motor_force="URDF per-joint effort limits" if config["paper"]["joint_motor_force"] is None else config["paper"]["joint_motor_force"],
        success=dict(combined_error_tolerance=config["paper"]["success"]["combined_error_tolerance"],
                     strict_combined_error_tolerance=config["paper"]["success"]["strict_combined_error_tolerance"],
                     report_success="ever met reporting pose tolerances and no full-episode collision/limit violation",
                     final_report_success="met reporting pose tolerances at the final state and no episode violation"),
        entropy="conditional on AE count; exact averaged unsquashed Gaussian plus tanh Jacobian",
        source="FTP4RM paper and author code; UR5 asset, IK/endpoint filtering, task sampling choices documented",
        path_feasibility_verified=False, checkpoint_contains_replay=True), indent=2) + "\n")
    start, end, due = counters["steps"], counters["steps"] + additional, 0
    stopping, last_saved = False, -1

    def request_stop(signum, _):
        nonlocal stopping
        stopping = True
        print(f"收到停止信号 {signum}，完成当前并行批次后保存完整断点。", flush=True)

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    pool = None
    started = time.perf_counter()
    progress_fields = ("steps", "episodes", "updates", "actor_updates", "replay_size", "elapsed_s", "steps_per_second",
        *(f"{key}_rate" for key in recent), *(f"mean_{key}" for key in recent_errors), "ae_progress", *METRIC_FIELDS)

    def update():
        nonlocal due, last_metrics
        for _ in range(due):
            collect = collect_sac_diagnostics(agent, counters["updates"] + 1, config["train"]["diagnostics_interval"])
            metrics = agent.update(replay.sample(config["sac"]["batch_size"]), collect_diagnostics=collect)
            counters["updates"] += 1
            if collect:
                last_metrics = metrics
        due = 0

    try:
        pool = ParallelPaperEnvPool(config, list(range(config["seed"] + 10, config["seed"] + 10 + num_envs)),
                                    step_info_keys=STEP_INFO_FIELDS)
        if loaded:
            observations.update(pool.restore_many(loaded["environment_states"]))
        # Startup/URDF loading is excluded from live throughput/ETA.
        started = time.perf_counter()

        def save():
            nonlocal last_saved
            if last_saved == counters["steps"]:
                return
            update()
            state = dict(protocol=PAPER_PROTOCOL, signature=signature, config=config, num_envs=num_envs,
                validation_mode=args.validation_mode, agent=agent.state_dict(), replay=replay.state_dict(),
                counters=dict(counters), workers=copy.deepcopy(workers), worker_cursor=cursor,
                environment_states=pool.states(list(observations)),
                recent={key: list(value) for key, value in recent.items()}, rng=rng.bit_generator.state,
                recent_errors={key: list(value) for key, value in recent_errors.items()},
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(agent.device) if agent.device.type == "cuda" else None,
                last_metrics=dict(last_metrics), elapsed_s=elapsed_before + time.perf_counter() - started)
            target = checkpoint_dir / f"step_{counters['steps']:07d}.pt"
            temporary = target.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(target)
            last_saved = counters["steps"]
            print(f"完整断点已保存：{target}", flush=True)

        with CsvLog(output / "episodes.csv", EPISODE_FIELDS) as episodes, CsvLog(
                output / "progress.csv", progress_fields) as progress:
            while counters["steps"] < end and not stopping:
                boundary = min(end,
                    (counters["steps"] // config["train"]["progress_interval"] + 1) * config["train"]["progress_interval"],
                    (counters["steps"] // config["train"]["save_interval"] + 1) * config["train"]["save_interval"])
                count = min(num_envs, boundary - counters["steps"])
                indices = [(cursor + j) % num_envs for j in range(count)]
                cursor = (cursor + count) % num_envs
                resetting = [i for i in indices if workers[i] is None or not workers[i]["active"]]
                # Evaluation reserves the high half of uint32; training can
                # never accidentally draw one of those held-out scene seeds.
                seeds = {i: int(rng.integers(2**31)) for i in resetting}
                for i, (observation, _) in pool.reset_many({i: {} for i in resetting}, seeds=seeds).items():
                    observations[i] = observation
                    workers[i] = dict(active=True, seed=seeds[i])
                agent.set_episode_progress(counters["episodes"])
                before = np.asarray([observations[i] for i in indices])
                if counters["steps"] < config["sac"]["warmup_steps"]:
                    action_array = rng.uniform(-1., 1., (count, 6)).astype(np.float32)
                else:
                    action_array = agent.select_actions(before)
                pool.send_steps(dict(zip(indices, action_array)))
                # Simulation proceeds concurrently with learning from previous data.
                update()
                results = pool.recv_steps()
                following, rewards, terminals = [], [], []
                for i in indices:
                    observation, reward, _, terminated, truncated, info = results[i]
                    observations[i] = observation
                    following.append(observation)
                    rewards.append(reward)
                    terminals.append(terminated)  # Time-limit reset still bootstraps Q.
                    counters["steps"] += 1
                    if terminated or truncated:
                        workers[i]["active"] = False
                        counters["episodes"] += 1
                        for key in recent:
                            recent[key].append(bool(info[key]))
                        for key in recent_errors:
                            recent_errors[key].append(float(info[key]))
                        episodes.writerow(dict(steps=counters["steps"], worker=i, seed=workers[i]["seed"],
                                               **{key: info[key] for key in STEP_INFO_FIELDS}))
                replay.add_many(before, action_array, rewards, following, terminals)
                # Uniform replay samples with replacement, as SB3 does: a
                # batch may be larger than the populated buffer during warmup.
                if replay.size and counters["steps"] >= config["sac"]["update_after"]:
                    counters["update_credit"] += count * config["sac"]["updates_per_transition"]
                    due = int(counters["update_credit"] + 1e-9)
                    counters["update_credit"] -= due
                final = counters["steps"] == end or stopping
                if final or counters["steps"] % config["train"]["progress_interval"] == 0:
                    update()
                    elapsed = time.perf_counter() - started
                    rates = {f"{key}_rate": float(np.mean(values)) if values else "" for key, values in recent.items()}
                    errors = {f"mean_{key}": float(np.mean(values)) if values else "" for key, values in recent_errors.items()}
                    progress.writerow(dict(steps=counters["steps"], episodes=counters["episodes"], updates=counters["updates"],
                        actor_updates=agent.update_steps // agent.actor_update_interval,
                        replay_size=replay.size, elapsed_s=elapsed_before + elapsed,
                        steps_per_second=(counters["steps"] - start) / max(elapsed, 1e-9), **rates, **errors,
                        ae_progress=agent.progress, **last_metrics))
                    episodes.flush()
                    progress.flush()
                    text = format_training_progress(counters["steps"], end, elapsed, recent["safe_success"],
                        label=f"单SAC无碰撞到达(e≤{config['paper']['success']['combined_error_tolerance']:g})",
                        completed_steps=counters["steps"] - start)
                    def percent(value):
                        return f"{value:.1%}" if value != "" else "--"
                    tolerance = config["paper"]["success"]
                    text += (f" | 到达率={percent(rates['reached_rate'])} 碰撞率={percent(rates['collision_seen_rate'])}"
                             f" {tolerance['report_position_tolerance'] * 1000:g}mm/"
                             f"{tolerance['report_orientation_tolerance']:g}rad途中/终点="
                             f"{percent(rates['report_success_rate'])}/{percent(rates['final_report_success_rate'])}"
                             f" 严格到达={percent(rates['strict_success_rate'])}")
                    mean_error = errors['mean_position_error_m']
                    if mean_error != "":
                        text += f" 平均终点误差={mean_error * 1000:.1f}mm"
                    text += f" | 环境={num_envs}"
                    print(text, flush=True)
                if final or counters["steps"] % config["train"]["save_interval"] == 0:
                    save()
            save()
    finally:
        if pool is not None:
            pool.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    (output / "summary.json").write_text(json.dumps(dict(
        protocol=PAPER_PROTOCOL, counters=counters, stopped=stopping,
        validation_mode=args.validation_mode, latest_checkpoint=str(checkpoint_dir / f"step_{last_saved:07d}.pt")),
        indent=2) + "\n")


if __name__ == "__main__":
    main()
