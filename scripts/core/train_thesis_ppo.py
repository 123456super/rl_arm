#!/usr/bin/env python3
"""PPO trainer for the thesis reaching task.

This is the on-policy PPO trainer for the thesis reaching task.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from rl_risk_sac.algorithms.thesis_ppo import PPONumericalError, ThesisPPOAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_NUM_ENVS = 8
PPO_ROLLOUT_TRANSITIONS = 4096
REFERENCE_ROLLOUT_LENGTH = PPO_ROLLOUT_TRANSITIONS // REFERENCE_NUM_ENVS


def compute_gae(rewards, values, next_values, terminated, dones, gammas, gae_lambda):
    """Bootstrap rollout cuts, never bootstrap task-terminal timeouts."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gammas[t] * next_values[t] * (1.0 - terminated[t]) - values[t]
        last = delta + gammas[t] * gae_lambda * (1.0 - dones[t]) * last
        advantages[t] = last
    return advantages, advantages + values


def fixed_discount_factors(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Use one stationary MDP discount for every transition."""
    discount = float(gamma)
    if not np.isfinite(discount) or not 0.0 < discount <= 1.0:
        raise ValueError("PPO gamma must be finite and in (0, 1]")
    return np.full(rewards.shape, discount, dtype=np.float32)


def episode_stats(info):
    distances = np.asarray(info["reference_point_distances"])
    reference_error = float(
        0.5 * distances.sum() + 0.25 * info["orientation_error_norm"]
    )
    return {
        "return": 0.0, "length": 0, "collision": False, "joint_limit": False,
        "success": False,
        "best_position_m": float(info["goal_error_norm"]),
        "best_orientation_rad": float(info["orientation_error_norm"]),
        "best_reference_error": reference_error,
        "max_hold": 0, "action_clip_fraction": 0.0,
        "dense_reward_sum": 0.0, "terminal_reward_sum": 0.0,
        "self_collision_penalty_sum": 0.0, "joint_limit_violation_max_rad": 0.0,
    }


def stage_scene() -> str:
    return "none"


def make_contract(stage: str, scene: str) -> dict:
    return {
        "scene": scene,
        "xi": 1.0 if scene != "none" else 1.0,
        "strict": bool(stage == "s2"),
        "goal_scale": 1.0,
        "orientation_scale": 1.0,
        "position_tolerance": 0.01,
        "orientation_tolerance": 0.10,
        "lambda_self": 1.0,
        # Keep the 100-step horizon after a threshold hit.  The dense
        # reference reward is positive near the goal, so terminating on a hit
        # would otherwise make an early success less valuable than hovering
        # just outside the threshold for the rest of the episode.
        "terminate_on_success": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train thesis reaching with PPO")
    parser.add_argument("--config", default="configs/experiments/paper_full.yaml")
    parser.add_argument("--stage", choices=("s0",), default="s0")
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--num-envs", type=int, default=None)
    # Controlled PPO collects 4096 transitions per update. With the default
    # eight parallel environments this is 512 decisions per environment.
    parser.add_argument("--rollout-length", type=int, default=None)
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", default="outputs/ppo")
    parser.add_argument("--seed", type=int, default=11001)
    parser.add_argument("--resume")
    parser.add_argument(
        "--initial-global-step", type=int, default=None,
        help="Override historical transitions when resuming a legacy checkpoint.",
    )
    parser.add_argument("--device", help="Override configured device, e.g. cpu or cuda:0")
    parser.add_argument("--log-interval", type=int, default=1)
    args = parser.parse_args()
    if min(args.steps, args.log_interval) < 1:
        parser.error("steps and log-interval must be positive")
    if args.num_envs is not None and args.num_envs < 1:
        parser.error("num-envs must be positive")
    if args.rollout_length is not None and args.rollout_length < 1:
        parser.error("rollout-length must be positive")

    config = load_config(ROOT / args.config)
    config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    config["thesis"].setdefault("reward", {})["reward_mode"] = "reference_ppo"
    if args.num_envs not in (None, REFERENCE_NUM_ENVS):
        raise ValueError(f"reference PPO requires exactly {REFERENCE_NUM_ENVS} environments")
    if args.rollout_length not in (None, REFERENCE_ROLLOUT_LENGTH):
        raise ValueError("reference PPO requires 512 decisions per environment")
    args.num_envs = REFERENCE_NUM_ENVS
    args.rollout_length = REFERENCE_ROLLOUT_LENGTH
    config["thesis"]["simple_reaching_mode"] = True
    config["thesis"]["blocking_obstacle_mode"] = False
    config["thesis"]["stage"] = args.stage
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    run_name = args.run_name or f"{datetime.now():%Y%m%d_%H%M%S}_{args.stage}_ppo_seed{args.seed}"
    output = ROOT / args.output_dir / run_name

    from rl_risk_sac.envs.thesis_homotopy_env_reduced_observation import ThesisHomotopyEnv
    template = ThesisHomotopyEnv(config)
    obs_dim = int(template.observation_space.shape[0]); action_dim = int(template.action_space.shape[0])
    template.close()
    agent = ThesisPPOAgent(obs_dim, action_dim, config)
    resume_payload = None
    initial_global_step = 0
    if args.resume:
        resume_payload = torch.load(args.resume, map_location=agent.device, weights_only=False)
        agent.load_state_dict(resume_payload["agent"] if "agent" in resume_payload else resume_payload)
        if args.initial_global_step is not None:
            initial_global_step = args.initial_global_step
        elif isinstance(resume_payload, dict):
            initial_global_step = int(resume_payload.get("global_step", resume_payload.get("step", 0)))
    elif args.initial_global_step not in (None, 0):
        parser.error("--initial-global-step requires --resume")
    if initial_global_step < 0:
        parser.error("initial global step must be non-negative")
    global_target_steps = initial_global_step + args.steps
    config.setdefault("train", {}).update(
        total_steps=global_target_steps,
        additional_steps=args.steps,
        initial_global_step=initial_global_step,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    seeds = [args.seed + i * 9973 for i in range(args.num_envs)]
    pool = ParallelThesisEnvPool(config, seeds)
    scenes = [stage_scene() for _ in range(args.num_envs)]
    obs = np.zeros((args.num_envs, obs_dim), dtype=np.float32)
    episodes = 0; completed_steps = 0; update = 0
    recent = deque(maxlen=100)
    episodes_csv = (output / "episodes.csv").open("w", newline="", encoding="utf-8")
    fields = ["episode", "step", "run_step", "scene", "return", "length", "success", "collision", "terminal_collision", "timeout", "joint_limit", "termination_reason", "best_position_m", "best_orientation_rad", "best_reference_error", "final_position_m", "final_orientation_rad", "final_reference_error", "max_hold", "action_clip_fraction", "dense_reward_sum", "terminal_reward_sum", "self_collision_penalty_sum", "joint_limit_violation_max_rad"]
    writer = csv.DictWriter(episodes_csv, fieldnames=fields)
    writer.writeheader()
    updates_file = (output / "updates.jsonl").open("w", encoding="utf-8")
    failed = False
    rollout = {}
    kl_stop = None if agent.target_kl is None else 1.5 * agent.target_kl
    print(f"PPO mode=reference_ppo epochs={agent.epochs} "
          f"normalize_advantage={agent.normalize_advantage} max_grad_norm={agent.max_grad_norm} "
          f"target_kl={agent.target_kl} kl_stop={kl_stop} "
          f"max_abs_log_ratio={agent.max_abs_log_ratio} "
          f"global_steps={initial_global_step}->{global_target_steps}", flush=True)

    def save_checkpoint():
        agent.validate_training_state()
        global_step = initial_global_step + completed_steps
        path = output / "checkpoints" / f"step_{global_step:07d}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({"agent": agent.state_dict(), "step": global_step,
                    "global_step": global_step, "run_step": completed_steps,
                    "stage": args.stage, "config": config, "update": update}, temporary)
        temporary.replace(path)

    try:
        reset = pool.reset_many({i: make_contract(args.stage, scenes[i]) for i in range(args.num_envs)})
        stats = {}
        for i, (value, info) in reset.items():
            obs[i] = value
            stats[i] = episode_stats(info)
        while completed_steps < args.steps:
            T = min(args.rollout_length, max(1, (args.steps - completed_steps + args.num_envs - 1) // args.num_envs))
            rollout = {k: [] for k in ("obs", "actions", "std_scale", "logp", "values", "rewards", "dones", "terminated", "next_obs")}
            for _ in range(T):
                episode_fraction = episodes / max(1.0, args.steps / float(config["thesis"].get("horizon", 100)))
                actions, logp, values, policy_info = agent.act(obs, episode_fraction=episode_fraction)
                result = pool.step_many({i: actions[i] for i in range(args.num_envs)})
                next_obs = np.empty_like(obs); raw_next_obs = np.empty_like(obs)
                rewards = np.zeros(args.num_envs); dones = np.zeros(args.num_envs, dtype=np.float32)
                terminated_flags = np.zeros(args.num_envs, dtype=np.float32)
                for i, (nobs, reward, _cost, terminated, truncated, info) in result.items():
                    raw_next_obs[i] = nobs
                    next_obs[i] = nobs; rewards[i] = reward
                    terminated_flags[i] = float(terminated)
                    dones[i] = float(terminated or truncated)
                    stat = stats[i]
                    stat["return"] += reward; stat["length"] += 1
                    stat["collision"] |= bool(info.get("collision", False))
                    stat["joint_limit"] |= bool(info.get("joint_limit", False))
                    # Preserve the old episode-success meaning (the threshold
                    # was reached at least once) now that a hit no longer ends
                    # the episode immediately.
                    stat["success"] |= bool(info.get("success", False))
                    reference_error = float(info["reference_error"])
                    stat["best_position_m"] = min(stat["best_position_m"], info["next_rho_position"])
                    stat["best_orientation_rad"] = min(stat["best_orientation_rad"], info["next_rho_orientation"])
                    stat["best_reference_error"] = min(
                        stat["best_reference_error"], reference_error
                    )
                    stat["max_hold"] = max(stat["max_hold"], info["success_hold_count"])
                    stat["action_clip_fraction"] += float(np.mean(np.abs(policy_info["raw_actions"][i] - actions[i]) > 1e-6))
                    self_penalty = float(info.get("self_collision_step_penalty", 0.0))
                    stat["self_collision_penalty_sum"] += self_penalty
                    stat["terminal_reward_sum" if dones[i] else "dense_reward_sum"] += reward - self_penalty
                    stat["joint_limit_violation_max_rad"] = max(stat["joint_limit_violation_max_rad"], info.get("joint_limit_violation_max_rad", 0.0))
                    if dones[i]:
                        episodes += 1
                        stat["action_clip_fraction"] /= stat["length"]
                        run_step = completed_steps + args.num_envs
                        episode_success = bool(stat["success"])
                        horizon_after_success = bool(
                            episode_success and info.get("timeout", False)
                        )
                        row = {**stat, "episode": episodes,
                               "step": initial_global_step + run_step,
                               "run_step": run_step, "scene": scenes[i],
                               "success": int(episode_success), "terminal_collision": int(info.get("collision", False)), "timeout": int(info.get("timeout", False) and not episode_success), "termination_reason": "success_horizon" if horizon_after_success else info.get("termination_reason", "unknown"), "final_position_m": info["next_rho_position"], "final_orientation_rad": info["next_rho_orientation"], "final_reference_error": reference_error}
                        row["collision"] = int(row["collision"]); row["joint_limit"] = int(row["joint_limit"])
                        writer.writerow(row); recent.append(row)
                        scenes[i] = stage_scene()
                        reset_one = pool.reset_many({i: make_contract(args.stage, scenes[i])})[i]
                        next_obs[i] = reset_one[0]
                        stats[i] = episode_stats(reset_one[1])
                rollout["obs"].append(obs.copy()); rollout["actions"].append(policy_info["raw_actions"]); rollout["std_scale"].append(policy_info["std_scale"]); rollout["logp"].append(logp); rollout["values"].append(values); rollout["rewards"].append(rewards); rollout["dones"].append(dones); rollout["terminated"].append(terminated_flags); rollout["next_obs"].append(raw_next_obs.copy())
                obs = next_obs; completed_steps += args.num_envs
                if completed_steps >= args.steps: break
            next_obs_rollout = np.asarray(rollout["next_obs"], dtype=np.float32)
            next_values_rollout = agent.predict_values(next_obs_rollout.reshape(-1, obs_dim))
            next_values_rollout = next_values_rollout.reshape(len(next_obs_rollout), args.num_envs)
            rewards = np.asarray(rollout["rewards"], dtype=np.float32); values = np.asarray(rollout["values"], dtype=np.float32); dones = np.asarray(rollout["dones"], dtype=np.float32); terminated = np.asarray(rollout["terminated"], dtype=np.float32)
            gammas = fixed_discount_factors(rewards, agent.gamma)
            adv, value_targets = compute_gae(rewards, values, next_values_rollout, terminated, dones, gammas, agent.gae_lambda)
            batch = {"obs": np.asarray(rollout["obs"], dtype=np.float32).reshape(-1, obs_dim), "actions": np.asarray(rollout["actions"], dtype=np.float32).reshape(-1, action_dim), "std_scale": np.asarray(rollout["std_scale"], dtype=np.float32).reshape(-1, 1), "logp": np.asarray(rollout["logp"], dtype=np.float32).reshape(-1), "advantages": adv.reshape(-1), "returns": value_targets.reshape(-1)}
            # The reference implementation applies its linear schedule after
            # an update.  Scheduling this update from the rollout's starting
            # step is equivalent and preserves the full initial learning rate.
            rollout_start_step = initial_global_step + completed_steps - len(batch["obs"])
            metrics = agent.update(
                batch,
                progress=rollout_start_step / max(1.0, float(global_target_steps)),
                diagnostic_dir=output / "diagnostics",
                diagnostic_context={"step": initial_global_step + completed_steps,
                                    "run_step": completed_steps, "update": update + 1,
                                    "stage": args.stage, "config": config},
            ); update += 1
            mean = lambda key: float(np.mean([r[key] for r in recent])) if recent else None
            record = {"step": initial_global_step + completed_steps,
                      "run_step": completed_steps, "update": update, "episodes": episodes,
                      "mean_return_last100": mean("return"), "success_rate_last100": mean("success"), "collision_rate_last100": mean("collision"), "mean_length_last100": mean("length"), "mean_reward": float(rewards.mean()), "gamma_mean": float(gammas.mean()), "action_clip_fraction": float(np.mean(np.abs(batch["actions"]) > float(config["thesis"].get("joint_delta_limit_rad", 0.25)))), **metrics}
            updates_file.write(json.dumps(record, allow_nan=False) + "\n"); updates_file.flush(); episodes_csv.flush()
            if update % args.log_interval == 0:
                print(f"step={record['step']}/{global_target_steps} run_step={completed_steps}/{args.steps} update={update} episodes={episodes} return100={record['mean_return_last100']} success100={record['success_rate_last100']} len100={record['mean_length_last100']} reward={record['mean_reward']:.3f} actor={metrics['actor_loss']:.4f} critic={metrics['critic_loss']:.4f} entropy={metrics['entropy']:.3f} std={metrics['action_std_mean']:.3f} std_ema={metrics['actor_std_ema']:.3f} kl={metrics['approx_kl']:.4f} kl_max={metrics['max_approx_kl']:.4f} kl_reject={int(metrics['actor_update_rejected'])} actor_steps={metrics['optimizer_steps']}/{metrics['actor_optimizer_attempts']} critic_steps={metrics['critic_optimizer_steps']} lr=({metrics['actor_lr']:.3g},{metrics['critic_lr']:.3g}) clip={metrics['clip_fraction']:.3f} action_clip={record['action_clip_fraction']:.3f} grad=({metrics['actor_grad_norm']:.2f},{metrics['critic_grad_norm']:.2f})", flush=True)
            if metrics["kl_early_stopped"]:
                print(f"KL early stop: update={update} epoch={metrics['stop_epoch']} "
                      f"kl={metrics['max_approx_kl']:.6g} > {kl_stop:.6g}; "
                      f"Actor proposal rolled back; actor_lr={metrics['actor_lr']:.6g}; "
                      "Critic completed its independent update", flush=True)
            if update % 10 == 0:
                save_checkpoint()
    except Exception as error:
        failed = True
        if isinstance(error, PPONumericalError):
            diagnostic_path = error.diagnostic_path
        else:
            # Also retain partial rollouts for failures before agent.update().
            directory = output / "diagnostics"
            directory.mkdir(exist_ok=True)
            diagnostic_path = directory / f"trainer_failure_step_{completed_steps:07d}.pt"
            torch.save({"error": repr(error), "traceback": traceback.format_exc(),
                        "step": initial_global_step + completed_steps,
                        "run_step": completed_steps, "update": update, "config": config,
                        "obs": obs, "rollout": rollout, "agent": agent.state_dict()}, diagnostic_path)
        print(f"Training stopped: {error}\nDiagnostic saved: {diagnostic_path}\n"
              "No failure checkpoint will overwrite a valid checkpoint.", file=sys.stderr, flush=True)
        raise
    finally:
        try:
            if not failed:
                save_checkpoint()
        finally:
            episodes_csv.close(); updates_file.close(); pool.close()
    (output / "summary.json").write_text(json.dumps({"algorithm": "PPO", "stage": args.stage,
        "steps": initial_global_step + completed_steps, "global_step": initial_global_step + completed_steps,
        "run_steps": completed_steps, "episodes": episodes,
        "mean_return_last100": float(np.mean([r["return"] for r in recent])) if recent else None}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
