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
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from rl_risk_sac.algorithms.thesis_ppo import PPONumericalError, ThesisPPOAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.tasks.reference_reaching import residual_action_penalty
from rl_risk_sac.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]


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


def capture_random_state(pool: ParallelThesisEnvPool) -> dict:
    """Capture every random stream that affects rollout generation."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "environments": pool.rng_states(),
    }


def restore_random_state(state: dict, pool: ParallelThesisEnvPool) -> None:
    """Restore random streams saved by :func:`capture_random_state`."""
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "environments"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"checkpoint random state is incomplete: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "checkpoint CUDA RNG state count does not match visible CUDA devices"
            )
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
    environment_states = {
        int(index): value for index, value in state["environments"].items()
    }
    if set(environment_states) != set(range(len(pool))):
        raise ValueError("checkpoint environment RNG count does not match num-envs")
    pool.set_rng_states(environment_states)


def episode_stats(info):
    distances = np.asarray(info["reference_point_distances"])
    reference_error = float(
        0.5 * distances.sum() + 0.25 * info["orientation_error_norm"]
    )
    return {
        "return": 0.0, "length": 0, "collision": False,
        "obstacle_count": int(info.get("obstacle_count", 0)),
        "dynamic_obstacle_count": int(info.get("dynamic_obstacle_count", 0)),
        "static_obstacle_count": int(info.get("static_obstacle_count", 0)),
        "obstacle_layout": str(info.get("obstacle_layout", "none")),
        "obstacle_collision": False, "self_collision": False,
        "environment_collision": False, "joint_limit": False,
        "success": False,
        "best_position_m": float(info["goal_error_norm"]),
        "best_orientation_rad": float(info["orientation_error_norm"]),
        "best_reference_error": reference_error,
        "max_hold": 0, "action_abs_mean": 0.0, "action_abs_max": 0.0,
        "dense_reward_sum": 0.0, "terminal_reward_sum": 0.0,
        "self_collision_penalty_sum": 0.0,
        "residual_penalty_sum": 0.0,
        "joint_limit_violation_max_rad": 0.0,
    }


def stage_scene(stage: str, config: dict) -> str:
    probabilities = config["thesis"]["scene_mixture"][stage]
    scenes = tuple(probabilities)
    weights = np.asarray([probabilities[scene] for scene in scenes], dtype=np.float64)
    if not scenes or any(
        scene not in {"none", "static", "dynamic", "hybrid"}
        for scene in scenes
    ):
        raise ValueError(f"invalid scene mixture for {stage}")
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError(f"invalid scene probabilities for {stage}")
    weights /= weights.sum()
    return str(np.random.choice(scenes, p=weights))


def make_contract(stage: str, scene: str, config: dict) -> dict:
    del stage, config
    return {
        "scene": scene,
        # Keep the 100-step horizon after a threshold hit.  The dense
        # reference reward is positive near the goal, so terminating on a hit
        # would otherwise make an early success less valuable than hovering
        # just outside the threshold for the rest of the episode.
        "terminate_on_success": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train thesis reaching with PPO")
    parser.add_argument("--config", default="configs/experiments/paper_full.yaml")
    parser.add_argument("--stage", choices=("s0", "s1", "s2"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--num-envs", type=int, default=None)
    # Controlled PPO collects 4096 transitions per update. With the default
    # sixteen parallel environments this is 256 decisions per environment.
    parser.add_argument("--rollout-length", type=int, default=None)
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume")
    parser.add_argument(
        "--initialize-from-stage-checkpoint",
        "--initialize-from-s0",
        dest="initialize_from_stage_checkpoint",
        help=(
            "Transfer compatible model weights into a fresh training run; "
            "do not restore clocks or optimizers. --initialize-from-s0 is a "
            "backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--initial-action-std", type=float,
        help="Gaussian action standard deviation used for fresh stage initialization.",
    )
    parser.add_argument("--actor-lr", type=float)
    parser.add_argument("--actor-lr-min", type=float)
    parser.add_argument("--critic-lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--initial-global-step", type=int, default=None,
        help="Override historical transitions when resuming a legacy checkpoint.",
    )
    parser.add_argument("--device", help="Override configured device, e.g. cpu or cuda:0")
    parser.add_argument("--log-interval", type=int)
    args = parser.parse_args()
    if (args.log_interval is not None and args.log_interval < 1) or (args.steps is not None and args.steps < 1):
        parser.error("steps and log-interval must be positive")
    if args.num_envs is not None and args.num_envs < 1:
        parser.error("num-envs must be positive")
    if args.rollout_length is not None and args.rollout_length < 1:
        parser.error("rollout-length must be positive")
    if args.resume and args.initialize_from_stage_checkpoint:
        parser.error(
            "--resume and --initialize-from-stage-checkpoint are mutually exclusive"
        )
    if args.initial_action_std is not None and args.initial_action_std <= 0.0:
        parser.error("--initial-action-std must be positive")
    if args.actor_lr is not None and args.actor_lr <= 0.0:
        parser.error("--actor-lr must be positive")
    if args.actor_lr_min is not None and args.actor_lr_min < 0.0:
        parser.error("--actor-lr-min must be non-negative")
    if args.critic_lr is not None and args.critic_lr <= 0.0:
        parser.error("--critic-lr must be positive")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")

    config = load_config(ROOT / args.config)
    args.stage = str(config["thesis"]["stage"]) if args.stage is None else args.stage
    args.steps = int(config["train"]["total_steps"]) if args.steps is None else args.steps
    args.seed = int(config["seed"]) if args.seed is None else args.seed
    args.output_dir = str(config["train"]["output_dir"]) if args.output_dir is None else args.output_dir
    args.num_envs = int(config["train"]["num_envs"]) if args.num_envs is None else args.num_envs
    args.rollout_length = int(config["train"]["rollout_length"]) if args.rollout_length is None else args.rollout_length
    args.log_interval = int(config["train"]["update_log_interval"]) if args.log_interval is None else args.log_interval
    args.initial_action_std = float(config["train"]["stage_transfer_action_std"]) if args.initial_action_std is None else args.initial_action_std
    config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    config["thesis"].setdefault("reward", {})["reward_mode"] = "reference_ppo"
    if args.actor_lr is not None:
        config["ppo"]["actor_lr"] = args.actor_lr
    if args.actor_lr_min is not None:
        config["ppo"]["actor_lr_min"] = args.actor_lr_min
    if args.critic_lr is not None:
        config["ppo"]["critic_lr"] = args.critic_lr
    if args.epochs is not None:
        config["ppo"]["epochs"] = args.epochs
    config["train"]["num_envs"] = args.num_envs
    config["train"]["rollout_length"] = args.rollout_length
    config["thesis"]["simple_reaching_mode"] = args.stage == "s0"
    # S1/S2 static scenes should challenge the nominal S0 path. Random free-
    # space spheres often require no avoidance and teach no useful residual.
    config["thesis"]["blocking_obstacle_mode"] = args.stage in {"s1", "s2"}
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
    if args.initialize_from_stage_checkpoint:
        initialization_payload = torch.load(
            args.initialize_from_stage_checkpoint,
            map_location=agent.device,
            weights_only=False,
        )
        initialization_state = (
            initialization_payload["agent"]
            if "agent" in initialization_payload else initialization_payload
        )
        agent.initialize_from_stage_checkpoint(
            initialization_state, action_std=args.initial_action_std
        )
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
    affinity_enabled = bool(config.get("train", {}).get("cpu_affinity", False))
    configured_cpu_ids = config.get("train", {}).get("cpu_ids")
    cpu_ids = None
    if affinity_enabled:
        if configured_cpu_ids is None:
            raise ValueError("train.cpu_ids is required when cpu_affinity is enabled")
        cpu_ids = [int(value) for value in configured_cpu_ids[:args.num_envs]]
        if len(cpu_ids) != args.num_envs or len(set(cpu_ids)) != len(cpu_ids):
            raise ValueError(
                "train.cpu_ids must contain at least one unique CPU per environment"
            )
    pool = ParallelThesisEnvPool(
        config,
        seeds,
        cpu_ids=cpu_ids,
        step_info_keys=(
            "collision",
            "obstacle_collision",
            "self_collision",
            "environment_collision",
            "joint_limit",
            "success",
            "reference_error",
            "next_rho_position",
            "next_rho_orientation",
            "success_hold_count",
            "self_collision_step_penalty",
            "joint_limit_violation_max_rad",
            "timeout",
            "termination_reason",
            "d_min",
            "goal_error_norm",
            "obstacle_count",
            "obstacle_layout",
        ),
    )
    if resume_payload is not None and "random_state" in resume_payload:
        restore_random_state(resume_payload["random_state"], pool)
    elif resume_payload is not None:
        print(
            "warning: legacy checkpoint has no random_state; continuation is "
            "load-compatible but not random-stream exact",
            file=sys.stderr,
            flush=True,
        )
    scenes = [stage_scene(args.stage, config) for _ in range(args.num_envs)]
    obs = np.zeros((args.num_envs, obs_dim), dtype=np.float32)
    episodes = int(resume_payload.get("episodes", 0)) if resume_payload else 0
    completed_steps = 0
    update = int(resume_payload.get("update", 0)) if resume_payload else 0
    recent = deque(maxlen=100)
    if resume_payload:
        recent.extend(resume_payload.get("recent_episodes", []))
    episodes_csv = (output / "episodes.csv").open("w", newline="", encoding="utf-8")
    fields = ["episode", "step", "run_step", "scene", "obstacle_count", "dynamic_obstacle_count", "static_obstacle_count", "obstacle_layout", "return", "length", "success", "collision", "obstacle_collision", "self_collision", "environment_collision", "terminal_collision", "timeout", "joint_limit", "termination_reason", "best_position_m", "best_orientation_rad", "best_reference_error", "final_position_m", "final_orientation_rad", "final_reference_error", "max_hold", "action_abs_mean", "action_abs_max", "dense_reward_sum", "terminal_reward_sum", "self_collision_penalty_sum", "residual_penalty_sum", "joint_limit_violation_max_rad"]
    writer = csv.DictWriter(episodes_csv, fieldnames=fields)
    writer.writeheader()
    updates_file = (output / "updates.jsonl").open("w", encoding="utf-8")
    failed = False
    rollout = {}
    kl_stop = None if agent.target_kl is None else 1.5 * agent.target_kl
    print(f"PPO mode=reference_ppo epochs={agent.epochs} "
          f"normalize_advantage={agent.normalize_advantage} advantage_clip={agent.advantage_clip} "
          f"max_grad_norm={agent.max_grad_norm} "
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
                    "stage": args.stage, "config": config, "update": update,
                    "episodes": episodes, "recent_episodes": list(recent),
                    "random_state": capture_random_state(pool)}, temporary)
        temporary.replace(path)

    last_checkpoint_step = initial_global_step

    try:
        reset = pool.reset_many({i: make_contract(args.stage, scenes[i], config) for i in range(args.num_envs)})
        stats = {}
        for i, (value, info) in reset.items():
            obs[i] = value
            stats[i] = episode_stats(info)
        while completed_steps < args.steps:
            rollout_started = time.perf_counter()
            T = min(args.rollout_length, max(1, (args.steps - completed_steps + args.num_envs - 1) // args.num_envs))
            rollout = {k: [] for k in ("obs", "actions", "std_scale", "logp", "values", "rewards", "residual_penalties", "dones", "terminated", "next_obs")}
            for _ in range(T):
                episode_fraction = episodes / max(1.0, args.steps / float(config["thesis"]["horizon"]))
                actions, logp, values, policy_info = agent.act(obs, episode_fraction=episode_fraction)
                result = pool.step_many({i: actions[i] for i in range(args.num_envs)})
                next_obs = np.empty_like(obs); raw_next_obs = np.empty_like(obs)
                rewards = np.zeros(args.num_envs); dones = np.zeros(args.num_envs, dtype=np.float32)
                residual_penalties = np.zeros(args.num_envs, dtype=np.float32)
                terminated_flags = np.zeros(args.num_envs, dtype=np.float32)
                reset_requests = {}
                for i, (nobs, reward, _cost, terminated, truncated, info) in result.items():
                    reward_config = config["thesis"]["reward"]
                    residual_penalty, _ = residual_action_penalty(
                        policy_info["residual_actions"][i],
                        float(info["goal_error_norm"]),
                        float(info["d_min"]),
                        penalty_weight=float(
                            reward_config["reference_residual_penalty_weight"]
                        ),
                        goal_start_m=float(
                            reward_config["reference_residual_penalty_goal_start_m"]
                        ),
                        goal_full_m=float(
                            reward_config["reference_residual_penalty_goal_full_m"]
                        ),
                        clearance_start_m=float(
                            reward_config[
                                "reference_residual_penalty_clearance_start_m"
                            ]
                        ),
                        clearance_full_m=float(
                            reward_config[
                                "reference_residual_penalty_clearance_full_m"
                            ]
                        ),
                    )
                    reward = float(reward) - residual_penalty
                    residual_penalties[i] = residual_penalty
                    raw_next_obs[i] = nobs
                    next_obs[i] = nobs; rewards[i] = reward
                    terminated_flags[i] = float(terminated)
                    dones[i] = float(terminated or truncated)
                    stat = stats[i]
                    stat["return"] += reward; stat["length"] += 1
                    stat["collision"] |= bool(info.get("collision", False))
                    stat["obstacle_collision"] |= bool(
                        info.get("obstacle_collision", False)
                    )
                    stat["self_collision"] |= bool(
                        info.get("self_collision", False)
                    )
                    stat["environment_collision"] |= bool(
                        info.get("environment_collision", False)
                    )
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
                    step_action_abs = np.abs(actions[i])
                    stat["action_abs_mean"] += float(np.mean(step_action_abs))
                    stat["action_abs_max"] = max(
                        stat["action_abs_max"], float(np.max(step_action_abs))
                    )
                    self_penalty = float(info.get("self_collision_step_penalty", 0.0))
                    stat["self_collision_penalty_sum"] += self_penalty
                    stat["residual_penalty_sum"] += residual_penalty
                    stat["terminal_reward_sum" if dones[i] else "dense_reward_sum"] += reward - self_penalty
                    stat["joint_limit_violation_max_rad"] = max(stat["joint_limit_violation_max_rad"], info.get("joint_limit_violation_max_rad", 0.0))
                    if dones[i]:
                        episodes += 1
                        stat["action_abs_mean"] /= stat["length"]
                        run_step = completed_steps + args.num_envs
                        episode_success = bool(stat["success"])
                        horizon_after_success = bool(
                            episode_success and info.get("timeout", False)
                        )
                        row = {**stat, "episode": episodes,
                               "step": initial_global_step + run_step,
                               "run_step": run_step, "scene": scenes[i],
                               "success": int(episode_success), "terminal_collision": int(info.get("collision", False)), "timeout": int(info.get("timeout", False) and not episode_success), "termination_reason": "success_horizon" if horizon_after_success else info.get("termination_reason", "unknown"), "final_position_m": info["next_rho_position"], "final_orientation_rad": info["next_rho_orientation"], "final_reference_error": reference_error}
                        for flag in (
                            "collision", "obstacle_collision", "self_collision",
                            "environment_collision", "joint_limit",
                        ):
                            row[flag] = int(row[flag])
                        writer.writerow(row); recent.append(row)
                        scenes[i] = stage_scene(args.stage, config)
                        reset_requests[i] = make_contract(args.stage, scenes[i], config)
                if reset_requests:
                    reset_results = pool.reset_many(reset_requests)
                    for i, (reset_observation, reset_info) in reset_results.items():
                        next_obs[i] = reset_observation
                        stats[i] = episode_stats(reset_info)
                rollout["obs"].append(obs.copy()); rollout["actions"].append(policy_info["raw_actions"]); rollout["std_scale"].append(policy_info["std_scale"]); rollout["logp"].append(logp); rollout["values"].append(values); rollout["rewards"].append(rewards); rollout["residual_penalties"].append(residual_penalties); rollout["dones"].append(dones); rollout["terminated"].append(terminated_flags); rollout["next_obs"].append(raw_next_obs.copy())
                obs = next_obs; completed_steps += args.num_envs
                if completed_steps >= args.steps: break
            rollout_seconds = time.perf_counter() - rollout_started
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
            update_started = time.perf_counter()
            metrics = agent.update(
                batch,
                progress=rollout_start_step / max(1.0, float(global_target_steps)),
                diagnostic_dir=output / "diagnostics",
                diagnostic_context={"step": initial_global_step + completed_steps,
                                    "run_step": completed_steps, "update": update + 1,
                                    "stage": args.stage, "config": config},
            ); update += 1
            update_seconds = time.perf_counter() - update_started
            training_seconds = rollout_seconds + update_seconds
            mean = lambda key: float(np.mean([r[key] for r in recent])) if recent else None
            record = {"step": initial_global_step + completed_steps,
                      "run_step": completed_steps, "update": update, "episodes": episodes,
                      "mean_return_last100": mean("return"), "success_rate_last100": mean("success"), "collision_rate_last100": mean("collision"),
                      "obstacle_collision_rate_last100": mean("obstacle_collision"),
                      "self_collision_rate_last100": mean("self_collision"),
                      "environment_collision_rate_last100": mean("environment_collision"),
                      "mean_length_last100": mean("length"), "mean_reward": float(rewards.mean()), "mean_residual_penalty": float(np.mean(rollout["residual_penalties"])), "gamma_mean": float(gammas.mean()), "action_abs_mean": float(np.mean(np.abs(batch["actions"]))), "action_abs_max": float(np.max(np.abs(batch["actions"]))), **metrics}
            record.update(
                rollout_seconds=float(rollout_seconds),
                update_seconds=float(update_seconds),
                training_steps_per_second=float(len(batch["obs"]) / training_seconds),
            )
            updates_file.write(json.dumps(record, allow_nan=False) + "\n"); updates_file.flush(); episodes_csv.flush()
            if metrics["actor_update_rejected"]:
                rejection_directory = output / "diagnostics"
                rejection_directory.mkdir(exist_ok=True)
                rejection = {
                    "step": record["step"],
                    "update": update,
                    "reason": metrics["actor_rejection_reason"],
                    "max_abs_log_ratio": metrics["max_abs_log_ratio"],
                    "max_approx_kl": metrics["max_approx_kl"],
                    "actor_lr": metrics["actor_lr"],
                    "advantage_clip_fraction": metrics["advantage_clip_fraction"],
                    "actor_optimizer_attempts": metrics["actor_optimizer_attempts"],
                    "accepted_actor_steps": metrics["optimizer_steps"],
                }
                rejection_path = rejection_directory / (
                    f"actor_rejection_step_{record['step']:07d}_update_{update:06d}.json"
                )
                rejection_path.write_text(
                    json.dumps(rejection, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            if update % args.log_interval == 0:
                print(f"step={record['step']}/{global_target_steps} run_step={completed_steps}/{args.steps} update={update} episodes={episodes} return100={record['mean_return_last100']} success100={record['success_rate_last100']} len100={record['mean_length_last100']} reward={record['mean_reward']:.3f} residual_penalty={record['mean_residual_penalty']:.4f} actor={metrics['actor_loss']:.4f} critic={metrics['critic_loss']:.4f} entropy={metrics['entropy']:.3f} std={metrics['action_std_mean']:.3f} std_ema={metrics['actor_std_ema']:.3f} residual_on={metrics['residual_enabled_fraction']:.3f} residual=({metrics['residual_action_abs_mean']:.4f},{metrics['residual_action_abs_max']:.4f}) action=({record['action_abs_mean']:.4f},{record['action_abs_max']:.4f}) kl={metrics['approx_kl']:.4f} kl_max={metrics['max_approx_kl']:.4f} kl_reject={int(metrics['actor_update_rejected'])} actor_steps={metrics['optimizer_steps']}/{metrics['actor_optimizer_attempts']} critic_steps={metrics['critic_optimizer_steps']} lr=({metrics['actor_lr']:.3g},{metrics['critic_lr']:.3g}) clip={metrics['clip_fraction']:.3f} adv_clip={metrics['advantage_clip_fraction']:.3f} grad=({metrics['actor_grad_norm']:.2f},{metrics['critic_grad_norm']:.2f}) time=({record['rollout_seconds']:.2f},{record['update_seconds']:.2f}) steps/s={record['training_steps_per_second']:.1f}", flush=True)
            if metrics["kl_early_stopped"]:
                if metrics["actor_rejection_reason"] == "target_kl":
                    rejection_detail = (
                        f"kl={metrics['max_approx_kl']:.6g} > {kl_stop:.6g}"
                    )
                else:
                    rejection_detail = (
                        f"max_abs_log_ratio={metrics['max_abs_log_ratio']:.6g} "
                        f"> {agent.max_abs_log_ratio:.6g}"
                    )
                print(f"Actor early stop: reason={metrics['actor_rejection_reason']} "
                      f"update={update} epoch={metrics['stop_epoch']} "
                      f"{rejection_detail}; "
                      f"Actor proposal rolled back; actor_lr={metrics['actor_lr']:.6g}; "
                      "Critic completed its independent update", flush=True)
            if record["step"] - last_checkpoint_step >= int(config["train"]["save_interval"]):
                save_checkpoint()
                last_checkpoint_step = record["step"]
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
