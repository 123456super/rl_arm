from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import trange

from rl_risk_sac.algorithms import SACAgent
from rl_risk_sac.algorithms.replay_buffer import ReplayBuffer
from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.device import resolve_device
from rl_risk_sac.utils.seeding import capture_rng_state, restore_rng_state, set_seed

try:
    from scripts.seed_manifest import load_seed_manifest
except ModuleNotFoundError:
    from seed_manifest import load_seed_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume-actor", default=None)
    parser.add_argument("--resume-state", default=None)
    parser.add_argument("--resume-replay", default=None)
    parser.add_argument(
        "--reset-reward-critics",
        action="store_true",
        help="keep the actor/cost state but reinitialize reward critics for a changed reward",
    )
    parser.add_argument(
        "--reset-cost-critics",
        action="store_true",
        help="reinitialize cost critics when the risk-cost definition changed",
    )
    parser.add_argument(
        "--no-restore-optimizers",
        action="store_true",
        help="load network state without optimizer moments",
    )
    parser.add_argument(
        "--reset-agent-state",
        action="store_true",
        help="load only the actor checkpoint and reinitialize SAC critics/alpha state",
    )
    parser.add_argument(
        "--reset-state-warmup-steps",
        type=int,
        default=10000,
        help="critic-only replay warmup after --reset-agent-state",
    )
    parser.add_argument(
        "--resume-replay-warmup-steps",
        type=int,
        default=None,
        help="collect fresh replay samples before updates after any resume",
    )
    parser.add_argument(
        "--resume-critic-warmup-steps",
        type=int,
        default=None,
        help="critic-only updates after replay warmup for a full-state resume",
    )
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def build_run_dir(config: dict[str, Any], method: str, run_name: str | None) -> Path:
    output_root = Path(config["train"]["output_dir"])
    if run_name:
        return output_root / run_name

    robot_name = Path(str(config["robot"]["urdf"])).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_steps = int(config["train"]["total_steps"])
    seed = int(config["seed"])
    name = f"{timestamp}_{robot_name}_{method}_seed{seed}_steps{total_steps}"
    return output_root / "runs" / name


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def inferred_checkpoint_path(actor_path: str | Path, prefix: str, extension: str) -> Path:
    actor = Path(actor_path)
    if actor.name.startswith("actor"):
        suffix = actor.stem[len("actor") :]
        return actor.with_name(f"{prefix}{suffix}{extension}")
    raise ValueError(f"actor checkpoint name must start with 'actor': {actor}")


def checkpoint_step(path: str | Path, prefix: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(prefix)}_step_(\d+)\.\w+", Path(path).name)
    return int(match.group(1)) if match is not None else None


def validate_resume_checkpoint_steps(
    actor_path: str | Path,
    state_path: str | Path | None,
    replay_path: str | Path | None,
    start_step: int,
) -> None:
    actor_step = checkpoint_step(actor_path, "actor")
    if actor_step is not None and actor_step != start_step:
        raise ValueError(
            f"--start-step={start_step} does not match actor checkpoint step {actor_step}"
        )
    for label, path, prefix in (
        ("agent-state", state_path, "agent_state"),
        ("replay", replay_path, "replay"),
    ):
        if path is None:
            continue
        persisted_step = checkpoint_step(path, prefix)
        if persisted_step is not None and persisted_step != start_step:
            raise ValueError(
                f"--start-step={start_step} does not match {label} checkpoint step {persisted_step}"
            )


def replay_action_signature(path: str | Path) -> str:
    with np.load(path, allow_pickle=False) as payload:
        if "action_signature" not in payload.files:
            return ""
        return ReplayBuffer.metadata_string(payload["action_signature"])


def save_training_checkpoint(
    agent: SACAgent,
    replay: ReplayBuffer,
    env: UR5DynamicObstacleEnv,
    run_dir: Path,
    *,
    suffix: str,
    step: int,
    episode: int,
    save_replay: bool,
) -> None:
    training_state = {
        "step": int(step),
        "episode": int(episode),
        "rng_state": capture_rng_state(),
        "env_rng_state": env.rng.bit_generator.state,
    }
    agent.save(run_dir, suffix=suffix, training_state=training_state)
    if save_replay:
        replay.save(run_dir / f"replay{suffix}.npz")


def reset_training_environment(
    env: UR5DynamicObstacleEnv,
    focused_seeds: list[int],
    focused_fraction: float,
    default_seed: int | None = None,
) -> tuple[np.ndarray, dict[str, Any], int | None, str, int | None]:
    if not focused_seeds:
        observation, info = env.reset(seed=default_seed)
        return observation, info, default_seed, "default", None
    if np.random.random() < focused_fraction:
        seed = int(np.random.choice(focused_seeds))
        source = "focused"
    else:
        seed = int(np.random.randint(0, np.iinfo(np.int32).max))
        source = "random"
    jitter_cfg = env.config["train"].get("focused_reset_jitter", {})
    jitter_seed = None
    options = None
    if source == "focused" and bool(jitter_cfg.get("enabled", False)):
        jitter_seed = int(np.random.randint(0, np.iinfo(np.int32).max))
        options = {"reset_jitter": {**jitter_cfg, "seed": jitter_seed}}
    observation, info = env.reset(seed=seed, options=options)
    return observation, info, seed, source, jitter_seed


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.run_name is not None:
        config["train"]["run_name"] = args.run_name
    method = config["train"]["method"]
    if args.start_step < 0:
        raise ValueError("--start-step must be non-negative")
    if args.reset_agent_state and args.resume_actor is None:
        raise ValueError("--reset-agent-state requires --resume-actor")
    if (args.reset_reward_critics or args.reset_cost_critics or args.no_restore_optimizers) and not args.resume_actor:
        raise ValueError("critic/optimizer resume options require --resume-actor")
    if args.reset_agent_state and (args.reset_reward_critics or args.reset_cost_critics):
        raise ValueError("--reset-agent-state already resets all critics")
    configured_replay_warmup = int(config["sac"].get("resume_replay_warmup_steps", 10000))
    configured_critic_warmup = int(config["sac"].get("resume_critic_warmup_steps", 10000))
    requested_replay_warmup = (
        configured_replay_warmup
        if args.resume_replay_warmup_steps is None
        else args.resume_replay_warmup_steps
    )
    requested_critic_warmup = (
        configured_critic_warmup
        if args.resume_critic_warmup_steps is None
        else args.resume_critic_warmup_steps
    )
    if args.reset_state_warmup_steps < 0:
        raise ValueError("--reset-state-warmup-steps must be non-negative")
    if requested_replay_warmup < 0:
        raise ValueError("--resume-replay-warmup-steps must be non-negative")
    if requested_critic_warmup < 0:
        raise ValueError("--resume-critic-warmup-steps must be non-negative")

    config["device"] = resolve_device(config)
    set_seed(int(config["seed"]))
    env = UR5DynamicObstacleEnv(config, method=method)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, action_dim, config, method=method)
    resume_load_info: dict[str, Any] = {}
    resume_training_state: dict[str, Any] = {}
    resume_rng_restored = False
    resume_state: str | None = None
    if args.resume_actor is not None:
        if args.reset_agent_state and args.resume_state is not None:
            raise ValueError("--reset-agent-state cannot be combined with --resume-state")
        resume_state = args.resume_state
        if resume_state is None and not args.reset_agent_state:
            actor_path = Path(args.resume_actor)
            resume_state = str(actor_path.with_name(actor_path.name.replace("actor", "agent_state", 1)))
        validate_resume_checkpoint_steps(
            args.resume_actor,
            resume_state,
            args.resume_replay,
            args.start_step,
        )
        if args.reset_agent_state:
            agent.load_actor(args.resume_actor)
        else:
            resume_load_info = agent.load(
                args.resume_actor,
                resume_state,
                reset_reward_critics=args.reset_reward_critics,
                reset_cost_critics=args.reset_cost_critics,
                restore_optimizers=not args.no_restore_optimizers,
            )
    config["git_revision"] = git_revision()
    replay = ReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        capacity=int(config["sac"]["replay_size"]),
        device=config.get("device", "cpu"),
        stratified_fraction=float(config["sac"].get("replay_stratified_fraction", 0.0)),
        action_signature=agent.actor_signature,
    )
    resume_replay: Path | None = None
    if args.resume_actor is not None:
        if args.resume_replay is not None:
            resume_replay = Path(args.resume_replay)
            if not resume_replay.is_file():
                raise FileNotFoundError(resume_replay)
        else:
            candidate = inferred_checkpoint_path(args.resume_actor, "replay", ".npz")
            if candidate.is_file():
                resume_replay = candidate
        if resume_replay is not None:
            checkpoint_replay_signature = replay_action_signature(resume_replay)
            if not checkpoint_replay_signature:
                raise ValueError(
                    "legacy replay checkpoint has no action semantics signature; "
                    "do not reuse full-action replay for residual-control training"
                )
            replay.load(resume_replay)
            replay.stratified_fraction = float(config["sac"].get("replay_stratified_fraction", 0.0))

        training_state = resume_load_info.get("training_state")
        if isinstance(training_state, dict):
            resume_training_state = training_state
            checkpoint_step = int(training_state.get("step", args.start_step))
            if checkpoint_step != args.start_step:
                raise ValueError(
                    f"--start-step={args.start_step} does not match checkpoint training step {checkpoint_step}"
                )
            rng_state = training_state.get("rng_state")
            if isinstance(rng_state, dict):
                restore_rng_state(rng_state)
            env_rng_state = training_state.get("env_rng_state")
            if isinstance(env_rng_state, dict):
                env.rng.bit_generator.state = env_rng_state
                resume_rng_restored = True
        config["resume"] = {
            "actor": args.resume_actor,
            "state": resume_state,
            "replay": str(resume_replay) if resume_replay is not None else None,
            "start_step": args.start_step,
            "agent_state_reset": bool(args.reset_agent_state),
            "reward_critics_reset": not bool(resume_load_info.get("reward_critics_loaded", False)),
            "cost_critics_reset": not bool(resume_load_info.get("cost_critics_loaded", False)),
            "optimizers_loaded": bool(resume_load_info.get("optimizers_loaded", False)),
            "agent_state_warmup_steps": args.reset_state_warmup_steps if args.reset_agent_state else 0,
            "resume_replay_warmup_steps": requested_replay_warmup,
            "resume_critic_warmup_steps": requested_critic_warmup,
        }

    run_dir = build_run_dir(config, method, config["train"].get("run_name"))
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = Path(config["train"]["output_dir"]) / "latest_run.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(run_dir) + "\n", encoding="utf-8")
    with open(run_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    metrics_path = run_dir / "train_metrics.csv"
    fieldnames = [
        "episode",
        "step",
        "reset_seed",
        "reset_source",
        "reset_jitter_seed",
        "episode_reward",
        "episode_cost",
        "episode_length",
        "success",
        "collision",
        "collision_any",
        "collision_capsule_overlap",
        "collision_pybullet_contact",
        "termination_collision",
        "termination_reason",
        "safety_violation_rate",
        "min_distance",
        "mean_risk",
        "safety_filter_intervention_rate",
        "mean_safety_filter_intervention_norm",
        "safety_filter_safe_stop_rate",
        "safety_filter_infeasible_rate",
        "safety_filter_projection_failure_rate",
        "predictive_near_miss_rate",
        "min_predictive_h_m",
        "mean_safety_filter_solve_time_s",
        "residual_control_enabled",
        "residual_control_base_qdot_norm",
        "residual_qdot_norm",
        "lambda",
        "alpha",
    ]
    metrics_file = open(metrics_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
    writer.writeheader()

    progress_path = run_dir / "progress.csv"
    progress_fieldnames = [
        "step",
        "total_steps",
        "episode",
        "reset_seed",
        "reset_source",
        "reset_jitter_seed",
        "episode_length",
        "latest_reward",
        "latest_cost",
        "episode_reward",
        "episode_cost",
        "risk_global",
        "d_min",
        "success",
        "collision",
        "collision_any",
        "collision_capsule_overlap",
        "collision_pybullet_contact",
        "termination_collision",
        "safety_filter_status",
        "safety_filter_intervention_norm",
        "safety_filter_safe_stop",
        "safety_filter_max_constraint_violation",
        "predictive_h_min_m",
        "predictive_max_link_speed_bound_mps",
        "predictive_link_velocity_norms_mps",
        "safety_filter_solve_time_s",
        "residual_control_enabled",
        "residual_control_mode",
        "residual_control_base_qdot_norm",
        "residual_qdot_norm",
        "lambda",
        "alpha",
        "loss_actor",
        "loss_actor_anchor",
        "loss_reward_q",
        "loss_cost_q",
        "replay_size",
    ]
    progress_file = open(progress_path, "w", newline="", encoding="utf-8")
    progress_writer = csv.DictWriter(progress_file, fieldnames=progress_fieldnames)
    progress_writer.writeheader()

    total_steps = int(config["train"]["total_steps"])
    if args.start_step > total_steps:
        raise ValueError("--start-step must be less than or equal to train.total_steps")
    warmup_steps = int(config["sac"]["warmup_steps"])
    update_after = int(config["sac"]["update_after"])
    update_every = int(config["sac"]["update_every"])
    resume_critic_warmup_steps = (
        args.reset_state_warmup_steps if args.reset_agent_state else requested_critic_warmup
    ) if args.resume_actor is not None else 0
    resume_replay_warmup_steps = requested_replay_warmup if args.resume_actor is not None else 0
    save_interval = int(config["train"]["save_interval"])
    save_replay = bool(config["train"].get("save_replay", True))
    save_step_replay = bool(config["train"].get("save_step_replay", False))
    log_interval = int(config["train"]["log_interval"])
    progress_interval = max(1, int(config["train"].get("progress_interval", 100)))
    focused_manifest = config["train"].get("focused_reset_seed_manifest")
    focused_seeds = load_seed_manifest(focused_manifest) if focused_manifest else []
    focused_fraction = float(config["train"].get("focused_reset_fraction", 0.0))
    if not 0.0 <= focused_fraction <= 1.0:
        raise ValueError("train.focused_reset_fraction must be in [0, 1]")

    observation, _, current_reset_seed, current_reset_source, current_reset_jitter_seed = reset_training_environment(
        env,
        focused_seeds,
        focused_fraction,
        default_seed=None if resume_rng_restored else int(config["seed"]),
    )
    episode = int(resume_training_state.get("episode", 0))
    episode_reward = 0.0
    episode_cost = 0.0
    episode_length = 0
    episode_costs: list[float] = []
    episode_risks: list[float] = []
    episode_distances: list[float] = []
    episode_violations = 0
    episode_collision_any = False
    episode_capsule_overlap = False
    episode_pybullet_contact = False
    episode_termination_collision = False
    episode_termination_reason = ""
    safety_filter_enabled = bool(config["env"].get("safety_filter", {}).get("enabled", False))
    episode_filter_interventions = 0
    episode_filter_intervention_norms: list[float] = []
    episode_filter_safe_stops = 0
    episode_filter_infeasible = 0
    episode_filter_projection_failures = 0
    episode_predictive_near_misses = 0
    episode_predictive_h_mins: list[float] = []
    episode_filter_solve_times_s: list[float] = []
    episode_replay_indices: list[int] = []
    recent_rewards: deque[float] = deque(maxlen=20)
    update_info: dict[str, float] = {"alpha": float(agent.alpha.detach().cpu()), "lambda": agent.lagrange_multiplier}

    print(
        f"start training: method={method} total_steps={total_steps} device={config['device']} "
        f"run_dir={run_dir} progress_interval={progress_interval} start_step={args.start_step}",
        flush=True,
    )
    if args.reset_agent_state and resume_critic_warmup_steps:
        print(
            f"agent-state warmup: steps={resume_critic_warmup_steps} mode=critic_only",
            flush=True,
        )
    if args.resume_actor is not None and resume_replay_warmup_steps:
        print(
            f"resume replay warmup: steps={resume_replay_warmup_steps} mode=collect_only",
            flush=True,
        )
    if args.resume_actor is not None and resume_critic_warmup_steps and not args.reset_agent_state:
        print(
            f"resume critic warmup: steps={resume_critic_warmup_steps} mode=critic_only",
            flush=True,
        )
    if args.resume_actor is not None:
        print(
            "resume state "
            f"reward_critics_loaded={resume_load_info.get('reward_critics_loaded', False)} "
            f"cost_critics_loaded={resume_load_info.get('cost_critics_loaded', False)} "
            f"optimizers_loaded={resume_load_info.get('optimizers_loaded', False)} "
            f"replay={str(resume_replay) if resume_replay is not None else 'fresh'}",
            flush=True,
        )
    progress = trange(args.start_step + 1, total_steps + 1, desc=f"train:{method}")
    for step in progress:
        schedule_step = step - args.start_step if args.resume_actor is not None else step
        if args.resume_actor is not None:
            action = agent.select_action(observation, deterministic=False)
        elif schedule_step <= warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(observation, deterministic=False)

        next_observation, reward, cost, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_index = replay.add(
            observation,
            action,
            reward,
            cost,
            next_observation,
            done=terminated,
            truncated=truncated,
        )
        episode_replay_indices.append(replay_index)
        observation = next_observation

        episode_reward += reward
        episode_cost += cost
        episode_length += 1
        episode_costs.append(cost)
        episode_risks.append(float(info["risk_global"]))
        episode_distances.append(float(info["d_min"]))
        episode_violations += int(info["safety_violation"])
        episode_collision_any = episode_collision_any or bool(info["collision_any"])
        episode_capsule_overlap = episode_capsule_overlap or bool(info["collision_capsule_overlap"])
        episode_pybullet_contact = episode_pybullet_contact or bool(info["collision_pybullet_contact"])
        if bool(info["termination_collision"]):
            episode_termination_collision = True
            episode_termination_reason = str(info["termination_reason"])
        if safety_filter_enabled:
            filter_status = str(info.get("safety_filter_status", "integration_error"))
            intervention_norm = float(info.get("safety_filter_intervention_norm", float("nan")))
            predictive_h_min = float(info.get("predictive_h_min_m", float("nan")))
            solve_time_s = float(info.get("safety_filter_solve_time_s", float("nan")))
            episode_filter_interventions += int(filter_status != "passthrough")
            episode_filter_safe_stops += int(bool(info.get("safety_filter_safe_stop", False)))
            episode_filter_infeasible += int(filter_status == "safe_stop_infeasible")
            episode_filter_projection_failures += int(filter_status == "safe_stop_projection_failed")
            if np.isfinite(intervention_norm):
                episode_filter_intervention_norms.append(intervention_norm)
            if np.isfinite(predictive_h_min):
                episode_predictive_h_mins.append(predictive_h_min)
                episode_predictive_near_misses += int(predictive_h_min < 0.0)
            if np.isfinite(solve_time_s):
                episode_filter_solve_times_s.append(solve_time_s)

        collect_only = (
            args.resume_actor is not None
            and schedule_step <= resume_replay_warmup_steps
        )
        critic_only = (
            args.resume_actor is not None
            and schedule_step > resume_replay_warmup_steps
            and schedule_step <= resume_replay_warmup_steps + resume_critic_warmup_steps
        )
        if (
            not collect_only
            and schedule_step >= update_after
            and len(replay) >= agent.batch_size
            and schedule_step % update_every == 0
        ):
            batch = replay.sample(agent.batch_size)
            if critic_only:
                update_info = agent.update_critics(batch)
            else:
                update_info = agent.update(batch)

        if step % progress_interval == 0 or step == total_steps:
            progress_row = {
                "step": step,
                "total_steps": total_steps,
                "episode": episode,
                "reset_seed": current_reset_seed,
                "reset_source": current_reset_source,
                "reset_jitter_seed": current_reset_jitter_seed,
                "episode_length": episode_length,
                "latest_reward": reward,
                "latest_cost": cost,
                "episode_reward": episode_reward,
                "episode_cost": episode_cost,
                "risk_global": float(info["risk_global"]),
                "d_min": float(info["d_min"]),
                "success": int(info["success"]),
                "collision": int(info["collision"]),
                "collision_any": int(info["collision_any"]),
                "collision_capsule_overlap": int(info["collision_capsule_overlap"]),
                "collision_pybullet_contact": int(info["collision_pybullet_contact"]),
                "termination_collision": int(info["termination_collision"]),
                "safety_filter_status": info.get("safety_filter_status", "not_enabled"),
                "safety_filter_intervention_norm": info.get("safety_filter_intervention_norm", float("nan")),
                "safety_filter_safe_stop": info.get("safety_filter_safe_stop", False),
                "safety_filter_max_constraint_violation": info.get(
                    "safety_filter_max_constraint_violation", float("nan")
                ),
                "predictive_h_min_m": info.get("predictive_h_min_m", float("nan")),
                "predictive_max_link_speed_bound_mps": info.get(
                    "predictive_max_link_speed_bound_mps", float("nan")
                ),
                "predictive_link_velocity_norms_mps": info.get("predictive_link_velocity_norms_mps", ""),
                "safety_filter_solve_time_s": info.get("safety_filter_solve_time_s", float("nan")),
                "residual_control_enabled": bool(info.get("residual_control_enabled", False)),
                "residual_control_mode": info.get("residual_control_mode", "disabled"),
                "residual_control_base_qdot_norm": float(
                    np.linalg.norm(info.get("residual_control_base_qdot", np.zeros(action_dim)))
                ),
                "residual_qdot_norm": float(np.linalg.norm(info.get("residual_qdot", np.zeros(action_dim)))),
                "lambda": agent.lagrange_multiplier,
                "alpha": float(agent.alpha.detach().cpu()),
                "loss_actor": update_info.get("loss/actor", float("nan")),
                "loss_actor_anchor": update_info.get("loss/actor_anchor", float("nan")),
                "loss_reward_q": update_info.get("loss/reward_q", float("nan")),
                "loss_cost_q": update_info.get("loss/cost_q", float("nan")),
                "replay_size": len(replay),
            }
            progress_writer.writerow(progress_row)
            progress_file.flush()
            print(
                "progress "
                f"step={step}/{total_steps} "
                f"ep={episode} "
                f"ep_len={episode_length} "
                f"ep_reward={episode_reward:.3f} "
                f"ep_cost={episode_cost:.3f} "
                f"risk={float(info['risk_global']):.3f} "
                f"d_min={float(info['d_min']):.3f} "
                f"lambda={agent.lagrange_multiplier:.3f} "
                f"alpha={float(agent.alpha.detach().cpu()):.3f} "
                f"replay={len(replay)}",
                flush=True,
            )

        if done:
            replay.label_episode(episode_replay_indices, success=bool(info["success"]))
            episode += 1
            mean_cost = float(np.mean(episode_costs)) if episode_costs else 0.0
            agent.update_lagrange(mean_cost)
            recent_rewards.append(episode_reward)
            writer.writerow(
                {
                    "episode": episode,
                    "step": step,
                    "reset_seed": current_reset_seed,
                    "reset_source": current_reset_source,
                    "reset_jitter_seed": current_reset_jitter_seed,
                    "episode_reward": episode_reward,
                    "episode_cost": episode_cost,
                    "episode_length": episode_length,
                    "success": int(info["success"]),
                    "collision": int(episode_collision_any),
                    "collision_any": int(episode_collision_any),
                    "collision_capsule_overlap": int(episode_capsule_overlap),
                    "collision_pybullet_contact": int(episode_pybullet_contact),
                    "termination_collision": int(episode_termination_collision),
                    "termination_reason": episode_termination_reason,
                    "safety_violation_rate": episode_violations / max(episode_length, 1),
                    "min_distance": min(episode_distances) if episode_distances else 0.0,
                    "mean_risk": float(np.mean(episode_risks)) if episode_risks else 0.0,
                    "safety_filter_intervention_rate": (
                        episode_filter_interventions / max(episode_length, 1) if safety_filter_enabled else float("nan")
                    ),
                    "mean_safety_filter_intervention_norm": (
                        float(np.mean(episode_filter_intervention_norms))
                        if episode_filter_intervention_norms
                        else float("nan")
                    ),
                    "safety_filter_safe_stop_rate": (
                        episode_filter_safe_stops / max(episode_length, 1) if safety_filter_enabled else float("nan")
                    ),
                    "safety_filter_infeasible_rate": (
                        episode_filter_infeasible / max(episode_length, 1) if safety_filter_enabled else float("nan")
                    ),
                    "safety_filter_projection_failure_rate": (
                        episode_filter_projection_failures / max(episode_length, 1)
                        if safety_filter_enabled
                        else float("nan")
                    ),
                    "predictive_near_miss_rate": (
                        episode_predictive_near_misses / max(episode_length, 1) if safety_filter_enabled else float("nan")
                    ),
                    "min_predictive_h_m": (
                        min(episode_predictive_h_mins) if episode_predictive_h_mins else float("nan")
                    ),
                    "mean_safety_filter_solve_time_s": (
                        float(np.mean(episode_filter_solve_times_s)) if episode_filter_solve_times_s else float("nan")
                    ),
                    "residual_control_enabled": bool(info.get("residual_control_enabled", False)),
                    "residual_control_base_qdot_norm": float(
                        np.linalg.norm(info.get("residual_control_base_qdot", np.zeros(action_dim)))
                    ),
                    "residual_qdot_norm": float(np.linalg.norm(info.get("residual_qdot", np.zeros(action_dim)))),
                    "lambda": agent.lagrange_multiplier,
                    "alpha": float(agent.alpha.detach().cpu()),
                }
            )
            metrics_file.flush()

            if episode % log_interval == 0:
                progress.set_postfix(
                    {
                        "ep": episode,
                        "r20": f"{np.mean(recent_rewards):.2f}" if recent_rewards else "0.00",
                        "cost": f"{mean_cost:.3f}",
                        "lambda": f"{agent.lagrange_multiplier:.3f}",
                        "alpha": f"{update_info.get('alpha', 0.0):.3f}",
                    }
                )

            observation, _, current_reset_seed, current_reset_source, current_reset_jitter_seed = reset_training_environment(
                env,
                focused_seeds,
                focused_fraction,
            )
            episode_reward = 0.0
            episode_cost = 0.0
            episode_length = 0
            episode_costs = []
            episode_risks = []
            episode_distances = []
            episode_violations = 0
            episode_collision_any = False
            episode_capsule_overlap = False
            episode_pybullet_contact = False
            episode_termination_collision = False
            episode_termination_reason = ""
            episode_filter_interventions = 0
            episode_filter_intervention_norms = []
            episode_filter_safe_stops = 0
            episode_filter_infeasible = 0
            episode_filter_projection_failures = 0
            episode_predictive_near_misses = 0
            episode_predictive_h_mins = []
            episode_filter_solve_times_s = []
            episode_replay_indices = []

        if step % save_interval == 0:
            save_training_checkpoint(
                agent,
                replay,
                env,
                run_dir,
                suffix=f"_step_{step}",
                step=step,
                episode=episode,
                save_replay=save_replay and save_step_replay,
            )

    save_training_checkpoint(
        agent,
        replay,
        env,
        run_dir,
        suffix="",
        step=total_steps,
        episode=episode,
        save_replay=save_replay,
    )
    progress_file.close()
    metrics_file.close()
    env.close()
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
