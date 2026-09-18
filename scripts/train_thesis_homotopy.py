#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.algorithms.homotopy_curriculum import HomotopyCurriculum
from rl_risk_sac.algorithms.homotopy_replay import HomotopyReplayBuffer
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_stage_block_steps(
    stage_total_step: int, requested_steps: int, max_stage_steps: int
) -> int:
    if stage_total_step < 0:
        raise ValueError("stage_total_step must be non-negative")
    if requested_steps < 1 or max_stage_steps < 1:
        raise ValueError("requested_steps and max_stage_steps must be positive")
    remaining = max_stage_steps - stage_total_step
    if remaining <= 0:
        raise ValueError(
            f"stage hard budget exhausted: {stage_total_step}/{max_stage_steps} transitions"
        )
    return min(requested_steps, remaining)


def save_checkpoint(path: Path, agent, replay, curriculum, counters, config, env, active_episode) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "protocol": config["thesis"]["protocol"], "agent": agent.state_dict(),
        "replay": replay.state_dict(), "curriculum": curriculum.state_dict(),
        "counters": counters, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "environment_rng": env.rng.bit_generator.state,
        "warmup_action_rng": config["_warmup_action_rng"].bit_generator.state,
        "rng_stream_seeds": config["_rng_stream_seeds"],
        "active_episode": active_episode,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    }, path)


def restore_torch_rng(state: dict) -> None:
    torch.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng"]])


def run_deterministic_position_probe(
    agent: ThesisSACAgent, config: dict, *, episodes: int, seed: int,
) -> dict[str, float]:
    """Evaluate the current deterministic actor on a private position-only gate."""
    probe_env = ThesisHomotopyEnv(config)
    outcomes = []
    try:
        for episode in range(int(episodes)):
            probe_env.configure_episode(
                "none", xi=1.0, strict=True, goal_scale=1.0,
                orientation_scale=0.0, orientation_tolerance=float(np.pi),
            )
            observation, _ = probe_env.reset(seed=int(seed) + episode)
            terminated = truncated = False
            info: dict = {}
            while not (terminated or truncated):
                observation, _, _, terminated, truncated, info = probe_env.step(
                    agent.select_action(observation, deterministic=True)
                )
            collision = bool(
                info["obstacle_collision"]
                or info["self_collision"]
                or info["environment_collision"]
            )
            outcomes.append((
                float(info["task_reached"]), float(collision),
                float(info["joint_limit"]), float(truncated),
            ))
    finally:
        probe_env.close()
    values = np.asarray(outcomes, dtype=np.float64)
    return {
        "success_rate": float(np.mean(values[:, 0])),
        "collision_rate": float(np.mean(values[:, 1])),
        "joint_limit_rate": float(np.mean(values[:, 2])),
        "timeout_rate": float(np.mean(values[:, 3])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Chapters 1--3 task-first safety homotopy")
    parser.add_argument("--config", default="configs/experiments/thesis_homotopy.yaml")
    parser.add_argument("--stage", choices=("s0", "s1", "s2"))
    parser.add_argument("--resume", help="complete checkpoint from the preceding block or stage")
    parser.add_argument(
        "--restart-pose-from",
        help="v5 S0 position checkpoint used once to start the v6 pose-retention curriculum",
    )
    parser.add_argument(
        "--restart-retention-from",
        help="v6 S0 pose checkpoint used once to start the v7 retention-aware curriculum",
    )
    parser.add_argument("--steps", type=int, help="override this block's environment steps")
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int, help="override training seed")
    parser.add_argument("--validation-mode", action="store_true", help="small replay/batch for implementation checks only")
    args = parser.parse_args()
    restart_flags = [args.resume, args.restart_pose_from, args.restart_retention_from]
    if sum(value is not None for value in restart_flags) > 1:
        parser.error("--resume, --restart-pose-from and --restart-retention-from are mutually exclusive")
    config = load_config(ROOT / args.config)
    thesis = config["thesis"]
    if str(thesis.get("protocol", "")).endswith(("_v8", "_v9", "_v10", "_v11")) and (
        args.restart_pose_from or args.restart_retention_from
    ):
        parser.error(
            "the configured protocol changes the observation and reward contracts; "
            "start S0 from scratch or use --resume with a matching checkpoint"
        )
    stage = args.stage or str(thesis["stage"])
    seed = int(args.seed if args.seed is not None else config["seed"]); config["seed"] = seed
    stream_names = ("python", "numpy", "torch", "environment", "curriculum", "replay", "warmup_action")
    children = np.random.SeedSequence(seed).spawn(len(stream_names))
    stream_seeds = {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(stream_names, children)
    }
    config["_rng_stream_seeds"] = stream_seeds
    random.seed(stream_seeds["python"]); np.random.seed(stream_seeds["numpy"])
    torch.manual_seed(stream_seeds["torch"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(stream_seeds["torch"])
    warmup_action_rng = np.random.default_rng(stream_seeds["warmup_action"])
    config["_warmup_action_rng"] = warmup_action_rng
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{stamp}_{stage}_seed{seed}"
    output = ROOT / str(config["train"]["output_dir"]) / run_name
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"; checkpoints.mkdir()
    env = ThesisHomotopyEnv(config)
    env.rng = np.random.default_rng(stream_seeds["environment"])
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    agent = ThesisSACAgent(obs_dim, action_dim, config)
    orientation_curriculum = thesis["orientation_curriculum"]
    replay_storage = orientation_curriculum["replay_storage"]
    self_curriculum = thesis["self_collision"]["curriculum"]
    capacities = ({"none": 512, "static": 512, "dynamic": 512} if args.validation_mode else None)
    storage_scale = 512 if args.validation_mode else None
    replay = HomotopyReplayBuffer(
        obs_dim, action_dim, str(agent.device), capacities=capacities, seed=seed,
        reward_gamma=float(config["sac"]["gamma"]), reward_horizon=int(thesis["horizon"]),
        s0_anchor_fraction=float(orientation_curriculum["replay_anchor_fraction"]),
        s0_current_fraction=float(orientation_curriculum["replay_current_fraction"]),
        s0_anchor_capacity=(
            storage_scale or int(replay_storage["anchor_capacity"])
        ),
        s0_current_capacity=(
            storage_scale or int(replay_storage["current_capacity"])
        ),
        s0_history_capacity_per_level=(
            storage_scale or int(replay_storage["history_capacity_per_level"])
        ),
        reward_parameters=env.reward_parameters,
    )
    replay.rng = np.random.default_rng(stream_seeds["replay"])
    goal_curriculum = thesis["goal_curriculum"]
    full_scale_min_transitions = int(goal_curriculum["full_scale_min_transitions"])
    if full_scale_min_transitions < 1:
        raise ValueError("goal_curriculum.full_scale_min_transitions must be positive")
    pose_full_scale_min_transitions = int(orientation_curriculum["full_scale_min_transitions"])
    if pose_full_scale_min_transitions < 1:
        raise ValueError("orientation_curriculum.full_scale_min_transitions must be positive")
    curriculum = HomotopyCurriculum(
        stage,
        seed=stream_seeds["curriculum"],
        ramp_steps=int(thesis["ramp_steps"]),
        goal_start_scale=float(goal_curriculum["start_scale"]),
        goal_end_scale=float(goal_curriculum["end_scale"]),
        goal_success_window=int(goal_curriculum["success_window"]),
        goal_success_floor=float(goal_curriculum["success_floor"]),
        goal_full_scale_min_steps=full_scale_min_transitions,
        goal_levels=goal_curriculum["levels"],
        goal_min_transitions_per_level=int(goal_curriculum["min_transitions_per_level"]),
        orientation_start_scale=float(orientation_curriculum["start_scale"]),
        orientation_end_scale=float(orientation_curriculum["end_scale"]),
        orientation_tolerance_start=float(orientation_curriculum["tolerance_start"]),
        orientation_tolerance_end=float(orientation_curriculum["tolerance_end"]),
        orientation_success_window=int(orientation_curriculum["success_window"]),
        orientation_success_floor=float(orientation_curriculum["success_floor"]),
        orientation_levels=orientation_curriculum["levels"],
        orientation_min_transitions_per_level=int(
            orientation_curriculum["min_transitions_per_level"]
        ),
        orientation_anchor_probability=float(orientation_curriculum["anchor_probability"]),
        orientation_anchor_floor=float(orientation_curriculum["anchor_success_floor"]),
        orientation_anchor_window=int(orientation_curriculum["anchor_success_window"]),
        orientation_anchor_probability_intermediate=float(
            orientation_curriculum["anchor_probability_intermediate"]
        ),
        orientation_anchor_probability_recovery=float(
            orientation_curriculum["anchor_probability_recovery"]
        ),
        orientation_retention_target=float(orientation_curriculum["retention_target"]),
        orientation_retention_min_observations=int(
            orientation_curriculum["retention_min_observations"]
        ),
        orientation_replay_anchor_normal=float(
            orientation_curriculum["replay_anchor_fraction"]
        ),
        orientation_replay_current_normal=float(
            orientation_curriculum["replay_current_fraction"]
        ),
        orientation_replay_anchor_intermediate=float(
            orientation_curriculum["replay_anchor_intermediate"]
        ),
        orientation_replay_current_intermediate=float(
            orientation_curriculum["replay_current_intermediate"]
        ),
        orientation_replay_anchor_recovery=float(
            orientation_curriculum["replay_anchor_recovery"]
        ),
        orientation_replay_current_recovery=float(
            orientation_curriculum["replay_current_recovery"]
        ),
        orientation_deterministic_probe_required=bool(
            orientation_curriculum["deterministic_probe_required"]
        ),
        orientation_full_scale_min_steps=pose_full_scale_min_transitions,
        self_start_weight=float(self_curriculum["start_weight"]),
        self_end_weight=float(self_curriculum["end_weight"]),
        self_ramp_steps=int(self_curriculum["ramp_steps"]),
        self_full_weight_min_steps=int(
            self_curriculum["full_weight_min_transitions"]
        ),
    )
    counters = {"global_step": 0, "stage_step": 0, "stage_total_step": 0,
                "updates": 0, "episodes": 0, "block": 1,
                "checkpoint_index": 0, "random_steps_used": 0, "replay_redistributions": 0}
    resume_mode = None
    active_episode = None
    checkpoint_source = args.resume or args.restart_pose_from or args.restart_retention_from
    if checkpoint_source:
        state = torch.load(checkpoint_source, map_location=agent.device, weights_only=False)
        pose_restart = bool(args.restart_pose_from)
        retention_restart = bool(args.restart_retention_from)
        expected_protocol = (
            "position_first_pose_then_safety_homotopy_v5"
            if pose_restart else thesis["protocol"]
        )
        if retention_restart:
            expected_protocol = "position_first_pose_then_safety_homotopy_v6"
        if state.get("protocol") != expected_protocol:
            raise ValueError(
                f"checkpoint protocol {state.get('protocol')!r} does not match "
                f"expected protocol {expected_protocol!r}"
            )
        if int(state["config"]["seed"]) != seed:
            raise ValueError(
                f"checkpoint root seed {state['config']['seed']} does not match requested seed {seed}"
            )
        if "warmup_action_rng" not in state or "rng_stream_seeds" not in state:
            raise ValueError("checkpoint predates deterministic RNG stream tracking and cannot be resumed")
        if state["rng_stream_seeds"] != stream_seeds:
            raise ValueError("checkpoint RNG stream derivation does not match the current trainer")
        previous_stage = state["curriculum"]["stage"]
        if retention_restart:
            if stage != "s0" or previous_stage != "s0":
                raise ValueError("--restart-retention-from requires an S0 checkpoint and --stage s0")
            agent.load_state_dict(state["agent"]); replay.load_state_dict(state["replay"])
            curriculum.load_state_dict(state["curriculum"])
            curriculum.configure_orientation_retention(orientation_curriculum)
            # The reward contract changed in v7.  Keep the learned network and
            # replay (whose atomic rewards are relabelled), but require the
            # current eta level to earn fresh counters and windows under v7.
            curriculum.orientation.current_level_steps = 0
            curriculum.orientation.outcomes.clear()
            curriculum.orientation.anchor_outcomes.clear()
            curriculum.orientation.deterministic_probe_passed = False
            curriculum.orientation.deterministic_probe_success_rate = None
            curriculum.orientation.deterministic_probe_collision_rate = None
            curriculum.orientation.deterministic_probe_joint_limit_rate = None
            curriculum.orientation.deterministic_probe_level_steps = -1
            curriculum.orientation.recovery_mode = False
            counters.update(state["counters"])
            counters["stage_total_step"] = int(
                state["counters"].get("stage_total_step", state["counters"]["global_step"])
            )
            counters["block"] += 1; counters["stage_step"] = 0
            random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
            restore_torch_rng(state)
            active_episode = None
            resume_mode = "pose_retention_restart_v6_to_v7"
        elif pose_restart:
            if stage != "s0" or previous_stage != "s0":
                raise ValueError("--restart-pose-from requires an S0 checkpoint and --stage s0")
            previous_goal = state["curriculum"]["goal"]
            previous_orientation = state["curriculum"]["orientation"]
            if (not np.isclose(float(previous_goal.scale), float(previous_goal.end))
                    or not np.isclose(float(previous_orientation.scale), 0.0)):
                raise ValueError(
                    "pose restart requires a full-range position-only checkpoint: "
                    f"goal_scale={previous_goal.scale}, "
                    f"orientation_scale={previous_orientation.scale}"
                )
            agent.load_state_dict(state["agent"]); replay.load_state_dict(state["replay"])
            curriculum.goal = previous_goal
            counters.update(state["counters"])
            counters["stage_total_step"] = int(
                state["counters"].get("stage_total_step", state["counters"]["global_step"])
            )
            counters["block"] += 1; counters["stage_step"] = 0
            random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
            restore_torch_rng(state)
            curriculum.rng.bit_generator.state = state["curriculum"]["rng_state"]
            active_episode = state.get("active_episode")
            resume_mode = "pose_curriculum_restart_v5_to_v6"
        elif {("s0", "s1"), ("s1", "s2")}.intersection({(previous_stage, stage)}):
            if previous_stage == "s0":
                previous_goal = state["curriculum"]["goal"]
                previous_orientation = state["curriculum"]["orientation"]
                previous_self_safety = state["curriculum"].get("self_safety")
                previous_full_steps = int(getattr(previous_goal, "full_scale_steps", 0))
                previous_at_full_scale = np.isclose(
                    float(previous_goal.scale), float(previous_goal.end), rtol=0.0, atol=1e-12
                )
                previous_pose_steps = int(previous_orientation.full_scale_steps)
                previous_orientation_at_full = np.isclose(
                    float(previous_orientation.scale), float(previous_orientation.end), rtol=0.0, atol=1e-12
                )
                previous_pose_rate = (
                    float(np.mean(previous_orientation.outcomes))
                    if previous_orientation.outcomes else None
                )
                previous_anchor_rate = (
                    float(np.mean(previous_orientation.anchor_outcomes))
                    if previous_orientation.anchor_outcomes else None
                )
                if (not previous_at_full_scale or previous_full_steps < full_scale_min_transitions
                        or not previous_orientation_at_full
                        or previous_pose_steps < pose_full_scale_min_transitions
                        or len(previous_orientation.outcomes)
                        != previous_orientation.outcomes.maxlen
                        or previous_pose_rate is None
                        or previous_pose_rate < previous_orientation.floor
                        or len(previous_orientation.anchor_outcomes)
                        != previous_orientation.anchor_outcomes.maxlen
                        or previous_anchor_rate is None
                        or previous_anchor_rate < previous_orientation.anchor_floor
                        or previous_self_safety is None
                        or not np.isclose(
                            float(previous_self_safety.weight),
                            float(previous_self_safety.end),
                            rtol=0.0,
                            atol=1e-12,
                        )
                        or int(getattr(previous_self_safety, "full_weight_steps", 0))
                        < int(self_curriculum["full_weight_min_transitions"])
                        or (getattr(previous_orientation, "deterministic_probe_required", False)
                            and not getattr(previous_orientation, "deterministic_probe_passed", False))):
                    raise ValueError(
                        "S0 checkpoint has not completed full-goal consolidation: "
                        f"goal_scale={previous_goal.scale}, "
                        f"full_scale_steps={previous_full_steps}, "
                        f"required={full_scale_min_transitions}"
                        f", orientation_scale={previous_orientation.scale}, "
                        f"full_pose_steps={previous_pose_steps}, "
                        f"pose_required={pose_full_scale_min_transitions}, "
                        f"pose_rate={previous_pose_rate}, anchor_rate={previous_anchor_rate}"
                    )
            agent.load_state_dict(state["agent"]); replay.load_state_dict(state["replay"])
            inherited = state["counters"]; counters["global_step"] = inherited["global_step"]
            counters["updates"] = inherited["updates"]; counters["episodes"] = inherited["episodes"]
            counters["random_steps_used"] = inherited["random_steps_used"]
            counters["stage_total_step"] = 0
            random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
            restore_torch_rng(state)
            curriculum.rng.bit_generator.state = state["curriculum"]["rng_state"]
            resume_mode = f"stage_transition_{previous_stage}_to_{stage}"
        elif previous_stage == stage:
            agent.load_state_dict(state["agent"]); replay.load_state_dict(state["replay"])
            curriculum.load_state_dict(state["curriculum"]); counters.update(state["counters"])
            if "stage_total_step" not in state["counters"]:
                if stage != "s0":
                    raise ValueError(
                        "same-stage checkpoint lacks stage_total_step; cannot audit its stage budget"
                    )
                counters["stage_total_step"] = int(state["counters"]["global_step"])
            random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
            restore_torch_rng(state)
            counters["block"] += 1; counters["stage_step"] = 0
            active_episode = state.get("active_episode")
            resume_mode = "same_stage_continuation"
        else:
            raise ValueError(f"invalid stage transition {previous_stage}->{stage}")
        env.rng.bit_generator.state = state["environment_rng"]
        warmup_action_rng.bit_generator.state = state["warmup_action_rng"]
    elif stage != "s0":
        raise ValueError("S1/S2 must use --resume with the preceding stage's complete checkpoint")
    curriculum.configure_orientation_retention(orientation_curriculum)

    manifest = {
        "protocol": config["thesis"]["protocol"], "stage": stage, "seed": seed,
        "resume": checkpoint_source, "resume_mode": resume_mode, "config": str(args.config),
        "urdf_sha256": file_sha256(ROOT / config["robot"]["urdf"]),
        "joint_names": config["robot"]["joint_names"], "tool_link_name": config["robot"]["tool_link_name"],
        "observation_dim": obs_dim, "action_dim": action_dim, "external_collision_bodies": [],
        "validation_mode": args.validation_mode,
        "rng_stream_seeds": stream_seeds,
        "replay_storage": {
            **replay_storage,
            "storage_version": replay.STORAGE_VERSION,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = output / "episodes.csv"
    fields = ["episode", "global_step", "stage_step", "stage_total_step", "scene", "length", "return", "curriculum_eligible",
              "orientation_anchor",
              "position_reached", "task_reached",
              "collision_assisted_reach", "safe_success", "obstacle_collision", "self_collision",
              "environment_collision", "joint_limit", "timeout", "xi", "lambda_self", "strict", "rolling_task_reach_rate",
              "eligible_steps", "strict_steps", "goal_scale", "rolling_goal_success_rate",
              "goal_eligible_steps", "goal_level_index", "goal_level_steps",
              "goal_full_scale_steps"]
    fields.extend(["orientation_scale", "orientation_tolerance",
                   "rolling_orientation_success_rate", "orientation_eligible_steps",
                   "orientation_level_index", "orientation_level_steps",
                   "rolling_anchor_position_success_rate", "full_pose_steps",
                   "position_phase_complete", "position_error_m", "orientation_error_rad",
                   "orientation_retention_mode", "orientation_anchor_probability",
                   "deterministic_probe_passed", "deterministic_probe_success_rate",
                   "s0_phase", "pose_phase_complete",
                   "self_safety_eligible_steps", "self_safety_full_weight_steps"])
    transition_fields = ["global_step", "stage_step", "stage_total_step", "episode", "episode_step", "scene", "xi", "lambda_self", "strict",
                         "reward", "r_goal", "c_proximity", "hard_penalty", "safety_penalty",
                         "external_safety_penalty", "self_safety_penalty",
                         "terminal_guard_penalty", "clearance_violation", "risk_max", "d_min", "goal_scale",
                         "self_clearance_violation", "self_risk_max", "self_d_min",
                         "self_ttc_min", "self_approach_max",
                         "goal_full_scale_steps", "orientation_scale", "orientation_tolerance",
                         "orientation_reward_scale", "orientation_reward_gate",
                         "rho_position", "next_rho_position", "rho_orientation",
                         "next_rho_orientation", "position_quality", "orientation_quality",
                         "pose_potential", "next_pose_potential", "pose_potential_progress",
                         "position_progress", "orientation_progress",
                         "orientation_anchor", "position_error_m", "orientation_error_rad",
                         "velocity_magnitude", "velocity_cost", "smooth_velocity", "smooth_cost",
                         "curriculum_eligible", "position_reached",
                         "task_reached", "obstacle_collision", "self_collision", "environment_collision", "joint_limit",
                         "sample_none", "sample_static", "sample_dynamic"]
    update_fields = [
        "global_step", "stage_step", "stage_total_step", "update", "critic_loss", "actor_loss", "alpha_loss", "alpha",
        "q1_mean", "q2_mean", "target_mean", "sample_none", "sample_static", "sample_dynamic",
        "sample_orientation_anchor", "sample_orientation_current", "sample_orientation_historical",
        "critic_gradient_norm", "actor_gradient_norm", "actor_updated",
    ]
    progress_fields = [
        "global_step", "stage_step", "stage_total_step", "episodes", "replay_size", "updates", "alpha",
        "critic_loss", "actor_loss", "q1_mean", "q2_mean", "target_mean", "steps_per_second",
        "goal_scale", "goal_level_index", "goal_level_steps",
        "rolling_goal_success_rate", "goal_eligible_steps",
        "goal_full_scale_steps", "s0_goal_gate_eligible",
        "orientation_scale", "orientation_tolerance", "rolling_orientation_success_rate",
        "orientation_eligible_steps", "orientation_level_index", "orientation_level_steps",
        "rolling_anchor_position_success_rate", "sample_orientation_anchor",
        "sample_orientation_current", "sample_orientation_historical",
        "full_pose_steps", "position_phase_complete", "orientation_retention_mode",
        "orientation_anchor_probability", "replay_orientation_anchor_fraction",
        "replay_orientation_current_fraction", "replay_orientation_historical_fraction",
        "stored_orientation_anchor", "stored_orientation_current",
        "stored_orientation_historical", "stored_orientation_historical_levels",
        "deterministic_probe_passed", "deterministic_probe_success_rate",
        "deterministic_probe_collision_rate", "deterministic_probe_joint_limit_rate",
        "lambda_self", "s0_phase", "pose_phase_complete",
        "self_safety_eligible_steps", "self_safety_full_weight_steps",
    ]
    requested_steps = int(args.steps or config["train"]["total_steps"])
    max_stage_steps = int(config["train"]["max_stage_steps"])
    total_steps = resolve_stage_block_steps(
        counters["stage_total_step"], requested_steps, max_stage_steps
    )
    if total_steps < requested_steps:
        print(
            f"stage budget leaves {total_steps} transitions; "
            f"shortening requested block from {requested_steps}",
            flush=True,
        )
    save_interval = int(config["train"]["save_interval"]); last_saved_step = -1
    progress_interval = int(config["train"].get("progress_interval", 1000))
    if progress_interval < 1:
        raise ValueError("train.progress_interval must be positive")
    batch_size = 8 if args.validation_mode else int(config["sac"]["batch_size"])
    update_after = 8 if args.validation_mode else int(config["sac"]["update_after"])
    with log_path.open("w", newline="", encoding="utf-8") as handle, \
            (output / "transitions.csv").open("w", newline="", encoding="utf-8") as transition_handle, \
            (output / "updates.csv").open("w", newline="", encoding="utf-8") as update_handle, \
            (output / "progress.csv").open("w", newline="", encoding="utf-8") as progress_handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        transition_writer = csv.DictWriter(transition_handle, fieldnames=transition_fields); transition_writer.writeheader()
        update_writer = csv.DictWriter(update_handle, fieldnames=update_fields); update_writer.writeheader()
        progress_writer = csv.DictWriter(progress_handle, fieldnames=progress_fields); progress_writer.writeheader()
        probe_path = output / "orientation_probes.csv"
        probe_handle = probe_path.open("w", newline="", encoding="utf-8")
        probe_fields = [
            "global_step", "stage_step", "stage_total_step",
            "orientation_level_index", "orientation_scale",
            "orientation_level_steps", "episodes", "seed", "success_rate",
            "collision_rate", "joint_limit_rate", "timeout_rate", "passed",
        ]
        probe_writer = csv.DictWriter(probe_handle, fieldnames=probe_fields)
        probe_writer.writeheader()
        last_update: dict[str, float] = {}
        start_time = time.perf_counter()
        while counters["stage_step"] < total_steps:
            if active_episode is None:
                scene = curriculum.choose_scene(); xi, strict = curriculum.contract(scene)
                lambda_self = curriculum.lambda_self
                goal_scale = curriculum.goal_scale
                orientation_anchor = curriculum.choose_orientation_anchor()
                orientation_scale = (
                    curriculum.orientation.start
                    if orientation_anchor else curriculum.orientation_scale
                )
                orientation_tolerance = (
                    curriculum.orientation.tolerance_start
                    if orientation_anchor else curriculum.orientation_tolerance
                )
                env.configure_episode(
                    scene, xi=xi, strict=strict, goal_scale=goal_scale,
                    orientation_scale=orientation_scale,
                    orientation_tolerance=orientation_tolerance,
                    lambda_self=lambda_self,
                )
                obs, _ = env.reset(); episode_return = 0.0; episode_length = 0
                final = {}; terminated = truncated = False; episode_used_warmup = False
            else:
                scene = active_episode["scene"]; xi, strict = curriculum.contract(scene)
                obs = env.restore_episode_state(active_episode["environment"])
                goal_scale = env.contract.goal_scale
                orientation_scale = env.contract.orientation_scale
                orientation_tolerance = env.contract.orientation_tolerance
                lambda_self = env.contract.lambda_self
                orientation_anchor = bool(active_episode.get("orientation_anchor", False))
                episode_return = active_episode["return"]; episode_length = active_episode["length"]
                episode_used_warmup = bool(active_episode["episode_used_warmup"])
                final = active_episode["final"]; terminated = truncated = False; active_episode = None
            while not (terminated or truncated) and counters["stage_step"] < total_steps:
                if stage == "s0" and counters["random_steps_used"] < int(config["sac"]["warmup_steps"]):
                    action = warmup_action_rng.uniform(-1.0, 1.0, size=6).astype(np.float32)
                    counters["random_steps_used"] += 1
                    used_warmup_transition = True
                else:
                    action = agent.select_action(obs)
                    used_warmup_transition = False
                episode_used_warmup = episode_used_warmup or used_warmup_transition
                next_obs, reward, _, terminated, truncated, info = env.step(action)
                curriculum.record_transition(
                    goal_scale, orientation_scale,
                    orientation_anchor=orientation_anchor,
                    curriculum_eligible=not used_warmup_transition,
                )
                replay.add(
                    scene, obs, action, next_obs, terminated or truncated, info,
                    counters["episodes"], episode_length,
                    orientation_anchor=orientation_anchor,
                )
                obs = next_obs; final = info; episode_return += reward; episode_length += 1
                counters["global_step"] += 1
                counters["stage_step"] += 1
                counters["stage_total_step"] += 1
                if len(replay) >= max(update_after, batch_size):
                    replay_anchor_fraction, replay_current_fraction, _ = (
                        curriculum.orientation_replay_mix
                    )
                    batch = replay.sample(
                        stage, curriculum.xi_map(), batch_size,
                        orientation_scale=curriculum.orientation.scale,
                        s0_anchor_fraction=replay_anchor_fraction,
                        s0_current_fraction=replay_current_fraction,
                        lambda_self=curriculum.lambda_self,
                    )
                    last_update = agent.update(batch); counters["updates"] += 1
                    update_writer.writerow({
                        "global_step": counters["global_step"], "stage_step": counters["stage_step"],
                        "stage_total_step": counters["stage_total_step"],
                        "update": counters["updates"], **last_update,
                        "sample_none": replay.last_sample_counts["none"],
                        "sample_static": replay.last_sample_counts["static"],
                        "sample_dynamic": replay.last_sample_counts["dynamic"],
                        "sample_orientation_anchor": replay.last_orientation_sample_counts["anchor"],
                        "sample_orientation_current": replay.last_orientation_sample_counts["current"],
                        "sample_orientation_historical": replay.last_orientation_sample_counts["historical"],
                        "critic_gradient_norm": last_update["critic_gradient_norm"],
                        "actor_gradient_norm": last_update["actor_gradient_norm"],
                        "actor_updated": last_update["actor_updated"],
                    })
                    update_handle.flush()
                transition_writer.writerow({
                    "global_step": counters["global_step"], "stage_step": counters["stage_step"],
                    "stage_total_step": counters["stage_total_step"],
                    "episode": counters["episodes"], "episode_step": episode_length, "scene": scene,
                    "xi": xi, "lambda_self": lambda_self, "strict": int(strict), "reward": reward, "r_goal": info["r_goal"],
                    "c_proximity": info["c_proximity"], "hard_penalty": info["hard_penalty"],
                    "safety_penalty": info["safety_penalty"],
                    "external_safety_penalty": info["external_safety_penalty"],
                    "self_safety_penalty": info["self_safety_penalty"],
                    "terminal_guard_penalty": info["terminal_guard_penalty"],
                    "clearance_violation": info["clearance_violation"],
                    "risk_max": info["control_max_risk"],
                    "d_min": info["control_min_distance"], "goal_scale": goal_scale,
                    "self_clearance_violation": info["self_clearance_violation"],
                    "self_risk_max": info["control_self_max_risk"],
                    "self_d_min": info["control_self_min_distance"],
                    "self_ttc_min": info["control_self_min_ttc"],
                    "self_approach_max": info["control_self_max_approach"],
                    "goal_full_scale_steps": curriculum.goal.full_scale_steps,
                    "orientation_scale": orientation_scale,
                    "orientation_tolerance": orientation_tolerance,
                    "orientation_reward_scale": info["orientation_reward_scale"],
                    "orientation_reward_gate": info["orientation_reward_gate"],
                    "rho_position": info["rho_position"],
                    "next_rho_position": info["next_rho_position"],
                    "rho_orientation": info["rho_orientation"],
                    "next_rho_orientation": info["next_rho_orientation"],
                    "position_quality": info["position_quality"],
                    "orientation_quality": info["orientation_quality"],
                    "pose_potential": info["pose_potential"],
                    "next_pose_potential": info["next_pose_potential"],
                    "pose_potential_progress": info["pose_potential_progress"],
                    "position_progress": info["position_progress"],
                    "orientation_progress": info["orientation_progress"],
                    "orientation_anchor": int(orientation_anchor),
                    "position_error_m": info["next_rho_position"],
                    "orientation_error_rad": info["next_rho_orientation"],
                    "velocity_magnitude": info["velocity_magnitude"],
                    "velocity_cost": info["velocity_cost"],
                    "smooth_velocity": info["smooth_velocity"],
                    "smooth_cost": info["smooth_cost"],
                    "curriculum_eligible": int(not used_warmup_transition),
                    "position_reached": int(info["position_reached"]),
                    "task_reached": int(info["task_reached"]),
                    "obstacle_collision": int(info["obstacle_collision"]), "self_collision": int(info["self_collision"]),
                    "environment_collision": int(info["environment_collision"]), "joint_limit": int(info["joint_limit"]),
                    "sample_none": replay.last_sample_counts["none"], "sample_static": replay.last_sample_counts["static"],
                    "sample_dynamic": replay.last_sample_counts["dynamic"],
                }); transition_handle.flush()
                if counters["stage_step"] % progress_interval == 0:
                    elapsed = max(time.perf_counter() - start_time, 1e-9)
                    goal_success_rate = (
                        float(np.mean(curriculum.goal.outcomes))
                        if curriculum.goal.outcomes else ""
                    )
                    orientation_success_rate = (
                        float(np.mean(curriculum.orientation.outcomes))
                        if curriculum.orientation.outcomes else ""
                    )
                    replay_anchor_fraction, replay_current_fraction, replay_historical_fraction = (
                        curriculum.orientation_replay_mix
                    )
                    replay_storage_counts = replay.orientation_storage_counts()
                    progress = {
                        "global_step": counters["global_step"], "stage_step": counters["stage_step"],
                        "stage_total_step": counters["stage_total_step"],
                        "episodes": counters["episodes"], "replay_size": len(replay),
                        "updates": counters["updates"], "alpha": last_update.get("alpha", float(agent.alpha.detach().cpu())),
                        "critic_loss": last_update.get("critic_loss", ""),
                        "actor_loss": last_update.get("actor_loss", ""),
                        "q1_mean": last_update.get("q1_mean", ""), "q2_mean": last_update.get("q2_mean", ""),
                        "target_mean": last_update.get("target_mean", ""),
                        "steps_per_second": counters["stage_step"] / elapsed,
                        "goal_scale": curriculum.goal_scale,
                        "goal_level_index": curriculum.goal.level_index,
                        "goal_level_steps": curriculum.goal.current_level_steps,
                        "rolling_goal_success_rate": goal_success_rate,
                        "goal_eligible_steps": curriculum.goal.eligible_steps,
                        "goal_full_scale_steps": curriculum.goal.full_scale_steps,
                        "s0_goal_gate_eligible": int(
                            curriculum.s0_goal_gate_eligible(
                                full_scale_min_transitions, pose_full_scale_min_transitions
                            )
                        ),
                        "orientation_scale": curriculum.orientation_scale,
                        "orientation_tolerance": curriculum.orientation_tolerance,
                        "rolling_orientation_success_rate": orientation_success_rate,
                        "orientation_eligible_steps": curriculum.orientation.eligible_steps,
                        "orientation_level_index": curriculum.orientation.level_index,
                        "orientation_level_steps": curriculum.orientation.current_level_steps,
                        "rolling_anchor_position_success_rate": (
                            "" if curriculum.orientation_anchor_success_rate is None
                            else curriculum.orientation_anchor_success_rate
                        ),
                        "sample_orientation_anchor": replay.last_orientation_sample_counts["anchor"],
                        "sample_orientation_current": replay.last_orientation_sample_counts["current"],
                        "sample_orientation_historical": replay.last_orientation_sample_counts["historical"],
                        "full_pose_steps": curriculum.orientation.full_scale_steps,
                        "position_phase_complete": int(curriculum.position_phase_complete),
                        "orientation_retention_mode": curriculum.orientation_retention_mode,
                        "orientation_anchor_probability": curriculum.orientation_anchor_probability,
                        "replay_orientation_anchor_fraction": replay_anchor_fraction,
                        "replay_orientation_current_fraction": replay_current_fraction,
                        "replay_orientation_historical_fraction": replay_historical_fraction,
                        "stored_orientation_anchor": replay_storage_counts["anchor"],
                        "stored_orientation_current": replay_storage_counts["current"],
                        "stored_orientation_historical": replay_storage_counts["historical"],
                        "stored_orientation_historical_levels": len(
                            replay_storage_counts["historical_levels"]
                        ),
                        "deterministic_probe_passed": int(
                            curriculum.orientation.deterministic_probe_passed
                        ),
                        "deterministic_probe_success_rate": (
                            "" if curriculum.orientation.deterministic_probe_success_rate is None
                            else curriculum.orientation.deterministic_probe_success_rate
                        ),
                        "deterministic_probe_collision_rate": (
                            "" if curriculum.orientation.deterministic_probe_collision_rate is None
                            else curriculum.orientation.deterministic_probe_collision_rate
                        ),
                        "deterministic_probe_joint_limit_rate": (
                            "" if curriculum.orientation.deterministic_probe_joint_limit_rate is None
                            else curriculum.orientation.deterministic_probe_joint_limit_rate
                        ),
                        "lambda_self": curriculum.lambda_self,
                        "s0_phase": curriculum.s0_phase if stage == "s0" else stage,
                        "pose_phase_complete": int(curriculum.pose_phase_complete),
                        "self_safety_eligible_steps": curriculum.self_safety.eligible_steps,
                        "self_safety_full_weight_steps": (
                            curriculum.self_safety.full_weight_steps
                        ),
                    }
                    progress_writer.writerow(progress); progress_handle.flush()
                    print(
                        f"step={counters['stage_step']}/{total_steps} episodes={counters['episodes']} "
                        f"replay={len(replay)} updates={counters['updates']} alpha={float(progress['alpha']):.5f} "
                        f"goal_scale={curriculum.goal_scale:.4f} "
                        f"goal_success={goal_success_rate if goal_success_rate == '' else f'{goal_success_rate:.3f}'} "
                        f"orientation_scale={curriculum.orientation_scale:.4f} "
                        f"orientation_tol={curriculum.orientation_tolerance:.3f} "
                        f"orientation_level={curriculum.orientation.level_index} "
                        f"anchor_success={curriculum.orientation_anchor_success_rate} "
                        f"retention={curriculum.orientation_retention_mode} "
                        f"anchor_p={curriculum.orientation_anchor_probability:.2f} "
                        f"phase={curriculum.s0_phase if stage == 's0' else stage} "
                        f"lambda_self={curriculum.lambda_self:.3f} "
                        f"full_goal_steps={curriculum.goal.full_scale_steps} "
                        f"gate_eligible={progress['s0_goal_gate_eligible']} "
                        f"steps/s={progress['steps_per_second']:.1f}",
                        flush=True,
                    )
                if counters["stage_step"] % save_interval == 0 and not (terminated or truncated):
                    active_episode = {"scene": scene, "return": episode_return, "length": episode_length,
                                      "orientation_anchor": orientation_anchor,
                                      "final": final, "episode_used_warmup": episode_used_warmup,
                                      "environment": env.episode_state_dict()}
                    counters["checkpoint_index"] += 1; counters["replay_redistributions"] = replay.redistribution_count
                    checkpoint = checkpoints / f"step_{counters['stage_step']:07d}.pt"
                    save_checkpoint(checkpoint, agent, replay, curriculum, counters, config, env, active_episode)
                    agent.save_actor(checkpoints / f"actor_step_{counters['stage_step']:07d}.pt")
                    print(f"saved {checkpoint} scene={scene} xi={xi:.4f} replay={len(replay)}", flush=True)
                    last_saved_step = counters["stage_step"]
                    active_episode = None
            if not (terminated or truncated):
                active_episode = {"scene": scene, "return": episode_return, "length": episode_length,
                                  "orientation_anchor": orientation_anchor,
                                  "final": final, "episode_used_warmup": episode_used_warmup,
                                  "environment": env.episode_state_dict()}
                break
            update = curriculum.finish_episode(
                scene, bool(final.get("position_reached", False)),
                bool(final.get("task_reached", False)), episode_length,
                orientation_anchor=orientation_anchor,
                curriculum_eligible=not episode_used_warmup,
            )
            if curriculum.orientation_probe_due(
                int(orientation_curriculum["deterministic_probe_interval_transitions"]),
                pose_full_scale_min_transitions,
            ):
                probe_level_index = curriculum.orientation.level_index
                probe_scale = curriculum.orientation.scale
                metrics = run_deterministic_position_probe(
                    agent, config,
                    episodes=int(orientation_curriculum["deterministic_probe_episodes"]),
                    seed=int(orientation_curriculum["deterministic_probe_seed"]),
                )
                probe_passed = bool(
                    metrics["success_rate"]
                    >= float(orientation_curriculum["deterministic_probe_success_floor"])
                    and metrics["collision_rate"] == 0.0
                    and metrics["joint_limit_rate"] == 0.0
                )
                curriculum.record_orientation_probe(
                    success_rate=metrics["success_rate"],
                    collision_rate=metrics["collision_rate"],
                    joint_limit_rate=metrics["joint_limit_rate"],
                    passed=probe_passed,
                )
                level_steps_at_probe = curriculum.orientation.current_level_steps
                advanced = curriculum.advance_orientation_if_ready()
                if advanced:
                    replay.begin_orientation_level(curriculum.orientation.scale)
                probe_writer.writerow({
                    "global_step": counters["global_step"],
                    "stage_step": counters["stage_step"],
                    "stage_total_step": counters["stage_total_step"],
                    "orientation_level_index": probe_level_index,
                    "orientation_scale": probe_scale,
                    "orientation_level_steps": level_steps_at_probe,
                    "episodes": orientation_curriculum["deterministic_probe_episodes"],
                    "seed": orientation_curriculum["deterministic_probe_seed"],
                    **metrics,
                    "passed": int(probe_passed),
                })
                probe_handle.flush()
                print(
                    f"orientation probe eta={probe_scale:.4f} "
                    f"success={metrics['success_rate']:.3f} "
                    f"collision={metrics['collision_rate']:.3f} "
                    f"joint_limit={metrics['joint_limit_rate']:.3f} "
                    f"passed={int(probe_passed)} advanced={int(advanced)}",
                    flush=True,
                )
            if update["became_strict"]:
                report = replay.strictify(scene)
                curriculum.states[scene].replay_strictified = True
                manifest.setdefault("strictification", []).append({"scene": scene, "global_step": counters["global_step"], **report})
                (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            state = curriculum.states.get(scene)
            writer.writerow({
                "episode": counters["episodes"], "global_step": counters["global_step"],
                "stage_step": counters["stage_step"],
                "stage_total_step": counters["stage_total_step"],
                "scene": scene, "length": episode_length, "return": episode_return,
                "curriculum_eligible": int(not episode_used_warmup),
                "orientation_anchor": int(orientation_anchor),
                "position_reached": int(final.get("position_reached", False)),
                "task_reached": int(final.get("task_reached", False)),
                "collision_assisted_reach": int(final.get("collision_assisted_reach", False)),
                "safe_success": int(final.get("safe_success", False)),
                "obstacle_collision": int(final.get("obstacle_collision", False)), "self_collision": int(final.get("self_collision", False)),
                "environment_collision": int(final.get("environment_collision", False)), "joint_limit": int(final.get("joint_limit", False)),
                "timeout": int(truncated), "xi": 1.0 if state is None else state.xi,
                "lambda_self": lambda_self,
                "strict": 1 if state is None else int(state.strict),
                "rolling_task_reach_rate": "" if state is None else update["rolling_task_reach_rate"],
                "eligible_steps": 0 if state is None else state.eligible_steps, "strict_steps": 0 if state is None else state.strict_steps,
                "goal_scale": goal_scale,
                "rolling_goal_success_rate": update["rolling_goal_success_rate"],
                "goal_eligible_steps": curriculum.goal.eligible_steps,
                "goal_level_index": curriculum.goal.level_index,
                "goal_level_steps": curriculum.goal.current_level_steps,
                "goal_full_scale_steps": curriculum.goal.full_scale_steps,
                "orientation_scale": orientation_scale,
                "orientation_tolerance": orientation_tolerance,
                "rolling_orientation_success_rate": update["rolling_orientation_success_rate"],
                "orientation_eligible_steps": curriculum.orientation.eligible_steps,
                "orientation_level_index": curriculum.orientation.level_index,
                "orientation_level_steps": curriculum.orientation.current_level_steps,
                "rolling_anchor_position_success_rate": (
                    "" if curriculum.orientation_anchor_success_rate is None
                    else curriculum.orientation_anchor_success_rate
                ),
                "full_pose_steps": curriculum.orientation.full_scale_steps,
                "position_phase_complete": int(curriculum.position_phase_complete),
                "position_error_m": final.get("next_rho_position", ""),
                "orientation_error_rad": final.get("next_rho_orientation", ""),
                "orientation_retention_mode": curriculum.orientation_retention_mode,
                "orientation_anchor_probability": curriculum.orientation_anchor_probability,
                "deterministic_probe_passed": int(
                    curriculum.orientation.deterministic_probe_passed
                ),
                "deterministic_probe_success_rate": (
                    "" if curriculum.orientation.deterministic_probe_success_rate is None
                    else curriculum.orientation.deterministic_probe_success_rate
                ),
                "s0_phase": curriculum.s0_phase if stage == "s0" else stage,
                "pose_phase_complete": int(curriculum.pose_phase_complete),
                "self_safety_eligible_steps": curriculum.self_safety.eligible_steps,
                "self_safety_full_weight_steps": curriculum.self_safety.full_weight_steps,
            }); handle.flush(); counters["episodes"] += 1
            if (counters["stage_step"] % save_interval == 0 or counters["stage_step"] >= total_steps) and last_saved_step != counters["stage_step"]:
                counters["checkpoint_index"] += 1
                checkpoint = checkpoints / f"step_{counters['stage_step']:07d}.pt"
                counters["replay_redistributions"] = replay.redistribution_count
                save_checkpoint(checkpoint, agent, replay, curriculum, counters, config, env, None)
                agent.save_actor(checkpoints / f"actor_step_{counters['stage_step']:07d}.pt")
                print(f"saved {checkpoint} scene={scene} xi={xi:.4f} replay={len(replay)}", flush=True)
                last_saved_step = counters["stage_step"]
        if active_episode is not None and last_saved_step != counters["stage_step"]:
            counters["checkpoint_index"] += 1; counters["replay_redistributions"] = replay.redistribution_count
            checkpoint = checkpoints / f"step_{counters['stage_step']:07d}.pt"
            save_checkpoint(checkpoint, agent, replay, curriculum, counters, config, env, active_episode)
            agent.save_actor(checkpoints / f"actor_step_{counters['stage_step']:07d}.pt")
            print(f"saved {checkpoint} active_episode=1 replay={len(replay)}", flush=True)
        probe_handle.close()
    env.close()
    summary = {"stage": stage, **counters, "replay_size": len(replay), "xi": curriculum.xi_map(),
               "replay_orientation_storage": replay.orientation_storage_counts(),
               "max_stage_steps": max_stage_steps,
               "stage_budget_exhausted": counters["stage_total_step"] >= max_stage_steps,
               "goal_scale": curriculum.goal_scale,
               "goal_eligible_steps": curriculum.goal.eligible_steps,
               "goal_level_index": curriculum.goal.level_index,
               "goal_level_steps": curriculum.goal.current_level_steps,
               "goal_full_scale_steps": curriculum.goal.full_scale_steps,
               "orientation_scale": curriculum.orientation_scale,
               "orientation_tolerance": curriculum.orientation_tolerance,
               "orientation_eligible_steps": curriculum.orientation.eligible_steps,
               "orientation_level_index": curriculum.orientation.level_index,
               "orientation_level_steps": curriculum.orientation.current_level_steps,
               "rolling_orientation_success_rate": curriculum.orientation_success_rate,
               "rolling_anchor_position_success_rate": curriculum.orientation_anchor_success_rate,
               "orientation_retention_mode": curriculum.orientation_retention_mode,
               "orientation_anchor_probability": curriculum.orientation_anchor_probability,
               "orientation_replay_mix": curriculum.orientation_replay_mix,
               "deterministic_probe_passed": curriculum.orientation.deterministic_probe_passed,
               "deterministic_probe_success_rate": curriculum.orientation.deterministic_probe_success_rate,
               "deterministic_probe_collision_rate": curriculum.orientation.deterministic_probe_collision_rate,
               "deterministic_probe_joint_limit_rate": curriculum.orientation.deterministic_probe_joint_limit_rate,
               "full_pose_steps": curriculum.orientation.full_scale_steps,
               "position_phase_complete": curriculum.position_phase_complete,
               "pose_phase_complete": curriculum.pose_phase_complete,
               "s0_phase": curriculum.s0_phase if stage == "s0" else stage,
               "lambda_self": curriculum.lambda_self,
               "self_safety_eligible_steps": curriculum.self_safety.eligible_steps,
               "self_safety_full_weight_steps": curriculum.self_safety.full_weight_steps,
               "s0_goal_gate_eligible": curriculum.s0_goal_gate_eligible(
                   full_scale_min_transitions, pose_full_scale_min_transitions
               ),
               "strict_steps": {k: v.strict_steps for k, v in curriculum.states.items()}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
