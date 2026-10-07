#!/usr/bin/env python3
"""Run the single, parallel frozen-policy evaluation for V13.5."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config
from scripts.core.train_thesis_homotopy import (
    run_deterministic_pose_probe,
    run_parallel_deterministic_pose_probe,
)


MOTION_METRICS = (
    "command_velocity_rms_rad_s",
    "command_velocity_peak_rad_s",
    "measured_velocity_rms_rad_s",
    "measured_velocity_peak_rad_s",
    "command_acceleration_rms_rad_s2",
    "command_acceleration_peak_rad_s2",
    "measured_acceleration_rms_rad_s2",
    "measured_acceleration_peak_rad_s2",
    "command_jerk_rms_rad_s3",
    "command_jerk_peak_rad_s3",
    "measured_jerk_rms_rad_s3",
    "measured_jerk_peak_rad_s3",
)

EVALUATION_INFO_KEYS = (
    "rho_orientation",
    "next_rho_orientation",
    "control_self_min_distance",
    "strict_pose_reached",
    "obstacle_collision",
    "self_collision",
    "environment_collision",
    "joint_limit",
    "task_reached",
    "next_rho_position",
    "keypoint_distance",
    "next_keypoint_distance",
    "keypoint_tracking_quality",
    "keypoint_precision_quality",
    "keypoint_precision_reward",
    "joint_bottleneck_shaping_reward",
    "keypoint_progress",
    "jacobian_clip_ratio",
)


def _rms_and_peak(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(np.square(values)))), float(np.max(np.abs(values)))


def motion_metrics(
    command_velocities: np.ndarray,
    measured_velocities: np.ndarray,
    control_dt: float,
) -> dict[str, float]:
    """Summarize motion and command oscillation over one control episode."""
    command = np.asarray(command_velocities, dtype=np.float64)
    measured = np.asarray(measured_velocities, dtype=np.float64)
    if command.ndim != 2 or measured.shape != command.shape:
        raise ValueError("command and measured velocities must have matching 2D shapes")
    if command.shape[0] < 1:
        raise ValueError("motion metrics require at least one control step")
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive")
    command_acceleration = np.diff(command, axis=0) / control_dt
    measured_acceleration = np.diff(measured, axis=0) / control_dt
    command_jerk = np.diff(command_acceleration, axis=0) / control_dt
    measured_jerk = np.diff(measured_acceleration, axis=0) / control_dt
    series = {
        "command_velocity": command,
        "measured_velocity": measured,
        "command_acceleration": command_acceleration,
        "measured_acceleration": measured_acceleration,
        "command_jerk": command_jerk,
        "measured_jerk": measured_jerk,
    }
    metrics: dict[str, float] = {}
    for name, values in series.items():
        rms, peak = _rms_and_peak(values)
        unit = "rad_s" if name.endswith("velocity") else (
            "rad_s2" if name.endswith("acceleration") else "rad_s3"
        )
        metrics[f"{name}_rms_{unit}"] = rms
        metrics[f"{name}_peak_{unit}"] = peak
    return metrics


def available_physical_cpu_ids() -> list[int] | None:
    """Return one allowed logical CPU per physical core when Linux exposes it."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    allowed = sorted(os.sched_getaffinity(0))
    selected: list[int] = []
    seen: set[tuple[str, str, str]] = set()
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            # On multi-die CPUs (including Threadripper), core_id is only
            # unique within a die rather than within the whole package.
            die_path = topology / "die_id"
            die = die_path.read_text().strip() if die_path.exists() else "0"
            core = (topology / "core_id").read_text().strip()
        except OSError:
            return allowed
        key = (package, die, core)
        if key in seen:
            continue
        seen.add(key)
        selected.append(cpu)
    return selected


def physical_cpu_ids(count: int) -> list[int] | None:
    """Choose ``count`` distinct physical cores from the allowed CPU set."""
    available = available_physical_cpu_ids()
    if available is None or len(available) < count:
        return None
    return available[:count]


def automatic_num_envs(episodes: int) -> int:
    """Use all available physical cores without creating idle workers."""
    physical = available_physical_cpu_ids()
    capacity = len(physical) if physical else (os.cpu_count() or 1)
    return max(1, min(int(episodes), capacity))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel frozen-policy evaluation for a V13.5 checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml"
    )
    parser.add_argument(
        "--level-index", type=int,
        help="pose level; inferred from a complete checkpoint when omitted",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int,
                        help="override the evaluation episode horizon (default: config horizon)")
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument(
        "--scene", choices=("auto", "none", "static", "dynamic"), default="auto",
        help="auto maps complete S0/S1/S2 checkpoints to none/static/dynamic",
    )
    parser.add_argument("--device", choices=("cpu", "cuda:auto"), default="cpu")
    parser.add_argument(
        "--num-envs", type=int, default=0,
        help="parallel environments; 0 (default) uses all available physical cores",
    )
    parser.add_argument(
        "--no-cpu-affinity", action="store_true",
        help="let the OS schedule workers instead of using distinct physical cores",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.max_episode_steps is not None and args.max_episode_steps < 1:
        parser.error("--max-episode-steps must be positive")
    if args.num_envs < 0:
        parser.error("--num-envs must be non-negative")
    num_envs = automatic_num_envs(args.episodes) if args.num_envs == 0 else min(
        args.num_envs, args.episodes
    )
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    config["device"] = args.device
    if args.max_episode_steps is not None:
        config["thesis"]["horizon"] = args.max_episode_steps
    env = ThesisHomotopyEnv(config)
    try:
        agent = ThesisSACAgent(
            int(env.observation_space.shape[0]),
            int(env.action_space.shape[0]),
            config,
        )
    finally:
        env.close()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    started = time.perf_counter()
    state = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    checkpoint_stage = None
    if isinstance(state, dict) and "curriculum" in state:
        checkpoint_stage = state["curriculum"].get("stage")
    scene = args.scene
    if scene == "auto":
        scene = {"s0": "none", "s1": "static", "s2": "dynamic"}.get(
            checkpoint_stage, "none"
        )
    level_index = args.level_index
    if level_index is None and isinstance(state, dict) and "curriculum" in state:
        orientation = state["curriculum"].get("orientation")
        level_index = int(orientation.level_index)
    if level_index is None:
        parser.error("--level-index is required for an actor-only checkpoint")
    levels = config["thesis"]["joint_pose_curriculum"]["levels"]
    if not 0 <= level_index < len(levels):
        parser.error(f"--level-index must be in [0, {len(levels) - 1}]")
    if isinstance(state, dict) and "agent" in state:
        actor_state = state["agent"]["actor"]
    elif isinstance(state, dict) and "actor" in state:
        actor_state = state["actor"]
    else:
        actor_state = state
    agent.actor.load_state_dict(actor_state)
    agent.actor.eval()

    level = levels[level_index]
    orientation_levels = config["thesis"]["orientation_curriculum"]["levels"]
    contract = dict(
        scene=scene,
        goal_scale=float(level["goal_scale"]),
        orientation_scale=float(orientation_levels[level_index]),
        position_tolerance=float(level["position_tolerance_m"]),
        orientation_tolerance=float(level["orientation_tolerance_rad"]),
        target_distance_min_m=float(level["target_distance_min_m"]),
        target_distance_max_m=float(level["target_distance_max_m"]),
        target_orientation_min_rad=float(level["target_orientation_min_rad"]),
        target_orientation_max_rad=float(level["target_orientation_max_rad"]),
    )
    cpu_ids = None
    if num_envs > 1 and not args.no_cpu_affinity:
        cpu_ids = physical_cpu_ids(num_envs)
        if cpu_ids is None:
            print("warning: distinct physical CPU affinity unavailable; using OS scheduling")
    if num_envs == 1:
        metrics = run_deterministic_pose_probe(
            agent, config, episodes=args.episodes, seed=args.seed, **contract,
        )
    else:
        worker_seeds = [args.seed + 1000000 + index for index in range(num_envs)]
        # CPU-only evaluation can safely fork after the temporary construction
        # environment has closed.  This avoids importing PyTorch and PyBullet
        # again in every worker.  CUDA retains spawn semantics.
        start_method = (
            "fork"
            if args.device == "cpu" and "fork" in mp.get_all_start_methods()
            else "spawn"
        )
        with ParallelThesisEnvPool(
            config, worker_seeds, cpu_ids=cpu_ids,
            step_info_keys=EVALUATION_INFO_KEYS,
            start_method=start_method,
        ) as pool:
            metrics = run_parallel_deterministic_pose_probe(
                agent, pool, config,
                episodes=args.episodes, seed=args.seed, **contract,
            )

    settings = config["thesis"]["orientation_curriculum"]
    thresholds = {
        "none": {
            "success": float(settings["deterministic_probe_success_floor"]),
            "collision": float(settings["deterministic_probe_collision_ceiling"]),
            "timeout": 0.05,
        },
        "static": {"success": 0.90, "collision": 0.03, "timeout": 0.10},
        "dynamic": {"success": 0.80, "collision": 0.05, "timeout": 0.20},
    }
    threshold = thresholds[scene]
    metrics.update({
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "level_index": level_index,
        "scene": scene,
        "episodes": args.episodes,
        "max_episode_steps": int(config["thesis"]["horizon"]),
        "seed": args.seed,
        "num_envs": num_envs,
        "worker_start_method": start_method if num_envs > 1 else None,
        "worker_cpu_ids": cpu_ids,
        "elapsed_seconds": time.perf_counter() - started,
        "success_floor": threshold["success"],
        "bin_success_floor": float(settings["deterministic_probe_bin_success_floor"]),
        "collision_ceiling": threshold["collision"],
        "timeout_ceiling": threshold["timeout"],
        "passed": bool(
            metrics["success_rate"]
            >= threshold["success"]
            and metrics["minimum_bin_success_rate"]
            >= float(settings["deterministic_probe_bin_success_floor"])
            and metrics["collision_rate"]
            <= threshold["collision"]
            and metrics["joint_limit_rate"] == 0.0
            and metrics["timeout_rate"] <= threshold["timeout"]
        ),
    })
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
