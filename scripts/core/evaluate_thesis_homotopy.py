#!/usr/bin/env python3
"""Run the single, parallel frozen-policy evaluation for V13.5."""

from __future__ import annotations

import argparse
import copy
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


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.utils.config import load_config


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


def _evaluation_backend(config: dict, variant: str):
    """Select the environment/probe implementation from one eval entrypoint."""
    if variant == "auto":
        configured = str(
            config.get("thesis", {}).get("observation", {}).get(
                "ablation_variant", ""
            )
        ).lower()
        variant = "reduced" if (
            configured in {"m3_no_jacobian", "kp_sac"}
        ) else "standard"
    if variant == "reduced":
        from rl_risk_sac.envs.thesis_homotopy_env_reduced_observation import ThesisHomotopyEnv
        from scripts.core.train_thesis_homotopy_reduced_observation import (
            run_deterministic_pose_probe, run_parallel_deterministic_pose_probe,
        )
    else:
        from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
        from scripts.core.train_thesis_homotopy import (
            run_deterministic_pose_probe, run_parallel_deterministic_pose_probe,
        )
    return ThesisHomotopyEnv, run_deterministic_pose_probe, run_parallel_deterministic_pose_probe, variant


def main(default_variant: str = "auto") -> None:
    parser = argparse.ArgumentParser(
        description="Parallel frozen-policy evaluation for a V13.5 checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--variant", choices=("auto", "standard", "reduced"), default=default_variant,
        help="evaluation backend; auto infers reduced keypoint observations from the config",
    )
    parser.add_argument(
        "--config", default="configs/experiments/paper_full.yaml"
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
        "--position-noise-std-mm", type=float, default=0.0,
        help="IID Gaussian end-effector position measurement noise, per-axis std in mm",
    )
    parser.add_argument("--external-force-n", type=float, default=0.0,
                        help="physical tool force magnitude in N, random world direction")
    parser.add_argument("--force-interval-s", type=float, default=0.25)
    parser.add_argument("--disturbance-seed", type=int, default=72001)
    parser.add_argument("--sensor-noise-seed", type=int, default=71001)
    parser.add_argument(
        "--orientation-noise-std-rad", type=float, default=0.0,
        help="world-frame SO(3) noise, std of each Gaussian rotation-vector component in rad",
    )
    parser.add_argument(
        "--checkpoint-config", action="store_true",
        help="use the resolved config saved in a complete checkpoint",
    )
    parser.add_argument(
        "--scene", choices=("auto", "none", "static", "dynamic"), default="auto",
        help="auto maps complete S0/S1/S2 checkpoints to none/static/dynamic",
    )
    parser.add_argument(
        "--blocking-obstacles", action="store_true",
        help=(
            "for static evaluation, place the sphere on intermediate capsules "
            "of the straight reset-to-goal path"
        ),
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
    if not np.isfinite(args.position_noise_std_mm) or args.position_noise_std_mm < 0:
        parser.error("--position-noise-std-mm must be finite and nonnegative")
    if not np.isfinite(args.orientation_noise_std_rad) or args.orientation_noise_std_rad < 0:
        parser.error("--orientation-noise-std-rad must be finite and nonnegative")
    if not np.isfinite(args.external_force_n) or args.external_force_n < 0:
        parser.error("--external-force-n must be finite and nonnegative")
    if not np.isfinite(args.force_interval_s) or args.force_interval_s <= 0:
        parser.error("--force-interval-s must be finite and positive")
    if args.disturbance_seed < 0:
        parser.error("--disturbance-seed must be nonnegative")
    if args.seed < 0 or args.sensor_noise_seed < 0:
        parser.error("evaluation and noise seeds must be nonnegative")
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
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
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    started = time.perf_counter()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if args.checkpoint_config:
        if not isinstance(state, dict) or "config" not in state:
            parser.error("--checkpoint-config requires a checkpoint containing config")
        config = copy.deepcopy(state["config"])
    else:
        config = load_config(config_path)
    source_config_sha256 = hashlib.sha256(json.dumps(
        config, sort_keys=True, default=_json_default,
    ).encode()).hexdigest()
    ThesisHomotopyEnv, run_deterministic_pose_probe, run_parallel_deterministic_pose_probe, variant = _evaluation_backend(config, args.variant)
    config["device"] = args.device
    config["thesis"]["evaluation_sensor_noise"] = {
        "position_std_m": args.position_noise_std_mm / 1000.0,
        "orientation_std_rad": args.orientation_noise_std_rad,
        "seed": args.sensor_noise_seed,
    }
    config["thesis"]["evaluation_mechanical_disturbance"] = {
        "force_n": args.external_force_n,
        "interval_s": args.force_interval_s,
        "seed": args.disturbance_seed,
    }
    config["thesis"]["blocking_obstacle_mode"] = bool(args.blocking_obstacles)
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

    checkpoint_stage = None
    if isinstance(state, dict) and "curriculum" in state:
        checkpoint_stage = state["curriculum"].get("stage")
    scene = args.scene
    if scene == "auto":
        scene = {"s0": "none", "s1": "static", "s2": "dynamic"}.get(
            checkpoint_stage, "none"
        )
    if args.blocking_obstacles and scene != "static":
        parser.error("--blocking-obstacles is only supported for static evaluation")
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
        "checkpoint": os.path.relpath(checkpoint_path, ROOT),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_source": "checkpoint" if args.checkpoint_config else str(config_path),
        "source_config_sha256": source_config_sha256,
        "training_seed": config.get("seed"),
        "position_tolerance_m": contract["position_tolerance"],
        "orientation_tolerance_rad": contract["orientation_tolerance"],
        "sensor_noise": {
            **config["thesis"]["evaluation_sensor_noise"],
            "model": "iid_world_frame_translation_and_left_SO3_rotation_vector_gaussian",
            "precision_scaling_uses_measured_position": True,
            "precision_scaling_uses_measured_orientation": True,
            "metrics_use_ground_truth": True,
            "orientation_measurement": "left_SO3_noise" if args.orientation_noise_std_rad else "exact",
            "joint_measurements_and_model_jacobian": "exact",
        },
        "mechanical_disturbance": {
            **config["thesis"]["evaluation_mechanical_disturbance"],
            "model": "piecewise_constant_fixed_magnitude_isotropic_world_force",
            "application_point": "current_tool_origin",
            "application_rate_hz": 1.0 / float(config["thesis"]["physics_dt"]),
            "measurements": "exact" if not (args.position_noise_std_mm or args.orientation_noise_std_rad) else "noisy",
        },
        "level_index": level_index,
        "scene": scene,
        "episodes": args.episodes,
        "max_episode_steps": int(config["thesis"]["horizon"]),
        "seed": args.seed,
        "num_envs": num_envs,
        "worker_start_method": start_method if num_envs > 1 else None,
        "worker_cpu_ids": cpu_ids,
        "evaluation_variant": variant,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
