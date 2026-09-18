#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict validation for a thesis homotopy checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/experiments/thesis_homotopy.yaml")
    parser.add_argument("--scenes", nargs="+", choices=("none", "static", "dynamic"), default=["none", "static", "dynamic"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=41001)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != config["thesis"]["protocol"]:
        raise ValueError(
            f"checkpoint protocol {checkpoint.get('protocol')!r} does not match "
            f"configured protocol {config['thesis']['protocol']!r}"
        )
    checkpoint_stage = checkpoint["curriculum"]["stage"]
    if checkpoint_stage == "s0":
        goal = checkpoint["curriculum"]["goal"]
        orientation = checkpoint["curriculum"]["orientation"]
        self_safety = checkpoint["curriculum"].get("self_safety")
        required = int(config["thesis"]["goal_curriculum"]["full_scale_min_transitions"])
        pose_required = int(config["thesis"]["orientation_curriculum"]["full_scale_min_transitions"])
        self_required = int(
            config["thesis"]["self_collision"]["curriculum"][
                "full_weight_min_transitions"
            ]
        )
        full_scale_steps = int(getattr(goal, "full_scale_steps", 0))
        at_full_scale = np.isclose(float(goal.scale), float(goal.end), rtol=0.0, atol=1e-12)
        full_pose_steps = int(orientation.full_scale_steps)
        orientation_at_full = np.isclose(
            float(orientation.scale), float(orientation.end), rtol=0.0, atol=1e-12
        )
        pose_rate = float(np.mean(orientation.outcomes)) if orientation.outcomes else None
        anchor_rate = (
            float(np.mean(orientation.anchor_outcomes))
            if orientation.anchor_outcomes else None
        )
        if (not at_full_scale or full_scale_steps < required or not orientation_at_full
                or full_pose_steps < pose_required
                or len(orientation.outcomes) != orientation.outcomes.maxlen
                or pose_rate is None or pose_rate < orientation.floor
                or len(orientation.anchor_outcomes) != orientation.anchor_outcomes.maxlen
                or anchor_rate is None or anchor_rate < orientation.anchor_floor
                or self_safety is None
                or not np.isclose(
                    float(self_safety.weight), float(self_safety.end),
                    rtol=0.0, atol=1e-12,
                )
                or int(getattr(self_safety, "full_weight_steps", 0)) < self_required
                or (getattr(orientation, "deterministic_probe_required", False)
                    and not getattr(orientation, "deterministic_probe_passed", False))):
            raise ValueError(
                "S0 checkpoint is not Gate-eligible: "
                f"goal_scale={goal.scale}, full_scale_steps={full_scale_steps}, required={required}"
                f", orientation_scale={orientation.scale}, full_pose_steps={full_pose_steps}, "
                f"pose_required={pose_required}, pose_rate={pose_rate}, anchor_rate={anchor_rate}, "
                f"self_full_weight_steps={getattr(self_safety, 'full_weight_steps', 0)}, "
                f"self_required={self_required}"
            )
    if checkpoint_stage in {"s1", "s2"}:
        new_scene = "static" if checkpoint_stage == "s1" else "dynamic"
        state = checkpoint["curriculum"]["states"][new_scene]
        if not (state.strict and state.replay_strictified and state.strict_steps >= int(config["thesis"]["strict_min_transitions"])):
            raise ValueError(f"checkpoint is not Gate-eligible: {new_scene} strict_steps={state.strict_steps}")
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=False)
    env = ThesisHomotopyEnv(config)
    agent = ThesisSACAgent(
        int(env.observation_space.shape[0]), int(env.action_space.shape[0]), config
    )
    agent.load_state_dict(checkpoint["agent"])
    rows = []
    for scene_index, scene in enumerate(args.scenes):
        for episode in range(args.episodes):
            env.configure_episode(
                scene, xi=1.0, strict=True, goal_scale=1.0,
                orientation_scale=1.0,
                orientation_tolerance=float(config["thesis"]["orientation_tolerance"]),
            )
            observation, _ = env.reset(seed=args.seed + scene_index * 100000 + episode)
            terminated = truncated = False; total = 0.0; length = 0; info = {}
            minimum_distance = float("inf")
            minimum_self_distance = float("inf")
            command_velocities = []
            measured_velocities = []
            while not (terminated or truncated):
                action = agent.select_action(observation, deterministic=True)
                observation, reward, _, terminated, truncated, info = env.step(action)
                total += reward; length += 1
                command_velocities.append(np.asarray(info["qdot_cmd"], dtype=np.float64))
                measured_velocities.append(np.asarray(info["qdot_measured"], dtype=np.float64))
                minimum_distance = min(minimum_distance, float(info["control_min_distance"]))
                minimum_self_distance = min(
                    minimum_self_distance, float(info["control_self_min_distance"])
                )
            collision = bool(
                info["obstacle_collision"] or info["self_collision"] or info["environment_collision"]
            )
            rows.append({
                "scene": scene, "episode": episode, "return": total, "length": length,
                "safe_success": int(info["safe_success"]),
                "obstacle_collision": int(info["obstacle_collision"]),
                "self_collision": int(info["self_collision"]),
                "environment_collision": int(info["environment_collision"]),
                "collision": int(collision), "joint_limit": int(info["joint_limit"]),
                "timeout": int(truncated), "position_error_m": float(info["goal_error_norm"]),
                "orientation_error_rad": float(info["orientation_error_norm"]),
                "minimum_distance_m": minimum_distance,
                "minimum_self_distance_m": minimum_self_distance,
                **motion_metrics(
                    np.asarray(command_velocities),
                    np.asarray(measured_velocities),
                    float(config["thesis"]["control_dt"]),
                ),
            })
    env.close()
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for scene in args.scenes:
        selected = [row for row in rows if row["scene"] == scene]; count = len(selected)
        summary[scene] = {key + "_rate": float(np.mean([row[key] for row in selected]))
                          for key in ("safe_success", "collision", "obstacle_collision", "self_collision",
                                      "environment_collision", "joint_limit", "timeout")}
        summary[scene]["episodes"] = count
        for field in ("position_error_m", "orientation_error_rad"):
            values = np.asarray([row[field] for row in selected], dtype=np.float64)
            summary[scene][field + "_mean"] = float(np.mean(values))
            summary[scene][field + "_p95"] = float(np.quantile(values, .95))
        summary[scene]["minimum_distance_m"] = float(
            np.min([row["minimum_distance_m"] for row in selected])
        )
        summary[scene]["minimum_self_distance_m"] = float(
            np.min([row["minimum_self_distance_m"] for row in selected])
        )
        for field in MOTION_METRICS:
            values = np.asarray([row[field] for row in selected], dtype=np.float64)
            summary[scene][field + "_mean"] = float(np.mean(values))
            summary[scene][field + "_p95"] = float(np.quantile(values, .95))
            summary[scene][field + "_max"] = float(np.max(values))
    thresholds = {
        "none": {"safe_success_rate": .95, "collision_rate": 0.0, "joint_limit_rate": 0.0, "timeout_rate": .05},
        "static": {"safe_success_rate": .90, "collision_rate": .03, "joint_limit_rate": 0.0, "timeout_rate": .10},
        "dynamic": {"safe_success_rate": .80, "collision_rate": .05, "joint_limit_rate": 0.0, "timeout_rate": .20},
    }
    for scene in args.scenes:
        actual, limit = summary[scene], thresholds[scene]
        actual["gate_pass"] = bool(actual["safe_success_rate"] >= limit["safe_success_rate"]
                                   and actual["collision_rate"] <= limit["collision_rate"]
                                   and actual["joint_limit_rate"] <= limit["joint_limit_rate"]
                                   and actual["timeout_rate"] <= limit["timeout_rate"])
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
