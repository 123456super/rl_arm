from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = _load_config_with_includes(config_path)
    validate_config(config)
    return config


def _load_config_with_includes(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        current = yaml.safe_load(file) or {}

    merged: dict[str, Any] = {}
    includes = current.pop("includes", []) or []
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        deep_update(merged, _load_config_with_includes(include_path))

    deep_update(merged, current)
    return merged


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def validate_config(config: dict[str, Any]) -> None:
    required_paths = [
        ("robot", "urdf"),
        ("robot", "base_position"),
        ("robot", "joint_names"),
        ("robot", "tool_link_name"),
        ("robot", "reset", "default_joint_positions"),
        ("robot", "reset", "joint_noise_range"),
        ("robot", "capsules"),
        ("env", "time_step"),
        ("env", "gravity"),
        ("env", "control_dt"),
        ("env", "max_episode_steps"),
        ("env", "action_scale"),
        ("env", "fixed_beta"),
        ("env", "success_tolerance"),
        ("env", "workspace"),
        ("env", "observation", "space_bound"),
        ("env", "observation", "distance_clip"),
        ("env", "observation", "no_obstacle_distance"),
        ("env", "execution", "joint_motor_force"),
        ("env", "visual"),
        ("env", "obstacle", "enabled"),
        ("env", "obstacle", "scenario"),
        ("env", "obstacle", "radius"),
        ("env", "obstacle", "speed_range"),
        ("env", "obstacle", "disabled_position"),
        ("env", "obstacle", "bounds"),
        ("env", "obstacle", "random"),
        ("env", "obstacle", "scenarios"),
        ("env", "residual_control"),
        ("env", "goal", "fixed"),
        ("env", "goal", "position"),
        ("risk", "weights"),
        ("risk", "cost"),
        ("smoothing",),
        ("reward",),
        ("sac",),
        ("train",),
        ("eval",),
        ("smoke",),
    ]
    for path in required_paths:
        current: Any = config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                dotted = ".".join(path)
                raise KeyError(f"Missing required config key: {dotted}")
            current = current[key]

    robot_cfg = config["robot"]
    joint_count = len(robot_cfg["joint_names"])
    if len(config["robot"]["reset"]["default_joint_positions"]) != joint_count:
        raise ValueError("robot.reset.default_joint_positions must match robot joint count")

    if len(config["robot"]["capsules"]) <= 0:
        raise ValueError("robot.capsules must contain at least one capsule")
    for capsule in config["robot"]["capsules"]:
        if "parent_link_name" not in capsule or "child_link_name" not in capsule:
            raise KeyError("robot.capsules entries must define parent_link_name and child_link_name")
        has_start = "start_local_position" in capsule
        has_end = "end_local_position" in capsule
        if has_start != has_end:
            raise KeyError("robot.capsules local endpoints must be provided together")

    weights = config.get("device_selection", {})
    memory_weight = float(weights.get("memory_weight", 0.7))
    compute_weight = float(weights.get("compute_weight", 0.3))
    if memory_weight < 0 or compute_weight < 0:
        raise ValueError("device_selection.memory_weight and compute_weight must be non-negative")
    if memory_weight == 0 and compute_weight == 0:
        raise ValueError("At least one of device_selection.memory_weight or compute_weight must be positive")

    reward = config["reward"]
    terminal_radius = float(reward.get("terminal_goal_radius_m", 0.0))
    terminal_weight = float(reward.get("w_terminal_progress", 0.0))
    if not isfinite(terminal_radius) or terminal_radius < 0.0:
        raise ValueError("reward.terminal_goal_radius_m must be finite and non-negative")
    if not isfinite(terminal_weight):
        raise ValueError("reward.w_terminal_progress must be finite")

    residual_control = config["env"]["residual_control"]
    if not isinstance(residual_control, dict):
        raise TypeError("env.residual_control must be a mapping")
    for key in (
        "residual_scale",
        "base_speed_scale",
        "terminal_goal_radius_m",
        "terminal_gain",
        "waypoint_gain",
        "damping",
        "clearance_margin_m",
        "waypoint_lateral_margin_m",
        "waypoint_height_offset_m",
        "link_avoidance_activation_margin_m",
        "link_avoidance_max_speed_mps",
    ):
        value = float(residual_control.get(key, 0.0))
        if not isfinite(value):
            raise ValueError(f"env.residual_control.{key} must be finite")
    for key in (
        "residual_scale",
        "base_speed_scale",
        "terminal_goal_radius_m",
        "terminal_gain",
        "waypoint_gain",
        "damping",
        "clearance_margin_m",
        "waypoint_lateral_margin_m",
        "link_avoidance_activation_margin_m",
        "link_avoidance_max_speed_mps",
    ):
        if float(residual_control.get(key, 0.0)) < 0.0:
            raise ValueError(f"env.residual_control.{key} must be non-negative")

    sac = config["sac"]
    replay_size = int(sac["replay_size"])
    if replay_size <= 0:
        raise ValueError("sac.replay_size must be positive")
    stratified_fraction = float(sac.get("replay_stratified_fraction", 0.0))
    if not isfinite(stratified_fraction) or not 0.0 <= stratified_fraction <= 1.0:
        raise ValueError("sac.replay_stratified_fraction must be in [0, 1]")
    actor_anchor_weight = float(sac.get("actor_anchor_weight", 0.0))
    if not isfinite(actor_anchor_weight) or actor_anchor_weight < 0.0:
        raise ValueError("sac.actor_anchor_weight must be finite and non-negative")
    for key in ("resume_replay_warmup_steps", "resume_critic_warmup_steps"):
        if int(sac.get(key, 0)) < 0:
            raise ValueError(f"sac.{key} must be non-negative")

    focused_fraction = float(config["train"].get("focused_reset_fraction", 0.0))
    if not isfinite(focused_fraction) or not 0.0 <= focused_fraction <= 1.0:
        raise ValueError("train.focused_reset_fraction must be in [0, 1]")
    if focused_fraction > 0.0 and not config["train"].get("focused_reset_seed_manifest"):
        raise ValueError("train.focused_reset_seed_manifest is required when focused_reset_fraction > 0")

    focused_jitter = config["train"].get("focused_reset_jitter", {})
    if focused_jitter is not None:
        if not isinstance(focused_jitter, dict):
            raise TypeError("train.focused_reset_jitter must be a mapping when provided")
        for key in ("joint_noise_range_rad", "goal_radius_m", "obstacle_radius_m"):
            value = float(focused_jitter.get(key, 0.0))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"train.focused_reset_jitter.{key} must be finite and non-negative")
