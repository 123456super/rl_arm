from __future__ import annotations

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

    weights = config.get("device_selection", {})
    memory_weight = float(weights.get("memory_weight", 0.7))
    compute_weight = float(weights.get("compute_weight", 0.3))
    if memory_weight < 0 or compute_weight < 0:
        raise ValueError("device_selection.memory_weight and compute_weight must be non-negative")
    if memory_weight == 0 and compute_weight == 0:
        raise ValueError("At least one of device_selection.memory_weight or compute_weight must be positive")
