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
        ("env", "hierarchical_control"),
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

    hierarchical = config["env"]["hierarchical_control"]
    if not isinstance(hierarchical, dict):
        raise TypeError("env.hierarchical_control must be a mapping")
    if bool(hierarchical.get("enabled", False)):
        if bool(residual_control.get("enabled", True)):
            raise ValueError("env.hierarchical_control and legacy env.residual_control cannot both be enabled")
        safety_filter = config["env"].get("safety_filter", {})
        if not bool(safety_filter.get("enabled", False)):
            raise ValueError("env.hierarchical_control requires env.safety_filter.enabled=true")
        for section_name in ("train", "eval", "smoke"):
            method = config.get(section_name, {}).get("method")
            if method != "hierarchical_residual":
                raise ValueError(
                    f"{section_name}.method must be hierarchical_residual when hierarchical control is enabled"
                )
    for key in ("hold_margin_m", "release_margin_m"):
        value = float(hierarchical.get(key, 0.0))
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"env.hierarchical_control.{key} must be finite and non-negative")
    if float(hierarchical.get("release_margin_m", 0.0)) <= float(hierarchical.get("hold_margin_m", 0.0)):
        raise ValueError("env.hierarchical_control.release_margin_m must exceed hold_margin_m")
    for key in ("replan_after_hold_steps", "plan_retry_interval_steps"):
        if int(hierarchical.get(key, 0)) <= 0:
            raise ValueError(f"env.hierarchical_control.{key} must be positive")
    planner = hierarchical.get("planner", {})
    tracker = hierarchical.get("tracker", {})
    hierarchical_residual = hierarchical.get("residual", {})
    if not all(isinstance(section, dict) for section in (planner, tracker, hierarchical_residual)):
        raise TypeError("hierarchical planner, tracker and residual sections must be mappings")
    architecture_version = hierarchical.get("architecture_version")
    if architecture_version is not None and not isinstance(architecture_version, str):
        raise TypeError("env.hierarchical_control.architecture_version must be a string")
    for key in ("ik_attempts", "ik_max_iterations", "max_iterations"):
        if int(planner.get(key, 0)) <= 0:
            raise ValueError(f"env.hierarchical_control.planner.{key} must be positive")
    for key in (
        "ik_residual_threshold",
        "ik_goal_tolerance_m",
        "clearance_m",
        "step_size_rad",
        "edge_resolution_rad",
        "goal_sample_probability",
    ):
        value = float(planner.get(key, 0.0))
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"env.hierarchical_control.planner.{key} must be finite and non-negative")
    if float(planner.get("step_size_rad", 0.0)) <= 0.0 or float(planner.get("edge_resolution_rad", 0.0)) <= 0.0:
        raise ValueError("hierarchical planner step and edge resolutions must be positive")
    probability = float(planner.get("goal_sample_probability", 0.0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("hierarchical planner goal_sample_probability must be in [0, 1]")
    if str(planner.get("candidate_selection", "first_success")) not in {
        "first_success",
        "clearance_then_length",
        "length_then_clearance",
        "predictive_clearance_then_length",
    }:
        raise ValueError("hierarchical planner candidate_selection is invalid")
    for key in (
        "boundary_start_tolerance_m",
        "dls_max_iterations",
        "dls_damping",
        "dls_step_size",
        "dls_goal_tolerance_m",
        "ik_fallback_attempts",
        "ik_fallback_dls_seeds",
        "predictive_candidate_limit",
        "ik_target_shell_attempts",
        "ik_target_shell_radius_m",
        "ik_goal_region_attempts",
        "ik_goal_region_radius_m",
        "ik_obstacle_aware_nullspace_attempts",
        "ik_obstacle_aware_nullspace_step_rad",
    ):
        value = float(planner.get(key, 0.0))
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"env.hierarchical_control.planner.{key} must be finite and non-negative")
    if int(planner.get("predictive_candidate_limit", 1)) <= 0:
        raise ValueError("hierarchical planner predictive_candidate_limit must be positive")
    if int(planner.get("ik_target_shell_attempts", 0)) < 0:
        raise ValueError("hierarchical planner ik_target_shell_attempts must be non-negative")
    if int(planner.get("ik_goal_region_attempts", 0)) < 0:
        raise ValueError("hierarchical planner ik_goal_region_attempts must be non-negative")
    shell_radius = float(planner.get("ik_target_shell_radius_m", 0.0))
    if shell_radius > float(planner.get("ik_goal_tolerance_m", 0.0)):
        raise ValueError("hierarchical planner ik_target_shell_radius_m must not exceed ik_goal_tolerance_m")
    goal_region_radius = float(planner.get("ik_goal_region_radius_m", 0.0))
    if goal_region_radius > float(planner.get("ik_goal_tolerance_m", 0.0)):
        raise ValueError("hierarchical planner ik_goal_region_radius_m must not exceed ik_goal_tolerance_m")
    if "ik_target_shell_enabled" in planner and not isinstance(planner["ik_target_shell_enabled"], bool):
        raise TypeError("hierarchical planner ik_target_shell_enabled must be boolean")
    if "ik_goal_region_enabled" in planner and not isinstance(planner["ik_goal_region_enabled"], bool):
        raise TypeError("hierarchical planner ik_goal_region_enabled must be boolean")
    for key in ("ik_obstacle_aware_nullspace_enabled",):
        if key in planner and not isinstance(planner[key], bool):
            raise TypeError(f"env.hierarchical_control.planner.{key} must be boolean")
    for key in (
        "waypoint_gain", "waypoint_tolerance_rad", "servo_trigger_m", "servo_gain", "servo_damping",
        "servo_velocity_damping", "servo_near_goal_radius_m", "servo_near_goal_gain_scale",
        "filter_intervention_threshold", "filter_aware_gain_floor", "filter_aware_blend_strength",
        "filter_aware_min_alignment",
        "servo_stall_improvement_m", "servo_stall_filter_ratio",
    ):
        value = float(tracker.get(key, 0.0))
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"env.hierarchical_control.tracker.{key} must be finite and non-negative")
    for key in ("servo_damping_near", "servo_damping_far", "adaptive_damping_error_m", "nullspace_gain", "clearance_gain_low_m", "clearance_gain_high_m"):
        value = float(tracker.get(key, 0.0))
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"env.hierarchical_control.tracker.{key} must be finite and non-negative")
    for key in ("adaptive_damping", "nullspace_limit_avoidance", "clearance_aware_gain", "filter_aware_servo"):
        if key in tracker and not isinstance(tracker[key], bool):
            raise TypeError(f"env.hierarchical_control.tracker.{key} must be boolean")
    if float(tracker.get("filter_aware_gain_floor", 0.35)) > 1.0 or float(tracker.get("servo_near_goal_gain_scale", 0.65)) > 1.0:
        raise ValueError("tracker gain scales must be at most 1")
    if float(tracker.get("filter_aware_min_alignment", 0.0)) < -1.0 or float(tracker.get("filter_aware_min_alignment", 0.0)) > 1.0:
        raise ValueError("tracker filter_aware_min_alignment must be in [-1, 1]")
    for key in ("servo_stall_steps", "servo_stall_replan_limit"):
        if int(tracker.get(key, 0)) < 0:
            raise ValueError(f"env.hierarchical_control.tracker.{key} must be non-negative")
    risk_start = float(hierarchical_residual.get("risk_start_m", 0.0))
    risk_stop = float(hierarchical_residual.get("risk_stop_m", 0.0))
    residual_scale = float(hierarchical_residual.get("residual_scale", 0.0))
    if not all(isfinite(value) and value >= 0.0 for value in (risk_start, risk_stop, residual_scale)):
        raise ValueError("hierarchical residual values must be finite and non-negative")
    if risk_start <= risk_stop:
        raise ValueError("hierarchical residual risk_start_m must exceed risk_stop_m")

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
