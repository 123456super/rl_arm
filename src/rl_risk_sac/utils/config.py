"""Single loader and validator for the active thesis PPO configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rl_risk_sac.utils.runtime_config import RuntimeConfig


def load_config(path: str | Path) -> dict[str, Any]:
    config = _load_config_with_includes(Path(path))
    validate_config(config)
    return config


def _load_config_with_includes(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        current = yaml.safe_load(file) or {}
    merged: dict[str, Any] = {}
    for include in current.pop("includes", []) or []:
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


def _require(config: dict[str, Any], *paths: tuple[str, ...]) -> None:
    for path in paths:
        current: Any = config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(f"Missing required config key: {'.'.join(path)}")
            current = current[key]


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    valid = np.isfinite(number) and (number >= 0.0 if allow_zero else number > 0.0)
    if not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return number


def validate_config(config: dict[str, Any]) -> None:
    """Validate only fields that affect the current S0/S1/S2 pipeline."""
    allowed = {"seed", "device", "device_selection", "robot", "thesis", "ppo", "train"}
    unknown = set(config) - allowed
    if unknown:
        raise KeyError(f"obsolete or unknown top-level config sections: {sorted(unknown)}")
    _require(
        config,
        ("seed",), ("device",), ("device_selection",), ("robot",),
        ("thesis",), ("ppo",), ("train",),
        ("robot", "urdf"), ("robot", "base_position"),
        ("robot", "joint_names"), ("robot", "tool_link_name"),
        ("robot", "reset", "default_joint_positions"), ("robot", "capsules"),
        ("thesis", "physics_dt"), ("thesis", "control_dt"),
        ("thesis", "horizon"), ("thesis", "action_mode"),
        ("thesis", "action_limit_rad"), ("thesis", "joint_delta_limit_rad"),
        ("thesis", "reference_ppo"), ("thesis", "reward"),
        ("thesis", "observation", "obstacle_set"), ("thesis", "scene_mixture"),
        ("thesis", "simple_reaching_mode"), ("thesis", "blocking_obstacle_mode"),
        ("thesis", "stage"), ("thesis", "terminate_on_collision"),
        ("thesis", "obstacle_geometry_check_interval_substeps"),
        ("thesis", "joint_target_margin_rad"),
        ("thesis", "joint_limit_tolerance_rad"),
        ("thesis", "qdot_observation_scale"), ("thesis", "success_hold_steps"),
        ("thesis", "ee_linear_velocity_scale"),
        ("thesis", "ee_angular_velocity_scale"),
        ("thesis", "obstacle_half_extents_min_m"),
        ("thesis", "obstacle_half_extents_max_m"),
        ("thesis", "obstacle_speed"), ("thesis", "d_safe"),
        ("thesis", "position_tolerance"), ("thesis", "orientation_tolerance"),
        ("thesis", "goal_sample_max_attempts"),
        ("thesis", "scene_sample_max_attempts"),
        ("thesis", "dynamic_motion"), ("thesis", "dynamic_prediction"),
        ("thesis", "self_collision"),
        ("thesis", "fixed_orientation_goal"),
        ("thesis", "static_obstacle_count_probabilities"),
        ("thesis", "hybrid_static_obstacle_count"),
        ("thesis", "static_obstacle_layout_probabilities"),
        ("thesis", "path_offset_min_m"), ("thesis", "path_offset_max_m"),
        ("thesis", "obstacle_surface_clearance_min_m"),
        ("thesis", "structured_offset_min_m"),
        ("thesis", "structured_offset_max_m"),
        ("ppo", "gamma"), ("ppo", "gae_lambda"), ("ppo", "clip_ratio"),
        ("ppo", "epochs"), ("ppo", "minibatch_size"),
        ("ppo", "actor_lr"), ("ppo", "critic_lr"), ("ppo", "hidden_dims"),
        ("ppo", "entropy_coef"), ("ppo", "value_coef"),
        ("ppo", "normalize_advantage"), ("ppo", "advantage_clip"),
        ("ppo", "max_grad_norm"), ("ppo", "target_kl"),
        ("ppo", "max_abs_log_ratio"), ("ppo", "action_std_init"),
        ("ppo", "lr_decay"), ("ppo", "actor_lr_final"),
        ("ppo", "critic_lr_final"), ("ppo", "adaptive_kl_lr"),
        ("ppo", "actor_std_lr_adaptation"),
        ("ppo", "actor_std_lr_ema_decay"), ("ppo", "actor_lr_min"),
        ("train", "num_envs"), ("train", "rollout_length"),
        ("train", "save_interval"), ("train", "update_log_interval"),
        ("train", "stage_transfer_action_std"),
        ("train", "output_dir"),
    )
    runtime = RuntimeConfig.from_mapping(config)
    if len(runtime.robot.joint_names) != len(runtime.robot.reset.default_joint_positions):
        raise ValueError("robot reset positions must match robot joint count")
    if not runtime.robot.capsules:
        raise ValueError("robot.capsules must not be empty")
    for capsule in runtime.robot.capsules:
        _positive(capsule.radius, f"robot capsule {capsule.name} radius")
    selection = runtime.device_selection
    _positive(selection.min_free_memory_gb, "device_selection.min_free_memory_gb", allow_zero=True)
    _positive(selection.memory_weight, "device_selection.memory_weight", allow_zero=True)
    _positive(selection.compute_weight, "device_selection.compute_weight", allow_zero=True)
    if selection.memory_weight == 0.0 and selection.compute_weight == 0.0:
        raise ValueError("device selection weights cannot both be zero")

    thesis = config["thesis"]
    obsolete_thesis = {
        "protocol", "self_safety_projection", "joint_pose_curriculum",
    } & set(thesis)
    if obsolete_thesis:
        raise KeyError(
            f"obsolete thesis config fields: {sorted(obsolete_thesis)}"
        )
    stage = str(thesis["stage"])
    if stage not in {"s0", "s1", "s2"}:
        raise ValueError("thesis.stage must be one of s0, s1, s2")
    if not bool(thesis["fixed_orientation_goal"]["enabled"]):
        raise ValueError(
            "the active S0/S1/S2 pipeline requires fixed_orientation_goal.enabled=true"
        )
    if bool(thesis["simple_reaching_mode"]) != (stage == "s0"):
        raise ValueError(
            "simple_reaching_mode must be true only for the S0 configuration"
        )
    physics_dt = _positive(thesis["physics_dt"], "thesis.physics_dt")
    control_dt = _positive(thesis["control_dt"], "thesis.control_dt")
    ratio = control_dt / physics_dt
    if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1.0e-6):
        raise ValueError("thesis.control_dt must be an integer multiple of physics_dt")
    if str(thesis["action_mode"]) != "joint_delta":
        raise ValueError("only thesis.action_mode=joint_delta is supported")
    for name in ("action_limit_rad", "joint_delta_limit_rad", "horizon"):
        _positive(thesis[name], f"thesis.{name}")
    for stage in ("s0", "s1", "s2"):
        values = thesis["scene_mixture"].get(stage)
        if not isinstance(values, dict) or not values:
            raise ValueError(f"thesis.scene_mixture.{stage} must be non-empty")
        if set(values) - {"none", "static", "dynamic", "hybrid"}:
            raise ValueError(f"invalid scene in thesis.scene_mixture.{stage}")
        probabilities = np.asarray(list(values.values()), dtype=np.float64)
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or probabilities.sum() <= 0.0:
            raise ValueError(f"invalid probabilities in thesis.scene_mixture.{stage}")
    if int(thesis["scene_sample_max_attempts"]) < 1:
        raise ValueError("thesis.scene_sample_max_attempts must be positive")
    max_obstacles = int(thesis["observation"]["obstacle_set"]["max_obstacles"])
    hybrid_static_count = int(thesis["hybrid_static_obstacle_count"])
    if hybrid_static_count < 1 or hybrid_static_count + 1 > max_obstacles:
        raise ValueError(
            "thesis.hybrid_static_obstacle_count must leave room for one "
            "dynamic obstacle"
        )

    ppo = config["ppo"]
    if not 0.0 < float(ppo["gamma"]) <= 1.0 or not 0.0 < float(ppo["gae_lambda"]) <= 1.0:
        raise ValueError("ppo.gamma and ppo.gae_lambda must be in (0, 1]")
    for name in ("clip_ratio", "actor_lr", "critic_lr"):
        _positive(ppo[name], f"ppo.{name}")
    if int(ppo["epochs"]) < 1 or int(ppo["minibatch_size"]) < 1:
        raise ValueError("PPO epochs and minibatch size must be positive")
    if len(ppo["hidden_dims"]) != 2 or any(int(value) < 1 for value in ppo["hidden_dims"]):
        raise ValueError("ppo.hidden_dims must contain two positive widths")

    train = config["train"]
    for name in ("num_envs", "rollout_length", "save_interval", "update_log_interval"):
        if int(train[name]) < 1:
            raise ValueError(f"train.{name} must be positive")
    _positive(train["stage_transfer_action_std"], "train.stage_transfer_action_std")

    dynamic = thesis.get("dynamic_prediction", {})
    if dynamic.get("enabled", False):
        _require(
            config,
            ("thesis", "dynamic_prediction", "mode"),
            ("thesis", "dynamic_prediction", "checkpoint"),
            ("thesis", "dynamic_prediction", "history_sample_hz"),
            ("thesis", "dynamic_prediction", "history_length"),
            ("thesis", "dynamic_prediction", "horizon_fractions"),
            ("thesis", "dynamic_prediction", "training"),
        )
        if dynamic["mode"] != "lstm":
            raise ValueError("only LSTM dynamic prediction is supported")
        _positive(dynamic["history_sample_hz"], "dynamic history frequency")
        if int(dynamic["history_length"]) < 2:
            raise ValueError("dynamic history length must be at least two")
        training = dynamic["training"]
        for name in ("train_samples", "validation_samples", "test_samples", "epochs", "batch_size", "hidden_size", "num_layers"):
            if int(training[name]) < 1:
                raise ValueError(f"dynamic_prediction.training.{name} must be positive")
        _positive(training["learning_rate"], "dynamic LSTM learning rate")
        checkpoint_parent = Path(str(dynamic["checkpoint"])).parent
        if Path(str(training["output_dir"])) != checkpoint_parent:
            raise ValueError(
                "dynamic LSTM training.output_dir must equal the checkpoint parent"
            )
