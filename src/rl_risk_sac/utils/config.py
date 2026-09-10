from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """加载并校验实验配置。

    配置文件支持 includes 递归组合：默认配置可以引用 robot/env/algo/run
    子配置，实验 YAML 再覆盖其中少量字段。
    """
    config_path = Path(path)
    config = _load_config_with_includes(config_path)
    validate_config(config)
    return config


def _load_config_with_includes(path: Path) -> dict[str, Any]:
    """递归读取 YAML，并让当前文件覆盖被 include 的基础配置。"""
    with open(path, "r", encoding="utf-8") as file:
        current = yaml.safe_load(file) or {}

    merged: dict[str, Any] = {}
    includes = current.pop("includes", []) or []
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            # include 使用相对路径时，以当前 YAML 所在目录为基准。
            include_path = path.parent / include_path
        deep_update(merged, _load_config_with_includes(include_path))

    # 当前文件优先级最高，所以最后 merge。
    deep_update(merged, current)
    return merged


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """递归合并字典，保留未被覆盖的嵌套配置。"""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def validate_config(config: dict[str, Any]) -> None:
    """对训练/评估所需配置做早期校验。

    这里的目标不是检查每个字段的物理合理性，而是尽早发现缺键、
    维度不一致、非法模式等会导致训练半路崩掉的问题。
    """
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
        ("env", "observation", "schema_version"),
        ("env", "observation", "distance_clip"),
        ("env", "observation", "no_obstacle_distance"),
        ("env", "execution", "joint_motor_force"),
        ("env", "execution", "max_policy_velocity_delta"),
        ("env", "execution", "fixed_smoothing_mode"),
        ("env", "execution", "rtb", "cutoff_angular_frequency"),
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
        ("env", "goal", "mode"),
        ("env", "goal", "position"),
        ("env", "goal", "speed_range"),
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
    # reset 初始关节角必须和可控关节数量一致，否则 PyBullet 重置时会错位。
    if len(config["robot"]["reset"]["default_joint_positions"]) != joint_count:
        raise ValueError("robot.reset.default_joint_positions must match robot joint count")

    # 胶囊体是连杆级风险的基础，没有 capsule 就无法构造 observation。
    if len(config["robot"]["capsules"]) <= 0:
        raise ValueError("robot.capsules must contain at least one capsule")
    for capsule in config["robot"]["capsules"]:
        if "parent_link_name" not in capsule or "child_link_name" not in capsule:
            raise KeyError("robot.capsules entries must define parent_link_name and child_link_name")

    time_step = float(config["env"]["time_step"])
    control_dt = float(config["env"]["control_dt"])
    if time_step <= 0.0 or control_dt <= 0.0:
        raise ValueError("env.time_step and env.control_dt must be positive")
    substep_ratio = control_dt / time_step
    if not np.isclose(substep_ratio, round(substep_ratio), rtol=0.0, atol=1e-6):
        raise ValueError("env.control_dt must be an integer multiple of env.time_step")

    risk_weights = config["risk"]["weights"]
    weight_values = [float(risk_weights[key]) for key in ("distance", "velocity", "ttc")]
    if any(value < 0.0 for value in weight_values):
        raise ValueError("risk.weights must be non-negative")
    if not np.isclose(sum(weight_values), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("risk.weights must sum to 1")

    execution_cfg = config["env"]["execution"]
    fixed_smoothing_mode = str(execution_cfg["fixed_smoothing_mode"])
    # 固定方法目前只支持历史 EMA 和论文式 Butterworth+quintic RTB。
    if fixed_smoothing_mode not in {"ema", "butterworth_quintic"}:
        raise ValueError("env.execution.fixed_smoothing_mode must be 'ema' or 'butterworth_quintic'")
    if float(execution_cfg["rtb"]["cutoff_angular_frequency"]) <= 0.0:
        raise ValueError("env.execution.rtb.cutoff_angular_frequency must be positive")
    max_delta = execution_cfg["max_policy_velocity_delta"]
    if max_delta is not None and float(max_delta) <= 0.0:
        raise ValueError("env.execution.max_policy_velocity_delta must be positive or null")
    safety_qp_cfg = execution_cfg.get("safety_qp", {})
    trajectory_mode = str(safety_qp_cfg.get("trajectory_mode", "post_qp_rtb"))
    if trajectory_mode not in {"post_qp_rtb", "filtered_endpoint_qp"}:
        raise ValueError(
            "env.execution.safety_qp.trajectory_mode must be 'post_qp_rtb' or 'filtered_endpoint_qp'"
        )
    motion_bounds = safety_qp_cfg.get("motion_bounds", {})
    for key in ("max_acceleration", "max_jerk"):
        value = motion_bounds.get(key)
        if value is not None and float(value) <= 0.0:
            raise ValueError(f"env.execution.safety_qp.motion_bounds.{key} must be positive or null")
    if any(motion_bounds.get(key) is not None for key in ("max_acceleration", "max_jerk")):
        if not bool(safety_qp_cfg.get("enabled", False)):
            raise ValueError("safety QP must be enabled when motion bounds are configured")
        if trajectory_mode != "filtered_endpoint_qp":
            raise ValueError("motion bounds require safety_qp.trajectory_mode=filtered_endpoint_qp")

    goal_cfg = config["env"]["goal"]
    # static 用于普通到达任务；linear_bounce 用于动态目标跟踪实验。
    if str(goal_cfg["mode"]) not in {"static", "linear_bounce"}:
        raise ValueError("env.goal.mode must be 'static' or 'linear_bounce'")
    if not _valid_range(goal_cfg["speed_range"], lower_bound=0.0):
        raise ValueError("env.goal.speed_range must contain two non-negative bounds")
    obstacle_cfg = config["env"]["obstacle"]
    if not _valid_range(obstacle_cfg["speed_range"], lower_bound=0.0):
        raise ValueError("env.obstacle.speed_range must contain two non-negative bounds")
    if int(obstacle_cfg.get("count", 1)) <= 0:
        raise ValueError("env.obstacle.count must be positive when provided")
    episode_enable_probability = float(obstacle_cfg.get("episode_enable_probability", 1.0))
    if not 0.0 <= episode_enable_probability <= 1.0:
        raise ValueError("env.obstacle.episode_enable_probability must be in [0, 1]")

    predictive_risk_penalty = float(config["sac"].get("predictive_risk_penalty", 0.0))
    if predictive_risk_penalty < 0.0:
        raise ValueError("sac.predictive_risk_penalty must be non-negative")
    predictive_penalty_mode = str(config["sac"].get("predictive_risk_penalty_mode", "raw"))
    if predictive_penalty_mode not in {"raw", "excess"}:
        raise ValueError("sac.predictive_risk_penalty_mode must be 'raw' or 'excess'")
    if float(config["risk"].get("geometry_margin", 0.0)) < 0.0:
        raise ValueError("risk.geometry_margin must be non-negative")

    weights = config.get("device_selection", {})
    memory_weight = float(weights.get("memory_weight", 0.7))
    compute_weight = float(weights.get("compute_weight", 0.3))
    if memory_weight < 0 or compute_weight < 0:
        raise ValueError("device_selection.memory_weight and compute_weight must be non-negative")
    if memory_weight == 0 and compute_weight == 0:
        raise ValueError("At least one of device_selection.memory_weight or compute_weight must be positive")


def _valid_range(value: Any, lower_bound: float | None = None) -> bool:
    """检查形如 [low, high] 的采样范围是否合法。"""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    low, high = float(value[0]), float(value[1])
    if high < low:
        return False
    if lower_bound is not None and low < lower_bound:
        return False
    return True
