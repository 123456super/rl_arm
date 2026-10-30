"""Small, independent configuration contract for the FTP4RM-inspired experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rl_risk_sac.utils.config import _load_config_with_includes
from rl_risk_sac.utils.runtime_config import (
    CapsuleSpec, DeviceSelectionConfig, RobotResetConfig, RobotRuntimeConfig,
)

PAPER_PROTOCOL = "paper_single_sac_v2"
ROOT = Path(__file__).resolve().parents[3]


def load_paper_config(path):
    config = _load_config_with_includes(Path(path))
    validate_paper_config(config)
    return config


def validate_paper_config(config):
    if config.get("protocol") != PAPER_PROTOCOL:
        raise ValueError("requires paper_single_sac_v2; v1 has different actions, networks and success criteria")
    task, sac, train = config["paper"], config["sac"], config["train"]
    if not np.isfinite(task["gravity"]) or task["gravity"] > 0:
        raise ValueError("gravity must be finite and nonpositive")
    if int(task["obstacles"]["count"]) != 3 or len(config["robot"]["joint_names"]) != 6:
        raise ValueError("the reference observation requires six joints and three obstacles")
    for mapping, keys in ((task, ("physics_dt", "action_repeat", "horizon", "action_limit",
                                  "position_gain", "velocity_gain")),
                          (sac, ("actor_lr", "critic_lr", "alpha_lr", "initial_alpha",
                                 "batch_size", "replay_capacity", "actor_update_interval", "updates_per_transition")),
                          (train, ("total_steps", "num_envs", "progress_interval", "save_interval", "diagnostics_interval"))):
        for key in keys:
            if not np.isfinite(mapping[key]) or float(mapping[key]) <= 0:
                raise ValueError(f"{key} must be finite and positive")
    for value in (task["joint_motor_force"], sac["gradient_clip"]):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError("motor force/gradient clip must be null or finite and positive")
    if task["action_mode"] != "initial_offset":
        raise ValueError("paper_single_sac_v2 requires the author's initial_offset SAC action interface")
    if sac["activation"] != "relu":
        raise ValueError("paper_single_sac_v2 uses ReLU hidden layers for SAC")
    for mapping, keys in ((task, ("action_repeat", "horizon")),
                          (sac, ("batch_size", "replay_capacity", "actor_update_interval", "warmup_steps", "update_after")),
                          (train, ("total_steps", "num_envs", "progress_interval", "save_interval", "diagnostics_interval"))):
        if any(int(mapping[k]) != mapping[k] or int(mapping[k]) < 0 for k in keys):
            raise ValueError("step counts and sizes must be integers")
    if not 0 < sac["gamma"] < 1 or not 0 < sac["tau"] <= 1:
        raise ValueError("invalid SAC discount or target update coefficient")
    if not np.isfinite(sac["target_entropy"]) or sac["target_entropy"] >= 0:
        raise ValueError("target_entropy must be finite and negative")
    if sac["batch_size"] > sac["replay_capacity"] or not sac["hidden_dims"] or any(
            int(v) != v or v <= 0 for v in sac["hidden_dims"]):
        raise ValueError("invalid replay or network size")
    goal, obstacles, reward = task["goal"], task["obstacles"], task["reward"]
    if not 0 < goal["base_exclusion_radius"] < goal["workspace_radius"] or not (
            0 < goal["min_z"] < goal["workspace_radius"]):
        raise ValueError("invalid goal workspace")
    if not 0 < obstacles["radius"] < obstacles["shell_inner_radius"] < obstacles["shell_outer_radius"]:
        raise ValueError("invalid obstacle size/shell")
    if (len(goal["rpy"]) != 3 or not np.isfinite(goal["rpy"]).all()
            or min(goal["ik_position_tolerance"], goal["ik_orientation_tolerance"], goal["max_attempts"],
                   goal["ik_max_iterations"], goal["ik_refine_iterations"], obstacles["max_attempts"])
            <= 0 or obstacles["endpoint_clearance"] < 0):
        raise ValueError("invalid pose or sampling checks")
    if (len(config["robot"]["reset"]["default_joint_positions"]) != 6
            or not np.isfinite(config["robot"]["reset"]["default_joint_positions"]).all()
            or config["robot"]["reset"]["joint_noise_range"] < 0):
        raise ValueError("invalid six-joint reset pose")
    if not set(task["distance_links"]).issubset({c["name"] for c in config["robot"]["capsules"]}):
        raise ValueError("distance_links contains an unknown robot link group")
    if not task["distance_links"] or any(not np.isfinite(v) or v < 0 for v in reward.values()):
        raise ValueError("invalid distance links/reward parameters")
    if min(reward["error_log_offset"], reward["obstacle_distance"]) <= 0:
        raise ValueError("reward logarithm offset and distance range must be positive")
    if any(not np.isfinite(v) or v <= 0 for v in task["success"].values()):
        raise ValueError("success tolerances must be finite and positive")
    if task["success"]["strict_combined_error_tolerance"] > task["success"]["combined_error_tolerance"]:
        raise ValueError("strict success tolerance cannot exceed the main tolerance")
    ae = sac["action_ensemble"]
    if (not np.isfinite([ae["shape_growth"], ae["scale_growth"]]).all()
            or min(ae["shape_growth"], ae["scale_growth"]) < 0
            or any(int(ae[k]) != ae[k] or ae[k] < 1 for k in ("count_draws", "total_episodes"))):
        raise ValueError("invalid action ensemble schedule")
    if any(not np.isfinite(value) or value <= 0 for value in config["planner"].values()):
        raise ValueError("planner parameters must be finite and positive")
    if not robot_urdf(config).is_file():
        raise FileNotFoundError(robot_urdf(config))


def robot_urdf(config):
    path = Path(config["robot"]["urdf"])
    return path if path.is_absolute() else ROOT / path


def robot_runtime(config):
    robot = config["robot"]
    return RobotRuntimeConfig(
        urdf=robot["urdf"], base_position=tuple(robot["base_position"]),
        joint_names=tuple(robot["joint_names"]), tool_link_name=robot["tool_link_name"],
        reset=RobotResetConfig(tuple(robot["reset"]["default_joint_positions"]),
                               float(robot["reset"]["joint_noise_range"])),
        capsules=tuple(CapsuleSpec(**entry) for entry in robot["capsules"]),
    )


def device_selection(config):
    return DeviceSelectionConfig(**config["device_selection"])


def experiment_signature(config):
    """Budget/logging/device may change; the task, optimizer and AE schedule may not."""
    payload = {key: config[key] for key in ("protocol", "robot", "paper", "sac", "seed")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
