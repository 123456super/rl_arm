"""Local passage semantics; no final-pose reward, planner or action labels."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pybullet as p

from rl_risk_sac.tasks.convex_sweep import ARM_GROUPS

LOCAL_CONTEXT_DIM = 27  # direction(24), remaining projected metres, length, time
LOCAL_OBSERVATION_DIM = 241 + LOCAL_CONTEXT_DIM
PROTOCOL = "s0_generated_local_passage_v1"
GEOMETRIC_PROTOCOL = "s0_instantaneous_geometric_passage_v1"


def geometric_mode(config):
    return config.get("local_avoidance", {}).get("reference_mode", "trajectory") == "instantaneous_geometry"


def local_protocol(config):
    return GEOMETRIC_PROTOCOL if geometric_mode(config) else PROTOCOL


def obstacle_count_probabilities(config):
    probabilities = np.asarray(config["local_avoidance"].get(
        "obstacle_count_probabilities", [0., 1.]), dtype=float)
    if (probabilities.ndim != 1 or len(probabilities) < 2
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or probabilities[0] != 0. or not np.isclose(probabilities.sum(), 1.)):
        raise ValueError("local obstacle count probabilities require P(0)=0, finite nonnegative entries and sum=1")
    return probabilities


def validate_local_config(config):
    from rl_risk_sac.tasks.local_path_verification import verification_settings
    verification_settings(config)
    local = config.get("local_avoidance", {})
    if not local.get("enabled", False):
        raise ValueError("local_avoidance.enabled must be true")
    if local.get("reference_mode", "trajectory") not in ("trajectory", "instantaneous_geometry"):
        raise ValueError("unknown local reference_mode")
    if geometric_mode(config):
        for key in ("kinematic_dt", "prediction_dt", "minimum_point_speed", "minimum_closing_cosine"):
            value = local.get("geometry", {}).get(key)
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"local_avoidance.geometry.{key} must be finite and positive")
        if (local["geometry"]["prediction_dt"] > config["thesis"]["control_dt"]
                or local["geometry"]["kinematic_dt"] > local["geometry"]["prediction_dt"]
                or local["geometry"]["minimum_closing_cosine"] >= 1):
            raise ValueError("geometric prediction requires 0 < kinematic_dt <= prediction_dt <= control_dt and cosine < 1")
        if config["thesis"].get("self_safety_projection", {}).get("enabled", False):
            raise ValueError("geometric intent currently requires the unprojected SAC execution contract")
    if (not config.get("dual_sac", {}).get("enabled", False)
            or config["dual_sac"].get("training_policy") != "avoidance"):
        raise ValueError("local passage requires the dual avoidance SAC")
    obstacles = config["thesis"]["obstacles"]
    if obstacles.get("dynamic", False) or config["thesis"]["stage"] != "s1":
        raise ValueError("local passage supports static obstacles in S1 only")
    obstacle_count_probabilities(config)
    safe = float(obstacles["risk"]["d_safe"])
    if not 0 < safe < local["release_distance_m"] <= local["trigger_distance_m"]:
        raise ValueError("local distances require 0 < safe < release <= trigger")
    for key in ("horizon", "recovery_steps", "nominal_max_steps", "obstacle_proposals_per_trace",
                "minimum_approach_steps", "replay_capacity", "batch_size", "update_after", "updates_per_step"):
        if int(local[key]) != local[key] or local[key] < 1:
            raise ValueError(f"local_avoidance.{key} must be a positive integer")
    for key in ("minimum_reference_motion_m", "recovery_progress", "obstacle_offset_std_m"):
        if not np.isfinite(local[key]) or local[key] <= 0:
            raise ValueError(f"local_avoidance.{key} must be finite and positive")
    if local["warmup_steps"] < 0:
        raise ValueError("warmup_steps must be nonnegative")
    for key, value in local["reward"].items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid local reward coefficient: {key}")


def arm_vector(env):
    capsules = {capsule.name: capsule for capsule in env.robot.capsules()}
    return np.concatenate([np.r_[capsules[name].start, capsules[name].end]
                           for name in ARM_GROUPS]).astype(np.float64)


def passage_progress(vector, sample):
    return float((np.asarray(vector) - sample["entry_arm"]) @ sample["direction"] / sample["length"])


def local_context(vector, sample, steps, horizon):
    progress = passage_progress(vector, sample)
    return np.r_[sample["direction"], sample["length"] * (1. - progress),
                 sample["length"], max(0., 1. - steps / horizon)].astype(np.float32)


def passage_candidate(progress, external_distance, local):
    return bool(progress >= 1. and external_distance >= local["release_distance_m"])


def local_reward(previous_progress, progress, external_risk, self_risk,
                 smoothness, success, failure, timeout, coefficients):
    # Clip the potential, not each increment: undiscounted forward/backward
    # cycles cancel instead of harvesting asymmetric clips. No final bonus.
    delta = float(np.clip(progress, -1., 2.) - np.clip(previous_progress, -1., 2.))
    return float(
        coefficients["progress_scale"] * delta
        - coefficients["external_risk_scale"] * np.clip(external_risk, 0., 1.)
        - coefficients["self_risk_scale"] * np.clip(self_risk, 0., 1.)
        - coefficients["time_cost"]
        - coefficients["smooth_cost"] * np.clip(smoothness, 0., 1.)
        + coefficients["success_bonus"] * bool(success)
        - coefficients["failure_penalty"] * bool(failure)
        - coefficients["timeout_penalty"] * bool(timeout)
    )


class MeshClearance:
    """Original meshes, sampled at the start and every 240 Hz physics step."""
    def __init__(self, env):
        self.env = env
        self.external = self.self_distance = float("inf")
        self.checks = 0
        self()

    def __call__(self):
        env = self.env
        self.external = min([self.external] + [env._static_clearance(o) for o in env.obstacles])
        for first, second in env._self_pair_link_ids:
            points = env._self_pair_points(first, second)
            self.self_distance = min(self.self_distance,
                                     min((float(q[8]) for q in points), default=env.self_query_distance))
        self.checks += 1

    @property
    def collision(self):
        return self.external <= 0 or self.self_distance <= 0


def actor_digest(actor):
    digest = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        digest.update(name.encode())
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def contract_digest(config):
    # Offline path-search budgets select samples; they do not change the MDP.
    # Their values and checking resolution are recorded in each witness instead.
    thesis = config["thesis"]
    contract = {"protocol": local_protocol(config), "robot": config["robot"],
                "local_avoidance": config["local_avoidance"],
                "thesis": {key: thesis.get(key) for key in (
                    "physics_dt", "control_dt", "action_scale", "joint_motor_force",
                    "precision_control", "observation", "self_collision", "self_safety_projection",
                    "qdot_observation_scale", "ee_linear_velocity_scale", "ee_angular_velocity_scale",
                    "joint_pose_curriculum", "orientation_curriculum", "obstacles")}}
    # Path strings alone do not identify collision geometry or normalization.
    files = [config["robot"]["urdf"], thesis["obstacles"]["predictor_checkpoint"]]
    from pathlib import Path
    robot_dir = Path(config["robot"]["urdf"]).parent
    files.extend(str(path) for path in sorted((robot_dir / "meshes").rglob("*.stl")))
    content = {}
    for filename in files:
        content[filename] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    contract["file_hashes"] = content
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
