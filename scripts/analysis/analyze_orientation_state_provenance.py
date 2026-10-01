#!/usr/bin/env python3
"""Compare native and descended states in the 0.6--1.0 rad orientation band.

The analysis separates the scalar orientation-error band from the full local
control state.  It records deterministic frozen-policy transitions from three
origins (native-band, medium descent, and O7--O9 large descent), constructs an
orientation-only damped-least-squares reference action at the same pre-action
state, and evaluates matched one-step DLS counterfactuals at descending
0.1-rad landmarks.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config
from scripts.core.evaluate_thesis_homotopy import automatic_num_envs, physical_cpu_ids


DEFAULT_BAND = (0.6, 1.0)
DEFAULT_DAMPING = 0.05
DAMPING_SENSITIVITY = (0.01, 0.05, 0.10)
MATCH_CALIPERS = (0.25, 0.5, 1.0)
PRIMARY_MATCH_CALIPER = 0.5
LANDMARK_WIDTH = 0.1

STATE_INFO_KEYS = (
    "goal_position", "ee_position", "goal_error_norm",
    "orientation_error_norm", "orientation_error_vector", "ee_jacobian",
)

REWARD_INFO_KEYS = (
    "reward", "r_goal", "keypoint_tracking_quality", "keypoint_progress",
    "keypoint_tracking_reward", "keypoint_progress_reward",
    "keypoint_precision_reward", "velocity_cost", "smooth_cost",
    "safety_penalty", "hard_penalty", "timeout_penalty",
)

STEP_INFO_KEYS = tuple(dict.fromkeys(STATE_INFO_KEYS + REWARD_INFO_KEYS + (
    "rho_orientation", "next_rho_orientation",
    "rho_position", "next_rho_position", "next_keypoint_distance",
    "policy_joint_velocity", "commanded_joint_velocity", "joint_velocity",
    "precision_action_scale", "self_projection_intervened",
    "task_reached", "strict_pose_reached", "timeout",
    "obstacle_collision", "self_collision", "environment_collision",
    "joint_limit",
)))

MATCH_OUTCOMES = (
    "actor_command_alignment",
    "actor_measured_alignment",
    "orientation_progress_rad_s",
    "actor_command_effective_capacity_fraction",
    "actor_dls_action_cosine",
    "actor_dls_same_norm_effective_ratio",
    "orientation_reversal",
    "future_cross_0p6",
    "future_success",
)

COUNTERFACTUAL_COMPONENTS = (
    "reward", "r_goal", "keypoint_tracking_reward",
    "keypoint_progress_reward", "keypoint_precision_reward",
    "velocity_cost", "smooth_cost", "safety_penalty",
)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0, "mean": None, "median": None, "p25": None,
            "p75": None, "p95": None, "min": None, "max": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def orientation_origin(
    initial_rho_orientation: float,
    initial_orientation_bin: int,
    *,
    band_low: float = DEFAULT_BAND[0],
    band_high: float = DEFAULT_BAND[1],
) -> str:
    """Classify an episode by how far it traveled before entering the band."""
    if band_low <= initial_rho_orientation <= band_high:
        return "native_band"
    if initial_rho_orientation > band_high and initial_orientation_bin <= 6:
        return "medium_descent"
    if initial_orientation_bin >= 7:
        return "large_descent"
    return "outside_control"


def _safe_unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1.0e-12)


def _uniform_bound(vector: np.ndarray, bound: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    maximum = float(np.max(np.abs(vector))) if vector.size else 0.0
    if maximum <= 1.0e-12:
        return np.zeros_like(vector)
    return vector * (float(bound) / maximum)


def action_orientation_metrics(
    orientation_error: np.ndarray,
    jacobian: np.ndarray,
    joint_velocity: np.ndarray,
    *,
    action_scale: float,
) -> dict[str, float]:
    """Measure how a joint velocity uses available orientation capacity."""
    error_direction = _safe_unit(orientation_error)
    angular_jacobian = np.asarray(jacobian, dtype=np.float64)[3:]
    joint_velocity = np.asarray(joint_velocity, dtype=np.float64)
    angular_velocity = angular_jacobian @ joint_velocity
    angular_speed = float(np.linalg.norm(angular_velocity))
    effective = float(np.dot(error_direction, angular_velocity))
    directional_row = error_direction @ angular_jacobian
    capacity = float(np.sum(np.abs(directional_row)) * action_scale)
    return {
        "angular_speed_rad_s": angular_speed,
        "effective_angular_speed_rad_s": effective,
        "angular_alignment": effective / max(angular_speed, 1.0e-12),
        "orientation_capacity_rad_s": capacity,
        "effective_capacity_fraction": effective / max(capacity, 1.0e-12),
        "joint_speed_norm_rad_s": float(np.linalg.norm(joint_velocity)),
        "joint_speed_rms_fraction": float(
            np.sqrt(np.mean(np.square(joint_velocity / action_scale)))
        ),
    }


def orientation_dls_actions(
    orientation_error: np.ndarray,
    jacobian: np.ndarray,
    actor_joint_velocity: np.ndarray,
    *,
    action_scale: float,
    damping: float = DEFAULT_DAMPING,
) -> dict[str, np.ndarray | float]:
    """Return full-bound and actor-norm-matched orientation-only DLS actions."""
    error_direction = _safe_unit(orientation_error)
    angular_jacobian = np.asarray(jacobian, dtype=np.float64)[3:]
    regularized = (
        angular_jacobian @ angular_jacobian.T
        + float(damping) ** 2 * np.eye(3, dtype=np.float64)
    )
    raw = angular_jacobian.T @ np.linalg.solve(regularized, error_direction)
    full_bound = _uniform_bound(raw, action_scale)
    actor_joint_velocity = np.asarray(actor_joint_velocity, dtype=np.float64)
    actor_norm = float(np.linalg.norm(actor_joint_velocity))
    raw_norm = float(np.linalg.norm(raw))
    if raw_norm <= 1.0e-12 or actor_norm <= 1.0e-12:
        same_norm = np.zeros_like(raw)
    else:
        same_norm = raw * (actor_norm / raw_norm)
        maximum = float(np.max(np.abs(same_norm)))
        if maximum > action_scale:
            same_norm *= action_scale / maximum
    cosine = float(
        np.dot(actor_joint_velocity, raw)
        / max(actor_norm * raw_norm, 1.0e-12)
    )
    return {
        "raw": raw,
        "full_bound": full_bound,
        "same_norm": same_norm,
        "actor_action_cosine": cosine,
        "damping": float(damping),
    }


def _landmark_bucket(rho: float, band_low: float, band_high: float) -> int:
    clipped = min(max(float(rho), band_low), np.nextafter(band_high, band_low))
    bucket = int(np.floor((clipped - band_low) / LANDMARK_WIDTH + 1.0e-10))
    return min(bucket, int(np.ceil((band_high - band_low) / LANDMARK_WIDTH)) - 1)


def is_first_descending_landmark(
    rho: float,
    previous_rho: float | None,
    bucket: int,
    visited_buckets: set[int],
    *,
    band_low: float,
    band_high: float,
    native_initial: bool = False,
) -> bool:
    """Select the first sampled entry below a bucket's upper edge on descent.

    Native episodes may use their reset state as a baseline. A step that skips
    buckets only marks the bucket actually occupied at the next decision state.
    """
    if bucket in visited_buckets or not band_low <= rho < band_high:
        return False
    if native_initial:
        return True
    upper = min(band_high, band_low + (bucket + 1) * LANDMARK_WIDTH)
    return previous_rho is not None and previous_rho >= upper and rho < upper


def _counterfactual_result(
    info: dict[str, Any], *, control_dt: float,
) -> dict[str, float | bool]:
    result: dict[str, float | bool] = {
        "orientation_progress_rad_s": float(
            (info["rho_orientation"] - info["next_rho_orientation"]) / control_dt
        ),
        "position_progress_m_s": float(
            (info["rho_position"] - info["next_rho_position"]) / control_dt
        ),
        "orientation_reversal": bool(
            info["next_rho_orientation"] > info["rho_orientation"]
        ),
        "next_rho_orientation": float(info["next_rho_orientation"]),
        "next_rho_position": float(info["next_rho_position"]),
        "next_keypoint_distance": float(info["next_keypoint_distance"]),
    }
    for key in COUNTERFACTUAL_COMPONENTS:
        result[key] = float(info[key])
    return result


def _state_features(
    observation: np.ndarray,
    info: dict[str, Any],
    *,
    action_scale: float,
) -> dict[str, Any]:
    error = np.asarray(info["orientation_error_vector"], dtype=np.float64)
    jacobian = np.asarray(info["ee_jacobian"], dtype=np.float64)
    error_direction = _safe_unit(error)
    position_error = (
        np.asarray(info["goal_position"], dtype=np.float64)
        - np.asarray(info["ee_position"], dtype=np.float64)
    )
    position_direction = _safe_unit(position_error)
    angular_jacobian = jacobian[3:]
    singular_values = np.linalg.svd(angular_jacobian, compute_uv=False)
    directional_row = error_direction @ angular_jacobian
    return {
        "q_normalized": np.asarray(observation[:6], dtype=np.float64),
        "qdot_normalized": np.asarray(observation[6:12], dtype=np.float64),
        "orientation_error_direction": error_direction,
        "position_error_direction": position_direction,
        "jacobian_directional_row": directional_row,
        "angular_jacobian": angular_jacobian,
        "jacobian_angular_singular_values": singular_values,
        "orientation_capacity_rad_s": float(
            np.sum(np.abs(directional_row)) * action_scale
        ),
        "observation": np.asarray(observation, dtype=np.float64),
    }


def _pre_action_row(
    *,
    episode: int,
    step: int,
    initial_orientation_bin: int,
    initial_position_bin: int,
    origin: str,
    observation: np.ndarray,
    info: dict[str, Any],
    actor_action: np.ndarray,
    horizon: int,
    action_scale: float,
    damping: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray | float]]:
    error = np.asarray(info["orientation_error_vector"], dtype=np.float64)
    jacobian = np.asarray(info["ee_jacobian"], dtype=np.float64)
    actor_joint_velocity = np.asarray(actor_action, dtype=np.float64) * action_scale
    actor_metrics = action_orientation_metrics(
        error, jacobian, actor_joint_velocity, action_scale=action_scale,
    )
    dls = orientation_dls_actions(
        error, jacobian, actor_joint_velocity,
        action_scale=action_scale, damping=damping,
    )
    dls_same_metrics = action_orientation_metrics(
        error, jacobian, np.asarray(dls["same_norm"]), action_scale=action_scale,
    )
    dls_full_metrics = action_orientation_metrics(
        error, jacobian, np.asarray(dls["full_bound"]), action_scale=action_scale,
    )
    state_features = _state_features(
        observation, info, action_scale=action_scale,
    )
    rho_orientation = float(info["orientation_error_norm"])
    rho_position = float(info["goal_error_norm"])
    row: dict[str, Any] = {
        "episode": int(episode),
        "step": int(step),
        "remaining_steps": int(horizon - step),
        "remaining_time_fraction": float((horizon - step) / horizon),
        "initial_orientation_bin": int(initial_orientation_bin),
        "initial_position_bin": int(initial_position_bin),
        "origin": origin,
        "rho_orientation": rho_orientation,
        "rho_position": rho_position,
        "actor_joint_velocity": actor_joint_velocity,
        "actor_command_alignment": actor_metrics["angular_alignment"],
        "actor_command_effective_angular_speed_rad_s": actor_metrics[
            "effective_angular_speed_rad_s"
        ],
        "actor_command_effective_capacity_fraction": actor_metrics[
            "effective_capacity_fraction"
        ],
        "actor_joint_speed_rms_fraction": actor_metrics[
            "joint_speed_rms_fraction"
        ],
        "actor_dls_action_cosine": float(dls["actor_action_cosine"]),
        "dls_same_norm_alignment": dls_same_metrics["angular_alignment"],
        "dls_same_norm_effective_angular_speed_rad_s": dls_same_metrics[
            "effective_angular_speed_rad_s"
        ],
        "dls_full_bound_alignment": dls_full_metrics["angular_alignment"],
        "dls_full_bound_effective_angular_speed_rad_s": dls_full_metrics[
            "effective_angular_speed_rad_s"
        ],
        "actor_dls_same_norm_effective_ratio": float(
            actor_metrics["effective_angular_speed_rad_s"]
            / max(dls_same_metrics["effective_angular_speed_rad_s"], 1.0e-12)
        ),
        **state_features,
    }
    return row, dls


def _finish_actor_row(
    row: dict[str, Any], info: dict[str, Any], *, control_dt: float,
    action_scale: float = .7,
) -> None:
    jacobian_directional = np.asarray(
        row["jacobian_directional_row"], dtype=np.float64,
    )
    angular_jacobian = np.asarray(row["angular_jacobian"], dtype=np.float64)
    # Endpoint qdot projected through the pre-action Jacobian is a local
    # kinematic proxy; finite-step rho progress below is the measured outcome.
    measured = np.asarray(info["joint_velocity"], dtype=np.float64)
    commanded = np.asarray(info["commanded_joint_velocity"], dtype=np.float64)
    measured_omega_projection = float(np.dot(jacobian_directional, measured))
    measured_omega = angular_jacobian @ measured
    row.update({
        "orientation_progress_rad_s": float(
            (info["rho_orientation"] - info["next_rho_orientation"]) / control_dt
        ),
        "position_progress_m_s": float(
            (info["rho_position"] - info["next_rho_position"]) / control_dt
        ),
        "orientation_reversal": float(
            info["next_rho_orientation"] > info["rho_orientation"]
        ),
        "actor_measured_alignment": float(
            measured_omega_projection / max(float(np.linalg.norm(measured_omega)), 1.0e-12)
        ),
        "actor_measured_effective_angular_speed_rad_s": measured_omega_projection,
        "measured_joint_speed_rms_fraction": float(np.sqrt(np.mean(
            np.square(measured / action_scale)
        ))),
        "precision_action_scale": float(info.get("precision_action_scale", 1.)),
        "command_delta_norm_rad_s": float(np.linalg.norm(
            commanded - np.asarray(row["actor_joint_velocity"], dtype=np.float64)
        )),
        "self_projection_intervened": bool(
            info.get("self_projection_intervened", False)
        ),
        "next_rho_orientation": float(info["next_rho_orientation"]),
        "next_rho_position": float(info["next_rho_position"]),
        "next_keypoint_distance": float(info["next_keypoint_distance"]),
    })
    for key in REWARD_INFO_KEYS:
        row[key] = float(info[key])


def _numeric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "rho_orientation", "rho_position", "remaining_time_fraction",
        "actor_command_alignment", "actor_measured_alignment",
        "orientation_progress_rad_s", "position_progress_m_s",
        "actor_command_effective_capacity_fraction",
        "actor_joint_speed_rms_fraction", "actor_dls_action_cosine",
        "actor_dls_same_norm_effective_ratio",
        "full_time_actor_alignment_delta", "reward",
    )
    return {
        "records": len(records),
        "episodes": len({int(row["episode"]) for row in records}),
        "orientation_reversal_rate": (
            float(np.mean([row["orientation_reversal"] for row in records]))
            if records else None
        ),
        "metrics": {
            metric: _describe(float(row[metric]) for row in records)
            for metric in metrics
        },
    }


def _feature_vector(row: dict[str, Any], feature_set: str) -> np.ndarray:
    task = np.asarray([
        row["rho_orientation"], row["rho_position"],
        row["remaining_time_fraction"],
    ], dtype=np.float64)
    if feature_set == "rho_only":
        return task[:1]
    if feature_set == "task_time":
        return task
    if feature_set == "physical_state":
        return np.concatenate((
            task,
            np.asarray(row["q_normalized"], dtype=np.float64),
            np.asarray(row["qdot_normalized"], dtype=np.float64),
            np.asarray(row["orientation_error_direction"], dtype=np.float64),
            np.asarray(row["position_error_direction"], dtype=np.float64),
            np.asarray(row["jacobian_directional_row"], dtype=np.float64),
            np.asarray(row["jacobian_angular_singular_values"], dtype=np.float64),
            np.asarray([row["orientation_capacity_rad_s"]], dtype=np.float64),
        ))
    if feature_set == "actor_observation":
        return np.asarray(row["observation"], dtype=np.float64)
    raise ValueError(f"unknown feature set {feature_set!r}")


def _cluster_bootstrap_interval(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    seed: int,
    repetitions: int = 2000,
) -> tuple[float | None, float | None]:
    if not len(values):
        return None, None
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) == 1:
        value = float(np.mean(values))
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled_clusters = rng.choice(
            unique_clusters, size=len(unique_clusters), replace=True,
        )
        sampled_values = [values[clusters == cluster] for cluster in sampled_clusters]
        means[index] = float(np.mean(np.concatenate(sampled_values)))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def nearest_state_matches(
    query_records: list[dict[str, Any]],
    control_records: list[dict[str, Any]],
    *,
    feature_set: str,
    caliper: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Nearest-neighbor matching with replacement inside 0.1-rad buckets."""
    if not query_records or not control_records:
        return {
            "feature_set": feature_set, "caliper_standardized_rms": caliper,
            "available_queries": len(query_records),
            "available_controls": len(control_records), "matched_pairs": 0,
        }
    all_records = query_records + control_records
    matrix = np.stack([_feature_vector(row, feature_set) for row in all_records])
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    keep = std > 1.0e-8
    if not np.any(keep):
        standardized = np.zeros((len(all_records), 1), dtype=np.float64)
    else:
        standardized = (matrix[:, keep] - mean[keep]) / std[keep]
    query_z = standardized[:len(query_records)]
    control_z = standardized[len(query_records):]
    controls_by_bucket: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(control_records):
        controls_by_bucket[int(row["orientation_band_bucket"])].append(index)
    pairs: list[tuple[int, int, float]] = []
    for query_index, row in enumerate(query_records):
        available = controls_by_bucket.get(int(row["orientation_band_bucket"]), [])
        if not available:
            continue
        indices = np.asarray(available, dtype=np.int64)
        differences = control_z[indices] - query_z[query_index]
        distances = np.sqrt(np.mean(np.square(differences), axis=1))
        best_local = int(np.argmin(distances))
        distance = float(distances[best_local])
        if distance <= caliper:
            pairs.append((query_index, int(indices[best_local]), distance))
    result: dict[str, Any] = {
        "feature_set": feature_set,
        "caliper_standardized_rms": float(caliper),
        "active_feature_dimensions": int(np.sum(keep)),
        "available_queries": len(query_records),
        "available_controls": len(control_records),
        "matched_pairs": len(pairs),
        "query_coverage": len(pairs) / len(query_records),
        "unique_controls": len({pair[1] for pair in pairs}),
        "match_distance": _describe(pair[2] for pair in pairs),
        "outcomes": {},
    }
    if not pairs:
        return result
    matched_query = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    matched_control = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    query_features = query_z[matched_query]
    control_features = control_z[matched_control]
    feature_smd = np.mean(query_features - control_features, axis=0)
    result["post_match_balance"] = {
        "mean_absolute_standardized_difference": float(np.mean(np.abs(feature_smd))),
        "max_absolute_standardized_difference": float(np.max(np.abs(feature_smd))),
    }
    result["adequate_balance"] = bool(
        result["post_match_balance"]["mean_absolute_standardized_difference"] <= 0.10
        and result["post_match_balance"]["max_absolute_standardized_difference"] <= 0.25
    )
    query_clusters = np.asarray([
        int(query_records[index]["episode"]) for index in matched_query
    ], dtype=np.int64)
    for outcome_index, outcome in enumerate(MATCH_OUTCOMES):
        query_values = np.asarray([
            float(query_records[index][outcome]) for index in matched_query
        ])
        control_values = np.asarray([
            float(control_records[index][outcome]) for index in matched_control
        ])
        difference = query_values - control_values
        ci_low, ci_high = _cluster_bootstrap_interval(
            difference, query_clusters, seed=seed + outcome_index,
        )
        result["outcomes"][outcome] = {
            "query_mean": float(np.mean(query_values)),
            "control_mean": float(np.mean(control_values)),
            "paired_difference_query_minus_control": float(np.mean(difference)),
            "bootstrap_95pct_ci": [ci_low, ci_high],
        }
    result["pair_ids"] = [
        {
            "query_episode": int(query_records[q]["episode"]),
            "query_step": int(query_records[q]["step"]),
            "control_episode": int(control_records[c]["episode"]),
            "control_step": int(control_records[c]["step"]),
            "orientation_band_bucket": int(
                query_records[q]["orientation_band_bucket"]
            ),
            "distance": distance,
        }
        for q, c, distance in pairs
    ]
    return result


def _counterfactual_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    eligible = [row for row in records if key in row]
    result: dict[str, Any] = {
        "records": len(eligible), "dls_minus_actor": {},
    }
    for component in COUNTERFACTUAL_COMPONENTS + (
        "orientation_progress_rad_s", "position_progress_m_s",
    ):
        result["dls_minus_actor"][component] = _describe(
            float(row[key][component]) - float(row["actor_counterfactual"][component])
            for row in eligible
        )
    result["dls_reward_win_rate"] = (
        float(np.mean([
            row[key]["reward"] > row["actor_counterfactual"]["reward"]
            for row in eligible
        ])) if eligible else None
    )
    result["dls_orientation_progress_win_rate"] = (
        float(np.mean([
            row[key]["orientation_progress_rad_s"]
            > row["actor_counterfactual"]["orientation_progress_rad_s"]
            for row in eligible
        ])) if eligible else None
    )
    return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def evaluate(
    agent: ThesisSACAgent,
    pool: ParallelThesisEnvPool,
    config: dict[str, Any],
    *,
    episodes: int,
    seed: int,
    level_index: int,
    band_low: float,
    band_high: float,
    damping: float,
    counterfactuals: bool,
    episode_ids: list[int] | None = None,
    capture_all_large_states: bool = False,
    capture_observations: bool = False,
    capture_episode_states: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    episode_ids = list(range(episodes)) if episode_ids is None else list(episode_ids)
    if len(episode_ids) != episodes or len(set(episode_ids)) != episodes:
        raise ValueError("episode_ids must contain one unique ID per episode")
    levels = config["thesis"]["joint_pose_curriculum"]["levels"]
    level = levels[level_index]
    orientation_levels = config["thesis"]["orientation_curriculum"]["levels"]
    sampler = config["thesis"]["orientation_curriculum"]
    position_bins = int(sampler.get("deterministic_probe_position_bins", 10))
    orientation_bins = int(sampler.get("deterministic_probe_orientation_bins", 10))
    position_edges = np.linspace(
        float(level["target_distance_min_m"]),
        float(level["target_distance_max_m"]), position_bins + 1,
    )
    orientation_edges = np.linspace(
        float(level["target_orientation_min_rad"]),
        float(level["target_orientation_max_rad"]), orientation_bins + 1,
    )
    contract_base = {
        "scene": "none", "xi": 1.0, "strict": True,
        "goal_scale": float(level["goal_scale"]),
        "orientation_scale": float(orientation_levels[level_index]),
        "position_tolerance": float(level["position_tolerance_m"]),
        "orientation_tolerance": float(level["orientation_tolerance_rad"]),
        "lambda_self": 1.0,
    }
    control_dt = float(config["thesis"]["control_dt"])
    horizon = int(config["thesis"]["horizon"])
    action_scale = float(config["thesis"]["action_scale"])
    active: dict[int, dict[str, Any]] = {}
    finished: dict[int, dict[str, Any]] = {}
    transitions: list[dict[str, Any]] = []
    landmarks: list[dict[str, Any]] = []
    next_episode = 0

    while len(finished) < episodes:
        assignments: dict[int, int] = {}
        for worker in range(len(pool)):
            if worker not in active and next_episode < episodes:
                assignments[worker] = episode_ids[next_episode]
                next_episode += 1
        if assignments:
            contracts: dict[int, dict[str, Any]] = {}
            for worker, episode in assignments.items():
                position_bin = (episode // orientation_bins) % position_bins
                orientation_bin = episode % orientation_bins
                contract = dict(contract_base)
                contract.update({
                    "target_distance_min_m": float(position_edges[position_bin]),
                    "target_distance_max_m": float(position_edges[position_bin + 1]),
                    "target_orientation_min_rad": float(orientation_edges[orientation_bin]),
                    "target_orientation_max_rad": float(orientation_edges[orientation_bin + 1]),
                })
                contracts[worker] = contract
                active[worker] = {
                    "episode": episode,
                    "position_bin": position_bin,
                    "orientation_bin": orientation_bin,
                    "step": 0,
                    "landmark_buckets": set(),
                    "last_pre_action_rho": None,
                    "crossings": {"0.6": None, "0.4": None},
                    "band_alignment_window": deque(maxlen=5),
                    "collapse": None,
                }
            resets = pool.reset_many(
                contracts,
                seeds={worker: seed + episode for worker, episode in assignments.items()},
            )
            for worker, (observation, info) in resets.items():
                state = active[worker]
                initial_rho = float(info["orientation_error_norm"])
                state.update({
                    "observation": observation,
                    "info": info,
                    "initial_rho_orientation": initial_rho,
                    "initial_rho_position": float(info["goal_error_norm"]),
                    "origin": orientation_origin(
                        initial_rho, state["orientation_bin"],
                        band_low=band_low, band_high=band_high,
                    ),
                })
                for threshold in (0.6, 0.4):
                    if initial_rho < threshold:
                        state["crossings"][str(threshold)] = 0

        workers = sorted(active)
        observations = np.stack([active[worker]["observation"] for worker in workers])
        actions = agent.select_actions(observations, deterministic=True)
        pending: dict[int, dict[str, Any]] = {}
        landmark_workers: list[int] = []
        for worker, action in zip(workers, actions):
            state = active[worker]
            info = state["info"]
            rho = float(info["orientation_error_norm"])
            in_band = band_low <= rho <= band_high
            if state["origin"] == "outside_control" or not (
                in_band or (capture_all_large_states and state["origin"] == "large_descent")
            ):
                continue
            row, dls = _pre_action_row(
                episode=state["episode"], step=state["step"],
                initial_orientation_bin=state["orientation_bin"],
                initial_position_bin=state["position_bin"], origin=state["origin"],
                observation=state["observation"], info=info, actor_action=action,
                horizon=horizon, action_scale=action_scale, damping=damping,
            )
            if capture_observations:
                row["observation"] = np.asarray(state["observation"], dtype=np.float32).copy()
            bucket = _landmark_bucket(rho, band_low, band_high)
            row["orientation_band_bucket"] = bucket
            row["time_bucket_10_steps"] = int(state["step"] // 10)
            if capture_all_large_states:
                row["linear_jacobian"] = np.asarray(info["ee_jacobian"], dtype=np.float64)[:3]
            is_landmark = in_band and is_first_descending_landmark(
                rho, state["last_pre_action_rho"], bucket,
                state["landmark_buckets"], band_low=band_low,
                band_high=band_high,
                native_initial=(state["step"] == 0 and state["origin"] == "native_band"),
            )
            if is_landmark:
                state["landmark_buckets"].add(bucket)
                row["landmark_bucket"] = bucket
                row["landmark_id"] = len(landmarks)
                row["dls_damping_sensitivity"] = {}
                error = np.asarray(info["orientation_error_vector"], dtype=np.float64)
                jacobian = np.asarray(info["ee_jacobian"], dtype=np.float64)
                actor_qdot = np.asarray(action, dtype=np.float64) * action_scale
                for candidate_damping in DAMPING_SENSITIVITY:
                    candidate = orientation_dls_actions(
                        error, jacobian, actor_qdot, action_scale=action_scale,
                        damping=candidate_damping,
                    )
                    candidate_metrics = action_orientation_metrics(
                        error, jacobian, np.asarray(candidate["same_norm"]),
                        action_scale=action_scale,
                    )
                    row["dls_damping_sensitivity"][str(candidate_damping)] = {
                        "actor_action_cosine": float(candidate["actor_action_cosine"]),
                        "same_norm_alignment": candidate_metrics["angular_alignment"],
                        "same_norm_effective_angular_speed_rad_s": candidate_metrics[
                            "effective_angular_speed_rad_s"
                        ],
                    }
                landmark_workers.append(worker)
            pending[worker] = {
                "row": row, "dls": dls, "is_landmark": is_landmark,
            }

        if pending:
            pending_workers = sorted(pending)
            full_time_observations = []
            for worker in pending_workers:
                observation = np.asarray(
                    active[worker]["observation"], dtype=np.float32,
                ).copy()
                full_time_observations.append(observation)
            full_time_actions = agent.select_actions(
                np.stack(full_time_observations), deterministic=True,
            )
            for worker, full_time_action in zip(pending_workers, full_time_actions):
                info = active[worker]["info"]
                metrics = action_orientation_metrics(
                    np.asarray(info["orientation_error_vector"], dtype=np.float64),
                    np.asarray(info["ee_jacobian"], dtype=np.float64),
                    np.asarray(full_time_action, dtype=np.float64) * action_scale,
                    action_scale=action_scale,
                )
                row = pending[worker]["row"]
                row["full_time_actor_alignment"] = metrics["angular_alignment"]
                row["full_time_actor_effective_angular_speed_rad_s"] = metrics[
                    "effective_angular_speed_rad_s"
                ]
                row["full_time_actor_alignment_delta"] = float(
                    metrics["angular_alignment"] - row["actor_command_alignment"]
                )

        state_workers = sorted(set(landmark_workers if counterfactuals else ()) | (
            {worker for worker, item in pending.items()
             if band_low <= item["row"]["rho_orientation"] <= band_high}
            if capture_episode_states else set()
        ))
        pre_states = pool.states(state_workers) if state_workers else {}
        if capture_episode_states:
            for worker in state_workers:
                item = pending[worker]
                item["row"]["episode_state"] = pre_states[worker]
        actor_results = pool.step_many(dict(zip(workers, actions)))
        actor_post_states = (
            pool.states(landmark_workers) if counterfactuals and landmark_workers else {}
        )

        for worker, pending_item in pending.items():
            info = actor_results[worker][-1]
            row = pending_item["row"]
            _finish_actor_row(row, info, control_dt=control_dt,
                              action_scale=action_scale)
            transitions.append(row)
            if pending_item["is_landmark"]:
                row["actor_counterfactual"] = _counterfactual_result(
                    info, control_dt=control_dt,
                )
                landmarks.append(row)

        if counterfactuals and landmark_workers:
            restored_pre = pool.restore_many(pre_states)
            for worker in landmark_workers:
                if not np.allclose(
                    restored_pre[worker], active[worker]["observation"], atol=1.0e-6,
                ):
                    raise RuntimeError("pre-state restore changed the actor observation")
            same_norm_actions = {
                worker: np.asarray(pending[worker]["dls"]["same_norm"], dtype=np.float32)
                / action_scale
                for worker in landmark_workers
            }
            same_norm_results = pool.step_many(same_norm_actions)
            for worker in landmark_workers:
                row = pending[worker]["row"]
                row["dls_same_norm_counterfactual"] = _counterfactual_result(
                    same_norm_results[worker][-1], control_dt=control_dt,
                )

            restored_pre = pool.restore_many(pre_states)
            for worker in landmark_workers:
                if not np.allclose(
                    restored_pre[worker], active[worker]["observation"], atol=1.0e-6,
                ):
                    raise RuntimeError("second pre-state restore changed the observation")
            full_bound_actions = {
                worker: np.asarray(pending[worker]["dls"]["full_bound"], dtype=np.float32)
                / action_scale
                for worker in landmark_workers
            }
            full_bound_results = pool.step_many(full_bound_actions)
            for worker in landmark_workers:
                row = pending[worker]["row"]
                row["dls_full_bound_counterfactual"] = _counterfactual_result(
                    full_bound_results[worker][-1], control_dt=control_dt,
                )
            restored_post = pool.restore_many(actor_post_states)
            for worker in landmark_workers:
                if not np.allclose(
                    restored_post[worker], actor_results[worker][0], atol=1.0e-6,
                ):
                    raise RuntimeError("post-state restore changed the actor trajectory")

        done_workers: list[int] = []
        for worker in workers:
            observation, _, _, terminated, truncated, info = actor_results[worker]
            state = active[worker]
            state["step"] += 1
            state["observation"] = observation
            state["info"] = info
            state["last_pre_action_rho"] = float(info["rho_orientation"])
            for threshold in (0.6, 0.4):
                key = str(threshold)
                if state["crossings"][key] is None and info["next_rho_orientation"] < threshold:
                    state["crossings"][key] = state["step"]
            pending_item = pending.get(worker)
            if pending_item is not None and state["origin"] == "large_descent" and (
                band_low <= pending_item["row"]["rho_orientation"] <= band_high
            ):
                row = pending_item["row"]
                state["band_alignment_window"].append(row)
                window = state["band_alignment_window"]
                if state["collapse"] is None and len(window) == window.maxlen:
                    alignments = np.asarray([
                        item["actor_command_alignment"] for item in window
                    ])
                    if float(np.mean(alignments)) < 0.25 and int(np.sum(alignments < 0.0)) >= 2:
                        state["collapse"] = {
                            "detected_step": state["step"],
                            "window_start_step": int(window[0]["step"]),
                            "window_mean_alignment": float(np.mean(alignments)),
                            "window_reversal_fraction": float(np.mean(alignments < 0.0)),
                            "rho_orientation": float(row["rho_orientation"]),
                            "rho_position": float(row["rho_position"]),
                            "reward": float(row["reward"]),
                            "keypoint_progress_reward": float(row["keypoint_progress_reward"]),
                            "keypoint_tracking_reward": float(row["keypoint_tracking_reward"]),
                            "keypoint_precision_reward": float(row["keypoint_precision_reward"]),
                            "velocity_cost": float(row["velocity_cost"]),
                            "smooth_cost": float(row["smooth_cost"]),
                            "actor_joint_speed_rms_fraction": float(
                                row["actor_joint_speed_rms_fraction"]
                            ),
                            "actor_dls_action_cosine": float(row["actor_dls_action_cosine"]),
                        }
            if not (terminated or truncated):
                continue
            collision = bool(
                info["obstacle_collision"] or info["self_collision"]
                or info["environment_collision"]
            )
            finished[state["episode"]] = {
                "episode": state["episode"],
                "initial_position_bin": state["position_bin"],
                "initial_orientation_bin": state["orientation_bin"],
                "origin": state["origin"],
                "initial_rho_position": state["initial_rho_position"],
                "initial_rho_orientation": state["initial_rho_orientation"],
                "steps": state["step"],
                "success": bool(info["task_reached"]),
                "timeout": bool(truncated),
                "collision": collision,
                "crossing_0p6_step": state["crossings"]["0.6"],
                "crossing_0p4_step": state["crossings"]["0.4"],
                "collapse": state["collapse"],
                "final_rho_position": float(info["next_rho_position"]),
                "final_rho_orientation": float(info["next_rho_orientation"]),
            }
            done_workers.append(worker)
        for worker in done_workers:
            del active[worker]

    episode_records = [finished[index] for index in episode_ids]
    by_episode = {record["episode"]: record for record in episode_records}
    for row in transitions:
        episode = by_episode[row["episode"]]
        row["future_success"] = float(episode["success"])
        crossing_06 = episode["crossing_0p6_step"]
        row["future_cross_0p6"] = float(
            crossing_06 is not None and crossing_06 >= row["step"]
        )
        row["steps_to_0p6"] = (
            None if crossing_06 is None else int(max(0, crossing_06 - row["step"]))
        )
    return episode_records, transitions, landmarks


def select_matching_records(
    transitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep one state per episode, orientation bucket, and ten-step window."""
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for row in transitions:
        key = (
            int(row["episode"]), int(row["orientation_band_bucket"]),
            int(row["time_bucket_10_steps"]),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml",
    )
    parser.add_argument("--level-index", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=51001)
    parser.add_argument("--band-low", type=float, default=DEFAULT_BAND[0])
    parser.add_argument("--band-high", type=float, default=DEFAULT_BAND[1])
    parser.add_argument("--dls-damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--num-envs", type=int, default=0)
    parser.add_argument("--no-cpu-affinity", action="store_true")
    parser.add_argument("--no-counterfactuals", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.episodes < 100:
        parser.error("--episodes must be at least 100")
    if not 0.0 < args.band_low < args.band_high < np.pi:
        parser.error("orientation band must satisfy 0 < low < high < pi")
    if args.dls_damping <= 0.0:
        parser.error("--dls-damping must be positive")

    checkpoint_path, config_path, output_path = map(
        _path, (args.checkpoint, args.config, args.output),
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    config = load_config(config_path)
    config["device"] = "cpu"
    num_envs = automatic_num_envs(args.episodes) if args.num_envs == 0 else min(
        args.num_envs, args.episodes
    )
    torch.set_num_threads(1)
    env = ThesisHomotopyEnv(config)
    try:
        agent = ThesisSACAgent(
            env.observation_space.shape[0], env.action_space.shape[0], config,
        )
    finally:
        env.close()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_state = checkpoint.get("agent", checkpoint).get("actor", checkpoint)
    agent.actor.load_state_dict(actor_state)
    agent.actor.eval()
    cpu_ids = None
    if num_envs > 1 and not args.no_cpu_affinity:
        cpu_ids = physical_cpu_ids(num_envs)

    start = time.perf_counter()
    with ParallelThesisEnvPool(
        config, [args.seed + 1_000_000 + index for index in range(num_envs)],
        cpu_ids=cpu_ids, step_info_keys=STEP_INFO_KEYS,
        start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn",
    ) as pool:
        episodes, transitions, landmarks = evaluate(
            agent, pool, config, episodes=args.episodes, seed=args.seed,
            level_index=args.level_index, band_low=args.band_low,
            band_high=args.band_high, damping=args.dls_damping,
            counterfactuals=not args.no_counterfactuals,
        )

    origins = ("native_band", "medium_descent", "large_descent")
    transition_summary = {
        origin: _numeric_summary([
            row for row in transitions if row["origin"] == origin
        ]) for origin in origins
    }
    landmark_summary = {
        origin: _numeric_summary([
            row for row in landmarks if row["origin"] == origin
        ]) for origin in origins
    }
    episode_summary = {}
    for origin in origins:
        subset = [record for record in episodes if record["origin"] == origin]
        episode_summary[origin] = {
            "episodes": len(subset),
            "success_rate": float(np.mean([row["success"] for row in subset]))
            if subset else None,
            "timeout_rate": float(np.mean([row["timeout"] for row in subset]))
            if subset else None,
            "collapse_rate": float(np.mean([row["collapse"] is not None for row in subset]))
            if subset else None,
        }

    matching_records = select_matching_records(transitions)
    matching: dict[str, Any] = {}
    native = [
        row for row in matching_records if row["origin"] == "native_band"
    ]
    for query_origin in ("medium_descent", "large_descent"):
        query = [
            row for row in matching_records if row["origin"] == query_origin
        ]
        matching[query_origin + "_vs_native_band"] = {}
        for feature_set in (
            "rho_only", "task_time", "physical_state", "actor_observation",
        ):
            sensitivity = {}
            for caliper in MATCH_CALIPERS:
                sensitivity[str(caliper)] = nearest_state_matches(
                    query, native, feature_set=feature_set, caliper=caliper,
                    seed=args.seed + int(caliper * 100),
                )
            matching[query_origin + "_vs_native_band"][feature_set] = {
                "primary_caliper": PRIMARY_MATCH_CALIPER,
                "primary": sensitivity[str(PRIMARY_MATCH_CALIPER)],
                "caliper_sensitivity": sensitivity,
            }

    counterfactual_summary = {}
    for origin in origins:
        subset = [row for row in landmarks if row["origin"] == origin]
        counterfactual_summary[origin] = {
            "same_norm_dls": _counterfactual_summary(
                subset, "dls_same_norm_counterfactual",
            ),
            "full_bound_dls": _counterfactual_summary(
                subset, "dls_full_bound_counterfactual",
            ),
        }

    collapse_summary: dict[str, Any] = {}
    for outcome, success in (("success", True), ("failure", False)):
        subset = [
            row for row in episodes
            if row["origin"] == "large_descent" and row["success"] is success
        ]
        collapsed = [row["collapse"] for row in subset if row["collapse"] is not None]
        collapse_summary[outcome] = {
            "episodes": len(subset),
            "collapse_count": len(collapsed),
            "collapse_rate": len(collapsed) / len(subset) if subset else None,
            "detected_step": _describe(row["detected_step"] for row in collapsed),
            "rho_orientation": _describe(row["rho_orientation"] for row in collapsed),
            "rho_position": _describe(row["rho_position"] for row in collapsed),
            "window_mean_alignment": _describe(
                row["window_mean_alignment"] for row in collapsed
            ),
            "reward": _describe(row["reward"] for row in collapsed),
            "actor_dls_action_cosine": _describe(
                row["actor_dls_action_cosine"] for row in collapsed
            ),
        }

    compact_landmarks = []
    for row in landmarks:
        compact = {key: value for key, value in row.items() if key != "observation"}
        compact_landmarks.append(compact)

    result = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config": str(config_path.relative_to(ROOT)),
        "level_index": args.level_index,
        "episodes": args.episodes,
        "seed": args.seed,
        "num_envs": num_envs,
        "orientation_band_rad": [args.band_low, args.band_high],
        "landmark_width_rad": LANDMARK_WIDTH,
        "dls_damping": args.dls_damping,
        "counterfactuals_enabled": not args.no_counterfactuals,
        "elapsed_seconds": time.perf_counter() - start,
        "episode_summary_by_origin": episode_summary,
        "transition_summary_by_origin": transition_summary,
        "landmark_summary_by_origin": landmark_summary,
        "matching_record_counts_by_origin": {
            origin: sum(row["origin"] == origin for row in matching_records)
            for origin in origins
        },
        "matched_state_analysis": matching,
        "counterfactual_summary_by_origin": counterfactual_summary,
        "large_descent_alignment_collapse": collapse_summary,
        "episode_records": episodes,
        "landmark_records": compact_landmarks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_ready(result), indent=2) + "\n", encoding="utf-8",
    )

    def compact_match(details: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value for key, value in details.items()
            if key != "pair_ids"
        }

    print(json.dumps(_json_ready({
        "episode_summary_by_origin": episode_summary,
        "transition_summary_by_origin": transition_summary,
        "matched_state_analysis": {
            comparison: {
                feature: compact_match(details["primary"])
                for feature, details in feature_sets.items()
            }
            for comparison, feature_sets in matching.items()
        },
        "counterfactual_summary_by_origin": counterfactual_summary,
        "large_descent_alignment_collapse": collapse_summary,
    }), indent=2))


if __name__ == "__main__":
    main()
