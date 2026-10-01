#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.algorithms.homotopy_curriculum import HomotopyCurriculum
from rl_risk_sac.algorithms.homotopy_replay import HomotopyReplayBuffer
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config


HYBRID_KEYPOINT_AUTO_CHAIN_PROTOCOL = (
    "task_space_precision_curriculum_hybrid_keypoint_jacobian_auto_chain_pcr"
)
SUPPORTED_PROTOCOLS = frozenset((
    HYBRID_KEYPOINT_AUTO_CHAIN_PROTOCOL,
))


def validate_training_architecture(config: dict) -> None:
    """Reject retired experiment modes before creating output or workers."""
    thesis = config["thesis"]
    observation = thesis.get("observation", {})
    reward = thesis.get("reward", {})
    sac = config["sac"]
    if (
        thesis.get("protocol") not in SUPPORTED_PROTOCOLS
        or not observation.get("keypoint_jacobian_pose", True)
        or not observation.get("hybrid_explicit_pose_error", True)
        or observation.get("include_orientation_error_vector", False)
        or reward.get("pose_objective", "unified_keypoint") != "unified_keypoint"
        or not reward.get("keypoint_pose_reward", True)
        or reward.get("rtpc_orientation_reward", False)
        or reward.get("pose_balanced_rtpc_reward", False)
        or not sac.get("chain_pcr_enabled", False)
        or not sac.get("chain_pcr_auto_enabled", False)
    ):
        raise ValueError("only Hybrid Keypoint + Jacobian + Auto-PCR is supported")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_stage_block_steps(
    stage_total_step: int, requested_steps: int, max_stage_steps: int
) -> int:
    if stage_total_step < 0:
        raise ValueError("stage_total_step must be non-negative")
    if requested_steps < 1 or max_stage_steps < 1:
        raise ValueError("requested_steps and max_stage_steps must be positive")
    remaining = max_stage_steps - stage_total_step
    if remaining <= 0:
        raise ValueError(
            f"stage hard budget exhausted: {stage_total_step}/{max_stage_steps} transitions"
        )
    return min(requested_steps, remaining)


def actor_updates_enabled(counters: dict) -> bool:
    """Whether a migrated/initialized actor has finished its critic warmup."""
    initialization = counters.get("actor_initialization", {})
    warmup = int(initialization.get("critic_warmup_transitions", 0))
    start = int(initialization.get("warmup_start_stage_total_step", 0))
    return int(counters["stage_total_step"]) - start >= warmup


def record_keypoint_reward_statistics(
    counters: dict, info: dict, *, enabled: bool,
) -> None:
    if not enabled:
        return
    progress = float(info["keypoint_progress"])
    counters["keypoint_transition_count"] = int(
        counters.get("keypoint_transition_count", 0)
    ) + 1
    counters["keypoint_tracking_quality_sum"] = float(
        counters.get("keypoint_tracking_quality_sum", 0.0)
    ) + float(info["keypoint_tracking_quality"])
    counters["keypoint_precision_quality_sum"] = float(
        counters.get("keypoint_precision_quality_sum", 0.0)
    ) + float(info.get("keypoint_precision_quality", 0.0))
    counters["keypoint_precision_reward_sum"] = float(
        counters.get("keypoint_precision_reward_sum", 0.0)
    ) + float(info.get("keypoint_precision_reward", 0.0))
    counters["keypoint_progress_sum"] = float(
        counters.get("keypoint_progress_sum", 0.0)
    ) + progress
    counters["jacobian_clip_ratio_sum"] = float(
        counters.get("jacobian_clip_ratio_sum", 0.0)
    ) + float(info.get("jacobian_clip_ratio", 0.0))
    if progress > 0.0:
        counters["keypoint_progress_positive_count"] = int(
            counters.get("keypoint_progress_positive_count", 0)
        ) + 1
        counters["keypoint_progress_positive_sum"] = float(
            counters.get("keypoint_progress_positive_sum", 0.0)
        ) + progress


def keypoint_reward_statistics(counters: dict) -> dict[str, float | str]:
    count = int(counters.get("keypoint_transition_count", 0))
    positive_count = int(counters.get("keypoint_progress_positive_count", 0))
    if count == 0:
        return {
            "keypoint_tracking_quality_mean": "",
            "keypoint_precision_quality_mean": "",
            "keypoint_precision_reward_mean": "",
            "keypoint_progress_mean": "",
            "keypoint_progress_positive_ratio": "",
            "keypoint_progress_mean_when_positive": "",
            "jacobian_clip_ratio_mean": "",
        }
    return {
        "keypoint_tracking_quality_mean": float(
            counters.get("keypoint_tracking_quality_sum", 0.0)
        ) / count,
        "keypoint_precision_quality_mean": float(
            counters.get("keypoint_precision_quality_sum", 0.0)
        ) / count,
        "keypoint_precision_reward_mean": float(
            counters.get("keypoint_precision_reward_sum", 0.0)
        ) / count,
        "keypoint_progress_mean": float(
            counters.get("keypoint_progress_sum", 0.0)
        ) / count,
        "keypoint_progress_positive_ratio": positive_count / count,
        "keypoint_progress_mean_when_positive": (
            "" if positive_count == 0 else float(
                counters.get("keypoint_progress_positive_sum", 0.0)
            ) / positive_count
        ),
        "jacobian_clip_ratio_mean": float(
            counters.get("jacobian_clip_ratio_sum", 0.0)
        ) / count,
    }


def resolve_num_envs(
    stage: str, requested_num_envs: int | None, configured_num_envs: int,
    validation_mode: bool,
) -> int:
    if requested_num_envs is not None:
        result = int(requested_num_envs)
    elif stage == "s0" and not validation_mode:
        result = int(configured_num_envs)
    else:
        result = 1
    if result < 1:
        raise ValueError("num_envs must be positive")
    return result


def accrue_update_credit(counters: dict, updates_per_transition: float) -> int:
    """Return gradient steps due while preserving fractional UTD credit."""
    if updates_per_transition <= 0.0:
        raise ValueError("updates_per_transition must be positive")
    credit = float(counters.get("update_credit", 0.0)) + updates_per_transition
    due = int(np.floor(credit + 1e-12))
    counters["update_credit"] = credit - due
    return due


def save_checkpoint(path: Path, agent, replay, curriculum, counters, config, env, active_episode) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "protocol": config["thesis"]["protocol"], "agent": agent.state_dict(),
        "replay": replay.state_dict(), "curriculum": curriculum.state_dict(),
        "counters": counters, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "environment_rng": env.rng.bit_generator.state,
        "warmup_action_rng": config["_warmup_action_rng"].bit_generator.state,
        "rng_stream_seeds": config["_rng_stream_seeds"],
        "active_episode": active_episode,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    }, path)


def save_parallel_checkpoint(
    path: Path, agent, replay, curriculum, counters, config,
    pool: ParallelThesisEnvPool, workers: list[dict | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    environment_states = pool.states()
    parallel_workers = []
    for index, worker in enumerate(workers):
        parallel_workers.append({
            "active": worker is not None,
            "episode": None if worker is None else {
                key: value for key, value in worker.items() if key != "observation"
            },
            "observation": None if worker is None else worker["observation"],
            "environment": environment_states[index],
        })
    torch.save({
        "protocol": config["thesis"]["protocol"], "agent": agent.state_dict(),
        "replay": replay.state_dict(), "curriculum": curriculum.state_dict(),
        "counters": counters, "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "environment_rng": environment_states[0]["rng_state"],
        "parallel_num_envs": len(pool), "parallel_workers": parallel_workers,
        "warmup_action_rng": config["_warmup_action_rng"].bit_generator.state,
        "rng_stream_seeds": config["_rng_stream_seeds"], "active_episode": None,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    }, path)


def checkpoint_actor_state(checkpoint: Path, device: torch.device) -> dict[str, torch.Tensor]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "agent" in state:
        actor_state = state["agent"]["actor"]
    elif isinstance(state, dict) and "actor" in state:
        actor_state = state["actor"]
    else:
        actor_state = state
    return actor_state


def load_reference_actor(agent: ThesisSACAgent, checkpoint: Path):
    actor_state = checkpoint_actor_state(checkpoint, agent.device)
    reference = agent.frozen_actor_copy()
    reference.load_state_dict(actor_state)
    reference.eval()
    return reference


def restore_torch_rng(state: dict) -> None:
    torch.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng"]])


def checkpoint_curriculum(config: dict, state: dict) -> HomotopyCurriculum:
    """Rebuild a checkpoint curriculum under the current serial contract."""
    thesis = config["thesis"]
    goal = thesis["goal_curriculum"]
    orientation = thesis["orientation_curriculum"]
    self_safety = thesis["self_collision"]["curriculum"]
    curriculum = HomotopyCurriculum(
        str(state["stage"]),
        ramp_steps=int(thesis["ramp_steps"]),
        goal_start_scale=float(goal["start_scale"]),
        goal_end_scale=float(goal["end_scale"]),
        goal_success_window=int(goal["success_window"]),
        goal_success_floor=float(goal["success_floor"]),
        goal_full_scale_min_steps=int(goal["full_scale_min_transitions"]),
        goal_levels=goal["levels"],
        goal_min_transitions_per_level=int(goal["min_transitions_per_level"]),
        orientation_start_scale=float(orientation["start_scale"]),
        orientation_end_scale=float(orientation["end_scale"]),
        orientation_tolerance_start=float(orientation["tolerance_start"]),
        orientation_tolerance_end=float(orientation["tolerance_end"]),
        orientation_success_window=int(orientation["success_window"]),
        orientation_success_floor=float(orientation["success_floor"]),
        orientation_levels=orientation["levels"],
        orientation_min_transitions_per_level=int(
            orientation["min_transitions_per_level"]
        ),
        orientation_anchor_probability=float(orientation["anchor_probability"]),
        orientation_anchor_floor=float(orientation["anchor_success_floor"]),
        orientation_anchor_window=int(orientation["anchor_success_window"]),
        orientation_deterministic_probe_required=bool(
            orientation["deterministic_probe_required"]
        ),
        orientation_deterministic_previous_success_floor=float(
            orientation["deterministic_previous_success_floor"]
        ),
        orientation_deterministic_previous_collision_ceiling=float(
            orientation["deterministic_previous_collision_ceiling"]
        ),
        orientation_full_scale_min_steps=int(
            orientation["full_scale_min_transitions"]
        ),
        joint_pose_levels=thesis["joint_pose_curriculum"]["levels"],
        self_start_weight=float(self_safety["start_weight"]),
        self_end_weight=float(self_safety["end_weight"]),
        self_ramp_steps=int(self_safety["ramp_steps"]),
        self_full_weight_min_steps=int(
            self_safety["full_weight_min_transitions"]
        ),
        self_probe_collision_ceiling=float(self_safety["probe_collision_ceiling"]),
    )
    curriculum.load_state_dict(state)
    curriculum.configure_orientation_retention(orientation)
    return curriculum


def validate_serial_stage_transition(
    config: dict, checkpoint_state: dict, destination_stage: str,
) -> None:
    """Validate S0->S1 or S1->S2 with the formal latest completion gate."""
    source_stage = str(checkpoint_state["curriculum"]["stage"])
    expected = {"s1": "s0", "s2": "s1"}.get(destination_stage)
    if expected is None or source_stage != expected:
        raise ValueError(f"invalid stage transition {source_stage}->{destination_stage}")
    curriculum = checkpoint_curriculum(config, checkpoint_state["curriculum"])
    thesis = config["thesis"]
    complete = curriculum.stage_complete(
        int(thesis["goal_curriculum"]["full_scale_min_transitions"]),
        int(thesis["orientation_curriculum"]["full_scale_min_transitions"]),
        int(thesis["strict_min_transitions"]),
    )
    if not complete:
        if source_stage == "s0":
            raise ValueError(
                "S0 checkpoint has not passed the final P5 frozen-probe, "
                "P4 retention, and self-safety completion gate"
            )
        raise ValueError(
            "S1 checkpoint has not completed strict static-scene consolidation"
        )


class FrozenActorEvaluator:
    """Detached deterministic actor snapshot used only by promotion probes."""

    def __init__(self, agent: ThesisSACAgent) -> None:
        self.actor = agent.frozen_actor_copy()
        self.device = agent.device

    def select_actions(
        self, observations: np.ndarray, deterministic: bool = True,
    ) -> np.ndarray:
        del deterministic
        tensor = torch.as_tensor(observations, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            return self.actor.deterministic(tensor).cpu().numpy()

    def select_action(
        self, observation: np.ndarray, deterministic: bool = True,
    ) -> np.ndarray:
        return self.select_actions(np.asarray(observation)[None, :], deterministic)[0]


def task_space_training_kwargs(
    config: dict, curriculum: HomotopyCurriculum, *, anchor: bool,
    bin_probabilities: list[float] | None = None,
) -> dict[str, object]:
    """Add V13 task-space bounds to a training episode."""
    contract = curriculum.pose_contract(previous=anchor)
    if "target_distance_max_m" not in contract:
        return {}
    result = {
        key: float(contract[key]) for key in (
            "target_distance_min_m", "target_distance_max_m",
            "target_orientation_min_rad", "target_orientation_max_rad",
        )
    }
    index = int(curriculum.orientation.level_index)
    sampling = config["thesis"]["joint_pose_curriculum"].get(
        "task_space_sampling", {}
    )
    if not anchor and index > 0 and not bool(sampling.get("balanced_bins", False)):
        previous = curriculum.pose_contract(previous=True)
        result.update({
            "target_history_distance_max_m": float(previous["target_distance_max_m"]),
            "target_history_orientation_max_rad": float(previous["target_orientation_max_rad"]),
            "target_history_probability": float(sampling.get("history_probability", 0.5)),
        })
    fixed_probabilities = (
        fixed_task_space_bin_probabilities(config)
        if curriculum.stage == "s0" else None
    )
    effective_probabilities = (
        fixed_probabilities
        if fixed_probabilities is not None else bin_probabilities
    )
    if not anchor and effective_probabilities is not None:
        result["task_space_bin_probabilities"] = list(effective_probabilities)
    return result


def fixed_task_space_bin_probabilities(config: dict) -> list[float] | None:
    """Return the configured uniform-plus-hard-orientation training mixture."""
    sampling = config.get("thesis", {}).get("joint_pose_curriculum", {}).get(
        "task_space_sampling", {}
    )
    mixture = sampling.get("fixed_orientation_mixture", {})
    if not bool(mixture.get("enabled", False)):
        return None
    position_bins = int(sampling.get("position_bins", 10))
    orientation_bins = int(sampling.get("orientation_bins", 10))
    uniform_fraction = float(mixture.get("uniform_fraction", 0.5))
    groups = list(mixture.get("orientation_groups", []))
    if not 0.0 < uniform_fraction < 1.0:
        raise ValueError("fixed orientation uniform_fraction must be in (0, 1)")
    if not groups:
        raise ValueError("fixed orientation mixture requires orientation_groups")
    total_bins = position_bins * orientation_bins
    probabilities = np.full(
        total_bins, uniform_fraction / total_bins, dtype=np.float64,
    )
    group_fraction_sum = 0.0
    for group in groups:
        fraction = float(group.get("fraction", 0.0))
        bins = [int(value) for value in group.get("bins", [])]
        if fraction <= 0.0:
            raise ValueError("orientation group fraction must be positive")
        if not bins or len(set(bins)) != len(bins):
            raise ValueError("orientation group bins must be non-empty and unique")
        if min(bins) < 0 or max(bins) >= orientation_bins:
            raise ValueError("orientation group bin is outside the configured grid")
        group_fraction_sum += fraction
        probability_per_cell = fraction / (position_bins * len(bins))
        for position_bin in range(position_bins):
            for orientation_bin in bins:
                probabilities[
                    position_bin * orientation_bins + orientation_bin
                ] += probability_per_cell
    if not np.isclose(
        uniform_fraction + group_fraction_sum, 1.0, rtol=0.0, atol=1e-12,
    ):
        raise ValueError("fixed orientation mixture fractions must sum to one")
    if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("fixed task-space probabilities do not sum to one")
    return probabilities.tolist()


def update_adaptive_task_space_sampling(
    config: dict, counters: dict, probe_result: dict[str, object],
) -> list[float] | None:
    """Mix uniform coverage with weakness/improvement-driven bin sampling."""
    sampling = config.get("thesis", {}).get("joint_pose_curriculum", {}).get(
        "task_space_sampling", {}
    )
    adaptive = sampling.get("adaptive", {})
    if not bool(adaptive.get("enabled", False)):
        return None
    current = probe_result.get("current")
    if not isinstance(current, dict) or "bin_success_rates" not in current:
        return None
    rates = np.asarray(current["bin_success_rates"], dtype=np.float64)
    total_bins = int(sampling.get("position_bins", 10)) * int(
        sampling.get("orientation_bins", 10)
    )
    if rates.shape != (total_bins,) or not np.all(np.isfinite(rates)):
        raise ValueError("probe bin success rates do not match task-space bins")
    previous_value = counters.get("adaptive_previous_bin_success_rates")
    previous = (
        rates if previous_value is None
        else np.asarray(previous_value, dtype=np.float64)
    )
    if previous.shape != rates.shape:
        previous = rates
    weakness = np.maximum(1.0 - rates, 0.0)
    improvement = np.maximum(rates - previous, 0.0)
    score_floor = float(adaptive.get("score_floor", 0.01))
    improvement_weight = float(adaptive.get("improvement_weight", 0.5))
    scores = weakness + improvement_weight * improvement + score_floor
    adaptive_distribution = scores / scores.sum()
    beta = float(adaptive.get("beta", 0.60))
    if not 0.0 <= beta < 1.0:
        raise ValueError("adaptive task-space beta must be in [0, 1)")
    probabilities = (1.0 - beta) / total_bins + beta * adaptive_distribution
    probabilities /= probabilities.sum()
    counters["adaptive_previous_bin_success_rates"] = rates.tolist()
    counters["task_space_bin_probabilities"] = probabilities.tolist()
    counters["adaptive_sampling_updates"] = int(
        counters.get("adaptive_sampling_updates", 0)
    ) + 1
    return probabilities.tolist()


def scalar_probe_metrics(metrics: dict[str, object]) -> dict[str, object]:
    """Return only columns belonging in the scalar probe CSV row."""
    return {
        key: value for key, value in metrics.items()
        if key not in {"bin_success_rates", "bin_strict_pose_hit_rates"}
    }


def summarize_pose_probe_outcomes(
    values: np.ndarray,
    *,
    has_bins: bool,
    orientation_bins: int,
    large_angle_steps: int,
    orientation_improvements: int,
    orientation_rebound_sum: float,
    keypoint_statistics: dict[str, float] | None = None,
) -> dict[str, object]:
    """Build the shared serial/parallel frozen-pose metric contract."""
    result: dict[str, object] = {
        "success_rate": float(np.mean(values[:, 0])),
        "collision_rate": float(np.mean(values[:, 1])),
        "joint_limit_rate": float(np.mean(values[:, 2])),
        "timeout_rate": float(np.mean(values[:, 3])),
        "mean_position_error_m": float(np.mean(values[:, 4])),
        "mean_orientation_error_rad": float(np.mean(values[:, 5])),
        "p95_orientation_error_rad": float(np.quantile(values[:, 5], .95)),
        "mean_minimum_self_distance_m": float(np.mean(values[:, 6])),
        "self_violation_rate": float(np.mean(values[:, 7])),
        "strict_pose_hit_rate": float(np.mean(values[:, 8])),
        "large_angle_orientation_steps": int(large_angle_steps),
        "large_angle_orientation_improvement_ratio": (
            float(orientation_improvements / large_angle_steps)
            if large_angle_steps else 0.0
        ),
        "large_angle_orientation_mean_rebound_rad": (
            float(orientation_rebound_sum / large_angle_steps)
            if large_angle_steps else 0.0
        ),
    }
    if keypoint_statistics is not None:
        result.update(keypoint_statistics)
    if not has_bins:
        return result

    bin_indices = values[:, 9].astype(int)
    ordered_bins = sorted(set(bin_indices))
    bin_rates = [
        float(np.mean(values[bin_indices == index, 0])) for index in ordered_bins
    ]
    result["minimum_bin_success_rate"] = min(bin_rates)
    result["bin_success_rates"] = bin_rates
    result["weak_bin_count_below_0_6"] = int(np.sum(np.asarray(bin_rates) < .60))
    strict_bin_rates = [
        float(np.mean(values[bin_indices == index, 8])) for index in ordered_bins
    ]
    result["minimum_bin_strict_pose_hit_rate"] = min(strict_bin_rates)
    result["bin_strict_pose_hit_rates"] = strict_bin_rates

    if orientation_bins == 10:
        orientation_index = bin_indices % orientation_bins
        for label, lower, upper in (
            ("o0_o4", 0, 4), ("o5_o6", 5, 6), ("o7_o9", 7, 9),
        ):
            mask = (orientation_index >= lower) & (orientation_index <= upper)
            result[f"success_rate_{label}"] = float(np.mean(values[mask, 0]))
            result[f"timeout_rate_{label}"] = float(np.mean(values[mask, 3]))
    return result


def _empty_keypoint_probe_accumulator() -> dict[str, object]:
    return {
        "steps": 0,
        "distance_sum": 0.0,
        "next_distance_sum": 0.0,
        "quality_sum": 0.0,
        "precision_quality_sum": 0.0,
        "precision_reward_sum": 0.0,
        "progress_sum": 0.0,
        "positive_count": 0,
        "jacobian_clip_ratio_sum": 0.0,
        "episode_start_distances": [],
        "episode_end_distances": [],
    }


def _record_keypoint_probe_step(
    accumulator: dict[str, object], info: dict[str, object],
) -> None:
    progress = float(info["keypoint_progress"])
    accumulator["steps"] = int(accumulator["steps"]) + 1
    for target, source in (
        ("distance_sum", "keypoint_distance"),
        ("next_distance_sum", "next_keypoint_distance"),
        ("quality_sum", "keypoint_tracking_quality"),
        ("precision_quality_sum", "keypoint_precision_quality"),
        ("precision_reward_sum", "keypoint_precision_reward"),
        ("progress_sum", "keypoint_progress"),
        ("jacobian_clip_ratio_sum", "jacobian_clip_ratio"),
    ):
        accumulator[target] = float(accumulator[target]) + float(info[source])
    accumulator["positive_count"] = int(accumulator["positive_count"]) + int(
        progress > 0.0
    )


def _keypoint_probe_statistics(
    accumulator: dict[str, object],
) -> dict[str, float]:
    steps = int(accumulator["steps"])
    if steps < 1:
        return {}
    starts = np.asarray(accumulator["episode_start_distances"], dtype=np.float64)
    ends = np.asarray(accumulator["episode_end_distances"], dtype=np.float64)
    reductions = starts - ends
    return {
        "keypoint_step_count": steps,
        "keypoint_distance_mean_m": float(accumulator["distance_sum"]) / steps,
        "keypoint_next_distance_mean_m": float(
            accumulator["next_distance_sum"]
        ) / steps,
        "keypoint_tracking_quality_mean": float(
            accumulator["quality_sum"]
        ) / steps,
        "keypoint_precision_quality_mean": float(
            accumulator["precision_quality_sum"]
        ) / steps,
        "keypoint_precision_reward_mean": float(
            accumulator["precision_reward_sum"]
        ) / steps,
        "keypoint_progress_mean_m": float(accumulator["progress_sum"]) / steps,
        "keypoint_progress_positive_ratio": int(
            accumulator["positive_count"]
        ) / steps,
        "jacobian_clip_ratio_mean": float(
            accumulator["jacobian_clip_ratio_sum"]
        ) / steps,
        "keypoint_episode_start_distance_mean_m": float(np.mean(starts)),
        "keypoint_episode_end_distance_mean_m": float(np.mean(ends)),
        "keypoint_episode_distance_reduction_mean_m": float(np.mean(reductions)),
        "keypoint_episode_improvement_ratio": float(np.mean(reductions > 0.0)),
    }


def run_deterministic_pose_probe(
    agent: ThesisSACAgent, config: dict, *, episodes: int, seed: int,
    goal_scale: float, orientation_scale: float,
    position_tolerance: float, orientation_tolerance: float,
    scene: str = "none",
    target_distance_min_m: float | None = None,
    target_distance_max_m: float | None = None,
    target_orientation_min_rad: float | None = None,
    target_orientation_max_rad: float | None = None,
) -> dict[str, object]:
    """Evaluate the current actor on a fixed joint-pose curriculum contract."""
    probe_env = ThesisHomotopyEnv(config)
    outcomes = []
    large_angle_steps = 0
    orientation_improvements = 0
    orientation_rebound_sum = 0.0
    keypoint_accumulator = _empty_keypoint_probe_accumulator()
    orientation_bins = int(config["thesis"]["orientation_curriculum"].get(
        "deterministic_probe_orientation_bins", 10
    ))
    self_safe_distance = float(config["thesis"]["self_collision"]["safe_distance_m"])
    try:
        for episode in range(int(episodes)):
            task_contract = {}
            bin_index = -1
            if target_distance_max_m is not None:
                position_bins = int(config["thesis"]["orientation_curriculum"].get(
                    "deterministic_probe_position_bins", 10
                ))
                position_bin = (episode // orientation_bins) % position_bins
                orientation_bin = episode % orientation_bins
                position_edges = np.linspace(target_distance_min_m, target_distance_max_m, position_bins + 1)
                orientation_edges = np.linspace(target_orientation_min_rad, target_orientation_max_rad, orientation_bins + 1)
                task_contract = {
                    "target_distance_min_m": float(position_edges[position_bin]),
                    "target_distance_max_m": float(position_edges[position_bin + 1]),
                    "target_orientation_min_rad": float(orientation_edges[orientation_bin]),
                    "target_orientation_max_rad": float(orientation_edges[orientation_bin + 1]),
                }
                bin_index = position_bin * orientation_bins + orientation_bin
            probe_env.configure_episode(
                scene, xi=1.0, strict=True, goal_scale=goal_scale,
                orientation_scale=orientation_scale,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
                **task_contract,
            )
            observation, _ = probe_env.reset(seed=int(seed) + episode)
            terminated = truncated = False
            info: dict = {}
            minimum_self_distance = float("inf")
            self_violation = False
            strict_pose_hit = False
            episode_start_keypoint_distance: float | None = None
            while not (terminated or truncated):
                observation, _, _, terminated, truncated, info = probe_env.step(
                    agent.select_action(observation, deterministic=True)
                )
                _record_keypoint_probe_step(keypoint_accumulator, info)
                if episode_start_keypoint_distance is None:
                    # The first transition exposes D_KP,t, which is the reset
                    # state's distance and therefore the episode start value.
                    episode_start_keypoint_distance = float(
                        info["keypoint_distance"]
                    )
                previous_orientation = float(info["rho_orientation"])
                next_orientation = float(info["next_rho_orientation"])
                if previous_orientation > 1.6:
                    large_angle_steps += 1
                    orientation_improvements += int(next_orientation < previous_orientation)
                    orientation_rebound_sum += max(
                        0.0, next_orientation - previous_orientation
                    )
                step_self_distance = float(info["control_self_min_distance"])
                minimum_self_distance = min(minimum_self_distance, step_self_distance)
                self_violation = self_violation or step_self_distance < self_safe_distance
                strict_pose_hit = bool(
                    strict_pose_hit or info.get("strict_pose_reached", False)
                )
            keypoint_accumulator["episode_start_distances"].append(
                float(episode_start_keypoint_distance)
            )
            collision = bool(
                info["obstacle_collision"]
                or info["self_collision"]
                or info["environment_collision"]
            )
            outcomes.append((
                float(info["task_reached"]), float(collision),
                float(info["joint_limit"]), float(truncated),
                float(info["next_rho_position"]),
                float(info["next_rho_orientation"]),
                minimum_self_distance, float(self_violation),
                float(strict_pose_hit),
                float(bin_index),
            ))
            keypoint_accumulator["episode_end_distances"].append(
                float(info["next_keypoint_distance"])
            )
    finally:
        probe_env.close()
    values = np.asarray(outcomes, dtype=np.float64)
    return summarize_pose_probe_outcomes(
        values,
        has_bins=target_distance_max_m is not None,
        orientation_bins=orientation_bins,
        large_angle_steps=large_angle_steps,
        orientation_improvements=orientation_improvements,
        orientation_rebound_sum=orientation_rebound_sum,
        keypoint_statistics=_keypoint_probe_statistics(keypoint_accumulator),
    )


def run_parallel_deterministic_pose_probe(
    agent: ThesisSACAgent, pool: ParallelThesisEnvPool, config: dict, *,
    episodes: int, seed: int, goal_scale: float, orientation_scale: float,
    position_tolerance: float, orientation_tolerance: float,
    scene: str = "none",
    target_distance_min_m: float | None = None,
    target_distance_max_m: float | None = None,
    target_orientation_min_rad: float | None = None,
    target_orientation_max_rad: float | None = None,
) -> dict[str, object]:
    """Evaluate fixed episode seeds in parallel without changing their order."""
    base_contract = {
        "scene": scene, "xi": 1.0, "strict": True,
        "goal_scale": goal_scale, "orientation_scale": orientation_scale,
        "position_tolerance": position_tolerance,
        "orientation_tolerance": orientation_tolerance, "lambda_self": 1.0,
    }
    position_bins = int(config["thesis"]["orientation_curriculum"].get(
        "deterministic_probe_position_bins", 10
    ))
    orientation_bins = int(config["thesis"]["orientation_curriculum"].get(
        "deterministic_probe_orientation_bins", 10
    ))
    position_edges = None if target_distance_max_m is None else np.linspace(
        target_distance_min_m, target_distance_max_m, position_bins + 1
    )
    orientation_edges = None if target_orientation_max_rad is None else np.linspace(
        target_orientation_min_rad, target_orientation_max_rad, orientation_bins + 1
    )
    active: dict[int, dict[str, object]] = {}
    outcomes: dict[int, tuple[float, ...]] = {}
    large_angle_steps = 0
    orientation_improvements = 0
    orientation_rebound_sum = 0.0
    keypoint_accumulator = _empty_keypoint_probe_accumulator()
    next_episode = 0
    self_safe_distance = float(
        config["thesis"]["self_collision"]["safe_distance_m"]
    )
    while len(outcomes) < int(episodes):
        assignments: dict[int, int] = {}
        for worker_index in range(len(pool)):
            if worker_index in active or next_episode >= int(episodes):
                continue
            episode = next_episode
            next_episode += 1
            assignments[worker_index] = episode
            active[worker_index] = {
                "episode": episode,
                "minimum_self_distance": float("inf"),
                "self_violation": False,
                "strict_pose_hit": False,
                "bin_index": -1,
            }
        if assignments:
            contracts = {}
            for worker_index, episode in assignments.items():
                contract = dict(base_contract)
                if position_edges is not None and orientation_edges is not None:
                    position_bin = (episode // orientation_bins) % position_bins
                    orientation_bin = episode % orientation_bins
                    contract.update({
                        "target_distance_min_m": float(position_edges[position_bin]),
                        "target_distance_max_m": float(position_edges[position_bin + 1]),
                        "target_orientation_min_rad": float(orientation_edges[orientation_bin]),
                        "target_orientation_max_rad": float(orientation_edges[orientation_bin + 1]),
                    })
                    active[worker_index]["bin_index"] = position_bin * orientation_bins + orientation_bin
                contracts[worker_index] = contract
            resets = pool.reset_many(
                contracts,
                seeds={
                    worker_index: int(seed) + episode
                    for worker_index, episode in assignments.items()
                },
            )
            for worker_index, (observation, _) in resets.items():
                active[worker_index]["observation"] = observation
        indices = sorted(active)
        observations = np.stack([
            active[index]["observation"] for index in indices
        ])
        actions = agent.select_actions(observations, deterministic=True)
        results = pool.step_many(dict(zip(indices, actions)))
        finished = []
        for index in indices:
            observation, _, _, terminated, truncated, info = results[index]
            previous_orientation = float(info["rho_orientation"])
            next_orientation = float(info["next_rho_orientation"])
            if previous_orientation > 1.6:
                large_angle_steps += 1
                orientation_improvements += int(next_orientation < previous_orientation)
                orientation_rebound_sum += max(
                    0.0, next_orientation - previous_orientation
                )
            state = active[index]
            _record_keypoint_probe_step(keypoint_accumulator, info)
            if "start_keypoint_distance" not in state:
                state["start_keypoint_distance"] = float(info["keypoint_distance"])
            state["observation"] = observation
            step_distance = float(info["control_self_min_distance"])
            state["minimum_self_distance"] = min(
                float(state["minimum_self_distance"]), step_distance,
            )
            state["self_violation"] = bool(
                state["self_violation"] or step_distance < self_safe_distance
            )
            state["strict_pose_hit"] = bool(
                state["strict_pose_hit"] or info.get("strict_pose_reached", False)
            )
            if not (terminated or truncated):
                continue
            collision = bool(
                info["obstacle_collision"]
                or info["self_collision"]
                or info["environment_collision"]
            )
            outcomes[int(state["episode"])] = (
                float(info["task_reached"]), float(collision),
                float(info["joint_limit"]), float(truncated),
                float(info["next_rho_position"]),
                float(info["next_rho_orientation"]),
                float(state["minimum_self_distance"]),
                float(state["self_violation"]),
                float(state["strict_pose_hit"]),
                float(state["bin_index"]),
            )
            keypoint_accumulator["episode_start_distances"].append(
                float(state["start_keypoint_distance"])
            )
            keypoint_accumulator["episode_end_distances"].append(
                float(info["next_keypoint_distance"])
            )
            finished.append(index)
        for index in finished:
            del active[index]
    values = np.asarray(
        [outcomes[index] for index in range(int(episodes))], dtype=np.float64,
    )
    return summarize_pose_probe_outcomes(
        values,
        has_bins=position_edges is not None,
        orientation_bins=orientation_bins,
        large_angle_steps=large_angle_steps,
        orientation_improvements=orientation_improvements,
        orientation_rebound_sum=orientation_rebound_sum,
        keypoint_statistics=_keypoint_probe_statistics(keypoint_accumulator),
    )


def deterministic_contract_passed(
    metrics: dict[str, float], success_floor: float, collision_ceiling: float,
    bin_success_floor: float | None = None,
) -> bool:
    return bool(
        metrics["success_rate"] >= float(success_floor)
        and (
            bin_success_floor is None
            or metrics.get("minimum_bin_success_rate", 0.0) >= float(bin_success_floor)
        )
        and metrics["collision_rate"] <= float(collision_ceiling)
        and metrics["joint_limit_rate"] == 0.0
    )


def run_pose_promotion_probe(
    agent: ThesisSACAgent, config: dict, curriculum: HomotopyCurriculum,
    settings: dict, pool: ParallelThesisEnvPool | None = None,
) -> dict[str, object]:
    """Screen cheaply, then confirm current performance and previous-level retention."""
    screen_episodes = int(settings["deterministic_probe_episodes"])
    screen_seed = int(settings["deterministic_probe_seed"])
    current_floor = float(settings["deterministic_probe_success_floor"])
    previous_floor = float(settings["deterministic_previous_success_floor"])
    current_collision_ceiling = float(
        settings["deterministic_probe_collision_ceiling"]
    )
    previous_collision_ceiling = float(
        settings["deterministic_previous_collision_ceiling"]
    )
    recovery_success_floor = float(
        settings.get("retention_recovery_success_floor", 0.90)
    )
    recovery_collision_ceiling = float(
        settings.get("retention_recovery_collision_ceiling", 0.05)
    )
    runner = (
        run_deterministic_pose_probe
        if pool is None else
        lambda agent, config, **kwargs: run_parallel_deterministic_pose_probe(
            agent, pool, config, **kwargs,
        )
    )
    saved_pool_states = None if pool is None else pool.states()
    evaluation_actor = (
        FrozenActorEvaluator(agent)
        if hasattr(agent, "frozen_actor_copy") else agent
    )
    try:
        current_contract = curriculum.pose_contract()
        current = runner(
            evaluation_actor, config, episodes=screen_episodes, seed=screen_seed,
            **current_contract,
        )
        task_space_probe = "target_distance_max_m" in current_contract
        cross_level: dict[str, dict[str, object]] = {}
        if bool(settings.get("cross_level_probe_enabled", False)):
            pose_levels = config["thesis"]["joint_pose_curriculum"]["levels"]
            orientation_levels = config["thesis"]["orientation_curriculum"]["levels"]
            for index, level in enumerate(pose_levels):
                label = f"P{index + 1}"
                if index == int(curriculum.orientation.level_index):
                    cross_level[label] = current
                    continue
                cross_level[label] = runner(
                    evaluation_actor, config,
                    episodes=screen_episodes, seed=screen_seed,
                    goal_scale=float(level["goal_scale"]),
                    orientation_scale=float(orientation_levels[index]),
                    position_tolerance=float(level["position_tolerance_m"]),
                    orientation_tolerance=float(level["orientation_tolerance_rad"]),
                    target_distance_min_m=float(level["target_distance_min_m"]),
                    target_distance_max_m=float(level["target_distance_max_m"]),
                    target_orientation_min_rad=float(level["target_orientation_min_rad"]),
                    target_orientation_max_rad=float(level["target_orientation_max_rad"]),
                )
        has_previous = curriculum.orientation.level_index > 0
        previous = None
        if has_previous:
            previous_label = f"P{int(curriculum.orientation.level_index)}"
            if previous_label in cross_level:
                previous = cross_level[previous_label]
            else:
                previous = runner(
                    evaluation_actor, config,
                    episodes=screen_episodes, seed=screen_seed,
                    **curriculum.pose_contract(previous=True),
                )
        screen_current_passed = deterministic_contract_passed(
            current, current_floor, current_collision_ceiling,
            settings.get("deterministic_probe_bin_success_floor") if task_space_probe else None,
        )
        screen_previous_passed = bool(
            previous is None
            or deterministic_contract_passed(
                previous, previous_floor, previous_collision_ceiling,
            )
        )
        screen_passed = bool(screen_current_passed and screen_previous_passed)
        confirmation_current = None
        confirmation_previous = None
        confirmation_passed = False
        if screen_passed and not task_space_probe:
            confirmation_episodes = int(settings["deterministic_confirmation_episodes"])
            confirmation_seed = int(settings["deterministic_confirmation_seed"])
            confirmation_current = runner(
                evaluation_actor, config, episodes=confirmation_episodes,
                seed=confirmation_seed, **curriculum.pose_contract(),
            )
            if has_previous:
                confirmation_previous = runner(
                    evaluation_actor, config, episodes=confirmation_episodes,
                    seed=confirmation_seed, **curriculum.pose_contract(previous=True),
                )
            confirmation_current_passed = deterministic_contract_passed(
                confirmation_current, current_floor, current_collision_ceiling,
            )
            confirmation_previous_passed = bool(
                confirmation_previous is None
                or deterministic_contract_passed(
                    confirmation_previous, previous_floor,
                    previous_collision_ceiling,
                )
            )
            confirmation_passed = bool(
                confirmation_current_passed and confirmation_previous_passed
            )
        else:
            confirmation_current_passed = bool(screen_current_passed if task_space_probe else False)
            confirmation_previous_passed = bool(screen_previous_passed if task_space_probe else False)
            if task_space_probe:
                confirmation_passed = bool(screen_passed)
    finally:
        if pool is not None and saved_pool_states is not None:
            pool.restore_many(saved_pool_states)
    recorded_previous = (
        confirmation_previous
        if confirmation_previous is not None
        else previous
    )
    recovery_required = bool(
        recorded_previous is not None
        and (
            recorded_previous["success_rate"] < recovery_success_floor
            or recorded_previous["collision_rate"] > recovery_collision_ceiling
            or recorded_previous["joint_limit_rate"] > 0.0
        )
    )
    return {
        "current": current,
        "cross_level": cross_level,
        "previous": previous,
        "screen_passed": screen_passed,
        "confirmation_current": confirmation_current,
        "confirmation_previous": confirmation_previous,
        "confirmation_passed": confirmation_passed,
        "retention_passed": (
            confirmation_previous_passed
            if confirmation_previous is not None
            else screen_previous_passed
        ),
        "recovery_required": recovery_required,
        "passed": confirmation_passed,
    }


def pose_probe_audit_fields(
    result: dict[str, object], settings: dict,
) -> dict[str, object]:
    previous = result["previous"]
    confirmation_current = result["confirmation_current"]
    confirmation_previous = result["confirmation_previous"]

    def metric(values: object, key: str) -> object:
        return "" if values is None else values[key]  # type: ignore[index]

    cross_level = result.get("cross_level", {})
    if not isinstance(cross_level, dict):
        cross_level = {}
    return {
        "previous_success_rate": metric(previous, "success_rate"),
        "previous_collision_rate": metric(previous, "collision_rate"),
        "previous_joint_limit_rate": metric(previous, "joint_limit_rate"),
        "screen_passed": int(bool(result["screen_passed"])),
        "confirmation_ran": int(confirmation_current is not None),
        "confirmation_episodes": int(settings["deterministic_confirmation_episodes"]),
        "confirmation_seed": int(settings["deterministic_confirmation_seed"]),
        "confirmation_success_rate": metric(confirmation_current, "success_rate"),
        "confirmation_collision_rate": metric(confirmation_current, "collision_rate"),
        "confirmation_joint_limit_rate": metric(confirmation_current, "joint_limit_rate"),
        "confirmation_previous_success_rate": metric(
            confirmation_previous, "success_rate"
        ),
        "confirmation_previous_collision_rate": metric(
            confirmation_previous, "collision_rate"
        ),
        "confirmation_previous_joint_limit_rate": metric(
            confirmation_previous, "joint_limit_rate"
        ),
        "confirmation_passed": int(bool(result["confirmation_passed"])),
        "retention_passed": int(bool(result["retention_passed"])),
        "recovery_required": int(bool(result["recovery_required"])),
        "cross_level_success_rates": json.dumps({
            label: float(values["success_rate"])
            for label, values in cross_level.items()
        }, sort_keys=True),
        "cross_level_minimum_bin_success_rates": json.dumps({
            label: float(values.get("minimum_bin_success_rate", float("nan")))
            for label, values in cross_level.items()
        }, sort_keys=True),
    }


def run_manual_pose_checkpoint_probe(
    agent: ThesisSACAgent, config: dict, curriculum: HomotopyCurriculum,
    settings: dict, counters: dict, pool: ParallelThesisEnvPool | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate an exact checkpoint actor without changing curriculum or policy state."""
    level_index = curriculum.orientation.level_index
    scale = curriculum.orientation.scale
    level_steps = curriculum.orientation.current_level_steps
    phase = curriculum.s0_phase
    lambda_self = curriculum.lambda_self
    result = run_pose_promotion_probe(
        agent, config, curriculum, settings, pool=pool,
    )
    # In manual-promotion mode the probe is observational for intermediate
    # levels, but the final P5 probe is the formal frozen evidence used by the
    # S0->S1 gate.  Persist those metrics because there is no later promotion
    # command that would otherwise record them in the curriculum checkpoint.
    if (
        curriculum.precision_only_task_space
        and curriculum.orientation_at_full_scale
    ):
        metrics_for_gate = result["current"]
        previous_for_gate = result["previous"]
        assert isinstance(metrics_for_gate, dict)
        assert previous_for_gate is None or isinstance(previous_for_gate, dict)
        curriculum.record_orientation_probe(
            success_rate=float(metrics_for_gate["success_rate"]),
            collision_rate=float(metrics_for_gate["collision_rate"]),
            joint_limit_rate=float(metrics_for_gate["joint_limit_rate"]),
            previous_success_rate=(
                None if previous_for_gate is None
                else float(previous_for_gate["success_rate"])
            ),
            previous_collision_rate=(
                None if previous_for_gate is None
                else float(previous_for_gate["collision_rate"])
            ),
            previous_joint_limit_rate=(
                None if previous_for_gate is None
                else float(previous_for_gate["joint_limit_rate"])
            ),
            confirmation_passed=bool(result["confirmation_passed"]),
            retention_passed=bool(result["retention_passed"]),
            recovery_required=bool(result["recovery_required"]),
            passed=bool(result["passed"]),
        )
    update_adaptive_task_space_sampling(config, counters, result)
    metrics = result["current"]
    assert isinstance(metrics, dict)
    counters["last_manual_pose_probe"] = {
        "global_step": int(counters["global_step"]),
        "stage_step": int(counters["stage_step"]),
        "stage_total_step": int(counters["stage_total_step"]),
        "orientation_level_index": int(level_index),
        "orientation_level_steps": int(level_steps),
        "passed": bool(result["passed"]),
        "metrics": dict(metrics),
        "previous_metrics": (
            None if result["previous"] is None
            else dict(result["previous"])
        ),
        "retention_passed": bool(result["retention_passed"]),
        "cross_level": {
            str(label): dict(values)
            for label, values in result.get("cross_level", {}).items()
        },
    }
    row = {
        "global_step": counters["global_step"],
        "stage_step": counters["stage_step"],
        "stage_total_step": counters["stage_total_step"],
        "orientation_level_index": level_index,
        "orientation_scale": scale,
        "orientation_level_steps": level_steps,
        "episodes": settings["deterministic_probe_episodes"],
        "seed": settings["deterministic_probe_seed"],
        **scalar_probe_metrics(metrics),
        **pose_probe_audit_fields(result, settings),
        "s0_phase": phase,
        "lambda_self": lambda_self,
        "probe_track": "manual_checkpoint",
        "passed": int(bool(result["passed"])),
        "is_best": 0,
        "bad_probe_count": "",
        "rolled_back": 0,
    }
    return result, row


def s0_probe_track_key(curriculum: HomotopyCurriculum) -> str:
    """Keep pose and self-safety rollback checkpoints in separate tracks."""
    level = curriculum.orientation.level_index
    phase = curriculum.s0_phase
    if phase in {"joint_pose", "position", "pose"}:
        return f"level_{level:02d}_pose"
    if phase == "self_safety_ramp":
        bucket = min(10, max(0, int(np.floor(curriculum.lambda_self * 10.0 + 1e-9))))
        return f"level_{level:02d}_self_ramp_{bucket:02d}"
    return f"level_{level:02d}_self_full"


S0_PROBE_SCORE_VERSION = 2


def s0_probe_score(
    track: str, probe_result: dict[str, object], settings: dict,
    *, self_collision_ceiling: float,
) -> tuple[float, ...]:
    """Rank rollback checkpoints by the same feasibility contract as the Gate.

    A confirmation result is statistically stronger than its 50-episode screen,
    so it is used whenever it exists. Gate feasibility and hard-safety
    feasibility precede raw success in the tuple: an unsafe high-success actor
    must not overwrite a feasible checkpoint. Self-safety tracks additionally
    use the stricter final S0 collision ceiling.
    """
    confirmation = probe_result["confirmation_current"]
    metrics = (
        confirmation if confirmation is not None else probe_result["current"]
    )
    assert isinstance(metrics, dict)
    confirmation_previous = probe_result["confirmation_previous"]
    previous = (
        confirmation_previous
        if confirmation is not None else probe_result["previous"]
    )
    assert previous is None or isinstance(previous, dict)

    current_collision_ceiling = float(
        self_collision_ceiling
        if "_self_" in track
        else settings["deterministic_probe_collision_ceiling"]
    )
    previous_collision_ceiling = float(
        settings["deterministic_previous_collision_ceiling"]
    )
    current_safety_feasible = bool(
        metrics["collision_rate"] <= current_collision_ceiling
        and metrics["joint_limit_rate"] == 0.0
    )
    previous_safety_feasible = bool(
        previous is None
        or (
            previous["collision_rate"] <= previous_collision_ceiling
            and previous["joint_limit_rate"] == 0.0
        )
    )
    current_task_feasible = bool(
        metrics["success_rate"]
        >= float(settings["deterministic_probe_success_floor"])
    )
    previous_task_feasible = bool(
        previous is None
        or previous["success_rate"]
        >= float(settings["deterministic_previous_success_floor"])
    )
    gate_feasible = bool(
        probe_result["passed"]
        and metrics["collision_rate"] <= current_collision_ceiling
    )
    score = (
        float(gate_feasible),
        float(current_safety_feasible and previous_safety_feasible),
        float(current_safety_feasible),
        float(previous_safety_feasible),
        float(current_task_feasible and previous_task_feasible),
        float(metrics["success_rate"]),
        1.0 if previous is None else float(previous["success_rate"]),
        -float(metrics["collision_rate"]),
        0.0 if previous is None else -float(previous["collision_rate"]),
    )
    if "_self_" in track:
        score += (
            -float(metrics["self_violation_rate"]),
            float(metrics["mean_minimum_self_distance_m"]),
        )
    return score + (
        -float(metrics["timeout_rate"]),
        -float(metrics["mean_position_error_m"]),
        -float(metrics["mean_orientation_error_rad"]),
    )


def s0_probe_score_is_better(
    track: str, score: tuple[float, ...], previous: dict[str, object] | None,
    settings: dict, *, self_collision_ceiling: float,
) -> bool:
    """Compare checkpoints produced by the current serial protocol only."""
    if previous is None:
        return True
    previous_score = tuple(float(value) for value in previous.get("score", ()))
    if int(previous.get("score_version", -1)) != S0_PROBE_SCORE_VERSION:
        return False
    return score > previous_score


def run_parallel_s0(
    *, config: dict, env_template: ThesisHomotopyEnv, agent: ThesisSACAgent,
    replay: HomotopyReplayBuffer, curriculum: HomotopyCurriculum,
    counters: dict, output: Path, checkpoints: Path, fields: list[str],
    transition_fields: list[str], update_fields: list[str],
    progress_fields: list[str], total_steps: int, max_stage_steps: int,
    save_interval: int, progress_interval: int, log_flush_interval: int,
    batch_size: int, update_after: int, num_envs: int,
    updates_per_transition: float,
    orientation_curriculum: dict, full_scale_min_transitions: int,
    pose_full_scale_min_transitions: int, loaded_checkpoint_state: dict | None,
    bad_state_starts: dict | None = None, bad_state_fraction: float = .25,
) -> None:
    """Run synchronous multi-process S0 collection with one centralized learner."""
    stream_seed = int(config["_rng_stream_seeds"]["environment"])
    seed_children = np.random.SeedSequence(stream_seed).spawn(num_envs)
    worker_seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in seed_children
    ]
    env_template.close()
    pool = ParallelThesisEnvPool(config, worker_seeds)
    workers: list[dict | None] = [None] * num_envs
    counters.setdefault("next_episode_id", counters["episodes"])
    manual_promotion = bool(orientation_curriculum.get("manual_promotion", False))
    evaluate_during_training = bool(
        config["train"].get("evaluate_during_training", False)
    )
    manual_probe_interval = int(
        orientation_curriculum.get(
            "manual_probe_interval_transitions", save_interval,
        )
    )
    if manual_probe_interval < 1 or manual_probe_interval % save_interval != 0:
        raise ValueError(
            "manual_probe_interval_transitions must be a positive multiple of "
            "train.save_interval"
        )
    pending_probe = False
    last_update: dict[str, float] = {}
    last_saved_step = -1
    best_probes: dict[str, dict[str, object]] = dict(counters.get("probe_best", {}))
    counters["probe_best"] = best_probes
    targeted = None
    target_rng = np.random.default_rng(stream_seed ^ 0xBAD57A7E)
    if bad_state_starts is not None:
        targeted = counters.setdefault("bad_state_sampling", {
            "source_checkpoint": bad_state_starts["source_checkpoint"],
            "fraction": bad_state_fraction, "starts": 0,
            "targeted_transitions": 0, "normal_transitions": 0,
            "normal_length_ema": max(1., counters["global_step"] / max(1, counters["episodes"])),
            "targeted_length_ema": float(np.mean([
                config["thesis"]["horizon"] - c["step"] for c in bad_state_starts["centers"]
            ])),
        })
        if "rng_state" in targeted:
            target_rng.bit_generator.state = targeted["rng_state"]

    if loaded_checkpoint_state is not None:
        saved_workers = loaded_checkpoint_state["parallel_workers"]
        states = {index: saved["environment"] for index, saved in enumerate(saved_workers)}
        restored_observations = pool.restore_many(states)
        for index, saved in enumerate(saved_workers):
            if saved["active"]:
                worker = dict(saved["episode"])
                worker["observation"] = restored_observations[index]
                workers[index] = worker
        if evaluate_during_training and not manual_promotion:
            pending_probe = curriculum.orientation_probe_due(
                int(orientation_curriculum["deterministic_probe_interval_transitions"]),
                pose_full_scale_min_transitions,
            )

    def reset_idle() -> None:
        requests: dict[int, dict] = {}
        targeted_requests: dict[int, tuple[dict, dict]] = {}
        metadata: dict[int, dict] = {}
        for index, worker in enumerate(workers):
            if worker is not None:
                continue
            scene = curriculum.choose_scene()
            xi, strict = curriculum.contract(scene)
            orientation_anchor = curriculum.choose_orientation_anchor()
            pose_contract = curriculum.pose_contract(previous=orientation_anchor)
            episode_id = int(counters["next_episode_id"])
            counters["next_episode_id"] += 1
            contract = {
                "scene": scene, "xi": xi, "strict": strict,
                "goal_scale": pose_contract["goal_scale"],
                "orientation_scale": pose_contract["orientation_scale"],
                "position_tolerance": pose_contract["position_tolerance"],
                "orientation_tolerance": pose_contract["orientation_tolerance"],
                "lambda_self": curriculum.lambda_self,
                **task_space_training_kwargs(
                    config, curriculum, anchor=orientation_anchor,
                    bin_probabilities=counters.get(
                        "task_space_bin_probabilities"
                    ),
                ),
            }
            requests[index] = contract
            metadata[index] = {
                **contract, "orientation_anchor": orientation_anchor,
                "episode_id": episode_id, "return": 0.0, "length": 0,
                "episode_used_warmup": False, "final": {},
                "pose_phase_complete_at_start": curriculum.pose_phase_complete,
                "self_weight_full_at_start": bool(np.isclose(
                    curriculum.lambda_self, curriculum.self_safety.end,
                    rtol=0.0, atol=1e-12,
                )),
            }
            if targeted is not None:
                normal_length = targeted["normal_length_ema"]
                bad_length = targeted["targeted_length_ema"]
                probability = bad_state_fraction * normal_length / (
                    (1. - bad_state_fraction) * bad_length + bad_state_fraction * normal_length
                )
                if target_rng.random() < probability:
                    center_index = int(target_rng.integers(len(bad_state_starts["centers"])))
                    center = bad_state_starts["centers"][center_index]
                    targeted_requests[index] = (center, contract)
                    del requests[index]
                    metadata[index].update({"length": int(center["step"]),
                                            "start_step": int(center["step"]),
                                            "bad_state_start": True,
                                            "frontier_center": center_index})
                    targeted["starts"] += 1
                targeted["rng_state"] = target_rng.bit_generator.state
        for index, (observation, _) in pool.reset_many(requests).items():
            workers[index] = {**metadata[index], "observation": observation}
        for index, (observation, _) in pool.reset_bad_states_many(targeted_requests).items():
            workers[index] = {**metadata[index], "observation": observation}

    def write_progress(writer: csv.DictWriter, started: float) -> None:
        goal_rate = float(np.mean(curriculum.goal.outcomes)) if curriculum.goal.outcomes else ""
        orientation_rate = (
            float(np.mean(curriculum.orientation.outcomes))
            if curriculum.orientation.outcomes else ""
        )
        replay_mix = curriculum.orientation_replay_mix
        storage = replay.orientation_storage_counts()
        elapsed = max(time.perf_counter() - started, 1e-9)
        row = {
            "global_step": counters["global_step"], "stage_step": counters["stage_step"],
            "stage_total_step": counters["stage_total_step"], "episodes": counters["episodes"],
            "replay_size": len(replay), "updates": counters["updates"],
            "alpha": last_update.get("alpha", float(agent.alpha.detach().cpu())),
            "bad_state_starts": 0 if targeted is None else targeted["starts"],
            "bad_state_transitions": 0 if targeted is None else targeted["targeted_transitions"],
            "bad_state_transition_fraction": 0. if targeted is None else (
                targeted["targeted_transitions"] / max(1, targeted["targeted_transitions"] + targeted["normal_transitions"])
            ),
            "critic_loss": last_update.get("critic_loss", ""),
            "actor_loss": last_update.get("actor_loss", ""),
            "actor_sac_loss": last_update.get("actor_sac_loss", ""),
            "policy_churn_kl": last_update.get("policy_churn_kl", ""),
            "policy_churn_penalty": last_update.get(
                "policy_churn_penalty", ""
            ),
            "policy_churn_to_sac_ratio": last_update.get(
                "policy_churn_to_sac_ratio", ""
            ),
            "chain_pcr_effective_coefficient": last_update.get(
                "chain_pcr_effective_coefficient", ""
            ),
            "chain_pcr_sac_loss_ema": last_update.get(
                "chain_pcr_sac_loss_ema", ""
            ),
            "chain_pcr_loss_ema": last_update.get("chain_pcr_loss_ema", ""),
            **keypoint_reward_statistics(counters),
            "q1_mean": last_update.get("q1_mean", ""),
            "q2_mean": last_update.get("q2_mean", ""),
            "target_mean": last_update.get("target_mean", ""),
            "steps_per_second": counters["stage_step"] / elapsed,
            "goal_scale": curriculum.goal_scale,
            "goal_level_index": curriculum.goal.level_index,
            "goal_level_steps": curriculum.goal.current_level_steps,
            "rolling_goal_success_rate": goal_rate,
            "goal_eligible_steps": curriculum.goal.eligible_steps,
            "goal_full_scale_steps": curriculum.goal.full_scale_steps,
            "s0_goal_gate_eligible": int(curriculum.s0_goal_gate_eligible(
                full_scale_min_transitions, pose_full_scale_min_transitions,
            )),
            "orientation_scale": curriculum.orientation.scale,
            "position_tolerance": curriculum.position_tolerance,
            "orientation_tolerance": curriculum.orientation_tolerance,
            "rolling_orientation_success_rate": orientation_rate,
            "orientation_eligible_steps": curriculum.orientation.eligible_steps,
            "orientation_level_index": curriculum.orientation.level_index,
            "orientation_level_steps": curriculum.orientation.current_level_steps,
            "rolling_anchor_position_success_rate": (
                "" if curriculum.orientation_anchor_success_rate is None
                else curriculum.orientation_anchor_success_rate
            ),
            "rolling_previous_pose_success_rate": (
                "" if curriculum.orientation_anchor_success_rate is None
                else curriculum.orientation_anchor_success_rate
            ),
            "sample_orientation_anchor": replay.last_orientation_sample_counts["anchor"],
            "sample_orientation_current": replay.last_orientation_sample_counts["current"],
            "sample_orientation_historical": replay.last_orientation_sample_counts["historical"],
            "sample_current_success": replay.last_orientation_sample_counts["current_success"],
            "sample_current_recent": replay.last_orientation_sample_counts["current_recent"],
            "sample_previous_level": replay.last_orientation_sample_counts["previous"],
            "sample_semantic_long_term": replay.last_orientation_sample_counts.get("semantic_long_term", 0),
            "sample_stability_anchor": replay.last_orientation_sample_counts.get("stability_anchor", 0),
            "sample_frontier": replay.last_orientation_sample_counts.get("frontier", 0),
            "stored_stability_anchor": sum(p.size for p in replay.stability_anchor.values()),
            "stored_frontier": sum(p.size for p in replay.frontier.values()),
            "full_pose_steps": curriculum.orientation.full_scale_steps,
            "position_phase_complete": int(curriculum.position_phase_complete),
            "orientation_retention_mode": curriculum.orientation_retention_mode,
            "orientation_anchor_probability": curriculum.orientation_anchor_probability,
            "replay_orientation_anchor_fraction": .25 if replay.four_pool_enabled else (0. if replay.four_pool_history_mode else replay_mix[0]),
            "replay_orientation_current_fraction": .375 if replay.four_pool_enabled else (replay.four_pool_current_fraction if replay.four_pool_history_mode else replay_mix[1]),
            "replay_orientation_historical_fraction": (1. - replay.four_pool_current_fraction) if replay.four_pool_history_mode else (0. if replay.four_pool_enabled else replay_mix[2]),
            "replay_current_success_fraction": 0. if (replay.four_pool_enabled or replay.four_pool_history_mode) else replay.effective_s0_success_fraction,
            "replay_current_recent_fraction": (
                .375 if replay.four_pool_enabled else
                replay.four_pool_current_fraction if replay.four_pool_history_mode else
                1.0 - replay_mix[0] - replay.effective_s0_success_fraction - replay.semantic_fraction
            ),
            "replay_semantic_long_term_fraction": .1875 if replay.four_pool_enabled else (0. if replay.four_pool_history_mode else replay.semantic_fraction),
            "replay_previous_level_fraction": (1. - replay.four_pool_current_fraction) if replay.four_pool_history_mode else (0. if replay.four_pool_enabled else replay_mix[0]),
            "stored_orientation_anchor": storage["anchor"],
            "stored_orientation_current": storage["current"],
            "stored_orientation_historical": storage["historical"],
            "stored_orientation_historical_levels": len(storage["historical_levels"]),
            "stored_current_success_episodes": replay.current_success_episode_count,
            "stored_semantic_long_term_transitions": storage.get("semantic_long_term_transitions", 0),
            "stored_semantic_long_term_episodes": storage.get("semantic_long_term_episodes", 0),
            "stored_semantic_long_term_nonempty_bins": storage.get("semantic_long_term_nonempty_bins", 0),
            "deterministic_probe_passed": int(curriculum.orientation.deterministic_probe_passed),
            "deterministic_probe_success_rate": (
                "" if curriculum.orientation.deterministic_probe_success_rate is None
                else curriculum.orientation.deterministic_probe_success_rate
            ),
            "deterministic_probe_collision_rate": (
                "" if curriculum.orientation.deterministic_probe_collision_rate is None
                else curriculum.orientation.deterministic_probe_collision_rate
            ),
            "deterministic_probe_joint_limit_rate": (
                "" if curriculum.orientation.deterministic_probe_joint_limit_rate is None
                else curriculum.orientation.deterministic_probe_joint_limit_rate
            ),
            "deterministic_probe_previous_success_rate": (
                "" if curriculum.orientation.deterministic_probe_previous_success_rate is None
                else curriculum.orientation.deterministic_probe_previous_success_rate
            ),
            "deterministic_probe_confirmation_passed": int(
                curriculum.orientation.deterministic_probe_confirmation_passed
            ),
            "lambda_self": curriculum.lambda_self, "s0_phase": curriculum.s0_phase,
            "pose_phase_complete": int(curriculum.pose_phase_complete),
            "self_safety_eligible_steps": curriculum.self_safety.eligible_steps,
            "self_safety_full_weight_steps": curriculum.self_safety.full_weight_steps,
        }
        writer.writerow(row)
        print(
            f"step={counters['stage_step']}/{total_steps} envs={num_envs} "
            f"episodes={counters['episodes']} replay={len(replay)} "
            f"updates={counters['updates']} level={curriculum.orientation.level_index} "
            f"success={orientation_rate} steps/s={row['steps_per_second']:.1f}",
            flush=True,
        )

    log_path = output / "episodes.csv"
    with log_path.open("w", newline="", encoding="utf-8") as episode_handle, \
            (output / "transitions.csv").open("w", newline="", encoding="utf-8") as transition_handle, \
            (output / "updates.csv").open("w", newline="", encoding="utf-8") as update_handle, \
            (output / "progress.csv").open("w", newline="", encoding="utf-8") as progress_handle, \
            (output / "orientation_probes.csv").open("w", newline="", encoding="utf-8") as probe_handle:
        episode_writer = csv.DictWriter(episode_handle, fieldnames=fields)
        transition_writer = csv.DictWriter(transition_handle, fieldnames=transition_fields)
        update_writer = csv.DictWriter(update_handle, fieldnames=update_fields)
        progress_writer = csv.DictWriter(progress_handle, fieldnames=progress_fields)
        probe_fields = [
            "global_step", "stage_step", "stage_total_step", "orientation_level_index",
            "orientation_scale", "orientation_level_steps", "episodes", "seed",
            "success_rate", "collision_rate", "joint_limit_rate", "timeout_rate",
            "mean_position_error_m", "mean_orientation_error_rad",
            "p95_orientation_error_rad",
            "mean_minimum_self_distance_m", "self_violation_rate",
            "strict_pose_hit_rate", "minimum_bin_success_rate",
            "minimum_bin_strict_pose_hit_rate",
            "weak_bin_count_below_0_6", "success_rate_o0_o4",
            "success_rate_o5_o6", "success_rate_o7_o9",
            "timeout_rate_o0_o4", "timeout_rate_o5_o6", "timeout_rate_o7_o9",
            "large_angle_orientation_steps",
            "large_angle_orientation_improvement_ratio",
            "large_angle_orientation_mean_rebound_rad",
            "previous_success_rate", "previous_collision_rate",
            "previous_joint_limit_rate", "screen_passed", "confirmation_ran",
            "confirmation_episodes", "confirmation_seed", "confirmation_success_rate",
            "confirmation_collision_rate", "confirmation_joint_limit_rate",
            "confirmation_previous_success_rate", "confirmation_previous_collision_rate",
            "confirmation_previous_joint_limit_rate", "confirmation_passed",
            "retention_passed", "recovery_required",
            "cross_level_success_rates",
            "cross_level_minimum_bin_success_rates",
            "s0_phase", "lambda_self", "probe_track", "passed", "is_best",
            "bad_probe_count", "rolled_back",
        ]
        probe_writer = csv.DictWriter(probe_handle, fieldnames=probe_fields)
        for writer in (episode_writer, transition_writer, update_writer, progress_writer, probe_writer):
            writer.writeheader()

        def perform_deferred_updates() -> None:
            """Consume updates earned by the preceding environment batch.

            The caller invokes this after dispatching the next simulator step,
            so PyBullet CPU work and SAC GPU work overlap.  At checkpoint and
            final boundaries it is also called without an in-flight step to
            make the configured UTD exact before state is serialized.
            """
            nonlocal last_update
            updates_due = int(counters.get("parallel_updates_due", 0))
            counters["parallel_updates_due"] = 0
            for _ in range(updates_due):
                anchor_fraction, current_fraction, _ = (
                    curriculum.orientation_replay_mix
                )
                update_actor = actor_updates_enabled(counters)
                reference_observations = None
                if (
                    agent.chain_pcr_enabled
                    and update_actor
                    and (agent.update_steps + 1) % agent.actor_update_interval == 0
                ):
                    reference_batch = replay.sample(
                        "s0", curriculum.xi_map(), batch_size,
                        orientation_scale=curriculum.orientation.scale,
                        s0_anchor_fraction=anchor_fraction,
                        s0_current_fraction=current_fraction,
                        lambda_self=curriculum.lambda_self,
                    )
                    reference_observations = reference_batch.observations
                batch = replay.sample(
                    "s0", curriculum.xi_map(), batch_size,
                    orientation_scale=curriculum.orientation.scale,
                    s0_anchor_fraction=anchor_fraction,
                    s0_current_fraction=current_fraction,
                    lambda_self=curriculum.lambda_self,
                )
                last_update = agent.update(
                    batch,
                    update_actor=update_actor,
                    reference_observations=reference_observations,
                )
                counters["updates"] += 1
                update_writer.writerow({
                    "global_step": counters["global_step"],
                    "stage_step": counters["stage_step"],
                    "stage_total_step": counters["stage_total_step"],
                    "update": counters["updates"], **last_update,
                    "sample_none": replay.last_sample_counts["none"],
                    "sample_static": replay.last_sample_counts["static"],
                    "sample_dynamic": replay.last_sample_counts["dynamic"],
                    "sample_orientation_anchor": replay.last_orientation_sample_counts["anchor"],
                    "sample_orientation_current": replay.last_orientation_sample_counts["current"],
                    "sample_orientation_historical": replay.last_orientation_sample_counts["historical"],
                    "sample_current_success": replay.last_orientation_sample_counts["current_success"],
                    "sample_current_recent": replay.last_orientation_sample_counts["current_recent"],
                    "sample_previous_level": replay.last_orientation_sample_counts["previous"],
                    "sample_semantic_long_term": replay.last_orientation_sample_counts.get("semantic_long_term", 0),
                    "sample_stability_anchor": replay.last_orientation_sample_counts.get("stability_anchor", 0),
                    "sample_frontier": replay.last_orientation_sample_counts.get("frontier", 0),
                })

        started = time.perf_counter()
        training_completed = False
        try:
            while counters["stage_step"] < total_steps:
                if pending_probe and all(worker is None for worker in workers):
                    perform_deferred_updates()
                    level_index = curriculum.orientation.level_index
                    scale = curriculum.orientation.scale
                    probe_result = run_pose_promotion_probe(
                        agent, config, curriculum, orientation_curriculum,
                        pool=pool,
                    )
                    update_adaptive_task_space_sampling(
                        config, counters, probe_result,
                    )
                    metrics = probe_result["current"]
                    retention_metrics = probe_result["previous"]
                    recorded_metrics = (
                        probe_result["confirmation_current"] or metrics
                    )
                    recorded_retention = (
                        probe_result["confirmation_previous"] or retention_metrics
                    )
                    probe_phase = curriculum.s0_phase
                    probe_lambda = curriculum.lambda_self
                    track = s0_probe_track_key(curriculum)
                    passed = bool(probe_result["passed"])
                    curriculum.record_orientation_probe(
                        success_rate=recorded_metrics["success_rate"],
                        collision_rate=recorded_metrics["collision_rate"],
                        joint_limit_rate=recorded_metrics["joint_limit_rate"],
                        previous_success_rate=(
                            None if recorded_retention is None
                            else recorded_retention["success_rate"]
                        ),
                        previous_collision_rate=(
                            None if recorded_retention is None
                            else recorded_retention["collision_rate"]
                        ),
                        previous_joint_limit_rate=(
                            None if recorded_retention is None
                            else recorded_retention["joint_limit_rate"]
                        ),
                        confirmation_passed=bool(
                            probe_result["confirmation_passed"]
                        ),
                        retention_passed=bool(probe_result["retention_passed"]),
                        recovery_required=bool(probe_result["recovery_required"]),
                        passed=passed,
                    )
                    steps_at_probe = curriculum.orientation.current_level_steps
                    previous = best_probes.get(track)
                    score = s0_probe_score(
                        track, probe_result, orientation_curriculum,
                        self_collision_ceiling=curriculum.self_probe_collision_ceiling,
                    )
                    is_best = s0_probe_score_is_better(
                        track, score, previous, orientation_curriculum,
                        self_collision_ceiling=curriculum.self_probe_collision_ceiling,
                    )
                    rolled_back = False
                    if is_best:
                        best_path = checkpoints / f"best_{track}.pt"
                        best_probes[track] = {
                            "score": score, "success_rate": metrics["success_rate"],
                            "score_version": S0_PROBE_SCORE_VERSION,
                            "bad_count": 0, "path": str(best_path),
                        }
                        counters["probe_best"] = best_probes
                        save_parallel_checkpoint(
                            best_path, agent, replay, curriculum, counters, config,
                            pool, workers,
                        )
                    else:
                        if metrics["success_rate"] <= float(previous["success_rate"]) - float(
                            orientation_curriculum["rollback_success_drop"]
                        ):
                            previous["bad_count"] = int(previous["bad_count"]) + 1
                        else:
                            previous["bad_count"] = 0
                        if int(previous["bad_count"]) >= int(
                            orientation_curriculum["rollback_patience"]
                        ):
                            best_state = torch.load(
                                previous["path"], map_location=agent.device,
                                weights_only=False,
                            )
                            agent.load_state_dict(best_state["agent"])
                            factor = float(orientation_curriculum["rollback_actor_lr_factor"])
                            for group in agent.actor_optimizer.param_groups:
                                group["lr"] *= factor
                            previous["bad_count"] = 0
                            rolled_back = True
                    advanced = curriculum.advance_orientation_if_ready()
                    if advanced:
                        replay.begin_orientation_level(curriculum.orientation.scale)
                        contract = curriculum.pose_contract()
                        replay.update_semantic_bounds(
                            distance_min=contract["target_distance_min_m"],
                            distance_max=contract["target_distance_max_m"],
                            orientation_min=contract["target_orientation_min_rad"],
                            orientation_max=contract["target_orientation_max_rad"],
                        )
                    probe_writer.writerow({
                        "global_step": counters["global_step"],
                        "stage_step": counters["stage_step"],
                        "stage_total_step": counters["stage_total_step"],
                        "orientation_level_index": level_index, "orientation_scale": scale,
                        "orientation_level_steps": steps_at_probe,
                        "episodes": orientation_curriculum["deterministic_probe_episodes"],
                        "seed": orientation_curriculum["deterministic_probe_seed"],
                        **scalar_probe_metrics(metrics),
                        **pose_probe_audit_fields(
                            probe_result, orientation_curriculum,
                        ),
                        "s0_phase": probe_phase,
                        "lambda_self": probe_lambda,
                        "probe_track": track,
                        "passed": int(passed), "is_best": int(is_best),
                        "bad_probe_count": int(best_probes[track]["bad_count"]),
                        "rolled_back": int(rolled_back),
                    })
                    probe_handle.flush()
                    pending_probe = False
                if not pending_probe:
                    reset_idle()

                active = [index for index, worker in enumerate(workers) if worker is not None]
                if not active:
                    continue
                boundary = min(
                    total_steps,
                    ((counters["stage_step"] // save_interval) + 1) * save_interval,
                )
                count = min(len(active), boundary - counters["stage_step"])
                selected = active[:count]
                actions: dict[int, np.ndarray] = {}
                policy_indices = []
                for index in selected:
                    if counters["random_steps_used"] < int(config["sac"]["warmup_steps"]):
                        actions[index] = config["_warmup_action_rng"].uniform(
                            -1.0, 1.0, size=6,
                        ).astype(np.float32)
                        counters["random_steps_used"] += 1
                        workers[index]["episode_used_warmup"] = True
                    else:
                        policy_indices.append(index)
                if policy_indices:
                    observations = np.stack([
                        workers[index]["observation"] for index in policy_indices
                    ])
                    selected_actions = agent.select_actions(observations)
                    actions.update(zip(policy_indices, selected_actions))
                pool.send_steps(actions)
                perform_deferred_updates()
                results = pool.recv_steps()

                for index in selected:
                    worker = workers[index]
                    if targeted is not None:
                        key = "targeted_transitions" if worker.get("bad_state_start", False) else "normal_transitions"
                        targeted[key] += 1
                    next_observation, reward, _, terminated, truncated, info = results[index]
                    curriculum.record_transition(
                        worker["goal_scale"], worker["orientation_scale"],
                        orientation_anchor=worker["orientation_anchor"],
                        curriculum_eligible=not worker["episode_used_warmup"],
                    )
                    replay.add(
                        worker["scene"], worker["observation"], actions[index],
                        next_observation, terminated or truncated, info,
                        worker["episode_id"], worker["length"],
                        orientation_anchor=worker["orientation_anchor"],
                        curriculum_level=int(curriculum.orientation.level_index),
                    )
                    if worker.get("bad_state_start", False) and replay.four_pool_enabled:
                        error = np.asarray(info["orientation_error_vector"], dtype=float)
                        angular_velocity = np.asarray(info["ee_angular_velocity"], dtype=float)
                        norm = np.linalg.norm(error) * np.linalg.norm(angular_velocity)
                        alignment = (float(np.dot(error, angular_velocity) / norm)
                                     if norm > 1e-8 else None)
                        replay.add_frontier_from_recent(
                            worker["episode_id"], worker["frontier_center"],
                            worker["start_step"], recovered=False, alignment=alignment,
                        )
                        if (terminated or truncated) and info["task_reached"]:
                            replay.add_frontier_from_recent(
                                worker["episode_id"], worker["frontier_center"],
                                worker["start_step"], recovered=True,
                            )
                    record_keypoint_reward_statistics(
                        counters, info,
                        enabled=bool(config["thesis"]["reward"].get(
                            "keypoint_pose_reward", False,
                        )),
                    )
                    worker["observation"] = next_observation
                    worker["return"] += reward
                    worker["length"] += 1
                    worker["final"] = info
                    counters["global_step"] += 1
                    counters["stage_step"] += 1
                    counters["stage_total_step"] += 1
                    updates_due = (
                        accrue_update_credit(counters, updates_per_transition)
                        if replay.has_training_batch("s0", batch_size, update_after) else 0
                    )
                    counters["parallel_updates_due"] = (
                        int(counters.get("parallel_updates_due", 0)) + updates_due
                    )
                    transition_writer.writerow({
                        "global_step": counters["global_step"],
                        "stage_step": counters["stage_step"],
                        "stage_total_step": counters["stage_total_step"],
                        "episode": worker["episode_id"], "episode_step": worker["length"],
                        "bad_state_start": int(worker.get("bad_state_start", False)),
                        "scene": worker["scene"], "xi": worker["xi"],
                        "lambda_self": worker["lambda_self"], "strict": int(worker["strict"]),
                        "reward": reward, "r_goal": info["r_goal"],
                        "c_proximity": info["c_proximity"], "hard_penalty": info["hard_penalty"],
                        "timeout_penalty": info["timeout_penalty"],
                        "safety_penalty": info["safety_penalty"],
                        "external_safety_penalty": info["external_safety_penalty"],
                        "self_safety_penalty": info["self_safety_penalty"],
                        "terminal_guard_penalty": info["terminal_guard_penalty"],
                        "clearance_violation": info["clearance_violation"],
                        "risk_max": info["control_max_risk"],
                        "d_min": info["control_min_distance"],
                        "goal_scale": worker["goal_scale"],
                        "self_clearance_violation": info["self_clearance_violation"],
                        "self_risk_max": info["control_self_max_risk"],
                        "self_d_min": info["control_self_min_distance"],
                        "self_ttc_min": info["control_self_min_ttc"],
                        "self_approach_max": info["control_self_max_approach"],
                        "self_projection_intervened": int(info["self_projection_intervened"]),
                        "self_projection_infeasible": int(info["self_projection_infeasible"]),
                        "self_projection_correction_norm": info["self_projection_correction_norm"],
                        "self_projection_max_slack": info["self_projection_max_slack"],
                        "goal_full_scale_steps": curriculum.goal.full_scale_steps,
                        "orientation_scale": worker["orientation_scale"],
                        "position_tolerance": worker["position_tolerance"],
                        "orientation_tolerance": worker["orientation_tolerance"],
                        "orientation_reward_scale": info["orientation_reward_scale"],
                        "orientation_reward_gate": info["orientation_reward_gate"],
                        "rho_position": info["rho_position"],
                        "next_rho_position": info["next_rho_position"],
                        "rho_orientation": info["rho_orientation"],
                        "next_rho_orientation": info["next_rho_orientation"],
                        "position_quality": info["position_quality"],
                        "orientation_quality": info["orientation_quality"],
                        "keypoint_tracking_quality": info["keypoint_tracking_quality"],
                        "jacobian_clip_ratio": info["jacobian_clip_ratio"],
                        "keypoint_distance": info["keypoint_distance"],
                        "next_keypoint_distance": info["next_keypoint_distance"],
                        "keypoint_progress": info["keypoint_progress"],
                        "keypoint_tracking_reward": info["keypoint_tracking_reward"],
                        "keypoint_progress_reward": info["keypoint_progress_reward"],
                        "keypoint_precision_quality": info["keypoint_precision_quality"],
                        "keypoint_precision_reward": info["keypoint_precision_reward"],
                        "pose_potential": info["pose_potential"],
                        "next_pose_potential": info["next_pose_potential"],
                        "pose_potential_progress": info["pose_potential_progress"],
                        "position_progress": info["position_progress"],
                        "orientation_progress": info["orientation_progress"],
                        "orientation_error_progress": info["orientation_error_progress"],
                        "orientation_absolute_penalty": info["orientation_absolute_penalty"],
                        "orientation_absolute_position_gate": info[
                            "orientation_absolute_position_gate"
                        ],
                        "orientation_error_progress_reward": info["orientation_error_progress_reward"],
                        "orientation_completion_reward": info["orientation_completion_reward"],
                        "orientation_completion_position_gate": info[
                            "orientation_completion_position_gate"
                        ],
                        "orientation_shaping_reward": info["orientation_shaping_reward"],
                        "fine_position_quality": info["fine_position_quality"],
                        "fine_orientation_quality": info["fine_orientation_quality"],
                        "fine_position_progress": info["fine_position_progress"],
                        "fine_orientation_progress": info["fine_orientation_progress"],
                        "precision_proximity": info["precision_proximity"],
                        "precision_stop_cost": info["precision_stop_cost"],
                        "joint_tolerance_ratio": info["joint_tolerance_ratio"],
                        "next_joint_tolerance_ratio": info["next_joint_tolerance_ratio"],
                        "joint_precision_quality": info["joint_precision_quality"],
                        "next_joint_precision_quality": info["next_joint_precision_quality"],
                        "joint_precision_progress": info["joint_precision_progress"],
                        "joint_precision_reward": info["joint_precision_reward"],
                        "position_completion": info["position_completion"],
                        "orientation_completion": info["orientation_completion"],
                        "partial_precision_reward": info[
                            "partial_precision_reward"
                        ],
                        "left_joint_tolerance_region": int(
                            info["left_joint_tolerance_region"]
                        ),
                        "leave_joint_tolerance_penalty": info[
                            "leave_joint_tolerance_penalty"
                        ],
                        "strict_pose_reached": int(info["strict_pose_reached"]),
                        "success_hold_count": info["success_hold_count"],
                        "success_hold_steps_required": info[
                            "success_hold_steps_required"
                        ],
                        "orientation_anchor": int(worker["orientation_anchor"]),
                        "position_error_m": info["next_rho_position"],
                        "orientation_error_rad": info["next_rho_orientation"],
                        "precision_action_scale": info["precision_action_scale"],
                        "velocity_magnitude": info["velocity_magnitude"],
                        "velocity_cost": info["velocity_cost"],
                        "smooth_velocity": info["smooth_velocity"],
                        "smooth_cost": info["smooth_cost"],
                        "curriculum_eligible": int(not worker["episode_used_warmup"]),
                        "position_reached": int(info["position_reached"]),
                        "task_reached": int(info["task_reached"]),
                        "obstacle_collision": int(info["obstacle_collision"]),
                        "self_collision": int(info["self_collision"]),
                        "environment_collision": int(info["environment_collision"]),
                        "joint_limit": int(info["joint_limit"]),
                        "sample_none": replay.last_sample_counts["none"],
                        "sample_static": replay.last_sample_counts["static"],
                        "sample_dynamic": replay.last_sample_counts["dynamic"],
                    })
                    if terminated or truncated:
                        update = curriculum.finish_episode(
                            worker["scene"], bool(info["position_reached"]),
                            bool(info["task_reached"]), worker["length"],
                            orientation_anchor=worker["orientation_anchor"],
                            curriculum_eligible=not worker["episode_used_warmup"],
                            pose_phase_complete_at_start=worker[
                                "pose_phase_complete_at_start"
                            ],
                            self_weight_full_at_start=worker[
                                "self_weight_full_at_start"
                            ],
                        )
                        episode_writer.writerow({
                            "episode": worker["episode_id"],
                            "global_step": counters["global_step"],
                            "stage_step": counters["stage_step"],
                            "stage_total_step": counters["stage_total_step"],
                            "scene": worker["scene"], "length": worker["length"],
                            "return": worker["return"],
                            "curriculum_eligible": int(not worker["episode_used_warmup"]),
                            "orientation_anchor": int(worker["orientation_anchor"]),
                            "position_reached": int(info["position_reached"]),
                            "strict_pose_reached": int(info["strict_pose_reached"]),
                            "success_hold_count": info["success_hold_count"],
                            "success_hold_steps_required": info[
                                "success_hold_steps_required"
                            ],
                            "task_reached": int(info["task_reached"]),
                            "collision_assisted_reach": int(info["collision_assisted_reach"]),
                            "safe_success": int(info["safe_success"]),
                            "obstacle_collision": int(info["obstacle_collision"]),
                            "self_collision": int(info["self_collision"]),
                            "environment_collision": int(info["environment_collision"]),
                            "joint_limit": int(info["joint_limit"]), "timeout": int(truncated),
                            "xi": worker["xi"], "lambda_self": worker["lambda_self"],
                            "strict": int(worker["strict"]), "rolling_task_reach_rate": "",
                            "eligible_steps": 0, "strict_steps": 0,
                            "goal_scale": worker["goal_scale"],
                            "rolling_goal_success_rate": update["rolling_goal_success_rate"],
                            "goal_eligible_steps": curriculum.goal.eligible_steps,
                            "goal_level_index": curriculum.goal.level_index,
                            "goal_level_steps": curriculum.goal.current_level_steps,
                            "goal_full_scale_steps": curriculum.goal.full_scale_steps,
                            "orientation_scale": worker["orientation_scale"],
                            "position_tolerance": worker["position_tolerance"],
                            "orientation_tolerance": worker["orientation_tolerance"],
                            "rolling_orientation_success_rate": update["rolling_orientation_success_rate"],
                            "orientation_eligible_steps": curriculum.orientation.eligible_steps,
                            "orientation_level_index": curriculum.orientation.level_index,
                            "orientation_level_steps": curriculum.orientation.current_level_steps,
                            "rolling_anchor_position_success_rate": (
                                "" if curriculum.orientation_anchor_success_rate is None
                                else curriculum.orientation_anchor_success_rate
                            ),
                            "rolling_previous_pose_success_rate": (
                                "" if curriculum.orientation_anchor_success_rate is None
                                else curriculum.orientation_anchor_success_rate
                            ),
                            "full_pose_steps": curriculum.orientation.full_scale_steps,
                            "position_phase_complete": int(curriculum.position_phase_complete),
                            "position_error_m": info["next_rho_position"],
                            "orientation_error_rad": info["next_rho_orientation"],
                            "orientation_retention_mode": curriculum.orientation_retention_mode,
                            "orientation_anchor_probability": curriculum.orientation_anchor_probability,
                            "deterministic_probe_passed": int(curriculum.orientation.deterministic_probe_passed),
                            "deterministic_probe_success_rate": (
                                "" if curriculum.orientation.deterministic_probe_success_rate is None
                                else curriculum.orientation.deterministic_probe_success_rate
                            ),
                            "s0_phase": curriculum.s0_phase,
                            "pose_phase_complete": int(curriculum.pose_phase_complete),
                            "self_safety_eligible_steps": curriculum.self_safety.eligible_steps,
                            "self_safety_full_weight_steps": curriculum.self_safety.full_weight_steps,
                        })
                        counters["episodes"] += 1
                        if targeted is not None:
                            key = "targeted_length_ema" if worker.get("bad_state_start", False) else "normal_length_ema"
                            length = worker["length"] - worker.get("start_step", 0)
                            targeted[key] = .95 * targeted[key] + .05 * length
                        workers[index] = None
                        if (
                            evaluate_during_training
                            and not manual_promotion
                            and curriculum.orientation_probe_due(
                                int(orientation_curriculum["deterministic_probe_interval_transitions"]),
                                pose_full_scale_min_transitions,
                            )
                        ):
                            pending_probe = True

                    if counters["stage_step"] % log_flush_interval == 0:
                        transition_handle.flush(); update_handle.flush(); episode_handle.flush()
                    if counters["stage_step"] % progress_interval == 0:
                        write_progress(progress_writer, started)
                        progress_handle.flush()

                if counters["stage_step"] % save_interval == 0:
                    perform_deferred_updates()
                    if (
                        evaluate_during_training
                        and manual_promotion
                        and counters["stage_step"] % manual_probe_interval == 0
                    ):
                        manual_result, manual_row = run_manual_pose_checkpoint_probe(
                            agent, config, curriculum, orientation_curriculum,
                            counters, pool=pool,
                        )
                        probe_writer.writerow(manual_row)
                        probe_handle.flush()
                        manual_metrics = manual_result["current"]
                        print(
                            "manual checkpoint probe "
                            f"step={counters['stage_step']} "
                            f"level={curriculum.orientation.level_index} "
                            f"success={manual_metrics['success_rate']:.3f} "
                            f"min_bin={manual_metrics.get('minimum_bin_success_rate', float('nan')):.3f} "
                            f"passed={int(bool(manual_result['passed']))}; "
                            "curriculum unchanged",
                            flush=True,
                        )
                    counters["checkpoint_index"] += 1
                    counters["replay_redistributions"] = replay.redistribution_count
                    checkpoint = checkpoints / f"step_{counters['stage_step']:07d}.pt"
                    save_parallel_checkpoint(
                        checkpoint, agent, replay, curriculum, counters, config, pool, workers,
                    )
                    agent.save_actor(checkpoints / f"actor_step_{counters['stage_step']:07d}.pt")
                    transition_handle.flush(); update_handle.flush(); episode_handle.flush()
                    print(f"saved {checkpoint} envs={num_envs} replay={len(replay)}", flush=True)
                    last_saved_step = counters["stage_step"]
            training_completed = True
        finally:
            try:
                if training_completed:
                    perform_deferred_updates()
                if last_saved_step != counters["stage_step"]:
                    counters["checkpoint_index"] += 1
                    counters["replay_redistributions"] = replay.redistribution_count
                    checkpoint = checkpoints / f"step_{counters['stage_step']:07d}.pt"
                    save_parallel_checkpoint(
                        checkpoint, agent, replay, curriculum, counters, config, pool, workers,
                    )
                    agent.save_actor(
                        checkpoints / f"actor_step_{counters['stage_step']:07d}.pt"
                    )
            finally:
                pool.close()

    summary = {
        "stage": "s0", **counters, "replay_size": len(replay),
        "xi": curriculum.xi_map(), "replay_orientation_storage": replay.orientation_storage_counts(),
        "max_stage_steps": max_stage_steps,
        "stage_budget_exhausted": counters["stage_total_step"] >= max_stage_steps,
        "num_envs": num_envs, "goal_scale": curriculum.goal_scale,
        "rolling_goal_success_rate": (
            float(np.mean(curriculum.goal.outcomes))
            if curriculum.goal.outcomes else None
        ),
        "goal_eligible_steps": curriculum.goal.eligible_steps,
        "goal_full_scale_steps": curriculum.goal.full_scale_steps,
        "goal_level_index": curriculum.goal.level_index,
        "goal_level_steps": curriculum.goal.current_level_steps,
        "orientation_scale": curriculum.orientation_scale,
        "position_tolerance": curriculum.position_tolerance,
        "orientation_tolerance": curriculum.orientation_tolerance,
        "orientation_level_index": curriculum.orientation.level_index,
        "orientation_level_steps": curriculum.orientation.current_level_steps,
        "orientation_eligible_steps": curriculum.orientation.eligible_steps,
        "rolling_orientation_success_rate": curriculum.orientation_success_rate,
        "rolling_anchor_position_success_rate": curriculum.orientation_anchor_success_rate,
        "rolling_previous_pose_success_rate": curriculum.orientation_anchor_success_rate,
        "orientation_retention_mode": curriculum.orientation_retention_mode,
        "orientation_anchor_probability": curriculum.orientation_anchor_probability,
        "orientation_replay_mix": curriculum.orientation_replay_mix,
        "deterministic_probe_passed": curriculum.orientation.deterministic_probe_passed,
        "deterministic_probe_success_rate": curriculum.orientation.deterministic_probe_success_rate,
        "deterministic_probe_collision_rate": curriculum.orientation.deterministic_probe_collision_rate,
        "deterministic_probe_joint_limit_rate": curriculum.orientation.deterministic_probe_joint_limit_rate,
        "deterministic_probe_previous_success_rate": (
            curriculum.orientation.deterministic_probe_previous_success_rate
        ),
        "deterministic_probe_confirmation_passed": (
            curriculum.orientation.deterministic_probe_confirmation_passed
        ),
        "full_pose_steps": curriculum.orientation.full_scale_steps,
        "lambda_self": curriculum.lambda_self, "s0_phase": curriculum.s0_phase,
        "pose_phase_complete": curriculum.pose_phase_complete,
        "self_safety_eligible_steps": curriculum.self_safety.eligible_steps,
        "self_safety_full_weight_steps": curriculum.self_safety.full_weight_steps,
        "self_safety_probe_collision_ceiling": curriculum.self_probe_collision_ceiling,
        "self_safety_probe_passed": curriculum.self_safety_probe_passed,
        "strict_steps": {key: state.strict_steps for key, state in curriculum.states.items()},
        "s0_goal_gate_eligible": curriculum.s0_goal_gate_eligible(
            full_scale_min_transitions, pose_full_scale_min_transitions,
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Hybrid Keypoint + Jacobian + Auto-PCR (S0 -> S1 -> S2)")
    parser.add_argument("--config", default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    parser.add_argument("--stage", choices=("s0", "s1", "s2"))
    parser.add_argument("--resume", help="complete checkpoint from the preceding block or stage")
    parser.add_argument("--bad-state-starts", type=Path, help="complete frozen-event start bank for targeted S0 continuation")
    parser.add_argument("--four-pool-seed", type=Path,
                        help="immutable frozen-1M anchor/frontier seed for a new four-pool branch")
    parser.add_argument("--bad-state-fraction", type=float, default=.25,
                        help="target rollout transition fraction (default: 0.25); start probability adapts to rollout length")
    parser.add_argument(
        "--approve-orientation-promotion", action="store_true",
        help=(
            "on an S0 resume, manually approve promotion using the latest "
            "25k checkpoint probe; the completed current level must still "
            "satisfy its minimum transition budget"
        ),
    )
    parser.add_argument(
        "--promote-to-level", type=int, metavar="LEVEL",
        help=(
            "on an S0 resume, explicitly skip to the specified harder pose "
            "level after the current level has completed its minimum budget"
        ),
    )
    parser.add_argument(
        "--promoted-current-fraction", type=float, metavar="FRACTION",
        help="current-level replay share after four-pool promotion (default: 0.5)",
    )
    parser.add_argument(
        "--allow-early-promotion", action="store_true",
        help=(
            "allow --promote-to-level before the current level reaches its "
            "minimum transition budget; the override is recorded in the checkpoint"
        ),
    )
    parser.add_argument(
        "--initialize-actor-from",
        help=(
            "actor or complete checkpoint used only to initialize actor weights; "
            "critics, replay, counters, and success labels start fresh"
        ),
    )
    parser.add_argument(
        "--start-level", type=int,
        help="initial S0 pose level for --initialize-actor-from",
    )
    parser.add_argument("--steps", type=int, help="override this block's environment steps")
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int, help="override training seed")
    parser.add_argument(
        "--num-envs", type=int,
        help="number of synchronous PyBullet worker processes (formal default from config)",
    )
    parser.add_argument("--validation-mode", action="store_true", help="small replay/batch for implementation checks only")
    args = parser.parse_args()
    if args.approve_orientation_promotion and not args.resume:
        parser.error("--approve-orientation-promotion requires --resume")
    if args.promote_to_level is not None and not args.resume:
        parser.error("--promote-to-level requires --resume")
    if args.promoted_current_fraction is not None:
        if args.promote_to_level != 1:
            parser.error("--promoted-current-fraction requires --promote-to-level 1")
        if not 0. <= args.promoted_current_fraction <= 1.:
            parser.error("--promoted-current-fraction must be in [0, 1]")
    if args.allow_early_promotion and args.promote_to_level is None:
        parser.error("--allow-early-promotion requires --promote-to-level")
    if args.approve_orientation_promotion and args.promote_to_level is not None:
        parser.error(
            "--approve-orientation-promotion and --promote-to-level are mutually exclusive"
        )
    restart_flags = [args.resume, args.initialize_actor_from]
    if sum(value is not None for value in restart_flags) > 1:
        parser.error("--resume and --initialize-actor-from are mutually exclusive")
    if (args.initialize_actor_from is None) != (args.start_level is None):
        parser.error("--initialize-actor-from and --start-level must be used together")
    config = load_config(ROOT / args.config)
    thesis = config["thesis"]
    stage = args.stage or str(thesis["stage"])
    try:
        num_envs = resolve_num_envs(
            stage, args.num_envs, int(config["train"].get("num_envs", 1)),
            args.validation_mode,
        )
    except ValueError as error:
        parser.error(str(error))
    if num_envs > 1 and stage != "s0":
        parser.error("parallel environment collection currently supports S0 only")
    if args.initialize_actor_from and stage != "s0":
        parser.error("--initialize-actor-from currently supports S0 only")
    if args.bad_state_starts and (stage != "s0" or not args.resume or num_envs < 2):
        parser.error("--bad-state-starts requires parallel S0 --resume")
    if args.bad_state_starts and (args.promote_to_level is not None or args.approve_orientation_promotion):
        parser.error("targeted continuation cannot be combined with level promotion")
    if args.four_pool_seed and (not args.bad_state_starts or not args.resume or stage != "s0"):
        parser.error("--four-pool-seed requires targeted S0 continuation from the complete 1M checkpoint")
    if not 0. < args.bad_state_fraction < 1.:
        parser.error("--bad-state-fraction must be in (0, 1)")
    if thesis.get("protocol") not in SUPPORTED_PROTOCOLS:
        parser.error(
            "the serial trainer only supports protocols "
            f"{sorted(SUPPORTED_PROTOCOLS)!r}"
        )
    try:
        validate_training_architecture(config)
    except ValueError as error:
        parser.error(str(error))
    seed = int(args.seed if args.seed is not None else config["seed"]); config["seed"] = seed
    stream_names = ("python", "numpy", "torch", "environment", "curriculum", "replay", "warmup_action")
    children = np.random.SeedSequence(seed).spawn(len(stream_names))
    stream_seeds = {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(stream_names, children)
    }
    config["_rng_stream_seeds"] = stream_seeds
    random.seed(stream_seeds["python"]); np.random.seed(stream_seeds["numpy"])
    torch.manual_seed(stream_seeds["torch"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(stream_seeds["torch"])
    warmup_action_rng = np.random.default_rng(stream_seeds["warmup_action"])
    config["_warmup_action_rng"] = warmup_action_rng
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{stamp}_{stage}_seed{seed}"
    output = ROOT / str(config["train"]["output_dir"]) / run_name
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"; checkpoints.mkdir()
    env = ThesisHomotopyEnv(config)
    env.rng = np.random.default_rng(stream_seeds["environment"])
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    agent = ThesisSACAgent(obs_dim, action_dim, config)
    orientation_curriculum = thesis["orientation_curriculum"]
    replay_storage = orientation_curriculum["replay_storage"]
    self_curriculum = thesis["self_collision"]["curriculum"]
    capacities = ({"none": 512, "static": 512, "dynamic": 512} if args.validation_mode else None)
    storage_scale = 512 if args.validation_mode else None
    replay = HomotopyReplayBuffer(
        obs_dim, action_dim, str(agent.device), capacities=capacities, seed=seed,
        reward_gamma=float(config["sac"]["gamma"]), reward_horizon=int(thesis["horizon"]),
        s0_anchor_fraction=float(orientation_curriculum["replay_anchor_fraction"]),
        s0_current_fraction=float(orientation_curriculum["replay_current_fraction"]),
        s0_anchor_capacity=(
            storage_scale or int(replay_storage["anchor_capacity"])
        ),
        s0_current_capacity=(
            storage_scale or int(replay_storage["current_capacity"])
        ),
        s0_history_capacity_per_level=(
            storage_scale or int(replay_storage["history_capacity_per_level"])
        ),
        s0_joint_pose=bool(thesis.get("joint_pose_curriculum")),
        s0_success_fraction=float(orientation_curriculum["replay_success_fraction"]),
        s0_success_schedule=orientation_curriculum.get("replay_success_schedule"),
        s0_success_capacity=(
            storage_scale or int(replay_storage["current_success_capacity"])
        ),
        reward_parameters=env.reward_parameters,
    )
    replay.rng = np.random.default_rng(stream_seeds["replay"])
    goal_curriculum = thesis["goal_curriculum"]
    full_scale_min_transitions = int(goal_curriculum["full_scale_min_transitions"])
    if full_scale_min_transitions < 1:
        raise ValueError("goal_curriculum.full_scale_min_transitions must be positive")
    pose_full_scale_min_transitions = int(orientation_curriculum["full_scale_min_transitions"])
    if pose_full_scale_min_transitions < 1:
        raise ValueError("orientation_curriculum.full_scale_min_transitions must be positive")
    curriculum = HomotopyCurriculum(
        stage,
        seed=stream_seeds["curriculum"],
        ramp_steps=int(thesis["ramp_steps"]),
        goal_start_scale=float(goal_curriculum["start_scale"]),
        goal_end_scale=float(goal_curriculum["end_scale"]),
        goal_success_window=int(goal_curriculum["success_window"]),
        goal_success_floor=float(goal_curriculum["success_floor"]),
        goal_full_scale_min_steps=full_scale_min_transitions,
        goal_levels=goal_curriculum["levels"],
        goal_min_transitions_per_level=int(goal_curriculum["min_transitions_per_level"]),
        orientation_start_scale=float(orientation_curriculum["start_scale"]),
        orientation_end_scale=float(orientation_curriculum["end_scale"]),
        orientation_tolerance_start=float(orientation_curriculum["tolerance_start"]),
        orientation_tolerance_end=float(orientation_curriculum["tolerance_end"]),
        orientation_success_window=int(orientation_curriculum["success_window"]),
        orientation_success_floor=float(orientation_curriculum["success_floor"]),
        orientation_levels=orientation_curriculum["levels"],
        orientation_min_transitions_per_level=int(
            orientation_curriculum["min_transitions_per_level"]
        ),
        orientation_anchor_probability=float(orientation_curriculum["anchor_probability"]),
        orientation_anchor_floor=float(orientation_curriculum["anchor_success_floor"]),
        orientation_anchor_window=int(orientation_curriculum["anchor_success_window"]),
        orientation_anchor_probability_intermediate=float(
            orientation_curriculum["anchor_probability_intermediate"]
        ),
        orientation_anchor_probability_recovery=float(
            orientation_curriculum["anchor_probability_recovery"]
        ),
        orientation_retention_target=float(orientation_curriculum["retention_target"]),
        orientation_retention_min_observations=int(
            orientation_curriculum["retention_min_observations"]
        ),
        orientation_replay_anchor_normal=float(
            orientation_curriculum["replay_anchor_fraction"]
        ),
        orientation_replay_current_normal=float(
            orientation_curriculum["replay_current_fraction"]
        ),
        orientation_replay_anchor_intermediate=float(
            orientation_curriculum["replay_anchor_intermediate"]
        ),
        orientation_replay_current_intermediate=float(
            orientation_curriculum["replay_current_intermediate"]
        ),
        orientation_replay_anchor_recovery=float(
            orientation_curriculum["replay_anchor_recovery"]
        ),
        orientation_replay_current_recovery=float(
            orientation_curriculum["replay_current_recovery"]
        ),
        orientation_deterministic_probe_required=bool(
            orientation_curriculum["deterministic_probe_required"]
        ),
        orientation_deterministic_previous_success_floor=float(
            orientation_curriculum["deterministic_previous_success_floor"]
        ),
        orientation_deterministic_previous_collision_ceiling=float(
            orientation_curriculum["deterministic_previous_collision_ceiling"]
        ),
        orientation_full_scale_min_steps=pose_full_scale_min_transitions,
        joint_pose_levels=thesis.get("joint_pose_curriculum", {}).get("levels"),
        self_start_weight=float(self_curriculum["start_weight"]),
        self_end_weight=float(self_curriculum["end_weight"]),
        self_ramp_steps=int(self_curriculum["ramp_steps"]),
        self_full_weight_min_steps=int(
            self_curriculum["full_weight_min_transitions"]
        ),
        self_probe_collision_ceiling=float(
            self_curriculum["probe_collision_ceiling"]
        ),
    )
    counters = {"global_step": 0, "stage_step": 0, "stage_total_step": 0,
                "updates": 0, "episodes": 0, "block": 1,
                "checkpoint_index": 0, "random_steps_used": 0, "replay_redistributions": 0}
    resume_mode = None
    active_episode = None
    loaded_checkpoint_state = None
    checkpoint_source = args.resume
    if checkpoint_source:
        state = torch.load(checkpoint_source, map_location=agent.device, weights_only=False)
        loaded_checkpoint_state = state
        previous_stage = state["curriculum"]["stage"]
        if state.get("protocol") != thesis["protocol"]:
            raise ValueError(
                f"checkpoint protocol {state.get('protocol')!r} does not match "
                f"expected protocol {thesis['protocol']!r}"
            )
        if int(state["config"]["seed"]) != seed:
            raise ValueError(
                f"checkpoint root seed {state['config']['seed']} does not match requested seed {seed}"
            )
        if "warmup_action_rng" not in state or "rng_stream_seeds" not in state:
            raise ValueError("checkpoint predates deterministic RNG stream tracking and cannot be resumed")
        if state["rng_stream_seeds"] != stream_seeds:
            raise ValueError("checkpoint RNG stream derivation does not match the current trainer")
        if {("s0", "s1"), ("s1", "s2")}.intersection({(previous_stage, stage)}):
            validate_serial_stage_transition(config, state, stage)
            agent.load_state_dict(state["agent"]); replay.load_state_dict(state["replay"])
            curriculum.inherit_task_state(state["curriculum"])
            if previous_stage == "s1":
                curriculum.inherit_scene_state(state["curriculum"], "static")
            inherited = state["counters"]; counters["global_step"] = inherited["global_step"]
            counters["updates"] = inherited["updates"]; counters["episodes"] = inherited["episodes"]
            counters["random_steps_used"] = inherited["random_steps_used"]
            counters["stage_total_step"] = 0
            random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
            restore_torch_rng(state)
            curriculum.rng.bit_generator.state = state["curriculum"]["rng_state"]
            resume_mode = f"stage_transition_{previous_stage}_to_{stage}"
        elif previous_stage == stage:
            agent.load_state_dict(state["agent"]); replay.load_state_dict(state["replay"])
            curriculum.load_state_dict(state["curriculum"]); counters.update(state["counters"])
            if "stage_total_step" not in state["counters"]:
                if stage != "s0":
                    raise ValueError(
                        "same-stage checkpoint lacks stage_total_step; cannot audit its stage budget"
                    )
                counters["stage_total_step"] = int(state["counters"]["global_step"])
            random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
            restore_torch_rng(state)
            counters["block"] += 1; counters["stage_step"] = 0
            active_episode = state.get("active_episode")
            resume_mode = "same_stage_continuation"
        else:
            raise ValueError(f"invalid stage transition {previous_stage}->{stage}")
        env.rng.bit_generator.state = state["environment_rng"]
        warmup_action_rng.bit_generator.state = state["warmup_action_rng"]
    elif args.initialize_actor_from:
        start_level = int(args.start_level)
        maximum_level = len(thesis["joint_pose_curriculum"]["levels"]) - 1
        if not 0 <= start_level <= maximum_level:
            raise ValueError(
                f"--start-level must be in [0, {maximum_level}], got {start_level}"
            )
        initialized_actor = load_reference_actor(
            agent, ROOT / args.initialize_actor_from
        )
        agent.actor.load_state_dict(initialized_actor.state_dict())
        # PCR must begin from the imported policy, not the random actor that
        # existed when the new agent was constructed.
        agent.reset_chain_reference_actor()
        if start_level > curriculum.orientation.level_index:
            curriculum.manually_advance_orientation_to(start_level, allow_early=True)
        replay.begin_orientation_level(curriculum.orientation.scale)
        # An imported actor should collect the new reward contract immediately;
        # only critic updates retain the normal update_after replay warm-up.
        counters["random_steps_used"] = int(config["sac"]["warmup_steps"])
        counters["actor_initialization"] = {
            "checkpoint": args.initialize_actor_from,
            "start_level": start_level,
            "replay_reused": False,
            "critic_reused": False,
            "critic_warmup_transitions": int(
                config["train"].get(
                    "actor_initialization_critic_warmup_transitions", 5000
                )
            ),
            "warmup_start_stage_total_step": int(counters["stage_total_step"]),
        }
        resume_mode = "actor_only_fresh_replay_and_critics"
    elif stage != "s0":
        raise ValueError("S1/S2 must use --resume with the preceding stage's complete checkpoint")
    curriculum.configure_orientation_retention(orientation_curriculum)
    bad_state_starts = None
    previous_sampling = counters.get("bad_state_sampling")
    four_pool_promotion = (
        replay.four_pool_enabled and args.promote_to_level == 1
        and curriculum.orientation.level_index == 0
    )
    if args.promoted_current_fraction is not None and not four_pool_promotion:
        raise ValueError("--promoted-current-fraction requires four-pool L0 to L1 promotion")
    if previous_sampling and not args.bad_state_starts and not four_pool_promotion:
        raise ValueError("resuming targeted training requires --bad-state-starts")
    if args.bad_state_starts:
        bad_state_starts = torch.load(args.bad_state_starts, map_location="cpu", weights_only=False)
        expected_source = previous_sampling["source_checkpoint"] if previous_sampling else args.resume
        if Path(bad_state_starts["source_checkpoint"]).resolve() != Path(expected_source).resolve():
            raise ValueError("bad-state bank was collected from a different source checkpoint")
        if previous_sampling and previous_sampling["fraction"] != args.bad_state_fraction:
            raise ValueError("cannot silently change the targeted sampling fraction on resume")
        if curriculum.orientation.level_index != 0 or resume_mode != "same_stage_continuation":
            raise ValueError("bad-state bank requires a same-stage S0 level-0 continuation")
        if state["config"]["thesis"] != thesis or state["config"]["sac"] != config["sac"]:
            raise ValueError("targeted continuation must retain checkpoint environment and SAC settings")
        if not bad_state_starts.get("centers") or not env.hybrid_explicit_pose_error:
            raise ValueError("expected a nonempty hybrid/keypoint bad-state bank")
        for center in bad_state_starts["centers"]:
            snapshot = center["state"]
            if (np.asarray(center["observation"]).shape != (obs_dim,) or
                    not .6 <= center["rho_R"] <= 1. or snapshot["contract"].scene != "none" or
                    int(center["step"]) != int(snapshot["step_count"]) or
                    not 0 <= int(center["step"]) < env.horizon):
                raise ValueError("invalid bad-state snapshot/band/episode clock")
            for key, shape in (("q", (6,)), ("qdot", (6,)), ("goal_position", (3,)),
                               ("goal_quaternion", (4,)), ("goal_joint_positions", (6,))):
                value = np.asarray(snapshot[key])
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid bad-state {key}")
    if args.four_pool_seed:
        if replay.four_pool_enabled:
            raise ValueError("four-pool replay is already loaded from the checkpoint")
        seed_bank = torch.load(args.four_pool_seed, map_location="cpu", weights_only=False)
        if Path(seed_bank["source_checkpoint"]).resolve() != Path(args.resume).resolve():
            raise ValueError("four-pool seed must originate from this complete 1M checkpoint")
        if seed_bank["source_sha256"] != file_sha256(Path(args.resume)):
            raise ValueError("four-pool seed frozen actor hash does not match the resumed checkpoint")
        if Path(seed_bank["bad_state_starts"]).resolve() != args.bad_state_starts.resolve():
            raise ValueError("four-pool seed must use the selected 85-state bank")
        replay.configure_four_pool(seed_bank, source=str(args.four_pool_seed.resolve()))
        counters["four_pool_seed"] = str(args.four_pool_seed.resolve())
    elif replay.four_pool_enabled:
        if not four_pool_promotion and (
            not args.bad_state_starts or counters.get("four_pool_seed") != replay.four_pool_source
        ):
            raise ValueError("resuming a four-pool branch requires its original bad-state bank")
    if args.approve_orientation_promotion:
        if stage != "s0" or not bool(
            orientation_curriculum.get("manual_promotion", False)
        ):
            raise ValueError(
                "--approve-orientation-promotion requires S0 manual-promotion mode"
            )
        latest = counters.get("last_manual_pose_probe")
        if not isinstance(latest, dict):
            raise ValueError(
                "checkpoint has no manual pose probe; finish a 25k boundary first"
            )
        if int(latest["orientation_level_index"]) != curriculum.orientation.level_index:
            raise ValueError("latest manual pose probe belongs to a different level")
        if (
            curriculum.orientation.current_level_steps
            < curriculum.orientation.min_transitions_per_level
        ):
            raise ValueError(
                "manual promotion requires the full per-level transition budget: "
                f"{curriculum.orientation.current_level_steps}/"
                f"{curriculum.orientation.min_transitions_per_level}"
            )
        metrics = latest["metrics"]
        if not bool(latest.get("passed", False)):
            raise ValueError(
                "manual promotion requires the latest frozen probe to pass "
                "both the current-level and previous-level retention gates"
            )
        if curriculum.orientation.level_index > 0 and (
            "retention_passed" not in latest
            or "previous_metrics" not in latest
        ):
            raise ValueError(
                "manual promotion requires a fresh cross-level frozen probe; "
                "the checkpoint predates retention-gate reporting"
            )
        if not bool(latest.get(
            "retention_passed", curriculum.orientation.level_index == 0,
        )):
            raise ValueError(
                "manual promotion blocked by the previous-level retention gate"
            )
        previous_metrics = latest.get("previous_metrics")
        curriculum.record_orientation_probe(
            success_rate=float(metrics["success_rate"]),
            collision_rate=float(metrics["collision_rate"]),
            joint_limit_rate=float(metrics["joint_limit_rate"]),
            previous_success_rate=(
                None if previous_metrics is None
                else float(previous_metrics["success_rate"])
            ),
            previous_collision_rate=(
                None if previous_metrics is None
                else float(previous_metrics["collision_rate"])
            ),
            previous_joint_limit_rate=(
                None if previous_metrics is None
                else float(previous_metrics["joint_limit_rate"])
            ),
            confirmation_passed=True,
            retention_passed=bool(latest.get("retention_passed", True)),
            recovery_required=False,
            passed=True,
        )
        previous_level = curriculum.orientation.level_index
        if not curriculum.advance_orientation_if_ready():
            raise ValueError(
                "manual promotion could not advance (already final level or "
                "online prerequisites are incomplete)"
            )
        replay.begin_orientation_level(curriculum.orientation.scale)
        counters.setdefault("manual_promotions", []).append({
            "from_level": int(previous_level),
            "to_level": int(curriculum.orientation.level_index),
            "approved_from_global_step": int(latest["global_step"]),
            "measured_passed": bool(latest["passed"]),
            "measured_metrics": dict(metrics),
        })
        # A boundary checkpoint may contain partial old-level episodes. They
        # cannot continue after promotion because their task contract differs.
        active_episode = None
        if loaded_checkpoint_state is not None:
            for saved in loaded_checkpoint_state.get("parallel_workers", []):
                saved["active"] = False
                saved["episode"] = None
                saved["observation"] = None
    elif args.promote_to_level is not None:
        if stage != "s0" or not bool(
            orientation_curriculum.get("manual_promotion", False)
        ):
            raise ValueError("--promote-to-level requires S0 manual-promotion mode")
        previous_level = int(curriculum.orientation.level_index)
        target_level = int(args.promote_to_level)
        previous_steps = int(curriculum.orientation.current_level_steps)
        early_promotion = bool(
            previous_steps < curriculum.orientation.min_transitions_per_level
        )
        curriculum.manually_advance_orientation_to(
            target_level, allow_early=bool(args.allow_early_promotion),
        )
        if four_pool_promotion:
            replay.promote_four_pool_to_history(
                curriculum.orientation.scale,
                current_fraction=(.5 if args.promoted_current_fraction is None
                                  else args.promoted_current_fraction),
            )
            counters["bad_state_sampling_previous_level"] = counters.pop("bad_state_sampling")
        else:
            replay.begin_orientation_level(curriculum.orientation.scale)
        counters.setdefault("manual_promotions", []).append({
            "from_level": previous_level,
            "to_level": target_level,
            "skipped_levels": list(range(previous_level + 1, target_level)),
            "source_level_steps": previous_steps,
            "source_minimum_level_steps": int(
                curriculum.orientation.min_transitions_per_level
            ),
            "early_promotion": early_promotion,
            "decision": "explicit_target_level",
            "measured_passed": None,
            "measured_metrics": None,
        })
        active_episode = None
        if loaded_checkpoint_state is not None:
            for saved in loaded_checkpoint_state.get("parallel_workers", []):
                saved["active"] = False
                saved["episode"] = None
                saved["observation"] = None
    semantic_config = thesis["semantic_long_term_replay"]
    semantic_enabled = bool(semantic_config["enabled"]) and stage == "s0"
    pose_contract = curriculum.pose_contract()
    semantic_contract_keys = (
        "target_distance_min_m", "target_distance_max_m",
        "target_orientation_min_rad", "target_orientation_max_rad",
    )
    if semantic_enabled and not all(key in pose_contract for key in semantic_contract_keys):
        raise ValueError(
            "semantic long-term replay requires task-space distance/orientation bounds"
        )
    replay.configure_semantic_long_term(
        enabled=semantic_enabled,
        fraction=(float(semantic_config["fraction"]) if semantic_enabled else 0.0),
        position_bins=int(semantic_config["position_bins"]),
        orientation_bins=int(semantic_config["orientation_bins"]),
        episodes_per_bin=int(semantic_config["episodes_per_bin"]),
        transitions_per_episode=int(semantic_config["transitions_per_episode"]),
        distance_min=float(pose_contract.get("target_distance_min_m", 0.0)),
        distance_max=float(pose_contract.get("target_distance_max_m", 1.0)),
        orientation_min=float(pose_contract.get("target_orientation_min_rad", 0.0)),
        orientation_max=float(pose_contract.get("target_orientation_max_rad", 1.0)),
        seed=stream_seeds["replay"] ^ 0x5EED5EED,
        reset_on_contract_change=args.promote_to_level is not None,
    )
    if num_envs > 1:
        if stage != "s0":
            raise ValueError("parallel environment collection currently supports S0 only")
        if loaded_checkpoint_state is not None:
            checkpoint_num_envs = int(loaded_checkpoint_state.get("parallel_num_envs", 1))
            if checkpoint_num_envs != num_envs:
                raise ValueError(
                    f"parallel checkpoint uses {checkpoint_num_envs} environments, "
                    f"requested {num_envs}"
                )
            if "parallel_workers" not in loaded_checkpoint_state:
                raise ValueError(
                    "a single-environment checkpoint cannot resume a parallel run"
                )

    manifest = {
        "protocol": config["thesis"]["protocol"], "stage": stage, "seed": seed,
        "resume": checkpoint_source, "resume_mode": resume_mode, "config": str(args.config),
        "initialize_actor_from": args.initialize_actor_from,
        "start_level": args.start_level,
        "bad_state_starts": None if args.bad_state_starts is None else str(args.bad_state_starts.resolve()),
        "bad_state_target_transition_fraction": None if bad_state_starts is None else args.bad_state_fraction,
        "four_pool_seed": counters.get("four_pool_seed"),
        "semantic_long_term_replay": {
            "enabled": semantic_enabled,
            "fraction": float(semantic_config["fraction"]),
            "position_bins": int(semantic_config["position_bins"]),
            "orientation_bins": int(semantic_config["orientation_bins"]),
            "episodes_per_bin": int(semantic_config["episodes_per_bin"]),
            "transitions_per_episode": int(semantic_config["transitions_per_episode"]),
            "bounds": (
                None if not semantic_enabled else {
                    key: float(pose_contract[key]) for key in semantic_contract_keys
                }
            ),
        },
        "approve_orientation_promotion": bool(args.approve_orientation_promotion),
        "promote_to_level": args.promote_to_level,
        "allow_early_promotion": bool(args.allow_early_promotion),
        "urdf_sha256": file_sha256(ROOT / config["robot"]["urdf"]),
        "joint_names": config["robot"]["joint_names"], "tool_link_name": config["robot"]["tool_link_name"],
        "observation_dim": obs_dim, "action_dim": action_dim, "external_collision_bodies": [],
        "validation_mode": args.validation_mode,
        "evaluate_during_training": bool(
            config["train"].get("evaluate_during_training", False)
        ),
        "num_envs": num_envs,
        "parallel_execution": {
            "mode": (
                "pipelined_environment_step_and_learner_update"
                if num_envs > 1 else "single_process"
            ),
            "max_policy_lag_transitions": num_envs if num_envs > 1 else 0,
            "checkpoint_flushes_deferred_updates": True,
        },
        "batch_size": (
            8 if args.validation_mode else int(config["sac"]["batch_size"])
        ),
        "updates_per_transition": float(
            config["sac"].get("updates_per_transition", 1.0)
        ),
        "rng_stream_seeds": stream_seeds,
        "replay_storage": {
            **replay_storage,
            "storage_version": replay.STORAGE_VERSION,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = output / "episodes.csv"
    fields = ["episode", "global_step", "stage_step", "stage_total_step", "scene", "length", "return", "curriculum_eligible",
              "orientation_anchor",
              "position_reached", "strict_pose_reached", "success_hold_count",
              "success_hold_steps_required", "task_reached",
              "collision_assisted_reach", "safe_success", "obstacle_collision", "self_collision",
              "environment_collision", "joint_limit", "timeout", "xi", "lambda_self", "strict", "rolling_task_reach_rate",
              "eligible_steps", "strict_steps", "goal_scale", "rolling_goal_success_rate",
              "goal_eligible_steps", "goal_level_index", "goal_level_steps",
              "goal_full_scale_steps"]
    fields.extend(["orientation_scale", "position_tolerance", "orientation_tolerance",
                   "rolling_orientation_success_rate", "orientation_eligible_steps",
                   "orientation_level_index", "orientation_level_steps",
                   "rolling_anchor_position_success_rate",
                   "rolling_previous_pose_success_rate", "full_pose_steps",
                   "position_phase_complete", "position_error_m", "orientation_error_rad",
                   "orientation_retention_mode", "orientation_anchor_probability",
                   "deterministic_probe_passed", "deterministic_probe_success_rate",
                   "s0_phase", "pose_phase_complete",
                   "self_safety_eligible_steps", "self_safety_full_weight_steps"])
    transition_fields = ["global_step", "stage_step", "stage_total_step", "episode", "episode_step", "scene", "xi", "lambda_self", "strict",
                         "bad_state_start",
                         "reward", "r_goal", "c_proximity", "hard_penalty", "timeout_penalty", "safety_penalty",
                         "external_safety_penalty", "self_safety_penalty",
                         "terminal_guard_penalty", "clearance_violation", "risk_max", "d_min", "goal_scale",
                         "self_clearance_violation", "self_risk_max", "self_d_min",
                         "self_ttc_min", "self_approach_max",
                         "self_projection_intervened", "self_projection_infeasible",
                         "self_projection_correction_norm", "self_projection_max_slack",
                         "goal_full_scale_steps", "orientation_scale", "position_tolerance", "orientation_tolerance",
                         "orientation_reward_scale", "orientation_reward_gate",
                         "rho_position", "next_rho_position", "rho_orientation",
                         "next_rho_orientation", "position_quality", "orientation_quality",
                         "keypoint_tracking_quality", "keypoint_distance",
                         "jacobian_clip_ratio",
                         "next_keypoint_distance", "keypoint_progress",
                         "keypoint_tracking_reward", "keypoint_progress_reward",
                         "keypoint_precision_quality", "keypoint_precision_reward",
                         "pose_potential", "next_pose_potential", "pose_potential_progress",
                         "position_progress", "orientation_progress",
                         "orientation_error_progress", "orientation_absolute_penalty",
                         "orientation_absolute_position_gate",
                         "orientation_error_progress_reward",
                         "orientation_completion_reward",
                         "orientation_completion_position_gate",
                         "orientation_shaping_reward",
                         "fine_position_quality", "fine_orientation_quality",
                         "fine_position_progress", "fine_orientation_progress",
                         "precision_proximity", "precision_stop_cost",
                         "joint_tolerance_ratio", "next_joint_tolerance_ratio",
                         "joint_precision_quality", "next_joint_precision_quality",
                         "joint_precision_progress", "joint_precision_reward",
                         "position_completion", "orientation_completion",
                         "partial_precision_reward",
                         "left_joint_tolerance_region", "leave_joint_tolerance_penalty",
                         "strict_pose_reached", "success_hold_count",
                         "success_hold_steps_required",
                         "orientation_anchor", "position_error_m", "orientation_error_rad",
                         "precision_action_scale", "velocity_magnitude", "velocity_cost",
                         "smooth_velocity", "smooth_cost",
                         "curriculum_eligible", "position_reached",
                         "task_reached", "obstacle_collision", "self_collision", "environment_collision", "joint_limit",
                         "sample_none", "sample_static", "sample_dynamic"]
    update_fields = [
        "global_step", "stage_step", "stage_total_step", "update", "critic_loss",
        "actor_loss", "actor_sac_loss", "policy_churn_kl",
        "policy_churn_penalty", "policy_churn_to_sac_ratio",
        "chain_pcr_effective_coefficient",
        "chain_pcr_sac_loss_ema", "chain_pcr_loss_ema",
        "alpha_loss", "alpha",
        "q1_mean", "q2_mean", "target_mean", "sample_none", "sample_static", "sample_dynamic",
        "sample_orientation_anchor", "sample_orientation_current", "sample_orientation_historical",
        "sample_current_success", "sample_current_recent", "sample_previous_level",
        "sample_semantic_long_term",
        "sample_stability_anchor", "sample_frontier",
        "critic_gradient_norm", "actor_gradient_norm", "actor_updated",
    ]
    progress_fields = [
        "global_step", "stage_step", "stage_total_step", "episodes", "replay_size", "updates", "alpha",
        "bad_state_starts", "bad_state_transitions", "bad_state_transition_fraction",
        "critic_loss", "actor_loss", "actor_sac_loss", "policy_churn_kl",
        "policy_churn_penalty", "policy_churn_to_sac_ratio",
        "chain_pcr_effective_coefficient",
        "chain_pcr_sac_loss_ema", "chain_pcr_loss_ema",
        "q1_mean", "q2_mean", "target_mean",
        "keypoint_tracking_quality_mean", "keypoint_precision_quality_mean",
        "keypoint_precision_reward_mean", "keypoint_progress_mean",
        "keypoint_progress_positive_ratio",
        "keypoint_progress_mean_when_positive",
        "jacobian_clip_ratio_mean",
        "steps_per_second",
        "goal_scale", "goal_level_index", "goal_level_steps",
        "rolling_goal_success_rate", "goal_eligible_steps",
        "goal_full_scale_steps", "s0_goal_gate_eligible",
        "orientation_scale", "position_tolerance", "orientation_tolerance", "rolling_orientation_success_rate",
        "orientation_eligible_steps", "orientation_level_index", "orientation_level_steps",
        "rolling_anchor_position_success_rate", "rolling_previous_pose_success_rate",
        "sample_orientation_anchor",
        "sample_orientation_current", "sample_orientation_historical",
        "sample_current_success", "sample_current_recent", "sample_previous_level",
        "sample_semantic_long_term",
        "sample_stability_anchor", "sample_frontier", "stored_stability_anchor", "stored_frontier",
        "full_pose_steps", "position_phase_complete", "orientation_retention_mode",
        "orientation_anchor_probability", "replay_orientation_anchor_fraction",
        "replay_orientation_current_fraction", "replay_orientation_historical_fraction",
        "replay_current_success_fraction", "replay_current_recent_fraction",
        "replay_previous_level_fraction", "replay_semantic_long_term_fraction",
        "stored_orientation_anchor", "stored_orientation_current",
        "stored_orientation_historical", "stored_orientation_historical_levels",
        "stored_current_success_episodes",
        "stored_semantic_long_term_transitions",
        "stored_semantic_long_term_episodes",
        "stored_semantic_long_term_nonempty_bins",
        "deterministic_probe_passed", "deterministic_probe_success_rate",
        "deterministic_probe_collision_rate", "deterministic_probe_joint_limit_rate",
        "deterministic_probe_previous_success_rate",
        "deterministic_probe_confirmation_passed",
        "lambda_self", "s0_phase", "pose_phase_complete",
        "self_safety_eligible_steps", "self_safety_full_weight_steps",
    ]
    requested_steps = int(args.steps or config["train"]["total_steps"])
    max_stage_steps = int(config["train"]["max_stage_steps"])
    total_steps = resolve_stage_block_steps(
        counters["stage_total_step"], requested_steps, max_stage_steps
    )
    if total_steps < requested_steps:
        print(
            f"stage budget leaves {total_steps} transitions; "
            f"shortening requested block from {requested_steps}",
            flush=True,
        )
    save_interval = int(config["train"]["save_interval"]); last_saved_step = -1
    progress_interval = int(config["train"].get("progress_interval", 1000))
    if progress_interval < 1:
        raise ValueError("train.progress_interval must be positive")
    log_flush_interval = int(config["train"].get("log_flush_interval", 100))
    if log_flush_interval < 1:
        raise ValueError("train.log_flush_interval must be positive")
    batch_size = 8 if args.validation_mode else int(config["sac"]["batch_size"])
    update_after = 8 if args.validation_mode else int(config["sac"]["update_after"])
    updates_per_transition = float(config["sac"].get("updates_per_transition", 1.0))
    if updates_per_transition <= 0.0:
        raise ValueError("sac.updates_per_transition must be positive")
    manual_promotion = bool(orientation_curriculum.get("manual_promotion", False))
    evaluate_during_training = bool(
        config["train"].get("evaluate_during_training", False)
    )
    manual_probe_interval = int(
        orientation_curriculum.get(
            "manual_probe_interval_transitions", save_interval,
        )
    )
    if manual_probe_interval < 1 or manual_probe_interval % save_interval != 0:
        raise ValueError(
            "manual_probe_interval_transitions must be a positive multiple of "
            "train.save_interval"
        )
    if num_envs > 1:
        run_parallel_s0(
            config=config, env_template=env, agent=agent, replay=replay,
            curriculum=curriculum, counters=counters, output=output,
            checkpoints=checkpoints, fields=fields,
            transition_fields=transition_fields, update_fields=update_fields,
            progress_fields=progress_fields, total_steps=total_steps,
            max_stage_steps=max_stage_steps, save_interval=save_interval,
            progress_interval=progress_interval,
            log_flush_interval=log_flush_interval, batch_size=batch_size,
            update_after=update_after, num_envs=num_envs,
            updates_per_transition=updates_per_transition,
            orientation_curriculum=orientation_curriculum,
            full_scale_min_transitions=full_scale_min_transitions,
            pose_full_scale_min_transitions=pose_full_scale_min_transitions,
            loaded_checkpoint_state=loaded_checkpoint_state,
            bad_state_starts=bad_state_starts, bad_state_fraction=args.bad_state_fraction,
        )
        return
    with log_path.open("w", newline="", encoding="utf-8") as handle, \
            (output / "transitions.csv").open("w", newline="", encoding="utf-8") as transition_handle, \
            (output / "updates.csv").open("w", newline="", encoding="utf-8") as update_handle, \
            (output / "progress.csv").open("w", newline="", encoding="utf-8") as progress_handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        transition_writer = csv.DictWriter(transition_handle, fieldnames=transition_fields); transition_writer.writeheader()
        update_writer = csv.DictWriter(update_handle, fieldnames=update_fields); update_writer.writeheader()
        progress_writer = csv.DictWriter(progress_handle, fieldnames=progress_fields); progress_writer.writeheader()
        probe_path = output / "orientation_probes.csv"
        probe_handle = probe_path.open("w", newline="", encoding="utf-8")
        probe_fields = [
            "global_step", "stage_step", "stage_total_step",
            "orientation_level_index", "orientation_scale",
            "orientation_level_steps", "episodes", "seed", "success_rate",
            "collision_rate", "joint_limit_rate", "timeout_rate",
            "mean_position_error_m", "mean_orientation_error_rad",
            "p95_orientation_error_rad",
            "mean_minimum_self_distance_m", "self_violation_rate",
            "strict_pose_hit_rate", "minimum_bin_success_rate",
            "minimum_bin_strict_pose_hit_rate",
            "weak_bin_count_below_0_6", "success_rate_o0_o4",
            "success_rate_o5_o6", "success_rate_o7_o9",
            "timeout_rate_o0_o4", "timeout_rate_o5_o6", "timeout_rate_o7_o9",
            "large_angle_orientation_steps",
            "large_angle_orientation_improvement_ratio",
            "large_angle_orientation_mean_rebound_rad",
            "previous_success_rate", "previous_collision_rate",
            "previous_joint_limit_rate", "screen_passed", "confirmation_ran",
            "confirmation_episodes", "confirmation_seed", "confirmation_success_rate",
            "confirmation_collision_rate", "confirmation_joint_limit_rate",
            "confirmation_previous_success_rate", "confirmation_previous_collision_rate",
            "confirmation_previous_joint_limit_rate", "confirmation_passed",
            "retention_passed", "recovery_required",
            "cross_level_success_rates",
            "cross_level_minimum_bin_success_rates",
            "s0_phase", "lambda_self", "probe_track", "passed", "is_best",
            "bad_probe_count", "rolled_back",
        ]
        probe_writer = csv.DictWriter(probe_handle, fieldnames=probe_fields)
        probe_writer.writeheader()

        def run_and_log_manual_probe() -> None:
            result, row = run_manual_pose_checkpoint_probe(
                agent, config, curriculum, orientation_curriculum, counters,
            )
            probe_writer.writerow(row)
            probe_handle.flush()
            metrics = result["current"]
            print(
                "manual checkpoint probe "
                f"step={counters['stage_step']} "
                f"level={curriculum.orientation.level_index} "
                f"success={metrics['success_rate']:.3f} "
                f"min_bin={metrics.get('minimum_bin_success_rate', float('nan')):.3f} "
                f"passed={int(bool(result['passed']))}; curriculum unchanged",
                flush=True,
            )

        best_probes: dict[str, dict[str, object]] = dict(
            counters.get("probe_best", {})
        )
        counters["probe_best"] = best_probes
        last_update: dict[str, float] = {}
        start_time = time.perf_counter()
        while counters["stage_step"] < total_steps:
            if active_episode is None:
                scene = curriculum.choose_scene(); xi, strict = curriculum.contract(scene)
                lambda_self = curriculum.lambda_self
                orientation_anchor = curriculum.choose_orientation_anchor()
                pose_contract = curriculum.pose_contract(previous=orientation_anchor)
                goal_scale = pose_contract["goal_scale"]
                orientation_scale = pose_contract["orientation_scale"]
                position_tolerance = pose_contract["position_tolerance"]
                orientation_tolerance = pose_contract["orientation_tolerance"]
                goal_sampling_kwargs = {}
                sampling_config = config["thesis"]["joint_pose_curriculum"].get(
                    "goal_scale_sampling", {}
                )
                if (
                    stage == "s0" and not orientation_anchor
                    and bool(sampling_config.get("enabled", False))
                ):
                    pose_levels = config["thesis"]["joint_pose_curriculum"]["levels"]
                    level_index = int(curriculum.orientation.level_index)
                    previous_index = max(0, level_index - 1)
                    goal_sampling_kwargs = {
                        "goal_scale_history_min": float(
                            sampling_config.get("history_min_scale", pose_levels[0]["goal_scale"])
                        ),
                        "goal_scale_frontier_min": float(
                            pose_levels[previous_index]["goal_scale"]
                        ),
                        "goal_scale_history_probability": float(
                            sampling_config.get("history_probability", 0.5)
                        ),
                    }
                goal_sampling_kwargs.update(task_space_training_kwargs(
                    config, curriculum, anchor=orientation_anchor,
                    bin_probabilities=counters.get(
                        "task_space_bin_probabilities"
                    ),
                ))
                env.configure_episode(
                    scene, xi=xi, strict=strict, goal_scale=goal_scale,
                    orientation_scale=orientation_scale,
                    position_tolerance=position_tolerance,
                    orientation_tolerance=orientation_tolerance,
                    lambda_self=lambda_self,
                    **goal_sampling_kwargs,
                )
                obs, _ = env.reset(); episode_return = 0.0; episode_length = 0
                final = {}; terminated = truncated = False; episode_used_warmup = False
            else:
                scene = active_episode["scene"]; xi, strict = curriculum.contract(scene)
                obs = env.restore_episode_state(active_episode["environment"])
                goal_scale = env.contract.goal_scale
                orientation_scale = env.contract.orientation_scale
                position_tolerance = env.contract.position_tolerance
                orientation_tolerance = env.contract.orientation_tolerance
                lambda_self = env.contract.lambda_self
                orientation_anchor = bool(active_episode.get("orientation_anchor", False))
                episode_return = active_episode["return"]; episode_length = active_episode["length"]
                episode_used_warmup = bool(active_episode["episode_used_warmup"])
                final = active_episode["final"]; terminated = truncated = False; active_episode = None
            while not (terminated or truncated) and counters["stage_step"] < total_steps:
                if stage == "s0" and counters["random_steps_used"] < int(config["sac"]["warmup_steps"]):
                    action = warmup_action_rng.uniform(-1.0, 1.0, size=6).astype(np.float32)
                    counters["random_steps_used"] += 1
                    used_warmup_transition = True
                else:
                    action = agent.select_action(obs)
                    used_warmup_transition = False
                episode_used_warmup = episode_used_warmup or used_warmup_transition
                next_obs, reward, _, terminated, truncated, info = env.step(action)
                curriculum.record_transition(
                    goal_scale, orientation_scale,
                    orientation_anchor=orientation_anchor,
                    curriculum_eligible=not used_warmup_transition,
                )
                replay.add(
                    scene, obs, action, next_obs, terminated or truncated, info,
                    counters["episodes"], episode_length,
                    orientation_anchor=orientation_anchor,
                    curriculum_level=int(curriculum.orientation.level_index),
                )
                record_keypoint_reward_statistics(
                    counters, info,
                    enabled=bool(config["thesis"]["reward"].get(
                        "keypoint_pose_reward", False,
                    )),
                )
                obs = next_obs; final = info; episode_return += reward; episode_length += 1
                counters["global_step"] += 1
                counters["stage_step"] += 1
                counters["stage_total_step"] += 1
                updates_due = (
                    accrue_update_credit(counters, updates_per_transition)
                    if replay.has_training_batch(stage, batch_size, update_after) else 0
                )
                for _ in range(updates_due):
                    replay_anchor_fraction, replay_current_fraction, _ = (
                        curriculum.orientation_replay_mix
                    )
                    update_actor = actor_updates_enabled(counters)
                    reference_observations = None
                    if (
                        agent.chain_pcr_enabled
                        and update_actor
                        and (agent.update_steps + 1) % agent.actor_update_interval == 0
                    ):
                        reference_batch = replay.sample(
                            stage, curriculum.xi_map(), batch_size,
                            orientation_scale=curriculum.orientation.scale,
                            s0_anchor_fraction=replay_anchor_fraction,
                            s0_current_fraction=replay_current_fraction,
                            lambda_self=curriculum.lambda_self,
                        )
                        reference_observations = reference_batch.observations
                    batch = replay.sample(
                        stage, curriculum.xi_map(), batch_size,
                        orientation_scale=curriculum.orientation.scale,
                        s0_anchor_fraction=replay_anchor_fraction,
                        s0_current_fraction=replay_current_fraction,
                        lambda_self=curriculum.lambda_self,
                    )
                    last_update = agent.update(
                        batch,
                        update_actor=update_actor,
                        reference_observations=reference_observations,
                    )
                    counters["updates"] += 1
                    update_writer.writerow({
                        "global_step": counters["global_step"], "stage_step": counters["stage_step"],
                        "stage_total_step": counters["stage_total_step"],
                        "update": counters["updates"], **last_update,
                        "sample_none": replay.last_sample_counts["none"],
                        "sample_static": replay.last_sample_counts["static"],
                        "sample_dynamic": replay.last_sample_counts["dynamic"],
                        "sample_orientation_anchor": replay.last_orientation_sample_counts["anchor"],
                        "sample_orientation_current": replay.last_orientation_sample_counts["current"],
                        "sample_orientation_historical": replay.last_orientation_sample_counts["historical"],
                        "sample_current_success": replay.last_orientation_sample_counts["current_success"],
                        "sample_current_recent": replay.last_orientation_sample_counts["current_recent"],
                        "sample_previous_level": replay.last_orientation_sample_counts["previous"],
                        "sample_semantic_long_term": replay.last_orientation_sample_counts.get("semantic_long_term", 0),
                        "critic_gradient_norm": last_update["critic_gradient_norm"],
                        "actor_gradient_norm": last_update["actor_gradient_norm"],
                        "actor_updated": last_update["actor_updated"],
                    })
                transition_writer.writerow({
                    "global_step": counters["global_step"], "stage_step": counters["stage_step"],
                    "stage_total_step": counters["stage_total_step"],
                    "episode": counters["episodes"], "episode_step": episode_length, "scene": scene,
                    "xi": xi, "lambda_self": lambda_self, "strict": int(strict), "reward": reward, "r_goal": info["r_goal"],
                    "c_proximity": info["c_proximity"], "hard_penalty": info["hard_penalty"],
                    "timeout_penalty": info["timeout_penalty"],
                    "safety_penalty": info["safety_penalty"],
                    "external_safety_penalty": info["external_safety_penalty"],
                    "self_safety_penalty": info["self_safety_penalty"],
                    "terminal_guard_penalty": info["terminal_guard_penalty"],
                    "clearance_violation": info["clearance_violation"],
                    "risk_max": info["control_max_risk"],
                    "d_min": info["control_min_distance"], "goal_scale": goal_scale,
                    "self_clearance_violation": info["self_clearance_violation"],
                    "self_risk_max": info["control_self_max_risk"],
                    "self_d_min": info["control_self_min_distance"],
                    "self_ttc_min": info["control_self_min_ttc"],
                    "self_approach_max": info["control_self_max_approach"],
                    "self_projection_intervened": int(info["self_projection_intervened"]),
                    "self_projection_infeasible": int(info["self_projection_infeasible"]),
                    "self_projection_correction_norm": info["self_projection_correction_norm"],
                    "self_projection_max_slack": info["self_projection_max_slack"],
                    "goal_full_scale_steps": curriculum.goal.full_scale_steps,
                    "orientation_scale": orientation_scale,
                    "position_tolerance": position_tolerance,
                    "orientation_tolerance": orientation_tolerance,
                    "orientation_reward_scale": info["orientation_reward_scale"],
                    "orientation_reward_gate": info["orientation_reward_gate"],
                    "rho_position": info["rho_position"],
                    "next_rho_position": info["next_rho_position"],
                    "rho_orientation": info["rho_orientation"],
                    "next_rho_orientation": info["next_rho_orientation"],
                    "position_quality": info["position_quality"],
                    "orientation_quality": info["orientation_quality"],
                    "keypoint_tracking_quality": info["keypoint_tracking_quality"],
                    "jacobian_clip_ratio": info["jacobian_clip_ratio"],
                    "keypoint_distance": info["keypoint_distance"],
                    "next_keypoint_distance": info["next_keypoint_distance"],
                    "keypoint_progress": info["keypoint_progress"],
                    "keypoint_tracking_reward": info["keypoint_tracking_reward"],
                    "keypoint_progress_reward": info["keypoint_progress_reward"],
                    "keypoint_precision_quality": info["keypoint_precision_quality"],
                    "keypoint_precision_reward": info["keypoint_precision_reward"],
                    "pose_potential": info["pose_potential"],
                    "next_pose_potential": info["next_pose_potential"],
                    "pose_potential_progress": info["pose_potential_progress"],
                    "position_progress": info["position_progress"],
                    "orientation_progress": info["orientation_progress"],
                    "orientation_error_progress": info["orientation_error_progress"],
                    "orientation_absolute_penalty": info["orientation_absolute_penalty"],
                    "orientation_absolute_position_gate": info[
                        "orientation_absolute_position_gate"
                    ],
                    "orientation_error_progress_reward": info["orientation_error_progress_reward"],
                    "orientation_completion_reward": info["orientation_completion_reward"],
                    "orientation_completion_position_gate": info[
                        "orientation_completion_position_gate"
                    ],
                    "orientation_shaping_reward": info["orientation_shaping_reward"],
                    "fine_position_quality": info["fine_position_quality"],
                    "fine_orientation_quality": info["fine_orientation_quality"],
                    "fine_position_progress": info["fine_position_progress"],
                    "fine_orientation_progress": info["fine_orientation_progress"],
                    "precision_proximity": info["precision_proximity"],
                    "precision_stop_cost": info["precision_stop_cost"],
                    "joint_tolerance_ratio": info["joint_tolerance_ratio"],
                    "next_joint_tolerance_ratio": info["next_joint_tolerance_ratio"],
                    "joint_precision_quality": info["joint_precision_quality"],
                    "next_joint_precision_quality": info["next_joint_precision_quality"],
                    "joint_precision_progress": info["joint_precision_progress"],
                    "joint_precision_reward": info["joint_precision_reward"],
                    "position_completion": info["position_completion"],
                    "orientation_completion": info["orientation_completion"],
                    "partial_precision_reward": info[
                        "partial_precision_reward"
                    ],
                    "left_joint_tolerance_region": int(
                        info["left_joint_tolerance_region"]
                    ),
                    "leave_joint_tolerance_penalty": info[
                        "leave_joint_tolerance_penalty"
                    ],
                    "strict_pose_reached": int(info["strict_pose_reached"]),
                    "success_hold_count": info["success_hold_count"],
                    "success_hold_steps_required": info[
                        "success_hold_steps_required"
                    ],
                    "orientation_anchor": int(orientation_anchor),
                    "position_error_m": info["next_rho_position"],
                    "orientation_error_rad": info["next_rho_orientation"],
                    "precision_action_scale": info["precision_action_scale"],
                    "velocity_magnitude": info["velocity_magnitude"],
                    "velocity_cost": info["velocity_cost"],
                    "smooth_velocity": info["smooth_velocity"],
                    "smooth_cost": info["smooth_cost"],
                    "curriculum_eligible": int(not used_warmup_transition),
                    "position_reached": int(info["position_reached"]),
                    "task_reached": int(info["task_reached"]),
                    "obstacle_collision": int(info["obstacle_collision"]), "self_collision": int(info["self_collision"]),
                    "environment_collision": int(info["environment_collision"]), "joint_limit": int(info["joint_limit"]),
                    "sample_none": replay.last_sample_counts["none"], "sample_static": replay.last_sample_counts["static"],
                    "sample_dynamic": replay.last_sample_counts["dynamic"],
                })
                if counters["stage_step"] % log_flush_interval == 0:
                    transition_handle.flush()
                    update_handle.flush()
                if counters["stage_step"] % progress_interval == 0:
                    elapsed = max(time.perf_counter() - start_time, 1e-9)
                    goal_success_rate = (
                        float(np.mean(curriculum.goal.outcomes))
                        if curriculum.goal.outcomes else ""
                    )
                    orientation_success_rate = (
                        float(np.mean(curriculum.orientation.outcomes))
                        if curriculum.orientation.outcomes else ""
                    )
                    replay_anchor_fraction, replay_current_fraction, replay_historical_fraction = (
                        curriculum.orientation_replay_mix
                    )
                    replay_storage_counts = replay.orientation_storage_counts()
                    progress = {
                        "global_step": counters["global_step"], "stage_step": counters["stage_step"],
                        "stage_total_step": counters["stage_total_step"],
                        "episodes": counters["episodes"], "replay_size": len(replay),
                        "updates": counters["updates"], "alpha": last_update.get("alpha", float(agent.alpha.detach().cpu())),
                        "critic_loss": last_update.get("critic_loss", ""),
                        "actor_loss": last_update.get("actor_loss", ""),
                        "actor_sac_loss": last_update.get("actor_sac_loss", ""),
                        "policy_churn_kl": last_update.get("policy_churn_kl", ""),
                        "policy_churn_penalty": last_update.get(
                            "policy_churn_penalty", ""
                        ),
                        "policy_churn_to_sac_ratio": last_update.get(
                            "policy_churn_to_sac_ratio", ""
                        ),
                        "chain_pcr_effective_coefficient": last_update.get(
                            "chain_pcr_effective_coefficient", ""
                        ),
                        "chain_pcr_sac_loss_ema": last_update.get(
                            "chain_pcr_sac_loss_ema", ""
                        ),
                        "chain_pcr_loss_ema": last_update.get(
                            "chain_pcr_loss_ema", ""
                        ),
                        **keypoint_reward_statistics(counters),
                        "q1_mean": last_update.get("q1_mean", ""), "q2_mean": last_update.get("q2_mean", ""),
                        "target_mean": last_update.get("target_mean", ""),
                        "steps_per_second": counters["stage_step"] / elapsed,
                        "goal_scale": curriculum.goal_scale,
                        "goal_level_index": curriculum.goal.level_index,
                        "goal_level_steps": curriculum.goal.current_level_steps,
                        "rolling_goal_success_rate": goal_success_rate,
                        "goal_eligible_steps": curriculum.goal.eligible_steps,
                        "goal_full_scale_steps": curriculum.goal.full_scale_steps,
                        "s0_goal_gate_eligible": int(
                            curriculum.s0_goal_gate_eligible(
                                full_scale_min_transitions, pose_full_scale_min_transitions
                            )
                        ),
                        "orientation_scale": curriculum.orientation_scale,
                        "position_tolerance": curriculum.position_tolerance,
                        "orientation_tolerance": curriculum.orientation_tolerance,
                        "rolling_orientation_success_rate": orientation_success_rate,
                        "orientation_eligible_steps": curriculum.orientation.eligible_steps,
                        "orientation_level_index": curriculum.orientation.level_index,
                        "orientation_level_steps": curriculum.orientation.current_level_steps,
                        "rolling_anchor_position_success_rate": (
                            "" if curriculum.orientation_anchor_success_rate is None
                            else curriculum.orientation_anchor_success_rate
                        ),
                        "rolling_previous_pose_success_rate": (
                            "" if curriculum.orientation_anchor_success_rate is None
                            else curriculum.orientation_anchor_success_rate
                        ),
                        "sample_orientation_anchor": replay.last_orientation_sample_counts["anchor"],
                        "sample_orientation_current": replay.last_orientation_sample_counts["current"],
                        "sample_orientation_historical": replay.last_orientation_sample_counts["historical"],
                        "sample_current_success": replay.last_orientation_sample_counts["current_success"],
                        "sample_current_recent": replay.last_orientation_sample_counts["current_recent"],
                        "sample_previous_level": replay.last_orientation_sample_counts["previous"],
                        "sample_semantic_long_term": replay.last_orientation_sample_counts.get("semantic_long_term", 0),
                        "full_pose_steps": curriculum.orientation.full_scale_steps,
                        "position_phase_complete": int(curriculum.position_phase_complete),
                        "orientation_retention_mode": curriculum.orientation_retention_mode,
                        "orientation_anchor_probability": curriculum.orientation_anchor_probability,
                        "replay_orientation_anchor_fraction": replay_anchor_fraction,
                        "replay_orientation_current_fraction": replay_current_fraction,
                        "replay_orientation_historical_fraction": replay_historical_fraction,
                        "replay_current_success_fraction": replay.effective_s0_success_fraction,
                        "replay_current_recent_fraction": (
                            1.0 - replay_anchor_fraction
                            - replay.effective_s0_success_fraction
                            - replay.semantic_fraction
                        ),
                        "replay_semantic_long_term_fraction": replay.semantic_fraction,
                        "replay_previous_level_fraction": replay_anchor_fraction,
                        "stored_orientation_anchor": replay_storage_counts["anchor"],
                        "stored_orientation_current": replay_storage_counts["current"],
                        "stored_orientation_historical": replay_storage_counts["historical"],
                        "stored_orientation_historical_levels": len(
                            replay_storage_counts["historical_levels"]
                        ),
                        "stored_current_success_episodes": (
                            replay.current_success_episode_count
                        ),
                        "stored_semantic_long_term_transitions": replay_storage_counts.get("semantic_long_term_transitions", 0),
                        "stored_semantic_long_term_episodes": replay_storage_counts.get("semantic_long_term_episodes", 0),
                        "stored_semantic_long_term_nonempty_bins": replay_storage_counts.get("semantic_long_term_nonempty_bins", 0),
                        "deterministic_probe_passed": int(
                            curriculum.orientation.deterministic_probe_passed
                        ),
                        "deterministic_probe_success_rate": (
                            "" if curriculum.orientation.deterministic_probe_success_rate is None
                            else curriculum.orientation.deterministic_probe_success_rate
                        ),
                        "deterministic_probe_collision_rate": (
                            "" if curriculum.orientation.deterministic_probe_collision_rate is None
                            else curriculum.orientation.deterministic_probe_collision_rate
                        ),
                        "deterministic_probe_joint_limit_rate": (
                            "" if curriculum.orientation.deterministic_probe_joint_limit_rate is None
                            else curriculum.orientation.deterministic_probe_joint_limit_rate
                        ),
                        "deterministic_probe_previous_success_rate": (
                            "" if curriculum.orientation.deterministic_probe_previous_success_rate is None
                            else curriculum.orientation.deterministic_probe_previous_success_rate
                        ),
                        "deterministic_probe_confirmation_passed": int(
                            curriculum.orientation.deterministic_probe_confirmation_passed
                        ),
                        "lambda_self": curriculum.lambda_self,
                        "s0_phase": curriculum.s0_phase if stage == "s0" else stage,
                        "pose_phase_complete": int(curriculum.pose_phase_complete),
                        "self_safety_eligible_steps": curriculum.self_safety.eligible_steps,
                        "self_safety_full_weight_steps": (
                            curriculum.self_safety.full_weight_steps
                        ),
                    }
                    progress_writer.writerow(progress); progress_handle.flush()
                    print(
                        f"step={counters['stage_step']}/{total_steps} episodes={counters['episodes']} "
                        f"replay={len(replay)} updates={counters['updates']} alpha={float(progress['alpha']):.5f} "
                        f"goal_scale={curriculum.goal_scale:.4f} "
                        f"goal_success={goal_success_rate if goal_success_rate == '' else f'{goal_success_rate:.3f}'} "
                        f"orientation_scale={curriculum.orientation_scale:.4f} "
                        f"orientation_tol={curriculum.orientation_tolerance:.3f} "
                        f"orientation_level={curriculum.orientation.level_index} "
                        f"anchor_success={curriculum.orientation_anchor_success_rate} "
                        f"retention={curriculum.orientation_retention_mode} "
                        f"anchor_p={curriculum.orientation_anchor_probability:.2f} "
                        f"phase={curriculum.s0_phase if stage == 's0' else stage} "
                        f"lambda_self={curriculum.lambda_self:.3f} "
                        f"full_goal_steps={curriculum.goal.full_scale_steps} "
                        f"gate_eligible={progress['s0_goal_gate_eligible']} "
                        f"steps/s={progress['steps_per_second']:.1f}",
                        flush=True,
                    )
                if counters["stage_step"] % save_interval == 0 and not (terminated or truncated):
                    if (
                        evaluate_during_training
                        and stage == "s0" and manual_promotion
                        and counters["stage_step"] % manual_probe_interval == 0
                    ):
                        run_and_log_manual_probe()
                    active_episode = {"scene": scene, "return": episode_return, "length": episode_length,
                                      "orientation_anchor": orientation_anchor,
                                      "final": final, "episode_used_warmup": episode_used_warmup,
                                      "environment": env.episode_state_dict()}
                    counters["checkpoint_index"] += 1; counters["replay_redistributions"] = replay.redistribution_count
                    checkpoint = checkpoints / f"step_{counters['stage_step']:07d}.pt"
                    save_checkpoint(checkpoint, agent, replay, curriculum, counters, config, env, active_episode)
                    agent.save_actor(checkpoints / f"actor_step_{counters['stage_step']:07d}.pt")
                    print(f"saved {checkpoint} scene={scene} xi={xi:.4f} replay={len(replay)}", flush=True)
                    last_saved_step = counters["stage_step"]
                    active_episode = None
            if not (terminated or truncated):
                active_episode = {"scene": scene, "return": episode_return, "length": episode_length,
                                  "orientation_anchor": orientation_anchor,
                                  "final": final, "episode_used_warmup": episode_used_warmup,
                                  "environment": env.episode_state_dict()}
                break
            update = curriculum.finish_episode(
                scene, bool(final.get("position_reached", False)),
                bool(final.get("task_reached", False)), episode_length,
                orientation_anchor=orientation_anchor,
                curriculum_eligible=not episode_used_warmup,
            )
            if (
                evaluate_during_training
                and not manual_promotion
                and curriculum.orientation_probe_due(
                    int(orientation_curriculum["deterministic_probe_interval_transitions"]),
                    pose_full_scale_min_transitions,
                )
            ):
                probe_level_index = curriculum.orientation.level_index
                probe_scale = curriculum.orientation.scale
                probe_result = run_pose_promotion_probe(
                    agent, config, curriculum, orientation_curriculum,
                )
                update_adaptive_task_space_sampling(
                    config, counters, probe_result,
                )
                metrics = probe_result["current"]
                retention_metrics = probe_result["previous"]
                recorded_metrics = probe_result["confirmation_current"] or metrics
                recorded_retention = (
                    probe_result["confirmation_previous"] or retention_metrics
                )
                probe_phase = curriculum.s0_phase
                probe_lambda = curriculum.lambda_self
                probe_track = s0_probe_track_key(curriculum)
                probe_passed = bool(probe_result["passed"])
                curriculum.record_orientation_probe(
                    success_rate=recorded_metrics["success_rate"],
                    collision_rate=recorded_metrics["collision_rate"],
                    joint_limit_rate=recorded_metrics["joint_limit_rate"],
                    previous_success_rate=(
                        None if recorded_retention is None
                        else recorded_retention["success_rate"]
                    ),
                    previous_collision_rate=(
                        None if recorded_retention is None
                        else recorded_retention["collision_rate"]
                    ),
                    previous_joint_limit_rate=(
                        None if recorded_retention is None
                        else recorded_retention["joint_limit_rate"]
                    ),
                    confirmation_passed=bool(
                        probe_result["confirmation_passed"]
                    ),
                    retention_passed=bool(probe_result["retention_passed"]),
                    recovery_required=bool(probe_result["recovery_required"]),
                    passed=probe_passed,
                )
                level_steps_at_probe = curriculum.orientation.current_level_steps
                previous_best = best_probes.get(probe_track)
                score = s0_probe_score(
                    probe_track, probe_result, orientation_curriculum,
                    self_collision_ceiling=curriculum.self_probe_collision_ceiling,
                )
                is_best = s0_probe_score_is_better(
                    probe_track, score, previous_best, orientation_curriculum,
                    self_collision_ceiling=curriculum.self_probe_collision_ceiling,
                )
                rolled_back = False
                if is_best:
                    best_path = checkpoints / f"best_{probe_track}.pt"
                    best_probes[probe_track] = {
                        "score": score,
                        "success_rate": metrics["success_rate"],
                        "score_version": S0_PROBE_SCORE_VERSION,
                        "bad_count": 0,
                        "path": str(best_path),
                    }
                    counters["probe_best"] = best_probes
                    save_checkpoint(
                        best_path, agent, replay, curriculum, counters, config, env, None,
                    )
                else:
                    assert previous_best is not None
                    if metrics["success_rate"] <= (
                        float(previous_best["success_rate"])
                        - float(orientation_curriculum["rollback_success_drop"])
                    ):
                        previous_best["bad_count"] = int(previous_best["bad_count"]) + 1
                    else:
                        previous_best["bad_count"] = 0
                    if int(previous_best["bad_count"]) >= int(
                        orientation_curriculum["rollback_patience"]
                    ):
                        best_state = torch.load(
                            previous_best["path"], map_location=agent.device,
                            weights_only=False,
                        )
                        agent.load_state_dict(best_state["agent"])
                        factor = float(orientation_curriculum["rollback_actor_lr_factor"])
                        for group in agent.actor_optimizer.param_groups:
                            group["lr"] *= factor
                        previous_best["bad_count"] = 0
                        rolled_back = True
                counters["probe_best"] = best_probes
                advanced = curriculum.advance_orientation_if_ready()
                if advanced:
                    replay.begin_orientation_level(curriculum.orientation.scale)
                    contract = curriculum.pose_contract()
                    replay.update_semantic_bounds(
                        distance_min=contract["target_distance_min_m"],
                        distance_max=contract["target_distance_max_m"],
                        orientation_min=contract["target_orientation_min_rad"],
                        orientation_max=contract["target_orientation_max_rad"],
                    )
                probe_writer.writerow({
                    "global_step": counters["global_step"],
                    "stage_step": counters["stage_step"],
                    "stage_total_step": counters["stage_total_step"],
                    "orientation_level_index": probe_level_index,
                    "orientation_scale": probe_scale,
                    "orientation_level_steps": level_steps_at_probe,
                    "episodes": orientation_curriculum["deterministic_probe_episodes"],
                    "seed": orientation_curriculum["deterministic_probe_seed"],
                    **scalar_probe_metrics(metrics),
                    **pose_probe_audit_fields(
                        probe_result, orientation_curriculum,
                    ),
                    "s0_phase": probe_phase,
                    "lambda_self": probe_lambda,
                    "probe_track": probe_track,
                    "passed": int(probe_passed),
                    "is_best": int(is_best),
                    "bad_probe_count": int(
                        best_probes[probe_track]["bad_count"]
                    ),
                    "rolled_back": int(rolled_back),
                })
                probe_handle.flush()
                print(
                    f"orientation probe eta={probe_scale:.4f} "
                    f"success={metrics['success_rate']:.3f} "
                    f"collision={metrics['collision_rate']:.3f} "
                    f"joint_limit={metrics['joint_limit_rate']:.3f} "
                    f"passed={int(probe_passed)} best={int(is_best)} "
                    f"rollback={int(rolled_back)} advanced={int(advanced)}",
                    flush=True,
                )
            if update["became_strict"]:
                report = replay.strictify(scene)
                curriculum.states[scene].replay_strictified = True
                manifest.setdefault("strictification", []).append({"scene": scene, "global_step": counters["global_step"], **report})
                (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            state = curriculum.states.get(scene)
            writer.writerow({
                "episode": counters["episodes"], "global_step": counters["global_step"],
                "stage_step": counters["stage_step"],
                "stage_total_step": counters["stage_total_step"],
                "scene": scene, "length": episode_length, "return": episode_return,
                "curriculum_eligible": int(not episode_used_warmup),
                "orientation_anchor": int(orientation_anchor),
                "position_reached": int(final.get("position_reached", False)),
                "strict_pose_reached": int(final.get("strict_pose_reached", False)),
                "success_hold_count": int(final.get("success_hold_count", 0)),
                "success_hold_steps_required": int(
                    final.get("success_hold_steps_required", 1)
                ),
                "task_reached": int(final.get("task_reached", False)),
                "collision_assisted_reach": int(final.get("collision_assisted_reach", False)),
                "safe_success": int(final.get("safe_success", False)),
                "obstacle_collision": int(final.get("obstacle_collision", False)), "self_collision": int(final.get("self_collision", False)),
                "environment_collision": int(final.get("environment_collision", False)), "joint_limit": int(final.get("joint_limit", False)),
                "timeout": int(truncated), "xi": 1.0 if state is None else state.xi,
                "lambda_self": lambda_self,
                "strict": 1 if state is None else int(state.strict),
                "rolling_task_reach_rate": "" if state is None else update["rolling_task_reach_rate"],
                "eligible_steps": 0 if state is None else state.eligible_steps, "strict_steps": 0 if state is None else state.strict_steps,
                "goal_scale": goal_scale,
                "rolling_goal_success_rate": update["rolling_goal_success_rate"],
                "goal_eligible_steps": curriculum.goal.eligible_steps,
                "goal_level_index": curriculum.goal.level_index,
                "goal_level_steps": curriculum.goal.current_level_steps,
                "goal_full_scale_steps": curriculum.goal.full_scale_steps,
                "orientation_scale": orientation_scale,
                "position_tolerance": position_tolerance,
                "orientation_tolerance": orientation_tolerance,
                "rolling_orientation_success_rate": update["rolling_orientation_success_rate"],
                "orientation_eligible_steps": curriculum.orientation.eligible_steps,
                "orientation_level_index": curriculum.orientation.level_index,
                "orientation_level_steps": curriculum.orientation.current_level_steps,
                "rolling_anchor_position_success_rate": (
                    "" if curriculum.orientation_anchor_success_rate is None
                    else curriculum.orientation_anchor_success_rate
                ),
                "rolling_previous_pose_success_rate": (
                    "" if curriculum.orientation_anchor_success_rate is None
                    else curriculum.orientation_anchor_success_rate
                ),
                "full_pose_steps": curriculum.orientation.full_scale_steps,
                "position_phase_complete": int(curriculum.position_phase_complete),
                "position_error_m": final.get("next_rho_position", ""),
                "orientation_error_rad": final.get("next_rho_orientation", ""),
                "orientation_retention_mode": curriculum.orientation_retention_mode,
                "orientation_anchor_probability": curriculum.orientation_anchor_probability,
                "deterministic_probe_passed": int(
                    curriculum.orientation.deterministic_probe_passed
                ),
                "deterministic_probe_success_rate": (
                    "" if curriculum.orientation.deterministic_probe_success_rate is None
                    else curriculum.orientation.deterministic_probe_success_rate
                ),
                "s0_phase": curriculum.s0_phase if stage == "s0" else stage,
                "pose_phase_complete": int(curriculum.pose_phase_complete),
                "self_safety_eligible_steps": curriculum.self_safety.eligible_steps,
                "self_safety_full_weight_steps": curriculum.self_safety.full_weight_steps,
            }); counters["episodes"] += 1
            if counters["episodes"] % log_flush_interval == 0:
                handle.flush()
            if (counters["stage_step"] % save_interval == 0 or counters["stage_step"] >= total_steps) and last_saved_step != counters["stage_step"]:
                if (
                    evaluate_during_training
                    and stage == "s0" and manual_promotion
                    and counters["stage_step"] % manual_probe_interval == 0
                ):
                    run_and_log_manual_probe()
                counters["checkpoint_index"] += 1
                checkpoint = checkpoints / f"step_{counters['stage_step']:07d}.pt"
                counters["replay_redistributions"] = replay.redistribution_count
                save_checkpoint(checkpoint, agent, replay, curriculum, counters, config, env, None)
                agent.save_actor(checkpoints / f"actor_step_{counters['stage_step']:07d}.pt")
                transition_handle.flush(); update_handle.flush(); handle.flush()
                print(f"saved {checkpoint} scene={scene} xi={xi:.4f} replay={len(replay)}", flush=True)
                last_saved_step = counters["stage_step"]
        if active_episode is not None and last_saved_step != counters["stage_step"]:
            counters["checkpoint_index"] += 1; counters["replay_redistributions"] = replay.redistribution_count
            checkpoint = checkpoints / f"step_{counters['stage_step']:07d}.pt"
            save_checkpoint(checkpoint, agent, replay, curriculum, counters, config, env, active_episode)
            agent.save_actor(checkpoints / f"actor_step_{counters['stage_step']:07d}.pt")
            print(f"saved {checkpoint} active_episode=1 replay={len(replay)}", flush=True)
        probe_handle.close()
    env.close()
    summary = {"stage": stage, **counters, "replay_size": len(replay),
               "num_envs": num_envs, "xi": curriculum.xi_map(),
               "replay_orientation_storage": replay.orientation_storage_counts(),
               "max_stage_steps": max_stage_steps,
               "stage_budget_exhausted": counters["stage_total_step"] >= max_stage_steps,
               "goal_scale": curriculum.goal_scale,
               "goal_eligible_steps": curriculum.goal.eligible_steps,
               "goal_level_index": curriculum.goal.level_index,
               "goal_level_steps": curriculum.goal.current_level_steps,
               "goal_full_scale_steps": curriculum.goal.full_scale_steps,
               "orientation_scale": curriculum.orientation_scale,
               "position_tolerance": curriculum.position_tolerance,
               "orientation_tolerance": curriculum.orientation_tolerance,
               "orientation_eligible_steps": curriculum.orientation.eligible_steps,
               "orientation_level_index": curriculum.orientation.level_index,
               "orientation_level_steps": curriculum.orientation.current_level_steps,
               "rolling_orientation_success_rate": curriculum.orientation_success_rate,
               "rolling_anchor_position_success_rate": curriculum.orientation_anchor_success_rate,
               "rolling_previous_pose_success_rate": curriculum.orientation_anchor_success_rate,
               "orientation_retention_mode": curriculum.orientation_retention_mode,
               "orientation_anchor_probability": curriculum.orientation_anchor_probability,
               "orientation_replay_mix": curriculum.orientation_replay_mix,
               "deterministic_probe_passed": curriculum.orientation.deterministic_probe_passed,
               "deterministic_probe_success_rate": curriculum.orientation.deterministic_probe_success_rate,
               "deterministic_probe_collision_rate": curriculum.orientation.deterministic_probe_collision_rate,
               "deterministic_probe_joint_limit_rate": curriculum.orientation.deterministic_probe_joint_limit_rate,
               "deterministic_probe_previous_success_rate": (
                   curriculum.orientation.deterministic_probe_previous_success_rate
               ),
               "deterministic_probe_confirmation_passed": (
                   curriculum.orientation.deterministic_probe_confirmation_passed
               ),
               "full_pose_steps": curriculum.orientation.full_scale_steps,
               "position_phase_complete": curriculum.position_phase_complete,
               "pose_phase_complete": curriculum.pose_phase_complete,
               "s0_phase": curriculum.s0_phase if stage == "s0" else stage,
               "lambda_self": curriculum.lambda_self,
               "self_safety_eligible_steps": curriculum.self_safety.eligible_steps,
               "self_safety_full_weight_steps": curriculum.self_safety.full_weight_steps,
               "self_safety_probe_collision_ceiling": curriculum.self_probe_collision_ceiling,
               "self_safety_probe_passed": curriculum.self_safety_probe_passed,
               "s0_goal_gate_eligible": curriculum.s0_goal_gate_eligible(
                   full_scale_min_transitions, pose_full_scale_min_transitions
               ),
               "strict_steps": {k: v.strict_steps for k, v in curriculum.states.items()}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
