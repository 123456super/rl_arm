#!/usr/bin/env python3
"""Diagnose SAC learning capacity from a sequence of complete checkpoints."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.algorithms.homotopy_replay import _vectorized_replay_rewards
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent


def _checkpoint_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_checkpoint(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"agent", "replay", "config", "counters", "curriculum"}
    missing = required.difference(state)
    if missing:
        raise ValueError(
            f"{path} is not a complete checkpoint; missing {sorted(missing)}"
        )
    return state


def _make_agent(state: dict[str, Any]) -> ThesisSACAgent:
    actor_state = state["agent"]["actor"]
    obs_dim = int(actor_state["backbone.0.weight"].shape[1])
    action_dim = int(actor_state["mean.weight"].shape[0])
    config = copy.deepcopy(state["config"])
    config["device"] = "cpu"
    agent = ThesisSACAgent(obs_dim, action_dim, config)
    agent.load_state_dict(state["agent"])
    agent.actor.eval()
    agent.q1.eval()
    agent.q2.eval()
    agent.target_q1.eval()
    agent.target_q2.eval()
    return agent


def _probe_from_replay(
    state: dict[str, Any], samples: int, seed: int,
) -> dict[str, np.ndarray]:
    part = state["replay"].get("s0_current")
    if part is None or int(part.size) < 1:
        raise ValueError("plasticity probe currently requires non-empty S0 current replay")
    order = part.chronological_indices()
    episodes = part.episode[order]
    steps = part.step[order]
    raw = part.raw[order]

    initial_orientation: dict[int, float] = {}
    for episode, step, row in zip(episodes, steps, raw):
        if int(step) == 0:
            initial_orientation[int(episode)] = float(row[2])
    valid_positions = np.asarray(
        [i for i, episode in enumerate(episodes) if int(episode) in initial_orientation],
        dtype=np.int64,
    )
    if not len(valid_positions):
        raise ValueError("S0 replay has no episodes whose initial transition is retained")
    config = state["config"]
    levels = config["thesis"]["joint_pose_curriculum"]["levels"]
    level_index = int(state["curriculum"]["orientation"].level_index)
    level = levels[level_index]
    lower = float(level["target_orientation_min_rad"])
    upper = float(level["target_orientation_max_rad"])
    bins = int(config["thesis"]["orientation_curriculum"].get(
        "deterministic_probe_orientation_bins", 10
    ))
    initial = np.asarray(
        [initial_orientation[int(episodes[position])] for position in valid_positions],
        dtype=np.float64,
    )
    all_orientation_bins = np.minimum(
        ((np.clip(initial, lower, upper) - lower) / (upper - lower) * bins).astype(int),
        bins - 1,
    )
    all_groups = np.full(len(valid_positions), "o7_o9", dtype="U5")
    all_groups[all_orientation_bins <= 6] = "o5_o6"
    all_groups[all_orientation_bins <= 4] = "o0_o4"

    # Long hard episodes contain more transitions than easy successes. Draw
    # approximately equal state counts per initial-orientation band so that
    # overall activation/churn statistics are not dominated by episode length.
    rng = np.random.default_rng(seed)
    count = min(int(samples), len(valid_positions))
    group_names = ("o0_o4", "o5_o6", "o7_o9")
    targets = {
        group: count // len(group_names) + int(i < count % len(group_names))
        for i, group in enumerate(group_names)
    }
    selected_local: list[int] = []
    for group in group_names:
        candidates = np.flatnonzero(all_groups == group)
        take = min(targets[group], len(candidates))
        if take:
            selected_local.extend(
                rng.choice(candidates, size=take, replace=False).tolist()
            )
    if len(selected_local) < count:
        remaining = np.setdiff1d(
            np.arange(len(valid_positions)), np.asarray(selected_local),
            assume_unique=False,
        )
        selected_local.extend(
            rng.choice(
                remaining, size=count - len(selected_local), replace=False
            ).tolist()
        )
    selected_local_array = np.asarray(selected_local, dtype=np.int64)
    selected_positions = valid_positions[selected_local_array]
    sort_order = np.argsort(selected_positions)
    selected_positions = selected_positions[sort_order]
    selected_local_array = selected_local_array[sort_order]
    selected = order[selected_positions]
    orientation_bin = all_orientation_bins[selected_local_array]
    groups = all_groups[selected_local_array]

    return {
        "observations": part.obs[selected].copy(),
        "actions": part.action[selected].copy(),
        "next_observations": part.next_obs[selected].copy(),
        "dones": part.done[selected].copy(),
        "raw": part.raw[selected].copy(),
        "groups": groups,
        "orientation_bins": orientation_bin,
        "episodes": part.episode[selected].copy(),
    }


def normalized_activation_scores(features: np.ndarray) -> np.ndarray:
    mean_abs = np.mean(np.abs(np.asarray(features, dtype=np.float64)), axis=0)
    denominator = float(np.mean(mean_abs))
    if denominator <= 1e-12:
        return np.zeros_like(mean_abs)
    return mean_abs / denominator


def dormant_ratios(
    features: np.ndarray, thresholds: tuple[float, ...],
) -> dict[str, Any]:
    scores = normalized_activation_scores(features)
    return {
        "units": int(len(scores)),
        "mean_absolute_activation": float(np.mean(np.abs(features))),
        "normalized_score_min": float(np.min(scores)),
        "normalized_score_median": float(np.median(scores)),
        "ratios": {
            str(threshold): float(np.mean(scores <= threshold))
            for threshold in thresholds
        },
    }


def effective_rank(features: np.ndarray) -> dict[str, float | int]:
    matrix = torch.as_tensor(features, dtype=torch.float32)
    singular = torch.linalg.svdvals(matrix).cpu().numpy().astype(np.float64)
    total = float(np.sum(singular))
    if total <= 1e-12:
        return {"effective_rank": 0.0, "rank_99pct_energy": 0}
    probabilities = singular / total
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    energy = np.square(singular)
    energy_total = float(np.sum(energy))
    rank_99 = int(np.searchsorted(np.cumsum(energy) / energy_total, .99) + 1)
    return {
        "effective_rank": float(np.exp(entropy)),
        "rank_99pct_energy": rank_99,
    }


def _sequential_features(
    sequential: nn.Sequential,
    inputs: np.ndarray,
    *,
    exclude_last_linear: bool,
    batch_size: int,
) -> dict[str, np.ndarray]:
    linear_indices = [
        index for index, layer in enumerate(sequential) if isinstance(layer, nn.Linear)
    ]
    if exclude_last_linear:
        linear_indices = linear_indices[:-1]
    collected: dict[int, list[np.ndarray]] = {index: [] for index in linear_indices}
    tensor = torch.as_tensor(inputs, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            value = tensor[start:start + batch_size]
            pending: int | None = None
            for index, layer in enumerate(sequential):
                value = layer(value)
                if isinstance(layer, nn.Linear) and index in collected:
                    pending = index
                    if index + 1 >= len(sequential) or not isinstance(
                        sequential[index + 1], nn.ReLU
                    ):
                        collected[index].append(value.cpu().numpy())
                        pending = None
                elif isinstance(layer, nn.ReLU) and pending is not None:
                    collected[pending].append(value.cpu().numpy())
                    pending = None
    return {
        f"layer_{position + 1}": np.concatenate(collected[index], axis=0)
        for position, index in enumerate(linear_indices)
    }


def network_activity(
    agent: ThesisSACAgent,
    observations: np.ndarray,
    critic_actions: np.ndarray,
    thresholds: tuple[float, ...],
    batch_size: int,
) -> dict[str, Any]:
    actor_features = _sequential_features(
        agent.actor.backbone, observations,
        exclude_last_linear=False, batch_size=batch_size,
    )
    critic_inputs = np.concatenate((observations, critic_actions), axis=1)
    result: dict[str, Any] = {}
    for name, features_by_layer in (
        ("actor", actor_features),
        ("q1", _sequential_features(
            agent.q1.net, critic_inputs,
            exclude_last_linear=True, batch_size=batch_size,
        )),
        ("q2", _sequential_features(
            agent.q2.net, critic_inputs,
            exclude_last_linear=True, batch_size=batch_size,
        )),
    ):
        result[name] = {
            layer: dormant_ratios(features, thresholds)
            for layer, features in features_by_layer.items()
        }
        last_layer = features_by_layer[list(features_by_layer)[-1]]
        result[name]["last_layer_rank"] = effective_rank(last_layer)
    return result


def _gradient_norm(module: nn.Module) -> float:
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += float(torch.sum(parameter.grad.detach().square()))
    return float(np.sqrt(squared))


def gradient_activity(
    agent: ThesisSACAgent,
    probe: dict[str, np.ndarray],
    rewards: np.ndarray,
    sample_count: int,
    seed: int,
) -> dict[str, float]:
    count = min(int(sample_count), len(probe["observations"]))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(probe["observations"]), size=count, replace=False)
    obs = torch.as_tensor(probe["observations"][indices], dtype=torch.float32)
    actions = torch.as_tensor(probe["actions"][indices], dtype=torch.float32)
    next_obs = torch.as_tensor(probe["next_observations"][indices], dtype=torch.float32)
    dones = torch.as_tensor(probe["dones"][indices], dtype=torch.float32)
    reward = torch.as_tensor(rewards[indices, None], dtype=torch.float32)

    agent.q_optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    with torch.no_grad():
        next_action, next_log_prob = agent.actor.sample(next_obs)
        next_q = torch.min(
            agent.target_q1(next_obs, next_action),
            agent.target_q2(next_obs, next_action),
        )
        target = reward + agent.gamma * (1.0 - dones) * (
            next_q - agent.alpha.detach() * next_log_prob
        )
    q1_loss = F.smooth_l1_loss(agent.q1(obs, actions), target)
    q2_loss = F.smooth_l1_loss(agent.q2(obs, actions), target)
    (q1_loss + q2_loss).backward()
    q1_gradient = _gradient_norm(agent.q1)
    q2_gradient = _gradient_norm(agent.q2)

    agent.q_optimizer.zero_grad(set_to_none=True)
    agent.actor_optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    sampled_action, log_prob = agent.actor.sample(obs)
    actor_loss = (
        agent.alpha.detach() * log_prob
        - torch.min(agent.q1(obs, sampled_action), agent.q2(obs, sampled_action))
    ).mean()
    actor_loss.backward()
    actor_gradient = _gradient_norm(agent.actor)
    agent.actor_optimizer.zero_grad(set_to_none=True)
    agent.q_optimizer.zero_grad(set_to_none=True)
    return {
        "samples": count,
        "actor_loss": float(actor_loss.detach()),
        "q1_loss": float(q1_loss.detach()),
        "q2_loss": float(q2_loss.detach()),
        "actor_gradient_norm": actor_gradient,
        "q1_gradient_norm": q1_gradient,
        "q2_gradient_norm": q2_gradient,
    }


def parameter_norms(agent: ThesisSACAgent) -> dict[str, float]:
    result = {}
    for name, module in (("actor", agent.actor), ("q1", agent.q1), ("q2", agent.q2)):
        squared = sum(float(torch.sum(p.detach().square())) for p in module.parameters())
        result[name] = float(np.sqrt(squared))
    return result


def relative_parameter_update(
    previous: dict[str, torch.Tensor], current: dict[str, torch.Tensor],
) -> float:
    if previous.keys() != current.keys():
        raise ValueError("parameter sets differ between checkpoints")
    difference = 0.0
    baseline = 0.0
    for key in previous:
        left = previous[key].detach().cpu().to(torch.float64)
        right = current[key].detach().cpu().to(torch.float64)
        if left.shape != right.shape:
            raise ValueError(f"parameter shape changed for {key}")
        difference += float(torch.sum((right - left).square()))
        baseline += float(torch.sum(left.square()))
    return float(np.sqrt(difference) / (np.sqrt(baseline) + 1e-12))


def churn_metrics(
    previous_actions: np.ndarray,
    current_actions: np.ndarray,
    groups: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    distances = np.linalg.norm(current_actions - previous_actions, axis=1)

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        selected = distances[mask]
        if not len(selected):
            return {"samples": 0, "mean_l2": None, "fractions": {}}
        return {
            "samples": int(len(selected)),
            "mean_l2": float(np.mean(selected)),
            "median_l2": float(np.median(selected)),
            "p95_l2": float(np.quantile(selected, .95)),
            "fractions": {
                str(threshold): float(np.mean(selected > threshold))
                for threshold in thresholds
            },
        }

    result = {"all": summarize(np.ones(len(groups), dtype=bool))}
    for group in ("o0_o4", "o5_o6", "o7_o9"):
        result[group] = summarize(groups == group)
    return result


def _grouped_scalar_summary(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    absolute: bool = False,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups)
    if len(values) != len(groups):
        raise ValueError("values and groups must contain the same number of samples")

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        selected = values[mask]
        if absolute:
            selected = np.abs(selected)
        if not len(selected):
            return {"samples": 0, "mean": None, "median": None, "p95": None}
        return {
            "samples": int(len(selected)),
            "mean": float(np.mean(selected)),
            "median": float(np.median(selected)),
            "p95": float(np.quantile(selected, .95)),
        }

    result = {"all": summarize(np.ones(len(groups), dtype=bool))}
    for group in ("o0_o4", "o5_o6", "o7_o9"):
        result[group] = summarize(groups == group)
    return result


def critic_values(
    agent: ThesisSACAgent,
    observations: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    observations_tensor = torch.as_tensor(observations, dtype=torch.float32)
    actions_tensor = torch.as_tensor(actions, dtype=torch.float32)
    q1_parts: list[np.ndarray] = []
    q2_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(observations_tensor), batch_size):
            obs = observations_tensor[start:start + batch_size]
            action = actions_tensor[start:start + batch_size]
            q1_parts.append(agent.q1(obs, action).cpu().numpy().reshape(-1))
            q2_parts.append(agent.q2(obs, action).cpu().numpy().reshape(-1))
    q1 = np.concatenate(q1_parts)
    q2 = np.concatenate(q2_parts)
    return {"q1": q1, "q2": q2, "min_q": np.minimum(q1, q2)}


def twin_q_disagreement(
    q1: np.ndarray,
    q2: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    q1 = np.asarray(q1, dtype=np.float64).reshape(-1)
    q2 = np.asarray(q2, dtype=np.float64).reshape(-1)
    disagreement = np.abs(q1 - q2)
    scale = .5 * (np.abs(q1) + np.abs(q2))
    result = _grouped_scalar_summary(disagreement, groups)
    for name, mask in {
        "all": np.ones(len(groups), dtype=bool),
        "o0_o4": groups == "o0_o4",
        "o5_o6": groups == "o5_o6",
        "o7_o9": groups == "o7_o9",
    }.items():
        if not np.any(mask):
            result[name]["relative_to_mean_abs_q"] = None
        else:
            result[name]["relative_to_mean_abs_q"] = float(
                np.mean(disagreement[mask]) / (np.mean(scale[mask]) + 1e-12)
            )
    return result


def q_value_churn(
    previous: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    groups: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("q1", "q2", "min_q"):
        left = np.asarray(previous[key], dtype=np.float64).reshape(-1)
        right = np.asarray(current[key], dtype=np.float64).reshape(-1)
        if left.shape != right.shape:
            raise ValueError(f"Q-value shape changed for {key}")
        absolute_delta = np.abs(right - left)
        summary = _grouped_scalar_summary(absolute_delta, groups)
        for name, mask in {
            "all": np.ones(len(groups), dtype=bool),
            "o0_o4": groups == "o0_o4",
            "o5_o6": groups == "o5_o6",
            "o7_o9": groups == "o7_o9",
        }.items():
            if not np.any(mask):
                summary[name]["relative_to_previous_mean_abs_q"] = None
            else:
                summary[name]["relative_to_previous_mean_abs_q"] = float(
                    np.mean(absolute_delta[mask])
                    / (np.mean(np.abs(left[mask])) + 1e-12)
                )
        result[key] = summary
    return result


def policy_update_direction(
    older_actions: np.ndarray,
    previous_actions: np.ndarray,
    current_actions: np.ndarray,
    groups: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    previous_delta = np.asarray(previous_actions) - np.asarray(older_actions)
    current_delta = np.asarray(current_actions) - np.asarray(previous_actions)
    previous_norm = np.linalg.norm(previous_delta, axis=1)
    current_norm = np.linalg.norm(current_delta, axis=1)
    valid = (previous_norm > epsilon) & (current_norm > epsilon)
    cosine = np.full(len(groups), np.nan, dtype=np.float64)
    cosine[valid] = np.sum(
        previous_delta[valid] * current_delta[valid], axis=1
    ) / (previous_norm[valid] * current_norm[valid])
    cosine[valid] = np.clip(cosine[valid], -1.0, 1.0)

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        requested = int(np.sum(mask))
        selected = cosine[mask & valid]
        if not len(selected):
            return {
                "samples": requested,
                "valid_samples": 0,
                "mean_cosine": None,
                "negative_fraction": None,
                "strong_reversal_fraction": None,
            }
        return {
            "samples": requested,
            "valid_samples": int(len(selected)),
            "mean_cosine": float(np.mean(selected)),
            "median_cosine": float(np.median(selected)),
            "p05_cosine": float(np.quantile(selected, .05)),
            "negative_fraction": float(np.mean(selected < 0.0)),
            "strong_reversal_fraction": float(np.mean(selected <= -.5)),
            "aligned_fraction": float(np.mean(selected >= .5)),
        }

    result = {"all": summarize(np.ones(len(groups), dtype=bool))}
    for group in ("o0_o4", "o5_o6", "o7_o9"):
        result[group] = summarize(groups == group)
    return result


def _evaluation(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    values = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "success_rate", "success_rate_o0_o4", "success_rate_o5_o6",
        "success_rate_o7_o9", "timeout_rate", "timeout_rate_o7_o9",
        "large_angle_orientation_improvement_ratio",
        "large_angle_orientation_mean_rebound_rad",
        "mean_orientation_error_rad",
    )
    return {key: values.get(key) for key in keys}


def _assessment(checkpoints: list[dict[str, Any]], transitions: list[dict[str, Any]]) -> dict[str, Any]:
    if not transitions:
        return {
            "classification": "single_checkpoint_capacity_snapshot_only",
            "note": "至少两个 checkpoint 才能判断 churn 和参数更新趋势。",
        }
    latest = transitions[-1]
    churn = latest["policy_churn"]["all"]["mean_l2"]
    hard_churn = latest["policy_churn"]["o7_o9"]["mean_l2"]
    actor_update = latest["parameter_update_ratio"]["actor"]
    previous_actor = checkpoints[-2]["dormant_and_rank"]["actor"]
    current_actor = checkpoints[-1]["dormant_and_rank"]["actor"]
    previous_layers = [
        value for key, value in previous_actor.items() if key.startswith("layer_")
    ]
    current_layers = [
        value for key, value in current_actor.items() if key.startswith("layer_")
    ]
    previous_dormant = float(np.mean([
        layer["ratios"]["0.1"] for layer in previous_layers
    ]))
    current_dormant = float(np.mean([
        layer["ratios"]["0.1"] for layer in current_layers
    ]))
    dormant_delta = current_dormant - previous_dormant
    previous_rank = float(previous_actor["last_layer_rank"]["effective_rank"])
    current_rank = float(current_actor["last_layer_rank"]["effective_rank"])
    rank_delta = current_rank - previous_rank
    current_eval = checkpoints[-1].get("evaluation")
    previous_eval = checkpoints[-2].get("evaluation")
    result: dict[str, Any] = {
        "classification": "learning_activity_without_performance_context",
        "latest_actor_churn_mean_l2": churn,
        "latest_hard_region_churn_mean_l2": hard_churn,
        "latest_actor_parameter_update_ratio": actor_update,
        "actor_mean_dormant_ratio_tau_0.1": current_dormant,
        "actor_mean_dormant_ratio_delta": dormant_delta,
        "actor_last_layer_effective_rank": current_rank,
        "actor_last_layer_effective_rank_delta": rank_delta,
        "note": "阈值仅用于诊断分型，不是理论上的 plasticity 常数。",
    }
    if "q_value_churn" in latest:
        result["latest_min_q_churn_mean_abs"] = latest[
            "q_value_churn"
        ]["min_q"]["all"]["mean"]
        result["latest_hard_min_q_churn_mean_abs"] = latest[
            "q_value_churn"
        ]["min_q"]["o7_o9"]["mean"]
    if "policy_update_direction" in latest:
        direction = latest["policy_update_direction"]
        result["latest_policy_update_negative_cosine_fraction"] = direction[
            "all"
        ]["negative_fraction"]
        result["latest_hard_policy_update_negative_cosine_fraction"] = direction[
            "o7_o9"
        ]["negative_fraction"]
    if current_eval is None or previous_eval is None:
        return result
    current_success = current_eval.get("success_rate")
    previous_success = previous_eval.get("success_rate")
    current_hard = current_eval.get("success_rate_o7_o9")
    previous_hard = previous_eval.get("success_rate_o7_o9")
    if any(value is None for value in (
        current_success, previous_success, current_hard, previous_hard,
    )):
        result["note"] = "评估 JSON 缺少 overall 或 O7-O9 success，无法进行平台分型。"
        return result
    success_delta = float(current_success) - float(previous_success)
    hard_delta = float(current_hard) - float(previous_hard)
    result.update({"success_delta": success_delta, "hard_success_delta": hard_delta})
    plateau = abs(success_delta) < .01 and abs(hard_delta) < .01
    if not plateau and (success_delta > 0.0 or hard_delta > 0.0):
        result["classification"] = "performance_still_improving"
    elif plateau and churn is not None and churn >= .02 and actor_update >= 1e-4:
        result["classification"] = "active_but_not_improving_high_churn"
    elif plateau and (churn or 0.0) < .005 and actor_update < 1e-5:
        if dormant_delta >= .02 or rank_delta <= -2.0:
            result["classification"] = "possible_plasticity_loss"
        else:
            result["classification"] = "stable_plateau_low_update_activity"
    else:
        result["classification"] = "mixed_or_inconclusive_plateau_signals"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline dormant-neuron, churn, rank and gradient diagnostics"
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument(
        "--evaluations", nargs="*",
        help="optional frozen-evaluation JSON files in checkpoint order",
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--gradient-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=71001)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.samples, args.gradient_samples, args.batch_size) < 1:
        parser.error("sample counts and batch size must be positive")
    if args.evaluations is not None and len(args.evaluations) not in {
        0, len(args.checkpoints)
    }:
        parser.error("--evaluations must contain one path per checkpoint")

    torch.set_num_threads(1)
    checkpoint_paths = [_checkpoint_path(value) for value in args.checkpoints]
    evaluation_paths: list[Path | None] = [None] * len(checkpoint_paths)
    if args.evaluations:
        evaluation_paths = [_checkpoint_path(value) for value in args.evaluations]

    reference = _load_checkpoint(checkpoint_paths[-1])
    probe = _probe_from_replay(reference, args.samples, args.seed)
    reference_agent = _make_agent(reference)
    reference_actions = reference_agent.select_actions(
        probe["observations"], deterministic=True
    )
    reward_parameters = reference["replay"].get("reward_parameters", {})
    reward_gamma = float(reference["replay"].get(
        "reward_gamma", reference["config"]["sac"]["gamma"]
    ))
    self_safety = reference["curriculum"]["self_safety"]
    lambda_self = float(self_safety.weight)
    rewards, _ = _vectorized_replay_rewards(
        probe["raw"], probe["dones"], np.ones(len(probe["raw"])),
        lambda_self, reward_gamma, reward_parameters,
    )
    reference_agent_state = copy.deepcopy(reference["agent"])
    reference_config = copy.deepcopy(reference["config"])
    reference_protocol = reference.get("protocol")
    reference_counters = copy.deepcopy(reference["counters"])
    del reference_agent, reference
    gc.collect()

    thresholds = (.01, .05, .10)
    churn_thresholds = (.10, .20)
    reports: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    previous_actions: np.ndarray | None = None
    older_actions: np.ndarray | None = None
    previous_q_values: dict[str, np.ndarray] | None = None
    previous_parameters: dict[str, dict[str, torch.Tensor]] | None = None

    for index, (path, evaluation_path) in enumerate(
        zip(checkpoint_paths, evaluation_paths)
    ):
        if index == len(checkpoint_paths) - 1:
            state = {
                "agent": reference_agent_state,
                "config": reference_config,
                "protocol": reference_protocol,
                "counters": reference_counters,
            }
        else:
            state = _load_checkpoint(path)
        if state.get("protocol") != reference_protocol:
            raise ValueError("all checkpoints must use the same protocol")
        agent = _make_agent(state)
        if int(agent.actor.backbone[0].in_features) != probe["observations"].shape[1]:
            raise ValueError("checkpoint observation dimensions do not match probe states")
        actions = agent.select_actions(probe["observations"], deterministic=True)
        q_values = critic_values(
            agent, probe["observations"], reference_actions, args.batch_size,
        )
        activity = network_activity(
            agent, probe["observations"], reference_actions,
            thresholds, args.batch_size,
        )
        gradients = gradient_activity(
            agent, probe, rewards, args.gradient_samples, args.seed,
        )
        parameters = {
            name: copy.deepcopy(module.state_dict())
            for name, module in (("actor", agent.actor), ("q1", agent.q1), ("q2", agent.q2))
        }
        report = {
            "checkpoint": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "global_step": int(state["counters"]["global_step"]),
            "stage_total_step": int(state["counters"].get(
                "stage_total_step", state["counters"]["global_step"]
            )),
            "dormant_and_rank": activity,
            "gradient_activity": gradients,
            "twin_q_disagreement": twin_q_disagreement(
                q_values["q1"], q_values["q2"], probe["groups"],
            ),
            "parameter_norms": parameter_norms(agent),
            "evaluation": _evaluation(evaluation_path),
        }
        reports.append(report)
        if previous_actions is not None and previous_parameters is not None:
            transitions.append({
                "from_checkpoint": reports[-2]["checkpoint"],
                "to_checkpoint": report["checkpoint"],
                "global_step_delta": report["global_step"] - reports[-2]["global_step"],
                "policy_churn": churn_metrics(
                    previous_actions, actions, probe["groups"], churn_thresholds,
                ),
                "q_value_churn": q_value_churn(
                    previous_q_values, q_values, probe["groups"],
                ),
                "parameter_update_ratio": {
                    name: relative_parameter_update(
                        previous_parameters[name], parameters[name]
                    )
                    for name in ("actor", "q1", "q2")
                },
            })
            if older_actions is not None:
                transitions[-1]["policy_update_direction"] = policy_update_direction(
                    older_actions, previous_actions, actions, probe["groups"],
                )
        older_actions = previous_actions
        previous_actions = actions
        previous_q_values = q_values
        previous_parameters = parameters
        del agent, state
        gc.collect()

    group_counts = {
        group: int(np.sum(probe["groups"] == group))
        for group in ("o0_o4", "o5_o6", "o7_o9")
    }
    result = {
        "protocol": reference_protocol,
        "probe": {
            "source_checkpoint": str(checkpoint_paths[-1]),
            "source_partition": "s0_current",
            "samples": int(len(probe["observations"])),
            "seed": args.seed,
            "orientation_group_counts": group_counts,
            "dormant_thresholds": thresholds,
            "churn_l2_thresholds": churn_thresholds,
            "critic_actions": "fixed deterministic actions from source checkpoint",
            "q_diagnostics": (
                "online Q1/Q2 on the same fixed states and source-checkpoint actions"
            ),
            "policy_update_direction": (
                "cosine between consecutive deterministic-action deltas; requires "
                "three chronologically ordered checkpoints"
            ),
            "gradient_batch": "fixed transitions and relabelled rewards from source checkpoint",
        },
        "checkpoints": reports,
        "transitions": transitions,
        "assessment": _assessment(reports, transitions),
    }
    output = _checkpoint_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["assessment"], ensure_ascii=False))


if __name__ == "__main__":
    main()
