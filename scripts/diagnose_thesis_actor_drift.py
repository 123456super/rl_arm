#!/usr/bin/env python3
"""Diagnose SAC actor drift on fixed replay states from thesis checkpoints."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.utils.config import load_config


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    return label, path if path.is_absolute() else ROOT / path


def chronological(partition: Any) -> np.ndarray:
    if partition.size < partition.capacity:
        return np.arange(partition.size, dtype=np.int64)
    return np.concatenate((
        np.arange(partition.ptr, partition.capacity, dtype=np.int64),
        np.arange(partition.ptr, dtype=np.int64),
    ))


def select_rows(partition: Any, count: int, seed: int) -> dict[str, np.ndarray]:
    indices = chronological(partition)
    if len(indices) > count:
        indices = np.sort(np.random.default_rng(seed).choice(indices, count, replace=False))
    return {
        "observations": partition.obs[indices].copy(),
        "actions": partition.action[indices].copy(),
        "raw": partition.raw[indices].copy(),
        "episode": partition.episode[indices].copy(),
        "step": partition.step[indices].copy(),
    }


def tensor_stats(values: torch.Tensor) -> dict[str, float]:
    array = values.detach().cpu().numpy().reshape(-1)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def actor_outputs(agent: ThesisSACAgent, observations: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        mean, log_std = agent.actor(observations)
        return {
            "mean": mean,
            "log_std": log_std,
            "std": log_std.exp(),
            "action": torch.tanh(mean),
        }


def gaussian_kl(reference: dict[str, torch.Tensor], current: dict[str, torch.Tensor]) -> torch.Tensor:
    variance_ratio = (reference["std"] / current["std"]).pow(2)
    mean_term = ((current["mean"] - reference["mean"]) / current["std"]).pow(2)
    return (
        current["log_std"] - reference["log_std"]
        + 0.5 * (variance_ratio + mean_term - 1.0)
    ).sum(dim=-1)


def conservative_q_and_gradient(
    agent: ThesisSACAgent, observations: torch.Tensor, actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    differentiable_actions = actions.detach().clone().requires_grad_(True)
    q1 = agent.q1(observations, differentiable_actions)
    q2 = agent.q2(observations, differentiable_actions)
    conservative_q = torch.minimum(q1, q2)
    gradient = torch.autograd.grad(conservative_q.sum(), differentiable_actions)[0]
    return conservative_q.detach().squeeze(-1), (q1 - q2).detach().abs().squeeze(-1), gradient.detach()


def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(left, right, dim=-1, eps=1e-8)


def actor_gradient_components(
    agent: ThesisSACAgent, observations: torch.Tensor, limit: int = 2048,
) -> dict[str, float]:
    observations = observations[:limit]
    parameters = tuple(agent.actor.parameters())
    torch.manual_seed(73001)
    action, log_probability = agent.actor.sample(observations)
    q_objective = -torch.minimum(
        agent.q1(observations, action), agent.q2(observations, action),
    ).mean()
    entropy_objective = (agent.alpha.detach() * log_probability).mean()
    q_gradients = torch.autograd.grad(q_objective, parameters, retain_graph=True)
    entropy_gradients = torch.autograd.grad(entropy_objective, parameters)
    q_vector = torch.cat([gradient.reshape(-1) for gradient in q_gradients])
    entropy_vector = torch.cat([gradient.reshape(-1) for gradient in entropy_gradients])
    q_norm = torch.linalg.vector_norm(q_vector)
    entropy_norm = torch.linalg.vector_norm(entropy_vector)
    return {
        "states": len(observations),
        "q_gradient_l2": float(q_norm.detach().cpu()),
        "entropy_gradient_l2": float(entropy_norm.detach().cpu()),
        "entropy_to_q_gradient_ratio": float(
            (entropy_norm / torch.clamp(q_norm, min=1e-12)).detach().cpu()
        ),
        "gradient_cosine": float(
            torch.nn.functional.cosine_similarity(
                q_vector.unsqueeze(0), entropy_vector.unsqueeze(0), eps=1e-12,
            ).detach().cpu()
        ),
        "per_parameter": {
            str(index): {
                "q_gradient_l2": float(torch.linalg.vector_norm(q_gradient).detach().cpu()),
                "entropy_gradient_l2": float(
                    torch.linalg.vector_norm(entropy_gradient).detach().cpu()
                ),
                "entropy_to_q_gradient_ratio": float(
                    (
                        torch.linalg.vector_norm(entropy_gradient)
                        / torch.clamp(torch.linalg.vector_norm(q_gradient), min=1e-12)
                    ).detach().cpu()
                ),
            }
            for index, (q_gradient, entropy_gradient) in enumerate(
                zip(q_gradients, entropy_gradients, strict=True)
            )
        },
    }


def load_agent(
    checkpoint_path: Path, config: dict[str, Any], observation_dim: int, action_dim: int,
) -> tuple[ThesisSACAgent, dict[str, Any]]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent = ThesisSACAgent(observation_dim, action_dim, config)
    agent.load_state_dict(state["agent"])
    agent.actor.eval(); agent.q1.eval(); agent.q2.eval()
    return agent, state


def parameter_drift(reference: ThesisSACAgent, current: ThesisSACAgent) -> dict[str, Any]:
    rows = {}
    total_delta_sq = 0.0
    total_reference_sq = 0.0
    for (name, old), (_, new) in zip(
        reference.actor.named_parameters(), current.actor.named_parameters(), strict=True,
    ):
        delta = new.detach() - old.detach()
        delta_norm = float(torch.linalg.vector_norm(delta).cpu())
        old_norm = float(torch.linalg.vector_norm(old.detach()).cpu())
        rows[name] = {
            "delta_l2": delta_norm,
            "reference_l2": old_norm,
            "relative_l2": delta_norm / max(old_norm, 1e-12),
        }
        total_delta_sq += delta_norm ** 2
        total_reference_sq += old_norm ** 2
    rows["all"] = {
        "delta_l2": total_delta_sq ** 0.5,
        "reference_l2": total_reference_sq ** 0.5,
        "relative_l2": (total_delta_sq / max(total_reference_sq, 1e-24)) ** 0.5,
    }
    return rows


def analyze_group(
    agents: dict[str, ThesisSACAgent], reference_label: str,
    data: dict[str, np.ndarray],
    position_tolerance: float, orientation_tolerance: float,
) -> dict[str, Any]:
    device = agents[reference_label].device
    observations = torch.as_tensor(data["observations"], dtype=torch.float32, device=device)
    replay_actions = torch.as_tensor(data["actions"], dtype=torch.float32, device=device)
    reference_outputs = actor_outputs(agents[reference_label], observations)
    reference_q, reference_disagreement, reference_gradient = conservative_q_and_gradient(
        agents[reference_label], observations, reference_outputs["action"],
    )
    normalized_error = np.maximum(
        data["raw"][:, 1] / position_tolerance,
        data["raw"][:, 3] / orientation_tolerance,
    )
    phase_masks = {
        "early_step_0_79": data["step"] < 80,
        "middle_step_80_159": (data["step"] >= 80) & (data["step"] < 160),
        "late_step_160_plus": data["step"] >= 160,
        "near_goal_within_2x": normalized_error <= 2.0,
        "outside_2x_goal": normalized_error > 2.0,
    }
    result: dict[str, Any] = {
        "count": len(observations),
        "unique_episodes": int(len(np.unique(data["episode"]))),
        "reference": {
            "mean_std": float(reference_outputs["std"].mean().cpu()),
            "mean_abs_action": float(reference_outputs["action"].abs().mean().cpu()),
            "mean_replay_action_l2": float(
                torch.linalg.vector_norm(reference_outputs["action"] - replay_actions, dim=-1).mean().cpu()
            ),
            "critic_disagreement": tensor_stats(reference_disagreement),
        },
        "comparisons": {},
    }
    for label, agent in agents.items():
        if label == reference_label:
            continue
        current_outputs = actor_outputs(agent, observations)
        action_delta = current_outputs["action"] - reference_outputs["action"]
        action_l2 = torch.linalg.vector_norm(action_delta, dim=-1)
        kl = gaussian_kl(reference_outputs, current_outputs)
        q_reference_action, disagreement_reference_action, gradient = conservative_q_and_gradient(
            agent, observations, reference_outputs["action"],
        )
        q_current_action, disagreement_current_action, _ = conservative_q_and_gradient(
            agent, observations, current_outputs["action"],
        )
        q_advantage = q_current_action - q_reference_action
        uncertainty = torch.maximum(disagreement_reference_action, disagreement_current_action)
        alignment = cosine(action_delta, gradient)
        gradient_rotation = cosine(reference_gradient, gradient)

        torch.manual_seed(73001)
        with torch.no_grad():
            sampled_action, log_probability = agent.actor.sample(observations)
            sampled_q = torch.minimum(
                agent.q1(observations, sampled_action),
                agent.q2(observations, sampled_action),
            ).squeeze(-1)
            entropy_term = agent.alpha.detach() * log_probability.squeeze(-1)

        phase_results = {}
        for phase, numpy_mask in phase_masks.items():
            mask = torch.as_tensor(numpy_mask, device=device)
            if int(mask.sum()) == 0:
                continue
            phase_results[phase] = {
                "count": int(mask.sum().cpu()),
                "action_l2_mean": float(action_l2[mask].mean().cpu()),
                "action_l2_p95": float(torch.quantile(action_l2[mask], 0.95).cpu()),
                "kl_mean": float(kl[mask].mean().cpu()),
                "q_advantage_mean": float(q_advantage[mask].mean().cpu()),
                "critic_disagreement_mean": float(uncertainty[mask].mean().cpu()),
                "gradient_alignment_mean": float(alignment[mask].mean().cpu()),
            }
        result["comparisons"][label] = {
            "alpha": float(agent.alpha.detach().cpu()),
            "action_l2": tensor_stats(action_l2),
            "action_l2_over_1_fraction": float((action_l2 > 1.0).float().mean().cpu()),
            "pre_tanh_reference_to_current_kl": tensor_stats(kl),
            "mean_std": float(current_outputs["std"].mean().cpu()),
            "mean_abs_action": float(current_outputs["action"].abs().mean().cpu()),
            "q_advantage_current_over_reference_action": tensor_stats(q_advantage),
            "q_prefers_current_action_fraction": float((q_advantage > 0.0).float().mean().cpu()),
            "q_advantage_exceeds_disagreement_fraction": float(
                (q_advantage > uncertainty).float().mean().cpu()
            ),
            "critic_disagreement_at_current_action": tensor_stats(disagreement_current_action),
            "actual_drift_vs_current_q_gradient_cosine": tensor_stats(alignment),
            "positive_q_gradient_alignment_fraction": float((alignment > 0.0).float().mean().cpu()),
            "reference_vs_current_q_gradient_cosine": tensor_stats(gradient_rotation),
            "sampled_q_mean": float(sampled_q.mean().cpu()),
            "alpha_log_probability_mean": float(entropy_term.mean().cpu()),
            "entropy_to_q_absolute_ratio": float(
                entropy_term.abs().mean().cpu() / max(sampled_q.abs().mean().cpu(), 1e-12)
            ),
            "actor_gradient_components": actor_gradient_components(agent, observations),
            "phases": phase_results,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--reference", required=True, help="label of the reference checkpoint")
    parser.add_argument("--states-per-group", type=int, default=4096)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoints = dict(args.checkpoint)
    if len(checkpoints) != len(args.checkpoint):
        parser.error("checkpoint labels must be unique")
    if args.reference not in checkpoints:
        parser.error("reference label is not one of the checkpoints")
    if args.states_per_group < 1:
        parser.error("states-per-group must be positive")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = deepcopy(load_config(config_path))
    config["device"] = args.device
    reference_state = torch.load(checkpoints[args.reference], map_location="cpu", weights_only=False)
    replay = reference_state["replay"]
    observation_dim = int(replay["s0_current"].obs.shape[1])
    action_dim = int(replay["s0_current"].action.shape[1])
    agents = {
        label: load_agent(path, config, observation_dim, action_dim)[0]
        for label, path in checkpoints.items()
    }

    previous_scale = max(replay["s0_history"])
    success_episodes = set(int(value) for value in np.unique(
        replay["s0_current_success"].episode[
            chronological(replay["s0_current_success"])
        ]
    ))
    current = replay["s0_current"]
    current_indices = chronological(current)
    nonsuccess_mask = np.asarray([
        int(current.episode[index]) not in success_episodes for index in current_indices
    ])
    nonsuccess_indices = current_indices[nonsuccess_mask]

    class PartitionView:
        pass

    current_nonsuccess = PartitionView()
    current_nonsuccess.size = len(nonsuccess_indices)
    current_nonsuccess.capacity = len(nonsuccess_indices)
    current_nonsuccess.ptr = 0
    for field in ("obs", "action", "raw", "episode", "step"):
        setattr(current_nonsuccess, field, getattr(current, field)[nonsuccess_indices])

    groups = {
        "previous_level_success_snapshot": replay["s0_history"][previous_scale],
        "current_level_protected_success": replay["s0_current_success"],
        "current_level_non_success": current_nonsuccess,
    }
    pose_levels = config["thesis"]["joint_pose_curriculum"]["levels"]
    current_level = int(reference_state["curriculum"]["orientation"].level_index)
    results = {
        "reference": args.reference,
        "checkpoints": {label: str(path) for label, path in checkpoints.items()},
        "reference_level": current_level,
        "previous_replay_scale": previous_scale,
        "states_per_group_limit": args.states_per_group,
        "parameter_drift": {
            label: parameter_drift(agents[args.reference], agent)
            for label, agent in agents.items() if label != args.reference
        },
        "groups": {},
    }
    for group_index, (name, partition) in enumerate(groups.items()):
        level = current_level - 1 if name.startswith("previous") else current_level
        contract = pose_levels[level]
        data = select_rows(partition, args.states_per_group, 91001 + group_index)
        results["groups"][name] = analyze_group(
            agents, args.reference, data,
            float(contract["position_tolerance_m"]),
            float(contract["orientation_tolerance_rad"]),
        )
        print(f"analyzed {name}: {len(data['observations'])} fixed states", flush=True)

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
