#!/usr/bin/env python3
"""Diagnose whether large-angle failures come from delay or bad configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config
from scripts.core.evaluate_thesis_homotopy import (
    automatic_num_envs,
    physical_cpu_ids,
)


STEP_INFO_KEYS = (
    "rho_orientation",
    "next_rho_orientation",
    "next_rho_position",
    "next_keypoint_distance",
    "control_self_min_distance",
    "task_reached",
    "strict_pose_reached",
    "obstacle_collision",
    "self_collision",
    "environment_collision",
    "joint_limit",
)

CROSSING_METRICS = (
    "step",
    "remaining_steps",
    "remaining_time_fraction",
    "rho_position",
    "rho_orientation",
    "keypoint_distance",
    "joint_limit_margin_min_rad",
    "joint_limit_margin_min_fraction",
    "ee_jacobian_sigma_min",
    "ee_jacobian_condition_number",
    "ee_manipulability",
    "ee_log_manipulability",
    "keypoint_jacobian_sigma_min",
    "keypoint_jacobian_condition_number",
    "keypoint_manipulability",
    "keypoint_log_manipulability",
    "self_distance",
    "control_self_min_distance",
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


def _describe(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None, "median": None, "p25": None, "p75": None,
            "p95": None, "min": None, "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, .25)),
        "p75": float(np.quantile(array, .75)),
        "p95": float(np.quantile(array, .95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def orientation_group(orientation_bin: int) -> str:
    if orientation_bin <= 4:
        return "o0_o4"
    if orientation_bin <= 6:
        return "o5_o6"
    return "o7_o9"


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    crossed = [record for record in records if record["crossed"]]
    not_crossed = [record for record in records if not record["crossed"]]
    successful = sum(bool(record["success"]) for record in records)
    successful_after_crossing = sum(bool(record["success"]) for record in crossed)
    result: dict[str, Any] = {
        "episodes": len(records),
        "success_rate": successful / len(records) if records else None,
        "crossing_count": len(crossed),
        "crossing_rate": len(crossed) / len(records) if records else None,
        "no_crossing_count": len(not_crossed),
        "no_crossing_rate": len(not_crossed) / len(records) if records else None,
        "success_rate_given_crossing": (
            successful_after_crossing / len(crossed) if crossed else None
        ),
        "success_rate_without_crossing": (
            sum(bool(record["success"]) for record in not_crossed)
            / len(not_crossed) if not_crossed else None
        ),
        "timeout_rate": (
            sum(bool(record["timeout"]) for record in records) / len(records)
            if records else None
        ),
        "collision_rate": (
            sum(bool(record["collision"]) for record in records) / len(records)
            if records else None
        ),
        "crossing": {},
        "crossing_by_final_outcome": {},
    }
    for metric in CROSSING_METRICS:
        result["crossing"][metric] = _describe([
            float(record["crossing"][metric]) for record in crossed
        ])
    for label, outcome in (("success", True), ("failure", False)):
        subset = [record for record in crossed if bool(record["success"]) is outcome]
        result["crossing_by_final_outcome"][label] = {
            "episodes": len(subset),
            "metrics": {
                metric: _describe([
                    float(record["crossing"][metric]) for record in subset
                ])
                for metric in CROSSING_METRICS
            },
        }
    return result


def nearest_matched_analysis(
    records: list[dict[str, Any]], *, caliper: float,
) -> dict[str, Any]:
    """Match O7--O9 crossings to O0--O4 on rho_p and remaining time."""
    large = [
        record for record in records
        if record["orientation_group"] == "o7_o9" and record["crossed"]
    ]
    controls = [
        record for record in records
        if record["orientation_group"] == "o0_o4" and record["crossed"]
    ]
    if not large or not controls:
        return {
            "match_covariates": ["rho_position", "remaining_time_fraction"],
            "caliper_standardized_l2": caliper,
            "available_large_crossings": len(large),
            "available_control_crossings": len(controls),
            "matched_pairs": 0,
        }
    large_x = np.asarray([
        [record["crossing"]["rho_position"],
         record["crossing"]["remaining_time_fraction"]]
        for record in large
    ], dtype=np.float64)
    control_x = np.asarray([
        [record["crossing"]["rho_position"],
         record["crossing"]["remaining_time_fraction"]]
        for record in controls
    ], dtype=np.float64)
    pooled = np.vstack((large_x, control_x))
    scale = np.std(pooled, axis=0)
    scale = np.where(scale > 1.0e-9, scale, 1.0)
    distances = np.sqrt(np.sum(
        ((large_x[:, None, :] - control_x[None, :, :]) / scale) ** 2,
        axis=2,
    ))
    nearest = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(large)), nearest]
    accepted = np.flatnonzero(nearest_distances <= float(caliper))
    accepted_set = set(int(index) for index in accepted)
    pairs = [
        (large[index], controls[int(nearest[index])], float(nearest_distances[index]))
        for index in accepted
    ]
    result: dict[str, Any] = {
        "match_covariates": ["rho_position", "remaining_time_fraction"],
        "covariate_standard_deviations": {
            "rho_position": float(scale[0]),
            "remaining_time_fraction": float(scale[1]),
        },
        "caliper_standardized_l2": float(caliper),
        "available_large_crossings": len(large),
        "available_control_crossings": len(controls),
        "matched_pairs": len(pairs),
        "unmatched_large_crossings": len(large) - len(pairs),
        "unique_control_episodes": len({pair[1]["episode"] for pair in pairs}),
        "match_distance": _describe([pair[2] for pair in pairs]),
        "unmatched_large_crossings_summary": summarize_group([
            large[index] for index in range(len(large)) if index not in accepted_set
        ]),
    }
    if not pairs:
        return result
    large_success = np.asarray(
        [float(pair[0]["success"]) for pair in pairs], dtype=np.float64,
    )
    control_success = np.asarray(
        [float(pair[1]["success"]) for pair in pairs], dtype=np.float64,
    )
    result.update({
        "large_success_rate": float(np.mean(large_success)),
        "matched_control_success_rate": float(np.mean(control_success)),
        "success_rate_difference_large_minus_control": float(
            np.mean(large_success - control_success)
        ),
        "paired_success": {
            "both_success": int(np.sum((large_success == 1) & (control_success == 1))),
            "large_only": int(np.sum((large_success == 1) & (control_success == 0))),
            "control_only": int(np.sum((large_success == 0) & (control_success == 1))),
            "neither": int(np.sum((large_success == 0) & (control_success == 0))),
        },
        "matched_crossing_metrics": {},
    })
    for metric in CROSSING_METRICS:
        large_values = np.asarray([
            float(pair[0]["crossing"][metric]) for pair in pairs
        ], dtype=np.float64)
        control_values = np.asarray([
            float(pair[1]["crossing"][metric]) for pair in pairs
        ], dtype=np.float64)
        result["matched_crossing_metrics"][metric] = {
            "large_mean": float(np.mean(large_values)),
            "large_median": float(np.median(large_values)),
            "control_mean": float(np.mean(control_values)),
            "control_median": float(np.median(control_values)),
            "large_minus_control_mean": float(np.mean(
                large_values - control_values
            )),
        }
    return result


def infer_mechanism(
    groups: dict[str, dict[str, Any]], matched: dict[str, Any],
) -> dict[str, Any]:
    large = groups["o7_o9"]
    control = groups["o0_o4"]
    crossing_gap = float(control["crossing_rate"] - large["crossing_rate"])
    matched_gap = matched.get("success_rate_difference_large_minus_control")
    evidence = []
    if crossing_gap > .10:
        evidence.append(
            "O7-O9 enters the small-angle region less often, supporting front-end delay/failure."
        )
    if matched_gap is not None and float(matched_gap) >= -.05:
        evidence.append(
            "After matching rho_position and remaining time, downstream success is similar; mechanism A is favored."
        )
        label = "A_front_end_time_cost_favored"
    elif matched_gap is not None and float(matched_gap) < -.10:
        evidence.append(
            "A residual downstream success deficit remains after matching; path dependence/configuration degradation is supported."
        )
        label = "B_path_dependence_supported"
    else:
        evidence.append(
            "Matched downstream evidence is inconclusive at the configured threshold."
        )
        label = "mixed_or_inconclusive"
    return {
        "label": label,
        "crossing_rate_gap_control_minus_large": crossing_gap,
        "matched_success_gap_large_minus_control": matched_gap,
        "evidence": evidence,
        "interpretation_guard": (
            "Nearest-neighbor matching controls only rho_position and remaining time; "
            "it is diagnostic evidence, not a causal estimate."
        ),
    }


def evaluate(
    agent: ThesisSACAgent, pool: ParallelThesisEnvPool, config: dict[str, Any],
    *, episodes: int, seed: int, level_index: int, threshold: float,
    scene: str,
) -> list[dict[str, Any]]:
    levels = config["thesis"]["joint_pose_curriculum"]["levels"]
    level = levels[level_index]
    orientation_levels = config["thesis"]["orientation_curriculum"]["levels"]
    position_bins = int(config["thesis"]["orientation_curriculum"].get(
        "deterministic_probe_position_bins", 10
    ))
    orientation_bins = int(config["thesis"]["orientation_curriculum"].get(
        "deterministic_probe_orientation_bins", 10
    ))
    position_edges = np.linspace(
        float(level["target_distance_min_m"]),
        float(level["target_distance_max_m"]), position_bins + 1,
    )
    orientation_edges = np.linspace(
        float(level["target_orientation_min_rad"]),
        float(level["target_orientation_max_rad"]), orientation_bins + 1,
    )
    base_contract = {
        "scene": scene,
        "xi": 1.0,
        "strict": True,
        "goal_scale": float(level["goal_scale"]),
        "orientation_scale": float(orientation_levels[level_index]),
        "position_tolerance": float(level["position_tolerance_m"]),
        "orientation_tolerance": float(level["orientation_tolerance_rad"]),
        "lambda_self": 1.0,
    }
    active: dict[int, dict[str, Any]] = {}
    records: dict[int, dict[str, Any]] = {}
    next_episode = 0
    while len(records) < episodes:
        assignments: dict[int, int] = {}
        for worker_index in range(len(pool)):
            if worker_index in active or next_episode >= episodes:
                continue
            episode = next_episode
            next_episode += 1
            assignments[worker_index] = episode
        if assignments:
            contracts: dict[int, dict[str, Any]] = {}
            for worker_index, episode in assignments.items():
                position_bin = (episode // orientation_bins) % position_bins
                orientation_bin = episode % orientation_bins
                contract = dict(base_contract)
                contract.update({
                    "target_distance_min_m": float(position_edges[position_bin]),
                    "target_distance_max_m": float(position_edges[position_bin + 1]),
                    "target_orientation_min_rad": float(
                        orientation_edges[orientation_bin]
                    ),
                    "target_orientation_max_rad": float(
                        orientation_edges[orientation_bin + 1]
                    ),
                })
                contracts[worker_index] = contract
                active[worker_index] = {
                    "episode": episode,
                    "position_bin": position_bin,
                    "orientation_bin": orientation_bin,
                    "orientation_group": orientation_group(orientation_bin),
                    "crossed": False,
                    "crossing": None,
                    "strict_pose_hit": False,
                    "steps": 0,
                }
            resets = pool.reset_many(
                contracts,
                seeds={
                    worker_index: seed + episode
                    for worker_index, episode in assignments.items()
                },
            )
            initial_crossers = []
            for worker_index, (observation, info) in resets.items():
                state = active[worker_index]
                state["observation"] = observation
                state["initial_rho_position"] = float(info["goal_error_norm"])
                state["initial_rho_orientation"] = float(
                    info["orientation_error_norm"]
                )
                state["initial_keypoint_distance"] = float(np.mean(
                    np.asarray(info["keypoint_distances"], dtype=np.float64)
                ))
                if state["initial_rho_orientation"] < threshold:
                    initial_crossers.append(worker_index)
            if initial_crossers:
                diagnostics = pool.pose_crossing_diagnostics_many(initial_crossers)
                for worker_index, diagnostic in diagnostics.items():
                    diagnostic["control_self_min_distance"] = diagnostic[
                        "self_distance"
                    ]
                    active[worker_index]["crossed"] = True
                    active[worker_index]["crossing"] = diagnostic

        indices = sorted(active)
        observations = np.stack([
            active[index]["observation"] for index in indices
        ])
        actions = agent.select_actions(observations, deterministic=True)
        results = pool.step_many(dict(zip(indices, actions)))
        crossing_indices = []
        for index in indices:
            observation, _, _, terminated, truncated, info = results[index]
            state = active[index]
            state["steps"] += 1
            state["observation"] = observation
            state["last_info"] = info
            state["terminated"] = bool(terminated)
            state["truncated"] = bool(truncated)
            state["strict_pose_hit"] = bool(
                state["strict_pose_hit"] or info["strict_pose_reached"]
            )
            if not state["crossed"] and float(info["next_rho_orientation"]) < threshold:
                crossing_indices.append(index)
        if crossing_indices:
            diagnostics = pool.pose_crossing_diagnostics_many(crossing_indices)
            for index, diagnostic in diagnostics.items():
                diagnostic["control_self_min_distance"] = float(
                    active[index]["last_info"]["control_self_min_distance"]
                )
                active[index]["crossed"] = True
                active[index]["crossing"] = diagnostic

        finished = []
        for index in indices:
            state = active[index]
            if not (state["terminated"] or state["truncated"]):
                continue
            info = state["last_info"]
            collision = bool(
                info["obstacle_collision"] or info["self_collision"]
                or info["environment_collision"]
            )
            record = {
                "episode": int(state["episode"]),
                "seed": int(seed + state["episode"]),
                "position_bin": int(state["position_bin"]),
                "orientation_bin": int(state["orientation_bin"]),
                "orientation_group": state["orientation_group"],
                "initial_rho_position": state["initial_rho_position"],
                "initial_rho_orientation": state["initial_rho_orientation"],
                "initial_keypoint_distance": state["initial_keypoint_distance"],
                "crossed": bool(state["crossed"]),
                "crossing": state["crossing"],
                "success": bool(info["task_reached"]),
                "strict_pose_hit": bool(state["strict_pose_hit"]),
                "timeout": bool(state["truncated"]),
                "collision": collision,
                "joint_limit": bool(info["joint_limit"]),
                "final_rho_position": float(info["next_rho_position"]),
                "final_rho_orientation": float(info["next_rho_orientation"]),
                "final_keypoint_distance": float(info["next_keypoint_distance"]),
            }
            if state["crossed"]:
                record["post_crossing_steps"] = int(
                    state["steps"] - int(state["crossing"]["step"])
                )
            records[int(state["episode"])] = record
            finished.append(index)
        for index in finished:
            del active[index]
    return [records[index] for index in range(episodes)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Record the first rho_R crossing and compare O7-O9 downstream "
            "success with rho_p/time-matched O0-O4 controls."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml",
    )
    parser.add_argument("--level-index", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=51001)
    parser.add_argument("--threshold-rad", type=float, default=.6)
    parser.add_argument("--match-caliper", type=float, default=.5)
    parser.add_argument("--num-envs", type=int, default=0)
    parser.add_argument("--no-cpu-affinity", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.episodes < 100:
        parser.error("--episodes must be at least 100 for 10x10 bin coverage")
    if args.num_envs < 0:
        parser.error("--num-envs must be non-negative")
    if args.threshold_rad <= 0.0:
        parser.error("--threshold-rad must be positive")
    if args.match_caliper <= 0.0:
        parser.error("--match-caliper must be positive")
    config_path = _path(args.config)
    checkpoint_path = _path(args.checkpoint)
    output_path = _path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    config = load_config(config_path)
    config["device"] = "cpu"
    levels = config["thesis"]["joint_pose_curriculum"]["levels"]
    if not 0 <= args.level_index < len(levels):
        parser.error(f"--level-index must be in [0, {len(levels) - 1}]")
    num_envs = automatic_num_envs(args.episodes) if args.num_envs == 0 else min(
        args.num_envs, args.episodes
    )
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    env = ThesisHomotopyEnv(config)
    try:
        agent = ThesisSACAgent(
            int(env.observation_space.shape[0]),
            int(env.action_space.shape[0]), config,
        )
    finally:
        env.close()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "agent" in state:
        actor_state = state["agent"]["actor"]
    elif isinstance(state, dict) and "actor" in state:
        actor_state = state["actor"]
    else:
        actor_state = state
    agent.actor.load_state_dict(actor_state)
    agent.actor.eval()
    cpu_ids = None
    if num_envs > 1 and not args.no_cpu_affinity:
        cpu_ids = physical_cpu_ids(num_envs)
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    started = time.perf_counter()
    with ParallelThesisEnvPool(
        config,
        [args.seed + 1_000_000 + index for index in range(num_envs)],
        cpu_ids=cpu_ids,
        step_info_keys=STEP_INFO_KEYS,
        start_method=start_method,
    ) as pool:
        records = evaluate(
            agent, pool, config,
            episodes=args.episodes,
            seed=args.seed,
            level_index=args.level_index,
            threshold=args.threshold_rad,
            scene="none",
        )
    groups = {
        group: summarize_group([
            record for record in records if record["orientation_group"] == group
        ])
        for group in ("o0_o4", "o5_o6", "o7_o9")
    }
    matched = nearest_matched_analysis(records, caliper=args.match_caliper)
    matching_sensitivity = {}
    for caliper in (.25, .5, .75, 1.0, 1.5, 2.0):
        sensitivity = nearest_matched_analysis(records, caliper=caliper)
        matching_sensitivity[str(caliper)] = {
            key: sensitivity.get(key)
            for key in (
                "matched_pairs", "unmatched_large_crossings",
                "unique_control_episodes", "large_success_rate",
                "matched_control_success_rate",
                "success_rate_difference_large_minus_control",
            )
        }
    result = {
        "checkpoint": str(
            checkpoint_path.relative_to(ROOT)
            if checkpoint_path.is_relative_to(ROOT) else checkpoint_path
        ),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config": str(
            config_path.relative_to(ROOT)
            if config_path.is_relative_to(ROOT) else config_path
        ),
        "level_index": args.level_index,
        "episodes": args.episodes,
        "seed": args.seed,
        "crossing_threshold_rad": args.threshold_rad,
        "num_envs": num_envs,
        "elapsed_seconds": time.perf_counter() - started,
        "groups": groups,
        "matched_analysis": matched,
        "matching_caliper_sensitivity": matching_sensitivity,
        "mechanism_assessment": infer_mechanism(groups, matched),
        "episode_records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "groups": groups,
        "matched_analysis": matched,
        "mechanism_assessment": result["mechanism_assessment"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
