#!/usr/bin/env python3
"""Diagnose why large orientation errors take many steps to contract.

The evaluator separates three mechanisms: low joint-speed use, poor task-space
alignment/kinematic leverage, and simultaneous position optimization.  It also
derives first-crossing statistics for several thresholds from one rollout.
"""

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
from scripts.analysis.analyze_orientation_crossing import orientation_group
from scripts.core.evaluate_thesis_homotopy import automatic_num_envs, physical_cpu_ids


STEP_INFO_KEYS = (
    "rho_orientation", "next_rho_orientation",
    "rho_position", "next_rho_position",
    "next_keypoint_distance", "orientation_error_vector",
    "ee_angular_velocity", "ee_jacobian",
    "policy_joint_velocity", "commanded_joint_velocity", "joint_velocity",
    "task_reached", "strict_pose_reached", "obstacle_collision",
    "self_collision", "environment_collision", "joint_limit",
)

TRANSITION_METRICS = (
    "orientation_progress_rad_s", "position_progress_m_s",
    "angular_speed_rad_s", "effective_angular_speed_rad_s",
    "angular_alignment", "joint_speed_utilization",
    "command_speed_utilization", "policy_speed_utilization",
    "angular_speed_per_joint_speed", "orientation_capacity_rad_s",
    "effective_capacity_fraction", "jacobian_omega_relative_error",
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
        return {key: None for key in (
            "mean", "median", "p25", "p75", "p95", "min", "max",
        )}
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


def summarize_transitions(rows: list[dict[str, float]]) -> dict[str, Any]:
    count = len(rows)
    result: dict[str, Any] = {
        "transitions": count,
        "positive_orientation_progress_rate": (
            sum(row["orientation_progress_rad_s"] > 0.0 for row in rows) / count
            if count else None
        ),
        "orientation_reversal_rate": (
            sum(row["orientation_progress_rad_s"] < 0.0 for row in rows) / count
            if count else None
        ),
        "positive_position_progress_rate": (
            sum(row["position_progress_m_s"] > 0.0 for row in rows) / count
            if count else None
        ),
        "both_progress_positive_rate": (
            sum(
                row["orientation_progress_rad_s"] > 0.0
                and row["position_progress_m_s"] > 0.0
                for row in rows
            ) / count if count else None
        ),
        "metrics": {
            metric: _describe([row[metric] for row in rows])
            for metric in TRANSITION_METRICS
        },
    }
    if count > 1:
        orientation = np.asarray([
            row["orientation_progress_rad_s"] for row in rows
        ])
        position = np.asarray([row["position_progress_m_s"] for row in rows])
        if np.std(orientation) > 1e-12 and np.std(position) > 1e-12:
            result["position_orientation_progress_correlation"] = float(
                np.corrcoef(orientation, position)[0, 1]
            )
        else:
            result["position_orientation_progress_correlation"] = None
    return result


def summarize_episodes(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"episodes": 0}
    return {
        "episodes": len(records),
        "success_rate": float(np.mean([record["success"] for record in records])),
        "initial_rho_position": _describe([
            record["initial_rho_position"] for record in records
        ]),
        "initial_rho_orientation": _describe([
            record["initial_rho_orientation"] for record in records
        ]),
        "pre_0p6_net_orientation_reduction": _describe([
            record["pre_0p6"]["net_orientation_reduction"] for record in records
            if record["pre_0p6"] is not None
        ]),
        "pre_0p6_path_efficiency": _describe([
            record["pre_0p6"]["orientation_path_efficiency"] for record in records
            if record["pre_0p6"] is not None
        ]),
        "pre_0p6_position_reduction": _describe([
            record["pre_0p6"]["position_reduction"] for record in records
            if record["pre_0p6"] is not None
        ]),
    }


def _transition_row(
    info: dict[str, Any], *, control_dt: float, action_scale: np.ndarray,
) -> dict[str, float]:
    jacobian = np.asarray(info["ee_jacobian"], dtype=np.float64)
    qdot = np.asarray(info["joint_velocity"], dtype=np.float64)
    command = np.asarray(info["commanded_joint_velocity"], dtype=np.float64)
    policy = np.asarray(info["policy_joint_velocity"], dtype=np.float64)
    omega_from_jacobian = jacobian[3:] @ qdot
    omega_measured = np.asarray(info["ee_angular_velocity"], dtype=np.float64)
    error = np.asarray(info["orientation_error_vector"], dtype=np.float64)
    error_norm = float(np.linalg.norm(error))
    error_direction = error / max(error_norm, 1e-12)
    angular_speed = float(np.linalg.norm(omega_from_jacobian))
    effective = float(np.dot(error_direction, omega_from_jacobian))
    capacity = float(np.sum(np.abs(error_direction @ jacobian[3:]) * action_scale))
    qdot_norm = float(np.linalg.norm(qdot))
    return {
        "rho_orientation": float(info["rho_orientation"]),
        "next_rho_orientation": float(info["next_rho_orientation"]),
        "rho_position": float(info["rho_position"]),
        "next_rho_position": float(info["next_rho_position"]),
        "orientation_progress_rad_s": float(
            (info["rho_orientation"] - info["next_rho_orientation"]) / control_dt
        ),
        "position_progress_m_s": float(
            (info["rho_position"] - info["next_rho_position"]) / control_dt
        ),
        "angular_speed_rad_s": angular_speed,
        "effective_angular_speed_rad_s": effective,
        "angular_alignment": effective / max(angular_speed, 1e-12),
        "joint_speed_utilization": float(np.sqrt(np.mean((qdot / action_scale) ** 2))),
        "command_speed_utilization": float(np.sqrt(np.mean((command / action_scale) ** 2))),
        "policy_speed_utilization": float(np.sqrt(np.mean((policy / action_scale) ** 2))),
        "angular_speed_per_joint_speed": angular_speed / max(qdot_norm, 1e-12),
        "orientation_capacity_rad_s": capacity,
        "effective_capacity_fraction": effective / max(capacity, 1e-12),
        "jacobian_omega_relative_error": float(
            np.linalg.norm(omega_from_jacobian - omega_measured)
            / max(float(np.linalg.norm(omega_measured)), 1e-12)
        ),
    }


def evaluate(
    agent: ThesisSACAgent, pool: ParallelThesisEnvPool, config: dict[str, Any],
    *, episodes: int, seed: int, level_index: int, thresholds: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
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
    action_scale = np.full(6, float(config["thesis"]["action_scale"]))
    active: dict[int, dict[str, Any]] = {}
    finished_records: dict[int, dict[str, Any]] = {}
    transition_rows: list[dict[str, float]] = []
    next_episode = 0
    while len(finished_records) < episodes:
        assignments: dict[int, int] = {}
        for worker in range(len(pool)):
            if worker not in active and next_episode < episodes:
                assignments[worker] = next_episode
                next_episode += 1
        if assignments:
            contracts = {}
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
                    "episode": episode, "position_bin": position_bin,
                    "orientation_bin": orientation_bin,
                    "group": orientation_group(orientation_bin), "steps": 0,
                    "crossings": {str(value): None for value in thresholds},
                    "pre_0p6_rows": [], "pre_0p6_abs_delta": 0.0,
                }
            resets = pool.reset_many(
                contracts,
                seeds={worker: seed + episode for worker, episode in assignments.items()},
            )
            for worker, (observation, info) in resets.items():
                state = active[worker]
                state["observation"] = observation
                state["initial_rho_position"] = float(info["goal_error_norm"])
                state["initial_rho_orientation"] = float(info["orientation_error_norm"])
                for threshold in thresholds:
                    if state["initial_rho_orientation"] < threshold:
                        state["crossings"][str(threshold)] = {
                            "step": 0, "rho_position": state["initial_rho_position"],
                        }

        indices = sorted(active)
        observations = np.stack([active[index]["observation"] for index in indices])
        actions = agent.select_actions(observations, deterministic=True)
        results = pool.step_many(dict(zip(indices, actions)))
        done_workers = []
        for worker in indices:
            observation, _, _, terminated, truncated, info = results[worker]
            state = active[worker]
            state["steps"] += 1
            state["observation"] = observation
            row = _transition_row(info, control_dt=control_dt, action_scale=action_scale)
            row.update({
                "episode": float(state["episode"]),
                "initial_orientation_bin": float(state["orientation_bin"]),
                "current_orientation_bin": float(np.clip(
                    np.searchsorted(orientation_edges, row["rho_orientation"], side="right") - 1,
                    0, orientation_bins - 1,
                )),
            })
            transition_rows.append(row)
            if (
                state["orientation_bin"] >= 7
                and row["rho_orientation"] >= .6
            ):
                state["pre_0p6_rows"].append(row)
                state["pre_0p6_abs_delta"] += abs(
                    row["rho_orientation"] - row["next_rho_orientation"]
                )
            for threshold in thresholds:
                key = str(threshold)
                if state["crossings"][key] is None and row["next_rho_orientation"] < threshold:
                    state["crossings"][key] = {
                        "step": state["steps"],
                        "remaining_steps": int(config["thesis"]["horizon"] - state["steps"]),
                        "rho_position": row["next_rho_position"],
                    }
            if not (terminated or truncated):
                continue
            collision = bool(
                info["obstacle_collision"] or info["self_collision"]
                or info["environment_collision"]
            )
            pre_rows = state["pre_0p6_rows"]
            pre_summary = None
            crossing_06 = state["crossings"].get("0.6")
            if state["orientation_bin"] >= 7 and pre_rows:
                final_pre_rho = (
                    .6 if crossing_06 is not None else pre_rows[-1]["next_rho_orientation"]
                )
                net = state["initial_rho_orientation"] - final_pre_rho
                pre_summary = {
                    "steps": len(pre_rows),
                    "net_orientation_reduction": net,
                    "orientation_path_efficiency": (
                        net / max(state["pre_0p6_abs_delta"], 1e-12)
                    ),
                    "position_reduction": (
                        state["initial_rho_position"] - pre_rows[-1]["next_rho_position"]
                    ),
                    "transition_summary": summarize_transitions(pre_rows),
                }
            finished_records[state["episode"]] = {
                "episode": state["episode"], "position_bin": state["position_bin"],
                "orientation_bin": state["orientation_bin"], "group": state["group"],
                "initial_rho_position": state["initial_rho_position"],
                "initial_rho_orientation": state["initial_rho_orientation"],
                "steps": state["steps"], "success": bool(info["task_reached"]),
                "timeout": bool(truncated), "collision": collision,
                "crossings": state["crossings"], "pre_0p6": pre_summary,
                "final_rho_position": row["next_rho_position"],
                "final_rho_orientation": row["next_rho_orientation"],
            }
            done_workers.append(worker)
        for worker in done_workers:
            del active[worker]
    return [finished_records[index] for index in range(episodes)], transition_rows


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
    parser.add_argument("--thresholds-rad", type=float, nargs="+", default=[.4, .6, .8])
    parser.add_argument("--num-envs", type=int, default=0)
    parser.add_argument("--no-cpu-affinity", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.episodes < 100:
        parser.error("--episodes must be at least 100")
    thresholds = tuple(sorted(set(float(value) for value in args.thresholds_rad)))
    if not thresholds or any(value <= 0.0 for value in thresholds):
        parser.error("all thresholds must be positive")
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
        agent = ThesisSACAgent(env.observation_space.shape[0], env.action_space.shape[0], config)
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
        records, transitions = evaluate(
            agent, pool, config, episodes=args.episodes, seed=args.seed,
            level_index=args.level_index, thresholds=thresholds,
        )
    groups = {
        group: summarize_episodes([record for record in records if record["group"] == group])
        for group in ("o0_o4", "o5_o6", "o7_o9")
    }
    crossing_summary = {}
    for threshold in thresholds:
        key = str(threshold)
        crossing_summary[key] = {}
        for group in ("o0_o4", "o5_o6", "o7_o9"):
            subset = [record for record in records if record["group"] == group]
            crossed = [record for record in subset if record["crossings"][key] is not None]
            crossing_summary[key][group] = {
                "episodes": len(subset), "crossing_count": len(crossed),
                "crossing_rate": len(crossed) / len(subset) if subset else None,
                "crossing_step": _describe([
                    record["crossings"][key]["step"] for record in crossed
                ]),
                "rho_position_at_crossing": _describe([
                    record["crossings"][key]["rho_position"] for record in crossed
                ]),
                "success_given_crossing": (
                    float(np.mean([record["success"] for record in crossed]))
                    if crossed else None
                ),
            }
    pre_large = [
        row for row in transitions
        if int(row["initial_orientation_bin"]) >= 7 and row["rho_orientation"] >= .6
    ]
    by_current_bin = {
        f"o{orientation_bin}": summarize_transitions([
            row for row in pre_large
            if int(row["current_orientation_bin"]) == orientation_bin
        ])
        for orientation_bin in range(9, 0, -1)
    }
    result = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config": str(config_path.relative_to(ROOT)), "level_index": args.level_index,
        "episodes": args.episodes, "seed": args.seed, "num_envs": num_envs,
        "control_dt_seconds": float(config["thesis"]["control_dt"]),
        "elapsed_seconds": time.perf_counter() - start,
        "groups": groups, "crossing_threshold_sensitivity": crossing_summary,
        "large_angle_pre_0p6": summarize_transitions(pre_large),
        "large_angle_pre_0p6_by_current_orientation_bin": by_current_bin,
        "episode_records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "groups": groups, "crossing_threshold_sensitivity": crossing_summary,
        "large_angle_pre_0p6": result["large_angle_pre_0p6"],
        "large_angle_pre_0p6_by_current_orientation_bin": by_current_bin,
    }, indent=2))


if __name__ == "__main__":
    main()
