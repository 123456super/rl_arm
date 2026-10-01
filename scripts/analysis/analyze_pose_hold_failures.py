#!/usr/bin/env python3
"""Analyze timeout episodes that enter the pose tolerance but fail its hold."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config
from scripts.core.evaluate_thesis_homotopy import physical_cpu_ids


INFO_KEYS = (
    "next_rho_position",
    "next_rho_orientation",
    "strict_pose_reached",
    "task_reached",
    "timeout",
    "obstacle_collision",
    "self_collision",
    "environment_collision",
    "joint_limit",
    "precision_action_scale",
    "commanded_joint_velocity",
    "joint_velocity",
    "orientation_error_vector",
)


def longest_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def percentile_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def sign_change_count(values: np.ndarray, epsilon: float) -> int:
    signs = np.sign(values[np.abs(values) > epsilon])
    return int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0


def analyze_episode(
    rows: list[dict[str, object]], position_tolerance: float,
    orientation_tolerance: float,
) -> dict[str, object]:
    strict = [bool(row["strict"]) for row in rows]
    first_hit = strict.index(True)
    exits = [index for index in range(first_hit + 1, len(rows)) if strict[index - 1] and not strict[index]]
    first_exit = exits[0] if exits else None

    def exit_cause(index: int | None) -> str:
        if index is None:
            return "no_exit_before_timeout"
        position_out = float(rows[index]["rho_p"]) > position_tolerance
        orientation_out = float(rows[index]["rho_r"]) > orientation_tolerance
        if position_out and orientation_out:
            return "both"
        if position_out:
            return "position"
        if orientation_out:
            return "orientation"
        return "neither"

    near_indices = [
        index for index, row in enumerate(rows)
        if float(row["rho_p"]) <= 2.0 * position_tolerance
        and float(row["rho_r"]) <= 2.0 * orientation_tolerance
    ]
    near = [rows[index] for index in range(near_indices[0], len(rows))] if near_indices else []

    def projected_sign_changes(key: str, epsilon: float) -> int:
        if not near:
            return 0
        vectors = np.stack([np.asarray(row[key], dtype=np.float64) for row in near])
        reference = vectors[0]
        norm = float(np.linalg.norm(reference))
        if norm <= epsilon:
            nonzero = np.flatnonzero(np.linalg.norm(vectors, axis=1) > epsilon)
            if nonzero.size == 0:
                return 0
            reference = vectors[int(nonzero[0])]
            norm = float(np.linalg.norm(reference))
        projections = vectors @ (reference / norm)
        return sign_change_count(projections, epsilon)

    hit_rows = [row for row in rows if bool(row["strict"])]
    exit_rows = [rows[index] for index in exits]
    return {
        "max_consecutive_strict_steps": longest_true_run(strict),
        "first_exit_cause": exit_cause(first_exit),
        "all_exit_causes": [exit_cause(index) for index in exits],
        "strict_hit_scales": [float(row["scale"]) for row in hit_rows],
        "exit_scales": [float(row["scale"]) for row in exit_rows],
        "strict_hit_command_rms": [float(row["command_rms"]) for row in hit_rows],
        "strict_hit_qdot_rms": [float(row["qdot_rms"]) for row in hit_rows],
        "exit_command_rms": [float(row["command_rms"]) for row in exit_rows],
        "exit_qdot_rms": [float(row["qdot_rms"]) for row in exit_rows],
        "near_command_rms": [float(row["command_rms"]) for row in near],
        "near_qdot_rms": [float(row["qdot_rms"]) for row in near],
        "position_projection_sign_changes": projected_sign_changes("position_error", 5e-4),
        "orientation_projection_sign_changes": projected_sign_changes("orientation_error", 5e-3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    parser.add_argument("--level-index", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config_path = config_path if config_path.is_absolute() else ROOT / config_path
    config = load_config(config_path)
    config["device"] = "cpu"
    level = config["thesis"]["joint_pose_curriculum"]["levels"][args.level_index]
    orientation_scale = float(config["thesis"]["orientation_curriculum"]["levels"][args.level_index])
    position_tolerance = float(level["position_tolerance_m"])
    orientation_tolerance = float(level["orientation_tolerance_rad"])

    env = ThesisHomotopyEnv(config)
    try:
        agent = ThesisSACAgent(env.observation_space.shape[0], env.action_space.shape[0], config)
    finally:
        env.close()
    checkpoint = Path(args.checkpoint)
    checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor_state = state["agent"]["actor"] if "agent" in state else state.get("actor", state)
    agent.actor.load_state_dict(actor_state)
    agent.actor.eval()

    base_contract = {
        "scene": "none", "xi": 1.0, "strict": True,
        "goal_scale": float(level["goal_scale"]),
        "orientation_scale": orientation_scale,
        "position_tolerance": position_tolerance,
        "orientation_tolerance": orientation_tolerance,
        "lambda_self": 1.0,
    }
    position_bins = 10
    orientation_bins = 10
    position_edges = np.linspace(level["target_distance_min_m"], level["target_distance_max_m"], 11)
    orientation_edges = np.linspace(level["target_orientation_min_rad"], level["target_orientation_max_rad"], 11)
    cpu_ids = physical_cpu_ids(args.num_envs)
    active: dict[int, dict[str, object]] = {}
    completed: dict[int, dict[str, object]] = {}
    next_episode = 0
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    worker_seeds = [args.seed + 1_000_000 + index for index in range(args.num_envs)]
    with ParallelThesisEnvPool(
        config, worker_seeds, cpu_ids=cpu_ids,
        step_info_keys=INFO_KEYS, start_method=start_method,
    ) as pool:
        while len(completed) < args.episodes:
            assignments = {}
            for worker in range(len(pool)):
                if worker in active or next_episode >= args.episodes:
                    continue
                episode = next_episode
                next_episode += 1
                assignments[worker] = episode
                active[worker] = {"episode": episode, "rows": []}
            if assignments:
                contracts = {}
                for worker, episode in assignments.items():
                    p_bin = (episode // orientation_bins) % position_bins
                    o_bin = episode % orientation_bins
                    contract = dict(base_contract)
                    contract.update({
                        "target_distance_min_m": float(position_edges[p_bin]),
                        "target_distance_max_m": float(position_edges[p_bin + 1]),
                        "target_orientation_min_rad": float(orientation_edges[o_bin]),
                        "target_orientation_max_rad": float(orientation_edges[o_bin + 1]),
                    })
                    contracts[worker] = contract
                resets = pool.reset_many(
                    contracts,
                    seeds={worker: args.seed + episode for worker, episode in assignments.items()},
                )
                for worker, (observation, _) in resets.items():
                    active[worker]["observation"] = observation

            workers = sorted(active)
            observations = np.stack([active[worker]["observation"] for worker in workers])
            actions = agent.select_actions(observations, deterministic=True)
            results = pool.step_many(dict(zip(workers, actions)))
            finished = []
            for worker in workers:
                observation, _, _, terminated, truncated, info = results[worker]
                command = np.asarray(info["commanded_joint_velocity"], dtype=np.float64)
                qdot = np.asarray(info["joint_velocity"], dtype=np.float64)
                active[worker]["rows"].append({
                    "rho_p": float(info["next_rho_position"]),
                    "rho_r": float(info["next_rho_orientation"]),
                    "strict": bool(info["strict_pose_reached"]),
                    "scale": float(info["precision_action_scale"]),
                    "command_rms": float(np.sqrt(np.mean(np.square(command)))),
                    "qdot_rms": float(np.sqrt(np.mean(np.square(qdot)))),
                    "position_error": np.asarray(observation[75:78], dtype=np.float64),
                    "orientation_error": np.asarray(info["orientation_error_vector"], dtype=np.float64),
                })
                active[worker]["observation"] = observation
                if terminated or truncated:
                    completed[int(active[worker]["episode"])] = {
                        "success": bool(info["task_reached"]),
                        "timeout": bool(truncated),
                        "rows": active[worker]["rows"],
                    }
                    finished.append(worker)
            for worker in finished:
                del active[worker]

    selected = []
    failures = 0
    never_hit = 0
    for episode in range(args.episodes):
        result = completed[episode]
        if result["success"]:
            continue
        failures += 1
        rows = result["rows"]
        if not any(bool(row["strict"]) for row in rows):
            never_hit += 1
            continue
        analysis = analyze_episode(rows, position_tolerance, orientation_tolerance)
        analysis["episode"] = episode
        selected.append(analysis)

    run_counts = {
        str(length): sum(row["max_consecutive_strict_steps"] == length for row in selected)
        for length in range(1, 5)
    }
    first_causes = {
        cause: sum(row["first_exit_cause"] == cause for row in selected)
        for cause in ("position", "orientation", "both", "no_exit_before_timeout")
    }
    all_causes = {cause: 0 for cause in ("position", "orientation", "both", "neither")}
    for row in selected:
        for cause in row["all_exit_causes"]:
            all_causes[cause] += 1

    def flatten(key: str) -> list[float]:
        return [value for row in selected for value in row[key]]

    position_flips = [int(row["position_projection_sign_changes"]) for row in selected]
    orientation_flips = [int(row["orientation_projection_sign_changes"]) for row in selected]
    output = {
        "checkpoint": str(checkpoint),
        "level_index": args.level_index,
        "episodes": args.episodes,
        "seed": args.seed,
        "position_tolerance_m": position_tolerance,
        "orientation_tolerance_rad": orientation_tolerance,
        "failure_episodes": failures,
        "failure_episodes_ever_strict": len(selected),
        "failure_episodes_never_strict": never_hit,
        "max_consecutive_strict_steps": run_counts,
        "first_exit_cause": first_causes,
        "all_exit_events": all_causes,
        "precision_action_scale_at_strict_hit": percentile_summary(flatten("strict_hit_scales")),
        "precision_action_scale_at_exit": percentile_summary(flatten("exit_scales")),
        "command_joint_velocity_rms_rad_s_at_strict_hit": percentile_summary(flatten("strict_hit_command_rms")),
        "measured_qdot_rms_rad_s_at_strict_hit": percentile_summary(flatten("strict_hit_qdot_rms")),
        "command_joint_velocity_rms_rad_s_at_exit": percentile_summary(flatten("exit_command_rms")),
        "measured_qdot_rms_rad_s_at_exit": percentile_summary(flatten("exit_qdot_rms")),
        "command_joint_velocity_rms_rad_s_near_target": percentile_summary(flatten("near_command_rms")),
        "measured_qdot_rms_rad_s_near_target": percentile_summary(flatten("near_qdot_rms")),
        "near_target_sign_change_definition": {
            "region": "rho_p <= 2*position_tolerance and rho_R <= 2*orientation_tolerance",
            "interval": "from first entry into the region through episode end, including later excursions outside it",
            "direction": "signed projection on the first near-target error-vector direction",
            "position_deadband_m": 5e-4,
            "orientation_deadband_rad": 5e-3,
            "repeated": "at least two sign changes",
        },
        "position_projection_sign_changes": percentile_summary(position_flips),
        "orientation_projection_sign_changes": percentile_summary(orientation_flips),
        "episodes_with_any_position_sign_change": sum(value >= 1 for value in position_flips),
        "episodes_with_repeated_position_sign_change": sum(value >= 2 for value in position_flips),
        "episodes_with_any_orientation_sign_change": sum(value >= 1 for value in orientation_flips),
        "episodes_with_repeated_orientation_sign_change": sum(value >= 2 for value in orientation_flips),
        "episode_details": selected,
    }
    output_path = Path(args.output)
    output_path = output_path if output_path.is_absolute() else ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "episode_details"}, indent=2))


if __name__ == "__main__":
    main()
