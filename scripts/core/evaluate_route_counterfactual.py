#!/usr/bin/env python3
"""Paired counterfactual route-choice evaluation for a thesis PPO policy."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from rl_risk_sac.algorithms.thesis_ppo import ThesisPPOAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.utils.config import deep_update, load_config


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("low_wide", "tall_narrow", "offset_left", "offset_right")


def metric(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def new_stat(info: dict) -> dict[str, object]:
    return {
        "return": 0.0,
        "length": 0,
        "collision": False,
        "obstacle_collision": False,
        "self_collision": False,
        "environment_collision": False,
        "success": False,
        "best_position_m": float(info["goal_error_norm"]),
        "best_orientation_rad": float(info["orientation_error_norm"]),
        "position_at_best_reference_m": float(info["goal_error_norm"]),
        "orientation_at_best_reference_rad": float(info["orientation_error_norm"]),
        "best_reference_error": float("inf"),
        "final_position_m": float(info["goal_error_norm"]),
        "final_orientation_rad": float(info["orientation_error_norm"]),
        "tcp_path_length_m": 0.0,
        "direct_tcp_distance_m": 0.0,
        "tcp_path_efficiency_ratio": 0.0,
        "route_lateral_min_m": 0.0,
        "route_lateral_max_m": 0.0,
        "route_vertical_deviation_min_m": 0.0,
        "route_vertical_deviation_max_m": 0.0,
        "selected_avoidance_route": "none",
        "crossing_route": "none",
        "crossing_lateral_m": 0.0,
        "crossing_vertical_m": 0.0,
        "crossing_progress_error_m": float("inf"),
        "episode_start_ee_position": np.asarray(info["ee_position"]).tolist(),
        "obstacle_position": np.asarray(info["obstacle_positions"])[0].tolist(),
        "obstacle_half_extents": np.asarray(info["obstacle_half_extents"])[0].tolist(),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        route_counts = {
            route: sum(row["crossing_route"] == route for row in selected)
            for route in ("left", "right", "over", "under")
        }
        result[variant] = {
            "episodes": len(selected),
            "collision_rate": float(np.mean([row["collision"] for row in selected])),
            "obstacle_collision_rate": float(np.mean([
                row["obstacle_collision"] for row in selected
            ])),
            "safe_pose_5mm_0.05rad_rate": float(np.mean([
                row["safe_pose_5mm_0.05rad"] for row in selected
            ])),
            "path_efficiency": metric([
                float(row["tcp_path_efficiency_ratio"]) for row in selected
            ]),
            "route_counts": route_counts,
            "route_rates": {
                route: count / len(selected) for route, count in route_counts.items()
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=51001)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--residual-cutoff-position-m", type=float, default=0.025,
        help="Deployment handoff threshold; use 0 to keep full Residual authority.",
    )
    args = parser.parse_args()
    if args.pairs < 1 or args.residual_cutoff_position_m < 0.0:
        parser.error("pairs must be positive and cutoff must be non-negative")

    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "config" not in payload:
        raise ValueError("checkpoint must contain its training config")
    config = load_config(ROOT / "configs/experiments/paper_full.yaml")
    deep_update(config, deepcopy(payload["config"]))
    if args.device:
        config["device"] = args.device
    config["seed"] = args.seed
    config["thesis"]["stage"] = "s1"
    config["thesis"]["simple_reaching_mode"] = False
    config["thesis"]["blocking_obstacle_mode"] = True
    # Counterfactual construction requires exactly one static obstacle.
    config["thesis"]["static_obstacle_count_probabilities"] = {1: 1.0}

    from rl_risk_sac.envs.thesis_homotopy_env_reduced_observation import (
        ThesisHomotopyEnv,
    )

    source = ThesisHomotopyEnv(config)
    obs_dim = int(source.observation_space.shape[0])
    action_dim = int(source.action_space.shape[0])
    agent = ThesisPPOAgent(obs_dim, action_dim, config)
    agent.load_state_dict(payload["agent"] if "agent" in payload else payload)

    info_keys = (
        "collision", "obstacle_collision", "self_collision",
        "environment_collision", "success", "reference_error",
        "next_rho_position", "next_rho_orientation", "goal_error_norm",
        "d_min", "ee_position", "tcp_path_length_m", "direct_tcp_distance_m",
        "tcp_path_efficiency_ratio", "route_lateral_min_m",
        "route_lateral_max_m", "route_vertical_deviation_min_m",
        "route_vertical_deviation_max_m", "selected_avoidance_route",
    )
    rows: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    rejected_seeds: list[dict[str, object]] = []
    candidate_index = 0
    started = perf_counter()
    worker_seeds = [args.seed + 900_000 + index for index in range(4)]
    try:
        with ParallelThesisEnvPool(
            config, worker_seeds, step_info_keys=info_keys
        ) as pool:
            while len(rows) // 4 < args.pairs:
                pair_id = len(rows) // 4
                pair_seed = args.seed + candidate_index
                candidate_index += 1
                source.configure_episode("none", terminate_on_success=False)
                _source_observation, goal_info = source.reset(seed=pair_seed)
                common = {
                    "forced_goal_position": goal_info["goal_position"],
                    "forced_goal_quaternion": goal_info["goal_quaternion"],
                    "forced_goal_joint_positions": goal_info["goal_joint_positions"],
                }
                requests = {
                    worker: {
                        "scene": "static",
                        "terminate_on_success": False,
                        "counterfactual_variant": variant,
                        **common,
                    }
                    for worker, variant in enumerate(VARIANTS)
                }
                resets = pool.try_reset_many(
                    requests, seeds={worker: pair_seed for worker in requests}
                )
                if not all(result[0] for result in resets.values()):
                    rejected_seeds.append({
                        "seed": pair_seed,
                        "errors": {
                            VARIANTS[worker]: result[3]
                            for worker, result in resets.items() if not result[0]
                        },
                    })
                    continue
                observations = np.stack([
                    np.asarray(resets[worker][1], dtype=np.float32)
                    for worker in range(4)
                ])
                initial_infos = [resets[worker][2] for worker in range(4)]
                goal_positions = np.stack([
                    np.asarray(info["goal_position"]) for info in initial_infos
                ])
                goal_joints = np.stack([
                    np.asarray(info["goal_joint_positions"]) for info in initial_infos
                ])
                if not (
                    np.allclose(goal_positions, goal_positions[0], atol=1e-8)
                    and np.allclose(goal_joints, goal_joints[0], atol=1e-8)
                ):
                    raise RuntimeError("counterfactual quartet does not share one goal")
                stats = [new_stat(info) for info in initial_infos]
                position_errors = np.asarray([
                    info["goal_error_norm"] for info in initial_infos
                ], dtype=np.float32)
                done = np.zeros(4, dtype=bool)
                while not bool(np.all(done)):
                    base_actions = agent.predict_base_actions(observations)
                    contexts = pool.residual_safety_context_many({
                        worker: base_actions[worker]
                        for worker in range(4) if not done[worker]
                    })
                    active = np.flatnonzero(~done)
                    context = np.stack([
                        contexts[int(worker)] for worker in active
                    ]).astype(np.float32)
                    cutoff = (
                        None if args.residual_cutoff_position_m == 0.0
                        else args.residual_cutoff_position_m
                    )
                    actions, _logp, _values, policy = agent.act_for_deployment(
                        observations[active], position_errors[active],
                        residual_cutoff_position_m=cutoff,
                        residual_safety_context=context,
                    )
                    results = pool.step_many(
                        {
                            int(worker): actions[index]
                            for index, worker in enumerate(active)
                        },
                        base_clearance_changes={
                            int(worker): context[index]
                            for index, worker in enumerate(active)
                        },
                    )
                    for worker, result in results.items():
                        active_index = int(np.flatnonzero(active == worker)[0])
                        observation, reward, _cost, terminated, truncated, info = result
                        observations[worker] = observation
                        position_errors[worker] = float(info["goal_error_norm"])
                        stat = stats[worker]
                        stat["return"] = float(stat["return"]) + float(reward)
                        stat["length"] = int(stat["length"]) + 1
                        for key in (
                            "collision", "obstacle_collision", "self_collision",
                            "environment_collision", "success",
                        ):
                            stat[key] = bool(stat[key]) or bool(info[key])
                        position = float(info["next_rho_position"])
                        orientation = float(info["next_rho_orientation"])
                        reference = float(info["reference_error"])
                        stat["best_position_m"] = min(float(stat["best_position_m"]), position)
                        stat["best_orientation_rad"] = min(
                            float(stat["best_orientation_rad"]), orientation
                        )
                        if reference < float(stat["best_reference_error"]):
                            stat["best_reference_error"] = reference
                            stat["position_at_best_reference_m"] = position
                            stat["orientation_at_best_reference_rad"] = orientation
                        stat["final_position_m"] = position
                        stat["final_orientation_rad"] = orientation
                        for key in (
                            "tcp_path_length_m", "direct_tcp_distance_m",
                            "tcp_path_efficiency_ratio", "route_lateral_min_m",
                            "route_lateral_max_m", "route_vertical_deviation_min_m",
                            "route_vertical_deviation_max_m",
                        ):
                            stat[key] = float(info[key])
                        stat["selected_avoidance_route"] = str(
                            info["selected_avoidance_route"]
                        )
                        start = np.asarray(
                            stat["episode_start_ee_position"], dtype=np.float64
                        )
                        goal = goal_positions[0].astype(np.float64)
                        current = np.asarray(info["ee_position"], dtype=np.float64)
                        obstacle = np.asarray(stat["obstacle_position"], dtype=np.float64)
                        horizontal = goal[:2] - start[:2]
                        horizontal_norm = float(np.linalg.norm(horizontal))
                        along = (
                            horizontal / horizontal_norm
                            if horizontal_norm > 1.0e-8
                            else np.asarray([1.0, 0.0])
                        )
                        normal = np.asarray([-along[1], along[0]])
                        obstacle_progress = float(np.dot(obstacle[:2] - start[:2], along))
                        current_progress = float(np.dot(current[:2] - start[:2], along))
                        progress_error = abs(current_progress - obstacle_progress)
                        lateral = float(np.dot(current[:2] - start[:2], normal))
                        fraction = float(np.clip(
                            current_progress / max(horizontal_norm, 1.0e-8), 0.0, 1.0
                        ))
                        vertical = float(
                            current[2] - (start[2] + fraction * (goal[2] - start[2]))
                        )
                        if progress_error < float(stat["crossing_progress_error_m"]):
                            stat["crossing_progress_error_m"] = progress_error
                            stat["crossing_lateral_m"] = lateral
                            stat["crossing_vertical_m"] = vertical
                            crossing_candidates = {
                                "left": lateral,
                                "right": -lateral,
                                "over": vertical,
                                "under": -vertical,
                            }
                            stat["crossing_route"] = max(
                                crossing_candidates, key=crossing_candidates.get
                            )
                        trace = {
                            "pair_id": pair_id,
                            "seed": pair_seed,
                            "variant": VARIANTS[worker],
                            "step": int(stat["length"]),
                            "ee_x": float(current[0]),
                            "ee_y": float(current[1]),
                            "ee_z": float(current[2]),
                            "position_error_m": position,
                            "orientation_error_rad": orientation,
                            "d_min_m": float(info["d_min"]),
                            "lateral_deviation_m": lateral,
                            "vertical_deviation_m": vertical,
                            "obstacle_progress_error_m": progress_error,
                            "collision": bool(info["collision"]),
                        }
                        for action_index in range(6):
                            trace[f"base_action_{action_index}"] = float(
                                policy["base_actions"][active_index, action_index]
                            )
                            trace[f"residual_action_{action_index}"] = float(
                                policy["residual_actions"][active_index, action_index]
                            )
                            trace[f"executed_action_{action_index}"] = float(
                                actions[active_index, action_index]
                            )
                        traces.append(trace)
                        done[worker] = bool(terminated or truncated)
                for worker, variant in enumerate(VARIANTS):
                    stat = stats[worker]
                    pose5 = (
                        float(stat["position_at_best_reference_m"]) <= 0.005
                        and float(stat["orientation_at_best_reference_rad"]) <= 0.05
                    )
                    rows.append({
                        "pair_id": pair_id,
                        "seed": pair_seed,
                        "variant": variant,
                        "goal_x": float(goal_positions[0, 0]),
                        "goal_y": float(goal_positions[0, 1]),
                        "goal_z": float(goal_positions[0, 2]),
                        **stat,
                        "safe_pose_5mm_0.05rad": bool(pose5 and not stat["collision"]),
                    })
                if (pair_id + 1) % 10 == 0:
                    print(
                        f"pairs={pair_id + 1}/{args.pairs} rejected={len(rejected_seeds)}",
                        flush=True,
                    )
    finally:
        source.close()

    paired = {int(row["pair_id"]): {} for row in rows}
    for row in rows:
        paired[int(row["pair_id"])][str(row["variant"])] = row
    low_tall_switch = np.mean([
        pair["low_wide"]["crossing_route"]
        != pair["tall_narrow"]["crossing_route"]
        for pair in paired.values()
    ])
    offset_mirror = np.mean([
        pair["offset_left"]["crossing_route"] == "right"
        and pair["offset_right"]["crossing_route"] == "left"
        for pair in paired.values()
    ])
    shape_expected = np.mean([
        pair["low_wide"]["crossing_route"] == "over"
        and pair["tall_narrow"]["crossing_route"] in {"left", "right"}
        for pair in paired.values()
    ])
    report = {
        "kind": "paired_route_counterfactual_evaluation",
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload.get("global_step", payload.get("step", 0))),
        "pairs": args.pairs,
        "candidate_seeds_tested": candidate_index,
        "rejected_seed_count": len(rejected_seeds),
        "seed_base": args.seed,
        "residual_cutoff_position_m": args.residual_cutoff_position_m,
        "elapsed_s": perf_counter() - started,
        "by_variant": summarize(rows),
        "paired_evidence": {
            "low_wide_vs_tall_narrow_route_switch_rate": float(low_tall_switch),
            "shape_expected_route_pair_rate": float(shape_expected),
            "left_right_offset_mirror_route_rate": float(offset_mirror),
        },
        "rejected_seeds": rejected_seeds,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".csv").exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_rows = []
    for row in rows:
        flat = dict(row)
        flat["obstacle_position"] = json.dumps(flat["obstacle_position"])
        flat["obstacle_half_extents"] = json.dumps(flat["obstacle_half_extents"])
        csv_rows.append(flat)
    with output.with_suffix(".csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    trace_output = output.with_name(output.stem + "_steps.csv")
    with trace_output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"saved summary: {output}", flush=True)
    print(f"saved paired rows: {output.with_suffix('.csv')}", flush=True)
    print(f"saved synchronized trajectories: {trace_output}", flush=True)


if __name__ == "__main__":
    main()
