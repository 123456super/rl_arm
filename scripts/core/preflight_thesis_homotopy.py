#!/usr/bin/env python3
"""Preflight audits for the task-first safety-homotopy training protocol."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.thesis_reaching import homotopy_reward
from rl_risk_sac.tasks.convex_sweep import ConvexSweep, signed_capsule_distance, swept_risk
from rl_risk_sac.robots.ur5_capsules import CapsuleState
from rl_risk_sac.utils.config import load_config


def discounted(values: list[float], gamma: float) -> float:
    return float(sum((gamma**index) * value for index, value in enumerate(values)))


def reward_audit(config: dict, samples: int) -> tuple[dict, list[dict]]:
    env = ThesisHomotopyEnv(config)
    rows: list[dict] = []
    gamma = float(config["sac"]["gamma"])
    horizon = int(config["thesis"]["horizon"])
    try:
        for index in range(samples):
            env.configure_episode("none", xi=1.0, strict=True)
            _, info = env.reset()
            rho_p = float(info["goal_error_norm"])
            rho_r = float(info["orientation_error_norm"])
            hard_failure, hard_failure_fields = homotopy_reward(
                rho_position=rho_p, next_rho_position=rho_p,
                rho_orientation=rho_r, next_rho_orientation=rho_r,
                smooth_velocity=0.0, velocity_magnitude=0.0, orientation_scale=1.0,
                task_reached=False, hard_failure=True,
                obstacle_collision=False, risk_max=0.0, distance_min=0.80, xi=1.0,
                gamma=gamma, horizon=horizon,
                **env.reward_parameters,
            )
            wait_step, _ = homotopy_reward(
                rho_position=rho_p, next_rho_position=rho_p,
                rho_orientation=rho_r, next_rho_orientation=rho_r,
                smooth_velocity=0.0, velocity_magnitude=0.0, orientation_scale=1.0,
                task_reached=False, hard_failure=False,
                obstacle_collision=False, risk_max=0.0, distance_min=0.80, xi=1.0,
                gamma=gamma, horizon=horizon,
                **env.reward_parameters,
            )
            timeout_step, _ = homotopy_reward(
                rho_position=rho_p, next_rho_position=rho_p,
                rho_orientation=rho_r, next_rho_orientation=rho_r,
                smooth_velocity=0.0, velocity_magnitude=0.0, orientation_scale=1.0,
                task_reached=False, hard_failure=False, timeout=True,
                obstacle_collision=False, risk_max=0.0, distance_min=0.80, xi=1.0,
                gamma=gamma, horizon=horizon,
                **env.reward_parameters,
            )
            timeout = discounted([wait_step] * (horizon - 1) + [timeout_step], gamma)
            def trajectory_return(
                positions: list[float], orientations: list[float], *, success: bool,
            ) -> float:
                values: list[float] = []
                previous_p, previous_r = rho_p, rho_r
                for step, (next_p, next_r) in enumerate(
                    zip(positions, orientations), start=1
                ):
                    final = step == len(positions)
                    reward, _ = homotopy_reward(
                        rho_position=previous_p, next_rho_position=next_p,
                        rho_orientation=previous_r, next_rho_orientation=next_r,
                        smooth_velocity=0.0, velocity_magnitude=0.0,
                        orientation_scale=1.0,
                        task_reached=bool(success and final),
                        hard_failure=False, timeout=bool(not success and final),
                        obstacle_collision=False, risk_max=0.0,
                        distance_min=0.80, xi=1.0, gamma=gamma, horizon=horizon,
                        **env.reward_parameters,
                    )
                    values.append(reward)
                    previous_p, previous_r = next_p, next_r
                return discounted(values, gamma)

            ideal_returns = {}
            for steps in (60, 120, 180, 239, 240):
                fractions = [1.0 - step / steps for step in range(1, steps + 1)]
                ideal_returns[steps] = trajectory_return(
                    [rho_p * value for value in fractions],
                    [rho_r * value for value in fractions], success=True,
                )
            half = horizon // 2
            retreat_fractions = (
                [1.0 - .5 * step / half for step in range(1, half + 1)]
                + [.5 + .5 * step / (horizon - half) for step in range(1, horizon - half + 1)]
            )
            approach_then_retreat = trajectory_return(
                [rho_p * value for value in retreat_fractions],
                [rho_r * value for value in retreat_fractions], success=False,
            )
            oscillation_fractions = [
                .75 if step % 2 else 1.0 for step in range(1, horizon + 1)
            ]
            periodic_oscillation = trajectory_return(
                [rho_p * value for value in oscillation_fractions],
                [rho_r * value for value in oscillation_fractions], success=False,
            )
            ideal_success = ideal_returns[60]
            rows.append({
                "sample": index, "rho_position": rho_p, "rho_orientation": rho_r,
                "hard_failure_now": hard_failure,
                "hard_failure_guard": hard_failure_fields["terminal_guard_penalty"],
                "stationary_timeout": timeout, "ideal_60_step_success": ideal_success,
                **{
                    f"ideal_{steps}_step_success": value
                    for steps, value in ideal_returns.items() if steps != 60
                },
                "approach_then_retreat_timeout": approach_then_retreat,
                "periodic_oscillation_timeout": periodic_oscillation,
                "hard_failure_better_than_timeout": hard_failure > timeout,
                "desired_order": ideal_success > timeout > hard_failure,
                "all_successes_beat_failure_loops": min(ideal_returns.values())
                > max(approach_then_retreat, periodic_oscillation),
            })
    finally:
        env.close()
    summary = {
        "samples": samples,
        "hard_failure_better_than_timeout_rate": float(np.mean([r["hard_failure_better_than_timeout"] for r in rows])),
        "desired_order_rate": float(np.mean([r["desired_order"] for r in rows])),
        "all_successes_beat_failure_loops_rate": float(np.mean([
            r["all_successes_beat_failure_loops"] for r in rows
        ])),
        "median_returns": {
            key: float(np.median([r[key] for r in rows]))
            for key in (
                "hard_failure_now",
                "stationary_timeout", "ideal_60_step_success",
                "ideal_120_step_success", "ideal_180_step_success",
                "ideal_239_step_success", "ideal_240_step_success",
                "approach_then_retreat_timeout", "periodic_oscillation_timeout",
            )
        },
        "median_terminal_guard": {
            "hard_failure": float(np.median([r["hard_failure_guard"] for r in rows])),
        },
    }
    return summary, rows


def _set_q(env: ThesisHomotopyEnv, q: np.ndarray) -> None:
    env._set_joint_state(q)  # The audit intentionally checks the generator's exact joint-space witness.
    p.performCollisionDetection(physicsClientId=env.client_id)


def goal_kinematics_audit(config: dict, samples: int) -> tuple[dict, list[dict]]:
    """Audit the configured task-space pose -> IK -> FK acceptance contract."""
    env = ThesisHomotopyEnv(config)
    levels = config["thesis"].get("joint_pose_curriculum", {}).get("levels", [])
    rows: list[dict] = []
    failures = 0
    try:
        for sample in range(samples):
            try:
                level_index = sample % len(levels) if levels else 0
                level = levels[level_index] if levels else {}
                task_bounds = {
                    key: level[key] for key in (
                        "target_distance_min_m", "target_distance_max_m",
                        "target_orientation_min_rad", "target_orientation_max_rad",
                    ) if key in level
                }
                env.configure_episode(
                    "none", xi=1.0, strict=True,
                    goal_scale=float(level.get("goal_scale", 1.0)),
                    position_tolerance=float(level.get(
                        "position_tolerance_m", config["thesis"]["position_tolerance"]
                    )),
                    orientation_tolerance=float(level.get(
                        "orientation_tolerance_rad", config["thesis"]["orientation_tolerance"]
                    )),
                    **task_bounds,
                )
                _, info = env.reset(seed=int(config["seed"]) + sample)
                validation = info["goal_ik_validation"]
                position_tolerance = float(env.contract.position_tolerance)
                orientation_tolerance = float(env.contract.orientation_tolerance)
                rows.append({
                    "sample": sample,
                    "level": level_index,
                    "accepted": bool(validation["reachable"]),
                    "attempts": int(info["goal_sample_attempts"]),
                    "position_tolerance_m": position_tolerance,
                    "orientation_tolerance_rad": orientation_tolerance,
                    "ik_finite": bool(validation["finite"]),
                    "ik_within_limits": bool(validation["within_limits"]),
                    "position_error_m": float(validation["position_error_m"]),
                    "orientation_error_rad": float(validation["orientation_error_rad"]),
                    "strict_fk_ok": bool(
                        validation["finite"]
                        and validation["within_limits"]
                        and validation["position_error_m"] <= position_tolerance
                        and validation["orientation_error_rad"] <= orientation_tolerance
                    ),
                    "target_distance_m": float(info.get("sampled_target_distance_m", 0.0)),
                    "target_orientation_rad": float(info.get("sampled_target_orientation_rad", 0.0)),
                })
            except RuntimeError:
                failures += 1
                rows.append({
                    "sample": sample, "level": level_index, "accepted": False,
                    "attempts": int(config["thesis"].get("goal_sample_max_attempts", 10000)),
                    "position_tolerance_m": float(level.get(
                        "position_tolerance_m", config["thesis"]["position_tolerance"]
                    )),
                    "orientation_tolerance_rad": float(level.get(
                        "orientation_tolerance_rad", config["thesis"]["orientation_tolerance"]
                    )),
                    "ik_finite": False, "ik_within_limits": False,
                    "position_error_m": float("inf"), "orientation_error_rad": float("inf"),
                    "strict_fk_ok": False,
                    "target_distance_m": float("nan"), "target_orientation_rad": float("nan"),
                })
    finally:
        env.close()
    accepted = [row for row in rows if row["accepted"]]
    summary = {
        "samples": samples,
        "reset_failures": failures,
        "acceptance_rate": float(len(accepted) / samples),
        "thresholds": {
            "by_level": {
                str(index): {
                    "position_error_m": float(level.get(
                        "position_tolerance_m", config["thesis"]["position_tolerance"]
                    )),
                    "orientation_error_rad": float(level.get(
                        "orientation_tolerance_rad", config["thesis"]["orientation_tolerance"]
                    )),
                }
                for index, level in enumerate(levels)
            }
        },
        "attempts": {
            "median": float(np.median([row["attempts"] for row in rows])),
            "p95": float(np.quantile([row["attempts"] for row in rows], 0.95)),
            "max": int(max(row["attempts"] for row in rows)),
        },
        "accepted_error_max": {
            "position_error_m": float(max(row["position_error_m"] for row in accepted)) if accepted else None,
            "orientation_error_rad": float(max(row["orientation_error_rad"] for row in accepted)) if accepted else None,
        },
        "accepted_strict_fk_rate": float(
            np.mean([row["strict_fk_ok"] for row in accepted])
        ) if accepted else 0.0,
    }
    return summary, rows


def task_space_cell_audit(
    config: dict,
    level_index: int = 6,
    seed: int | None = None,
    resets_per_cell: int = 1,
) -> tuple[dict, list[dict]]:
    """Audit every balanced L6 task-space cell under the strict IK contract.

    Each trial fixes one valid initial robot configuration and lets the normal
    sampler spend at most ``goal_sample_max_attempts`` proposals in that cell.
    A failed trial is therefore a cell that had no strict IK/self-clearance
    witness for that audited reset, not a claim about all possible robot
    configurations.
    """
    if resets_per_cell < 1:
        raise ValueError("resets_per_cell must be positive")
    levels = config["thesis"].get("joint_pose_curriculum", {}).get("levels", [])
    if not levels:
        raise ValueError("joint_pose_curriculum.levels is required for cell audit")
    if level_index < 0 or level_index >= len(levels):
        raise ValueError(f"level_index must be in [0, {len(levels) - 1}]")
    level = levels[level_index]
    position_bins = int(config["thesis"]["joint_pose_curriculum"]["task_space_sampling"]["position_bins"])
    orientation_bins = int(config["thesis"]["joint_pose_curriculum"]["task_space_sampling"]["orientation_bins"])
    max_attempts = int(config["thesis"].get("goal_sample_max_attempts", 10000))
    base_seed = int(config["seed"] if seed is None else seed)
    env = ThesisHomotopyEnv(config)
    rows: list[dict] = []
    try:
        env.configure_episode(
            "none", xi=1.0, strict=True,
            goal_scale=float(level["goal_scale"]),
            position_tolerance=float(level["position_tolerance_m"]),
            orientation_tolerance=float(level["orientation_tolerance_rad"]),
            target_distance_min_m=float(level["target_distance_min_m"]),
            target_distance_max_m=float(level["target_distance_max_m"]),
            target_orientation_min_rad=float(level["target_orientation_min_rad"]),
            target_orientation_max_rad=float(level["target_orientation_max_rad"]),
        )
        for position_bin in range(position_bins):
            for orientation_bin in range(orientation_bins):
                flat_bin = position_bin * orientation_bins + orientation_bin
                for trial in range(resets_per_cell):
                    env.rng = np.random.default_rng(
                        base_seed + flat_bin * resets_per_cell + trial
                    )
                    env._selected_task_space_position_bin = position_bin
                    env._selected_task_space_orientation_bin = orientation_bin
                    proposal_before = int(env.goal_proposal_counts[flat_bin])
                    try:
                        env._sample_valid_reset()
                    except RuntimeError as error:
                        rows.append({
                            "level": level_index,
                            "position_bin": position_bin,
                            "orientation_bin": orientation_bin,
                            "trial": trial,
                            "accepted": False,
                            "strict_ik_feasible": False,
                            "failure_stage": "reset",
                            "failure_detail": str(error),
                            "attempts": int(env.goal_proposal_counts[flat_bin] - proposal_before),
                            "max_attempts": max_attempts,
                            "position_error_m": float("inf"),
                            "orientation_error_rad": float("inf"),
                            "target_distance_m": float("nan"),
                            "target_orientation_rad": float("nan"),
                        })
                        continue
                    try:
                        env._sample_goal_pose()
                    except RuntimeError as error:
                        rows.append({
                            "level": level_index,
                            "position_bin": position_bin,
                            "orientation_bin": orientation_bin,
                            "trial": trial,
                            "accepted": False,
                            "strict_ik_feasible": False,
                            "failure_stage": "goal_sampling",
                            "failure_detail": str(error),
                            "attempts": int(env.goal_proposal_counts[flat_bin] - proposal_before),
                            "max_attempts": max_attempts,
                            "position_error_m": float("inf"),
                            "orientation_error_rad": float("inf"),
                            "target_distance_m": float("nan"),
                            "target_orientation_rad": float("nan"),
                        })
                        continue
                    validation = env.goal_ik_validation
                    rows.append({
                        "level": level_index,
                        "position_bin": position_bin,
                        "orientation_bin": orientation_bin,
                        "trial": trial,
                        "accepted": True,
                        "strict_ik_feasible": True,
                        "failure_stage": None,
                        "failure_detail": None,
                        "attempts": int(env.goal_proposal_counts[flat_bin] - proposal_before),
                        "max_attempts": max_attempts,
                        "position_error_m": float(validation["position_error_m"]),
                        "orientation_error_rad": float(validation["orientation_error_rad"]),
                        "target_distance_m": float(env.sampled_target_distance_m),
                        "target_orientation_rad": float(env.sampled_target_orientation_rad),
                    })
    finally:
        env.close()
    accepted = [row for row in rows if row["accepted"]]
    failed = [row for row in rows if not row["accepted"]]
    cell_summary: dict[str, dict] = {}
    for position_bin in range(position_bins):
        for orientation_bin in range(orientation_bins):
            key = f"{position_bin}:{orientation_bin}"
            subset = [
                row for row in rows
                if row["position_bin"] == position_bin
                and row["orientation_bin"] == orientation_bin
            ]
            cell_summary[key] = {
                "position_bin": position_bin,
                "orientation_bin": orientation_bin,
                "trials": len(subset),
                "accepted_trials": int(sum(row["accepted"] for row in subset)),
                "failed_trials": int(sum(not row["accepted"] for row in subset)),
                "min_attempts_to_accept": (
                    int(min(row["attempts"] for row in subset if row["accepted"]))
                    if any(row["accepted"] for row in subset) else None
                ),
                "max_attempts_to_accept": (
                    int(max(row["attempts"] for row in subset if row["accepted"]))
                    if any(row["accepted"] for row in subset) else None
                ),
            }
    summary = {
        "level": level_index,
        "position_bins": position_bins,
        "orientation_bins": orientation_bins,
        "cells": position_bins * orientation_bins,
        "resets_per_cell": resets_per_cell,
        "goal_sample_max_attempts": max_attempts,
        "thresholds": {
            "position_error_m": float(level["position_tolerance_m"]),
            "orientation_error_rad": float(level["orientation_tolerance_rad"]),
        },
        "accepted_trials": len(accepted),
        "failed_trials": len(failed),
        "accepted_rate": float(len(accepted) / max(len(rows), 1)),
        "accepted_fk_position_max_m": (
            float(max(row["position_error_m"] for row in accepted))
            if accepted else None
        ),
        "accepted_fk_orientation_max_rad": (
            float(max(row["orientation_error_rad"] for row in accepted))
            if accepted else None
        ),
        "failed_cells": [
            key for key, value in cell_summary.items()
            if value["accepted_trials"] == 0
        ],
        "goal_sampling_failed_cells": [
            key for key, value in cell_summary.items()
            if value["accepted_trials"] == 0
            and all(
                row["failure_stage"] == "goal_sampling"
                for row in rows
                if row["position_bin"] == value["position_bin"]
                and row["orientation_bin"] == value["orientation_bin"]
            )
        ],
        "cell_summary": cell_summary,
        "interpretation": (
            "A failed cell had no strict IK/self-clearance witness within "
            "goal_sample_max_attempts for the audited reset(s). It is a "
            "current-constraint infeasibility flag, not a global proof over "
            "all possible initial joint configurations."
        ),
    }
    return summary, rows


def self_contact_geometry_audit(config: dict, samples: int) -> tuple[dict, list[dict]]:
    """Cross-check exact self-contact events against mesh clearance and risk."""
    env = ThesisHomotopyEnv(config)
    rows: list[dict] = []
    rng = np.random.default_rng(int(config["seed"]) + 917)
    try:
        env.configure_episode("none", xi=1.0, strict=True)
        env.reset(seed=int(config["seed"]) + 917)
        for sample in range(samples):
            q = rng.uniform(
                env.robot.joint_lower_limits + 0.1,
                env.robot.joint_upper_limits - 0.1,
            )
            env._set_joint_state(q)
            p.performCollisionDetection(physicsClientId=env.client_id)
            contact = env._collision_events()["self_collision"]
            geometry = env._self_geometry()
            mesh_contact = geometry.distance_min <= 0.0
            rows.append({
                "sample": sample,
                "pybullet_contact": contact,
                "mesh_contact": mesh_contact,
                "distance_min_m": geometry.distance_min,
                "risk_max": geometry.risk_max,
                "ttc_min_s": geometry.ttc_min,
                "approach_max_mps": geometry.approach_max,
                "closest_first_slot": geometry.closest_pair[0],
                "closest_second_slot": geometry.closest_pair[1],
                "consistent": bool(
                    contact == mesh_contact
                    and (not contact or np.isclose(geometry.risk_max, 1.0))
                ),
            })
    finally:
        env.close()
    contacts = [row for row in rows if row["pybullet_contact"]]
    safe = [row for row in rows if not row["pybullet_contact"]]
    return {
        "samples": samples,
        "contacts": len(contacts),
        "collision_free": len(safe),
        "consistency_rate": float(np.mean([row["consistent"] for row in rows])),
        "maximum_contact_distance_m": (
            float(max(row["distance_min_m"] for row in contacts)) if contacts else None
        ),
        "minimum_collision_free_distance_m": (
            float(min(row["distance_min_m"] for row in safe)) if safe else None
        ),
        "safe_distance_m": env.d_self_safe,
        "query_distance_m": env.self_query_distance,
    }, rows


def write_results(output: Path, name: str, summary: dict, rows: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if rows:
        with (output / f"{name}_samples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def external_risk_audit(config: dict, samples: int) -> tuple[dict, list[dict]]:
    """Review reward distributions before freezing engineering risk settings."""
    settings = config["thesis"]["obstacles"]
    risk = settings["risk"]
    rng = np.random.default_rng(int(config["seed"]))
    capsule = CapsuleState(np.array([0., 0., -.2]), np.array([0., 0., .2]), .08, "audit")
    rows = []
    states = {"empty": None, "separated": (.25, .5), "approaching": (.01, .15),
              "contact": (0., 0.), "penetrating": (-.08, -.005)}
    for name, interval in states.items():
        for index in range(samples):
            distance = float(rng.uniform(*interval)) if interval is not None else .8
            if interval is None:
                features = np.empty((0, 37))
            else:
                sphere = ConvexSweep("sphere", [.07], [distance + .15, 0., 0.], np.eye(3))
                query = signed_capsule_distance(sphere, capsule)
                features = np.zeros((1, 37))
                features[0, :4] = query.distance
            cost = swept_risk(features, **risk)
            rows.append(dict(state=name, sample=index, signed_distance_m=distance, cost=cost,
                             reward_delta=-float(settings["lambda_ext"]) * cost))
    groups = {}
    for name in states:
        values = [r["reward_delta"] for r in rows if r["state"] == name]
        groups[name] = dict(mean=float(np.mean(values)), p05=float(np.percentile(values, 5)),
                            p95=float(np.percentile(values, 95)))
    return dict(risk=risk, lambda_ext=settings["lambda_ext"], distributions=groups,
                calibrated=False, note="Review these distributions with reaching rewards before freezing formal experiment values."), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("reward", "goal", "cells", "self-contact", "external-risk"),
    )
    parser.add_argument("--config", default="configs/experiments/paper_full.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--level", type=int, default=6)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resets-per-cell", type=int, default=1)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    output = ROOT / args.output
    if args.mode == "reward":
        summary, rows = reward_audit(config, args.samples)
    elif args.mode == "goal":
        summary, rows = goal_kinematics_audit(config, args.samples)
    elif args.mode == "cells":
        summary, rows = task_space_cell_audit(
            config, level_index=args.level, seed=args.seed,
            resets_per_cell=args.resets_per_cell,
        )
    elif args.mode == "self-contact":
        summary, rows = self_contact_geometry_audit(config, args.samples)
    elif args.mode == "external-risk":
        summary, rows = external_risk_audit(config, args.samples)
    write_results(output, args.mode, summary, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
