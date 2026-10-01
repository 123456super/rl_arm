#!/usr/bin/env python3
"""Replay O7--O9 trajectories and contrast first contiguous alignment collapses."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.analyze_orientation_state_provenance import (
    ROOT, STEP_INFO_KEYS, _describe, _json_ready, _path, _sha256, evaluate,
)
from scripts.core.evaluate_thesis_homotopy import physical_cpu_ids
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config


WINDOW = 5
METRICS = (
    "step", "rho_orientation", "rho_position", "remaining_time_fraction",
    "actor_command_alignment", "actor_measured_alignment",
    "orientation_progress_rad_s", "position_progress_m_s",
    "actor_joint_speed_rms_fraction", "measured_joint_speed_rms_fraction",
    "orientation_capacity_rad_s", "actor_command_effective_capacity_fraction",
    "actor_dls_action_cosine", "jacobian_angular_sigma_min",
    "jacobian_angular_condition", "jacobian_linear_sigma_min",
    "jacobian_linear_condition", "joint_limit_margin_fraction",
    "qdot_state_rms_fraction", "command_delta_norm_rad_s", "reward",
    "keypoint_progress_reward", "keypoint_tracking_reward",
    "keypoint_precision_reward", "velocity_cost", "smooth_cost",
    "safety_penalty", "precision_action_scale", "orientation_reversal",
    "joint_task_direction_cosine", "actor_action_change_norm_rad_s",
    "actor_saturated_joint_fraction", "wrist_bend_sin_abs",
)


def contiguous_collapse(rows: list[dict], low: float = .6, high: float = 1.) -> dict | None:
    """Require five successive actual control steps inside the band."""
    window: deque[dict] = deque(maxlen=WINDOW)
    for row in rows:
        if not low <= row["rho_orientation"] <= high or (
            window and row["step"] != window[-1]["step"] + 1
        ):
            window.clear()
        if not low <= row["rho_orientation"] <= high:
            continue
        window.append(row)
        if len(window) == WINDOW:
            alignment = [r["actor_command_alignment"] for r in window]
            if np.mean(alignment) < .25 and sum(a < 0 for a in alignment) >= 2:
                return {
                    "window_start_step": int(window[0]["step"]),
                    "onset_step": int(next(r["step"] for r in window
                                           if r["actor_command_alignment"] < 0)),
                    "detection_step": int(row["step"]),
                    "mean_alignment": float(np.mean(alignment)),
                    "negative_steps": sum(a < 0 for a in alignment),
                }
    return None


def compact(row: dict, action_scale: float) -> dict:
    angular = np.asarray(row["angular_jacobian"])
    linear = np.asarray(row["linear_jacobian"])
    sv = np.linalg.svd(linear, compute_uv=False)
    angular_sv = np.asarray(row["jacobian_angular_singular_values"])
    q = np.asarray(row["q_normalized"])
    qdot = np.asarray(row["qdot_normalized"])
    actor = np.asarray(row["actor_joint_velocity"])
    directional = np.asarray(row["jacobian_directional_row"])
    position_directional = np.asarray(row["position_error_direction"]) @ linear
    q_rad = np.asarray(row["q_rad"])
    result = {k: float(row[k]) for k in METRICS if k in row}
    result.update({
        "episode": int(row["episode"]),
        "initial_orientation_bin": int(row["initial_orientation_bin"]),
        "initial_position_bin": int(row["initial_position_bin"]),
        "q_normalized": q.tolist(), "qdot_normalized": qdot.tolist(),
        "actor_joint_velocity_rad_s": actor.tolist(),
        "q_rad": q_rad.tolist(),
        "angular_jacobian": angular.tolist(),
        "angular_jacobian_singular_values": angular_sv.tolist(),
        "jacobian_angular_sigma_min": float(angular_sv[-1]),
        "jacobian_angular_condition": float(angular_sv[0] / max(angular_sv[-1], 1.e-12)),
        "jacobian_linear_sigma_min": float(sv[-1]),
        "jacobian_linear_condition": float(sv[0] / max(sv[-1], 1.e-12)),
        "joint_limit_margin_fraction": float(np.min(1. - np.abs(q))),
        "qdot_state_rms_fraction": float(np.sqrt(np.mean(qdot**2))),
        "orientation_error_direction": np.asarray(row["orientation_error_direction"]).tolist(),
        "jacobian_directional_row": np.asarray(row["jacobian_directional_row"]).tolist(),
        "measured_joint_speed_rms_fraction": float(row["measured_joint_speed_rms_fraction"]),
        "precision_action_scale": float(row["precision_action_scale"]),
        "joint_task_direction_cosine": float(np.dot(directional, position_directional)
            / max(np.linalg.norm(directional) * np.linalg.norm(position_directional), 1.e-12)),
        "actor_action_change_norm_rad_s": float(row["actor_action_change_norm_rad_s"]),
        "actor_saturated_joint_fraction": float(np.mean(np.abs(actor) >= .95 * action_scale)),
        "wrist_bend_sin_abs": float(abs(np.sin(q_rad[4]))),
        "crossing_phase": row["crossing_phase"],
    })
    return result


def phase(step: int, crossing_step: int | None) -> str:
    return "after_crossing" if crossing_step is not None and step >= crossing_step else "before_crossing"


def select_success_controls(event: dict, candidates: list[dict], *,
                            step_tolerance: int = 20, rho_tolerance: float = .1,
                            position_tolerance: float = .12) -> tuple[dict | None, dict]:
    """One control per failure, exact initial bins and crossing phase."""
    eligible = [r for r in candidates if
                r["initial_orientation_bin"] == event["initial_orientation_bin"]
                and r["initial_position_bin"] == event["initial_position_bin"]
                and r["crossing_phase"] == event["crossing_phase"]]
    within = [r for r in eligible if
              abs(r["step"] - event["step"]) <= step_tolerance
              and abs(r["rho_orientation"] - event["rho_orientation"]) <= rho_tolerance
              and abs(r["rho_position"] - event["rho_position"]) <= position_tolerance]
    if not within:
        return None, {"same_bin_and_phase_candidates": len(eligible)}
    best = min(within, key=lambda r: (
        abs(r["rho_orientation"] - event["rho_orientation"]) / rho_tolerance
        + abs(r["rho_position"] - event["rho_position"]) / position_tolerance
        + abs(r["step"] - event["step"]) / step_tolerance,
        r["episode"], r["step"],
    ))
    return best, {
        "same_bin_and_phase_candidates": len(eligible),
        "delta_rho_orientation": best["rho_orientation"] - event["rho_orientation"],
        "delta_rho_position": best["rho_position"] - event["rho_position"],
        "delta_step": best["step"] - event["step"],
    }


def compare(pairs: list[dict], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    result = {"pairs": len(pairs), "unique_success_controls": len({
        p["success"]["episode"] for p in pairs
    }), "metrics": {}}
    if not pairs:
        return result
    for metric in METRICS:
        values = np.asarray([
            [p["failure"][metric], p["success"][metric]] for p in pairs
        ], dtype=np.float64)
        delta = values[:, 0] - values[:, 1]
        means = np.mean(delta[rng.integers(len(delta), size=(2000, len(delta)))], axis=1)
        result["metrics"][metric] = {
            "failure": _describe(values[:, 0]), "success": _describe(values[:, 1]),
            "paired_delta": _describe(delta),
            "bootstrap_95pct_ci": np.quantile(means, [.025, .975]).tolist(),
        }
    for vector in ("q_normalized", "q_rad", "qdot_normalized", "actor_joint_velocity_rad_s",
                   "angular_jacobian_singular_values"):
        failure = np.asarray([p["failure"][vector] for p in pairs])
        success = np.asarray([p["success"][vector] for p in pairs])
        delta = failure - success
        draws = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(axis=1)
        result["metrics"][vector] = {
            "failure_mean": failure.mean(axis=0).tolist(),
            "success_mean": success.mean(axis=0).tolist(),
            "paired_delta_mean": delta.mean(axis=0).tolist(),
            "paired_delta_bootstrap_95pct_ci": np.quantile(draws, [.025, .975], axis=0).T.tolist(),
        }
    return result


def summarize(rows: list[dict]) -> dict:
    return {"states": len(rows), "metrics": {
        metric: _describe(r[metric] for r in rows) for metric in METRICS
    }}


def analyze(episodes: list[dict], transitions: list[dict], *, action_scale: float,
            seed: int) -> dict:
    by_episode: dict[int, list[dict]] = defaultdict(list)
    episode_info = {e["episode"]: e for e in episodes}
    for row in transitions:
        info = episode_info[row["episode"]]
        row["crossing_phase"] = phase(row["step"], info["crossing_0p6_step"])
        by_episode[row["episode"]].append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda r: r["step"])
        for i, row in enumerate(rows):
            previous = np.zeros(6) if i == 0 else np.asarray(rows[i - 1]["actor_joint_velocity"])
            row["actor_action_change_norm_rad_s"] = float(np.linalg.norm(
                np.asarray(row["actor_joint_velocity"]) - previous))

    events = []
    controls = []
    event_rows = {}
    success_episodes = 0
    failure_episodes = 0
    for ep in episodes:
        rows = by_episode[ep["episode"]]
        if ep["success"]:
            success_episodes += 1
            controls.extend(compact(row, action_scale) for row in rows
                            if .6 <= row["rho_orientation"] <= 1.)
        else:
            failure_episodes += 1
        event = contiguous_collapse(rows)
        if event is None:
            continue
        event["episode"] = ep["episode"]
        event["success"] = ep["success"]
        event["initial_orientation_bin"] = ep["initial_orientation_bin"]
        event["initial_position_bin"] = ep["initial_position_bin"]
        event["crossing_0p6_step"] = ep["crossing_0p6_step"]
        event["first_entry_step"] = next((r["step"] for r in rows if .6 <= r["rho_orientation"] <= 1.), None)
        event["onset_phase"] = phase(event["onset_step"], ep["crossing_0p6_step"])
        event["detection_phase"] = phase(event["detection_step"], ep["crossing_0p6_step"])
        event["timeline"] = [compact(row, action_scale) for row in rows
                             if event["onset_step"] - 10 <= row["step"] <= event["detection_step"] + 5]
        event_rows[ep["episode"]] = rows
        events.append(event)

    comparisons = {}
    for phase_name in ("before_crossing", "after_crossing"):
        for label, step_key in (("onset", "onset_step"), ("detection", "detection_step"),
                                ("pre_onset_5", None)):
            pairs = []
            attempted = 0
            unmatched = []
            for event in events:
                if event["success"] or event["onset_phase"] != phase_name:
                    continue
                attempted += 1
                step = event[step_key] if step_key else event["onset_step"] - 5
                candidate = next((r for r in event_rows[event["episode"]]
                                  if r["step"] == step), None)
                if candidate is None or not .6 <= candidate["rho_orientation"] <= 1.:
                    unmatched.append({"episode": event["episode"], "reason": "not_in_band"})
                    continue
                failure = compact(candidate, action_scale)
                matched, balance = select_success_controls(failure, controls)
                if matched is None:
                    unmatched.append({"episode": event["episode"], **balance})
                    continue
                pairs.append({"failure": failure, "success": matched, "balance": balance})
            comparisons[f"{phase_name}_{label}"] = {
                "attempted_failure_episodes": attempted,
                "unmatched": unmatched,
                **compare(pairs, seed), "pairs_detail": pairs,
            }
    temporal = {}
    for phase_name in ("before_crossing", "after_crossing"):
        subset = [e for e in events if not e["success"] and e["onset_phase"] == phase_name]
        temporal[phase_name] = {}
        for offset in (-10, -5, 0, 5, 10):
            selected = []
            for event in subset:
                row = next((r for r in event_rows[event["episode"]]
                            if r["step"] == event["onset_step"] + offset), None)
                if row is not None:
                    selected.append(compact(row, action_scale))
            temporal[phase_name][str(offset)] = summarize(selected)
    entry = {}
    for threshold in (1., .8, .6):
        entry[str(threshold)] = {}
        for success in (True, False):
            selected = []
            for ep in episodes:
                if ep["success"] != success:
                    continue
                row = next((r for r in by_episode[ep["episode"]]
                            if r["rho_orientation"] < threshold), None)
                if row is not None:
                    selected.append(compact(row, action_scale))
            entry[str(threshold)]["success" if success else "failure"] = summarize(selected)
    stage_pairs = []
    for event in events:
        if event["success"]:
            continue
        failure = next(r for r in event["timeline"] if r["step"] == event["onset_step"])
        eligible = [r for r in controls if
                    r["initial_orientation_bin"] == failure["initial_orientation_bin"]
                    and r["initial_position_bin"] == failure["initial_position_bin"]
                    and r["crossing_phase"] == "before_crossing"
                    and abs(r["rho_orientation"] - failure["rho_orientation"]) <= .03]
        if eligible:
            success = min(eligible, key=lambda r: (
                abs(r["rho_orientation"] - failure["rho_orientation"]), r["step"], r["episode"]))
            stage_pairs.append({"failure": failure, "success": success})
    sensitivity = {}
    for low in (.4, .6):
        for high in (1., float(np.pi)):
            detected = [(ep, contiguous_collapse(by_episode[ep["episode"]], low, high))
                        for ep in episodes]
            sensitivity[f"{low}_{high:.3f}"] = {
                "success_detected": sum(ep["success"] and ev is not None for ep, ev in detected),
                "failure_detected": sum(not ep["success"] and ev is not None for ep, ev in detected),
                "first_failure_onsets": [
                    {"episode": ep["episode"], **ev}
                    for ep, ev in detected if not ep["success"] and ev is not None
                ],
            }
    return {
        "episode_counts": {"success": success_episodes, "failure": failure_episodes},
        "detection_rule": "first five consecutive control steps with rho_R in [0.6,1.0], mean command alignment <0.25 and at least two negative alignments; onset is first negative command alignment in that window",
        "matching_rule": "exact O7/O8/O9 and initial position bin and first-0.6-crossing phase; success step +/-20, rho_R +/-0.1 rad, rho_p +/-0.12 m; nearest normalized L1; with replacement",
        "events": events,
        "first_entry_comparison": entry,
        "failure_event_temporal_summary": temporal,
        "same_angle_healthy_descent_reference": {
            "warning": "descriptive reference only: same initial O/P bins and rho_R within 0.03 rad; success state before first crossing; time, position and phase NOT controlled",
            **compare(stage_pairs, seed), "pairs_detail": stage_pairs,
        },
        "band_sensitivity": sensitivity,
        "event_counts": {
            status: {p: sum(e["success"] == (status == "success") and e["onset_phase"] == p for e in events)
                     for p in ("before_crossing", "after_crossing")}
            for status in ("success", "failure")
        },
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=51001)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--reference-episodes", type=int, default=1000)
    args = parser.parse_args()
    output = _path(args.output)
    if output.exists():
        raise FileExistsError(output)
    config_path, checkpoint_path = _path(args.config), _path(args.checkpoint)
    config = load_config(config_path)
    config["device"] = "cpu"
    torch.set_num_threads(1)
    orientation_bins = int(config["thesis"]["orientation_curriculum"].get("deterministic_probe_orientation_bins", 10))
    if orientation_bins != 10 or args.reference_episodes < 100 or args.reference_episodes % 100:
        parser.error("requires ten orientation bins and reference episodes divisible by 100")
    episode_ids = [i for i in range(args.reference_episodes) if i % orientation_bins in (7, 8, 9)]
    env = ThesisHomotopyEnv(config)
    try:
        agent = ThesisSACAgent(env.observation_space.shape[0], env.action_space.shape[0], config)
        joint_lower = env.robot.joint_lower_limits
        joint_upper = env.robot.joint_upper_limits
        joint_names = env.robot.config.joint_names
    finally:
        env.close()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent.actor.load_state_dict(checkpoint.get("agent", checkpoint).get("actor", checkpoint))
    agent.actor.eval()
    start = time.perf_counter()
    with ParallelThesisEnvPool(config, [args.seed + 1_000_000 + i for i in range(args.num_envs)],
                               cpu_ids=physical_cpu_ids(args.num_envs), step_info_keys=STEP_INFO_KEYS,
                               start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn") as pool:
        episodes, transitions, _ = evaluate(
            agent, pool, config, episodes=len(episode_ids), seed=args.seed,
            level_index=0, band_low=.6, band_high=1., damping=.05,
            counterfactuals=False, episode_ids=episode_ids,
            capture_all_large_states=True,
        )
    for row in transitions:
        q_normalized = np.asarray(row["q_normalized"])
        row["q_rad"] = joint_lower + (q_normalized + 1.) * .5 * (joint_upper - joint_lower)
    result = analyze(episodes, transitions,
                     action_scale=float(config["thesis"]["action_scale"]), seed=args.seed)
    result.update({"checkpoint": str(checkpoint_path.relative_to(ROOT)),
                   "checkpoint_sha256": _sha256(checkpoint_path),
                   "seed": args.seed, "episode_ids": episode_ids,
                   "joint_names": joint_names,
                   "joint_lower_rad": joint_lower, "joint_upper_rad": joint_upper,
                   "elapsed_seconds": time.perf_counter() - start,
                   "episode_outcomes": episodes})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episode_counts": result["episode_counts"],
                      "event_counts": result["event_counts"],
                      "comparisons": {k: {"attempted": v["attempted_failure_episodes"],
                                          "pairs": v["pairs"]}
                                      for k, v in result["comparisons"].items()}}, indent=2))


if __name__ == "__main__":
    main()
