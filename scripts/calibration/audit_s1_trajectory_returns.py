#!/usr/bin/env python3
"""Same-start S1 reward audit using recorded actions and physical execution.

No learning, reward changes, horizon extension, teleporting during execution,
or synthetic collision rewards. Inputs are trusted local diagnostic snapshots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

POSITIVE = (
    "keypoint_tracking_reward", "keypoint_progress_reward",
    "keypoint_precision_reward", "joint_bottleneck_shaping_reward", "hold_reward",
)
NEGATIVE = (
    "leave_joint_tolerance_penalty", "precision_stop_cost", "velocity_cost",
    "smooth_cost", "hard_penalty", "timeout_penalty", "external_safety_penalty",
    "self_safety_penalty", "terminal_guard_penalty",
)


def summarize_trace(trace, gamma):
    """Finite physical episode return, first reward undiscounted; no entropy."""
    if not trace:
        raise ValueError("empty trajectory")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    discount = np.power(gamma, np.arange(len(trace), dtype=np.float64))
    components = {
        name: float(sum(w * row[name] for w, row in zip(discount, trace)))
        for name in (*POSITIVE, "success_reward", *NEGATIVE)
    }
    total = float(sum(w * row["reward"] for w, row in zip(discount, trace)))
    reconstructed = sum(components[k] for k in (*POSITIVE, "success_reward"))
    reconstructed -= sum(components[k] for k in NEGATIVE)
    residual = total - reconstructed
    if abs(residual) > 1e-7:
        raise ValueError(f"reward decomposition mismatch: {residual}")
    return {
        "discounted_return": total,
        "undiscounted_return": float(sum(row["reward"] for row in trace)),
        "discounted_components": components,
        "decomposition_residual": float(residual),
        "steps": len(trace),
    }


def paired_comparisons(rows, reference_mode="witness"):
    """Only call a route successful when the executed episode says so."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["episode"], {})[row["mode"]] = row
    comparisons = []
    for episode, modes in sorted(grouped.items()):
        witness = modes.get(reference_mode)
        if witness is None or witness["cause"] != "success":
            continue
        for mode, row in sorted(modes.items()):
            if mode == reference_mode or row["cause"] == "success":
                continue
            delta = witness["discounted_return"] - row["discounted_return"]
            comparisons.append({
                "episode": episode, "seed": row["seed"], "band": row["band"],
                "success_mode": reference_mode,
                "failure_mode": mode, "failure_cause": row["cause"],
                "success_return": witness["discounted_return"],
                "failure_return": row["discounted_return"],
                "success_minus_failure": delta,
                "successful_route_preferred": bool(delta > 0),
                "witness_sampled_mesh_margin_passed": witness["sampled_mesh_margin_passed"],
            })
    return comparisons


def rescore_self_weight(rows, target_weight):
    """Exact reward-only relabel of fixed trajectories; no new policy rollout."""
    if not 0 <= target_weight <= 1:
        raise ValueError("self weight must be in [0, 1]")
    result = []
    for source in rows:
        source_weight = source["lambda_self"]
        if source_weight <= 0:
            raise ValueError("positive source weight required to recover the unweighted cost")
        scale = target_weight / source_weight
        row = {k: v for k, v in source.items() if k != "trace"}
        components = dict(source["discounted_components"])
        old = components["self_safety_penalty"]
        components["self_safety_penalty"] = old * scale
        row["discounted_return"] += (1 - scale) * old
        row["undiscounted_return"] += (1 - scale) * sum(
            t["self_safety_penalty"] for t in source["trace"])
        row.update(discounted_components=components, lambda_self=target_weight,
                   source_lambda_self=source_weight,
                   reward_relabel="self penalty only; identical fixed physical trajectory")
        result.append(row)
    return result


def export_summary(primary_path, sensitivity_path, output_dir):
    """Generate small reproducible tables without copying the full step traces."""
    primary = json.loads(Path(primary_path).read_text())
    sensitivity = json.loads(Path(sensitivity_path).read_text())
    rows = primary["trajectories"] + sensitivity["trajectories"]
    references = sorted({r["mode"] for r in rows if r["mode"].startswith("witness")})
    comparisons = [c for mode in references for c in paired_comparisons(rows, mode)]
    selected = {}
    for row in rows:
        if (not row["mode"].startswith("witness") or row["cause"] != "success"
                or not row["sampled_mesh_margin_passed"]):
            continue
        if (row["episode"] not in selected
                or row["discounted_return"] > selected[row["episode"]]["discounted_return"]):
            selected[row["episode"]] = row
    selected_keys = {(r["episode"], r["mode"]) for r in selected.values()}
    selected_comparisons = [c for c in comparisons
                            if (c["episode"], c["success_mode"]) in selected_keys]
    training_rows = rescore_self_weight(rows, .2)
    training_comparisons = [c for mode in references for c in paired_comparisons(training_rows, mode)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [k for k in rows[0] if k not in ("trace", "discounted_components", "original_replay_parity")]
    with (output_dir / "trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    with (output_dir / "comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    summary = dict(
        gamma=primary["gamma"], horizon=primary["horizon"],
        primary_trajectory_count=len(primary["trajectories"]),
        primary_comparison_count=len(primary["comparisons"]),
        primary_counterexamples=[c for c in primary["comparisons"] if not c["successful_route_preferred"]],
        followup_trajectory_count=len(sensitivity["trajectories"]),
        all_comparison_count=len(comparisons),
        all_counterexamples=[c for c in comparisons if not c["successful_route_preferred"]],
        selected_reference_comparison_count=len(selected_comparisons),
        selected_reference_counterexamples=[c for c in selected_comparisons if not c["successful_route_preferred"]],
        training_self_weight_rescore=dict(
            lambda_self=.2,
            method="exact self-penalty-only relabel of fixed executed paths; not new policy evaluations",
            counterexamples=[c for c in training_comparisons if not c["successful_route_preferred"]],
            comparisons=training_comparisons,
            trajectories=training_rows,
        ),
        selection_caveat="post-hoc best measured return among executed, endpoint-margin-passing paths; not held-out validation or continuous safety certification",
        selected_references=[{k: r[k] for k in fields} for _, r in sorted(selected.items())],
        comparisons=comparisons,
        trajectories=[{k: v for k, v in r.items() if k != "trace"} for r in rows],
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def init_worker(source, config_path):
    import torch
    from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
    from rl_risk_sac.utils.config import load_config
    from scripts.core.train_thesis_homotopy import apply_stage_profile

    global ENV, STATES, RECORDED
    torch.set_num_threads(1)
    config = load_config(config_path)
    apply_stage_profile(config, "s1")
    ENV = ThesisHomotopyEnv(config)
    source = Path(source)
    STATES = torch.load(source / "latest1250k_states.pt", map_location="cpu", weights_only=False)
    RECORDED = {
        tag: {r["episode"]: r for r in json.loads((source / f"{tag}_replays.json").read_text())}
        for tag in ("best1071k", "latest1250k")
    }


def mesh_clearance():
    import pybullet as p
    return min(
        (float(point[8]) for obstacle in ENV.obstacles
         for point in p.getClosestPoints(ENV.robot_id, obstacle.body_id, 2.,
                                         physicsClientId=ENV.client_id)),
        default=2.,
    )


def run_case(task):
    episode, mode = task
    saved = STATES[episode]
    initial = saved["initial"]
    ENV.restore_episode_state(initial)
    if ENV.horizon != 500 or initial["step_count"] != 0:
        raise ValueError("audit requires the original 500-step episode from its initial state")
    path = np.asarray(initial["obstacle_sampling"]["witness_waypoints"], dtype=np.float64)
    target_index = 1
    trace = []
    min_mesh = mesh_clearance()
    max_reward_error = max_position_error = max_orientation_error = 0.
    recorded = RECORDED.get(mode, {}).get(episode)
    for index in range(ENV.horizon):
        if recorded is not None:
            if index >= len(recorded["trace"]):
                raise ValueError(f"recorded actions exhausted without terminal: {episode} {mode}")
            action = np.asarray(recorded["trace"][index]["action"], dtype=np.float32)
        elif mode == "wait":
            action = np.zeros(6, dtype=np.float32)
        elif mode.startswith("witness"):
            q, _ = ENV.robot.joint_state_exact()
            while target_index < len(path) - 1 and np.max(np.abs(q - path[target_index])) < .025:
                target_index += 1
            gain = float(mode.split("gain")[1]) if "gain" in mode else 3.
            action = np.clip(gain * (path[target_index] - q) / ENV.action_scale, -1, 1)
        else:
            raise ValueError(f"unknown mode {mode}")
        _, reward, _, terminated, truncated, info = ENV.step(action)
        sampled_mesh = mesh_clearance()
        min_mesh = min(min_mesh, sampled_mesh)
        row = {name: float(info[name]) for name in (*POSITIVE, *NEGATIVE)}
        row.update(
            step=index + 1, reward=float(reward),
            success_reward=float(ENV.reward_parameters["success_bonus"]) * bool(info["task_reached"]),
            position_error_m=float(info["next_rho_position"]),
            orientation_error_rad=float(info["next_rho_orientation"]),
            encoded_external_distance_m=float(info["control_min_distance"]),
            control_self_distance_m=float(info["control_self_min_distance"]),
            sampled_mesh_clearance_m=sampled_mesh,
            action=np.asarray(action).tolist(),
            commanded_joint_velocity=np.asarray(info["commanded_joint_velocity"]).tolist(),
            success_hold_count=int(info["success_hold_count"]),
        )
        trace.append(row)
        if recorded is not None:
            expected = recorded["trace"][index]
            max_reward_error = max(max_reward_error, abs(reward - expected["reward"]))
            max_position_error = max(max_position_error, abs(row["position_error_m"] - expected["next_rho_position"]))
            max_orientation_error = max(max_orientation_error, abs(row["orientation_error_rad"] - expected["next_rho_orientation"]))
        if terminated or truncated:
            break
    if not (terminated or truncated):
        raise ValueError("trajectory did not terminate within the unchanged horizon")
    cause = (
        "success" if info["task_reached"] else "timeout" if truncated
        else "obstacle_collision" if info["obstacle_collision"]
        else "self_collision" if info["self_collision"]
        else "environment_collision" if info["environment_collision"] else "joint_limit"
    )
    parity = None
    if recorded is not None:
        parity = dict(cause_matches=cause == recorded["cause"],
                      length_matches=len(trace) == len(recorded["trace"]),
                      max_reward_error=max_reward_error,
                      max_position_error_m=max_position_error,
                      max_orientation_error_rad=max_orientation_error)
        if not (parity["cause_matches"] and parity["length_matches"]
                and max_reward_error < 1e-5 and max_position_error < 1e-6
                and max_orientation_error < 1e-5):
            raise ValueError(f"original replay mismatch {episode} {mode}: {parity}")
    self_min = min(row["control_self_distance_m"] for row in trace)
    result = dict(
        episode=episode, seed=saved["case"]["seed"], band=saved["case"]["band"],
        mode=mode, cause=cause, original_replay_parity=parity,
        position_tolerance_m=ENV.contract.position_tolerance,
        orientation_tolerance_rad=ENV.contract.orientation_tolerance,
        lambda_self=ENV.contract.lambda_self,
        control_dt=ENV.control_dt,
        external_safe_distance_m=ENV.d_safe,
        self_safe_distance_m=ENV.d_self_safe,
        **summarize_trace(trace, float(ENV.config["sac"]["gamma"])),
        final_position_error_m=trace[-1]["position_error_m"],
        final_orientation_error_rad=trace[-1]["orientation_error_rad"],
        min_sampled_mesh_clearance_m=min_mesh,
        min_control_self_distance_m=self_min,
        sampled_mesh_margin_passed=bool(min_mesh >= ENV.d_safe and self_min >= ENV.d_self_safe),
        witness_waypoint_count=len(path),
        witness_joint_path_length=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        straight_joint_length=float(np.linalg.norm(path[-1] - path[0])),
        trace=trace,
    )
    return result


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/paper_full.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--modes", nargs="+", default=["best1071k", "latest1250k", "witness", "wait"],
                        choices=["best1071k", "latest1250k", "witness", "wait",
                                 "witness_gain5", "witness_gain8", "witness_gain12"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    latest = json.loads((args.source / "latest1250k_replays.json").read_text())
    episodes = sorted(r["episode"] for r in latest) if args.episodes is None else args.episodes
    tasks = [(ep, mode) for ep in episodes for mode in args.modes]
    rows = []
    with mp.get_context("spawn").Pool(args.workers, initializer=init_worker,
                                       initargs=(str(args.source), str(args.config))) as pool:
        for result in pool.imap_unordered(run_case, tasks):
            rows.append(result)
            print(f"AUDIT ep={result['episode']} mode={result['mode']} cause={result['cause']} "
                  f"steps={result['steps']} G={result['discounted_return']:.6f} "
                  f"margin={result['sampled_mesh_margin_passed']}", flush=True)
    from rl_risk_sac.utils.config import load_config
    config = load_config(args.config)
    result = dict(
        method="actual env.step from matched initial snapshots, original horizon, recorded actor actions, no entropy or bootstrap",
        scope="failure-directed L0 cases; not an unbiased success-rate or all-level reward guarantee",
        clearance_caveat="mesh sampled only at control endpoints; substep contact checks remain active; not a continuous clearance certificate",
        gamma=config["sac"]["gamma"], horizon=config["thesis"]["horizon"],
        config_path=str(args.config.resolve()), config_sha256=fingerprint(args.config),
        source=str(args.source.resolve()),
        source_sha256={name: fingerprint(args.source / name) for name in (
            "latest1250k_states.pt", "latest1250k_replays.json", "best1071k_replays.json")},
        reward_config=config["thesis"]["reward"],
        comparisons=paired_comparisons(rows),
        trajectories=sorted(rows, key=lambda r: (r["episode"], r["mode"])),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    failed = [r for r in result["comparisons"] if not r["successful_route_preferred"]]
    print(f"DONE trajectories={len(rows)} comparisons={len(result['comparisons'])} "
          f"ranking_counterexamples={len(failed)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
