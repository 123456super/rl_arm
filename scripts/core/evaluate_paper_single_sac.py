#!/usr/bin/env python3
"""Evaluate deterministic single-SAC reaching, optionally following geometric plans."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch

from rl_risk_sac.algorithms.paper_sac import PaperSACAgent
from rl_risk_sac.envs.parallel_paper_env import ParallelPaperEnvPool
from rl_risk_sac.tasks.paper_task_space_planner import SimulationTaskSpacePlanner
from rl_risk_sac.tasks.paper_reaching import PAPER_POSITION_THRESHOLDS_MM
from rl_risk_sac.utils.paper_config import PAPER_PROTOCOL, experiment_signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=90300, help="held-out seed offset in the reserved high-half seed domain")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--planner", action="store_true", help="kinodynamic task-space search + checked B-spline")
    parser.add_argument("--output")
    parser.add_argument("--allow-validation", action="store_true")
    parser.add_argument("--combined-error-tolerance", type=float,
                        help="optional pose-error threshold sweep; defaults to the checkpoint's training tolerance")
    args = parser.parse_args()
    if min(args.episodes, args.num_envs) < 1 or not 0 <= args.seed < 2**31 - args.episodes:
        parser.error("invalid episode/worker count or held-out seed offset")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != PAPER_PROTOCOL:
        parser.error("requires paper_single_sac_v2; v1 used different actions/networks")
    if checkpoint["validation_mode"] and not args.allow_validation:
        parser.error("validation checkpoints require --allow-validation and do not establish policy quality")
    config = copy.deepcopy(checkpoint["config"])
    config["device"] = args.device
    if experiment_signature(config) != checkpoint["signature"]:
        raise ValueError("checkpoint configuration signature mismatch")
    if args.combined_error_tolerance is not None:
        if (not np.isfinite(args.combined_error_tolerance)
                or args.combined_error_tolerance < config["paper"]["success"]["strict_combined_error_tolerance"]):
            parser.error("evaluation tolerance must be finite and at least the configured strict IK tolerance")
        config["paper"]["success"]["combined_error_tolerance"] = args.combined_error_tolerance
    torch.set_num_threads(1)
    agent = PaperSACAgent(config)
    agent.load_state_dict(checkpoint["agent"])
    agent.actor.eval()
    workers = min(args.num_envs, args.episodes)
    pool = ParallelPaperEnvPool(config, list(range(600, 600 + workers)))
    observations, active, rows = {}, {}, []
    started, next_episode = time.perf_counter(), 0
    try:
        while len(rows) < args.episodes:
            new = {}
            for i in range(workers):
                if i not in active and next_episode < args.episodes:
                    active[i] = dict(episode=next_episode, seed=2**31 + args.seed + next_episode,
                                     waypoint=1, reference=-1, planner_time_s=0.)
                    new[i] = active[i]["seed"]
                    next_episode += 1
            for i, (observation, info) in pool.reset_many({i: {} for i in new}, seeds=new).items():
                observations[i] = observation
                active[i]["info"] = info
                if args.planner:
                    scene = info["scene"]
                    planner = SimulationTaskSpacePlanner(config["planner"], scene["obstacle_centers"],
                        config["paper"]["obstacles"]["radius"],
                        workspace_radius=config["paper"]["goal"]["workspace_radius"],
                        base=config["robot"]["base_position"])
                    planning_started = time.perf_counter()
                    try:
                        plan = planner.plan(info["ee_position"], scene["goal_position"])
                        spacing = config["planner"]["waypoint_spacing"]
                        path = plan.waypoints(spacing)
                        for _ in range(6):
                            if planner.free(path):
                                break
                            spacing *= .5
                            path = plan.waypoints(spacing)
                        if not planner.free(path):
                            raise RuntimeError("resampled waypoint edges fail collision checks")
                        active[i].update(path=path, quaternion=scene["goal_quaternion"],
                                         expanded=plan.expanded, smoothing=plan.smoothing)
                    except RuntimeError as error:
                        row = dict(episode=active[i]["episode"], seed=active[i]["seed"], status="planning_failed",
                            length=0, reached=False, safe_success=False, report_success=False, collision_seen=False,
                            final_report_success=False, strict_success=False, joint_limit=False, joint_limit_seen=False,
                            position_thresholds_seen=[False] * len(PAPER_POSITION_THRESHOLDS_MM),
                            pose_thresholds_seen=[False] * len(PAPER_POSITION_THRESHOLDS_MM),
                            position_error_m=float(info["position_error_m"]),
                            orientation_error_rad=float(info["orientation_error_rad"]), reason=str(error),
                            planner_time_s=time.perf_counter() - planning_started)
                        rows.append(row)
                        del active[i]
                        print(f"单SAC评估 {len(rows)}/{args.episodes}: planning_failed", flush=True)
                        continue
                    active[i]["planner_time_s"] = time.perf_counter() - planning_started
            if not active:
                continue
            references = {}
            if args.planner:
                for i, worker in active.items():
                    path, k = worker["path"], worker["waypoint"]
                    position = worker["info"]["ee_position"]
                    # Advance monotonically; a waypoint is only a reference,
                    # never a separate success/episode termination condition.
                    while k < len(path) - 1 and np.linalg.norm(position - path[k]) <= .04:
                        k += 1
                    worker["waypoint"] = k
                    if worker["reference"] != k:
                        references[i] = (path[k], worker["quaternion"])
                        worker["reference"] = k
                observations.update(pool.set_references(references))
            indices = list(active)
            actions = agent.select_actions([observations[i] for i in indices], deterministic=True)
            for i, result in pool.step_many(dict(zip(indices, actions))).items():
                observation, _, _, terminated, truncated, info = result
                observations[i] = observation
                active[i]["info"] = info
                if terminated or truncated:
                    worker = active.pop(i)
                    if info["safe_success"]:
                        status = "success"
                    elif info["reached"]:
                        status = "reached_with_collision" if info["collision_seen"] else "reached_with_joint_limit"
                    elif info["joint_limit_seen"]:
                        status = "joint_limit"
                    else:
                        status = "timeout_with_collision" if info["collision_seen"] else "timeout"
                    row = {key: info[key] for key in ("length", "reached", "safe_success", "report_success",
                            "final_report_success", "strict_success", "joint_limit_seen",
                            "collision_seen", "position_error_m", "orientation_error_rad", "final_combined_error",
                            "episode_return", "capsule_clearance_m", "joint_limit", "min_position_error_m",
                            "min_orientation_error_rad", "min_combined_error")}
                    row.update({key: info[key].tolist() for key in ("position_thresholds_seen", "pose_thresholds_seen")})
                    row.update(episode=worker["episode"], seed=worker["seed"], status=status,
                               planner_time_s=worker["planner_time_s"])
                    rows.append(row)
                    print(f"单SAC评估 {len(rows)}/{args.episodes}: {status} steps={row['length']} "
                          f"位置={row['position_error_m'] * 1000:.2f}mm 姿态={row['orientation_error_rad']:.4f}rad "
                          f"累计无碰撞成功={sum(r['safe_success'] for r in rows) / len(rows):.1%}", flush=True)
    finally:
        pool.close()
    rows.sort(key=lambda row: row["episode"])
    rates = {f"{key}_rate": float(np.mean([row[key] for row in rows]))
             for key in ("reached", "safe_success", "report_success", "final_report_success", "strict_success", "collision_seen")}
    orientation_limit = config["paper"]["success"]["report_orientation_tolerance"]
    threshold_rates = {}
    for pose in (False, True):
        key = "pose" if pose else "position"
        final_rates, ever_rates = {}, {}
        for index, mm in enumerate(PAPER_POSITION_THRESHOLDS_MM):
            label = f"{mm}mm_{orientation_limit:g}rad" if pose else f"{mm}mm"
            safe = lambda row: (row["status"] != "planning_failed" and not row["collision_seen"]
                                and not row["joint_limit_seen"])
            final_rates[label] = float(np.mean([safe(row) and row["position_error_m"] <= mm / 1000
                and (not pose or row["orientation_error_rad"] <= orientation_limit) for row in rows]))
            ever_rates[label] = float(np.mean([safe(row) and row[f"{key}_thresholds_seen"][index] for row in rows]))
        threshold_rates[f"final_collision_free_{key}_threshold_rates"] = final_rates
        threshold_rates[f"ever_collision_free_{key}_threshold_rates"] = ever_rates
    report = dict(protocol=PAPER_PROTOCOL, checkpoint=str(Path(args.checkpoint).resolve()),
        signature=checkpoint["signature"], validation_mode=checkpoint["validation_mode"],
        evaluation_signature=experiment_signature(config), success=config["paper"]["success"],
        deterministic=True, planner=args.planner, seed_domain="reserved_high_half_uint32",
        seed_offset=args.seed, elapsed_s=time.perf_counter() - started, episodes=args.episodes,
        planning_failures=sum(row["status"] == "planning_failed" for row in rows),
        **rates, **threshold_rates, results=rows)
    default_name = f"evaluation_{'planned' if args.planner else 'direct'}_seed{args.seed}.json"
    output = Path(args.output) if args.output else Path(args.checkpoint).parent.parent / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"评估已保存：{output}", flush=True)


if __name__ == "__main__":
    main()
