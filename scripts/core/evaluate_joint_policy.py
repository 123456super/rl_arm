#!/usr/bin/env python3
"""Evaluate reaching -> local passage -> reaching from full held-out scenes."""
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

from rl_risk_sac.algorithms.dual_sac import build_thesis_sac
from rl_risk_sac.control.joint_policy import JointControlFailure, JointPolicyController
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.local_avoidance import (
    LOCAL_OBSERVATION_DIM, PROTOCOL, GEOMETRIC_PROTOCOL, MeshClearance, actor_digest, local_protocol, geometric_mode,
)
from rl_risk_sac.tasks.local_samples import configure_nominal_episode, simulation_config, validate_bank


def validate_evaluation_bank(bank, state, reaching):
    identities = validate_bank(bank, state["config"], reaching, split=("validation", "test"))
    trace_seeds = {sample["provenance"]["seed"] for sample in bank["samples"]}
    if (identities.intersection(state["train_sample_ids"])
            or trace_seeds.intersection(state["train_trace_seeds"])):
        raise ValueError("evaluation scenes/source traces overlap with training")


def restore_full_scene(env, sample, config):
    """Recreate the pre-approach S0 start, using only held-out scene metadata.

    The bank's captured local entry, direction and exit are NOT supplied to
    the joint controller. It must generate its own reference during execution.
    """
    configure_nominal_episode(env, config)
    env.reset(seed=int(sample["provenance"]["seed"]))
    initial = copy.deepcopy(env.episode_state_dict())
    for key in ("goal_position", "goal_quaternion"):
        if not np.allclose(initial[key], sample["state"][key], rtol=0., atol=1e-6):
            raise ValueError("source seed no longer reproduces the bank goal; refusing a different scene")
    initial["obstacles"] = copy.deepcopy(sample["state"]["obstacles"])
    initial["contract"].terminate_on_success = True
    initial["contract"].strict_obstacle_collision = True
    return env.restore_episode_state(initial)


def evaluate_episode(env, controller, observation):
    controller.reset()
    first_physics_step = env.physics_step_count
    margins = MeshClearance(env)
    minimum_external, minimum_self = margins.external, margins.self_distance
    info = {}
    status = None
    if margins.collision:
        status = "initial_collision"
    else:
        while True:
            try:
                observation, terminated, truncated, info = controller.step(env, observation)
            except JointControlFailure as error:
                status = f"reference_unavailable:{error}"
                break
            minimum_external = min(minimum_external, info["min_external_mesh_distance_m"])
            minimum_self = min(minimum_self, info["min_self_mesh_distance_m"])
            if terminated or truncated:
                if info["collision"]:
                    status = "collision"
                elif info["joint_limit"]:
                    status = "joint_limit"
                elif info["safe_success"]:
                    status = "success"
                elif info["local_timeout"]:
                    status = "local_timeout"
                elif truncated:
                    status = "global_timeout"
                else:
                    status = "terminated"
                break
    position, quaternion = env.robot.end_effector_pose()
    # Use the same explicit pose-error routine as the live environment.
    from rl_risk_sac.tasks.thesis_reaching import pose_error
    position_error, orientation_error = pose_error(env.goal_position, env.goal_quaternion, position, quaternion)
    return dict(
        status=status, success=status == "success", collision=status in ("collision", "initial_collision"),
        length=controller.steps, simulated_seconds=(env.physics_step_count - first_physics_step) * env.physics_dt,
        policy_steps=controller.policy_steps.copy(), avoidance_attempts=controller.attempts,
        verified_passages=controller.passages, reference_calls=controller.reference_calls,
        final_position_error_m=float(np.linalg.norm(position_error)),
        final_orientation_error_rad=float(np.linalg.norm(orientation_error)),
        min_external_mesh_distance_m=minimum_external if np.isfinite(minimum_external) else None,
        min_self_mesh_distance_m=minimum_self if np.isfinite(minimum_self) else None,
        switches=copy.deepcopy(controller.events))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="complete local step_XXXXXXX.pt, containing both actors")
    parser.add_argument("--bank", required=True, help="held-out validation/test scene bank, never training bank")
    parser.add_argument("--output", required=True, help="new JSON report path")
    parser.add_argument("--episodes", type=int, help="evaluate this many bank scenes (default: all)")
    parser.add_argument("--max-episode-steps", type=int, default=1000, help="total live steps, including every passage")
    parser.add_argument("--device", choices=("cpu", "cuda:auto"), default="cpu")
    parser.add_argument("--reaching-checkpoint", help="optional fine-tuned reaching policy; original checkpoint still defines scene provenance")
    args = parser.parse_args()
    if args.max_episode_steps < 1 or (args.episodes is not None and args.episodes < 1):
        parser.error("episode counts/budgets must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("protocol") not in (PROTOCOL, GEOMETRIC_PROTOCOL):
        parser.error("joint evaluation requires a complete local passage checkpoint")
    config = copy.deepcopy(state["config"])
    if state["protocol"] != local_protocol(config):
        raise ValueError("checkpoint reference mode/protocol mismatch")
    config["device"] = args.device
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    agent.load_state_dict(state["agent"], actor_only=True)
    for policy in (agent.reaching, agent.avoidance):
        policy.actor.eval().requires_grad_(False)
    source_reaching_digest = actor_digest(agent.reaching.actor)
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    validate_evaluation_bank(bank, state, agent.reaching)
    override = None
    if args.reaching_checkpoint:
        from rl_risk_sac.tasks.reaching_handoff import file_digest, load_reaching_override
        override = torch.load(args.reaching_checkpoint, map_location="cpu", weights_only=False)
        load_reaching_override(agent, override, file_digest(args.checkpoint),
                               {s["provenance"]["seed"] for s in bank["samples"]})
    digests = {role: actor_digest(agent.policy(role).actor) for role in ("reaching", "avoidance")}
    if args.episodes is not None and args.episodes > len(bank["samples"]):
        parser.error("requested episodes exceed held-out scenes; scenes are not repeated")
    samples = bank["samples"][:args.episodes]
    env = ThesisHomotopyEnv(simulation_config(config), cache_geometry_queries=True)
    outcomes = []
    started = time.perf_counter()
    try:
        controller = JointPolicyController(config, agent)
        try:
            for sample in samples:
                observation = restore_full_scene(env, sample, config)
                env.horizon = args.max_episode_steps
                outcome = evaluate_episode(env, controller, observation)
                outcomes.append(dict(sample_id=sample["sample_id"],
                                     source_seed=int(sample["provenance"]["seed"]),
                                     obstacle_count=len(sample["state"]["obstacles"]), **outcome))
                success_rate = np.mean([row["success"] for row in outcomes])
                print(f"joint {len(outcomes)}/{len(samples)}: {outcome['status']} "
                      f"steps={outcome['length']} passages={outcome['verified_passages']}/"
                      f"{outcome['avoidance_attempts']} success={success_rate:.1%}", flush=True)
        finally:
            controller.close()
    finally:
        env.close()
    unchanged = all(actor_digest(agent.policy(role).actor) == digest for role, digest in digests.items())
    attempts = sum(row["avoidance_attempts"] for row in outcomes)
    passages = sum(row["verified_passages"] for row in outcomes)
    task_decisions = sum(row["policy_steps"]["reaching"] for row in outcomes) + attempts
    control_steps = sum(row["length"] for row in outcomes)
    report = dict(
        protocol="joint_geometric_passage_v1" if geometric_mode(config) else "joint_reaching_local_passage_v1",
        checkpoint=str(Path(args.checkpoint).resolve()),
        bank=str(Path(args.bank).resolve()), scene_distribution="held_out_s0_blocking_scenes",
        start="full_source_episode_before_approach", reference_source=("instantaneous_s0_geometry" if geometric_mode(config)
                                                                    else "online_frozen_s0_rollout"),
        independent_of_training=True, training_checkpoint_validation_mode=state["validation_mode"],
        actors_unchanged=unchanged, samples=len(outcomes),
        success_rate=float(np.mean([row["success"] for row in outcomes])),
        collision_rate=float(np.mean([row["collision"] for row in outcomes])),
        reference_failure_rate=float(np.mean([row["status"].startswith("reference_unavailable:") for row in outcomes])),
        timeout_rate=float(np.mean([row["status"] in ("local_timeout", "global_timeout") for row in outcomes])),
        avoidance_attempts=attempts, verified_passages=passages,
        task_decisions=task_decisions,
        task_decision_adjustment_rate=attempts / task_decisions if task_decisions else None,
        avoidance_control_fraction=(sum(row["policy_steps"]["avoidance"] for row in outcomes) / control_steps
                                    if control_steps else None),
        passage_success_rate=passages / attempts if attempts else None,
        max_episode_steps=args.max_episode_steps, local_horizon=config["local_avoidance"]["horizon"],
        trigger_distance_m=config["local_avoidance"]["trigger_distance_m"],
        release_distance_m=config["local_avoidance"]["release_distance_m"],
        actor_sha256=digests, wall_seconds=time.perf_counter() - started, outcomes=outcomes)
    report.update(source_scene_reaching_actor_sha256=source_reaching_digest,
                  reaching_override=args.reaching_checkpoint,
                  reaching_obstacle_aware=bool(getattr(agent.reaching, "obstacle_aware", False)),
                  reaching_override_validation_mode=override["settings"]["validation_mode"] if override else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "outcomes"}), flush=True)


if __name__ == "__main__":
    main()
