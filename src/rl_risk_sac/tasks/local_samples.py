"""Capture frozen-S0 starts; use planning only for offline feasibility filtering."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter

import numpy as np
import pybullet as p
import torch

from rl_risk_sac.envs.obstacle_scene import sample_obstacle, create_body
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv, ResetSamplingError
from rl_risk_sac.tasks.local_avoidance import (
    PROTOCOL, MeshClearance, actor_digest, arm_vector, contract_digest,
    obstacle_count_probabilities,
    geometric_mode, local_protocol,
)


def simulation_config(config):
    result = copy.deepcopy(config)
    # The generator itself places its static obstacles. No RRT sampler.
    obstacles = result["thesis"]["obstacles"]
    obstacles["path_sampling"]["enabled"] = False
    obstacles["count_probabilities"] = [1., 0., 0., 0.]
    obstacles["dynamic"] = False
    return result


def nominal_episode_contract(config):
    """The shared S0 goal distribution, also usable by parallel reset workers."""
    index = config["local_avoidance"]["nominal_level"]
    level = config["thesis"]["joint_pose_curriculum"]["levels"][index]
    return dict(
        scene="none", xi=1., strict=True, goal_scale=level["goal_scale"],
        orientation_scale=config["thesis"]["orientation_curriculum"]["levels"][index],
        position_tolerance=level["position_tolerance_m"],
        orientation_tolerance=level["orientation_tolerance_rad"], lambda_self=1.,
        **{key: level[key] for key in ("target_distance_min_m", "target_distance_max_m",
                                      "target_orientation_min_rad", "target_orientation_max_rad")})


def configure_nominal_episode(env, config):
    """Use the same goal distribution for source traces and full joint evaluation."""
    env.configure_episode(**nominal_episode_contract(config))


class ReachingSampleGenerator:
    def __init__(self, config, reaching, seed, *, verify_paths=True):
        self.config, self.reaching = config, reaching
        self.settings = config["local_avoidance"]
        self.rng = np.random.default_rng(seed)
        self.env = ThesisHomotopyEnv(simulation_config(config))
        self.rejections = Counter()
        self.policy_sha256 = actor_digest(reaching.actor)
        self.geometry = None
        if geometric_mode(config):
            from rl_risk_sac.tasks.geometric_passage import GeometricPassageReference
            self.geometry = GeometricPassageReference(config, reaching)
        self.path_verifier = None
        if verify_paths:
            from rl_risk_sac.tasks.local_path_verification import LocalPathVerifier
            self.path_verifier = LocalPathVerifier(config, reaching)

    def close(self):
        self.env.close()
        if self.geometry is not None:
            self.geometry.close()
        if self.path_verifier is not None:
            self.path_verifier.close()

    def _verify_sample(self, sample, goal):
        if self.path_verifier is None:
            return sample
        record = self.path_verifier.verify(sample, goal)
        if record["status"] != "verified":
            self.rejections["path_" + record["status"]] += 1
            return None
        sample["path_verification"] = record
        return sample

    def _capture_geometric(self, states, seed, nominal_min_distance):
        """Use the deployed instantaneous trigger during the physical S0 approach."""
        env = self.env
        initial = copy.deepcopy(states[0])
        initial["obstacles"] = [obj.metadata() for obj in env.obstacles]
        observation = env.restore_episode_state(initial)
        margins = MeshClearance(env)
        safe = self.config["thesis"]["obstacles"]["risk"]["d_safe"]
        for step in range(len(states)):
            intent = self.geometry.prepare(env, observation)
            current = min(env._static_clearance(obj) for obj in env.obstacles)
            if min(current, intent.next_clearance) <= self.settings["trigger_distance_m"]:
                reference = self.geometry.build(env, intent=intent)
                if reference.status == "blocked":
                    if step < self.settings["minimum_approach_steps"] or margins.external < safe or margins.collision:
                        self.rejections["geometric_approach_unsafe_or_short"] += 1
                        return None
                    captured = copy.deepcopy(env.episode_state_dict())
                    captured["step_count"] = captured["success_hold_count"] = 0
                    captured["contract"].terminate_on_success = False
                    sample = dict(reference.sample, state=captured,
                        provenance=dict(source="frozen_s0_policy", policy_sha256=self.policy_sha256,
                            seed=int(seed), approach_steps=step, nominal_min_distance_m=float(nominal_min_distance),
                            approach_min_distance_m=margins.external, physics_substep_checks=margins.checks,
                            obstacle_count=len(env.obstacles), planner_used=False,
                            reaching_actions_used_as_demonstrations=False,
                            reference_source="instantaneous_s0_geometry"))
                    sample["sample_id"] = hashlib.sha256(json.dumps({"seed": int(seed),
                        "q": captured["q"].tolist(), "goal": captured["goal_position"].tolist(),
                        "obstacles": captured["obstacles"]}, sort_keys=True).encode()).hexdigest()
                    return sample
            observation, _, _, terminated, truncated, info = env.step(intent.action, substep_observer=margins)
            if terminated or truncated or info["joint_limit"] or margins.collision or margins.external < safe:
                break
        self.rejections["geometric_no_safe_trigger"] += 1
        return None

    def _add_context_obstacles(self, count, vectors, index, states):
        """Add nearby or background objects, all checked against start/goal.

        At least the first proposed obstacle must obstruct the nominal trace.
        Additional objects may also obstruct it; every object enters the set
        encoder and all physical safety queries regardless of its role.
        """
        env = self.env
        settings = self.config["thesis"]["obstacles"]
        safe = float(settings["risk"]["d_safe"])
        for identity in range(1, count):
            for _ in range(16):
                center = None
                if self.rng.random() < .5:
                    point = vectors[index].reshape(-1, 3)[int(self.rng.integers(8))]
                    center = point + self.rng.normal(0., .12, 3)
                obstacle = sample_obstacle(
                    self.rng, identity, settings["workspace_low"], settings["workspace_high"],
                    center=center, motion_modes=["constant"])
                create_body(obstacle, env.client_id)
                overlapping = any(p.getClosestPoints(
                    obstacle.body_id, other.body_id, .005, physicsClientId=env.client_id)
                    for other in env.obstacles)
                env._set_joint_state(states[0]["q"], states[0]["qdot"])
                start_clear = env._static_clearance(obstacle)
                env._set_joint_state(states[-1]["q"], states[-1]["qdot"])
                goal_clear = env._static_clearance(obstacle)
                if (overlapping or start_clear < safe
                        or goal_clear < self.settings["release_distance_m"]):
                    p.removeBody(obstacle.body_id, physicsClientId=env.client_id)
                    self.rejections["context_obstacle_proposal"] += 1
                    continue
                env.obstacles.append(obstacle)
                break
            else:
                return False
        return True

    def _nominal_trace(self, seed):
        env = self.env
        configure_nominal_episode(env, self.config)
        observation, _ = env.reset(seed=seed)
        states, vectors = [copy.deepcopy(env.episode_state_dict())], [arm_vector(env)]
        for _ in range(self.settings["nominal_max_steps"]):
            action = self.reaching.select_action(observation, deterministic=True)
            observation, _, _, terminated, truncated, info = env.step(action)
            if info["collision"] or info["joint_limit"]:
                self.rejections["nominal_unsafe"] += 1
                return None
            states.append(copy.deepcopy(env.episode_state_dict()))
            vectors.append(arm_vector(env))
            if terminated or truncated:
                if not info["safe_success"]:
                    self.rejections["nominal_not_reached"] += 1
                    return None
                return states, np.asarray(vectors)
        self.rejections["nominal_budget"] += 1
        return None

    def sample(self, seed):
        trace = self._nominal_trace(seed)
        if trace is None:
            return None
        states, vectors = trace
        env, local = self.env, self.settings
        safe = self.config["thesis"]["obstacles"]["risk"]["d_safe"]
        minimum_trace = (local["minimum_approach_steps"] + 2 if self.geometry is not None
                         else local["recovery_steps"] + 5)
        if len(states) < minimum_trace:
            self.rejections["trace_too_short"] += 1
            return None
        probabilities = obstacle_count_probabilities(self.config)
        # Retain the original single-obstacle RNG sequence when the optional
        # count distribution is absent, including an already running bank.
        count = (int(self.rng.choice(len(probabilities), p=probabilities))
                 if "obstacle_count_probabilities" in local else 1)
        obstacle_config = self.config["thesis"]["obstacles"]
        for _ in range(local["obstacle_proposals_per_trace"]):
            env.restore_episode_state(states[0])
            first = max(2, len(states) // 5)
            last = min(len(states) - 2, max(first + 1, 4 * len(states) // 5))
            index = int(self.rng.integers(first, last))
            # One of the eight moving-arm capsule endpoints from the S0 trace.
            movement = np.linalg.norm((vectors[index] - vectors[0]).reshape(-1, 3), axis=1)
            if movement.max() < .01:
                self.rejections["nominal_arm_motion_too_small"] += 1
                continue
            point = int(self.rng.choice(8, p=movement / movement.sum()))
            anchor = vectors[index].reshape(-1, 3)[point]
            obstacle = sample_obstacle(
                self.rng, 0, obstacle_config["workspace_low"], obstacle_config["workspace_high"],
                center=anchor + self.rng.normal(0., local["obstacle_offset_std_m"], 3),
                motion_modes=["constant"])
            create_body(obstacle, env.client_id)
            env.obstacles.append(obstacle)
            # Reject unsafe endpoints before scanning the full trajectory.
            env._set_joint_state(states[0]["q"], states[0]["qdot"])
            first_distance = env._static_clearance(obstacle)
            env._set_joint_state(states[-1]["q"], states[-1]["qdot"])
            last_distance = env._static_clearance(obstacle)
            if first_distance < safe or last_distance < local["release_distance_m"]:
                self.rejections["unsafe_endpoints"] += 1
                continue
            if not self._add_context_obstacles(count, vectors, index, states):
                self.rejections["context_obstacle_budget"] += 1
                continue
            distances, primary_distances = [], []
            for state in states:
                env._set_joint_state(state["q"], state["qdot"])
                row = [env._static_clearance(obj) for obj in env.obstacles]
                distances.append(min(row))
                primary_distances.append(row[0])
            distances = np.asarray(distances)
            collisions = np.flatnonzero(distances <= 0.)
            if not len(collisions) or min(primary_distances) > 0.:
                self.rejections["nominal_not_blocked"] += 1
                continue
            collision = int(collisions[0])
            if self.geometry is not None:
                sample = self._capture_geometric(states, seed, distances.min())
                if sample is not None:
                    sample = self._verify_sample(sample, states[-1]["q"])
                    if sample is not None:
                        return sample
                continue
            before = np.flatnonzero(distances[:collision] >= local["trigger_distance_m"])
            if not len(before) or before[-1] < local["minimum_approach_steps"]:
                self.rejections["no_policy_approach"] += 1
                continue
            entry = int(before[-1])
            unsafe = np.flatnonzero(distances < safe)
            after = np.flatnonzero((np.arange(len(states)) > unsafe[-1])
                                   & (distances >= local["release_distance_m"]))
            if not len(after):
                self.rejections["no_local_exit"] += 1
                continue
            exit_index = int(after[0])
            if exit_index + local["recovery_steps"] >= len(states):
                self.rejections["no_continuation_budget"] += 1
                continue
            delta = vectors[exit_index] - vectors[entry]
            length = float(np.linalg.norm(delta))
            if length < local["minimum_reference_motion_m"]:
                self.rejections["reference_motion_too_small"] += 1
                continue

            # Replay the original actor physically WITH the obstacle to obtain
            # the actual pre-obstruction start, including nonzero joint speed.
            initial = copy.deepcopy(states[0])
            initial["obstacles"] = [obj.metadata() for obj in env.obstacles]
            observation = env.restore_episode_state(initial)
            approach = MeshClearance(env)
            invalid = False
            for _step in range(entry):
                action = self.reaching.select_action(observation, deterministic=True)
                observation, _, _, terminated, truncated, info = env.step(action, substep_observer=approach)
                if approach.external < safe or approach.collision or terminated or truncated:
                    invalid = True
                    break
            if invalid:
                self.rejections["approach_unsafe"] += 1
                continue
            captured = copy.deepcopy(env.episode_state_dict())
            current_arm = arm_vector(env)
            # Projection direction remains derived from the original S0 trace.
            # Replay discrepancies are recorded and bounded, never teleported.
            error = float(np.linalg.norm(current_arm - vectors[entry]))
            if error > .005 or min(env._static_clearance(obj) for obj in env.obstacles) < safe:
                self.rejections["approach_state_mismatch"] += 1
                continue
            delta = vectors[exit_index] - current_arm
            length = float(np.linalg.norm(delta))
            if length < local["minimum_reference_motion_m"]:
                self.rejections["captured_reference_too_small"] += 1
                continue
            captured["step_count"] = 0
            captured["success_hold_count"] = 0
            captured["contract"].terminate_on_success = False
            sample = dict(
                state=captured, entry_arm=current_arm.copy(),
                direction=delta / length, length=length,
                reference_exit_arm=vectors[exit_index].copy(),
                provenance=dict(source="frozen_s0_policy", policy_sha256=self.policy_sha256,
                                seed=int(seed), approach_steps=entry,
                                nominal_collision_step=collision, nominal_exit_step=exit_index,
                                nominal_min_distance_m=float(distances.min()),
                                nominal_entry_distance_m=float(distances[entry]),
                                approach_min_distance_m=approach.external,
                                approach_state_error_m=error,
                                physics_substep_checks=approach.checks,
                                obstacle_count=count,
                                planner_used=False,
                                reaching_actions_used_as_demonstrations=False))
            sample["sample_id"] = hashlib.sha256(
                json.dumps({"seed": int(seed), "q": captured["q"].tolist(),
                            "goal": captured["goal_position"].tolist(),
                            "obstacles": captured["obstacles"]}, sort_keys=True).encode()).hexdigest()
            sample = self._verify_sample(sample, states[-1]["q"])
            if sample is not None:
                return sample
        self.rejections["obstacle_budget"] += 1
        return None


def validate_bank(bank, config, reaching, *, split=None, require_path_verification=False):
    if bank.get("protocol") != local_protocol(config) or bank.get("format_version") != 1:
        raise ValueError("unsupported local sample bank; global-reaching replay is not a local bank")
    if bank.get("contract_sha256") != contract_digest(config):
        raise ValueError("local bank geometry/observation/task contract differs")
    if bank.get("policy_sha256") != actor_digest(reaching.actor):
        raise ValueError("local bank must come from this frozen reaching actor")
    if split is not None and bank.get("split") not in split:
        raise ValueError(f"local bank split must be one of {split}")
    if not bank.get("samples"):
        raise ValueError("local bank contains no samples")
    ids = set()
    task_digest = bank["contract_sha256"]
    probabilities = obstacle_count_probabilities(config)
    for sample in bank["samples"]:
        provenance = sample["provenance"]
        if (provenance.get("source") != "frozen_s0_policy"
                or provenance.get("policy_sha256") != bank["policy_sha256"]
                or provenance.get("planner_used", False) is not False
                or provenance.get("reaching_actions_used_as_demonstrations") is not False
                or provenance["nominal_min_distance_m"] > 0
                or provenance["approach_steps"] < config["local_avoidance"]["minimum_approach_steps"]
                or (not geometric_mode(config) and not provenance["approach_steps"]
                    < provenance["nominal_collision_step"] < provenance["nominal_exit_step"])):
            raise ValueError("invalid S0 sample provenance")
        direction, entry = np.asarray(sample["direction"]), np.asarray(sample["entry_arm"])
        if (direction.shape != (24,) or entry.shape != (24,)
                or not np.isfinite(direction).all() or not np.isfinite(entry).all()
                or not np.isclose(np.linalg.norm(direction), 1.)
                or not np.isfinite(sample["length"]) or sample["length"] < config["local_avoidance"]["minimum_reference_motion_m"]
                or not np.allclose(entry + sample["length"] * direction, sample["reference_exit_arm"])):
            raise ValueError("invalid actor-derived passage reference")
        if geometric_mode(config):
            from rl_risk_sac.tasks.geometric_passage import validate_geometric_reference
            validate_geometric_reference(sample, config)
            if (provenance.get("reference_source") != "instantaneous_s0_geometry"
                    or provenance.get("approach_min_distance_m", -1.) < config["thesis"]["obstacles"]["risk"]["d_safe"]):
                raise ValueError("geometric sample must come from a safe frozen-S0 approach")
        state = sample["state"]
        count = len(state["obstacles"])
        if (not 1 <= count < len(probabilities) or probabilities[count] <= 0
                or provenance.get("obstacle_count", count) != count
                or len({obj["identity"] for obj in state["obstacles"]}) != count
                or state["obstacle_contact_seen"]
                or state["step_count"] != 0 or state["contract"].terminate_on_success):
            raise ValueError("local samples require collision-free starts with an allowed obstacle count and unique identities")
        if sample["sample_id"] in ids:
            raise ValueError("duplicate local sample identity")
        ids.add(sample["sample_id"])
        if require_path_verification or "path_verification" in sample:
            from rl_risk_sac.tasks.local_path_verification import validate_path_certificate
            validate_path_certificate(sample, config, task_digest=task_digest)
    return ids


def relabel_geometric_sample(sample, env, guide):
    """Reuse a captured physical S0 start, never its old trajectory reference."""
    observation = env.restore_episode_state(copy.deepcopy(sample["state"]))
    margins = MeshClearance(env)
    if margins.collision or margins.external < guide.config["thesis"]["obstacles"]["risk"]["d_safe"]:
        return None, "unsafe_captured_start"
    intent = guide.prepare(env, observation)
    reference = guide.build(env, intent=intent)
    if reference.status != "blocked":
        return None, reference.status
    result = copy.deepcopy(sample)
    # Changing the reference invalidates any previous local-exit certificate.
    result.pop("path_verification", None)
    result.update(reference.sample)
    result["provenance"].update(reference_source="instantaneous_s0_geometry", relabelled_from_protocol=PROTOCOL)
    return result, "accepted"
