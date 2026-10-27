"""Offline, bounded whole-arm witnesses for static local-passage samples.

A miss is unconfirmed, never proof of infeasibility. The witness is geometric:
it does not certify tracking dynamics, initial velocity or the episode horizon.
"""
from __future__ import annotations

import copy
import hashlib
import json
from time import perf_counter

import numpy as np
import pybullet as p

from rl_risk_sac.tasks.static_scene_planning import joint_segment, rrt_connect, rrt_goal_region

VERIFICATION_PROTOCOL = "whole_arm_static_passage_witness_v1"
DEFAULT_SETTINGS = dict(joint_resolution_rad=.02, verification_resolution_rad=.005,
                        rrt_extension_rad=.15, rrt_iterations=400, rrt_max_checks=4000,
                        max_verification_checks=20000, max_exit_checks=32)


def verification_settings(config):
    settings = {**DEFAULT_SETTINGS, **config.get("local_path_verification", {})}
    if set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError("unknown local_path_verification setting")
    for key, value in settings.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"local_path_verification.{key} must be finite and positive")
        if key in ("rrt_iterations", "rrt_max_checks", "max_verification_checks", "max_exit_checks") and int(value) != value:
            raise ValueError(f"local_path_verification.{key} must be an integer")
    if (settings["rrt_max_checks"] < 2
            or settings["verification_resolution_rad"] > settings["joint_resolution_rad"]):
        raise ValueError("path verification requires at least two checks and a finer final resolution")
    for key in ("rrt_iterations", "rrt_max_checks", "max_verification_checks", "max_exit_checks"):
        settings[key] = int(settings[key])
    return settings


def _digest(value):
    def encode(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    return hashlib.sha256(json.dumps(value, default=encode, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def sample_path_digest(sample):
    """Bind a witness to the physical start, all obstacles and the local exit."""
    state = sample["state"]
    return _digest(dict(sample_id=sample["sample_id"],
        state={key: state[key] for key in ("q", "qdot", "goal_position", "goal_quaternion",
                                          "obstacles", "physics_step_count")},
        reference={key: sample.get(key) for key in ("entry_arm", "direction", "length", "geometry",
                                                   "reference_protocol", "reference_exit_arm")}))


def validate_path_certificate(sample, config, *, task_digest):
    """Cheap integrity checks at load time; never rerun planning in training."""
    record = sample.get("path_verification")
    if not record or record.get("protocol") != VERIFICATION_PROTOCOL or record.get("status") != "verified":
        raise ValueError("sample has no verified whole-arm path; run audit_local_avoidance_bank.py first")
    if (record.get("sample_sha256") != sample_path_digest(sample)
            or record.get("task_sha256") != task_digest
            or record.get("record_sha256") != _digest({k: v for k, v in record.items() if k != "record_sha256"})):
        raise ValueError("whole-arm path witness differs from the sample/task or was modified")
    route = np.asarray(record["waypoints"], dtype=float)
    exit_q = np.asarray(record["local_exit_q"], dtype=float)
    goal = np.asarray(record["goal_q"], dtype=float)
    local_route = np.asarray(record["local_waypoints"], dtype=float)
    settings = verification_settings(config)
    safe = config["thesis"]["obstacles"]["risk"]["d_safe"]
    if (route.ndim != 2 or route.shape[1:] != (6,) or len(route) < 2
            or not np.isfinite(route).all() or exit_q.shape != (6,) or not np.isfinite(exit_q).all()
            or goal.shape != (6,) or not np.isfinite(goal).all()
            or not np.allclose(route[0], sample["state"]["q"], rtol=0., atol=1e-8)
            or not np.allclose(route[-1], goal, rtol=0., atol=1e-8)
            or local_route.ndim != 2 or local_route.shape[1:] != (6,) or len(local_route) < 2
            or not np.isfinite(local_route).all()
            or not np.allclose(local_route[0], sample["state"]["q"], rtol=0., atol=1e-8)
            or not np.allclose(local_route[-1], exit_q, rtol=0., atol=1e-8)
            or record["obstacle_count"] != len(sample["state"]["obstacles"])
            or record["external_margin_m"] < safe
            or record["self_margin_m"] < config["thesis"]["self_collision"]["safe_distance_m"]
            or not 0 < record["verification_resolution_rad"] <= settings["verification_resolution_rad"]
            or record["local_exit_progress"] < 1.
            or record["local_exit_external_m"] < config["local_avoidance"]["release_distance_m"]
            or record.get("local_handoff_passed") is not True):
        raise ValueError("invalid whole-arm path/local-exit witness")


class LocalPathVerifier:
    """Independent Bullet clients; source scene, velocities and RNG stay intact."""
    def __init__(self, config, reaching):
        from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
        from rl_risk_sac.tasks.local_samples import simulation_config
        from rl_risk_sac.tasks.local_avoidance import contract_digest, geometric_mode
        self.config, self.reaching = config, reaching
        self.settings = verification_settings(config)
        self.task_digest = contract_digest(config)
        self.query = ThesisHomotopyEnv(simulation_config(config), compute_external_features=False,
                                      cache_geometry_queries=True)
        if geometric_mode(config):
            from rl_risk_sac.tasks.geometric_passage import GeometricPassageReference
            self.checker = GeometricPassageReference(config, reaching)
        else:
            self.checker = ThesisHomotopyEnv(simulation_config(config), compute_external_features=False,
                                           cache_geometry_queries=True)
        self.total_attempts = self.total_seconds = 0
        self.status_counts = {}

    def close(self):
        self.query.close()
        self.checker.close()

    def summary(self):
        return dict(attempts=self.total_attempts, statuses=dict(self.status_counts),
                    wall_seconds=self.total_seconds,
                    mean_seconds=self.total_seconds / self.total_attempts if self.total_attempts else None,
                    settings=self.settings, protocol=VERIFICATION_PROTOCOL)

    def _valid(self, q):
        env = self.query
        if (not np.isfinite(q).all() or np.any(q < env.robot.joint_lower_limits)
                or np.any(q > env.robot.joint_upper_limits)):
            return False
        env._set_joint_state(q)
        safe = self.config["thesis"]["obstacles"]["risk"]["d_safe"]
        for obj in env.obstacles:
            if any(point[8] < safe for point in p.getClosestPoints(
                    env.robot_id, obj.body_id, safe, physicsClientId=env.client_id)):
                return False
        for first, second in env._self_pair_link_ids:
            if any(point[8] < env.d_self_safe for point in p.getClosestPoints(
                    env.robot_id, env.robot_id, env.d_self_safe,
                    linkIndexA=first, linkIndexB=second, physicsClientId=env.client_id)):
                return False
        return True

    def verify(self, sample, goal_q):
        started = perf_counter()
        result = self._verify(sample, np.asarray(goal_q, dtype=float))
        result.update(wall_seconds=perf_counter() - started)
        self.total_attempts += 1
        self.total_seconds += result["wall_seconds"]
        self.status_counts[result["status"]] = self.status_counts.get(result["status"], 0) + 1
        if result["status"] == "verified":
            result["record_sha256"] = _digest(result)
        return result

    def _verify(self, sample, goal):
        from rl_risk_sac.tasks.local_avoidance import arm_vector, passage_progress, MeshClearance
        from rl_risk_sac.envs.local_avoidance_env import verify_reaching_recovery
        from rl_risk_sac.tasks.thesis_reaching import pose_error
        env, settings = self.query, self.settings
        env.restore_episode_state(copy.deepcopy(sample["state"]))
        start = np.asarray(sample["state"]["q"], dtype=float)
        result = dict(protocol=VERIFICATION_PROTOCOL, status="search_unconfirmed", global_path_verified=False,
                      search_settings=dict(settings),
                      sample_sha256=sample_path_digest(sample), task_sha256=self.task_digest,
                      obstacle_count=len(env.obstacles), goal_q=goal.tolist(), goal_source="successful_s0_endpoint",
                      search_checks=0, verification_checks=0, exit_checks=0,
                      external_margin_m=float(self.config["thesis"]["obstacles"]["risk"]["d_safe"]),
                      self_margin_m=env.d_self_safe,
                      verification_resolution_rad=settings["verification_resolution_rad"])
        if goal.shape != (6,) or start.shape != (6,):
            raise ValueError("path endpoints must contain six joint angles")
        if not self._valid(start):
            return dict(result, status="invalid_start")
        if not self._valid(goal):
            return dict(result, status="invalid_goal")
        position, quaternion = env.robot.end_effector_pose()
        position_error, orientation_error = pose_error(env.goal_position, env.goal_quaternion, position, quaternion)
        if (np.linalg.norm(position_error) > env.contract.position_tolerance
                or np.linalg.norm(orientation_error) > env.contract.orientation_tolerance):
            return dict(result, status="goal_pose_mismatch")
        if min(env._static_clearance(obj) for obj in env.obstacles) < self.config["local_avoidance"]["release_distance_m"]:
            return dict(result, status="goal_release_unconfirmed")
        # Separate, reproducible RNG: planning cannot change obstacle sampling.
        rng = np.random.default_rng(int(sample_path_digest(sample)[:16], 16))
        stats = {}
        path = rrt_connect(start, goal, env.robot.joint_lower_limits, env.robot.joint_upper_limits,
                           self._valid, rng, resolution=settings["joint_resolution_rad"],
                           extension=settings["rrt_extension_rad"], iterations=settings["rrt_iterations"],
                           max_checks=settings["rrt_max_checks"], statistics=stats)
        result.update(search_checks=stats["checks"], method=stats["method"])
        if path is None:
            return result
        result["waypoints"] = [q.tolist() for q in path]
        candidates = []
        max_progress = float("-inf")
        external_min, self_min = float("inf"), float("inf")
        for index, (a, b) in enumerate(zip(path, path[1:])):
            segment = joint_segment(a, b, settings["verification_resolution_rad"])
            for q in segment[int(index > 0):]:
                if result["verification_checks"] >= settings["max_verification_checks"]:
                    return dict(result, status="verification_budget_unconfirmed")
                result["verification_checks"] += 1
                if not self._valid(q):
                    return dict(result, status="fine_check_rejected")
                with env._geometry_query_scope():
                    margins = MeshClearance(env)
                external_min, self_min = min(external_min, margins.external), min(self_min, margins.self_distance)
                progress = passage_progress(arm_vector(env), sample)
                max_progress = max(max_progress, progress)
                if progress >= 1. and margins.external >= self.config["local_avoidance"]["release_distance_m"]:
                    candidates.append((q.copy(), progress, margins.external, index))
        result.update(global_path_verified=True, min_external_m=external_min, min_self_m=self_min,
                      global_path_max_progress=max_progress, global_path_exit_candidates=len(candidates))
        # Test a bounded, evenly distributed subset, including both ends.
        indices = np.unique(np.linspace(0, len(candidates) - 1,
                            min(len(candidates), settings["max_exit_checks"]), dtype=int))
        for index in indices:
            q, progress, clearance, segment_index = candidates[index]
            env._set_joint_state(q)  # Zero-speed kinematic witness, not a dynamic rollout.
            observation, info, *_ = env._observe()
            env.previous_rho_position = info["goal_error_norm"]
            env.previous_rho_orientation = info["orientation_error_norm"]
            result["exit_checks"] += 1
            if verify_reaching_recovery(env, self.checker, self.reaching, sample, self.config, observation=observation):
                return dict(result, status="verified", local_exit_q=q.tolist(),
                            local_waypoints=[v.tolist() for v in path[:segment_index + 1]] + [q.tolist()],
                            local_exit_progress=progress, local_exit_external_m=clearance,
                            local_handoff_passed=True, exit_velocity="zero_kinematic_witness")
        return self._search_local_exit(sample, start, goal, rng, result)

    def _search_local_exit(self, sample, start, goal, rng, result):
        """A global route need not cross the frozen plane; search its goal region."""
        from rl_risk_sac.tasks.local_avoidance import arm_vector, passage_progress, MeshClearance
        from rl_risk_sac.envs.local_avoidance_env import verify_reaching_recovery
        env, settings = self.query, self.settings
        remaining = settings["rrt_max_checks"] - result["search_checks"]
        if remaining < 2 or result["exit_checks"] >= settings["max_exit_checks"]:
            return dict(result, status="local_exit_unconfirmed")
        found = {}

        def exit_test(q):
            if result["exit_checks"] >= settings["max_exit_checks"]:
                return False
            env._set_joint_state(q)
            progress = passage_progress(arm_vector(env), sample)
            if progress < 1.:
                return False
            with env._geometry_query_scope():
                margins = MeshClearance(env)
                if margins.external < self.config["local_avoidance"]["release_distance_m"]:
                    return False
                observation, info, *_ = env._observe()
                env.previous_rho_position = info["goal_error_norm"]
                env.previous_rho_orientation = info["orientation_error_norm"]
                result["exit_checks"] += 1
                passed = verify_reaching_recovery(env, self.checker, self.reaching, sample,
                                                  self.config, observation=observation)
            if passed:
                found.update(local_exit_q=q.tolist(), local_exit_progress=progress,
                             local_exit_external_m=margins.external, local_handoff_passed=True,
                             exit_velocity="zero_kinematic_witness")
            return passed

        lower, upper = env.robot.joint_lower_limits, env.robot.joint_upper_limits

        def target():
            draw = rng.random()
            if draw < .2:
                return goal
            if draw < .6:
                return np.clip(start + rng.normal(0., .6, len(start)), lower, upper)
            return rng.uniform(lower, upper)

        statistics = {}
        path = rrt_goal_region(start, lower, upper, self._valid, exit_test, rng,
                              resolution=settings["joint_resolution_rad"], extension=settings["rrt_extension_rad"],
                              iterations=settings["rrt_iterations"], max_checks=remaining,
                              sample_target=target, statistics=statistics)
        result["search_checks"] += statistics["checks"]
        result["local_search_checks"] = statistics["checks"]
        if path is None:
            return dict(result, status="local_exit_unconfirmed")
        # Independently recheck every local edge at the fine resolution too.
        for index, (a, b) in enumerate(zip(path, path[1:])):
            for q in joint_segment(a, b, settings["verification_resolution_rad"])[int(index > 0):]:
                if result["verification_checks"] >= settings["max_verification_checks"]:
                    return dict(result, status="local_verification_budget_unconfirmed")
                result["verification_checks"] += 1
                if not self._valid(q):
                    return dict(result, status="local_fine_check_rejected")
        return dict(result, status="verified", local_waypoints=[q.tolist() for q in path], **found)


def nominal_goal_for_sample(generator, sample):
    """Recreate the successful obstacle-free endpoint for an old bank entry."""
    trace = generator._nominal_trace(int(sample["provenance"]["seed"]))
    if trace is None:
        return None
    states, _ = trace
    if any(not np.allclose(states[-1][key], sample["state"][key], rtol=0., atol=1e-6)
           for key in ("goal_position", "goal_quaternion")):
        raise ValueError("source seed does not reproduce this sample's goal")
    return states[-1]["q"].copy()
