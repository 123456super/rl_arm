"""Distance-triggered composition of frozen reaching and local-passage actors.

All hypothetical motion uses separate Bullet clients. The live environment is
stepped exactly once per call, and is never reset at a policy handoff.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from rl_risk_sac.envs.local_avoidance_env import verify_reaching_recovery
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.local_avoidance import (
    MeshClearance, arm_vector, local_context, passage_candidate, passage_progress,
    validate_local_config,
    geometric_mode,
)
from rl_risk_sac.tasks.local_samples import simulation_config
from rl_risk_sac.tasks.obstacle_observation import ObstacleObservation, obstacle_set


@dataclass
class PassageReference:
    status: str
    sample: dict | None = None
    nominal_steps: int = 0


class JointControlFailure(RuntimeError):
    """The bounded nominal rollout cannot provide a valid trained context."""


class OnlinePassageReference:
    """Derive the trained 24D direction from an obstacle-free S0 continuation.

    Probe its configurations against *all* live obstacles in another client.
    The reference is a progress coordinate, not a collision-free path or an
    action sequence for the avoidance actor to imitate.
    """

    def __init__(self, config, reaching):
        validate_local_config(config)
        self.config, self.reaching = config, reaching
        self.nominal = ThesisHomotopyEnv(simulation_config(config),
                                        compute_external_features=False, cache_geometry_queries=True)
        try:
            self.probe = ThesisHomotopyEnv(simulation_config(config),
                                          compute_external_features=False, cache_geometry_queries=True)
        except BaseException:
            self.nominal.close()
            raise

    def build(self, base):
        snapshot = copy.deepcopy(base.episode_state_dict())
        self.probe.restore_episode_state(copy.deepcopy(snapshot))
        snapshot["obstacles"] = []
        snapshot["predictor_history"] = {}
        snapshot["step_count"] = snapshot["success_hold_count"] = 0
        snapshot["obstacle_contact_seen"] = False
        snapshot["contract"].terminate_on_success = True
        observation = self.nominal.restore_episode_state(snapshot)
        budget = self.config["local_avoidance"]["nominal_max_steps"]
        self.nominal.horizon = budget + 1
        configurations = [self.nominal.robot.joint_state_exact()]
        vectors = [arm_vector(self.nominal)]
        for _ in range(budget):
            action = self.reaching.select_action(observation, deterministic=True)
            observation, _, _, terminated, truncated, info = self.nominal.step(action)
            if info["collision"] or info["joint_limit"]:
                return PassageReference("nominal_unsafe", nominal_steps=len(vectors) - 1)
            configurations.append(self.nominal.robot.joint_state_exact())
            vectors.append(arm_vector(self.nominal))
            if terminated or truncated:
                if not info["safe_success"]:
                    return PassageReference("nominal_not_reached", nominal_steps=len(vectors) - 1)
                break
        else:
            return PassageReference("nominal_budget", nominal_steps=budget)

        distances = []
        for q, qdot in configurations:
            self.probe._set_joint_state(q, qdot)
            distances.append(min((self.probe._static_clearance(o) for o in self.probe.obstacles),
                                 default=float("inf")))
        return self.from_trace(np.asarray(vectors), np.asarray(distances))

    def next_clearance(self, base, observation):
        """Detect entry into the trigger region before executing the S0 step.

        Banks capture the last state outside 10 cm, not the first state inside
        it. A single physical-step rollout in the probe preserves that timing.
        """
        self.probe.restore_episode_state(copy.deepcopy(base.episode_state_dict()))
        margins = MeshClearance(self.probe)
        action = self.reaching.select_action(observation, deterministic=True)
        self.probe.step(action, substep_observer=margins)
        return margins.external

    def from_trace(self, vectors, distances):
        """Apply the bank's through-last-unsafe-region reference convention."""
        local = self.config["local_avoidance"]
        unsafe = np.flatnonzero(distances < self.config["thesis"]["obstacles"]["risk"]["d_safe"])
        steps = len(vectors) - 1
        if not len(unsafe):
            # Being within the trigger radius is not itself an obstruction.
            # This also prevents 6--10 cm handoff oscillation on a safe route.
            return PassageReference("clear", nominal_steps=steps)
        exits = np.flatnonzero((np.arange(len(vectors)) > unsafe[-1])
                              & (distances >= local["release_distance_m"]))
        if not len(exits):
            return PassageReference("no_local_exit", nominal_steps=steps)
        index = int(exits[0])
        if index + local["recovery_steps"] >= len(vectors):
            return PassageReference("no_continuation_budget", nominal_steps=steps)
        delta = vectors[index] - vectors[0]
        length = float(np.linalg.norm(delta))
        if length < local["minimum_reference_motion_m"]:
            return PassageReference("reference_motion_too_small", nominal_steps=steps)
        sample = dict(entry_arm=vectors[0].copy(), direction=delta / length, length=length,
                      reference_exit_arm=vectors[index].copy())
        return PassageReference("blocked", sample, steps)

    def close(self):
        self.nominal.close()
        self.probe.close()


class JointPolicyController:
    def __init__(self, config, agent):
        validate_local_config(config)
        self.config, self.local, self.agent = config, config["local_avoidance"], agent
        self.geometric = geometric_mode(config)
        if self.geometric:
            from rl_risk_sac.tasks.geometric_passage import GeometricPassageReference
            self.references = GeometricPassageReference(config, agent.reaching)
            self.checker = self.references
            self.reset()
            return
        self.references = OnlinePassageReference(config, agent.reaching)
        try:
            self.checker = ThesisHomotopyEnv(simulation_config(config),
                                            compute_external_features=False, cache_geometry_queries=True)
        except BaseException:
            self.references.close()
            raise
        self.reset()

    def reset(self):
        self.role, self.sample, self.local_steps = "reaching", None, 0
        self.steps = self.attempts = self.passages = self.reference_calls = 0
        self.policy_steps = {"reaching": 0, "avoidance": 0}
        self.events = []
        self.finished = False

    def _event(self, role, reason, distance, **fields):
        self.events.append(dict(step=self.steps, source=self.role, destination=role,
                                reason=reason, clearance_m=float(distance), **fields))
        self.role = role

    def step(self, base, observation):
        if self.finished:
            raise RuntimeError("reset the controller before another episode")
        distance = min((base._static_clearance(o) for o in base.obstacles), default=float("inf"))
        intent = (self.references.prepare(base, observation)
                  if self.geometric and self.role == "reaching" else None)
        approaching = False
        if self.role == "reaching" and base.obstacles:
            approaching = (distance <= self.local["trigger_distance_m"]
                           or (intent.next_clearance if intent is not None else
                               self.references.next_clearance(base, observation)) <= self.local["trigger_distance_m"])
        if approaching:
            self.reference_calls += 1
            reference = (self.references.build(base, intent=intent) if self.geometric else self.references.build(base))
            if reference.status == "blocked":
                self.sample, self.local_steps = reference.sample, 0
                self.attempts += 1
                self._event("avoidance", "local_motion_blocked" if self.geometric else "nominal_path_blocked", distance,
                            reference_length_m=self.sample["length"])
            elif reference.status != "clear":
                self.finished = True
                raise JointControlFailure(reference.status)

        action_role = self.role
        if action_role == "avoidance":
            context = local_context(arm_vector(base), self.sample, self.local_steps, self.local["horizon"])
            policy_observation = ObstacleObservation(np.r_[np.asarray(observation), context],
                                                      obstacle_set(observation))
        else:
            policy_observation = observation
        action = (intent.action if action_role == "reaching" and intent is not None else
                  self.agent.select_action(policy_observation, deterministic=True, policy=action_role))
        margins = MeshClearance(base)
        precision = base.precision_control_enabled
        if action_role == "avoidance":
            base.precision_control_enabled = False
        try:
            observation, _, _, terminated, truncated, info = base.step(action, substep_observer=margins)
        finally:
            base.precision_control_enabled = precision
        self.steps += 1
        self.policy_steps[action_role] += 1
        failure = bool(info["collision"] or info["joint_limit"] or margins.collision)
        success = bool(info["safe_success"] and not failure)
        local_timeout = False
        progress = None
        if action_role == "avoidance":
            self.local_steps += 1
            progress = passage_progress(arm_vector(base), self.sample)
            distance = min(base._static_clearance(o) for o in base.obstacles)
            if not (failure or success):
                candidate = passage_candidate(progress, distance, self.local)
                recovered = False
                if candidate:
                    recovered = (verify_reaching_recovery(base, self.checker, self.agent.reaching,
                                 self.sample, self.config, observation=observation) if self.geometric else
                                 verify_reaching_recovery(base, self.checker, self.agent.reaching, self.sample, self.config))
                if recovered:
                    self.passages += 1
                    self._event("reaching", "geometric_passage_and_local_check" if self.geometric else
                                "passage_and_recovery_verified", distance,
                                local_progress=progress, local_steps=self.local_steps)
                else:
                    local_timeout = self.local_steps >= self.local["horizon"]
        terminated = bool(terminated or failure)
        truncated = bool((truncated or local_timeout) and not terminated)
        self.finished = bool(terminated or truncated)
        return observation, terminated, truncated, dict(
            info, success=success, safe_success=success, collision=bool(info["collision"] or margins.collision),
            action_policy=action_role, next_policy=self.role, local_progress=progress,
            local_timeout=local_timeout, timeout=truncated,
            min_external_mesh_distance_m=margins.external, min_self_mesh_distance_m=margins.self_distance)

    def close(self):
        self.references.close()
        if self.checker is not self.references:
            self.checker.close()
