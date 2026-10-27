"""A finite-horizon passage subtask starting at a captured S0 state."""
from __future__ import annotations

import copy

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.local_samples import simulation_config
from rl_risk_sac.tasks.local_avoidance import (
    LOCAL_OBSERVATION_DIM, MeshClearance, arm_vector, local_context,
    local_reward, passage_progress, passage_candidate, validate_local_config,
    geometric_mode, local_protocol,
)
from rl_risk_sac.tasks.obstacle_observation import ObstacleObservation, obstacle_set


def verify_reaching_recovery(base, checker, reaching, sample, config, *, observation=None):
    """Read-only handoff test shared by training and the joint controller."""
    local = config["local_avoidance"]
    if geometric_mode(config):
        return checker.handoff(base, base._observe()[0] if observation is None else observation, sample)
    snapshot = copy.deepcopy(base.episode_state_dict())
    snapshot["step_count"] = 0
    snapshot["success_hold_count"] = 0
    snapshot["contract"].terminate_on_success = False
    observation = checker.restore_episode_state(snapshot)
    checker.horizon = local["recovery_steps"] + 1
    margins = MeshClearance(checker)
    safe = config["thesis"]["obstacles"]["risk"]["d_safe"]

    def unsafe():
        return margins.external < safe or margins.self_distance < checker.d_self_safe

    if unsafe():
        return False
    start = passage_progress(arm_vector(checker), sample)
    for _ in range(local["recovery_steps"]):
        action = reaching.select_action(observation, deterministic=True)
        observation, _, _, terminated, truncated, info = checker.step(action, substep_observer=margins)
        if terminated or truncated or margins.collision or info["joint_limit"] or unsafe():
            return False
    advance = passage_progress(arm_vector(checker), sample) - start
    return bool(margins.external >= safe and margins.self_distance >= checker.d_self_safe
                and advance >= local["recovery_progress"])


class LocalAvoidanceEnv(gym.Env):
    def __init__(self, config, reaching):
        validate_local_config(config)
        self.config, self.local, self.reaching = config, config["local_avoidance"], reaching
        self.base = ThesisHomotopyEnv(simulation_config(config), cache_geometry_queries=True)
        # A separate Bullet client ensures success lookahead never mutates the
        # live state, solver, velocities, clock, histories or controller.
        if geometric_mode(config):
            from rl_risk_sac.tasks.geometric_passage import GeometricPassageReference
            self.checker = GeometricPassageReference(config, reaching)
        else:
            self.checker = ThesisHomotopyEnv(simulation_config(config), compute_external_features=False,
                                           cache_geometry_queries=True)
        self.action_space = self.base.action_space
        self.observation_space = spaces.Box(-np.inf, np.inf, (LOCAL_OBSERVATION_DIM,), np.float32)
        self.sample = None
        self.steps = 0
        self.finished = True

    def _observation(self, original):
        context = local_context(arm_vector(self.base), self.sample, self.steps, self.local["horizon"])
        return ObstacleObservation(np.r_[np.asarray(original), context], obstacle_set(original))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options is None or "sample" not in options:
            raise ValueError("local reset requires a captured frozen-S0 sample")
        self.sample = copy.deepcopy(options["sample"])
        if geometric_mode(self.config):
            from rl_risk_sac.tasks.geometric_passage import validate_geometric_reference
            validate_geometric_reference(self.sample, self.config)
        original = self.base.restore_episode_state(self.sample["state"])
        self._original = original
        self.base.contract.terminate_on_success = False
        self.base.horizon = self.local["horizon"] + 1
        self.steps, self.finished = 0, False
        self.previous_progress = passage_progress(arm_vector(self.base), self.sample)
        margins = MeshClearance(self.base)
        if margins.collision:
            raise ValueError("captured local start is already colliding")
        return self._observation(original), {"sample_id": self.sample["sample_id"],
                                            "source": "frozen_s0_policy"}

    def _recovery(self):
        return verify_reaching_recovery(self.base, self.checker, self.reaching,
                                        self.sample, self.config, observation=self._original)

    def step(self, action):
        with self.base._geometry_query_scope():
            return self._step(action)

    def _step(self, action):
        if self.finished:
            raise RuntimeError("reset the local episode before stepping")
        margins = MeshClearance(self.base)
        # Local evasive motion must not inherit final-pose precision slowdown.
        precision = self.base.precision_control_enabled
        self.base.precision_control_enabled = False
        try:
            original, _, _, _, _, info = self.base.step(action, substep_observer=margins)
        finally:
            self.base.precision_control_enabled = precision
        self.steps += 1
        self._original = original
        progress = passage_progress(arm_vector(self.base), self.sample)
        clearance = min(self.base._static_clearance(o) for o in self.base.obstacles)
        failure = bool(info["collision"] or info["joint_limit"] or margins.collision)
        candidate = passage_candidate(progress, clearance, self.local)
        success = bool(not failure and candidate and self._recovery())
        terminated = bool(failure or success)
        truncated = bool(self.steps >= self.local["horizon"] and not terminated)
        reward = local_reward(self.previous_progress, progress, info["control_max_risk"],
                              info["control_self_max_risk"], info["smooth_velocity"],
                              success, failure, truncated, self.local["reward"])
        self.previous_progress = progress
        self.finished = bool(terminated or truncated)
        # Keep global pose measurements only as diagnostics. All success and
        # reward fields consumed by the local trainer are local-task fields.
        result = dict(sample_id=self.sample["sample_id"], success=success,
                      local_success=success, local_failure=failure, timeout=truncated,
                      local_progress=progress, passage_candidate=candidate,
                      recovery_passed=success, min_external_mesh_distance_m=margins.external,
                      min_self_mesh_distance_m=margins.self_distance,
                      current_external_mesh_distance_m=clearance,
                      physics_substep_checks=margins.checks,
                      position_error_m=info["next_rho_position"],
                      orientation_error_rad=info["next_rho_orientation"])
        return self._observation(original), reward, terminated, truncated, result

    def state_dict(self):
        return {"protocol": local_protocol(self.config), "sample": copy.deepcopy(self.sample), "steps": self.steps,
                "finished": self.finished, "previous_progress": self.previous_progress,
                "environment": self.base.episode_state_dict()}

    def load_state_dict(self, state):
        from rl_risk_sac.tasks.local_avoidance import PROTOCOL
        if state.get("protocol", PROTOCOL) != local_protocol(self.config):
            raise ValueError("local environment protocol differs")
        self.sample = copy.deepcopy(state["sample"])
        self.steps, self.finished = state["steps"], state["finished"]
        self.previous_progress = state["previous_progress"]
        original = self.base.restore_episode_state(state["environment"])
        self._original = original
        self.base.horizon = self.local["horizon"] + 1
        return self._observation(original)

    def close(self):
        self.base.close()
        self.checker.close()
