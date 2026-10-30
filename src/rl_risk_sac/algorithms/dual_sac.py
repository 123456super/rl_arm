"""Two independent SACs, selected explicitly without an automatic controller."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rl_risk_sac.algorithms.replay_buffer import Batch
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.tasks.obstacle_observation import ObstacleBatch, ObstacleObservation, obstacle_set


POLICIES = ("reaching", "avoidance")


def validate_dual_sac_config(config: dict) -> None:
    settings = config.get("dual_sac", {})
    if not settings.get("enabled", False):
        return
    role = settings.get("training_policy")
    if role not in POLICIES:
        raise ValueError("dual_sac.training_policy must be reaching or avoidance")
    thesis = config["thesis"]
    obstacles = thesis.get("obstacles", {})
    if role == "reaching" and (thesis["stage"] != "s0" or obstacles.get("enabled", False)):
        raise ValueError("reaching SAC training requires the obstacle-free S0 stage")
    if role == "avoidance" and (thesis["stage"] == "s0" or not obstacles.get("enabled", False)):
        raise ValueError("avoidance SAC training requires an obstacle-enabled S1/S2/S3 stage")


def checkpoint_policy_state(checkpoint: dict, role: str | None = None) -> dict:
    """Select a named SAC from a full checkpoint, SAC state, or raw actor."""
    if role is not None and role not in POLICIES:
        raise ValueError(f"unknown SAC policy: {role!r}")
    state = checkpoint.get("agent", checkpoint)
    dual = state.get("dual_sac")
    if dual is None:
        return state
    if dual.get("format_version") != DualSACAgent.FORMAT_VERSION:
        raise ValueError("unsupported dual SAC checkpoint format")
    selected = role or dual["training_policy"]
    if set(dual.get("policies", {})) != set(POLICIES):
        raise ValueError("dual SAC checkpoint must contain both complete policies")
    return dual["policies"][selected]


class ReachingSACAgent(ThesisSACAgent):
    """Original S0 by default; explicit handoff fine-tuning can enable perception."""

    def __init__(self, obs_dim, action_dim, config):
        super().__init__(obs_dim, action_dim, config)
        self.set_obstacle_awareness(config.get("thesis", {}).get("observation", {}).get(
            "reaching_obstacle_aware", False))

    def set_obstacle_awareness(self, enabled):
        self.obstacle_aware = bool(enabled)
        # Actor-only CPU snapshots must use the same input contract as the learner.
        self.actor.reaching_obstacle_aware = self.obstacle_aware

    def initialize_obstacle_awareness(self, feature_scale):
        """Enable existing independent encoders without disturbing mature outputs.

        Legacy S0 never trained its encoders. Zero only their final linear
        projections, so all initial embeddings remain zero while gradients can
        learn obstacle effects. Retain all reaching backbone/critic weights.
        """
        if self.obstacle_aware:
            return False
        scale = torch.as_tensor(feature_scale, device=self.device, dtype=self.actor.obstacle_encoder.scale.dtype)
        if scale.shape != (37,) or not bool(torch.isfinite(scale).all() and (scale > 0).all()):
            raise ValueError("invalid reaching obstacle feature scale")
        for module in (self.actor, self.q1, self.q2, self.target_q1, self.target_q2):
            encoder = module.obstacle_encoder
            with torch.no_grad():
                encoder.scale.copy_(scale)
                encoder.fusion[-1].weight.zero_()
                encoder.fusion[-1].bias.zero_()
            for optimizer in (self.actor_optimizer, self.q_optimizer):
                for parameter in encoder.parameters():
                    optimizer.state.pop(parameter, None)
        self.set_obstacle_awareness(True)
        self.reset_chain_reference_actor()
        return True

    def select_actions(self, observations, deterministic=False):
        if self.obstacle_aware:
            if isinstance(observations, (list, tuple)):
                observations = [ObstacleObservation(np.asarray(o)[:self.actor_obs_dim], obstacle_set(o))
                                for o in observations]
            elif getattr(observations, "obstacle_sets", None) is not None:
                observations = ObstacleBatch(np.asarray(observations)[:, :self.actor_obs_dim], observations.obstacle_sets)
            else:
                observations = np.asarray(observations)[:, :self.actor_obs_dim]
            return super().select_actions(observations, deterministic)
        observations = np.stack([np.asarray(o)[:self.actor_obs_dim] for o in observations])
        return super().select_actions(observations, deterministic)

    def update(self, batch: Batch, **kwargs):
        if self.obstacle_aware:
            return super().update(batch, **kwargs)
        def without_obstacles(value):
            return replace(value, obstacle_features=None, obstacle_mask=None,
                           next_obstacle_features=None, next_obstacle_mask=None)
        batch = without_obstacles(batch)
        if isinstance(kwargs.get("reference_observations"), Batch):
            kwargs["reference_observations"] = without_obstacles(kwargs["reference_observations"])
        return super().update(batch, **kwargs)

    def state_dict(self):
        state = super().state_dict()
        if self.obstacle_aware:
            state["reaching_obstacle_aware"] = True
        return state

    def load_state_dict(self, state, *, actor_only=False):
        super().load_state_dict(state, actor_only=actor_only)
        # Old checkpoints always retain the original obstacle-blind behavior.
        self.set_obstacle_awareness(state.get("reaching_obstacle_aware", False))


class DualSACAgent:
    """Keep the inactive SAC frozen while the existing trainer updates one SAC.

    Both policies have their own actor, twin critics, target critics, set
    encoders, alpha and optimizers. Reaching retains 241D; legacy S0 ignores
    external sets, while an explicit fine-tuned checkpoint can enable them.
    Local avoidance adds observable passage context to the Full state;
    its dedicated environment/replay defines a separate local reward contract.
    The active run owns its replay/curriculum; there is no cross-policy replay
    transfer, action fusion, safety gate or automatic switching.
    """

    FORMAT_VERSION = 1

    def __init__(self, obs_dim: int, action_dim: int, config: dict[str, Any]):
        validate_dual_sac_config(config)
        from rl_risk_sac.tasks.local_avoidance import local_protocol
        self.task_protocol = local_protocol(config) if config.get("local_avoidance", {}).get("enabled", False) else None
        self.training_policy = config["dual_sac"]["training_policy"]
        reaching_config = copy.deepcopy(config)
        reaching_config["thesis"]["obstacles"]["enabled"] = False
        reaching_config["thesis"]["obstacles"]["dynamic"] = False
        reaching_dim = 241 if config.get("local_avoidance", {}).get("enabled", False) else obs_dim
        self.reaching = ReachingSACAgent(reaching_dim, action_dim, reaching_config)
        self.avoidance = ThesisSACAgent(obs_dim, action_dim, config)
        self._freeze_inactive()

    def policy(self, role: str) -> ThesisSACAgent:
        if role not in POLICIES:
            raise ValueError(f"unknown SAC policy: {role!r}")
        return object.__getattribute__(self, role)

    @property
    def active(self) -> ThesisSACAgent:
        return self.policy(self.training_policy)

    def __getattr__(self, name):
        # Keep the existing single-policy training/probe loop and metrics.
        if name in POLICIES or name == "training_policy":
            raise AttributeError(name)
        return getattr(self.active, name)

    def _freeze_inactive(self):
        inactive = self.policy(next(role for role in POLICIES if role != self.training_policy))
        for module in (inactive.actor, inactive.q1, inactive.q2,
                       inactive.target_q1, inactive.target_q2):
            module.eval()
            module.requires_grad_(False)
        inactive.log_alpha.requires_grad_(False)

    def select_action(self, observation, deterministic=False, *, policy=None):
        return self.select_actions([observation], deterministic, policy=policy)[0]

    def select_actions(self, observations, deterministic=False, *, policy=None):
        role = policy or self.training_policy
        selected = self.policy(role)
        return selected.select_actions(observations, deterministic)

    def update(self, batch: Batch, **kwargs):
        return self.active.update(batch, **kwargs)

    def state_dict(self):
        states = {role: self.policy(role).state_dict() for role in POLICIES}
        # Retain the active fields for existing actor-extraction tools. The
        # nested dictionary explicitly stores both complete independent SACs.
        return {**states[self.training_policy], "dual_sac": {
            "format_version": self.FORMAT_VERSION,
            "training_policy": self.training_policy,
            "task_protocol": self.task_protocol,
            "policies": states,
        }}

    def load_state_dict(self, state, *, actor_only=False):
        dual = state.get("dual_sac")
        if dual is None:
            raise ValueError("dual SAC resume requires both policies; initialize a policy explicitly from a single SAC")
        if dual.get("training_policy") != self.training_policy:
            raise ValueError("dual SAC training_policy differs; replay/curriculum cannot resume across policies")
        from rl_risk_sac.tasks.local_avoidance import PROTOCOL
        saved_protocol = dual.get("task_protocol", PROTOCOL if self.task_protocol is not None else None)
        if saved_protocol != self.task_protocol:
            raise ValueError("dual SAC local task protocol differs; use explicit actor initialization, never old replay")
        states = {role: checkpoint_policy_state(state, role) for role in POLICIES}
        for role in POLICIES:
            self.policy(role).load_state_dict(states[role], actor_only=actor_only)
        self._freeze_inactive()

    def save_actor(self, path: str | Path):
        path = Path(path)
        self.active.save_actor(path)
        for role in POLICIES:
            self.policy(role).save_actor(path.with_name(f"{role}_{path.name}"))


def build_thesis_sac(obs_dim: int, action_dim: int, config: dict):
    if config.get("dual_sac", {}).get("enabled", False):
        return DualSACAgent(obs_dim, action_dim, config)
    return ThesisSACAgent(obs_dim, action_dim, config)
