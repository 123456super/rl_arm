from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_risk_sac.algorithms.networks import (
    GaussianActor,
    QNetwork,
    hard_update,
    soft_update,
)
from rl_risk_sac.algorithms.replay_buffer import Batch
from rl_risk_sac.utils.device import resolve_device


def diagonal_gaussian_kl_old_to_current(
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    current_mean: torch.Tensor,
    current_log_std: torch.Tensor,
) -> torch.Tensor:
    """KL(old || current), averaged over states and summed over actions.

    The tanh transformation is the same invertible map for both policies, so
    the KL of the squashed policies equals the KL of their base Gaussians.
    """
    old_variance = torch.exp(2.0 * old_log_std)
    current_variance = torch.exp(2.0 * current_log_std)
    per_dimension = (
        current_log_std - old_log_std
        + (old_variance + (old_mean - current_mean).square())
        / (2.0 * current_variance)
        - 0.5
    )
    return per_dimension.sum(dim=-1).mean()


class ThesisSACAgent:
    """Reward-only SAC for the single S0 -> S1 -> S2 serial protocol."""

    def __init__(self, obs_dim: int, action_dim: int, config: dict[str, Any]) -> None:
        sac = config["sac"]
        self.device = torch.device(resolve_device(config))
        hidden = list(sac.get("hidden_dims", [256, 256]))
        self.gamma = float(sac.get("gamma", 0.99))
        self.tau = float(sac.get("tau", 0.005))
        self.critic_gradient_clip = float(sac.get("critic_gradient_clip", 10.0))
        self.actor_gradient_clip = float(sac.get("actor_gradient_clip", 10.0))
        self.actor_update_interval = int(sac.get("actor_update_interval", 1))
        self.chain_pcr_enabled = bool(sac.get("chain_pcr_enabled", False))
        self.chain_pcr_coefficient = float(sac.get("chain_pcr_coefficient", 0.0))
        self.chain_pcr_max_loss_ratio = float(
            sac.get("chain_pcr_max_loss_ratio", float("inf"))
        )
        self.chain_pcr_auto_enabled = bool(
            sac.get("chain_pcr_auto_enabled", False)
        )
        self.chain_pcr_target_ratio = float(
            sac.get("chain_pcr_target_ratio", 0.003)
        )
        self.chain_pcr_ema_decay = float(sac.get("chain_pcr_ema_decay", 0.99))
        self.chain_pcr_auto_min_coefficient = float(
            sac.get("chain_pcr_auto_min_coefficient", 1.0e-4)
        )
        self.chain_pcr_auto_max_coefficient = float(
            sac.get("chain_pcr_auto_max_coefficient", 100.0)
        )
        if self.actor_update_interval < 1:
            raise ValueError("actor_update_interval must be positive")
        if self.chain_pcr_coefficient < 0.0:
            raise ValueError("chain_pcr_coefficient must be non-negative")
        if self.chain_pcr_enabled and self.chain_pcr_coefficient == 0.0:
            raise ValueError("CHAIN-PCR requires a positive coefficient")
        if self.chain_pcr_max_loss_ratio <= 0.0:
            raise ValueError("chain_pcr_max_loss_ratio must be positive")
        if not 0.0 < self.chain_pcr_target_ratio < 1.0:
            raise ValueError("chain_pcr_target_ratio must be in (0, 1)")
        if not 0.0 <= self.chain_pcr_ema_decay < 1.0:
            raise ValueError("chain_pcr_ema_decay must be in [0, 1)")
        if not 0.0 < self.chain_pcr_auto_min_coefficient <= self.chain_pcr_auto_max_coefficient:
            raise ValueError("invalid Auto-PCR coefficient bounds")
        self.chain_pcr_sac_loss_ema = 0.0
        self.chain_pcr_loss_ema = 0.0
        self.chain_pcr_sac_ema_initialized = False
        self.chain_pcr_loss_ema_initialized = False
        self.update_steps = 0
        self.last_actor_metrics = {
            "actor_loss": 0.0,
            "actor_sac_loss": 0.0,
            "policy_churn_kl": 0.0,
            "policy_churn_penalty": 0.0,
            "policy_churn_to_sac_ratio": 0.0,
            "chain_pcr_effective_coefficient": 0.0,
            "chain_pcr_sac_loss_ema": 0.0,
            "chain_pcr_loss_ema": 0.0,
            "alpha_loss": 0.0,
        }
        self.target_entropy = -float(action_dim)
        self.actor = GaussianActor(obs_dim, action_dim, hidden).to(self.device)
        self.chain_reference_actor = (
            self.frozen_actor_copy() if self.chain_pcr_enabled else None
        )
        self.q1 = QNetwork(obs_dim, action_dim, hidden).to(self.device)
        self.q2 = QNetwork(obs_dim, action_dim, hidden).to(self.device)
        self.target_q1 = QNetwork(obs_dim, action_dim, hidden).to(self.device)
        self.target_q2 = QNetwork(obs_dim, action_dim, hidden).to(self.device)
        hard_update(self.q1, self.target_q1)
        hard_update(self.q2, self.target_q2)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(sac["actor_lr"])
        )
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=float(sac["critic_lr"]),
        )
        self.log_alpha = torch.tensor(
            np.log(float(sac.get("initial_alpha", 0.2))),
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=float(sac["alpha_lr"])
        )

    def frozen_actor_copy(self) -> nn.Module:
        reference = copy.deepcopy(self.actor).to(self.device)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        return reference

    def reset_chain_reference_actor(self) -> None:
        """Align PCR's frozen reference with the currently loaded actor."""
        if self.chain_reference_actor is not None:
            hard_update(self.actor, self.chain_reference_actor)
        self.chain_pcr_sac_loss_ema = 0.0
        self.chain_pcr_loss_ema = 0.0
        self.chain_pcr_sac_ema_initialized = False
        self.chain_pcr_loss_ema_initialized = False

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(
        self, observation: np.ndarray, deterministic: bool = False,
    ) -> np.ndarray:
        return self.select_actions(
            np.asarray(observation, dtype=np.float32)[None, :],
            deterministic=deterministic,
        )[0]

    def select_actions(
        self, observations: np.ndarray, deterministic: bool = False,
    ) -> np.ndarray:
        obs = torch.as_tensor(
            observations, device=self.device, dtype=torch.float32
        )
        if obs.ndim != 2:
            raise ValueError("observations must have shape (batch, observation_dim)")
        with torch.no_grad():
            action = (
                self.actor.deterministic(obs)
                if deterministic
                else self.actor.sample(obs)[0]
            )
        return action.cpu().numpy()

    def update(
        self,
        batch: Batch,
        *,
        update_actor: bool = True,
        reference_observations: torch.Tensor | None = None,
        collect_diagnostics: bool = True,
    ) -> dict[str, float]:
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(batch.next_observations)
            next_q = torch.min(
                self.target_q1(batch.next_observations, next_action),
                self.target_q2(batch.next_observations, next_action),
            )
            target = batch.rewards + self.gamma * (1.0 - batch.dones) * (
                next_q - self.alpha.detach() * next_log_prob
            )

        q1_prediction = self.q1(batch.observations, batch.actions)
        q2_prediction = self.q2(batch.observations, batch.actions)
        q_loss = F.smooth_l1_loss(
            q1_prediction, target
        ) + F.smooth_l1_loss(q2_prediction, target)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.critic_gradient_clip,
        )
        self.q_optimizer.step()

        self.update_steps += 1
        actor_updated = bool(
            update_actor and self.update_steps % self.actor_update_interval == 0
        )
        actor_gradient_norm = 0.0
        if actor_updated:
            if self.chain_pcr_enabled and reference_observations is None:
                raise ValueError("CHAIN-PCR actor update requires a reference batch")
            action, log_prob = self.actor.sample(batch.observations)
            q1 = self.q1(batch.observations, action)
            q2 = self.q2(batch.observations, action)
            actor_sac_loss = (
                self.alpha.detach() * log_prob - torch.min(q1, q2)
            ).mean()
            policy_churn_kl = torch.zeros((), device=self.device)
            pre_update_actor: dict[str, torch.Tensor] | None = None
            if self.chain_pcr_enabled:
                assert self.chain_reference_actor is not None
                reference_observations = reference_observations.to(self.device)
                with torch.no_grad():
                    old_mean, old_log_std = self.chain_reference_actor(
                        reference_observations
                    )
                current_mean, current_log_std = self.actor(reference_observations)
                policy_churn_kl = diagonal_gaussian_kl_old_to_current(
                    old_mean, old_log_std, current_mean, current_log_std,
                )
                pre_update_actor = {
                    key: value.detach().clone()
                    for key, value in self.actor.state_dict().items()
                }
            effective_pcr_coefficient = torch.as_tensor(
                self.chain_pcr_coefficient,
                dtype=actor_sac_loss.dtype,
                device=self.device,
            )
            sac_magnitude = float(actor_sac_loss.detach().abs().cpu())
            pcr_magnitude = float(policy_churn_kl.detach().abs().cpu())
            decay = self.chain_pcr_ema_decay
            if not self.chain_pcr_sac_ema_initialized:
                self.chain_pcr_sac_loss_ema = sac_magnitude
                self.chain_pcr_sac_ema_initialized = True
            else:
                self.chain_pcr_sac_loss_ema = (
                    decay * self.chain_pcr_sac_loss_ema
                    + (1.0 - decay) * sac_magnitude
                )
            if pcr_magnitude > 1.0e-12:
                if not self.chain_pcr_loss_ema_initialized:
                    self.chain_pcr_loss_ema = pcr_magnitude
                    self.chain_pcr_loss_ema_initialized = True
                else:
                    self.chain_pcr_loss_ema = (
                        decay * self.chain_pcr_loss_ema
                        + (1.0 - decay) * pcr_magnitude
                    )
            if self.chain_pcr_auto_enabled and self.chain_pcr_loss_ema_initialized:
                automatic_coefficient = (
                    self.chain_pcr_target_ratio
                    * self.chain_pcr_sac_loss_ema
                    / (self.chain_pcr_loss_ema + 1.0e-12)
                )
                automatic_coefficient = float(np.clip(
                    automatic_coefficient,
                    self.chain_pcr_auto_min_coefficient,
                    self.chain_pcr_auto_max_coefficient,
                ))
                effective_pcr_coefficient = torch.as_tensor(
                    automatic_coefficient,
                    dtype=actor_sac_loss.dtype,
                    device=self.device,
                )
            if np.isfinite(self.chain_pcr_max_loss_ratio):
                maximum_coefficient = (
                    self.chain_pcr_max_loss_ratio
                    * actor_sac_loss.detach().abs().clamp_min(1.0e-8)
                    / policy_churn_kl.detach().clamp_min(1.0e-12)
                )
                effective_pcr_coefficient = torch.minimum(
                    effective_pcr_coefficient, maximum_coefficient,
                )
            policy_churn_penalty = effective_pcr_coefficient * policy_churn_kl
            actor_loss = actor_sac_loss + policy_churn_penalty
            policy_churn_to_sac_ratio = (
                policy_churn_penalty.detach()
                / actor_sac_loss.detach().abs().clamp_min(1.0e-8)
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.actor_gradient_clip
            )
            self.actor_optimizer.step()
            if pre_update_actor is not None:
                assert self.chain_reference_actor is not None
                self.chain_reference_actor.load_state_dict(pre_update_actor)

            alpha_loss = -(
                self.log_alpha * (log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            if collect_diagnostics:
                self.last_actor_metrics = {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "actor_sac_loss": float(actor_sac_loss.detach().cpu()),
                    "policy_churn_kl": float(policy_churn_kl.detach().cpu()),
                    "policy_churn_penalty": float(policy_churn_penalty.detach().cpu()),
                    "policy_churn_to_sac_ratio": float(
                        policy_churn_to_sac_ratio.cpu()
                    ),
                    "chain_pcr_effective_coefficient": float(
                        effective_pcr_coefficient.detach().cpu()
                    ),
                    "chain_pcr_sac_loss_ema": self.chain_pcr_sac_loss_ema,
                    "chain_pcr_loss_ema": self.chain_pcr_loss_ema,
                    "alpha_loss": float(alpha_loss.detach().cpu()),
                }

        soft_update(self.q1, self.target_q1, self.tau)
        soft_update(self.q2, self.target_q2, self.tau)
        if not collect_diagnostics:
            return {}
        if not actor_updated:
            q1 = q1_prediction
            q2 = q2_prediction
        return {
            "critic_loss": float(q_loss.detach().cpu()),
            **self.last_actor_metrics,
            "alpha": float(self.alpha.detach().cpu()),
            "q1_mean": float(q1.detach().mean().cpu()),
            "q2_mean": float(q2.detach().mean().cpu()),
            "target_mean": float(target.detach().mean().cpu()),
            "critic_gradient_norm": float(critic_gradient_norm),
            "actor_gradient_norm": float(actor_gradient_norm),
            "actor_updated": float(actor_updated),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "update_steps": self.update_steps,
            "last_actor_metrics": self.last_actor_metrics,
            "chain_reference_actor": (
                None if self.chain_reference_actor is None
                else self.chain_reference_actor.state_dict()
            ),
            "chain_pcr_auto_state": {
                "sac_loss_ema": self.chain_pcr_sac_loss_ema,
                "loss_ema": self.chain_pcr_loss_ema,
                "sac_initialized": self.chain_pcr_sac_ema_initialized,
                "loss_initialized": self.chain_pcr_loss_ema_initialized,
            },
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any], *, actor_only: bool = False) -> None:
        self.actor.load_state_dict(state["actor"])
        if actor_only:
            self.reset_chain_reference_actor()
            return
        for key, module in (
            ("q1", self.q1),
            ("q2", self.q2),
            ("target_q1", self.target_q1),
            ("target_q2", self.target_q2),
        ):
            module.load_state_dict(state[key])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.update_steps = int(state["update_steps"])
        self.last_actor_metrics = {
            key: float(state.get("last_actor_metrics", {}).get(key, 0.0))
            for key in (
                "actor_loss", "actor_sac_loss", "policy_churn_kl",
                "policy_churn_penalty", "policy_churn_to_sac_ratio",
                "chain_pcr_effective_coefficient",
                "chain_pcr_sac_loss_ema", "chain_pcr_loss_ema",
                "alpha_loss",
            )
        }
        if self.chain_reference_actor is not None:
            reference_state = state.get("chain_reference_actor")
            if reference_state is None:
                self.reset_chain_reference_actor()
            else:
                self.chain_reference_actor.load_state_dict(reference_state)
        auto_state = state.get("chain_pcr_auto_state", {})
        self.chain_pcr_sac_loss_ema = float(auto_state.get("sac_loss_ema", 0.0))
        self.chain_pcr_loss_ema = float(auto_state.get("loss_ema", 0.0))
        self.chain_pcr_sac_ema_initialized = bool(
            auto_state.get("sac_initialized", False)
        )
        self.chain_pcr_loss_ema_initialized = bool(
            auto_state.get("loss_initialized", False)
        )
        for key, optimizer in (
            ("actor_optimizer", self.actor_optimizer),
            ("q_optimizer", self.q_optimizer),
            ("alpha_optimizer", self.alpha_optimizer),
        ):
            optimizer.load_state_dict(state[key])
            for values in optimizer.state.values():
                for name, value in values.items():
                    if torch.is_tensor(value):
                        values[name] = value.to(self.device)

    def save_actor(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path)
