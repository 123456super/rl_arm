from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rl_risk_sac.algorithms.networks import GaussianActor, QNetwork, hard_update, soft_update
from rl_risk_sac.algorithms.replay_buffer import Batch
from rl_risk_sac.utils.device import resolve_device


class ThesisSACAgent:
    """Reward-only SAC used by the task-first safety-homotopy protocol."""

    def __init__(self, obs_dim: int, action_dim: int, config: dict[str, Any]) -> None:
        sac = config["sac"]
        self.device = torch.device(resolve_device(config))
        hidden = list(sac.get("hidden_dims", [256, 256]))
        self.gamma = float(sac.get("gamma", 0.99))
        self.tau = float(sac.get("tau", 0.005))
        self.target_entropy = -float(action_dim)
        self.actor = GaussianActor(obs_dim, action_dim, hidden).to(self.device)
        self.q1 = QNetwork(obs_dim, action_dim, hidden).to(self.device)
        self.q2 = QNetwork(obs_dim, action_dim, hidden).to(self.device)
        self.target_q1 = QNetwork(obs_dim, action_dim, hidden).to(self.device)
        self.target_q2 = QNetwork(obs_dim, action_dim, hidden).to(self.device)
        hard_update(self.q1, self.target_q1)
        hard_update(self.q2, self.target_q2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(sac["actor_lr"]))
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=float(sac["critic_lr"])
        )
        self.log_alpha = torch.tensor(
            np.log(float(sac.get("initial_alpha", 0.2))), device=self.device,
            dtype=torch.float32, requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(sac["alpha_lr"]))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = torch.as_tensor(observation, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.deterministic(obs) if deterministic else self.actor.sample(obs)[0]
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: Batch) -> dict[str, float]:
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(batch.next_observations)
            next_q = torch.min(
                self.target_q1(batch.next_observations, next_action),
                self.target_q2(batch.next_observations, next_action),
            )
            target = batch.rewards + self.gamma * (1.0 - batch.dones) * (
                next_q - self.alpha.detach() * next_log_prob
            )
        q_loss = F.mse_loss(self.q1(batch.observations, batch.actions), target)
        q_loss = q_loss + F.mse_loss(self.q2(batch.observations, batch.actions), target)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        action, log_prob = self.actor.sample(batch.observations)
        actor_loss = (self.alpha.detach() * log_prob - torch.min(
            self.q1(batch.observations, action), self.q2(batch.observations, action)
        )).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        soft_update(self.q1, self.target_q1, self.tau)
        soft_update(self.q2, self.target_q2, self.tau)
        return {
            "critic_loss": float(q_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(), "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(), "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(), "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any], *, actor_only: bool = False) -> None:
        self.actor.load_state_dict(state["actor"])
        if actor_only:
            return
        for key, module in (("q1", self.q1), ("q2", self.q2), ("target_q1", self.target_q1), ("target_q2", self.target_q2)):
            module.load_state_dict(state[key])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        for key, optimizer in (("actor_optimizer", self.actor_optimizer), ("q_optimizer", self.q_optimizer), ("alpha_optimizer", self.alpha_optimizer)):
            optimizer.load_state_dict(state[key])
            for values in optimizer.state.values():
                for name, value in values.items():
                    if torch.is_tensor(value):
                        values[name] = value.to(self.device)

    def save_actor(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path)
