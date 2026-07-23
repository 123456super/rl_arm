from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rl_risk_sac.algorithms.networks import GaussianActor, QNetwork, hard_update, soft_update
from rl_risk_sac.algorithms.replay_buffer import Batch


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: dict[str, Any],
        method: str,
    ) -> None:
        self.method = method
        self.constrained = method.startswith("ldrc")
        self.device = torch.device(config.get("device", "cpu"))
        sac_cfg = config["sac"]
        hidden_dims = [int(v) for v in sac_cfg["hidden_dims"]]

        self.gamma = float(sac_cfg["gamma"])
        self.tau = float(sac_cfg["tau"])
        self.batch_size = int(sac_cfg["batch_size"])
        self.target_entropy = -float(action_dim)
        self.cost_safe = float(config["risk"]["cost"]["c_safe"])
        self.lambda_lr = float(sac_cfg["lambda_lr"])
        self.cost_ema_rho = float(sac_cfg["cost_ema_rho"])
        self.cost_ema = self.cost_safe
        self.lagrange_multiplier = float(sac_cfg["initial_lambda"]) if self.constrained else 0.0

        self.actor = GaussianActor(obs_dim, action_dim, hidden_dims).to(self.device)
        self.reward_q1 = QNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.reward_q2 = QNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.reward_target_q1 = QNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.reward_target_q2 = QNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        hard_update(self.reward_q1, self.reward_target_q1)
        hard_update(self.reward_q2, self.reward_target_q2)

        self.cost_q1 = QNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.cost_q2 = QNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.cost_target_q1 = QNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.cost_target_q2 = QNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        hard_update(self.cost_q1, self.cost_target_q1)
        hard_update(self.cost_q2, self.cost_target_q2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(sac_cfg["actor_lr"]))
        self.reward_q_optimizer = torch.optim.Adam(
            list(self.reward_q1.parameters()) + list(self.reward_q2.parameters()),
            lr=float(sac_cfg["critic_lr"]),
        )
        self.cost_q_optimizer = torch.optim.Adam(
            list(self.cost_q1.parameters()) + list(self.cost_q2.parameters()),
            lr=float(sac_cfg["critic_lr"]),
        )
        self.log_alpha = torch.tensor(
            np.log(float(sac_cfg["initial_alpha"])),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(sac_cfg["alpha_lr"]))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(obs)
            else:
                action, _ = self.actor.sample(obs)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: Batch) -> dict[str, float]:
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(batch.next_observations)
            target_reward_q = torch.min(
                self.reward_target_q1(batch.next_observations, next_action),
                self.reward_target_q2(batch.next_observations, next_action),
            )
            reward_target = batch.rewards + self.gamma * (1.0 - batch.dones) * (
                target_reward_q - self.alpha.detach() * next_log_prob
            )

            target_cost_q = torch.min(
                self.cost_target_q1(batch.next_observations, next_action),
                self.cost_target_q2(batch.next_observations, next_action),
            )
            cost_target = batch.costs + self.gamma * (1.0 - batch.dones) * target_cost_q

        reward_q1 = self.reward_q1(batch.observations, batch.actions)
        reward_q2 = self.reward_q2(batch.observations, batch.actions)
        reward_q_loss = F.mse_loss(reward_q1, reward_target) + F.mse_loss(reward_q2, reward_target)
        self.reward_q_optimizer.zero_grad(set_to_none=True)
        reward_q_loss.backward()
        self.reward_q_optimizer.step()

        cost_q1 = self.cost_q1(batch.observations, batch.actions)
        cost_q2 = self.cost_q2(batch.observations, batch.actions)
        cost_q_loss = F.mse_loss(cost_q1, cost_target) + F.mse_loss(cost_q2, cost_target)
        self.cost_q_optimizer.zero_grad(set_to_none=True)
        cost_q_loss.backward()
        self.cost_q_optimizer.step()

        action, log_prob = self.actor.sample(batch.observations)
        reward_q = torch.min(self.reward_q1(batch.observations, action), self.reward_q2(batch.observations, action))
        cost_q = torch.min(self.cost_q1(batch.observations, action), self.cost_q2(batch.observations, action))
        actor_loss = (self.alpha.detach() * log_prob - reward_q).mean()
        if self.constrained:
            actor_loss = actor_loss + self.lagrange_multiplier * cost_q.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        soft_update(self.reward_q1, self.reward_target_q1, self.tau)
        soft_update(self.reward_q2, self.reward_target_q2, self.tau)
        soft_update(self.cost_q1, self.cost_target_q1, self.tau)
        soft_update(self.cost_q2, self.cost_target_q2, self.tau)

        return {
            "loss/actor": float(actor_loss.detach().cpu()),
            "loss/reward_q": float(reward_q_loss.detach().cpu()),
            "loss/cost_q": float(cost_q_loss.detach().cpu()),
            "loss/alpha": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "lambda": float(self.lagrange_multiplier),
            "cost_ema": float(self.cost_ema),
        }

    def update_lagrange(self, episode_mean_cost: float) -> None:
        if not self.constrained:
            return
        self.cost_ema = (1.0 - self.cost_ema_rho) * self.cost_ema + self.cost_ema_rho * float(episode_mean_cost)
        self.lagrange_multiplier = max(
            0.0,
            self.lagrange_multiplier + self.lambda_lr * (self.cost_ema - self.cost_safe),
        )

    def save(self, directory: str | Path, suffix: str = "") -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path / f"actor{suffix}.pt")
        torch.save(
            {
                "reward_q1": self.reward_q1.state_dict(),
                "reward_q2": self.reward_q2.state_dict(),
                "cost_q1": self.cost_q1.state_dict(),
                "cost_q2": self.cost_q2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "lagrange_multiplier": self.lagrange_multiplier,
                "cost_ema": self.cost_ema,
                "method": self.method,
            },
            path / f"agent_state{suffix}.pt",
        )

    def load(self, actor_path: str | Path, state_path: str | Path | None = None) -> None:
        self.load_actor(actor_path)
        if state_path is None:
            return

        state = torch.load(state_path, map_location=self.device)
        self.reward_q1.load_state_dict(state["reward_q1"])
        self.reward_q2.load_state_dict(state["reward_q2"])
        self.cost_q1.load_state_dict(state["cost_q1"])
        self.cost_q2.load_state_dict(state["cost_q2"])
        hard_update(self.reward_q1, self.reward_target_q1)
        hard_update(self.reward_q2, self.reward_target_q2)
        hard_update(self.cost_q1, self.cost_target_q1)
        hard_update(self.cost_q2, self.cost_target_q2)
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.lagrange_multiplier = float(state["lagrange_multiplier"])
        self.cost_ema = float(state["cost_ema"])

    def load_actor(self, actor_path: str | Path) -> None:
        state = torch.load(actor_path, map_location=self.device)
        self.actor.load_state_dict(state)
