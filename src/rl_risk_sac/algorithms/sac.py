from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rl_risk_sac.algorithms.networks import GaussianActor, QNetwork, hard_update, soft_update
from rl_risk_sac.algorithms.replay_buffer import Batch


def _config_signature(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def residual_control_signature(config: dict[str, Any]) -> dict[str, Any]:
    residual_cfg = config.get("env", {}).get("residual_control")
    if not residual_cfg or not bool(residual_cfg.get("enabled", True)):
        return {"chain": "direct_actor"}
    keys = (
        "residual_scale",
        "base_speed_scale",
        "terminal_goal_radius_m",
        "terminal_gain",
        "waypoint_gain",
        "damping",
        "clearance_margin_m",
        "waypoint_lateral_margin_m",
        "waypoint_height_offset_m",
        "link_avoidance_enabled",
        "link_avoidance_activation_margin_m",
        "link_avoidance_max_speed_mps",
    )
    return {"chain": "residual_actor", **{key: residual_cfg.get(key) for key in keys}}


def actor_signature(config: dict[str, Any], method: str) -> str:
    return _config_signature(
        {
            "method": method,
            "residual_control": residual_control_signature(config),
        }
    )


def reward_signature(config: dict[str, Any], method: str) -> str:
    return _config_signature(
        {
            "method": method,
            "reward": config["reward"],
            "fixed_risk_penalty": config["sac"].get("fixed_risk_penalty", 0.0),
            "residual_control": residual_control_signature(config),
            # ``link_fixed`` subtracts fixed_risk_penalty * cost, and the
            # cost/risk values depend on the complete risk configuration.
            # Include all of it so a changed distance/velocity model cannot
            # silently reuse a critic trained for a different reward target.
            "risk": config["risk"],
        }
    )


def cost_signature(config: dict[str, Any], method: str) -> str:
    return _config_signature(
        {
            "method": method,
            "risk": config["risk"],
            "residual_control": residual_control_signature(config),
        }
    )


def inferred_state_path(actor_path: str | Path) -> Path:
    actor = Path(actor_path)
    if actor.name.startswith("actor"):
        suffix = actor.stem[len("actor") :]
        return actor.with_name(f"agent_state{suffix}{actor.suffix}")
    raise ValueError(f"actor checkpoint name must start with 'actor': {actor}")


def _reset_parameters(module: torch.nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()


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
        self.actor_lr = float(sac_cfg["actor_lr"])
        self.critic_lr = float(sac_cfg["critic_lr"])
        self.alpha_lr = float(sac_cfg["alpha_lr"])
        self.actor_anchor_weight = float(sac_cfg.get("actor_anchor_weight", 0.0))
        if self.actor_anchor_weight < 0.0:
            raise ValueError("sac.actor_anchor_weight must be non-negative")
        self.actor_signature = actor_signature(config, method)
        self.reward_signature = reward_signature(config, method)
        self.cost_signature = cost_signature(config, method)
        self.last_load_info: dict[str, Any] = {}

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

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.reward_q_optimizer = torch.optim.Adam(
            list(self.reward_q1.parameters()) + list(self.reward_q2.parameters()),
            lr=self.critic_lr,
        )
        self.cost_q_optimizer = torch.optim.Adam(
            list(self.cost_q1.parameters()) + list(self.cost_q2.parameters()),
            lr=self.critic_lr,
        )
        self.log_alpha = torch.tensor(
            np.log(float(sac_cfg["initial_alpha"])),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.actor_reference: GaussianActor | None = None

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
        reward_q_loss, cost_q_loss = self._update_critics(batch)
        action, log_prob = self.actor.sample(batch.observations)
        reward_q = torch.min(self.reward_q1(batch.observations, action), self.reward_q2(batch.observations, action))
        cost_q = torch.min(self.cost_q1(batch.observations, action), self.cost_q2(batch.observations, action))
        actor_loss = (self.alpha.detach() * log_prob - reward_q).mean()
        if self.constrained:
            actor_loss = actor_loss + self.lagrange_multiplier * cost_q.mean()
        actor_anchor_loss = torch.zeros((), dtype=actor_loss.dtype, device=self.device)
        if self.actor_reference is not None and self.actor_anchor_weight > 0.0:
            actor_mean, _ = self.actor(batch.observations)
            with torch.no_grad():
                reference_mean, _ = self.actor_reference(batch.observations)
            actor_anchor_loss = F.mse_loss(torch.tanh(actor_mean), torch.tanh(reference_mean))
            actor_loss = actor_loss + self.actor_anchor_weight * actor_anchor_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update_critics()

        return {
            "loss/actor": float(actor_loss.detach().cpu()),
            "loss/actor_anchor": float(actor_anchor_loss.detach().cpu()),
            "loss/reward_q": reward_q_loss,
            "loss/cost_q": cost_q_loss,
            "loss/alpha": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "lambda": float(self.lagrange_multiplier),
            "cost_ema": float(self.cost_ema),
        }

    def _update_critics(self, batch: Batch) -> tuple[float, float]:
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

        return float(reward_q_loss.detach().cpu()), float(cost_q_loss.detach().cpu())

    def _soft_update_critics(self) -> None:
        soft_update(self.reward_q1, self.reward_target_q1, self.tau)
        soft_update(self.reward_q2, self.reward_target_q2, self.tau)
        soft_update(self.cost_q1, self.cost_target_q1, self.tau)
        soft_update(self.cost_q2, self.cost_target_q2, self.tau)

    def update_critics(self, batch: Batch) -> dict[str, float]:
        reward_q_loss, cost_q_loss = self._update_critics(batch)
        self._soft_update_critics()

        return {
            "loss/reward_q": reward_q_loss,
            "loss/cost_q": cost_q_loss,
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

    def set_actor_reference(self, state: dict[str, torch.Tensor] | None = None) -> None:
        if self.actor_anchor_weight <= 0.0:
            self.actor_reference = None
            return
        self.actor_reference = copy.deepcopy(self.actor).to(self.device)
        if state is not None:
            self.actor_reference.load_state_dict(state)
        self.actor_reference.eval()
        for parameter in self.actor_reference.parameters():
            parameter.requires_grad_(False)

    def reset_reward_critics(self) -> None:
        self.reward_q1.apply(_reset_parameters)
        self.reward_q2.apply(_reset_parameters)
        hard_update(self.reward_q1, self.reward_target_q1)
        hard_update(self.reward_q2, self.reward_target_q2)
        self.reward_q_optimizer = torch.optim.Adam(
            list(self.reward_q1.parameters()) + list(self.reward_q2.parameters()),
            lr=self.critic_lr,
        )

    def reset_cost_critics(self) -> None:
        self.cost_q1.apply(_reset_parameters)
        self.cost_q2.apply(_reset_parameters)
        hard_update(self.cost_q1, self.cost_target_q1)
        hard_update(self.cost_q2, self.cost_target_q2)
        self.cost_q_optimizer = torch.optim.Adam(
            list(self.cost_q1.parameters()) + list(self.cost_q2.parameters()),
            lr=self.critic_lr,
        )

    def save(
        self,
        directory: str | Path,
        suffix: str = "",
        training_state: dict[str, Any] | None = None,
    ) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path / f"actor{suffix}.pt")
        torch.save(
            {
                "format_version": 2,
                "reward_q1": self.reward_q1.state_dict(),
                "reward_q2": self.reward_q2.state_dict(),
                "reward_target_q1": self.reward_target_q1.state_dict(),
                "reward_target_q2": self.reward_target_q2.state_dict(),
                "cost_q1": self.cost_q1.state_dict(),
                "cost_q2": self.cost_q2.state_dict(),
                "cost_target_q1": self.cost_target_q1.state_dict(),
                "cost_target_q2": self.cost_target_q2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "reward_q_optimizer": self.reward_q_optimizer.state_dict(),
                "cost_q_optimizer": self.cost_q_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "lagrange_multiplier": self.lagrange_multiplier,
                "cost_ema": self.cost_ema,
                "method": self.method,
                "actor_signature": self.actor_signature,
                "reward_signature": self.reward_signature,
                "cost_signature": self.cost_signature,
                "actor_reference": (
                    self.actor_reference.state_dict() if self.actor_reference is not None else None
                ),
                "training_state": training_state,
            },
            path / f"agent_state{suffix}.pt",
        )

    def load(
        self,
        actor_path: str | Path,
        state_path: str | Path | None = None,
        *,
        reset_reward_critics: bool = False,
        reset_cost_critics: bool = False,
        restore_optimizers: bool = True,
    ) -> dict[str, Any]:
        if state_path is None:
            self.load_actor(actor_path, validate_signature=True)
            self.last_load_info = {
                "reward_critics_loaded": False,
                "cost_critics_loaded": False,
                "optimizers_loaded": False,
                "training_state": None,
            }
            return self.last_load_info

        state_path = Path(state_path)
        state = torch.load(state_path, map_location=self.device, weights_only=False)
        checkpoint_actor_signature = state.get("actor_signature")
        checkpoint_reward_signature = state.get("reward_signature")
        checkpoint_cost_signature = state.get("cost_signature")
        checkpoint_method = str(state.get("method", self.method))
        (
            checkpoint_actor_signature,
            checkpoint_reward_signature,
            checkpoint_cost_signature,
        ) = self._fill_checkpoint_signatures_from_config(
            state_path.parent / "config.json",
            checkpoint_method,
            checkpoint_actor_signature,
            checkpoint_reward_signature,
            checkpoint_cost_signature,
        )
        if checkpoint_actor_signature is None:
            raise ValueError("agent-state checkpoint is missing actor action semantics")

        actor_compatible = checkpoint_actor_signature == self.actor_signature
        reward_compatible = checkpoint_reward_signature == self.reward_signature
        cost_compatible = checkpoint_cost_signature == self.cost_signature
        if not actor_compatible:
            raise ValueError(
                "actor checkpoint action semantics do not match this config; "
                "train or load a checkpoint from the residual-actor chain"
            )
        self.load_actor(actor_path, validate_signature=False)
        load_reward = not reset_reward_critics and reward_compatible
        load_cost = not reset_cost_critics and cost_compatible
        if load_reward:
            self.reward_q1.load_state_dict(state["reward_q1"])
            self.reward_q2.load_state_dict(state["reward_q2"])
            if "reward_target_q1" in state and "reward_target_q2" in state:
                self.reward_target_q1.load_state_dict(state["reward_target_q1"])
                self.reward_target_q2.load_state_dict(state["reward_target_q2"])
            else:
                hard_update(self.reward_q1, self.reward_target_q1)
                hard_update(self.reward_q2, self.reward_target_q2)
        else:
            self.reset_reward_critics()
        if load_cost:
            self.cost_q1.load_state_dict(state["cost_q1"])
            self.cost_q2.load_state_dict(state["cost_q2"])
            if "cost_target_q1" in state and "cost_target_q2" in state:
                self.cost_target_q1.load_state_dict(state["cost_target_q1"])
                self.cost_target_q2.load_state_dict(state["cost_target_q2"])
            else:
                hard_update(self.cost_q1, self.cost_target_q1)
                hard_update(self.cost_q2, self.cost_target_q2)
        else:
            self.reset_cost_critics()
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.lagrange_multiplier = float(state["lagrange_multiplier"])
        self.cost_ema = float(state["cost_ema"])
        optimizer_keys = {
            "actor_optimizer",
            "reward_q_optimizer",
            "cost_q_optimizer",
            "alpha_optimizer",
        }
        optimizers_loaded = bool(restore_optimizers and optimizer_keys.issubset(state))
        if optimizers_loaded:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            if load_reward:
                self.reward_q_optimizer.load_state_dict(state["reward_q_optimizer"])
            if load_cost:
                self.cost_q_optimizer.load_state_dict(state["cost_q_optimizer"])
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
            self._set_optimizer_lr(self.actor_optimizer, self.actor_lr)
            self._set_optimizer_lr(self.reward_q_optimizer, self.critic_lr)
            self._set_optimizer_lr(self.cost_q_optimizer, self.critic_lr)
            self._set_optimizer_lr(self.alpha_optimizer, self.alpha_lr)
        reference_state = state.get("actor_reference")
        if reference_state is not None:
            self.set_actor_reference(reference_state)
        elif self.actor_anchor_weight > 0.0:
            self.set_actor_reference()
        self.last_load_info = {
            "reward_critics_loaded": load_reward,
            "cost_critics_loaded": load_cost,
            "actor_signature_match": actor_compatible,
            "reward_signature_match": reward_compatible,
            "cost_signature_match": cost_compatible,
            "optimizers_loaded": optimizers_loaded,
            "training_state": state.get("training_state"),
        }
        return self.last_load_info

    @staticmethod
    def _set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

    def load_actor(
        self,
        actor_path: str | Path,
        *,
        validate_signature: bool = True,
    ) -> None:
        if validate_signature:
            checkpoint_actor_signature = self._infer_actor_checkpoint_signature(actor_path)
            if checkpoint_actor_signature is not None and checkpoint_actor_signature != self.actor_signature:
                raise ValueError(
                    "actor checkpoint action semantics do not match this config; "
                    "train or load a checkpoint from the residual-actor chain"
                )
        state = torch.load(actor_path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(state)
        if self.actor_anchor_weight > 0.0:
            self.set_actor_reference()

    def _infer_actor_checkpoint_signature(self, actor_path: str | Path) -> str | None:
        actor_path = Path(actor_path)
        state_path = inferred_state_path(actor_path)
        if state_path.is_file():
            state = torch.load(state_path, map_location=self.device, weights_only=False)
            checkpoint_actor_signature = state.get("actor_signature")
            checkpoint_method = str(state.get("method", self.method))
            checkpoint_actor_signature, _, _ = self._fill_checkpoint_signatures_from_config(
                state_path.parent / "config.json",
                checkpoint_method,
                checkpoint_actor_signature,
                None,
                None,
            )
            if checkpoint_actor_signature is not None:
                return str(checkpoint_actor_signature)

        source_config = actor_path.parent / "config.json"
        if source_config.is_file():
            with source_config.open(encoding="utf-8") as file:
                source = json.load(file)
            checkpoint_method = str(source.get("train", {}).get("method", self.method))
            return actor_signature(source, checkpoint_method)
        return None

    @staticmethod
    def _fill_checkpoint_signatures_from_config(
        source_config: Path,
        checkpoint_method: str,
        checkpoint_actor_signature: object,
        checkpoint_reward_signature: object,
        checkpoint_cost_signature: object,
    ) -> tuple[object, object, object]:
        if source_config.is_file() and (
            checkpoint_actor_signature is None
            or checkpoint_reward_signature is None
            or checkpoint_cost_signature is None
        ):
            with source_config.open(encoding="utf-8") as file:
                source = json.load(file)
            checkpoint_actor_signature = checkpoint_actor_signature or actor_signature(source, checkpoint_method)
            checkpoint_reward_signature = checkpoint_reward_signature or reward_signature(source, checkpoint_method)
            checkpoint_cost_signature = checkpoint_cost_signature or cost_signature(source, checkpoint_method)
        return checkpoint_actor_signature, checkpoint_reward_signature, checkpoint_cost_signature
