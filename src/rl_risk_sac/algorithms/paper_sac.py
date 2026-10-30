"""Standard SAC for the paper task, with an optional conditional Gaussian AE ablation."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rl_risk_sac.algorithms.networks import soft_update
from rl_risk_sac.algorithms.replay_buffer import Batch
from rl_risk_sac.utils.device import select_device
from rl_risk_sac.utils.paper_config import device_selection


def sac_mlp(input_dim, hidden, output_dim):
    layers, previous = [], input_dim
    for width in hidden:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    return nn.Sequential(*layers, nn.Linear(previous, output_dim))


def optimizer_gradient_norm(parameters, clip, collect):
    """Keep diagnostics independent of the optimizer's clipping policy."""
    parameters = tuple(parameters)
    if clip is not None:
        return nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=collect)
    if collect:
        return torch.linalg.vector_norm(torch.stack([
            parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None]))
    return None


class PaperGaussianActor(nn.Module):
    def __init__(self, obs_dim=25, action_dim=6, hidden=(256, 256)):
        super().__init__()
        self.action_dim = action_dim
        # ReLU backbone and ordinary Linear initialization match SB3 SAC.
        # The final tanh squashes actions only; it is not a hidden activation.
        self.net = sac_mlp(obs_dim, hidden, 2 * action_dim)

    def forward(self, observations):
        mean, log_std = self.net(observations).chunk(2, dim=-1)
        return mean, log_std.clamp(-20., 2.)

    def sample(self, observations, counts=None):
        mean, log_std = self(observations)
        if counts is not None:
            # Mean of N independent *unsquashed* Gaussians. Sampling the
            # resulting Gaussian is exact and avoids N network/sample passes.
            log_std = log_std - .5 * counts.log()
        normal = torch.distributions.Normal(mean, log_std.exp(), validate_args=False)
        raw = normal.rsample()
        action = raw.tanh()
        correction = 2 * (math.log(2.) - raw - F.softplus(-2 * raw))
        log_prob = (normal.log_prob(raw) - correction).sum(dim=-1, keepdim=True)
        return action, log_prob

    def log_prob(self, observations, actions, counts=None, reduce=True):
        """Log probability of replay actions, including the tanh Jacobian.

        ``reduce=False`` is used by SAC policy feedback to reproduce the
        author's per-action-dimension probability transform before averaging.
        """
        mean, log_std = self(observations)
        if counts is not None:
            log_std = log_std - .5 * counts.log()
        eps = torch.finfo(actions.dtype).eps
        bounded = actions.clamp(-1. + eps, 1. - eps)
        raw = torch.atanh(bounded)
        normal = torch.distributions.Normal(mean, log_std.exp(), validate_args=False)
        correction = 2 * (math.log(2.) - raw - F.softplus(-2 * raw))
        values = normal.log_prob(raw) - correction
        return values.sum(dim=-1, keepdim=True) if reduce else values

    def deterministic(self, observations):
        return self(observations)[0].tanh()


class PaperQNetwork(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = sac_mlp(31, hidden, 1)

    def forward(self, observations, actions):
        return self.net(torch.cat((observations, actions), dim=-1))


class PaperSACAgent:
    def __init__(self, config):
        sac = config["sac"]
        self.device = torch.device(select_device(config["device"], device_selection(config)))
        hidden = sac["hidden_dims"]
        self.actor = PaperGaussianActor(hidden=hidden).to(self.device)
        self.q1, self.q2 = PaperQNetwork(hidden).to(self.device), PaperQNetwork(hidden).to(self.device)
        self.target_q1, self.target_q2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=sac["actor_lr"])
        self.q_optimizer = torch.optim.Adam((*self.q1.parameters(), *self.q2.parameters()), lr=sac["critic_lr"])
        self.log_alpha = torch.tensor(math.log(sac["initial_alpha"]), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=sac["alpha_lr"])
        # SAC on the verified short-horizon manipulation task otherwise drives
        # alpha almost to zero during the first replay phase.  Keep a small
        # exploration floor configurable; older configs retain the old behavior.
        self.alpha_min = float(sac.get("alpha_min", 0.0))
        self.alpha_max = float(sac.get("alpha_max", float("inf")))
        if self.alpha_min < 0 or self.alpha_max <= 0 or self.alpha_min > self.alpha_max:
            raise ValueError("invalid SAC alpha bounds")
        self.gamma, self.tau = float(sac["gamma"]), float(sac["tau"])
        self.target_entropy = float(sac["target_entropy"])
        feedback = sac.get("policy_feedback", {})
        self.policy_feedback = bool(feedback.get("enabled", False))
        self.feedback_gamma_min = float(feedback.get("gamma_min", 0.6))
        self.feedback_gamma_max = float(feedback.get("gamma_max", 0.99))
        self.feedback_logprob_scale = float(feedback.get("logprob_scale", 0.1))
        self.feedback_transform = str(feedback.get("transform", "sac_sigmoid"))
        self.feedback_logprob_center = float(feedback.get("logprob_center", 0.0))
        self.feedback_temperature = float(feedback.get("temperature", 0.5))
        if not 0 < self.feedback_gamma_min <= self.feedback_gamma_max <= 1:
            raise ValueError("invalid policy feedback gamma bounds")
        if self.feedback_transform not in ("sac_sigmoid", "paper") or self.feedback_temperature <= 0:
            raise ValueError("invalid policy feedback transform")
        self.actor_update_interval = int(sac["actor_update_interval"])
        self.gradient_clip = None if sac["gradient_clip"] is None else float(sac["gradient_clip"])
        self.ae = copy.deepcopy(sac["action_ensemble"])
        self.rng = np.random.default_rng(config["seed"] + 101)
        self.progress = 0.
        self.update_steps = 0
        self.last_actor_metrics = {}

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def set_episode_progress(self, episodes):
        self.progress = float(np.clip(episodes / self.ae["total_episodes"], 0., 1.))

    def _counts(self, batch_size):
        if not self.ae["enabled"]:
            return None
        shape = 1 + self.ae["shape_growth"] * self.progress
        scale = 1 + self.ae["scale_growth"] * self.progress
        samples = scale * self.rng.weibull(shape, size=(batch_size, self.ae["count_draws"]))
        counts = np.clip(samples, 1., scale).mean(axis=1).astype(np.int64).clip(1)
        return torch.as_tensor(counts[:, None], device=self.device, dtype=torch.float32)

    def select_actions(self, observations, deterministic=False):
        obs = torch.as_tensor(np.asarray(observations), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            actions = (self.actor.deterministic(obs) if deterministic else
                       self.actor.sample(obs, self._counts(len(obs)))[0])
        return actions.cpu().numpy()

    def update(self, batch, *, collect_diagnostics=False):
        with torch.no_grad():
            actions, log_prob = self.actor.sample(batch.next_observations, self._counts(len(batch.actions)))
            q = torch.minimum(self.target_q1(batch.next_observations, actions),
                              self.target_q2(batch.next_observations, actions))
            if self.policy_feedback:
                counts = self._counts(len(batch.actions))
                per_action_log_prob = self.actor.log_prob(batch.observations, batch.actions,
                                                          counts=counts, reduce=False)
                if self.feedback_transform == "paper":
                    positive = torch.tanh(self.feedback_logprob_scale * per_action_log_prob)
                    negative = per_action_log_prob.exp()
                    confidence = torch.where(per_action_log_prob > 0, positive, negative)
                    feedback_gamma = confidence.mean(dim=-1, keepdim=True).clamp(
                        self.feedback_gamma_min, self.feedback_gamma_max)
                else:
                    # SAC's tanh-corrected log-probabilities are not distributed
                    # like PPO's Gaussian log-probabilities.  A centered sigmoid
                    # avoids collapsing every transition to gamma_min.
                    confidence = torch.sigmoid(
                        (per_action_log_prob - self.feedback_logprob_center) / self.feedback_temperature)
                    feedback_gamma = self.feedback_gamma_min + (
                        self.feedback_gamma_max - self.feedback_gamma_min) * confidence.mean(dim=-1, keepdim=True)
            else:
                feedback_gamma = torch.full_like(batch.rewards, self.gamma)
            target = batch.rewards + feedback_gamma * (1 - batch.dones) * (q - self.alpha.detach() * log_prob)
        q1, q2 = self.q1(batch.observations, batch.actions), self.q2(batch.observations, batch.actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_parameters = (*self.q1.parameters(), *self.q2.parameters())
        critic_norm = optimizer_gradient_norm(critic_parameters, self.gradient_clip, collect_diagnostics)
        self.q_optimizer.step()
        self.update_steps += 1
        actor_updated = self.update_steps % self.actor_update_interval == 0
        if actor_updated:
            self.q1.requires_grad_(False)
            self.q2.requires_grad_(False)
            try:
                action, log_prob = self.actor.sample(batch.observations, self._counts(len(batch.actions)))
                q = torch.minimum(self.q1(batch.observations, action), self.q2(batch.observations, action))
                actor_loss = (self.alpha.detach() * log_prob - q).mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_norm = optimizer_gradient_norm(self.actor.parameters(), self.gradient_clip, collect_diagnostics)
                self.actor_optimizer.step()
                alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                if self.alpha_min > 0 or np.isfinite(self.alpha_max):
                    with torch.no_grad():
                        low = math.log(max(self.alpha_min, 1e-12))
                        high = math.log(self.alpha_max)
                        self.log_alpha.clamp_(min=low, max=high)
                if collect_diagnostics:
                    values = torch.stack((actor_loss.detach(), alpha_loss.detach(),
                                          -log_prob.detach().mean(), actor_norm.detach())).cpu().tolist()
                    self.last_actor_metrics = dict(zip(
                        ("actor_loss", "alpha_loss", "policy_entropy", "actor_gradient_norm"), values))
            finally:
                self.q1.requires_grad_(True)
                self.q2.requires_grad_(True)
        soft_update(self.q1, self.target_q1, self.tau)
        soft_update(self.q2, self.target_q2, self.tau)
        if not collect_diagnostics:
            return {}
        values = torch.stack((critic_loss.detach(), self.alpha.detach(), q1.detach().mean(),
                              q2.detach().mean(), target.mean(), critic_norm.detach(),
                              feedback_gamma.detach().mean())).cpu().tolist()
        metrics = dict(zip(("critic_loss", "alpha", "q1_mean", "q2_mean", "target_mean",
                            "critic_gradient_norm", "policy_feedback_gamma"), values))
        metrics.update(self.last_actor_metrics, actor_updated=float(actor_updated))
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError("nonfinite single-SAC diagnostics")
        return metrics

    def state_dict(self):
        return {**{name: getattr(self, name).state_dict() for name in (
            "actor", "q1", "q2", "target_q1", "target_q2", "actor_optimizer", "q_optimizer", "alpha_optimizer")},
            "log_alpha": self.log_alpha.detach().cpu(), "rng": self.rng.bit_generator.state,
            "progress": self.progress, "update_steps": self.update_steps,
            "last_actor_metrics": dict(self.last_actor_metrics)}

    def load_state_dict(self, state):
        for name in ("actor", "q1", "q2", "target_q1", "target_q2",
                     "actor_optimizer", "q_optimizer", "alpha_optimizer"):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.rng.bit_generator.state = state["rng"]
        self.progress, self.update_steps = state["progress"], state["update_steps"]
        self.last_actor_metrics = dict(state["last_actor_metrics"])


class PaperReplayBuffer:
    """Vectorized insertion, uniform replay, compact complete checkpoint snapshots."""
    shapes = {"observations": 25, "actions": 6, "rewards": 1, "next_observations": 25, "dones": 1}

    def __init__(self, capacity, device, seed):
        self.capacity, self.device = int(capacity), torch.device(device)
        self.arrays = {key: np.empty((capacity, width), np.float32) for key, width in self.shapes.items()}
        self.size, self.ptr = 0, 0
        self.rng = np.random.default_rng(seed)

    def add_many(self, observations, actions, rewards, next_observations, terminals):
        rows = dict(observations=observations, actions=actions, rewards=rewards,
                    next_observations=next_observations, dones=terminals)
        count = len(observations)
        if not 0 < count <= self.capacity:
            raise ValueError("invalid replay insertion size")
        indices = (self.ptr + np.arange(count)) % self.capacity
        for key, width in self.shapes.items():
            self.arrays[key][indices] = np.asarray(rows[key], dtype=np.float32).reshape(count, width)
        self.ptr = (self.ptr + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size):
        if not self.size:
            raise ValueError("cannot sample empty replay")
        indices = self.rng.integers(self.size, size=batch_size)
        values = {key: torch.as_tensor(value[indices], device=self.device) for key, value in self.arrays.items()}
        return Batch(**values, costs=torch.zeros_like(values["rewards"]))

    def state_dict(self):
        return dict(capacity=self.capacity, size=self.size, ptr=self.ptr,
                    arrays={key: value[:self.size].copy() for key, value in self.arrays.items()},
                    rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity or not 0 <= state["size"] <= self.capacity:
            raise ValueError("incompatible replay checkpoint")
        self.size, self.ptr = state["size"], state["ptr"]
        for key in self.shapes:
            self.arrays[key][:self.size] = state["arrays"][key]
        self.rng.bit_generator.state = state["rng"]
