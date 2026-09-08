from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN = -20
LOG_STD_MAX = 2


def mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    """Create the simple feed-forward MLP used by actor and critics."""
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU()])
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy used by SAC.

    网络先预测未限幅高斯分布的 mean/log_std，再采样并经过 tanh，
    所以最终动作天然落在环境要求的 [-1, 1]。
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.backbone = mlp(obs_dim, hidden_dims, hidden_dims[-1])
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        # tanh 改变了概率密度，log_prob 必须减去雅可比修正项；
        # 这是 SAC 连续动作实现里的标准写法。
        log_prob = normal.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    """Critic network Q(s, a) -> scalar value."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.net = mlp(obs_dim + action_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Polyak update: slowly move target network toward the online network."""
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


def hard_update(source: nn.Module, target: nn.Module) -> None:
    """Copy all parameters exactly, used when target networks are initialized."""
    target.load_state_dict(source.state_dict())
