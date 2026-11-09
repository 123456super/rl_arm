from __future__ import annotations

import torch
import torch.nn as nn


def mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    *,
    activation: str = "relu",
    orthogonal_init: bool = False,
    output_gain: float = 1.0,
) -> nn.Sequential:
    """Build a feed-forward network.

    The reference FTP4RM PPO uses two tanh hidden layers with orthogonal
    initialization.  The options are kept explicit so non-PPO callers retain
    the historical ReLU behavior.
    """
    if activation == "tanh":
        activation_layer: type[nn.Module] = nn.Tanh
    elif activation == "relu":
        activation_layer = nn.ReLU
    else:
        raise ValueError(f"unsupported activation: {activation}")
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layer = nn.Linear(last_dim, hidden_dim)
        if orthogonal_init:
            nn.init.orthogonal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        layers.extend([layer, activation_layer()])
        last_dim = hidden_dim
    output = nn.Linear(last_dim, output_dim)
    if orthogonal_init:
        nn.init.orthogonal_(output.weight, gain=output_gain)
        nn.init.constant_(output.bias, 0.0)
    layers.append(output)
    return nn.Sequential(*layers)


class ObstacleSetEncoder(nn.Module):
    """Permutation-invariant encoder for a masked set of obstacle features."""

    def __init__(
        self,
        feature_dim: int,
        max_obstacles: int,
        embedding_dim: int = 32,
        output_dim: int = 32,
    ) -> None:
        super().__init__()
        if min(feature_dim, max_obstacles, embedding_dim, output_dim) < 1:
            raise ValueError("obstacle encoder dimensions must be positive")
        self.feature_dim = feature_dim
        self.max_obstacles = max_obstacles
        self.output_dim = output_dim
        self.phi = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
        )
        self.rho = nn.Sequential(
            nn.Linear(2 * embedding_dim + 1, output_dim),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2 or features.shape[-2:] != (
            self.max_obstacles,
            self.feature_dim,
        ):
            raise ValueError("invalid obstacle feature tensor shape")
        if mask.shape != features.shape[:-1]:
            raise ValueError("obstacle mask shape does not match features")
        mask = mask.to(dtype=features.dtype).clamp(0.0, 1.0)
        present = (mask.sum(dim=-1, keepdim=True) > 0.0).to(features.dtype)
        embedded = self.phi(features)
        expanded_mask = mask.unsqueeze(-1)
        count = expanded_mask.sum(dim=-2).clamp_min(1.0)
        mean_pool = (embedded * expanded_mask).sum(dim=-2) / count
        minimum = torch.finfo(embedded.dtype).min
        max_pool = embedded.masked_fill(expanded_mask == 0.0, minimum).amax(dim=-2)
        max_pool = torch.where(present.bool(), max_pool, torch.zeros_like(max_pool))
        count_fraction = mask.sum(dim=-1, keepdim=True) / float(self.max_obstacles)
        encoded = self.rho(torch.cat([mean_pool, max_pool, count_fraction], dim=-1))
        # Exact zero is important: a legacy S0 policy sees the same effective
        # input after migration, independent of newly initialized encoder bias.
        return encoded * present
