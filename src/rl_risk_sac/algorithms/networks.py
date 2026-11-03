from __future__ import annotations

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
