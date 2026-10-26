"""Independent, jointly trained permutation-invariant Full set encoders."""
from __future__ import annotations

import torch
from torch import nn

from rl_risk_sac.algorithms.networks import GaussianActor, QNetwork, mlp


class ObstacleSetEncoder(nn.Module):
    def __init__(self, scale):
        super().__init__()
        scale = torch.as_tensor(scale, dtype=torch.float32)
        if scale.shape != (37,) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("obstacle feature scale must contain 37 finite positive training scales")
        self.register_buffer("scale", scale)
        self.phi = mlp(37, [64], 64)
        self.attention = mlp(37, [32], 1)
        self.fusion = mlp(193, [128], 128)

    def forward(self, features, mask):
        result = features.new_zeros((features.shape[0], 128))
        if features.shape[1] == 0:
            return result
        active = mask.any(dim=1)
        if not active.any():
            return result
        valid = mask[active]
        x = features[active] / self.scale
        # Distances and displacements use clipped fixed scales. Directions
        # and support ranges retain their training-statistic normalization.
        x = torch.cat((x[..., :4].clamp(-5, 5), x[..., 4:16],
                       x[..., 16:25].clamp(-5, 5), x[..., 25:]), dim=-1)
        x = x.masked_fill(~valid[..., None], 0)
        h = self.phi(x)
        count = valid.sum(dim=1, keepdim=True)
        mean = (h * valid[..., None]).sum(dim=1) / count
        maximum = h.masked_fill(~valid[..., None], -torch.inf).amax(dim=1)
        logits = self.attention(x).squeeze(-1).masked_fill(~valid, -torch.inf)
        attention = (logits.softmax(dim=1)[..., None] * h).sum(dim=1)
        pooled = torch.cat((mean, maximum, attention, torch.log1p(count.to(h.dtype))), dim=-1)
        result[active] = self.fusion(pooled)
        return result


class SetGaussianActor(GaussianActor):
    def __init__(self, obs_dim, action_dim, hidden_dims, feature_scale):
        super().__init__(obs_dim, action_dim, hidden_dims)
        self.obstacle_encoder = ObstacleSetEncoder(feature_scale)

    def forward(self, obs, features=None, mask=None):
        if features is None:
            z = obs.new_zeros((len(obs), 128))
        else:
            z = self.obstacle_encoder(features, mask)
        return super().forward(torch.cat((obs[..., :89], z, obs[..., 217:]), dim=-1))

    def sample(self, obs, features=None, mask=None):
        # Reuse SAC's Gaussian/tanh calculation while retaining the raw set.
        from torch.distributions import Normal
        mean, log_std = self(obs, features, mask)
        normal = Normal(mean, log_std.exp())
        raw_action = normal.rsample()
        action = raw_action.tanh()
        log_prob = (normal.log_prob(raw_action)
                    - torch.log(1 - action.square() + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs, features=None, mask=None):
        return self(obs, features, mask)[0].tanh()


class SetQNetwork(QNetwork):
    def __init__(self, obs_dim, action_dim, hidden_dims, feature_scale, external_start):
        super().__init__(obs_dim, action_dim, hidden_dims)
        self.external_start = external_start
        self.obstacle_encoder = ObstacleSetEncoder(feature_scale)

    def forward(self, obs, action, features=None, mask=None):
        start = self.external_start
        z = obs.new_zeros((len(obs), 128)) if features is None else self.obstacle_encoder(features, mask)
        state = torch.cat((obs[..., :start], z, obs[..., start + 128:]), dim=-1)
        return super().forward(state, action)
