from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device: str) -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.costs = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        cost: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.costs[self.ptr] = cost
        self.next_observations[self.ptr] = next_observation
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        indices = np.random.randint(0, self.size, size=batch_size)
        return Batch(
            observations=self._tensor(self.observations[indices]),
            actions=self._tensor(self.actions[indices]),
            rewards=self._tensor(self.rewards[indices]),
            costs=self._tensor(self.costs[indices]),
            next_observations=self._tensor(self.next_observations[indices]),
            dones=self._tensor(self.dones[indices]),
        )

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, device=self.device, dtype=torch.float32)

    def __len__(self) -> int:
        return self.size
