"""Independent local-task replay, including raw sets and observable context."""
from __future__ import annotations

import numpy as np
import torch

from rl_risk_sac.algorithms.replay_buffer import Batch
from rl_risk_sac.tasks.local_avoidance import LOCAL_OBSERVATION_DIM, PROTOCOL
from rl_risk_sac.tasks.obstacle_observation import obstacle_set, FEATURE_DIM, cache_obstacle_rows


class LocalReplayBuffer:
    def __init__(self, capacity, device, seed, *, protocol=PROTOCOL):
        self.protocol = protocol
        self.capacity, self.device = int(capacity), device
        self.rng = np.random.default_rng(seed)
        self.ptr = self.size = 0
        self.observations = np.zeros((capacity, LOCAL_OBSERVATION_DIM), np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.actions = np.zeros((capacity, 6), np.float32)
        self.rewards = np.zeros((capacity, 1), np.float32)
        self.dones = np.zeros_like(self.rewards)
        self.sets, self.next_sets = [None] * capacity, [None] * capacity
        # Derived raw-feature caches, not neural embeddings. Grow on demand so
        # obstacle count stays variable and old ragged checkpoints remain valid.
        self._feature_capacity = 0
        self._features = np.zeros((capacity, 0, FEATURE_DIM), np.float32)
        self._next_features = np.zeros_like(self._features)
        self._counts = np.zeros(capacity, np.int64)
        self._next_counts = np.zeros_like(self._counts)

    def _store_sets(self, index, current, following):
        needed = max(len(current), len(following))
        if needed > self._feature_capacity:
            for key in ("_features", "_next_features"):
                previous = getattr(self, key)
                grown = np.zeros((self.capacity, needed, FEATURE_DIM), np.float32)
                grown[:, :self._feature_capacity] = previous
                setattr(self, key, grown)
            self._feature_capacity = needed
        for storage, counts, values in ((self._features, self._counts, current),
                                        (self._next_features, self._next_counts, following)):
            storage[index] = 0.
            storage[index, :len(values)] = values
            counts[index] = len(values)

    def add(self, observation, action, reward, next_observation, done):
        if np.shape(observation) != (LOCAL_OBSERVATION_DIM,) or np.shape(next_observation) != (LOCAL_OBSERVATION_DIM,):
            raise ValueError("local replay requires passage context; global-reaching observations are incompatible")
        for value in (observation, action, reward, next_observation):
            if not np.isfinite(value).all():
                raise ValueError("local replay rejects nonfinite transitions")
        index = self.ptr
        self.observations[index], self.next_observations[index] = observation, next_observation
        self.actions[index], self.rewards[index], self.dones[index] = action, reward, bool(done)
        self.sets[index] = obstacle_set(observation).copy()
        self.next_sets[index] = obstacle_set(next_observation).copy()
        self._store_sets(index, self.sets[index], self.next_sets[index])
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, size):
        if not self.size:
            raise ValueError("cannot sample empty local replay")
        index = self.rng.integers(self.size, size=size)
        counts, next_counts = self._counts[index], self._next_counts[index]
        maximum, next_maximum = int(counts.max(initial=0)), int(next_counts.max(initial=0))
        arrays = [self.observations[index], self.actions[index], self.rewards[index],
                  self.next_observations[index], self.dones[index],
                  self._features[index, :maximum], self._next_features[index, :next_maximum],
                  np.arange(maximum)[None] < counts[:, None],
                  np.arange(next_maximum)[None] < next_counts[:, None]]
        # One host-to-device transfer instead of separate transfers for every
        # state/action/feature/mask array. Contiguous outputs preserve the old
        # network input layouts and the same minibatch/RNG sequence.
        widths = [int(np.prod(array.shape[1:])) for array in arrays]
        packed = np.concatenate([array.reshape(size, width) for array, width in zip(arrays, widths)],
                                axis=1).astype(np.float32, copy=False)
        values = torch.as_tensor(packed, device=self.device).split(widths, dim=1)
        tensors = [value.reshape(array.shape).contiguous() for value, array in zip(values, arrays)]
        observation, action, reward, following, done, features, next_features, mask, next_mask = tensors
        return Batch(observations=observation, actions=action, rewards=reward,
                     costs=torch.zeros((size, 1), device=self.device), next_observations=following,
                     dones=done, obstacle_features=features,
                     obstacle_mask=cache_obstacle_rows(mask.bool(), counts),
                     next_obstacle_features=next_features,
                     next_obstacle_mask=cache_obstacle_rows(next_mask.bool(), next_counts))

    def state_dict(self):
        return {"protocol": self.protocol, "capacity": self.capacity, "ptr": self.ptr, "size": self.size,
                "rng": self.rng.bit_generator.state,
                **{key: getattr(self, key)[:self.size].copy() for key in (
                    "observations", "next_observations", "actions", "rewards", "dones")},
                "sets": self.sets[:self.size], "next_sets": self.next_sets[:self.size]}

    def load_state_dict(self, state):
        if state.get("protocol") != self.protocol or state["capacity"] != self.capacity:
            raise ValueError("incompatible local replay protocol/capacity")
        self.ptr, self.size = state["ptr"], state["size"]
        if not 0 <= self.size <= self.capacity or not 0 <= self.ptr < self.capacity:
            raise ValueError("invalid local replay indices")
        for key in ("observations", "next_observations", "actions", "rewards", "dones"):
            getattr(self, key)[:self.size] = state[key]
        self.sets[:self.size], self.next_sets[:self.size] = state["sets"], state["next_sets"]
        for index in range(self.size):
            self._store_sets(index, self.sets[index], self.next_sets[index])
        self.rng.bit_generator.state = state["rng"]
