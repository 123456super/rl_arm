"""Raw variable-size obstacle sets transported alongside the Full state.

The 128 external columns are placeholders, never cached neural embeddings.
Each policy/critic substitutes its own current set encoder on every forward.
"""
from __future__ import annotations

import numpy as np
import torch

FEATURE_DIM = 37
EXTERNAL_START = 89
EXTERNAL_DIM = 128
FULL_OBSERVATION_DIM = 241


class ObstacleObservation(np.ndarray):
    def __new__(cls, state, features=()):
        obj = np.asarray(state, dtype=np.float32).view(cls)
        obj.obstacle_features = np.asarray(features, dtype=np.float32).reshape(-1, FEATURE_DIM).copy()
        return obj

    def __array_finalize__(self, source):
        self.obstacle_features = getattr(source, "obstacle_features", np.empty((0, FEATURE_DIM), np.float32))

    def __reduce__(self):
        # ndarray's default pickle drops additional attributes (worker pipes).
        return (type(self), (np.asarray(self), self.obstacle_features))


class ObstacleBatch(np.ndarray):
    def __new__(cls, states, sets):
        obj = np.asarray(states, dtype=np.float32).view(cls)
        obj.obstacle_sets = tuple(sets)
        return obj

    def __array_finalize__(self, source):
        self.obstacle_sets = getattr(source, "obstacle_sets", ())

    def __reduce__(self):
        return (type(self), (np.asarray(self), self.obstacle_sets))


def obstacle_set(observation):
    return getattr(observation, "obstacle_features", np.empty((0, FEATURE_DIM), np.float32))


def stack_observations(observations):
    return ObstacleBatch(np.stack([np.asarray(o) for o in observations]),
                         [obstacle_set(o) for o in observations])


def pad_obstacle_sets(sets, device):
    """Only this minibatch's largest set determines temporary padding."""
    sets = tuple(sets)
    maximum = max((len(s) for s in sets), default=0)
    features = np.zeros((len(sets), maximum, FEATURE_DIM), np.float32)
    mask = np.zeros((len(sets), maximum), bool)
    for i, values in enumerate(sets):
        features[i, :len(values)] = values
        mask[i, :len(values)] = True
    return (torch.as_tensor(features, device=device), torch.as_tensor(mask, device=device))


def observation_tensors(observations, device):
    sets = getattr(observations, "obstacle_sets", None)
    if sets is None:
        sets = [np.empty((0, FEATURE_DIM), np.float32) for _ in range(len(observations))]
    features, mask = pad_obstacle_sets(sets, device)
    return torch.as_tensor(np.asarray(observations), device=device), features, mask
