from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int,
        device: str,
        stratified_fraction: float = 0.0,
        action_signature: str | None = None,
    ) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.stratified_fraction = float(stratified_fraction)
        if not 0.0 <= self.stratified_fraction <= 1.0:
            raise ValueError("stratified_fraction must be in [0, 1]")
        self.action_signature = action_signature
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.costs = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        # ``dones`` is the Bellman terminal mask. Time-limit truncations are
        # deliberately stored separately so they can still bootstrap.
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.truncateds = np.zeros((capacity, 1), dtype=np.float32)
        # -1 means the episode outcome is not known yet, 0 is failure, 1 success.
        # The training loop retroactively labels every transition in an episode.
        self.episode_outcomes = np.full(capacity, -1, dtype=np.int8)
        self.ptr = 0
        self.size = 0

    @staticmethod
    def metadata_string(value: object) -> str:
        array = np.asarray(value)
        if array.shape == ():
            return str(array.item())
        if array.size == 1:
            return str(array.reshape(-1)[0])
        return str(value)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        cost: float,
        next_observation: np.ndarray,
        done: bool,
        truncated: bool = False,
    ) -> int:
        index = self.ptr
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.costs[self.ptr] = cost
        self.next_observations[self.ptr] = next_observation
        self.dones[self.ptr] = float(done)
        self.truncateds[self.ptr] = float(truncated)
        self.episode_outcomes[self.ptr] = -1
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return index

    def label_episode(self, indices: list[int], success: bool) -> None:
        """Label all transitions from a completed episode for stratified sampling."""
        if not indices:
            return
        valid = np.asarray(indices, dtype=np.int64)
        if np.any(valid < 0) or np.any(valid >= self.capacity):
            raise ValueError("episode transition index is outside replay capacity")
        self.episode_outcomes[valid] = 1 if success else 0

    def sample(self, batch_size: int) -> Batch:
        if self.size < 1:
            raise ValueError("Cannot sample from an empty replay buffer")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = self._sample_indices(batch_size)
        return Batch(
            observations=self._tensor(self.observations[indices]),
            actions=self._tensor(self.actions[indices]),
            rewards=self._tensor(self.rewards[indices]),
            costs=self._tensor(self.costs[indices]),
            next_observations=self._tensor(self.next_observations[indices]),
            dones=self._tensor(self.dones[indices]),
        )

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        if self.stratified_fraction <= 0.0 or self.size < 2:
            return np.random.randint(0, self.size, size=batch_size)

        desired = min(batch_size, int(round(batch_size * self.stratified_fraction)))
        if desired <= 0:
            return np.random.randint(0, self.size, size=batch_size)
        known = self.episode_outcomes[: self.size]
        success_indices = np.flatnonzero(known == 1)
        failure_indices = np.flatnonzero(known == 0)
        if not len(success_indices) and not len(failure_indices):
            return np.random.randint(0, self.size, size=batch_size)

        half = desired // 2
        selected: list[np.ndarray] = []
        if len(success_indices) and half:
            selected.append(np.random.choice(success_indices, size=half, replace=True))
        if len(failure_indices) and desired - half:
            selected.append(np.random.choice(failure_indices, size=desired - half, replace=True))
        selected_count = sum(len(values) for values in selected)
        if selected_count < desired:
            selected.append(np.random.randint(0, self.size, size=desired - selected_count))
        remaining = batch_size - desired
        if remaining:
            selected.append(np.random.randint(0, self.size, size=remaining))
        indices = np.concatenate(selected).astype(np.int64, copy=False)
        np.random.shuffle(indices)
        return indices

    def state_dict(self) -> dict[str, object]:
        stored = self.capacity if self.size == self.capacity else self.size
        return {
            "version": 2,
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "ptr": self.ptr,
            "size": self.size,
            "stratified_fraction": self.stratified_fraction,
            "action_signature": "" if self.action_signature is None else self.action_signature,
            "observations": self.observations[:stored].copy(),
            "actions": self.actions[:stored].copy(),
            "rewards": self.rewards[:stored].copy(),
            "costs": self.costs[:stored].copy(),
            "next_observations": self.next_observations[:stored].copy(),
            "dones": self.dones[:stored].copy(),
            "truncateds": self.truncateds[:stored].copy(),
            "episode_outcomes": self.episode_outcomes[:stored].copy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        source_capacity = int(state["capacity"])
        source_size = int(state["size"])
        if int(state["obs_dim"]) != self.obs_dim or int(state["action_dim"]) != self.action_dim:
            raise ValueError("ReplayBuffer dimensions do not match the checkpoint")
        checkpoint_action_signature = self.metadata_string(state.get("action_signature", ""))
        if self.action_signature is not None and checkpoint_action_signature:
            if checkpoint_action_signature != self.action_signature:
                raise ValueError("ReplayBuffer action semantics do not match the current config")
        if source_size > self.capacity:
            raise ValueError(
                f"Replay checkpoint contains {source_size} transitions but capacity is only {self.capacity}"
            )

        source_ptr = int(state["ptr"])
        arrays = {
            "observations": np.asarray(state["observations"], dtype=np.float32),
            "actions": np.asarray(state["actions"], dtype=np.float32),
            "rewards": np.asarray(state["rewards"], dtype=np.float32),
            "costs": np.asarray(state["costs"], dtype=np.float32),
            "next_observations": np.asarray(state["next_observations"], dtype=np.float32),
            "dones": np.asarray(state["dones"], dtype=np.float32),
            "truncateds": np.asarray(
                state.get("truncateds", np.zeros((source_size, 1))), dtype=np.float32
            ),
            "episode_outcomes": np.asarray(
                state.get("episode_outcomes", np.full(source_size, -1)), dtype=np.int8
            ),
        }
        if source_size == source_capacity and self.capacity != source_capacity:
            order = np.concatenate((np.arange(source_ptr, source_capacity), np.arange(0, source_ptr)))
            target_ptr = source_size % self.capacity
        else:
            order = np.arange(source_size)
            target_ptr = source_ptr if self.capacity == source_capacity else source_size % self.capacity
        for name, values in arrays.items():
            values = values[order]
            target = getattr(self, name)
            target[...] = -1 if name == "episode_outcomes" else 0
            target[:source_size] = values
        self.size = source_size
        self.ptr = target_ptr
        self.stratified_fraction = float(state.get("stratified_fraction", self.stratified_fraction))
        self.action_signature = checkpoint_action_signature or self.action_signature

    def save(self, path: str | Path) -> None:
        """Persist replay data without requiring torch pickle deserialization."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        np.savez_compressed(target, **state)

    def load(self, path: str | Path) -> None:
        source = Path(path)
        with np.load(source, allow_pickle=False) as payload:
            state: dict[str, object] = {key: payload[key] for key in payload.files}
        self.load_state_dict(state)

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, device=self.device, dtype=torch.float32)

    def __len__(self) -> int:
        return self.size
