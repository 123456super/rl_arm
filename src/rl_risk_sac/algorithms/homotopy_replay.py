from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import torch

from rl_risk_sac.algorithms.replay_buffer import Batch
from rl_risk_sac.tasks.thesis_reaching import homotopy_reward


SCENES = ("none", "static", "dynamic")
SCENE_ID = {name: index for index, name in enumerate(SCENES)}


class _Partition:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int) -> None:
        self.capacity, self.ptr, self.size = int(capacity), 0, 0
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.action = np.zeros((capacity, action_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)
        # rho_p(2), rho_R(2), smooth, task, hard, obstacle, self,
        # environment, joint-limit, Rmax, dmin, contact-before, contact-after.
        self.raw = np.zeros((capacity, 15), np.float32)
        self.episode = np.zeros(capacity, np.int64)
        self.step = np.zeros(capacity, np.int32)

    def add(self, obs, action, next_obs, done, raw, episode, step) -> None:
        index = self.ptr
        self.obs[index], self.action[index], self.next_obs[index] = obs, action, next_obs
        self.done[index], self.raw[index] = float(done), raw
        self.episode[index], self.step[index] = episode, step
        self.ptr = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def chronological_indices(self) -> np.ndarray:
        if self.size < self.capacity:
            return np.arange(self.size)
        return np.concatenate((np.arange(self.ptr, self.capacity), np.arange(self.ptr)))

    def compact(self, keep: np.ndarray, done_override: np.ndarray | None = None) -> None:
        order = self.chronological_indices()
        selected = order[keep]
        size = len(selected)
        for field in ("obs", "action", "next_obs", "raw", "episode", "step"):
            array = getattr(self, field)
            array[:size] = array[selected].copy()
        values = self.done[selected].copy()
        if done_override is not None:
            values[:] = done_override.reshape(-1, 1)
        self.done[:size] = values
        self.size, self.ptr = size, size % self.capacity


class HomotopyReplayBuffer:
    """Scene-stratified replay with current-xi reward relabelling."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str,
        capacities=None,
        seed: int = 0,
        *,
        reward_gamma: float = 0.99,
        reward_horizon: int = 240,
    ) -> None:
        capacities = capacities or {"none": 60000, "static": 90000, "dynamic": 150000}
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device("cuda" if device in {"cuda", "cuda:auto", "auto"} else device)
        self.parts = {s: _Partition(obs_dim, action_dim, capacities[s]) for s in SCENES}
        self.rng = np.random.default_rng(seed)
        self.reward_gamma = float(reward_gamma)
        self.reward_horizon = int(reward_horizon)
        self.last_sample_counts = {s: 0 for s in SCENES}
        self.redistribution_count = 0

    def add(self, scene: str, observation, action, next_observation, terminal: bool, info: dict[str, Any], episode_id: int, step_in_episode: int) -> None:
        raw = np.asarray([
            info["rho_position"], info["next_rho_position"], info["rho_orientation"],
            info["next_rho_orientation"], info["smooth_velocity"], float(info["task_reached"]),
            float(info["hard_failure"]), float(info["obstacle_collision"]), float(info["self_collision"]),
            float(info["environment_collision"]), float(info["joint_limit"]), info["control_max_risk"],
            info["control_min_distance"], float(info["obstacle_contact_seen_before"]), float(info["obstacle_contact_seen"]),
        ], np.float32)
        self.parts[scene].add(observation, action, next_observation, terminal, raw, episode_id, step_in_episode)

    def __len__(self) -> int:
        return sum(p.size for p in self.parts.values())

    def sample(self, stage: str, xi: dict[str, float], batch_size: int = 256) -> Batch:
        ratios = {"s0": (1, 0, 0), "s1": (64, 192, 0), "s2": (51, 77, 128)}[stage]
        desired = np.asarray(ratios, dtype=int)
        if batch_size != 256:
            desired = np.floor(np.asarray(ratios) * batch_size / 256).astype(int)
            desired[0] += batch_size - int(desired.sum())
        counts = np.minimum(desired, np.asarray([self.parts[s].size for s in SCENES]))
        missing = batch_size - int(counts.sum())
        available = np.asarray([self.parts[s].size - counts[i] for i, s in enumerate(SCENES)], dtype=int)
        redistributed = missing > 0
        while missing and available.sum() > 0:
            shares = missing * available / available.sum()
            allocation = np.minimum(available, np.floor(shares).astype(int))
            if allocation.sum() == 0:
                candidates = np.flatnonzero(available > 0)
                allocation[candidates[:missing]] = 1
            counts += allocation; available -= allocation; missing -= int(allocation.sum())
        if missing:
            raise ValueError(f"replay has only {batch_size - missing} transitions, need {batch_size}")
        self.last_sample_counts = {scene: int(count) for scene, count in zip(SCENES, counts)}
        if redistributed:
            self.redistribution_count += 1
        rows = []
        for scene, count in zip(SCENES, counts):
            if count:
                part = self.parts[scene]
                index = self.rng.choice(part.size, size=int(count), replace=False)
                for i in index:
                    rows.append((scene, part, int(i)))
        self.rng.shuffle(rows)
        rewards, costs = [], []
        for scene, part, i in rows:
            r = part.raw[i]
            reward, fields = homotopy_reward(
                rho_position=r[0], next_rho_position=r[1], rho_orientation=r[2], next_rho_orientation=r[3],
                smooth_velocity=r[4], task_reached=bool(r[5]), hard_failure=bool(r[6]),
                obstacle_collision=bool(r[7]),
                terminal_obstacle_collision=bool(r[7] and part.done[i, 0] and not r[5] and not r[6]),
                risk_max=r[11], distance_min=r[12],
                xi=1.0 if scene == "none" else xi[scene],
                gamma=self.reward_gamma, horizon=self.reward_horizon,
            )
            rewards.append(reward); costs.append(fields["c_proximity"])
        tensor = lambda value: torch.as_tensor(np.asarray(value), device=self.device, dtype=torch.float32)
        return Batch(
            observations=tensor([p.obs[i] for _, p, i in rows]), actions=tensor([p.action[i] for _, p, i in rows]),
            rewards=tensor(rewards).reshape(-1, 1), costs=tensor(costs).reshape(-1, 1),
            next_observations=tensor([p.next_obs[i] for _, p, i in rows]),
            dones=tensor([p.done[i] for _, p, i in rows]).reshape(-1, 1),
        )

    def strictify(self, scene: str) -> dict[str, Any]:
        """Cut each stored trajectory at its first obstacle contact."""
        part = self.parts[scene]; before_hash = self.partition_sha256(scene)
        order = part.chronological_indices()
        keep = np.ones(len(order), dtype=bool); dones = part.done[order, 0].copy()
        blocked: set[int] = set()
        for j, index in enumerate(order):
            episode = int(part.episode[index]); before, after = bool(part.raw[index, 13]), bool(part.raw[index, 14])
            if episode in blocked or before:
                keep[j] = False; continue
            if after:
                dones[j] = 1.0; part.raw[index, 5] = 0.0; blocked.add(episode)
        old = part.size
        part.compact(keep, dones[keep])
        return {"before": old, "after": part.size, "removed": old - part.size,
                "sha256_before": before_hash, "sha256_after": self.partition_sha256(scene)}

    def partition_sha256(self, scene: str) -> str:
        part = self.parts[scene]; digest = hashlib.sha256()
        order = part.chronological_indices()
        for field in (part.obs, part.action, part.next_obs, part.done, part.raw, part.episode, part.step):
            digest.update(np.ascontiguousarray(field[order]).tobytes())
        return digest.hexdigest()

    def state_dict(self) -> dict[str, Any]:
        return {"parts": self.parts, "rng_state": self.rng.bit_generator.state,
                "reward_gamma": self.reward_gamma, "reward_horizon": self.reward_horizon,
                "last_sample_counts": self.last_sample_counts, "redistribution_count": self.redistribution_count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.parts = state["parts"]; self.rng.bit_generator.state = state["rng_state"]
        self.reward_gamma = float(state.get("reward_gamma", self.reward_gamma))
        self.reward_horizon = int(state.get("reward_horizon", self.reward_horizon))
        self.last_sample_counts = state.get("last_sample_counts", {s: 0 for s in SCENES})
        self.redistribution_count = state.get("redistribution_count", 0)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
