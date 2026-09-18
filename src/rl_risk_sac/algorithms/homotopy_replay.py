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
        # environment, joint-limit, Rmax, dmin, contact-before, contact-after,
        # velocity magnitude, episode-fixed orientation scale, self risk,
        # self distance, self violation, self approach, and self TTC.
        self.raw = np.zeros((capacity, 22), np.float32)
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

    def snapshot(self, capacity: int, rng: np.random.Generator) -> _Partition:
        """Return a fixed, uniformly sampled snapshot in chronological order."""
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError("snapshot capacity must be positive")
        result = _Partition(self.obs.shape[1], self.action.shape[1], capacity)
        order = self.chronological_indices()
        if len(order) > capacity:
            selected_positions = np.sort(
                rng.choice(len(order), size=capacity, replace=False)
            )
            order = order[selected_positions]
        for index in order:
            result.add(
                self.obs[index], self.action[index], self.next_obs[index],
                self.done[index, 0], self.raw[index], self.episode[index], self.step[index],
            )
        return result


class HomotopyReplayBuffer:
    """Scene/pose-stratified replay with current safety-weight relabelling.

    S0 uses three physically separate stores: a position-anchor FIFO, a large
    FIFO for the current non-zero eta level, and immutable per-level snapshots
    archived whenever eta advances.  Sampling strata therefore remain backed
    by real data even after a difficult eta level lasts longer than the total
    replay capacity.
    """

    STORAGE_VERSION = 2

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
        s0_anchor_fraction: float = 0.25,
        s0_current_fraction: float = 0.50,
        s0_anchor_capacity: int = 15000,
        s0_current_capacity: int = 30000,
        s0_history_capacity_per_level: int = 3000,
        reward_parameters: dict[str, float] | None = None,
    ) -> None:
        capacities = capacities or {"none": 60000, "static": 90000, "dynamic": 150000}
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device("cuda" if device in {"cuda", "cuda:auto", "auto"} else device)
        self.parts = {
            scene: _Partition(obs_dim, action_dim, capacities[scene])
            for scene in ("static", "dynamic")
        }
        self.s0_anchor = _Partition(obs_dim, action_dim, int(s0_anchor_capacity))
        self.s0_current = _Partition(obs_dim, action_dim, int(s0_current_capacity))
        self.s0_current_scale: float | None = None
        self.s0_history: dict[float, _Partition] = {}
        self.s0_history_capacity_per_level = int(s0_history_capacity_per_level)
        if min(
            self.s0_anchor.capacity,
            self.s0_current.capacity,
            self.s0_history_capacity_per_level,
        ) < 1:
            raise ValueError("all S0 replay storage capacities must be positive")
        self.rng = np.random.default_rng(seed)
        self.reward_gamma = float(reward_gamma)
        self.reward_horizon = int(reward_horizon)
        if not 0.0 <= s0_anchor_fraction < 1.0:
            raise ValueError("s0_anchor_fraction must be in [0, 1)")
        if not 0.0 <= s0_current_fraction <= 1.0 - s0_anchor_fraction:
            raise ValueError("s0_current_fraction leaves an invalid historical fraction")
        self.s0_anchor_fraction = float(s0_anchor_fraction)
        self.s0_current_fraction = float(s0_current_fraction)
        self.reward_parameters = dict(reward_parameters or {})
        self.last_sample_counts = {s: 0 for s in SCENES}
        self.last_orientation_sample_counts = {"anchor": 0, "current": 0, "historical": 0}
        self.redistribution_count = 0

    @staticmethod
    def _scale_key(scale: float) -> float:
        return float(np.round(float(scale), decimals=6))

    def _archive_current_level(self) -> None:
        if self.s0_current_scale is None or self.s0_current.size == 0:
            return
        scale = self._scale_key(self.s0_current_scale)
        if scale in self.s0_history:
            raise ValueError(f"S0 eta level {scale} was archived more than once")
        self.s0_history[scale] = self.s0_current.snapshot(
            self.s0_history_capacity_per_level, self.rng,
        )
        self.s0_current = _Partition(
            self.s0_current.obs.shape[1], self.s0_current.action.shape[1],
            self.s0_current.capacity,
        )

    def begin_orientation_level(self, scale: float) -> None:
        """Atomically move S0 replay storage to a newly activated eta level."""
        scale = self._scale_key(scale)
        if np.isclose(scale, 0.0, rtol=0.0, atol=1e-7):
            return
        if self.s0_current_scale is None:
            self.s0_current_scale = scale
            return
        if np.isclose(scale, self.s0_current_scale, rtol=0.0, atol=1e-7):
            return
        if scale < self.s0_current_scale:
            raise ValueError(
                f"cannot move S0 replay eta backward from {self.s0_current_scale} to {scale}"
            )
        self._archive_current_level()
        self.s0_current_scale = scale

    def add(
        self, scene: str, observation, action, next_observation, terminal: bool,
        info: dict[str, Any], episode_id: int, step_in_episode: int,
        *, orientation_anchor: bool = False,
    ) -> None:
        raw = np.asarray([
            info["rho_position"], info["next_rho_position"], info["rho_orientation"],
            info["next_rho_orientation"], info["smooth_velocity"], float(info["task_reached"]),
            float(info["hard_failure"]), float(info["obstacle_collision"]), float(info["self_collision"]),
            float(info["environment_collision"]), float(info["joint_limit"]), info["control_max_risk"],
            info["control_min_distance"], float(info["obstacle_contact_seen_before"]), float(info["obstacle_contact_seen"]),
            info["velocity_magnitude"], info["orientation_scale"],
            info.get("control_self_max_risk", 0.0),
            info.get("control_self_min_distance", 0.25),
            info.get("self_clearance_violation", 0.0),
            info.get("control_self_max_approach", 0.0),
            info.get("control_self_min_ttc", 3.0),
        ], np.float32)
        if scene != "none":
            self.parts[scene].add(
                observation, action, next_observation, terminal, raw,
                episode_id, step_in_episode,
            )
            return
        scale = self._scale_key(info["orientation_scale"])
        if orientation_anchor or np.isclose(scale, 0.0, rtol=0.0, atol=1e-7):
            self.s0_anchor.add(
                observation, action, next_observation, terminal, raw,
                episode_id, step_in_episode,
            )
            return
        self.begin_orientation_level(scale)
        self.s0_current.add(
            observation, action, next_observation, terminal, raw,
            episode_id, step_in_episode,
        )

    def __len__(self) -> int:
        return (
            self.s0_anchor.size
            + self.s0_current.size
            + sum(part.size for part in self.s0_history.values())
            + sum(part.size for part in self.parts.values())
        )

    def scene_size(self, scene: str) -> int:
        if scene == "none":
            return (
                self.s0_anchor.size
                + self.s0_current.size
                + sum(part.size for part in self.s0_history.values())
            )
        return self.parts[scene].size

    def orientation_storage_counts(self) -> dict[str, Any]:
        return {
            "anchor": self.s0_anchor.size,
            "current": self.s0_current.size,
            "current_scale": self.s0_current_scale,
            "historical": sum(part.size for part in self.s0_history.values()),
            "historical_levels": {
                str(scale): part.size
                for scale, part in sorted(self.s0_history.items())
            },
        }

    def _choice_without_replacement(self, candidates: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or len(candidates) == 0:
            return np.empty(0, dtype=np.int64)
        return self.rng.choice(candidates, size=min(int(count), len(candidates)), replace=False)

    def _stratified_historical_rows(
        self, count: int,
    ) -> list[tuple[_Partition, int, str]]:
        groups = [
            (part, np.arange(part.size, dtype=np.int64))
            for _, part in sorted(self.s0_history.items())
            if part.size
        ]
        selected: list[tuple[_Partition, int, str]] = []
        while len(selected) < count and groups:
            next_groups = []
            for part, available in groups:
                choice = int(self.rng.choice(available))
                selected.append((part, choice, "historical"))
                remaining = available[available != choice]
                if len(remaining):
                    next_groups.append((part, remaining))
                if len(selected) == count:
                    break
            groups = next_groups
        return selected

    def _sample_partition_rows(
        self, part: _Partition, count: int, category: str,
    ) -> list[tuple[_Partition, int, str]]:
        indices = self._choice_without_replacement(
            np.arange(part.size, dtype=np.int64), count,
        )
        return [(part, int(index), category) for index in indices]

    def _fill_s0_rows(
        self, selected: list[tuple[_Partition, int, str]], count: int,
    ) -> list[tuple[_Partition, int, str]]:
        if len(selected) >= count:
            return selected
        used: dict[int, set[int]] = {}
        for part, index, _ in selected:
            used.setdefault(id(part), set()).add(index)
        partitions = [(self.s0_anchor, "anchor")]
        if self.s0_current.size:
            partitions.append((self.s0_current, "current"))
        partitions.extend(
            (part, "historical")
            for _, part in sorted(self.s0_history.items())
        )
        missing = count - len(selected)
        for part, category in partitions:
            blocked = used.get(id(part), set())
            available = np.arange(part.size, dtype=np.int64)
            if blocked:
                mask = np.ones(part.size, dtype=np.bool_)
                mask[np.fromiter(blocked, dtype=np.int64)] = False
                available = available[mask]
            take = min(missing, len(available))
            if take:
                chosen = self.rng.choice(available, size=take, replace=False)
                selected.extend((part, int(index), category) for index in chosen)
                missing -= take
            if missing == 0:
                return selected
        if missing:
            raise ValueError(f"S0 replay has only {count - missing} rows, need {count}")
        return selected

    def _sample_s0_rows(
        self, count: int, orientation_scale: float | None,
        anchor_fraction: float | None = None, current_fraction: float | None = None,
    ) -> list[tuple[_Partition, int, str]]:
        if self.s0_current_scale is None or orientation_scale is None or np.isclose(
            float(orientation_scale), 0.0, rtol=0.0, atol=1e-7,
        ):
            selected = self._sample_partition_rows(self.s0_anchor, count, "anchor")
            selected = self._fill_s0_rows(selected, count)
            self.last_orientation_sample_counts = {
                category: sum(row[2] == category for row in selected)
                for category in ("anchor", "current", "historical")
            }
            self.rng.shuffle(selected)
            return selected
        if not np.isclose(
            float(orientation_scale), self.s0_current_scale, rtol=0.0, atol=1e-7,
        ):
            raise ValueError(
                f"requested current eta {orientation_scale} does not match replay "
                f"current eta {self.s0_current_scale}"
            )
        anchor_fraction = self.s0_anchor_fraction if anchor_fraction is None else float(anchor_fraction)
        current_fraction = self.s0_current_fraction if current_fraction is None else float(current_fraction)
        if not 0.0 <= anchor_fraction < 1.0:
            raise ValueError("anchor_fraction must be in [0, 1)")
        if not 0.0 <= current_fraction <= 1.0 - anchor_fraction:
            raise ValueError("current_fraction leaves an invalid historical fraction")
        desired_anchor = round(count * anchor_fraction)
        desired_current = round(count * current_fraction)
        selected = self._sample_partition_rows(
            self.s0_anchor, desired_anchor, "anchor",
        )
        selected.extend(self._sample_partition_rows(
            self.s0_current, desired_current, "current",
        ))
        selected.extend(self._stratified_historical_rows(count - len(selected)))
        selected = self._fill_s0_rows(selected, count)
        self.last_orientation_sample_counts = {
            category: sum(row[2] == category for row in selected)
            for category in ("anchor", "current", "historical")
        }
        self.rng.shuffle(selected)
        return selected

    def _sample_none_rows(self, count: int) -> list[tuple[_Partition, int, str]]:
        """Sample the retained full-task/anchor data for S1 and S2."""
        partitions = [self.s0_anchor]
        if self.s0_current.size:
            partitions.append(self.s0_current)
        retained = sum(part.size for part in partitions)
        for _, part in sorted(self.s0_history.items(), reverse=True):
            if retained >= count:
                break
            partitions.append(part)
            retained += part.size
        sizes = np.asarray([part.size for part in partitions], dtype=np.int64)
        if int(sizes.sum()) < count:
            raise ValueError(f"no-obstacle replay has only {int(sizes.sum())} rows, need {count}")
        remaining = count
        remaining_total = int(sizes.sum())
        rows: list[tuple[_Partition, int, str]] = []
        for index, (part, size) in enumerate(zip(partitions, sizes)):
            if index == len(partitions) - 1:
                take = remaining
            else:
                take = int(self.rng.hypergeometric(int(size), remaining_total - int(size), remaining))
            rows.extend(self._sample_partition_rows(part, take, "none"))
            remaining -= take
            remaining_total -= int(size)
        return rows

    def sample(
        self, stage: str, xi: dict[str, float], batch_size: int = 256,
        *, orientation_scale: float | None = None,
        s0_anchor_fraction: float | None = None,
        s0_current_fraction: float | None = None,
        lambda_self: float = 1.0,
    ) -> Batch:
        ratios = {"s0": (1, 0, 0), "s1": (64, 192, 0), "s2": (51, 77, 128)}[stage]
        desired = np.asarray(ratios, dtype=int)
        if batch_size != 256:
            desired = np.floor(np.asarray(ratios) * batch_size / 256).astype(int)
            desired[0] += batch_size - int(desired.sum())
        scene_sizes = np.asarray([self.scene_size(scene) for scene in SCENES])
        counts = np.minimum(desired, scene_sizes)
        missing = batch_size - int(counts.sum())
        available = scene_sizes - counts
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
                if stage == "s0" and scene == "none":
                    selected = self._sample_s0_rows(
                        int(count), orientation_scale,
                        s0_anchor_fraction, s0_current_fraction,
                    )
                    rows.extend((scene, part, index) for part, index, _ in selected)
                elif scene == "none":
                    self.last_orientation_sample_counts = {
                        "anchor": 0, "current": 0, "historical": 0,
                    }
                    selected = self._sample_none_rows(int(count))
                    rows.extend((scene, part, index) for part, index, _ in selected)
                else:
                    part = self.parts[scene]
                    index = self.rng.choice(part.size, size=int(count), replace=False)
                    rows.extend((scene, part, int(i)) for i in index)
        self.rng.shuffle(rows)
        rewards, costs = [], []
        for scene, part, i in rows:
            r = part.raw[i]
            reward, fields = homotopy_reward(
                rho_position=r[0], next_rho_position=r[1], rho_orientation=r[2], next_rho_orientation=r[3],
                smooth_velocity=r[4], task_reached=bool(r[5]), hard_failure=bool(r[6]),
                velocity_magnitude=r[15], orientation_scale=r[16],
                obstacle_collision=bool(r[7]),
                terminal_obstacle_collision=bool(r[7] and part.done[i, 0] and not r[5] and not r[6]),
                risk_max=r[11], distance_min=r[12],
                self_risk_max=r[17], self_distance_min=r[18],
                lambda_self=lambda_self,
                xi=1.0 if scene == "none" else xi[scene],
                gamma=self.reward_gamma, horizon=self.reward_horizon,
                **self.reward_parameters,
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
        if scene == "none":
            raise ValueError("no-obstacle replay is banked and cannot be strictified")
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
        digest = hashlib.sha256()
        if scene == "none":
            partitions = [("anchor", self.s0_anchor), ("current", self.s0_current)]
            partitions.extend(
                (f"history:{scale}", part)
                for scale, part in sorted(self.s0_history.items())
            )
        else:
            partitions = [(scene, self.parts[scene])]
        for label, part in partitions:
            digest.update(label.encode("utf-8"))
            order = part.chronological_indices()
            for field in (part.obs, part.action, part.next_obs, part.done, part.raw, part.episode, part.step):
                digest.update(np.ascontiguousarray(field[order]).tobytes())
        return digest.hexdigest()

    def state_dict(self) -> dict[str, Any]:
        return {"storage_version": self.STORAGE_VERSION,
                "parts": self.parts,
                "s0_anchor": self.s0_anchor,
                "s0_current": self.s0_current,
                "s0_current_scale": self.s0_current_scale,
                "s0_history": self.s0_history,
                "s0_history_capacity_per_level": self.s0_history_capacity_per_level,
                "rng_state": self.rng.bit_generator.state,
                "reward_gamma": self.reward_gamma, "reward_horizon": self.reward_horizon,
                "s0_anchor_fraction": self.s0_anchor_fraction,
                "s0_current_fraction": self.s0_current_fraction,
                "reward_parameters": self.reward_parameters,
                "last_sample_counts": self.last_sample_counts,
                "last_orientation_sample_counts": self.last_orientation_sample_counts,
                "redistribution_count": self.redistribution_count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("storage_version", 0)) != self.STORAGE_VERSION:
            raise ValueError(
                "checkpoint replay storage predates persistent S0 eta banks"
            )
        self.parts = state["parts"]
        self.s0_anchor = state["s0_anchor"]
        self.s0_current = state["s0_current"]
        self.s0_current_scale = state["s0_current_scale"]
        self.s0_history = state["s0_history"]
        self.s0_history_capacity_per_level = int(
            state["s0_history_capacity_per_level"]
        )
        self.rng.bit_generator.state = state["rng_state"]
        self.reward_gamma = float(state.get("reward_gamma", self.reward_gamma))
        self.reward_horizon = int(state.get("reward_horizon", self.reward_horizon))
        self.reward_parameters = dict(state.get("reward_parameters", self.reward_parameters))
        # Sampling policy is a property of the current protocol, not old replay data.
        self.last_sample_counts = state.get("last_sample_counts", {s: 0 for s in SCENES})
        self.last_orientation_sample_counts = state.get(
            "last_orientation_sample_counts", {"anchor": 0, "current": 0, "historical": 0},
        )
        self.redistribution_count = state.get("redistribution_count", 0)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
