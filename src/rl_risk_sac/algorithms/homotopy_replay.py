from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import torch

from rl_risk_sac.algorithms.replay_buffer import Batch
SCENES = ("none", "static", "dynamic")
SCENE_ID = {name: index for index, name in enumerate(SCENES)}


def _vectorized_replay_rewards(
    raw: np.ndarray,
    dones: np.ndarray,
    xi: np.ndarray,
    lambda_self: float,
    reward_gamma: float,
    reward_parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute replay rewards for a whole batch without Python row loops.

    This is the array form of ``thesis_reaching.homotopy_reward``.  Replay only
    consumes the scalar reward and ``c_proximity``, so calculating the other
    diagnostic fields here would add work without changing the training batch.
    Float64 intermediates match the scalar function's Python-float arithmetic;
    the caller performs the existing float32 conversion when creating tensors.
    """
    r = np.asarray(raw, dtype=np.float64)
    done = np.asarray(dones, dtype=np.float64).reshape(-1)
    scene_xi = np.asarray(xi, dtype=np.float64).reshape(-1)
    p = {
        "d_safe": 0.12, "d_self_safe": 0.005,
        "joint_precision_temperature": 1.0,
        "keypoint_pose_reward": True,
        "keypoint_tracking_scale": 0.20,
        "keypoint_progress_scale": 10.0,
        "keypoint_precision_reward_scale": 0.0,
        "success_bonus": 20.0, "velocity_cost_weight": 0.04,
        "smooth_cost_weight": 0.01, "hard_failure_penalty": 20.0,
        "timeout_penalty": 2.0,
        "safety_risk_weight": 2.0, "safety_clearance_weight": 8.0,
        "self_risk_weight": 2.0, "self_clearance_weight": 8.0,
        "external_safety_scale": 0.05, "self_safety_scale": 0.05,
        **reward_parameters,
    }
    if (
        not p["keypoint_pose_reward"]
        or p.get("rtpc_orientation_reward", False)
        or p.get("pose_balanced_rtpc_reward", False)
    ):
        raise ValueError("only unified keypoint pose reward is supported")
    if r.shape[1] < 29:
        raise ValueError("keypoint replay rows require current/next distance and tracking quality")
    next_tolerance_ratio = np.maximum(
        np.maximum(r[:, 1], 0.0) / r[:, 24],
        np.maximum(r[:, 3], 0.0) / r[:, 25],
    )
    velocity_cost = float(p["velocity_cost_weight"]) * np.clip(r[:, 15], 0.0, 1.0)
    smooth_cost = float(p["smooth_cost_weight"]) * np.clip(r[:, 4], 0.0, 1.0)
    task_reached = r[:, 5] != 0.0
    hard_failure = r[:, 6] != 0.0
    obstacle_collision = r[:, 7] != 0.0
    timeout = r[:, 22] != 0.0
    r_goal = (
        float(p["keypoint_tracking_scale"]) * np.clip(r[:, 28], 0.0, 1.0)
        + float(p["keypoint_progress_scale"]) * (r[:, 26] - r[:, 27])
        + float(p["keypoint_precision_reward_scale"])
        * np.exp(-next_tolerance_ratio / float(p["joint_precision_temperature"]))
        - velocity_cost
        - smooth_cost
        + float(p["success_bonus"]) * task_reached
    )

    clearance = np.clip(
        (float(p["d_safe"]) - r[:, 12]) / float(p["d_safe"]), 0.0, 1.0,
    )
    risk = np.clip(r[:, 11], 0.0, 1.0)
    self_clearance = np.clip(
        (float(p["d_self_safe"]) - r[:, 18]) / float(p["d_self_safe"]),
        0.0,
        1.0,
    )
    self_risk = np.clip(r[:, 17], 0.0, 1.0)
    proximity = risk + clearance + self_risk + self_clearance
    external_cost = (
        float(p["safety_risk_weight"]) * risk
        + float(p["safety_clearance_weight"]) * clearance
    ) / (float(p["safety_risk_weight"]) + float(p["safety_clearance_weight"]))
    self_cost = (
        float(p["self_risk_weight"]) * self_risk
        + float(p["self_clearance_weight"]) * self_clearance
    ) / (float(p["self_risk_weight"]) + float(p["self_clearance_weight"]))
    safety_penalty = (
        scene_xi * float(p["external_safety_scale"]) * external_cost
        + float(lambda_self) * float(p["self_safety_scale"]) * self_cost
    )
    hard_penalty = float(p["hard_failure_penalty"]) * hard_failure
    timeout_penalty = float(p["timeout_penalty"]) * (
        timeout & ~task_reached & ~hard_failure
    )
    terminal_obstacle = obstacle_collision & (done != 0.0) & ~task_reached & ~hard_failure
    terminal_guard = float(p["hard_failure_penalty"]) * terminal_obstacle
    rewards = r_goal - hard_penalty - timeout_penalty - safety_penalty - terminal_guard
    return rewards, proximity


class _Partition:
    def __init__(
        self, obs_dim: int, action_dim: int, capacity: int, raw_dim: int = 29,
    ) -> None:
        self.capacity, self.ptr, self.size = int(capacity), 0, 0
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.action = np.zeros((capacity, action_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)
        # rho_p(2), rho_R(2), smooth, task, hard, obstacle, self,
        # environment, joint-limit, Rmax, dmin, contact-before, contact-after,
        # velocity magnitude, episode-fixed orientation scale, self risk,
        # self distance, self violation, self approach, self TTC, timeout,
        # curriculum level, episode position tolerance, and episode orientation
        # tolerance.  The final three fields make replay reward relabelling use
        # the contract that actually generated each transition. Keypoint-mode
        # partitions append D_KP(t), D_KP(t+1), and Q_KP(t+1).
        self.raw = np.zeros((capacity, int(raw_dim)), np.float32)
        self.episode = np.zeros(capacity, np.int64)
        self.step = np.zeros(capacity, np.int32)
        # Built on first episode-balanced sample, then maintained incrementally.
        # This avoids repeatedly evaluating one full-buffer boolean mask per
        # episode (an O(transitions * episodes) operation).
        self._episode_rows: dict[int, set[int]] | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # The index is derived data and can be rebuilt after checkpoint load.
        state["_episode_rows"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._episode_rows = None

    def episode_rows(self) -> dict[int, set[int]]:
        cached = getattr(self, "_episode_rows", None)
        if cached is None:
            cached = {}
            valid = (
                np.arange(self.size, dtype=np.int64)
                if self.size < self.capacity
                else np.arange(self.capacity, dtype=np.int64)
            )
            for index in valid:
                cached.setdefault(int(self.episode[index]), set()).add(int(index))
            self._episode_rows = cached
        return cached

    def add(self, obs, action, next_obs, done, raw, episode, step) -> None:
        index = self.ptr
        cached = getattr(self, "_episode_rows", None)
        if cached is not None and self.size == self.capacity:
            old_episode = int(self.episode[index])
            cached[old_episode].remove(index)
            if not cached[old_episode]:
                del cached[old_episode]
        self.obs[index], self.action[index], self.next_obs[index] = obs, action, next_obs
        self.done[index], self.raw[index] = float(done), raw
        self.episode[index], self.step[index] = episode, step
        if cached is not None:
            cached.setdefault(int(episode), set()).add(index)
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
        self._episode_rows = None

    def snapshot(self, capacity: int, rng: np.random.Generator) -> _Partition:
        """Return a fixed, uniformly sampled snapshot in chronological order."""
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError("snapshot capacity must be positive")
        result = _Partition(
            self.obs.shape[1], self.action.shape[1], capacity,
            raw_dim=self.raw.shape[1],
        )
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

    def episode_indices(self, episode: int) -> np.ndarray:
        indices = np.fromiter(
            self.episode_rows().get(int(episode), ()), dtype=np.int64,
        )
        if len(indices):
            indices = indices[np.argsort(self.step[indices], kind="stable")]
        return indices

    def copy_episode_to(self, episode: int, destination: _Partition) -> None:
        for index in self.episode_indices(episode):
            destination.add(
                self.obs[index], self.action[index], self.next_obs[index],
                self.done[index, 0], self.raw[index], self.episode[index], self.step[index],
            )


class HomotopyReplayBuffer:
    """Scene/pose-stratified replay with current safety-weight relabelling.

    V12 S0 separates online previous-level anchors, current recent data,
    protected current successes, and immutable successful level snapshots.
    Current data is sampled episode-uniformly so long timeouts do not dominate
    merely because they contain more transitions.
    """

    STORAGE_VERSION = 6

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str,
        capacities=None,
        seed: int = 0,
        *,
        reward_gamma: float = 0.99,
        reward_horizon: int = 500,
        s0_anchor_fraction: float = 0.25,
        s0_current_fraction: float = 0.50,
        s0_anchor_capacity: int = 15000,
        s0_current_capacity: int = 30000,
        s0_history_capacity_per_level: int = 3000,
        s0_joint_pose: bool = False,
        s0_success_fraction: float = 0.30,
        s0_success_schedule: dict[str, float | int] | None = None,
        s0_success_capacity: int = 15000,
        reward_parameters: dict[str, float] | None = None,
    ) -> None:
        capacities = capacities or {"none": 60000, "static": 90000, "dynamic": 150000}
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device("cuda" if device in {"cuda", "cuda:auto", "auto"} else device)
        self.reward_parameters = dict(reward_parameters or {})
        if (
            not self.reward_parameters.get("keypoint_pose_reward", True)
            or self.reward_parameters.get("rtpc_orientation_reward", False)
            or self.reward_parameters.get("pose_balanced_rtpc_reward", False)
        ):
            raise ValueError("only unified keypoint pose reward is supported")
        self.raw_dim = 29
        self.parts = {
            scene: _Partition(obs_dim, action_dim, capacities[scene], self.raw_dim)
            for scene in ("static", "dynamic")
        }
        self.s0_anchor = _Partition(obs_dim, action_dim, int(s0_anchor_capacity), self.raw_dim)
        self.s0_current = _Partition(obs_dim, action_dim, int(s0_current_capacity), self.raw_dim)
        self.s0_current_success = _Partition(
            obs_dim, action_dim, int(s0_success_capacity), self.raw_dim
        )
        self.s0_current_scale: float | None = None
        self.s0_history: dict[float, _Partition] = {}
        self.s0_history_capacity_per_level = int(s0_history_capacity_per_level)
        if min(
            self.s0_anchor.capacity,
            self.s0_current.capacity,
            self.s0_current_success.capacity,
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
        self.s0_joint_pose = bool(s0_joint_pose)
        self.s0_success_fraction = float(s0_success_fraction)
        if not 0.0 <= self.s0_success_fraction < 1.0:
            raise ValueError("s0_success_fraction must be in [0, 1)")
        self.s0_success_schedule = self._validate_success_schedule(
            s0_success_schedule
        )
        self.last_sample_counts = {s: 0 for s in SCENES}
        self.last_orientation_sample_counts = {
            "anchor": 0, "current": 0, "historical": 0,
            "current_success": 0, "current_recent": 0, "previous": 0,
        }
        self.redistribution_count = 0
        # Task-semantic long-term replay is part of the formal S0 protocol.
        self.semantic_enabled = False
        self.semantic_fraction = 0.0
        self.semantic_position_bins = 0
        self.semantic_orientation_bins = 0
        self.semantic_episodes_per_bin = 0
        self.semantic_transitions_per_episode = 0
        self.semantic_bounds: tuple[float, float, float, float] | None = None
        self.semantic_slots: dict[int, list[_Partition]] = {}
        self.semantic_seen = np.zeros(0, dtype=np.int64)
        self.semantic_rng = np.random.default_rng(seed ^ 0x5EED5EED)
        self._loaded_semantic_state: dict[str, Any] | None = None
        self.four_pool_enabled = False
        self.stability_anchor: dict[str, _Partition] = {}
        self.frontier: dict[int, _Partition] = {}
        self.four_pool_source: str | None = None
        self.four_pool_history_mode = False
        self.four_pool_current_fraction = .5

    def promote_four_pool_to_history(
        self, scale: float, *, current_fraction: float = .5,
    ) -> None:
        """Carry the four-pool L0 online data and protected examples into L1 history."""
        if not self.four_pool_enabled or self.four_pool_history_mode:
            raise ValueError("four-pool history promotion requires an active four-pool level")
        if not 0. <= current_fraction <= 1.:
            raise ValueError("promoted current replay fraction must be in [0, 1]")
        if self.s0_current_scale is None or self.s0_current.size == 0:
            raise ValueError("cannot promote an empty four-pool online replay")
        previous_scale = self._scale_key(self.s0_current_scale)
        self.s0_history_capacity_per_level = max(self.s0_history_capacity_per_level, 60000)
        self.begin_orientation_level(scale)
        history = self.s0_history[previous_scale]
        available = history.capacity - history.size
        # Preserve every online row first; use remaining space for frozen
        # successful anchors and hard-state examples from the previous level.
        anchor_budget = min(sum(part.size for part in self.stability_anchor.values()),
                            available * 3 // 4)
        tier_fractions = (("o0_o4", .4), ("o5_o6", .3), ("o7_o9", .3))
        anchor_quotas = {
            tier: min(self.stability_anchor[tier].size, round(anchor_budget * fraction))
            for tier, fraction in tier_fractions
        }
        for tier, _ in tier_fractions:
            remaining = anchor_budget - sum(anchor_quotas.values())
            anchor_quotas[tier] += min(
                remaining, self.stability_anchor[tier].size - anchor_quotas[tier],
            )
        for tier, _ in tier_fractions:
            part = self.stability_anchor[tier]
            for _, index, _ in self._sample_episode_balanced_rows(
                part, anchor_quotas[tier], "previous",
            ):
                history.add(part.obs[index], part.action[index], part.next_obs[index],
                            part.done[index, 0], part.raw[index], part.episode[index], part.step[index])
        centers = list(self.frontier)
        self.rng.shuffle(centers)
        frontier_budget = history.capacity - history.size
        for ordinal, center in enumerate(centers):
            if history.size >= history.capacity:
                break
            part = self.frontier[center]
            for _, index, _ in self._sample_partition_rows(
                part, min(part.size, frontier_budget // len(centers)
                          + int(ordinal < frontier_budget % len(centers))), "previous",
            ):
                history.add(part.obs[index], part.action[index], part.next_obs[index],
                            part.done[index, 0], part.raw[index], part.episode[index], part.step[index])
        saved_semantic = self._loaded_semantic_state
        semantic_slots = (
            saved_semantic["slots"] if saved_semantic and saved_semantic.get("enabled")
            else self.semantic_slots
        )
        for part, index, _ in self._sample_semantic_rows(
            history.capacity - history.size, slots=semantic_slots,
        ):
            history.add(part.obs[index], part.action[index], part.next_obs[index],
                        part.done[index, 0], part.raw[index], part.episode[index], part.step[index])
        self.four_pool_enabled = False
        self.four_pool_history_mode = True
        self.four_pool_current_fraction = float(current_fraction)
        self.stability_anchor = {}
        self.frontier = {}

    def has_training_batch(self, stage: str, batch_size: int, update_after: int) -> bool:
        if len(self) < max(update_after, batch_size):
            return False
        if stage == "s0" and self.four_pool_history_mode and self.four_pool_current_fraction == 1.:
            return self.s0_current.size >= batch_size
        return True

    def configure_four_pool(self, seed: dict[str, Any], *, source: str) -> None:
        """Replace the inherited 1M sampling mix with a fixed four-pool protocol."""
        if self.four_pool_enabled:
            raise ValueError("four-pool replay is already configured")
        tiers = ("o0_o4", "o5_o6", "o7_o9")
        if set(seed["anchor"]) != set(tiers) or len(seed["frontier"]) != 85:
            raise ValueError("four-pool seed has incomplete anchor/frontier coverage")
        if any(seed["anchor"][tier].size < 256 for tier in tiers):
            raise ValueError("each frozen anchor tier needs at least 256 rows")
        if any(part.size < 1 for part in seed["frontier"].values()):
            raise ValueError("every bad-state center needs frontier observations")
        self.stability_anchor = seed["anchor"]
        self.frontier = seed["frontier"]
        self.four_pool_source = str(source)
        self.four_pool_enabled = True
        self.s0_current = _Partition(
            self.s0_current.obs.shape[1], self.s0_current.action.shape[1],
            30000, self.raw_dim,
        )
        # The inherited success partition is neither retained nor sampled.
        self.s0_current_success = _Partition(
            self.s0_current.obs.shape[1], self.s0_current.action.shape[1],
            1, self.raw_dim,
        )

    def add_frontier_from_recent(self, episode: int, center: int, start_step: int,
                                 *, recovered: bool, alignment: float | None = None) -> None:
        """Keep the first 40 steps, hard events, and final 20 recovery steps."""
        if not self.four_pool_enabled:
            return
        indices = self.s0_current.episode_indices(episode) if recovered else [
            (self.s0_current.ptr - 1) % self.s0_current.capacity
        ]
        last_step = int(self.s0_current.step[indices[-1]])
        for index in indices:
            step = int(self.s0_current.step[index])
            raw = self.s0_current.raw[index]
            stalled = .6 <= raw[2] <= 1. and raw[2] - raw[3] <= .0005
            rebounded = raw[2] < .6 <= raw[3]
            misaligned = (.6 <= raw[2] <= 1. and alignment is not None
                          and alignment < .25)
            if (not recovered and step - start_step < 40) or (
                recovered and step - start_step >= 40 and step >= last_step - 19
            ) or (not recovered and (stalled or rebounded or misaligned)):
                part = self.frontier[center]
                part.add(self.s0_current.obs[index], self.s0_current.action[index],
                         self.s0_current.next_obs[index], self.s0_current.done[index, 0],
                         self.s0_current.raw[index], episode, step)

    def _validate_success_schedule(
        self, schedule: dict[str, float | int] | None,
    ) -> dict[str, float | int] | None:
        if schedule is None:
            return None
        required = {
            "low_start_episodes", "medium_start_episodes", "full_start_episodes",
            "low_fraction", "medium_fraction",
        }
        missing = required.difference(schedule)
        if missing:
            raise ValueError(
                "s0_success_schedule is missing: " + ", ".join(sorted(missing))
            )
        normalized: dict[str, float | int] = {
            "low_start_episodes": int(schedule["low_start_episodes"]),
            "medium_start_episodes": int(schedule["medium_start_episodes"]),
            "full_start_episodes": int(schedule["full_start_episodes"]),
            "low_fraction": float(schedule["low_fraction"]),
            "medium_fraction": float(schedule["medium_fraction"]),
        }
        low_start = int(normalized["low_start_episodes"])
        medium_start = int(normalized["medium_start_episodes"])
        full_start = int(normalized["full_start_episodes"])
        if not 1 <= low_start < medium_start < full_start:
            raise ValueError(
                "success replay episode thresholds must satisfy "
                "1 <= low_start < medium_start < full_start"
            )
        low_fraction = float(normalized["low_fraction"])
        medium_fraction = float(normalized["medium_fraction"])
        if not 0.0 <= low_fraction <= medium_fraction <= self.s0_success_fraction:
            raise ValueError(
                "success replay fractions must satisfy "
                "0 <= low <= medium <= s0_success_fraction"
            )
        return normalized

    @property
    def current_success_episode_count(self) -> int:
        return len(self.s0_current_success.episode_rows())

    @property
    def effective_s0_success_fraction(self) -> float:
        """Success quota after the episode-diversity cold-start schedule."""
        if self.s0_success_schedule is None:
            return self.s0_success_fraction
        episodes = self.current_success_episode_count
        schedule = self.s0_success_schedule
        if episodes < int(schedule["low_start_episodes"]):
            return 0.0
        if episodes < int(schedule["medium_start_episodes"]):
            return float(schedule["low_fraction"])
        if episodes < int(schedule["full_start_episodes"]):
            return float(schedule["medium_fraction"])
        return self.s0_success_fraction

    @staticmethod
    def _scale_key(scale: float) -> float:
        return float(np.round(float(scale), decimals=6))

    def _archive_current_level(self) -> None:
        if self.s0_current_scale is None or self.s0_current.size == 0:
            return
        scale = self._scale_key(self.s0_current_scale)
        if scale in self.s0_history:
            raise ValueError(f"S0 eta level {scale} was archived more than once")
        source = (
            self.s0_current_success
            if self.s0_joint_pose and self.s0_current_success.size
            else self.s0_current
        )
        self.s0_history[scale] = source.snapshot(
            self.s0_history_capacity_per_level, self.rng,
        )
        self.s0_current = _Partition(
            self.s0_current.obs.shape[1], self.s0_current.action.shape[1],
            self.s0_current.capacity, raw_dim=self.s0_current.raw.shape[1],
        )
        self.s0_current_success = _Partition(
            self.s0_current_success.obs.shape[1],
            self.s0_current_success.action.shape[1],
            self.s0_current_success.capacity,
            raw_dim=self.s0_current_success.raw.shape[1],
        )
        if self.s0_joint_pose:
            self.s0_anchor = _Partition(
                self.s0_anchor.obs.shape[1], self.s0_anchor.action.shape[1],
                self.s0_anchor.capacity, raw_dim=self.s0_anchor.raw.shape[1],
            )
        # The semantic bank spans the whole S0 task.  A precision-only level
        # change must not discard broad pose coverage collected at earlier
        # tolerances.  update_semantic_bounds() clears it only when the physical
        # target-space bounds really change.

    def _clear_semantic_bank(self) -> None:
        if not self.semantic_enabled:
            return
        total_bins = self.semantic_position_bins * self.semantic_orientation_bins
        self.semantic_slots = {}
        self.semantic_seen = np.zeros(total_bins, dtype=np.int64)

    def configure_semantic_long_term(
        self, *, enabled: bool, fraction: float,
        position_bins: int, orientation_bins: int,
        episodes_per_bin: int, transitions_per_episode: int,
        distance_min: float, distance_max: float,
        orientation_min: float, orientation_max: float,
        seed: int, reset_on_contract_change: bool = False,
    ) -> None:
        """Configure a bounded, task-semantic long-term episodic reservoir.

        Bins are defined by target end-effector distance and target orientation
        change.  Sampling is uniform over non-empty bins, then over episodes,
        so frequently visited regions cannot evict or dominate sparse regions.
        """
        if enabled and not self.s0_joint_pose:
            raise ValueError("semantic long-term replay requires joint-pose S0")
        if not 0.0 <= float(fraction) <= self.s0_current_fraction:
            raise ValueError(
                "semantic replay fraction must be in [0, s0_current_fraction]"
            )
        if min(
            int(position_bins), int(orientation_bins), int(episodes_per_bin),
            int(transitions_per_episode),
        ) < 1:
            raise ValueError("semantic replay bin and storage sizes must be positive")
        bounds = (
            float(distance_min), float(distance_max),
            float(orientation_min), float(orientation_max),
        )
        if not bounds[0] < bounds[1] or not bounds[2] < bounds[3]:
            raise ValueError("semantic replay bounds must have positive width")
        self.semantic_enabled = bool(enabled)
        if self.semantic_enabled:
            self.last_orientation_sample_counts.setdefault("semantic_long_term", 0)
        self.semantic_fraction = float(fraction) if enabled else 0.0
        self.semantic_position_bins = int(position_bins)
        self.semantic_orientation_bins = int(orientation_bins)
        self.semantic_episodes_per_bin = int(episodes_per_bin)
        self.semantic_transitions_per_episode = int(transitions_per_episode)
        self.semantic_bounds = bounds
        self.semantic_rng = np.random.default_rng(int(seed))
        loaded = self._loaded_semantic_state
        self._loaded_semantic_state = None
        if (
            self.semantic_enabled
            and loaded is not None
            and bool(loaded.get("enabled", False))
        ):
            expected = {
                "fraction": self.semantic_fraction,
                "position_bins": self.semantic_position_bins,
                "orientation_bins": self.semantic_orientation_bins,
                "episodes_per_bin": self.semantic_episodes_per_bin,
                "transitions_per_episode": self.semantic_transitions_per_episode,
                "bounds": self.semantic_bounds,
            }
            observed = {
                "fraction": float(loaded["fraction"]),
                "position_bins": int(loaded["position_bins"]),
                "orientation_bins": int(loaded["orientation_bins"]),
                "episodes_per_bin": int(loaded["episodes_per_bin"]),
                "transitions_per_episode": int(loaded["transitions_per_episode"]),
                "bounds": tuple(float(value) for value in loaded["bounds"]),
            }
            if observed != expected:
                if not reset_on_contract_change:
                    raise ValueError(
                        "checkpoint semantic replay contract does not match CLI settings"
                    )
            else:
                self.semantic_slots = loaded["slots"]
                self.semantic_seen = np.asarray(loaded["seen"], dtype=np.int64)
                self.semantic_rng.bit_generator.state = loaded["rng_state"]
                return
        self._clear_semantic_bank()
        if self.semantic_enabled:
            self._bootstrap_semantic_bank()

    def update_semantic_bounds(
        self, *, distance_min: float, distance_max: float,
        orientation_min: float, orientation_max: float,
    ) -> None:
        """Update semantic bounds, preserving the bank when space is unchanged."""
        if not self.semantic_enabled:
            return
        bounds = (
            float(distance_min), float(distance_max),
            float(orientation_min), float(orientation_max),
        )
        if not bounds[0] < bounds[1] or not bounds[2] < bounds[3]:
            raise ValueError("semantic replay bounds must have positive width")
        if self.semantic_bounds is not None and np.allclose(
            self.semantic_bounds, bounds, rtol=0.0, atol=1e-12,
        ):
            return
        self.semantic_bounds = bounds
        self._clear_semantic_bank()

    def _semantic_bin(self, distance: float, orientation: float) -> int:
        if self.semantic_bounds is None:
            raise ValueError("semantic replay bounds are not configured")
        d_min, d_max, o_min, o_max = self.semantic_bounds
        d_fraction = np.clip((float(distance) - d_min) / (d_max - d_min), 0.0, 1.0)
        o_fraction = np.clip(
            (float(orientation) - o_min) / (o_max - o_min), 0.0, 1.0,
        )
        d_bin = min(int(d_fraction * self.semantic_position_bins), self.semantic_position_bins - 1)
        o_bin = min(int(o_fraction * self.semantic_orientation_bins), self.semantic_orientation_bins - 1)
        return d_bin * self.semantic_orientation_bins + o_bin

    def _sparse_episode_snapshot(
        self, source: _Partition, indices: np.ndarray,
    ) -> _Partition:
        count = min(len(indices), self.semantic_transitions_per_episode)
        selected_positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
        selected = indices[selected_positions]
        result = _Partition(
            source.obs.shape[1], source.action.shape[1],
            self.semantic_transitions_per_episode, raw_dim=source.raw.shape[1],
        )
        for index in selected:
            result.add(
                source.obs[index], source.action[index], source.next_obs[index],
                source.done[index, 0], source.raw[index], source.episode[index],
                source.step[index],
            )
        return result

    @staticmethod
    def _episode_quality(part: _Partition, indices: np.ndarray) -> float:
        """Return the final normalized pose error used by semantic retention."""
        if len(indices) == 0:
            return float("inf")
        final = indices[int(np.argmax(part.step[indices]))]
        position_error = max(float(part.raw[final, 1]), 0.0)
        orientation_error = max(float(part.raw[final, 3]), 0.0)
        return max(position_error / 0.05, orientation_error / 0.10)

    @staticmethod
    def _snapshot_quality(snapshot: _Partition) -> float:
        return HomotopyReplayBuffer._episode_quality(
            snapshot, snapshot.chronological_indices()
        )

    def _archive_semantic_episode(
        self, episode: int, distance: float | None = None,
        orientation: float | None = None,
    ) -> None:
        if not self.semantic_enabled:
            return
        indices = self.s0_current.episode_indices(episode)
        if len(indices) == 0:
            return
        # Only retain complete episodes.  This also excludes partial episodes
        # encountered while bootstrapping an old ring buffer checkpoint.
        if not bool(self.s0_current.done[indices[-1], 0]):
            return
        if distance is None or orientation is None:
            first = indices[np.argmin(self.s0_current.step[indices])]
            if int(self.s0_current.step[first]) != 0:
                return
            distance = float(self.s0_current.raw[first, 0])
            orientation = float(self.s0_current.raw[first, 2])
        bin_index = self._semantic_bin(distance, orientation)
        snapshot = self._sparse_episode_snapshot(self.s0_current, indices)
        quality = self._episode_quality(self.s0_current, indices)
        self.semantic_seen[bin_index] += 1
        slots = self.semantic_slots.setdefault(bin_index, [])
        if len(slots) < self.semantic_episodes_per_bin:
            slots.append(snapshot)
            return
        # Keep the best trajectories for each semantic cell.  A random
        # reservoir can evict a rare, high-quality route after a curriculum
        # transition, so replacement is based on final normalized pose error.
        worst_index, worst_quality = max(
            enumerate(self._snapshot_quality(candidate) for candidate in slots),
            key=lambda item: item[1],
        )
        if quality < worst_quality:
            slots[worst_index] = snapshot

    def _bootstrap_semantic_bank(self) -> None:
        order = self.s0_current.chronological_indices()
        if len(order) == 0:
            return
        episodes = np.unique(self.s0_current.episode[order])
        for episode in episodes:
            self._archive_semantic_episode(int(episode))

    def begin_orientation_level(self, scale: float) -> None:
        """Atomically move S0 replay storage to a newly activated eta level."""
        scale = self._scale_key(scale)
        if not self.s0_joint_pose and np.isclose(scale, 0.0, rtol=0.0, atol=1e-7):
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
        curriculum_level: int | None = None,
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
            float(info.get("timeout", False)),
            float(
                info.get("curriculum_level", -1)
                if curriculum_level is None else curriculum_level
            ),
            float(info.get(
                "position_tolerance",
                self.reward_parameters.get("joint_position_tolerance", 0.01),
            )),
            float(info.get(
                "orientation_tolerance",
                self.reward_parameters.get("joint_orientation_tolerance", 0.03),
            )),
            float(info.get("keypoint_distance", 0.0)),
            float(info.get("next_keypoint_distance", 0.0)),
            float(info.get("keypoint_tracking_quality", 0.0)),
        ], np.float32)
        if scene != "none":
            self.parts[scene].add(
                observation, action, next_observation, terminal, raw,
                episode_id, step_in_episode,
            )
            return
        scale = self._scale_key(info["orientation_scale"])
        if self.s0_joint_pose and orientation_anchor:
            self.s0_anchor.add(
                observation, action, next_observation, terminal, raw,
                episode_id, step_in_episode,
            )
            return
        if (not self.s0_joint_pose) and (
            orientation_anchor or np.isclose(scale, 0.0, rtol=0.0, atol=1e-7)
        ):
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
        if self.s0_joint_pose and terminal and bool(info["task_reached"]):
            if not self.four_pool_enabled and not self.four_pool_history_mode:
                self.s0_current.copy_episode_to(episode_id, self.s0_current_success)
        if self.s0_joint_pose and terminal:
            self._archive_semantic_episode(
                episode_id,
                info.get("sampled_target_distance_m"),
                info.get("sampled_target_orientation_rad"),
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
        counts = {
            "anchor": self.s0_anchor.size,
            "current": self.s0_current.size,
            "current_scale": self.s0_current_scale,
            "historical": sum(part.size for part in self.s0_history.values()),
            "historical_levels": {
                str(scale): part.size
                for scale, part in sorted(self.s0_history.items())
            },
        }
        if self.s0_joint_pose:
            counts["current_success"] = self.s0_current_success.size
        if self.semantic_enabled:
            counts.update({
                "semantic_long_term_transitions": sum(
                    part.size
                    for slots in self.semantic_slots.values() for part in slots
                ),
                "semantic_long_term_episodes": sum(
                    len(slots) for slots in self.semantic_slots.values()
                ),
                "semantic_long_term_nonempty_bins": len(self.semantic_slots),
                "semantic_long_term_seen_episodes": int(self.semantic_seen.sum()),
            })
        return counts

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

    def _sample_episode_balanced_rows(
        self, part: _Partition, count: int, category: str,
        excluded_episodes: set[int] | None = None,
    ) -> list[tuple[_Partition, int, str]]:
        """Give each stored episode equal weight regardless of its length."""
        if count <= 0 or part.size == 0:
            return []
        indexed = part.episode_rows()
        episodes = [
            episode for episode in indexed
            if not excluded_episodes or episode not in excluded_episodes
        ]
        pools: dict[int, np.ndarray] = {}
        cursors: dict[int, int] = {}
        for episode in episodes:
            indices = np.fromiter(indexed[episode], dtype=np.int64)
            self.rng.shuffle(indices)
            pools[episode] = indices
            cursors[episode] = 0
        selected: list[tuple[_Partition, int, str]] = []
        active = episodes
        while active and len(selected) < count:
            self.rng.shuffle(active)
            remaining = []
            for episode in active:
                cursor = cursors[episode]
                selected.append((part, int(pools[episode][cursor]), category))
                cursor += 1
                cursors[episode] = cursor
                if cursor < len(pools[episode]):
                    remaining.append(episode)
                if len(selected) >= count:
                    break
            active = remaining
        return selected

    def _sample_semantic_rows(
        self, count: int, *, slots: dict[int, list[_Partition]] | None = None,
    ) -> list[tuple[_Partition, int, str]]:
        """Sample uniformly over semantic bins, then episodes and transitions."""
        if count <= 0 or (not self.semantic_enabled and slots is None):
            return []
        # Build transition permutations lazily.  The previous implementation
        # copied and shuffled every stored transition (up to 51,200) for every
        # 205-row semantic quota, even though most episode pools were never
        # selected during that call.
        pools: dict[int, list[list[Any]]] = {}
        for bin_index, bin_slots in (self.semantic_slots if slots is None else slots).items():
            episode_pools = [
                [part, None] for part in bin_slots if part.size
            ]
            if episode_pools:
                pools[int(bin_index)] = episode_pools
        selected: list[tuple[_Partition, int, str]] = []
        active_bins = list(pools)
        while active_bins and len(selected) < count:
            self.semantic_rng.shuffle(active_bins)
            remaining_bins = []
            for bin_index in active_bins:
                episode_pools = pools[bin_index]
                available = [
                    pool for pool in episode_pools
                    if pool[1] is None or pool[1]
                ]
                if not available:
                    continue
                pool = available[int(self.semantic_rng.integers(len(available)))]
                if pool[1] is None:
                    pool[1] = list(pool[0].chronological_indices())
                    self.semantic_rng.shuffle(pool[1])
                selected.append((
                    pool[0], int(pool[1].pop()), "semantic_long_term",
                ))
                if any(
                    candidate[1] is None or candidate[1]
                    for candidate in episode_pools
                ):
                    remaining_bins.append(bin_index)
                if len(selected) >= count:
                    break
            active_bins = remaining_bins
        return selected

    def _sample_joint_pose_rows(
        self, count: int, orientation_scale: float | None,
        previous_fraction: float | None = None,
        current_fraction: float | None = None,
    ) -> list[tuple[_Partition, int, str]]:
        if self.four_pool_history_mode:
            return self._sample_four_pool_history_rows(count, orientation_scale)
        if self.four_pool_enabled:
            return self._sample_four_pool_rows(count, orientation_scale)
        if self.s0_current_scale is None or orientation_scale is None:
            raise ValueError("joint-pose replay has no active level")
        if not np.isclose(
            float(orientation_scale), self.s0_current_scale, rtol=0.0, atol=1e-7,
        ):
            raise ValueError(
                f"requested current level {orientation_scale} does not match replay "
                f"current level {self.s0_current_scale}"
            )
        previous_fraction = (
            self.s0_anchor_fraction
            if previous_fraction is None else float(previous_fraction)
        )
        current_fraction = (
            self.s0_current_fraction
            if current_fraction is None else float(current_fraction)
        )
        if not 0.0 <= previous_fraction < 1.0:
            raise ValueError("previous_fraction must be in [0, 1)")
        if not 0.0 <= current_fraction <= 1.0 - previous_fraction:
            raise ValueError("current_fraction leaves an invalid success fraction")
        effective_success_fraction = self.effective_s0_success_fraction
        if not np.isclose(
            previous_fraction + current_fraction + self.s0_success_fraction,
            1.0, rtol=0.0, atol=1e-9,
        ):
            raise ValueError(
                "joint-pose replay fractions must sum to one: "
                f"success={self.s0_success_fraction}, current={current_fraction}, "
                f"previous={previous_fraction}"
            )
        desired_success = round(count * effective_success_fraction)
        desired_previous = round(count * previous_fraction)
        desired_semantic = round(count * self.semantic_fraction)
        desired_recent = count - desired_success - desired_previous - desired_semantic
        if desired_recent < 0:
            raise ValueError(
                "semantic replay quota exceeds the current/recent replay quota"
            )
        selected = self._sample_episode_balanced_rows(
            self.s0_current_success, desired_success, "current_success",
        )
        successful_episodes = (
            set(self.s0_current_success.episode_rows()) if desired_success else set()
        )
        semantic_rows = self._sample_semantic_rows(desired_semantic)
        selected.extend(semantic_rows)
        # During the initial cold start, unfilled semantic quota remains
        # ordinary recent replay; total batch size and update count stay fixed.
        desired_recent += desired_semantic - len(semantic_rows)
        selected.extend(self._sample_episode_balanced_rows(
            self.s0_current, desired_recent, "current_recent", successful_episodes,
        ))
        previous_candidates: list[tuple[_Partition, int, str]] = []
        if self.s0_history:
            _, previous = max(self.s0_history.items())
            previous_candidates.extend(self._sample_episode_balanced_rows(
                previous, previous.size, "previous",
            ))
        previous_candidates.extend(self._sample_episode_balanced_rows(
            self.s0_anchor, self.s0_anchor.size, "previous",
        ))
        self.rng.shuffle(previous_candidates)
        selected.extend(previous_candidates[:desired_previous])
        # At a new level one or more strata can be empty. Fill from current
        # data first, then protected successes and the immediately prior level.
        if len(selected) < count:
            used = {(id(part), index) for part, index, _ in selected}
            candidates: list[tuple[_Partition, int, str]] = []
            sources = [
                (self.s0_current, "current_recent"),
                (self.s0_current_success, "current_success"),
            ]
            if self.s0_history:
                _, previous = max(self.s0_history.items())
                sources.append((previous, "previous"))
            sources.append((self.s0_anchor, "previous"))
            for part, category in sources:
                for row in self._sample_episode_balanced_rows(part, part.size, category):
                    if (id(row[0]), row[1]) not in used:
                        candidates.append(row)
            self.rng.shuffle(candidates)
            selected.extend(candidates[:count - len(selected)])
        if len(selected) < count:
            raise ValueError(f"joint-pose replay has only {len(selected)} rows, need {count}")
        self.rng.shuffle(selected)
        categories = ["current_success", "current_recent", "previous"]
        if self.semantic_enabled:
            categories.append("semantic_long_term")
        counts = {
            category: sum(row[2] == category for row in selected)
            for category in categories
        }
        self.last_orientation_sample_counts = {
            "anchor": counts["current_success"],
            "current": counts["current_recent"],
            "historical": counts["previous"],
            **counts,
        }
        return selected

    def _sample_four_pool_rows(
        self, count: int, orientation_scale: float | None,
    ) -> list[tuple[_Partition, int, str]]:
        if count != 1024 or orientation_scale is None or not np.isclose(
            orientation_scale, self.s0_current_scale, rtol=0, atol=1e-7,
        ):
            raise ValueError("four-pool replay requires a 1024-row current-level S0 batch")
        selected = self._sample_episode_balanced_rows(
            self.s0_current, 384, "current_recent",
        )
        for tier, count_tier in (("o0_o4", 102), ("o5_o6", 77), ("o7_o9", 77)):
            selected.extend(self._sample_episode_balanced_rows(
                self.stability_anchor[tier], count_tier, "stability_anchor",
            ))
        centers = sorted(key for key, part in self.frontier.items() if part.size)
        if len(centers) != 85:
            raise ValueError("four-pool frontier lost a bad-state center")
        self.rng.shuffle(centers)
        # Center-uniform round robin prevents a few long failed episodes from
        # dominating the 192-row hard-state quota.
        frontier_rows = []
        for center in centers:
            frontier_rows.extend(self._sample_partition_rows(
                self.frontier[center], min(3, self.frontier[center].size), "frontier",
            ))
        self.rng.shuffle(frontier_rows)
        selected.extend(frontier_rows[:192])
        selected.extend(self._sample_semantic_rows(192))
        if len(selected) != count:
            raise ValueError(f"four-pool replay yielded {len(selected)} of {count} rows")
        self.rng.shuffle(selected)
        counts = {category: sum(row[2] == category for row in selected)
                  for category in ("current_recent", "stability_anchor", "frontier", "semantic_long_term")}
        self.last_orientation_sample_counts = {
            "anchor": counts["stability_anchor"], "current": counts["current_recent"],
            "historical": 0, "current_success": 0, "previous": 0,
            **counts,
        }
        return selected

    def _sample_four_pool_history_rows(
        self, count: int, orientation_scale: float | None,
    ) -> list[tuple[_Partition, int, str]]:
        if orientation_scale is None or not np.isclose(
            orientation_scale, self.s0_current_scale, rtol=0, atol=1e-7,
        ) or not self.s0_history:
            raise ValueError("history replay requires an active promoted S0 level")
        online_quota = round(count * self.four_pool_current_fraction)
        current_count = min(online_quota, self.s0_current.size)
        selected = self._sample_episode_balanced_rows(
            self.s0_current, current_count, "current_recent",
        )
        if self.four_pool_current_fraction == 1. and len(selected) != count:
            raise ValueError("current-only replay requires a full current-level batch")
        # The curriculum may still collect previous-level anchor episodes;
        # these are new online rollouts, not part of the frozen offline bank.
        online_target = min(online_quota, self.s0_current.size + self.s0_anchor.size)
        selected.extend(self._sample_episode_balanced_rows(
            self.s0_anchor, online_target - len(selected), "current_recent",
        ))
        histories = [part for _, part in sorted(self.s0_history.items()) if part.size]
        remaining = count - len(selected)
        allocations = [min(part.size, remaining // len(histories)
                           + int(index < remaining % len(histories)))
                       for index, part in enumerate(histories)]
        for index, part in enumerate(histories):
            allocations[index] += min(
                remaining - sum(allocations), part.size - allocations[index],
            )
        for part, quota in zip(histories, allocations):
            selected.extend(self._sample_episode_balanced_rows(part, quota, "previous"))
        if len(selected) != count:
            raise ValueError(f"history replay yielded {len(selected)} of {count} rows")
        self.rng.shuffle(selected)
        online_count = count - remaining
        self.last_orientation_sample_counts = {
            "anchor": 0, "current": online_count, "historical": remaining,
            "current_success": 0, "current_recent": online_count,
            "previous": remaining, "semantic_long_term": 0,
            "stability_anchor": 0, "frontier": 0,
        }
        return selected

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
        if self.s0_joint_pose:
            return self._sample_joint_pose_rows(
                count, orientation_scale, anchor_fraction, current_fraction,
            )
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

    @staticmethod
    def _gather_rows(
        rows: list[tuple[str, _Partition, int, bool]], field: str,
    ) -> np.ndarray:
        """Gather mixed-partition rows with one NumPy take per partition."""
        first = getattr(rows[0][1], field)
        result = np.empty((len(rows), *first.shape[1:]), dtype=first.dtype)
        groups: dict[int, tuple[_Partition, list[int], list[int]]] = {}
        for output_index, (_, part, row_index, _) in enumerate(rows):
            key = id(part)
            if key not in groups:
                groups[key] = (part, [], [])
            groups[key][1].append(output_index)
            groups[key][2].append(row_index)
        for part, output_indices, row_indices in groups.values():
            result[np.asarray(output_indices, dtype=np.int64)] = getattr(
                part, field
            )[np.asarray(row_indices, dtype=np.int64)]
        return result

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
        protected_previous = (
            max(self.s0_history.items())[1] if self.s0_history else None
        )
        rows: list[tuple[str, _Partition, int, bool]] = []
        for scene, count in zip(SCENES, counts):
            if count:
                if stage == "s0" and scene == "none":
                    selected = self._sample_s0_rows(
                        int(count), orientation_scale,
                        s0_anchor_fraction, s0_current_fraction,
                    )
                    rows.extend(
                        (
                            scene, part, index,
                            category in ("current_success", "stability_anchor")
                            or (category == "previous" and (
                                self.four_pool_history_mode or part is protected_previous
                            )),
                        )
                        for part, index, category in selected
                    )
                elif scene == "none":
                    self.last_orientation_sample_counts = {
                        "anchor": 0, "current": 0, "historical": 0,
                        "current_success": 0, "current_recent": 0,
                        "previous": 0,
                    }
                    selected = self._sample_none_rows(int(count))
                    rows.extend(
                        (scene, part, index, False)
                        for part, index, _ in selected
                    )
                else:
                    part = self.parts[scene]
                    index = self.rng.choice(part.size, size=int(count), replace=False)
                    rows.extend((scene, part, int(i), False) for i in index)
        self.rng.shuffle(rows)
        raw = self._gather_rows(rows, "raw")
        dones = self._gather_rows(rows, "done")
        scene_xi = np.fromiter(
            (1.0 if scene == "none" else xi[scene] for scene, _, _, _ in rows),
            dtype=np.float64,
            count=len(rows),
        )
        rewards, costs = _vectorized_replay_rewards(
            raw, dones, scene_xi, lambda_self, self.reward_gamma,
            self.reward_parameters,
        )
        observations = self._gather_rows(rows, "obs")
        actions = self._gather_rows(rows, "action")
        next_observations = self._gather_rows(rows, "next_obs")
        tensor = lambda value: torch.as_tensor(value, device=self.device, dtype=torch.float32)
        return Batch(
            observations=tensor(observations),
            actions=tensor(actions),
            rewards=tensor(rewards).reshape(-1, 1), costs=tensor(costs).reshape(-1, 1),
            next_observations=tensor(next_observations),
            dones=tensor(dones).reshape(-1, 1),
            protected_mask=torch.as_tensor(
                [protected for _, _, _, protected in rows],
                device=self.device, dtype=torch.bool,
            ).reshape(-1, 1),
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
            partitions = [
                ("anchor", self.s0_anchor),
                ("current", self.s0_current),
                ("current_success", self.s0_current_success),
            ]
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
        semantic_state = None
        if self.semantic_enabled:
            semantic_state = {
                "enabled": True,
                "fraction": self.semantic_fraction,
                "position_bins": self.semantic_position_bins,
                "orientation_bins": self.semantic_orientation_bins,
                "episodes_per_bin": self.semantic_episodes_per_bin,
                "transitions_per_episode": self.semantic_transitions_per_episode,
                "bounds": self.semantic_bounds,
                "slots": self.semantic_slots,
                "seen": self.semantic_seen,
                "rng_state": self.semantic_rng.bit_generator.state,
            }
        return {"storage_version": self.STORAGE_VERSION,
                "four_pool_history_mode": self.four_pool_history_mode,
                "four_pool_current_fraction": self.four_pool_current_fraction,
                "four_pool_history_source": self.four_pool_source,
                "four_pool": ({"source": self.four_pool_source,
                               "anchor": self.stability_anchor, "frontier": self.frontier}
                              if self.four_pool_enabled else None),
                "parts": self.parts,
                "s0_anchor": self.s0_anchor,
                "s0_current": self.s0_current,
                "s0_current_success": self.s0_current_success,
                "s0_current_scale": self.s0_current_scale,
                "s0_history": self.s0_history,
                "s0_history_capacity_per_level": self.s0_history_capacity_per_level,
                "rng_state": self.rng.bit_generator.state,
                "reward_gamma": self.reward_gamma, "reward_horizon": self.reward_horizon,
                "s0_anchor_fraction": self.s0_anchor_fraction,
                "s0_current_fraction": self.s0_current_fraction,
                "s0_joint_pose": self.s0_joint_pose,
                "s0_success_fraction": self.s0_success_fraction,
                "s0_success_schedule": self.s0_success_schedule,
                "reward_parameters": self.reward_parameters,
                "last_sample_counts": self.last_sample_counts,
                "last_orientation_sample_counts": self.last_orientation_sample_counts,
                "redistribution_count": self.redistribution_count,
                "semantic_long_term": semantic_state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("storage_version", 0)) != self.STORAGE_VERSION:
            raise ValueError(
                "checkpoint replay storage is incompatible with the success-replay schedule"
            )
        self.parts = state["parts"]
        four_pool = state.get("four_pool")
        self.four_pool_enabled = four_pool is not None
        self.four_pool_history_mode = bool(state.get("four_pool_history_mode", False))
        self.four_pool_current_fraction = float(state.get("four_pool_current_fraction", .5))
        self.four_pool_source = state.get("four_pool_history_source")
        if four_pool is not None:
            self.four_pool_source = four_pool["source"]
            self.stability_anchor = four_pool["anchor"]
            self.frontier = four_pool["frontier"]
        self.s0_anchor = state["s0_anchor"]
        self.s0_current = state["s0_current"]
        self.s0_current_success = state["s0_current_success"]
        self.s0_current_scale = state["s0_current_scale"]
        self.s0_history = state["s0_history"]
        self.s0_history_capacity_per_level = int(
            state["s0_history_capacity_per_level"]
        )
        self.rng.bit_generator.state = state["rng_state"]
        self.reward_gamma = float(state.get("reward_gamma", self.reward_gamma))
        self.reward_horizon = int(state.get("reward_horizon", self.reward_horizon))
        self.reward_parameters = dict(state.get("reward_parameters", self.reward_parameters))
        self.s0_joint_pose = bool(state.get("s0_joint_pose", self.s0_joint_pose))
        self.s0_success_fraction = float(
            state.get("s0_success_fraction", self.s0_success_fraction)
        )
        self.s0_success_schedule = self._validate_success_schedule(
            state.get("s0_success_schedule", self.s0_success_schedule)
        )
        # Sampling policy is a property of the current protocol, not old replay data.
        self.last_sample_counts = state.get("last_sample_counts", {s: 0 for s in SCENES})
        self.last_orientation_sample_counts = state.get(
            "last_orientation_sample_counts", {
                "anchor": 0, "current": 0, "historical": 0,
                "current_success": 0, "current_recent": 0, "previous": 0,
            },
        )
        self.redistribution_count = state.get("redistribution_count", 0)
        self._loaded_semantic_state = state.get("semantic_long_term")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
