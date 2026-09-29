from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SceneHomotopy:
    xi: float
    floor: float
    strict: bool = False
    eligible_steps: int = 0
    strict_steps: int = 0
    replay_strictified: bool = False
    outcomes: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class GoalCurriculum:
    scale: float
    start: float
    end: float
    floor: float
    levels: tuple[float, ...] = ()
    level_index: int = 0
    min_transitions_per_level: int = 5000
    current_level_steps: int = 0
    eligible_steps: int = 0
    full_scale_steps: int = 0
    outcomes: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class OrientationCurriculum:
    scale: float
    start: float
    end: float
    tolerance_start: float
    tolerance_end: float
    floor: float
    levels: tuple[float, ...] = ()
    level_index: int = 0
    min_transitions_per_level: int = 5000
    current_level_steps: int = 0
    anchor_probability: float = 0.25
    anchor_floor: float = 0.95
    eligible_steps: int = 0
    full_scale_steps: int = 0
    outcomes: deque = field(default_factory=lambda: deque(maxlen=100))
    anchor_outcomes: deque = field(default_factory=lambda: deque(maxlen=100))
    anchor_probability_intermediate: float = 0.40
    anchor_probability_recovery: float = 0.50
    retention_target: float = 0.98
    retention_min_observations: int = 20
    replay_anchor_intermediate: float = 0.40
    replay_current_intermediate: float = 0.40
    replay_anchor_recovery: float = 0.50
    replay_current_recovery: float = 0.35
    replay_anchor_normal: float = 0.25
    replay_current_normal: float = 0.50
    deterministic_probe_required: bool = False
    deterministic_previous_success_floor: float = 0.0
    deterministic_previous_collision_ceiling: float = 1.0
    deterministic_probe_passed: bool = False
    deterministic_probe_attempts: int = 0
    deterministic_probe_success_rate: float | None = None
    deterministic_probe_collision_rate: float | None = None
    deterministic_probe_joint_limit_rate: float | None = None
    deterministic_probe_previous_success_rate: float | None = None
    deterministic_probe_previous_collision_rate: float | None = None
    deterministic_probe_previous_joint_limit_rate: float | None = None
    deterministic_probe_confirmation_passed: bool = False
    deterministic_probe_level_steps: int = -1
    retention_warning: bool = False
    recovery_mode: bool = False


@dataclass
class SelfSafetyCurriculum:
    weight: float
    start: float
    end: float
    ramp_steps: int
    eligible_steps: int = 0
    full_weight_steps: int = 0


class HomotopyCurriculum:
    """Episode-level S0/S1/S2 mixture and gated safety continuation."""

    def __init__(
        self,
        stage: str,
        seed: int = 0,
        ramp_steps: int = 50000,
        *,
        goal_start_scale: float = 0.03,
        goal_end_scale: float = 1.0,
        goal_success_window: int = 100,
        goal_success_floor: float = 0.80,
        goal_full_scale_min_steps: int = 25000,
        goal_levels: tuple[float, ...] | list[float] | None = None,
        goal_min_transitions_per_level: int = 5000,
        orientation_start_scale: float = 0.0,
        orientation_end_scale: float = 1.0,
        orientation_tolerance_start: float = np.pi,
        orientation_tolerance_end: float = 0.10,
        orientation_success_window: int = 100,
        orientation_success_floor: float = 0.80,
        orientation_levels: tuple[float, ...] | list[float] | None = None,
        orientation_min_transitions_per_level: int = 5000,
        orientation_anchor_probability: float = 0.25,
        orientation_anchor_floor: float = 0.95,
        orientation_anchor_window: int = 100,
        orientation_anchor_probability_intermediate: float = 0.40,
        orientation_anchor_probability_recovery: float = 0.50,
        orientation_retention_target: float = 0.98,
        orientation_retention_min_observations: int = 20,
        orientation_replay_anchor_intermediate: float = 0.40,
        orientation_replay_current_intermediate: float = 0.40,
        orientation_replay_anchor_recovery: float = 0.50,
        orientation_replay_current_recovery: float = 0.35,
        orientation_replay_anchor_normal: float = 0.25,
        orientation_replay_current_normal: float = 0.50,
        orientation_deterministic_probe_required: bool = False,
        orientation_deterministic_previous_success_floor: float = 0.0,
        orientation_deterministic_previous_collision_ceiling: float = 1.0,
        orientation_full_scale_min_steps: int = 25000,
        joint_pose_levels: list[dict[str, float]] | None = None,
        self_start_weight: float = 0.0,
        self_end_weight: float = 1.0,
        self_ramp_steps: int = 50000,
        self_full_weight_min_steps: int = 25000,
        self_probe_collision_ceiling: float = 0.01,
    ) -> None:
        if stage not in {"s0", "s1", "s2"}:
            raise ValueError("stage must be s0, s1 or s2")
        self.stage, self.ramp_steps = stage, int(ramp_steps)
        self.rng = np.random.default_rng(seed)
        if not 0.0 <= self_start_weight <= self_end_weight <= 1.0:
            raise ValueError("self safety weights must satisfy 0 <= start <= end <= 1")
        if self_ramp_steps < 1:
            raise ValueError("self_ramp_steps must be positive")
        if self_full_weight_min_steps < 1:
            raise ValueError("self_full_weight_min_steps must be positive")
        if not 0.0 <= self_probe_collision_ceiling <= 1.0:
            raise ValueError("self_probe_collision_ceiling must be in [0, 1]")
        self.self_full_weight_min_steps = int(self_full_weight_min_steps)
        self.self_probe_collision_ceiling = float(self_probe_collision_ceiling)
        self.self_safety = SelfSafetyCurriculum(
            weight=float(self_start_weight if stage == "s0" else self_end_weight),
            start=float(self_start_weight),
            end=float(self_end_weight),
            ramp_steps=int(self_ramp_steps),
        )
        if not 0.0 < goal_start_scale <= goal_end_scale <= 1.0:
            raise ValueError("goal scales must satisfy 0 < start <= end <= 1")
        if not 0.0 <= goal_success_floor <= 1.0:
            raise ValueError("goal_success_floor must be in [0, 1]")
        if goal_success_window < 1:
            raise ValueError("goal_success_window must be positive")
        if goal_full_scale_min_steps < 1:
            raise ValueError("goal_full_scale_min_steps must be positive")
        levels = tuple(float(value) for value in (
            goal_levels if goal_levels is not None else (goal_start_scale, goal_end_scale)
        ))
        invalid_goal_order = any(
            (right < left if joint_pose_levels is not None else right <= left)
            for left, right in zip(levels, levels[1:])
        )
        if (len(levels) < 2 or not np.isclose(levels[0], goal_start_scale)
                or not np.isclose(levels[-1], goal_end_scale)
                or invalid_goal_order):
            qualifier = "non-decreasing" if joint_pose_levels is not None else "strictly increasing"
            raise ValueError(
                f"goal_levels must be {qualifier} from start_scale to end_scale"
            )
        if goal_min_transitions_per_level < 1:
            raise ValueError("goal_min_transitions_per_level must be positive")
        if not 0.0 <= orientation_start_scale <= orientation_end_scale <= 1.0:
            raise ValueError("orientation scales must satisfy 0 <= start <= end <= 1")
        if not 0.0 < orientation_tolerance_end <= orientation_tolerance_start <= np.pi:
            raise ValueError("orientation tolerances must satisfy 0 < end <= start <= pi")
        if orientation_success_window < 1:
            raise ValueError("orientation_success_window must be positive")
        if not 0.0 <= orientation_success_floor <= 1.0:
            raise ValueError("orientation_success_floor must be in [0, 1]")
        orientation_levels_tuple = tuple(float(value) for value in (
            orientation_levels
            if orientation_levels is not None
            else (orientation_start_scale, orientation_end_scale)
        ))
        if (len(orientation_levels_tuple) < 2
                or not np.isclose(orientation_levels_tuple[0], orientation_start_scale)
                or not np.isclose(orientation_levels_tuple[-1], orientation_end_scale)
                or any(right <= left for left, right in zip(
                    orientation_levels_tuple, orientation_levels_tuple[1:]
                ))):
            raise ValueError(
                "orientation_levels must be strictly increasing from start_scale to end_scale"
            )
        if orientation_min_transitions_per_level < 1:
            raise ValueError("orientation_min_transitions_per_level must be positive")
        if orientation_full_scale_min_steps < 1:
            raise ValueError("orientation_full_scale_min_steps must be positive")
        self.orientation_full_scale_min_steps = int(orientation_full_scale_min_steps)
        if not 0.0 <= orientation_anchor_probability < 1.0:
            raise ValueError("orientation_anchor_probability must be in [0, 1)")
        if not 0.0 <= orientation_anchor_floor <= 1.0:
            raise ValueError("orientation_anchor_floor must be in [0, 1]")
        if orientation_anchor_window < 1:
            raise ValueError("orientation_anchor_window must be positive")
        if not orientation_anchor_probability <= orientation_anchor_probability_intermediate <= 1.0:
            raise ValueError("intermediate anchor probability must be >= the base probability")
        if not orientation_anchor_probability_intermediate <= orientation_anchor_probability_recovery <= 1.0:
            raise ValueError("recovery anchor probability must be >= the intermediate probability")
        if not 0.0 <= orientation_retention_target <= 1.0:
            raise ValueError("retention target must be in [0, 1]")
        orientation_retention_target = max(
            float(orientation_retention_target), float(orientation_anchor_floor)
        )
        if orientation_retention_min_observations < 1:
            raise ValueError("retention_min_observations must be positive")
        for anchor_fraction, current_fraction in (
            (orientation_replay_anchor_normal, orientation_replay_current_normal),
            (orientation_replay_anchor_intermediate, orientation_replay_current_intermediate),
            (orientation_replay_anchor_recovery, orientation_replay_current_recovery),
        ):
            if not 0.0 <= anchor_fraction < 1.0 or not 0.0 <= current_fraction <= 1.0 - anchor_fraction:
                raise ValueError("invalid adaptive replay fractions")
        self.goal_full_scale_min_steps = int(goal_full_scale_min_steps)
        initial_goal_index = 0 if stage == "s0" else len(levels) - 1
        initial_goal_scale = levels[initial_goal_index]
        self.goal = GoalCurriculum(
            scale=float(initial_goal_scale),
            start=float(goal_start_scale),
            end=float(goal_end_scale),
            floor=float(goal_success_floor),
            levels=levels,
            level_index=initial_goal_index,
            min_transitions_per_level=int(goal_min_transitions_per_level),
            outcomes=deque(maxlen=int(goal_success_window)),
        )
        initial_orientation_index = 0 if stage == "s0" else len(orientation_levels_tuple) - 1
        initial_orientation_scale = orientation_levels_tuple[initial_orientation_index]
        self.orientation = OrientationCurriculum(
            scale=float(initial_orientation_scale),
            start=float(orientation_start_scale),
            end=float(orientation_end_scale),
            tolerance_start=float(orientation_tolerance_start),
            tolerance_end=float(orientation_tolerance_end),
            floor=float(orientation_success_floor),
            levels=orientation_levels_tuple,
            level_index=initial_orientation_index,
            min_transitions_per_level=int(orientation_min_transitions_per_level),
            anchor_probability=float(orientation_anchor_probability),
            anchor_floor=float(orientation_anchor_floor),
            outcomes=deque(maxlen=int(orientation_success_window)),
            anchor_outcomes=deque(maxlen=int(orientation_anchor_window)),
            anchor_probability_intermediate=float(orientation_anchor_probability_intermediate),
            anchor_probability_recovery=float(orientation_anchor_probability_recovery),
            retention_target=float(orientation_retention_target),
            retention_min_observations=int(orientation_retention_min_observations),
            replay_anchor_intermediate=float(orientation_replay_anchor_intermediate),
            replay_current_intermediate=float(orientation_replay_current_intermediate),
            replay_anchor_recovery=float(orientation_replay_anchor_recovery),
            replay_current_recovery=float(orientation_replay_current_recovery),
            replay_anchor_normal=float(orientation_replay_anchor_normal),
            replay_current_normal=float(orientation_replay_current_normal),
            deterministic_probe_required=bool(orientation_deterministic_probe_required),
            deterministic_previous_success_floor=float(
                orientation_deterministic_previous_success_floor
            ),
            deterministic_previous_collision_ceiling=float(
                orientation_deterministic_previous_collision_ceiling
            ),
        )
        self.joint_pose_levels: tuple[dict[str, float], ...] = ()
        self.precision_only_task_space = False
        if joint_pose_levels is not None:
            parsed = tuple(
                {
                    "goal_scale": float(level["goal_scale"]),
                    "position_tolerance": float(level["position_tolerance_m"]),
                    "orientation_tolerance": float(level["orientation_tolerance_rad"]),
                    **({
                        "target_distance_min_m": float(level["target_distance_min_m"]),
                        "target_distance_max_m": float(level["target_distance_max_m"]),
                        "target_orientation_min_rad": float(level["target_orientation_min_rad"]),
                        "target_orientation_max_rad": float(level["target_orientation_max_rad"]),
                    } if "target_distance_max_m" in level else {}),
                }
                for level in joint_pose_levels
            )
            if len(parsed) != len(self.orientation.levels):
                raise ValueError("joint_pose_levels must match orientation_levels length")
            goal_scales = [level["goal_scale"] for level in parsed]
            position_tolerances = [level["position_tolerance"] for level in parsed]
            orientation_tolerances = [level["orientation_tolerance"] for level in parsed]
            if any(not 0.0 < value <= 1.0 for value in goal_scales):
                raise ValueError("joint pose goal scales must be in (0, 1]")
            if any(right < left for left, right in zip(goal_scales, goal_scales[1:])):
                raise ValueError("joint pose goal scales must be non-decreasing")
            if any(not 0.0 < value <= 0.20 for value in position_tolerances):
                raise ValueError("joint pose position tolerances must be in (0, 0.20]")
            if any(right > left for left, right in zip(
                position_tolerances, position_tolerances[1:]
            )):
                raise ValueError("joint pose position tolerances must be non-increasing")
            if any(not 0.0 < value <= np.pi for value in orientation_tolerances):
                raise ValueError("joint pose orientation tolerances must be in (0, pi]")
            if any(right > left for left, right in zip(
                orientation_tolerances, orientation_tolerances[1:]
            )):
                raise ValueError("joint pose orientation tolerances must be non-increasing")
            task_space = ["target_distance_max_m" in level for level in parsed]
            if any(task_space) and not all(task_space):
                raise ValueError("task-space curriculum bounds must be present at every level")
            if all(task_space):
                distance_maxima = [level["target_distance_max_m"] for level in parsed]
                orientation_maxima = [level["target_orientation_max_rad"] for level in parsed]
                if any(not 0.0 < level["target_distance_min_m"] < level["target_distance_max_m"] for level in parsed):
                    raise ValueError("invalid task-space distance range")
                if any(not 0.0 <= level["target_orientation_min_rad"] < level["target_orientation_max_rad"] <= np.pi for level in parsed):
                    raise ValueError("invalid task-space orientation range")
                task_bounds = [(
                    level["target_distance_min_m"],
                    level["target_distance_max_m"],
                    level["target_orientation_min_rad"],
                    level["target_orientation_max_rad"],
                ) for level in parsed]
                self.precision_only_task_space = all(
                    np.allclose(bounds, task_bounds[0], rtol=0.0, atol=1e-12)
                    for bounds in task_bounds[1:]
                )
                if not self.precision_only_task_space:
                    if any(right <= left for left, right in zip(distance_maxima, distance_maxima[1:])):
                        raise ValueError("task-space distance maxima must strictly increase")
                    if any(right <= left for left, right in zip(orientation_maxima, orientation_maxima[1:])):
                        raise ValueError("task-space orientation maxima must strictly increase")
            self.joint_pose_levels = parsed
            index = self.orientation.level_index
            self.goal.levels = tuple(goal_scales)
            self.goal.level_index = index
            self.goal.start = goal_scales[0]
            self.goal.end = goal_scales[-1]
            self.goal.scale = goal_scales[index]
        self.probabilities = {"s0": [1, 0, 0], "s1": [.25, .75, 0], "s2": [.20, .30, .50]}[stage]
        self.states = {
            "static": SceneHomotopy(1.0 if stage == "s2" else .02, .90, strict=stage == "s2", replay_strictified=stage == "s2"),
            "dynamic": SceneHomotopy(.02, .80),
        }

    def choose_scene(self) -> str:
        return str(self.rng.choice(["none", "static", "dynamic"], p=self.probabilities))

    def choose_orientation_anchor(self) -> bool:
        """Rehearse the previous joint-pose level while learning a harder one."""
        if self.joint_pose_levels:
            if "target_distance_max_m" in self.joint_pose_levels[0]:
                return False
            return bool(
                self.stage == "s0"
                and self.orientation.level_index > 0
                and self.rng.random() < self.orientation_anchor_probability
            )
        return bool(
            self.stage == "s0"
            and self.position_phase_complete
            and self.orientation.scale > self.orientation.start
            and self.rng.random() < self.orientation_anchor_probability
        )

    @property
    def orientation_retention_mode(self) -> str:
        outcomes = self.orientation.anchor_outcomes
        rate = self.orientation_anchor_success_rate
        enough = len(outcomes) >= self.orientation.retention_min_observations
        if self.orientation.recovery_mode or (
            enough and rate is not None and rate < self.orientation.anchor_floor
        ):
            return "recovery"
        if self.orientation.retention_warning or (
            enough and rate is not None and rate < self.orientation.retention_target
        ):
            return "intermediate"
        return "normal"

    @property
    def orientation_anchor_probability(self) -> float:
        mode = self.orientation_retention_mode
        if mode == "recovery":
            return self.orientation.anchor_probability_recovery
        if mode == "intermediate":
            return self.orientation.anchor_probability_intermediate
        return self.orientation.anchor_probability

    @property
    def orientation_replay_mix(self) -> tuple[float, float, float]:
        mode = self.orientation_retention_mode
        if mode == "recovery":
            anchor = self.orientation.replay_anchor_recovery
            current = self.orientation.replay_current_recovery
        elif mode == "intermediate":
            anchor = self.orientation.replay_anchor_intermediate
            current = self.orientation.replay_current_intermediate
        else:
            anchor = self.orientation.replay_anchor_normal
            current = self.orientation.replay_current_normal
        return float(anchor), float(current), float(1.0 - anchor - current)

    def configure_orientation_retention(self, config: dict[str, Any]) -> None:
        """Apply the current protocol policy after loading an older checkpoint."""
        values = {
            "anchor_probability": float(config["anchor_probability"]),
            "anchor_probability_intermediate": float(config["anchor_probability_intermediate"]),
            "anchor_probability_recovery": float(config["anchor_probability_recovery"]),
            "retention_target": float(config["retention_target"]),
            "retention_min_observations": int(config["retention_min_observations"]),
            "replay_anchor_intermediate": float(config["replay_anchor_intermediate"]),
            "replay_current_intermediate": float(config["replay_current_intermediate"]),
            "replay_anchor_recovery": float(config["replay_anchor_recovery"]),
            "replay_current_recovery": float(config["replay_current_recovery"]),
            "replay_anchor_normal": float(config["replay_anchor_fraction"]),
            "replay_current_normal": float(config["replay_current_fraction"]),
            "deterministic_probe_required": bool(config["deterministic_probe_required"]),
            "deterministic_previous_success_floor": float(
                config["deterministic_previous_success_floor"]
            ),
            "deterministic_previous_collision_ceiling": float(
                config["deterministic_previous_collision_ceiling"]
            ),
        }
        for name, value in values.items():
            setattr(self.orientation, name, value)
        for name, default in (
            ("deterministic_probe_passed", False),
            ("deterministic_probe_attempts", 0),
            ("deterministic_probe_success_rate", None),
            ("deterministic_probe_collision_rate", None),
            ("deterministic_probe_joint_limit_rate", None),
            ("deterministic_probe_previous_success_rate", None),
            ("deterministic_probe_previous_collision_rate", None),
            ("deterministic_probe_previous_joint_limit_rate", None),
            ("deterministic_probe_confirmation_passed", False),
            ("deterministic_probe_level_steps", -1),
            ("retention_warning", False),
            ("recovery_mode", False),
        ):
            if not hasattr(self.orientation, name):
                setattr(self.orientation, name, default)

    def _orientation_online_ready(self) -> bool:
        if self.joint_pose_levels and "target_distance_max_m" in self.joint_pose_levels[0]:
            return bool(
                self.orientation.current_level_steps
                >= self.orientation.min_transitions_per_level
            )
        rate = self.orientation_success_rate
        anchor_rate = self.orientation_anchor_success_rate
        anchor_ready = bool(
            self.joint_pose_levels and self.orientation.level_index == 0
        ) or bool(
            len(self.orientation.anchor_outcomes) == self.orientation.anchor_outcomes.maxlen
            and anchor_rate is not None and anchor_rate >= self.orientation.anchor_floor
        )
        return bool(
            len(self.orientation.outcomes) == self.orientation.outcomes.maxlen
            and rate is not None and rate >= self.orientation.floor
            and anchor_ready
            and self.orientation.current_level_steps >= self.orientation.min_transitions_per_level
        )

    def orientation_probe_due(
        self, interval_transitions: int, minimum_full_pose_steps: int = 0,
    ) -> bool:
        if not self.orientation.deterministic_probe_required:
            return False
        if (self.orientation_at_full_scale
                and self.orientation.full_scale_steps < int(minimum_full_pose_steps)):
            return False
        # The first probe cannot advance a level until the online and anchor
        # windows pass, so running it earlier only burns evaluation episodes.
        task_space_curriculum = bool(
            self.joint_pose_levels
            and "target_distance_max_m" in self.joint_pose_levels[0]
        )
        if (not task_space_curriculum
                and self.orientation.deterministic_probe_attempts == 0
                and not self._orientation_online_ready()):
            return False
        last = self.orientation.deterministic_probe_level_steps
        minimum = max(int(interval_transitions), self.orientation.min_transitions_per_level)
        return bool(
            self.orientation.current_level_steps >= minimum
            and (last < 0 or self.orientation.current_level_steps - last >= int(interval_transitions))
        )

    def record_orientation_probe(
        self, *, success_rate: float, collision_rate: float, joint_limit_rate: float,
        passed: bool, previous_success_rate: float | None = None,
        previous_collision_rate: float | None = None,
        previous_joint_limit_rate: float | None = None,
        confirmation_passed: bool = False,
        retention_passed: bool | None = None,
        recovery_required: bool | None = None,
    ) -> None:
        self.orientation.deterministic_probe_attempts += 1
        self.orientation.deterministic_probe_success_rate = float(success_rate)
        self.orientation.deterministic_probe_collision_rate = float(collision_rate)
        self.orientation.deterministic_probe_joint_limit_rate = float(joint_limit_rate)
        self.orientation.deterministic_probe_previous_success_rate = (
            None if previous_success_rate is None else float(previous_success_rate)
        )
        self.orientation.deterministic_probe_previous_collision_rate = (
            None if previous_collision_rate is None else float(previous_collision_rate)
        )
        self.orientation.deterministic_probe_previous_joint_limit_rate = (
            None if previous_joint_limit_rate is None else float(previous_joint_limit_rate)
        )
        self.orientation.deterministic_probe_confirmation_passed = bool(
            confirmation_passed
        )
        self.orientation.deterministic_probe_level_steps = self.orientation.current_level_steps
        self.orientation.deterministic_probe_passed = bool(passed)
        # Promotion and recovery have different thresholds. A small miss of
        # the promotion contract freezes promotion without forcing half of
        # collection and replay back to the previous level.
        self.orientation.recovery_mode = bool(
            retention_passed is False
            if recovery_required is None
            else recovery_required
        )
        self.orientation.retention_warning = bool(
            retention_passed is False and not self.orientation.recovery_mode
        )

    def advance_orientation_if_ready(self) -> bool:
        if self.orientation_at_full_scale or not self._orientation_online_ready():
            return False
        if (self.orientation.deterministic_probe_required
                and not self.orientation.deterministic_probe_passed):
            return False
        self.orientation.eligible_steps += self.orientation.current_level_steps
        self.orientation.level_index += 1
        self.orientation.scale = self.orientation.levels[self.orientation.level_index]
        if self.joint_pose_levels:
            self.goal.level_index = self.orientation.level_index
            self.goal.scale = self.joint_pose_levels[self.orientation.level_index]["goal_scale"]
            self.goal.current_level_steps = 0
            self.goal.outcomes.clear()
        self.orientation.current_level_steps = 0
        self.orientation.outcomes.clear()
        self.orientation.anchor_outcomes.clear()
        self.orientation.deterministic_probe_passed = False
        self.orientation.deterministic_probe_success_rate = None
        self.orientation.deterministic_probe_collision_rate = None
        self.orientation.deterministic_probe_joint_limit_rate = None
        self.orientation.deterministic_probe_previous_success_rate = None
        self.orientation.deterministic_probe_previous_collision_rate = None
        self.orientation.deterministic_probe_previous_joint_limit_rate = None
        self.orientation.deterministic_probe_confirmation_passed = False
        self.orientation.deterministic_probe_level_steps = -1
        self.orientation.retention_warning = False
        self.orientation.recovery_mode = False
        return True

    def manually_advance_orientation_to(
        self, target_level: int, *, allow_early: bool = False,
    ) -> None:
        """Activate a harder S0 pose level after an explicit human decision."""
        target_level = int(target_level)
        current_level = int(self.orientation.level_index)
        if self.stage != "s0":
            raise ValueError("manual pose promotion is only valid in S0")
        if target_level <= current_level:
            raise ValueError(
                f"target level L{target_level} must be above current L{current_level}"
            )
        if target_level >= len(self.orientation.levels):
            raise ValueError(
                f"target level L{target_level} is outside L0--"
                f"L{len(self.orientation.levels) - 1}"
            )
        if (
            not allow_early
            and self.orientation.current_level_steps
            < self.orientation.min_transitions_per_level
        ):
            raise ValueError(
                "manual promotion requires the full per-level transition budget: "
                f"{self.orientation.current_level_steps}/"
                f"{self.orientation.min_transitions_per_level}"
            )
        self.orientation.eligible_steps += self.orientation.current_level_steps
        self.orientation.level_index = target_level
        self.orientation.scale = self.orientation.levels[target_level]
        if self.joint_pose_levels:
            self.goal.level_index = target_level
            self.goal.scale = self.joint_pose_levels[target_level]["goal_scale"]
            self.goal.current_level_steps = 0
            self.goal.outcomes.clear()
        self.orientation.current_level_steps = 0
        self.orientation.outcomes.clear()
        self.orientation.anchor_outcomes.clear()
        self.orientation.deterministic_probe_passed = False
        self.orientation.deterministic_probe_success_rate = None
        self.orientation.deterministic_probe_collision_rate = None
        self.orientation.deterministic_probe_joint_limit_rate = None
        self.orientation.deterministic_probe_previous_success_rate = None
        self.orientation.deterministic_probe_previous_collision_rate = None
        self.orientation.deterministic_probe_previous_joint_limit_rate = None
        self.orientation.deterministic_probe_confirmation_passed = False
        self.orientation.deterministic_probe_level_steps = -1
        self.orientation.retention_warning = False
        self.orientation.recovery_mode = False

    def contract(self, scene: str) -> tuple[float, bool]:
        return (1.0, True) if scene == "none" else (self.states[scene].xi, self.states[scene].strict)

    @property
    def goal_scale(self) -> float:
        return self.goal.scale

    @property
    def goal_at_full_scale(self) -> bool:
        return bool(np.isclose(self.goal.scale, self.goal.end, rtol=0.0, atol=1e-12))

    @property
    def orientation_scale(self) -> float:
        return self.orientation.scale

    @property
    def lambda_self(self) -> float:
        return self.self_safety.weight

    @property
    def orientation_tolerance(self) -> float:
        if self.joint_pose_levels:
            return self.joint_pose_levels[self.orientation.level_index][
                "orientation_tolerance"
            ]
        fraction = (self.orientation.scale - self.orientation.start) / max(
            self.orientation.end - self.orientation.start, 1e-12
        )
        return float(
            self.orientation.tolerance_start
            + np.clip(fraction, 0.0, 1.0)
            * (self.orientation.tolerance_end - self.orientation.tolerance_start)
        )

    @property
    def position_tolerance(self) -> float:
        if self.joint_pose_levels:
            return self.joint_pose_levels[self.orientation.level_index][
                "position_tolerance"
            ]
        return 0.055

    def pose_contract(self, *, previous: bool = False) -> dict[str, float]:
        if not self.joint_pose_levels:
            return {
                "goal_scale": self.goal_scale,
                "orientation_scale": self.orientation_scale,
                "position_tolerance": self.position_tolerance,
                "orientation_tolerance": self.orientation_tolerance,
            }
        index = self.orientation.level_index
        if previous:
            index = max(0, index - 1)
        level = self.joint_pose_levels[index]
        contract = {
            "goal_scale": level["goal_scale"],
            # Keep the level coordinate here because replay partitions use it
            # as their key.  Explicit task-space bounds, rather than this
            # bookkeeping value, control target-orientation sampling.
            "orientation_scale": self.orientation.levels[index],
            "position_tolerance": level["position_tolerance"],
            "orientation_tolerance": level["orientation_tolerance"],
        }
        if "target_distance_max_m" in level:
            contract.update({
                "target_distance_min_m": level["target_distance_min_m"],
                "target_distance_max_m": level["target_distance_max_m"],
                "target_orientation_min_rad": level["target_orientation_min_rad"],
                "target_orientation_max_rad": level["target_orientation_max_rad"],
            })
        return contract
    @property
    def position_phase_complete(self) -> bool:
        if self.joint_pose_levels:
            return self.orientation_at_full_scale
        return bool(
            self.goal_at_full_scale
            and self.goal.full_scale_steps >= self.goal_full_scale_min_steps
        )

    @property
    def pose_phase_complete(self) -> bool:
        """Whether the complete pose task is ready for dense safety shaping."""
        if self.precision_only_task_space:
            retention_ready = bool(
                self.orientation.level_index == 0
                or (
                    self.orientation.deterministic_probe_previous_success_rate
                    is not None
                    and self.orientation.deterministic_probe_previous_success_rate
                    >= self.orientation.deterministic_previous_success_floor
                    and (
                        self.orientation.deterministic_probe_previous_collision_rate
                        is None
                        or self.orientation.deterministic_probe_previous_collision_rate
                        <= self.orientation.deterministic_previous_collision_ceiling
                    )
                    and (
                        self.orientation.deterministic_probe_previous_joint_limit_rate
                        is None
                        or self.orientation.deterministic_probe_previous_joint_limit_rate
                        == 0.0
                    )
                )
            )
        else:
            retention_ready = bool(
                (self.joint_pose_levels and self.orientation.level_index == 0)
                or (
                    len(self.orientation.anchor_outcomes)
                    == self.orientation.anchor_outcomes.maxlen
                    and self.orientation_anchor_success_rate is not None
                    and self.orientation_anchor_success_rate
                    >= self.orientation.deterministic_previous_success_floor
                )
            )
        return bool(
            self.position_phase_complete
            and self.orientation_at_full_scale
            and self.orientation.full_scale_steps >= self.orientation_full_scale_min_steps
            and len(self.orientation.outcomes) == self.orientation.outcomes.maxlen
            and self.orientation_success_rate is not None
            and self.orientation_success_rate >= self.orientation.floor
            and retention_ready
            and (
                not self.orientation.deterministic_probe_required
                or self.orientation.deterministic_probe_passed
            )
        )

    @property
    def self_safety_probe_passed(self) -> bool:
        return bool(
            not self.orientation.deterministic_probe_required
            or (
                self.orientation.deterministic_probe_passed
                and self.orientation.deterministic_probe_collision_rate is not None
                and self.orientation.deterministic_probe_collision_rate
                <= self.self_probe_collision_ceiling
            )
        )

    @property
    def self_safety_phase_complete(self) -> bool:
        return bool(
            np.isclose(
                self.self_safety.weight,
                self.self_safety.end,
                rtol=0.0,
                atol=1e-12,
            )
            and self.self_safety.full_weight_steps >= self.self_full_weight_min_steps
            and self.self_safety_probe_passed
        )

    @property
    def s0_phase(self) -> str:
        if self.joint_pose_levels and not self.pose_phase_complete:
            return "joint_pose"
        if not self.position_phase_complete:
            return "position"
        # Once dense self-safety shaping has started, a retention drop freezes
        # its counters but must not make the reported state regress to pose.
        self_safety_started = bool(
            self.self_safety.eligible_steps > 0
            or self.self_safety.weight > self.self_safety.start
        )
        if not self.pose_phase_complete and not self_safety_started:
            return "pose"
        if not np.isclose(
            self.self_safety.weight, self.self_safety.end, rtol=0.0, atol=1e-12
        ):
            return "self_safety_ramp"
        if not self.self_safety_phase_complete:
            return "self_safety_consolidation"
        return "complete"

    @property
    def orientation_at_full_scale(self) -> bool:
        return bool(np.isclose(self.orientation.scale, self.orientation.end, rtol=0.0, atol=1e-12))

    def record_transition(
        self, goal_scale: float, orientation_scale: float, *, orientation_anchor: bool = False,
        curriculum_eligible: bool = True,
    ) -> None:
        """Count position and full-pose consolidation from their actual contracts."""
        if self.stage != "s0" or not curriculum_eligible:
            return
        if self.joint_pose_levels:
            if np.isclose(float(goal_scale), self.goal.scale, rtol=0.0, atol=1e-12):
                self.goal.current_level_steps += 1
            if self.orientation_at_full_scale and not orientation_anchor:
                self.goal.full_scale_steps += 1
                self.orientation.full_scale_steps += 1
            return
        if np.isclose(float(goal_scale), self.goal.scale, rtol=0.0, atol=1e-12):
            self.goal.current_level_steps += 1
        if np.isclose(
            float(goal_scale), self.goal.end, rtol=0.0, atol=1e-12
        ):
            self.goal.full_scale_steps += 1
            if np.isclose(
                float(orientation_scale), self.orientation.end, rtol=0.0, atol=1e-12
            ) and not orientation_anchor:
                self.orientation.full_scale_steps += 1

    @property
    def orientation_anchor_success_rate(self) -> float | None:
        if not self.orientation.anchor_outcomes:
            return None
        return float(np.mean(self.orientation.anchor_outcomes))

    @property
    def orientation_success_rate(self) -> float | None:
        if not self.orientation.outcomes:
            return None
        return float(np.mean(self.orientation.outcomes))

    def s0_goal_gate_eligible(
        self, minimum_goal_full_scale_steps: int, minimum_pose_full_scale_steps: int
    ) -> bool:
        if self.precision_only_task_space:
            retention_ready = bool(
                self.orientation.level_index == 0
                or (
                    self.orientation.deterministic_probe_previous_success_rate
                    is not None
                    and self.orientation.deterministic_probe_previous_success_rate
                    >= self.orientation.anchor_floor
                    and (
                        self.orientation.deterministic_probe_previous_collision_rate
                        is None
                        or self.orientation.deterministic_probe_previous_collision_rate
                        <= self.orientation.deterministic_previous_collision_ceiling
                    )
                    and (
                        self.orientation.deterministic_probe_previous_joint_limit_rate
                        is None
                        or self.orientation.deterministic_probe_previous_joint_limit_rate
                        == 0.0
                    )
                )
            )
        else:
            retention_ready = bool(
                len(self.orientation.anchor_outcomes)
                == self.orientation.anchor_outcomes.maxlen
                and self.orientation_anchor_success_rate is not None
                and self.orientation_anchor_success_rate
                >= self.orientation.anchor_floor
            )
        return bool(
            self.stage == "s0"
            and self.goal_at_full_scale
            and self.goal.full_scale_steps >= int(minimum_goal_full_scale_steps)
            and self.orientation_at_full_scale
            and self.orientation.full_scale_steps >= int(minimum_pose_full_scale_steps)
            and len(self.orientation.outcomes) == self.orientation.outcomes.maxlen
            and self.orientation_success_rate is not None
            and self.orientation_success_rate >= self.orientation.floor
            and retention_ready
            and np.isclose(
                self.self_safety.weight,
                self.self_safety.end,
                rtol=0.0,
                atol=1e-12,
            )
            and self.self_safety.full_weight_steps >= self.self_full_weight_min_steps
            and self.self_safety_phase_complete
            and (
                not self.orientation.deterministic_probe_required
                or self.orientation.deterministic_probe_passed
            )
        )

    def scene_stage_complete(self, scene: str, minimum_strict_steps: int) -> bool:
        """Return whether a formal obstacle stage may hand off its checkpoint."""
        if scene not in {"static", "dynamic"}:
            raise ValueError("scene must be 'static' or 'dynamic'")
        state = self.states[scene]
        return bool(
            state.strict
            and state.replay_strictified
            and int(state.strict_steps) >= int(minimum_strict_steps)
        )

    def stage_complete(
        self, minimum_goal_full_scale_steps: int,
        minimum_pose_full_scale_steps: int, minimum_strict_steps: int,
    ) -> bool:
        """Formal completion contract for the active serial stage."""
        if self.stage == "s0":
            return self.s0_goal_gate_eligible(
                minimum_goal_full_scale_steps, minimum_pose_full_scale_steps,
            )
        scene = "static" if self.stage == "s1" else "dynamic"
        return self.scene_stage_complete(scene, minimum_strict_steps)

    def finish_episode(
        self, scene: str, position_reached: bool, task_reached: bool, transitions: int,
        *, orientation_anchor: bool = False, curriculum_eligible: bool = True,
        pose_phase_complete_at_start: bool | None = None,
        self_weight_full_at_start: bool | None = None,
    ) -> dict[str, Any]:
        goal_rate = None
        orientation_rate = None
        if self.stage == "s0":
            if self.joint_pose_levels:
                current_episode_full_steps = (
                    0 if orientation_anchor or not self.orientation_at_full_scale
                    else int(transitions)
                )
                if pose_phase_complete_at_start is None:
                    pose_phase_complete_at_episode_start = bool(
                        self.pose_phase_complete
                        and self.orientation.full_scale_steps - current_episode_full_steps
                        >= self.orientation_full_scale_min_steps
                    )
                else:
                    pose_phase_complete_at_episode_start = bool(
                        pose_phase_complete_at_start
                    )
                if self_weight_full_at_start is None:
                    self_weight_full_at_episode_start = bool(np.isclose(
                        self.self_safety.weight, self.self_safety.end,
                        rtol=0.0, atol=1e-12,
                    ))
                else:
                    self_weight_full_at_episode_start = bool(self_weight_full_at_start)
                if curriculum_eligible:
                    self.goal.outcomes.append(bool(task_reached))
                    goal_rate = float(np.mean(self.goal.outcomes))
                    if orientation_anchor:
                        self.orientation.anchor_outcomes.append(bool(task_reached))
                    else:
                        self.orientation.current_level_steps += int(transitions)
                        self.orientation.outcomes.append(bool(task_reached))
                        orientation_rate = float(np.mean(self.orientation.outcomes))
                if (curriculum_eligible and not orientation_anchor
                        and pose_phase_complete_at_episode_start):
                    self.self_safety.eligible_steps += int(transitions)
                    fraction = min(
                        self.self_safety.eligible_steps / self.self_safety.ramp_steps, 1.0
                    )
                    self.self_safety.weight = float(
                        self.self_safety.start
                        + fraction * (self.self_safety.end - self.self_safety.start)
                    )
                    if self_weight_full_at_episode_start:
                        self.self_safety.full_weight_steps += int(transitions)
                return {
                    "became_strict": False,
                    "rolling_task_reach_rate": None,
                    "rolling_goal_success_rate": goal_rate,
                    "rolling_orientation_success_rate": orientation_rate,
                    "goal_scale": self.goal.scale,
                    "orientation_scale": self.orientation.scale,
                    "position_tolerance": self.position_tolerance,
                    "orientation_tolerance": self.orientation_tolerance,
                    "lambda_self": self.lambda_self,
                }
            # Phase predicates are sampled before incorporating this episode so
            # its fixed reward contract cannot qualify itself retroactively.
            pose_phase_complete_at_episode_start = self.pose_phase_complete
            self_weight_full_at_episode_start = bool(
                np.isclose(
                    self.self_safety.weight,
                    self.self_safety.end,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            if curriculum_eligible:
                self.goal.outcomes.append(bool(position_reached))
                goal_rate = float(np.mean(self.goal.outcomes))
                if (not self.goal_at_full_scale
                        and len(self.goal.outcomes) == self.goal.outcomes.maxlen
                        and goal_rate >= self.goal.floor
                        and self.goal.current_level_steps >= self.goal.min_transitions_per_level):
                    self.goal.eligible_steps += self.goal.current_level_steps
                    self.goal.level_index += 1
                    self.goal.scale = self.goal.levels[self.goal.level_index]
                    self.goal.current_level_steps = 0
                    self.goal.outcomes.clear()
                    goal_rate = None
            # Do not use the episode that merely crossed the position-consolidation
            # threshold to advance the pose curriculum.  Its contract was chosen
            # before S0-R became eligible.
            position_phase_complete_at_episode_start = bool(
                self.goal_at_full_scale
                and self.goal.full_scale_steps - int(transitions)
                >= self.goal_full_scale_min_steps
            )
            if curriculum_eligible and pose_phase_complete_at_episode_start:
                self.self_safety.eligible_steps += int(transitions)
                fraction = min(
                    self.self_safety.eligible_steps / self.self_safety.ramp_steps, 1.0
                )
                self.self_safety.weight = float(
                    self.self_safety.start
                    + fraction * (self.self_safety.end - self.self_safety.start)
                )
                if self_weight_full_at_episode_start:
                    self.self_safety.full_weight_steps += int(transitions)
            if curriculum_eligible and position_phase_complete_at_episode_start:
                if orientation_anchor:
                    self.orientation.anchor_outcomes.append(bool(position_reached))
                else:
                    self.orientation.current_level_steps += int(transitions)
                    self.orientation.outcomes.append(bool(task_reached))
                    # At eta=0 the current task is itself a position anchor, which
                    # bootstraps the retention window before anchor mixing starts.
                    if np.isclose(
                        self.orientation.scale, self.orientation.start,
                        rtol=0.0, atol=1e-12,
                    ):
                        self.orientation.anchor_outcomes.append(bool(position_reached))
                    orientation_rate = float(np.mean(self.orientation.outcomes))
                    anchor_rate = self.orientation_anchor_success_rate
                    if self.advance_orientation_if_ready():
                        orientation_rate = None
        if scene == "none":
            return {
                "became_strict": False,
                "rolling_task_reach_rate": None,
                "rolling_goal_success_rate": goal_rate,
                "rolling_orientation_success_rate": orientation_rate,
                "goal_scale": self.goal.scale,
                "orientation_scale": self.orientation.scale,
                "orientation_tolerance": self.orientation_tolerance,
                "lambda_self": self.lambda_self,
            }
        state = self.states[scene]; was_strict = state.strict; state.outcomes.append(bool(task_reached))
        rate = float(np.mean(state.outcomes))
        became = False
        if not state.strict and len(state.outcomes) == 100 and rate >= state.floor:
            state.eligible_steps += int(transitions)
            old = state.xi
            state.xi = min(1.0, .02 + .98 * state.eligible_steps / self.ramp_steps)
            if old < 1.0 and state.xi >= 1.0:
                state.strict = True; became = True
        if was_strict:
            state.strict_steps += int(transitions)
        return {
            "became_strict": became,
            "rolling_task_reach_rate": rate,
            "rolling_goal_success_rate": goal_rate,
            "rolling_orientation_success_rate": orientation_rate,
            "goal_scale": self.goal.scale,
            "orientation_scale": self.orientation.scale,
            "orientation_tolerance": self.orientation_tolerance,
            "lambda_self": self.lambda_self,
            "xi": state.xi,
            "strict": state.strict,
        }

    def xi_map(self) -> dict[str, float]:
        return {name: state.xi for name, state in self.states.items()}

    def inherit_task_state(self, previous: dict[str, Any]) -> None:
        """Carry the completed S0 task and self-safety contract into S1/S2."""
        self.goal = copy.deepcopy(previous["goal"])
        self.orientation = copy.deepcopy(previous["orientation"])
        self.joint_pose_levels = tuple(copy.deepcopy(previous.get("joint_pose_levels", ())))
        self._refresh_precision_only_task_space()
        self.self_safety = copy.deepcopy(previous["self_safety"])

    def _refresh_precision_only_task_space(self) -> None:
        self.precision_only_task_space = False
        if not self.joint_pose_levels:
            return
        required = (
            "target_distance_min_m", "target_distance_max_m",
            "target_orientation_min_rad", "target_orientation_max_rad",
        )
        if not all(all(key in level for key in required) for level in self.joint_pose_levels):
            return
        first = tuple(self.joint_pose_levels[0][key] for key in required)
        self.precision_only_task_space = all(
            np.allclose(
                tuple(level[key] for key in required), first,
                rtol=0.0, atol=1e-12,
            )
            for level in self.joint_pose_levels[1:]
        )

    def inherit_scene_state(self, previous: dict[str, Any], scene: str) -> None:
        """Carry one completed scene curriculum without replacing stage policy."""
        if scene not in self.states or scene not in previous["states"]:
            raise ValueError(f"cannot inherit unknown scene {scene!r}")
        self.states[scene] = copy.deepcopy(previous["states"][scene])

    def state_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "ramp_steps": self.ramp_steps, "probabilities": self.probabilities,
                "goal_full_scale_min_steps": self.goal_full_scale_min_steps,
                "orientation_full_scale_min_steps": self.orientation_full_scale_min_steps,
                "self_full_weight_min_steps": self.self_full_weight_min_steps,
                "self_probe_collision_ceiling": self.self_probe_collision_ceiling,
                "goal": self.goal, "orientation": self.orientation,
                "joint_pose_levels": self.joint_pose_levels,
                "self_safety": self.self_safety,
                "states": self.states, "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["stage"] != self.stage: raise ValueError("checkpoint curriculum stage mismatch")
        self.ramp_steps = state["ramp_steps"]; self.probabilities = state["probabilities"]
        self.goal_full_scale_min_steps = int(state["goal_full_scale_min_steps"])
        self.orientation_full_scale_min_steps = int(
            state["orientation_full_scale_min_steps"]
        )
        self.self_full_weight_min_steps = int(state["self_full_weight_min_steps"])
        self.self_probe_collision_ceiling = float(
            state.get(
                "self_probe_collision_ceiling", self.self_probe_collision_ceiling,
            )
        )
        self.goal = state["goal"]
        self.orientation = state["orientation"]
        self.joint_pose_levels = tuple(state.get("joint_pose_levels", ()))
        self._refresh_precision_only_task_space()
        self.self_safety = state["self_safety"]
        self.states = state["states"]
        self.rng.bit_generator.state = state["rng_state"]
