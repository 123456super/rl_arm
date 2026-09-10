from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class ObstacleState:
    """Position and velocity of one spherical obstacle."""

    center: np.ndarray
    velocity: np.ndarray
    enabled: bool


class ObstacleProvider(Protocol):
    """Interface for obstacle state machines used by the environment."""

    def reset(self, rng: np.random.Generator) -> tuple[ObstacleState, ...]: ...

    def advance(self, dt: float) -> tuple[ObstacleState, ...]: ...


class SphericalObstacleProvider:
    """Spherical obstacle generator with random or named crossing scenarios.

    它只维护障碍物的运动状态，不创建 PyBullet 物体。环境会把这里的状态
    同步到可视化/碰撞球上。
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.enabled = bool(config.get("enabled", True))
        self.episode_enable_probability = float(config.get("episode_enable_probability", 1.0))
        self.scenario = str(config.get("scenario", "random"))
        self.count = max(1, int(config.get("count", 1)))
        self._states = tuple(self._disabled_state() for _ in range(self.count))

    def reset(self, rng: np.random.Generator) -> tuple[ObstacleState, ...]:
        """Sample initial obstacle positions and velocities for a new episode."""
        # A small no-obstacle training mixture prevents the zero-risk placeholder
        # observation from becoming an unseen deployment state.  Avoid drawing
        # from the RNG at probability one so legacy/default trajectories remain
        # bitwise reproducible.
        episode_enabled = self.enabled
        if episode_enabled and self.episode_enable_probability <= 0.0:
            episode_enabled = False
        elif episode_enabled and self.episode_enable_probability < 1.0:
            episode_enabled = bool(rng.random() < self.episode_enable_probability)
        if not episode_enabled:
            self._states = tuple(self._disabled_state() for _ in range(self.count))
            return self._states

        states = []
        for _ in range(self.count):
            if self.scenario == "random":
                center, velocity = self._sample_random(rng)
            else:
                center, velocity = self._sample_named(rng, self.scenario)
            states.append(ObstacleState(center=center, velocity=velocity, enabled=True))
        self._states = tuple(states)
        return self._states

    def advance(self, dt: float) -> tuple[ObstacleState, ...]:
        """Move obstacles with constant velocity and bounce at configured bounds."""
        if not self.enabled:
            return self._states

        next_states = []
        bounds = self.config["bounds"]
        for state in self._states:
            if not state.enabled:
                next_states.append(state)
                continue
            center = (state.center + state.velocity * dt).astype(np.float32)
            velocity = state.velocity.copy()
            for axis_index, axis in enumerate(("x", "y", "z")):
                low, high = bounds[axis]
                if center[axis_index] < low or center[axis_index] > high:
                    velocity[axis_index] *= -1.0
                    center[axis_index] = np.clip(center[axis_index], low, high)
            next_states.append(ObstacleState(center=center, velocity=velocity, enabled=state.enabled))
        self._states = tuple(next_states)
        return self._states

    def _sample_random(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Sample a crossing path from one side of the workspace to the other."""
        random_cfg = self.config["random"]
        side = -1.0 if rng.random() < 0.5 else 1.0
        center = np.array(
            [
                rng.uniform(*random_cfg["x_range"]),
                side * rng.uniform(*random_cfg["start_y_abs_range"]),
                rng.uniform(*random_cfg["z_range"]),
            ],
            dtype=np.float32,
        )
        target = np.array(
            [
                rng.uniform(*random_cfg["x_range"]),
                -side * rng.uniform(*random_cfg["target_y_abs_range"]),
                rng.uniform(*random_cfg["z_range"]),
            ],
            dtype=np.float32,
        )
        return center, self._velocity_toward(rng, center, target)

    def _sample_named(self, rng: np.random.Generator, scenario: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a controlled crossing path near a named robot-link region."""
        scenarios = self.config["scenarios"]
        if scenario not in scenarios:
            raise ValueError(f"Unknown obstacle scenario {scenario!r}; expected random or one of {sorted(scenarios)}")

        scenario_cfg = scenarios[scenario]
        side = -1.0 if rng.random() < 0.5 else 1.0
        x_center = rng.uniform(*scenario_cfg["x_range"])
        z_range = scenario_cfg["z_range"]
        center = np.array(
            [x_center, side * float(scenario_cfg["start_y_abs"]), rng.uniform(*z_range)],
            dtype=np.float32,
        )
        target = np.array(
            [x_center, -side * float(scenario_cfg["target_y_abs"]), rng.uniform(*z_range)],
            dtype=np.float32,
        )
        return center, self._velocity_toward(rng, center, target)

    def _velocity_toward(
        self,
        rng: np.random.Generator,
        center: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Choose a random speed and point the velocity from center to target."""
        direction = target - center
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        speed = rng.uniform(*self.config["speed_range"])
        return (direction * speed).astype(np.float32)

    def _disabled_state(self) -> ObstacleState:
        return ObstacleState(
            center=np.asarray(self.config["disabled_position"], dtype=np.float32).copy(),
            velocity=np.zeros(3, dtype=np.float32),
            enabled=False,
        )
