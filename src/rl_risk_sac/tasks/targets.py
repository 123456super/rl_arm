from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class TargetState:
    position: np.ndarray
    velocity: np.ndarray


class TargetProvider(Protocol):
    def reset(self, rng: np.random.Generator) -> TargetState: ...

    def advance(self, dt: float) -> TargetState: ...


class WorkspaceTargetProvider:
    """Static or linearly moving target independent from simulator state."""

    def __init__(self, config: dict[str, Any], workspace: dict[str, list[float]]) -> None:
        self.config = config
        self.workspace = workspace
        self.mode = str(config.get("mode", "static"))
        self._position = np.zeros(3, dtype=np.float32)
        self._velocity = np.zeros(3, dtype=np.float32)

    def reset(self, rng: np.random.Generator) -> TargetState:
        if self.config.get("fixed", False):
            self._position = np.asarray(self.config["position"], dtype=np.float32).copy()
        else:
            self._position = np.asarray(
                [rng.uniform(*self.workspace[axis]) for axis in ("x", "y", "z")],
                dtype=np.float32,
            )

        if self.mode == "static":
            self._velocity = np.zeros(3, dtype=np.float32)
        elif self.mode == "linear_bounce":
            direction = rng.normal(size=3).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-8
            speed = rng.uniform(*self.config["speed_range"])
            self._velocity = (direction * speed).astype(np.float32)
        else:
            raise ValueError(f"Unknown goal mode: {self.mode}")
        return self._state()

    def advance(self, dt: float) -> TargetState:
        if self.mode == "linear_bounce":
            self._position = (self._position + self._velocity * dt).astype(np.float32)
            for axis_index, axis in enumerate(("x", "y", "z")):
                low, high = self.workspace[axis]
                if self._position[axis_index] < low or self._position[axis_index] > high:
                    self._velocity[axis_index] *= -1.0
                    self._position[axis_index] = np.clip(self._position[axis_index], low, high)
        return self._state()

    def _state(self) -> TargetState:
        return TargetState(self._position.copy(), self._velocity.copy())
