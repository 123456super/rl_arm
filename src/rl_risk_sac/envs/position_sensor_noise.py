"""Position-measurement stress tests with a task-independent noise stream."""

from __future__ import annotations

import numpy as np


class PositionSensorNoise:
    """Add world-frame IID Gaussian translation noise, once per control sample.

    Keying samples by episode seed and control step makes repeated observation
    reads idempotent, and pairs the same standard-normal draws across noise
    levels without consuming the environment's target/reset RNG.
    """

    def __init__(self, config: dict) -> None:
        self.std_m = float(config.get("position_std_m", 0.0))
        self.seed = int(config.get("seed", 71001))
        if not np.isfinite(self.std_m) or self.std_m < 0.0:
            raise ValueError("position noise standard deviation must be finite and nonnegative")
        if self.seed < 0:
            raise ValueError("sensor noise seed must be nonnegative")
        self.episode_seed = 0

    @property
    def enabled(self) -> bool:
        return self.std_m > 0.0

    def reset(self, seed: int | None) -> None:
        self.episode_seed = self.episode_seed + 1 if seed is None else int(seed)

    def measure(self, true_position: np.ndarray, step: int) -> np.ndarray:
        if not self.enabled:
            return true_position
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, self.episode_seed, int(step)]
        ))
        return (np.asarray(true_position, dtype=np.float64)
                + self.std_m * rng.standard_normal(3)).astype(np.float32)
