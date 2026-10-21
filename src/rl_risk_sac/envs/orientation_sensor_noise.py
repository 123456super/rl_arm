"""World-frame SO(3) measurement noise, independent of the task/position RNG."""

from __future__ import annotations

import numpy as np


class OrientationSensorNoise:
    """Apply exp([eta]x) R, with IID Gaussian rotation-vector components.

    Quaternions use PyBullet's [x, y, z, w] convention. The standard deviation
    is in radians per rotation-vector component, not a bound on the angle.
    """

    def __init__(self, config: dict) -> None:
        self.std_rad = float(config.get("orientation_std_rad", 0.0))
        self.seed = int(config.get("seed", 71001))
        if not np.isfinite(self.std_rad) or self.std_rad < 0.0:
            raise ValueError("orientation noise standard deviation must be finite and nonnegative")
        if self.seed < 0:
            raise ValueError("sensor noise seed must be nonnegative")
        self.episode_seed = 0

    @property
    def enabled(self) -> bool:
        return self.std_rad > 0.0

    def reset(self, seed: int | None) -> None:
        self.episode_seed = self.episode_seed + 1 if seed is None else int(seed)

    def measure(self, true_quaternion: np.ndarray, step: int) -> np.ndarray:
        if not self.enabled:
            return true_quaternion
        # Separate namespace from the existing translation stream. Re-reading
        # a control sample is idempotent and does not consume the task RNG.
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, self.episode_seed, int(step), 1]
        ))
        rotation_vector = self.std_rad * rng.standard_normal(3)
        angle = float(np.linalg.norm(rotation_vector))
        delta_xyz = .5 * np.sinc(angle / (2 * np.pi)) * rotation_vector
        delta_w = np.cos(.5 * angle)
        quaternion = np.asarray(true_quaternion, dtype=np.float64)
        xyz = (delta_w * quaternion[:3] + quaternion[3] * delta_xyz
               + np.cross(delta_xyz, quaternion[:3]))
        w = delta_w * quaternion[3] - np.dot(delta_xyz, quaternion[:3])
        measured = np.append(xyz, w)
        return (measured / np.linalg.norm(measured)).astype(np.float32)
