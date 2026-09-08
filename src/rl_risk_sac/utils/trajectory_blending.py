from __future__ import annotations

import numpy as np


def quintic_smoothstep(s: float | np.ndarray) -> float | np.ndarray:
    """Quintic blend with zero first and second derivatives at both ends."""
    clipped = np.clip(s, 0.0, 1.0)
    return 6.0 * clipped**5 - 15.0 * clipped**4 + 10.0 * clipped**3


class ButterworthQuinticRTB:
    """Fixed two-stage real-time trajectory blender for joint velocities.

    A first-order Butterworth low-pass filter is followed by a quintic
    interpolation from the currently executed velocity to the filtered target.
    """

    def __init__(self, joint_count: int, control_dt: float, cutoff_angular_frequency: float) -> None:
        if joint_count <= 0:
            raise ValueError("joint_count must be positive")
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if cutoff_angular_frequency <= 0.0:
            raise ValueError("cutoff_angular_frequency must be positive")
        self.joint_count = int(joint_count)
        self.control_dt = float(control_dt)
        self.cutoff_angular_frequency = float(cutoff_angular_frequency)
        self.previous_policy_velocity = np.zeros(self.joint_count, dtype=np.float32)
        self.previous_filtered_velocity = np.zeros(self.joint_count, dtype=np.float32)

    def reset(self, velocity: np.ndarray | None = None) -> None:
        initial = (
            np.zeros(self.joint_count, dtype=np.float32)
            if velocity is None
            else self._as_velocity(velocity)
        )
        self.previous_policy_velocity = initial.copy()
        self.previous_filtered_velocity = initial.copy()

    def filter(self, policy_velocity: np.ndarray) -> np.ndarray:
        policy_velocity = self._as_velocity(policy_velocity)
        k = 2.0 / (self.cutoff_angular_frequency * self.control_dt)
        filtered = (
            policy_velocity
            + self.previous_policy_velocity
            - (1.0 - k) * self.previous_filtered_velocity
        ) / (1.0 + k)
        self.previous_policy_velocity = policy_velocity.copy()
        self.previous_filtered_velocity = filtered.astype(np.float32)
        return self.previous_filtered_velocity.copy()

    def trajectory(
        self,
        start_velocity: np.ndarray,
        policy_velocity: np.ndarray,
        sample_count: int,
    ) -> np.ndarray:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        start = self._as_velocity(start_velocity)
        target = self.filter(policy_velocity)
        phase = np.arange(1, sample_count + 1, dtype=np.float32) / float(sample_count)
        blend = np.asarray(quintic_smoothstep(phase), dtype=np.float32)[:, None]
        return (start[None, :] + blend * (target - start)[None, :]).astype(np.float32)

    def _as_velocity(self, velocity: np.ndarray) -> np.ndarray:
        value = np.asarray(velocity, dtype=np.float32)
        if value.shape != (self.joint_count,):
            raise ValueError(f"Expected velocity shape {(self.joint_count,)}, got {value.shape}")
        return value
