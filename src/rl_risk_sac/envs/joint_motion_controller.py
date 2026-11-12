"""Joint-velocity- and acceleration-bounded execution of policy targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointMotionCommand:
    velocity: np.ndarray
    joint_velocity_limited: bool
    joint_acceleration_limited: bool
    tcp_velocity_limited: bool
    tcp_linear_speed: float
    tcp_angular_speed: float


class BoundedJointMotionController:
    """Convert a joint-position goal into a conservative velocity command.

    The policy may request an arbitrarily large joint increment.  This layer
    preserves that goal but approaches it through the same velocity and
    acceleration contract used by the CB3.1 deployment controller.
    """

    def __init__(
        self,
        *,
        joint_velocity_limits: np.ndarray,
        joint_acceleration_limits: np.ndarray,
        position_gain: float,
        position_tolerance: float,
    ) -> None:
        self.velocity_limits = self._positive_vector(
            joint_velocity_limits, "joint_velocity_limits"
        )
        self.acceleration_limits = self._positive_vector(
            joint_acceleration_limits, "joint_acceleration_limits"
        )
        self.position_gain = self._positive_scalar(position_gain, "position_gain")
        self.position_tolerance = self._positive_scalar(
            position_tolerance, "position_tolerance"
        )
        self.previous_velocity = np.zeros(6, dtype=np.float64)

    @staticmethod
    def _positive_vector(value: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (6,) or not np.isfinite(array).all() or np.any(array <= 0.0):
            raise ValueError(f"{name} must contain six finite positive values")
        return array

    @staticmethod
    def _positive_scalar(value: float, name: str) -> float:
        number = float(value)
        if not np.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return number

    def reset(self, velocity: np.ndarray | None = None) -> None:
        initial = (
            np.zeros(6, dtype=np.float64)
            if velocity is None else np.asarray(velocity, dtype=np.float64)
        )
        if initial.shape != (6,) or not np.isfinite(initial).all():
            raise ValueError("initial velocity must contain six finite values")
        self.previous_velocity = np.clip(
            initial, -self.velocity_limits, self.velocity_limits
        )

    def command(
        self,
        q: np.ndarray,
        goal: np.ndarray,
        dt: float,
    ) -> JointMotionCommand:
        q = np.asarray(q, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        dt = self._positive_scalar(dt, "dt")
        if q.shape != (6,) or goal.shape != (6,):
            raise ValueError("q and goal must have shape (6,)")

        error = goal - q
        # The stopping-speed bound prevents the command from overshooting a
        # target even when the proportional term is still large.
        stopping_speed = np.sqrt(
            np.maximum(0.0, 2.0 * self.acceleration_limits * np.abs(error))
        )
        raw = np.sign(error) * np.minimum(
            self.position_gain * np.abs(error), stopping_speed
        )
        raw[np.abs(error) <= self.position_tolerance] = 0.0
        velocity_clipped = np.clip(raw, -self.velocity_limits, self.velocity_limits)
        joint_velocity_limited = bool(np.any(np.abs(raw) > self.velocity_limits + 1e-12))

        maximum_change = self.acceleration_limits * dt
        accelerated = np.clip(
            velocity_clipped,
            self.previous_velocity - maximum_change,
            self.previous_velocity + maximum_change,
        )
        joint_acceleration_limited = bool(
            np.any(np.abs(accelerated - velocity_clipped) > 1e-12)
        )

        commanded = accelerated
        self.previous_velocity = commanded
        return JointMotionCommand(
            velocity=commanded.astype(np.float32),
            joint_velocity_limited=joint_velocity_limited,
            joint_acceleration_limited=joint_acceleration_limited,
            # Kept in the diagnostic contract so old CSV/evaluation tooling can
            # explicitly report that Cartesian limiting is disabled.
            tcp_velocity_limited=False,
            tcp_linear_speed=0.0,
            tcp_angular_speed=0.0,
        )
