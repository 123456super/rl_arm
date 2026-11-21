"""Joint-velocity- and acceleration-bounded execution of policy targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointMotionCommand:
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
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
        trajectory_smoothing_enabled: bool = False,
        butterworth_cutoff_hz: float = 0.5,
        quintic_blend_duration_s: float = 0.4,
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
        self.trajectory_smoothing_enabled = bool(trajectory_smoothing_enabled)
        self.butterworth_cutoff_hz = self._positive_scalar(
            butterworth_cutoff_hz, "butterworth_cutoff_hz"
        )
        self.quintic_blend_duration_s = self._positive_scalar(
            quintic_blend_duration_s, "quintic_blend_duration_s"
        )
        self.previous_velocity = np.zeros(6, dtype=np.float64)
        self.previous_acceleration = np.zeros(6, dtype=np.float64)
        self.previous_policy_velocity = np.zeros(6, dtype=np.float64)
        self.previous_filtered_velocity = np.zeros(6, dtype=np.float64)
        self.blend_start_velocity = np.zeros(6, dtype=np.float64)
        self.blend_target_velocity = np.zeros(6, dtype=np.float64)
        self.blend_elapsed_s = 0.0
        self.policy_step_prepared = False

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
        self.previous_acceleration = np.zeros(6, dtype=np.float64)
        self.previous_policy_velocity = self.previous_velocity.copy()
        self.previous_filtered_velocity = self.previous_velocity.copy()
        self.blend_start_velocity = self.previous_velocity.copy()
        self.blend_target_velocity = self.previous_velocity.copy()
        self.blend_elapsed_s = 0.0
        self.policy_step_prepared = False

    def _target_velocity(self, q: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, bool]:
        error = goal - q
        stopping_speed = np.sqrt(
            np.maximum(0.0, 2.0 * self.acceleration_limits * np.abs(error))
        )
        raw = np.sign(error) * np.minimum(
            self.position_gain * np.abs(error), stopping_speed
        )
        raw[np.abs(error) <= self.position_tolerance] = 0.0
        clipped = np.clip(raw, -self.velocity_limits, self.velocity_limits)
        limited = bool(np.any(np.abs(raw) > self.velocity_limits + 1e-12))
        return clipped, limited

    def begin_policy_step(
        self,
        q: np.ndarray,
        goal: np.ndarray,
        policy_dt: float,
    ) -> None:
        """Latch and smooth one policy target before the 125 Hz servo loop.

        The Butterworth stage runs once per policy decision.  The resulting
        velocity is then joined to the currently executed velocity by a
        fifth-order smoothstep at the real-controller frequency.
        """
        if not self.trajectory_smoothing_enabled:
            return
        q = np.asarray(q, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        policy_dt = self._positive_scalar(policy_dt, "policy_dt")
        if q.shape != (6,) or goal.shape != (6,):
            raise ValueError("q and goal must have shape (6,)")
        policy_velocity, _ = self._target_velocity(q, goal)
        omega_c = 2.0 * np.pi * self.butterworth_cutoff_hz
        k = 2.0 / (omega_c * policy_dt)
        filtered = (
            policy_velocity
            + self.previous_policy_velocity
            - (1.0 - k) * self.previous_filtered_velocity
        ) / (1.0 + k)
        self.previous_policy_velocity = policy_velocity.copy()
        self.previous_filtered_velocity = filtered.copy()
        self.blend_start_velocity = self.previous_velocity.copy()
        self.blend_target_velocity = np.clip(
            filtered, -self.velocity_limits, self.velocity_limits
        )
        self.blend_elapsed_s = 0.0
        self.policy_step_prepared = True

    @staticmethod
    def _quintic_smoothstep(s: float) -> float:
        s = float(np.clip(s, 0.0, 1.0))
        return 6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3

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

        velocity_clipped, joint_velocity_limited = self._target_velocity(q, goal)
        if self.trajectory_smoothing_enabled:
            if not self.policy_step_prepared:
                raise RuntimeError(
                    "begin_policy_step must be called before the smoothed servo loop"
                )
            self.blend_elapsed_s = min(
                self.blend_elapsed_s + dt, self.quintic_blend_duration_s
            )
            alpha = self._quintic_smoothstep(
                self.blend_elapsed_s / self.quintic_blend_duration_s
            )
            blended = self.blend_start_velocity + alpha * (
                self.blend_target_velocity - self.blend_start_velocity
            )
            # The filtered target was already computed with the velocity and
            # stopping-speed bounds at the policy boundary.  Reapplying the
            # instantaneous stopping envelope here would introduce a new
            # discontinuity inside the blend and recreate the jerk that RTB
            # is intended to remove.
            velocity_clipped = np.clip(
                blended, -self.velocity_limits, self.velocity_limits
            )

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
        acceleration = (commanded - self.previous_velocity) / dt
        jerk = (acceleration - self.previous_acceleration) / dt
        self.previous_velocity = commanded
        self.previous_acceleration = acceleration
        return JointMotionCommand(
            velocity=commanded.astype(np.float32),
            acceleration=acceleration.astype(np.float32),
            jerk=jerk.astype(np.float32),
            joint_velocity_limited=joint_velocity_limited,
            joint_acceleration_limited=joint_acceleration_limited,
            # Kept in the diagnostic contract so old CSV/evaluation tooling can
            # explicitly report that Cartesian limiting is disabled.
            tcp_velocity_limited=False,
            tcp_linear_speed=0.0,
            tcp_angular_speed=0.0,
        )

    def predict_reachable_position(
        self,
        q: np.ndarray,
        goal: np.ndarray,
        dt: float,
        updates: int,
    ) -> np.ndarray:
        """Predict one policy period without mutating controller state.

        The prediction applies the same velocity, acceleration, proportional,
        stopping-speed and tolerance rules as :meth:`command` at every real
        controller tick.  Integrating the commanded velocity gives the ideal
        joint position reachable during the requested number of updates.  The
        rollout keeps its velocity history local so a Residual-policy feature
        query cannot change the command subsequently sent to the simulated or
        real robot.
        """
        position = np.asarray(q, dtype=np.float64).copy()
        target = np.asarray(goal, dtype=np.float64)
        dt = self._positive_scalar(dt, "dt")
        if position.shape != (6,) or target.shape != (6,):
            raise ValueError("q and goal must have shape (6,)")
        if not np.isfinite(position).all() or not np.isfinite(target).all():
            raise ValueError("q and goal must contain finite values")
        if not isinstance(updates, (int, np.integer)) or int(updates) < 1:
            raise ValueError("updates must be a positive integer")

        state = self._state_snapshot()
        try:
            if self.trajectory_smoothing_enabled:
                self.begin_policy_step(position, target, dt * int(updates))
            for _ in range(int(updates)):
                command = self.command(position, target, dt)
                position += command.velocity.astype(np.float64) * dt
            return position
        finally:
            self._restore_state(state)

    def _state_snapshot(self) -> tuple[np.ndarray, ...] | tuple:
        return (
            self.previous_velocity.copy(),
            self.previous_acceleration.copy(),
            self.previous_policy_velocity.copy(),
            self.previous_filtered_velocity.copy(),
            self.blend_start_velocity.copy(),
            self.blend_target_velocity.copy(),
            float(self.blend_elapsed_s),
            bool(self.policy_step_prepared),
        )

    def _restore_state(self, state: tuple) -> None:
        (
            self.previous_velocity,
            self.previous_acceleration,
            self.previous_policy_velocity,
            self.previous_filtered_velocity,
            self.blend_start_velocity,
            self.blend_target_velocity,
            self.blend_elapsed_s,
            self.policy_step_prepared,
        ) = state
