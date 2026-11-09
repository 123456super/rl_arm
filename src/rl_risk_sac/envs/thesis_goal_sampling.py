"""Fixed-orientation goal sampling for the active S0/S1/S2 pipeline."""

from __future__ import annotations

import numpy as np
import pybullet as p

from rl_risk_sac.tasks.reference_reaching import pose_error


class GoalSamplingMixin:
    def _propose_reference_goal_position(self) -> np.ndarray | None:
        """Draw one FTP4RM ``xyztorus`` position proposal."""
        x = float(self.rng.uniform(-1.0, 1.0))
        y = float(self.rng.uniform(0.0, 1.0))
        z = float(self.rng.uniform(0.0, 1.0))
        if x * x + y * y + z * z > 0.5625:
            return None
        if x * x + y * y < 0.09 or z < 0.05:
            return None
        return np.asarray([x, y, z], dtype=np.float32)

    def _reset_fixed_initial_state(self) -> None:
        q = self.fixed_initial_joint_positions
        if np.any(q < self.robot.joint_lower_limits) or np.any(
            q > self.robot.joint_upper_limits
        ):
            raise ValueError("fixed initial joint positions exceed robot joint limits")
        self._set_joint_state(q)
        p.performCollisionDetection(physicsClientId=self.client_id)
        events = self._collision_events()
        if any(events.values()) or self._self_geometry().distance_min < self.d_self_safe:
            raise ValueError(
                "fixed initial configuration is colliding or lacks self clearance"
            )

    def _sample_fixed_orientation_goal_pose(self) -> None:
        """Sample a position and retain only an IK-valid, collision-free target."""
        initial_q, initial_qdot = self.robot.joint_state_exact()
        initial_position, initial_quaternion = self.robot.end_effector_pose()
        target_quaternion = self.fixed_goal_quaternion
        try:
            for attempt in range(1, self.goal_sample_max_attempts + 1):
                target_position = self._propose_reference_goal_position()
                if target_position is None:
                    continue
                position_error, orientation_error = pose_error(
                    target_position,
                    target_quaternion,
                    initial_position,
                    initial_quaternion,
                )
                if (
                    np.linalg.norm(position_error) <= self.position_tolerance
                    and np.linalg.norm(orientation_error) <= self.orientation_tolerance
                ):
                    continue
                validation = self.robot.check_pose_ik(
                    target_position,
                    target_quaternion,
                    position_tolerance=self.position_tolerance,
                    orientation_tolerance=self.orientation_tolerance,
                    rest_pose=initial_q,
                )
                if (
                    not validation["reachable"]
                    or not validation["finite"]
                    or not validation["within_limits"]
                    or not np.isfinite(validation["position_error_m"])
                    or not np.isfinite(validation["orientation_error_rad"])
                    or validation["position_error_m"] > self.position_tolerance
                    or validation["orientation_error_rad"] > self.orientation_tolerance
                ):
                    continue
                candidate = np.asarray(validation["q_ik"], dtype=np.float64)
                candidate = (candidate + np.pi) % (2.0 * np.pi) - np.pi
                command_lower = np.maximum(
                    self.robot.joint_lower_limits + self.joint_target_margin,
                    -self.action_limit,
                )
                command_upper = np.minimum(
                    self.robot.joint_upper_limits - self.joint_target_margin,
                    self.action_limit,
                )
                if np.any(candidate < command_lower) or np.any(candidate > command_upper):
                    continue
                self._set_joint_state(candidate)
                p.performCollisionDetection(physicsClientId=self.client_id)
                events = self._collision_events()
                clearance = self._self_geometry().distance_min
                self._set_joint_state(initial_q, initial_qdot)
                if any(events.values()) or clearance < self.d_self_safe:
                    continue
                self.goal_position = target_position.copy()
                self.goal_quaternion = target_quaternion.copy()
                self.goal_joint_positions = candidate.astype(np.float32)
                self.goal_ik_validation = {
                    **validation,
                    "q_ik": candidate.copy(),
                    "collision_free": True,
                    "self_clearance_m": float(clearance),
                }
                self.goal_sample_attempts_last = attempt
                return
        finally:
            self._set_joint_state(initial_q, initial_qdot)
            p.performCollisionDetection(physicsClientId=self.client_id)
        raise RuntimeError(
            "failed to sample an IK-valid, collision-free fixed-orientation goal "
            f"in {self.goal_sample_max_attempts} proposals"
        )
