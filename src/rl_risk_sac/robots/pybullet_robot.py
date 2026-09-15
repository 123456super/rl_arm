from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p

from rl_risk_sac.robots.ur5_capsules import CapsuleState, UR5CapsuleModel
from rl_risk_sac.utils.risk import closest_point_on_segment
from rl_risk_sac.utils.runtime_config import RobotRuntimeConfig


class PyBulletRobot:
    """Own PyBullet-specific robot loading, state reads, and geometry queries."""

    def __init__(self, config: RobotRuntimeConfig, urdf: Path, physics_client_id: int) -> None:
        self.config = config
        self.urdf = urdf
        self.physics_client_id = physics_client_id
        self.capsule_model = UR5CapsuleModel(config.capsules)
        self.robot_id: int | None = None
        self.joint_ids: list[int] = []
        self.tool_link_id = -1
        self._joint_lower: np.ndarray | None = None
        self._joint_upper: np.ndarray | None = None

    @property
    def joint_count(self) -> int:
        return len(self.config.joint_names)

    def load(self, rng: np.random.Generator, *, enable_self_collision: bool = False) -> int:
        flags = p.URDF_USE_SELF_COLLISION if enable_self_collision else 0
        self.robot_id = p.loadURDF(
            str(self.urdf),
            basePosition=self.config.base_position,
            useFixedBase=True,
            flags=flags,
            physicsClientId=self.physics_client_id,
        )
        self._resolve_references()
        self._read_joint_limits()
        self.reset_joints(rng)
        return self.robot_id

    @property
    def joint_lower_limits(self) -> np.ndarray:
        """Hard lower joint limits copied from the URDF (rad)."""
        self._require_loaded()
        if self._joint_lower is None:
            self._read_joint_limits()
        return self._joint_lower.copy()

    @property
    def joint_upper_limits(self) -> np.ndarray:
        """Hard upper joint limits copied from the URDF (rad)."""
        self._require_loaded()
        if self._joint_upper is None:
            self._read_joint_limits()
        return self._joint_upper.copy()

    def _read_joint_limits(self) -> None:
        """Cache the per-joint hard limits reported by PyBullet.

        Non-finite or degenerate bounds fall back to +/-pi so that downstream
        limit-aware diagnostics never divide by zero or compare against NaN.
        """
        infos = [
            p.getJointInfo(self._require_loaded(), joint_id, physicsClientId=self.physics_client_id)
            for joint_id in self.joint_ids
        ]
        lower = np.asarray([float(info[8]) for info in infos], dtype=np.float64)
        upper = np.asarray([float(info[9]) for info in infos], dtype=np.float64)
        invalid = ~np.isfinite(lower) | ~np.isfinite(upper) | (upper <= lower)
        lower[invalid], upper[invalid] = -np.pi, np.pi
        self._joint_lower = lower
        self._joint_upper = upper

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        states = p.getJointStates(self._require_loaded(), self.joint_ids, physicsClientId=self.physics_client_id)
        q = np.asarray([state[0] for state in states], dtype=np.float32)
        q_dot = np.asarray([state[1] for state in states], dtype=np.float32)
        return q, q_dot

    def end_effector_state(self) -> tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(
            self._require_loaded(),
            self.tool_link_id,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
            physicsClientId=self.physics_client_id,
        )
        return np.asarray(state[4], dtype=np.float32), np.asarray(state[6], dtype=np.float32)

    def end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the configured end-link world position and quaternion."""
        state = p.getLinkState(
            self._require_loaded(),
            self.tool_link_id,
            computeForwardKinematics=True,
            physicsClientId=self.physics_client_id,
        )
        return np.asarray(state[4], dtype=np.float32), np.asarray(state[5], dtype=np.float32)

    def capsules(self) -> list[CapsuleState]:
        return self.capsule_model.states(self._require_loaded(), self.physics_client_id)

    def reset_joints(self, rng: np.random.Generator) -> None:
        reset = self.config.reset
        default = np.asarray(reset.default_joint_positions, dtype=np.float32)
        noise = rng.uniform(-reset.joint_noise_range, reset.joint_noise_range, size=self.joint_count).astype(np.float32)
        for joint_id, joint_value in zip(self.joint_ids, default + noise):
            p.resetJointState(
                self._require_loaded(),
                joint_id,
                float(joint_value),
                targetVelocity=0.0,
                physicsClientId=self.physics_client_id,
            )

    def check_position_ik(self, target: np.ndarray, fk_tolerance: float) -> dict[str, object]:
        """Solve position-only IK and verify limits plus FK without changing simulator state."""
        robot_id = self._require_loaded()
        joint_info = [
            p.getJointInfo(robot_id, joint_id, physicsClientId=self.physics_client_id)
            for joint_id in self.joint_ids
        ]
        lower = np.asarray([float(info[8]) for info in joint_info], dtype=np.float64)
        upper = np.asarray([float(info[9]) for info in joint_info], dtype=np.float64)
        invalid = ~np.isfinite(lower) | ~np.isfinite(upper) | (upper <= lower)
        lower[invalid], upper[invalid] = -np.pi, np.pi
        current_q, current_qdot = self.joint_state()
        solution = p.calculateInverseKinematics(
            robot_id,
            self.tool_link_id,
            np.asarray(target, dtype=np.float64).tolist(),
            lowerLimits=lower.tolist(),
            upperLimits=upper.tolist(),
            jointRanges=(upper - lower).tolist(),
            restPoses=current_q.astype(np.float64).tolist(),
            maxNumIterations=200,
            residualThreshold=1.0e-6,
            physicsClientId=self.physics_client_id,
        )
        q_ik = np.asarray(solution[: self.joint_count], dtype=np.float64)
        finite = q_ik.shape == (self.joint_count,) and bool(np.isfinite(q_ik).all())
        within_limits = finite and bool(np.all(q_ik >= lower - 1e-8) and np.all(q_ik <= upper + 1e-8))
        fk_error = float("inf")
        if within_limits:
            state_id = p.saveState(physicsClientId=self.physics_client_id)
            try:
                for joint_id, value in zip(self.joint_ids, q_ik):
                    p.resetJointState(
                        robot_id,
                        joint_id,
                        float(value),
                        targetVelocity=0.0,
                        physicsClientId=self.physics_client_id,
                    )
                fk_position, _ = self.end_effector_state()
                fk_error = float(np.linalg.norm(fk_position - np.asarray(target, dtype=np.float32)))
            finally:
                p.restoreState(stateId=state_id, physicsClientId=self.physics_client_id)
                p.removeState(stateUniqueId=state_id, physicsClientId=self.physics_client_id)
        return {
            "q_ik": q_ik,
            "finite": finite,
            "within_limits": within_limits,
            "fk_error_m": fk_error,
            "reachable": bool(within_limits and fk_error <= fk_tolerance),
        }

    def check_pose_ik(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        *,
        position_tolerance: float = 0.01,
        orientation_tolerance: float = 0.05,
        rest_pose: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Solve pose IK and verify the solution by an explicit FK round trip.

        The simulator state is restored before returning. Quaternion comparison
        uses ``abs(dot(q1, q2))`` so that the equivalent signs ``q`` and ``-q``
        produce the same angular error.
        """
        robot_id = self._require_loaded()
        lower = self.joint_lower_limits.astype(np.float64)
        upper = self.joint_upper_limits.astype(np.float64)
        current_q, _ = self.joint_state()
        rest = current_q.astype(np.float64) if rest_pose is None else np.asarray(rest_pose, dtype=np.float64)
        if rest.shape != (self.joint_count,) or not np.isfinite(rest).all():
            raise ValueError(f"rest_pose must contain {self.joint_count} finite joint values")

        target_position = np.asarray(target_position, dtype=np.float64)
        target_quaternion = np.asarray(target_quaternion, dtype=np.float64)
        target_norm = float(np.linalg.norm(target_quaternion))
        if target_position.shape != (3,) or target_quaternion.shape != (4,) or target_norm <= 0.0:
            raise ValueError("target pose must contain a 3D position and a non-zero quaternion")
        target_quaternion = target_quaternion / target_norm

        solution = p.calculateInverseKinematics(
            robot_id,
            self.tool_link_id,
            target_position.tolist(),
            targetOrientation=target_quaternion.tolist(),
            lowerLimits=lower.tolist(),
            upperLimits=upper.tolist(),
            jointRanges=(upper - lower).tolist(),
            restPoses=rest.tolist(),
            maxNumIterations=200,
            residualThreshold=1.0e-6,
            physicsClientId=self.physics_client_id,
        )
        q_ik = np.asarray(solution[: self.joint_count], dtype=np.float64)
        finite = q_ik.shape == (self.joint_count,) and bool(np.isfinite(q_ik).all())
        within_limits = finite and bool(np.all(q_ik >= lower - 1e-8) and np.all(q_ik <= upper + 1e-8))
        position_error = float("inf")
        orientation_error = float("inf")
        if within_limits:
            state_id = p.saveState(physicsClientId=self.physics_client_id)
            try:
                for joint_id, value in zip(self.joint_ids, q_ik):
                    p.resetJointState(
                        robot_id,
                        joint_id,
                        float(value),
                        targetVelocity=0.0,
                        physicsClientId=self.physics_client_id,
                    )
                fk_position, fk_quaternion = self.end_effector_pose()
                position_error = float(np.linalg.norm(fk_position.astype(np.float64) - target_position))
                fk_quaternion = fk_quaternion.astype(np.float64)
                fk_quaternion /= max(float(np.linalg.norm(fk_quaternion)), np.finfo(np.float64).eps)
                quaternion_dot = float(np.clip(abs(np.dot(fk_quaternion, target_quaternion)), 0.0, 1.0))
                orientation_error = float(2.0 * np.arccos(quaternion_dot))
            finally:
                p.restoreState(stateId=state_id, physicsClientId=self.physics_client_id)
                p.removeState(stateUniqueId=state_id, physicsClientId=self.physics_client_id)
        return {
            "q_ik": q_ik,
            "finite": finite,
            "within_limits": within_limits,
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "reachable": bool(
                within_limits
                and position_error <= position_tolerance
                and orientation_error <= orientation_tolerance
            ),
        }

    def surface_distance_jacobian(
        self,
        capsule_index: int,
        obstacle_center: np.ndarray,
        finite_difference_epsilon: float,
        numerical_epsilon: float,
    ) -> np.ndarray:
        """Estimate one capsule-to-point distance Jacobian by finite differences."""
        q, q_dot = self.joint_state()
        base_capsule = self.capsules()[capsule_index]
        closest, _ = closest_point_on_segment(obstacle_center, base_capsule.start, base_capsule.end)
        delta = obstacle_center - closest
        direction = delta / (float(np.linalg.norm(delta)) + numerical_epsilon)
        jacobian = np.zeros(self.joint_count, dtype=np.float32)

        for offset, joint_id in enumerate(self.joint_ids):
            p.resetJointState(
                self._require_loaded(),
                joint_id,
                float(q[offset] + finite_difference_epsilon),
                targetVelocity=float(q_dot[offset]),
                physicsClientId=self.physics_client_id,
            )
            perturbed = self.capsules()[capsule_index]
            perturbed_closest, _ = closest_point_on_segment(
                obstacle_center, perturbed.start, perturbed.end
            )
            point_jacobian = (perturbed_closest - closest) / finite_difference_epsilon
            jacobian[offset] = -float(np.dot(direction, point_jacobian))
            p.resetJointState(
                self._require_loaded(),
                joint_id,
                float(q[offset]),
                targetVelocity=float(q_dot[offset]),
                physicsClientId=self.physics_client_id,
            )
        return jacobian

    def _resolve_references(self) -> None:
        joint_name_to_id: dict[str, int] = {}
        link_name_to_id = {"base": -1}
        robot_id = self._require_loaded()
        for joint_id in range(p.getNumJoints(robot_id, physicsClientId=self.physics_client_id)):
            info = p.getJointInfo(robot_id, joint_id, physicsClientId=self.physics_client_id)
            joint_name_to_id[info[1].decode("utf-8")] = joint_id
            link_name_to_id[info[12].decode("utf-8")] = joint_id
        self.joint_ids = [self._resolve_name(name, joint_name_to_id, "joint") for name in self.config.joint_names]
        self.tool_link_id = self._resolve_name(self.config.tool_link_name, link_name_to_id, "link")
        self.capsule_model.resolve_link_names(link_name_to_id)

    @staticmethod
    def _resolve_name(name: str, name_to_id: dict[str, int], kind: str) -> int:
        if name not in name_to_id:
            available = ", ".join(sorted(name_to_id))
            raise KeyError(f"Unknown {kind} name {name!r}; available {kind}s: {available}")
        return name_to_id[name]

    def _require_loaded(self) -> int:
        if self.robot_id is None:
            raise RuntimeError("Robot must be loaded before its state is queried")
        return self.robot_id
