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

    @property
    def joint_count(self) -> int:
        return len(self.config.joint_names)

    def load(self, rng: np.random.Generator) -> int:
        self.robot_id = p.loadURDF(
            str(self.urdf),
            basePosition=self.config.base_position,
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        self._resolve_references()
        self.reset_joints(rng)
        return self.robot_id

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
