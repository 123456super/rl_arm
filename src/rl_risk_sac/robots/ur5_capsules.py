from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p


@dataclass(frozen=True)
class CapsuleSpec:
    name: str
    parent_link: int
    child_link: int
    radius: float
    child_offset: np.ndarray


@dataclass
class CapsuleState:
    start: np.ndarray
    end: np.ndarray
    radius: float
    name: str


class UR5CapsuleModel:
    """Capsule approximation for the main links of the bundled UR5-like arm."""

    def __init__(self) -> None:
        self.specs = [
            CapsuleSpec("shoulder", -1, 0, 0.075, np.array([0.0, 0.0, 0.0])),
            CapsuleSpec("upper_arm", 1, 2, 0.06, np.array([0.0, 0.0, 0.0])),
            CapsuleSpec("forearm", 2, 3, 0.052, np.array([0.0, 0.0, 0.0])),
            CapsuleSpec("wrist_1", 3, 4, 0.045, np.array([0.0, 0.0, 0.0])),
            CapsuleSpec("wrist_2", 4, 5, 0.043, np.array([0.0, 0.0, 0.0])),
            CapsuleSpec("wrist_3", 5, 6, 0.04, np.array([0.0, 0.0, 0.0])),
        ]

    @property
    def count(self) -> int:
        return len(self.specs)

    def states(self, robot_id: int, physics_client_id: int) -> list[CapsuleState]:
        states: list[CapsuleState] = []
        base_pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client_id)
        base = np.asarray(base_pos, dtype=np.float32)

        for spec in self.specs:
            if spec.parent_link < 0:
                start = base
            else:
                start = self._link_world_position(robot_id, spec.parent_link, physics_client_id)
            end = self._link_world_position(robot_id, spec.child_link, physics_client_id) + spec.child_offset
            states.append(CapsuleState(start=start, end=end, radius=spec.radius, name=spec.name))
        return states

    @staticmethod
    def _link_world_position(robot_id: int, link_id: int, physics_client_id: int) -> np.ndarray:
        if link_id < 0:
            pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client_id)
            return np.asarray(pos, dtype=np.float32)
        state = p.getLinkState(
            robot_id,
            link_id,
            computeForwardKinematics=True,
            physicsClientId=physics_client_id,
        )
        return np.asarray(state[4], dtype=np.float32)
