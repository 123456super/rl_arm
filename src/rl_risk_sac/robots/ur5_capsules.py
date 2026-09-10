from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pybullet as p

from rl_risk_sac.utils.runtime_config import CapsuleSpec


@dataclass
class CapsuleState:
    """World-space capsule used by risk detection at one simulator step."""

    start: np.ndarray
    end: np.ndarray
    radius: float
    name: str


class UR5CapsuleModel:
    """Capsule approximation for the main links of the bundled UR5 arm.

    PyBullet 的 URDF 几何很复杂；避障风险只需要每段连杆的大致占据体，
    所以这里用“父 link 世界坐标 -> 子 link 世界坐标 + 半径”的胶囊近似。
    """

    def __init__(self, specs: Sequence[CapsuleSpec]) -> None:
        self.specs = list(specs)
        self.link_name_to_id: dict[str, int] = {}

    @property
    def count(self) -> int:
        return len(self.specs)

    def resolve_link_names(self, link_name_to_id: dict[str, int]) -> None:
        """Cache the mapping from URDF link names to PyBullet link ids."""
        self.link_name_to_id = dict(link_name_to_id)

    def states(self, robot_id: int, physics_client_id: int) -> list[CapsuleState]:
        """Read current PyBullet link poses and return all capsule states."""
        states: list[CapsuleState] = []
        base_pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client_id)
        base = np.asarray(base_pos, dtype=np.float32)

        for spec in self.specs:
            # base 在 PyBullet 中没有普通 link id，这里用 -1 表示机器人基座。
            parent_link = self._resolve_link_id(spec.parent_link_name)
            child_link = self._resolve_link_id(spec.child_link_name)
            start = self._link_world_point(
                robot_id,
                parent_link,
                spec.parent_offset,
                physics_client_id,
                base_position=base,
            )
            end = self._link_world_point(
                robot_id,
                child_link,
                spec.child_offset,
                physics_client_id,
                base_position=base,
            )
            if float(np.linalg.norm(end - start)) <= 1e-6 and not spec.allow_degenerate:
                raise ValueError(
                    f"Capsule {spec.name!r} is degenerate; set allow_degenerate=true "
                    "only for an intentional sphere approximation"
                )
            states.append(CapsuleState(start=start, end=end, radius=spec.radius, name=spec.name))
        return states

    def _resolve_link_id(self, link_name: str) -> int:
        if link_name not in self.link_name_to_id:
            available = ", ".join(sorted(self.link_name_to_id))
            raise KeyError(f"Unknown link name {link_name!r}; available links: {available}")
        return self.link_name_to_id[link_name]

    @staticmethod
    def _link_world_point(
        robot_id: int,
        link_id: int,
        local_offset: np.ndarray,
        physics_client_id: int,
        *,
        base_position: np.ndarray,
    ) -> np.ndarray:
        if link_id < 0:
            _, orientation = p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client_id)
            position = base_position
        else:
            state = p.getLinkState(
                robot_id,
                link_id,
                computeForwardKinematics=True,
                physicsClientId=physics_client_id,
            )
            position = np.asarray(state[4], dtype=np.float32)
            orientation = state[5]
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation), dtype=np.float32).reshape(3, 3)
        return position + rotation @ np.asarray(local_offset, dtype=np.float32)
