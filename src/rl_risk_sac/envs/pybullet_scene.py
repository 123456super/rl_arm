from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pybullet as p

from rl_risk_sac.scene import ObstacleState
from rl_risk_sac.utils.runtime_config import ObstacleRuntimeConfig, VisualRuntimeConfig


class PyBulletScene:
    """Own PyBullet bodies used for the floor, obstacles, and goal marker."""

    def __init__(
        self,
        physics_client_id: int,
        obstacle_config: ObstacleRuntimeConfig,
        visual_config: VisualRuntimeConfig,
    ) -> None:
        self.physics_client_id = physics_client_id
        self.obstacle_config = obstacle_config
        self.visual_config = visual_config
        self.obstacle_ids: list[int] = []
        self.goal_marker_id: int | None = None

    def create_floor(self) -> None:
        self.obstacle_ids = []
        self.goal_marker_id = None
        self._create_floor()

    def populate(self, obstacles: Sequence[ObstacleState], goal: np.ndarray) -> None:
        self.obstacle_ids = [self._create_obstacle(state.center) for state in obstacles if state.enabled]
        self.goal_marker_id = self._create_goal_marker(goal)

    def sync_obstacles(self, obstacles: Sequence[ObstacleState]) -> None:
        active_states = [state for state in obstacles if state.enabled]
        for obstacle_id, state in zip(self.obstacle_ids, active_states):
            p.resetBasePositionAndOrientation(
                obstacle_id,
                state.center.tolist(),
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.physics_client_id,
            )

    def sync_goal(self, goal: np.ndarray) -> None:
        if self.goal_marker_id is None:
            return
        p.resetBasePositionAndOrientation(
            self.goal_marker_id,
            goal.tolist(),
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.physics_client_id,
        )

    def has_robot_contact(self, robot_id: int) -> bool:
        return any(
            p.getContactPoints(
                bodyA=robot_id,
                bodyB=obstacle_id,
                physicsClientId=self.physics_client_id,
            )
            for obstacle_id in self.obstacle_ids
        )

    def _create_floor(self) -> None:
        collision = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.physics_client_id)
        visual = p.createVisualShape(
            p.GEOM_PLANE,
            rgbaColor=self.visual_config.floor_rgba,
            physicsClientId=self.physics_client_id,
        )
        p.createMultiBody(
            0,
            collision,
            visual,
            self.visual_config.floor_position,
            physicsClientId=self.physics_client_id,
        )

    def _create_obstacle(self, center: np.ndarray) -> int:
        radius = self.obstacle_config.radius
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.physics_client_id)
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=self.visual_config.obstacle_rgba,
            physicsClientId=self.physics_client_id,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=center.tolist(),
            physicsClientId=self.physics_client_id,
        )

    def _create_goal_marker(self, goal: np.ndarray) -> int:
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.visual_config.goal_marker_radius,
            rgbaColor=self.visual_config.goal_rgba,
            physicsClientId=self.physics_client_id,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=goal.tolist(),
            physicsClientId=self.physics_client_id,
        )
