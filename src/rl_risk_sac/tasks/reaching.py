from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_risk_sac.utils.risk import LinkRisk


@dataclass(frozen=True)
class ReachingObservationBuilder:
    action_scale: float
    distance_clip: tuple[float, float]
    v_max: float
    ttc_max: float

    @staticmethod
    def dimension(joint_count: int, link_count: int) -> int:
        # q, qdot, previous command, goal position/velocity errors,
        # and distance/direction/approach/TTC/risk for every link.
        return joint_count * 3 + link_count * 7 + 7

    def build(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        goal_error: np.ndarray,
        goal_velocity_error: np.ndarray,
        risk: LinkRisk,
        previous_command: np.ndarray,
        beta: float,
    ) -> np.ndarray:
        return np.concatenate(
            [
                q / np.pi,
                q_dot / max(self.action_scale, 1e-6),
                goal_error,
                goal_velocity_error,
                np.clip(risk.distances, *self.distance_clip),
                risk.directions.reshape(-1),
                np.clip(risk.approach_velocities / max(self.v_max, 1e-6), 0.0, 1.0),
                risk.ttc / max(self.ttc_max, 1e-6),
                risk.risks,
                previous_command / max(self.action_scale, 1e-6),
                np.array([beta], dtype=np.float32),
            ]
        ).astype(np.float32)


@dataclass(frozen=True)
class ReachingObjective:
    reward_config: dict[str, Any]
    cost_config: dict[str, Any]
    safe_distance: float

    def reward(
        self,
        goal_error_norm: float,
        previous_goal_error_norm: float,
        command: np.ndarray,
        previous_command: np.ndarray,
        success: bool,
        collision: bool,
    ) -> float:
        progress = previous_goal_error_norm - goal_error_norm
        smooth = float(np.sum(np.square(command - previous_command)))
        value = (
            -float(self.reward_config["w_position"]) * goal_error_norm**2
            + float(self.reward_config["w_progress"]) * progress
            - float(self.reward_config["w_smooth"]) * smooth
        )
        if success:
            value += float(self.reward_config["success_bonus"])
        if collision:
            value -= float(self.reward_config["collision_penalty"])
        return float(value)

    def cost(self, risk: LinkRisk, collision: bool) -> float:
        violation = float(risk.d_min < self.safe_distance)
        return float(
            float(self.cost_config["k_risk"]) * risk.risk_global
            + float(self.cost_config["k_violation"]) * violation
            + float(self.cost_config["k_collision"]) * float(collision)
        )
