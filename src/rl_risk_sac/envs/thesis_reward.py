"""Reward and safety-cost calculation for one thesis environment transition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from rl_risk_sac.tasks.reference_reaching import reference_ppo_reward


@dataclass(frozen=True)
class StepReward:
    reward: float
    fields: dict[str, float]
    cost: float
    self_collision_step_penalty: float


def compute_step_reward(
    *,
    reference_error: float,
    obstacle_collision: bool,
    self_collision: bool,
    environment_collision: bool,
    obstacle_clearances: Sequence[float],
    parameters: Mapping[str, float],
    obstacle_risk: float,
    self_risk: float,
    obstacle_distance: float,
    self_distance: float,
    obstacle_safe_distance: float,
    self_safe_distance: float,
) -> StepReward:
    reward, fields = reference_ppo_reward(
        reference_error=reference_error,
        obstacle_collision=obstacle_collision,
        self_collision=self_collision,
        environment_collision=environment_collision,
        obstacle_clearances=tuple(float(value) for value in obstacle_clearances),
        obstacle_threshold=parameters["reference_obstacle_threshold"],
        obstacle_clearance_weight=parameters["reference_obstacle_clearance_weight"],
        obstacle_clearance_power=parameters["reference_obstacle_clearance_power"],
        obstacle_collision_penalty=parameters["reference_obstacle_collision_penalty"],
        self_collision_penalty=parameters["reference_self_collision_penalty"],
        environment_collision_penalty=parameters[
            "reference_environment_collision_penalty"
        ],
    )
    any_collision = obstacle_collision or self_collision or environment_collision
    cost = float(
        obstacle_risk
        + self_risk
        + 3.0 * (obstacle_distance < obstacle_safe_distance)
        + 3.0 * (self_distance < self_safe_distance)
        + 8.0 * any_collision
    )
    return StepReward(
        reward=float(reward),
        fields=fields,
        cost=cost,
        self_collision_step_penalty=-fields["reference_self_collision_penalty"],
    )
