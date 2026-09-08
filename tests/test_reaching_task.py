from __future__ import annotations

import numpy as np

from rl_risk_sac.tasks import ReachingObservationBuilder, ReachingObjective
from rl_risk_sac.utils.risk import LinkRisk


def test_observation_builder_has_stable_schema_dimension() -> None:
    builder = ReachingObservationBuilder(0.7, (-1.0, 1.5), 0.7, 3.0)
    risk = LinkRisk(
        closest_points=np.zeros((2, 3), dtype=np.float32),
        distances=np.ones(2, dtype=np.float32),
        directions=np.zeros((2, 3), dtype=np.float32),
        link_velocities=np.zeros((2, 3), dtype=np.float32),
        approach_velocities=np.zeros(2, dtype=np.float32),
        ttc=np.full(2, 3.0, dtype=np.float32),
        risks=np.zeros(2, dtype=np.float32),
        risk_global=0.0,
        d_min=1.0,
        closest_link=0,
    )
    observation = builder.build(
        q=np.zeros(6, dtype=np.float32),
        q_dot=np.zeros(6, dtype=np.float32),
        goal_error=np.zeros(3, dtype=np.float32),
        goal_velocity_error=np.zeros(3, dtype=np.float32),
        risk=risk,
        previous_command=np.zeros(6, dtype=np.float32),
        beta=0.35,
    )

    assert observation.shape == (builder.dimension(6, 2),)


def test_reaching_objective_keeps_reward_and_cost_separate() -> None:
    objective = ReachingObjective(
        reward_config={
            "w_position": 2.0,
            "w_progress": 18.0,
            "w_smooth": 0.04,
            "success_bonus": 20.0,
            "collision_penalty": 2.0,
        },
        cost_config={"k_risk": 1.0, "k_violation": 3.0, "k_collision": 8.0},
        safe_distance=0.12,
    )
    risk = LinkRisk(
        closest_points=np.zeros((1, 3), dtype=np.float32),
        distances=np.asarray([0.1], dtype=np.float32),
        directions=np.zeros((1, 3), dtype=np.float32),
        link_velocities=np.zeros((1, 3), dtype=np.float32),
        approach_velocities=np.zeros(1, dtype=np.float32),
        ttc=np.zeros(1, dtype=np.float32),
        risks=np.asarray([0.5], dtype=np.float32),
        risk_global=0.5,
        d_min=0.1,
        closest_link=0,
    )

    assert objective.cost(risk, collision=False) == 3.5
    assert objective.reward(0.1, 0.2, np.zeros(1), np.zeros(1), True, False) > 20.0
