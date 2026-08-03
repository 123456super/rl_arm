import numpy as np
import pybullet as p

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config


def test_predictive_link_velocities_and_bound_use_linear_units() -> None:
    config = load_config("configs/experiments/p2_safety_filter_dev.yaml")
    env = UR5DynamicObstacleEnv(config, method="ldrc_fixed")
    try:
        env.reset(seed=11)
        joint_positions, _ = env._joint_state()
        joint_velocities = np.full(env.joint_count, 0.1, dtype=np.float64)
        for joint_id, position, velocity in zip(
            env.joint_ids,
            joint_positions,
            joint_velocities,
            strict=True,
        ):
            p.resetJointState(
                env.robot_id,
                joint_id,
                float(position),
                targetVelocity=float(velocity),
                physicsClientId=env.physics_client_id,
            )

        predictive_risk = env._compute_predictive_risk()
        speed_norms_mps = np.linalg.norm(predictive_risk.link_velocities_mps, axis=1)

        assert np.max(speed_norms_mps) > 0.0
        assert np.max(speed_norms_mps) <= predictive_risk.max_link_speed_bound_mps + 1.0e-9
        assert predictive_risk.max_link_speed_bound_mps != env.action_scale
    finally:
        env.close()
