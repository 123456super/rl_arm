import numpy as np
import pybullet as p

from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config


def test_analytic_constraint_jacobians_match_finite_difference() -> None:
    config = load_config("configs/experiments/p2_safety_filter_dev.yaml")
    env = UR5DynamicObstacleEnv(config, method="ldrc_adaptive")
    try:
        env.reset(seed=7)
        predictive_risk = env._compute_predictive_risk()
        analytic_safety, _, analytic_ee = env._analytic_constraint_jacobians(predictive_risk)
        numeric_safety, numeric_ee = _finite_difference_jacobians(env, predictive_risk)

        np.testing.assert_allclose(analytic_safety, numeric_safety, atol=5e-4, rtol=1e-2)
        np.testing.assert_allclose(analytic_ee, numeric_ee, atol=5e-4, rtol=1e-2)
    finally:
        env.close()


def _finite_difference_jacobians(
    env: UR5DynamicObstacleEnv,
    predictive_risk,
) -> tuple[np.ndarray, np.ndarray]:
    joint_positions, _ = env._joint_state()
    ee_position, _ = env._end_effector_state()
    step_rad = 1e-4
    safety_jacobian = np.zeros((env.capsule_model.count, env.joint_count), dtype=np.float64)
    ee_jacobian = np.zeros((3, env.joint_count), dtype=np.float64)
    state_id = p.saveState(physicsClientId=env.physics_client_id)
    try:
        for index in range(env.joint_count):
            perturbed_positions = joint_positions.copy()
            perturbed_positions[index] += step_rad
            for joint_id, position in zip(env.joint_ids, perturbed_positions, strict=True):
                p.resetJointState(
                    env.robot_id,
                    joint_id,
                    float(position),
                    targetVelocity=0.0,
                    physicsClientId=env.physics_client_id,
                )
            perturbed_risk = env._compute_predictive_risk()
            perturbed_ee_position, _ = env._end_effector_state()
            safety_jacobian[:, index] = (
                perturbed_risk.safety_functions_m - predictive_risk.safety_functions_m
            ) / step_rad
            ee_jacobian[:, index] = (perturbed_ee_position - ee_position) / step_rad
            p.restoreState(state_id, physicsClientId=env.physics_client_id)
    finally:
        p.removeState(state_id, physicsClientId=env.physics_client_id)
    return safety_jacobian, ee_jacobian
