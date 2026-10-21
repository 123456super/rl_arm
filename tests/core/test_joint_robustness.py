"""Check simultaneous physical and measurement perturbations stay independent."""
import copy
from pathlib import Path

import numpy as np

from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def test_joint_perturbations_share_physics_but_use_noisy_observations():
    config = load_config(ROOT/'configs/experiments/paper_full.yaml')
    config['thesis']['precision_control']['enabled'] = False
    config['thesis']['evaluation_mechanical_disturbance'] = {
        'force_n': 20, 'interval_s': .25, 'seed': 74001,
    }
    noisy = copy.deepcopy(config)
    noisy['thesis']['evaluation_sensor_noise'] = {
        'position_std_m': .00075, 'orientation_std_rad': 0, 'seed': 73001,
    }
    force_only, joint = ThesisHomotopyEnv(config), ThesisHomotopyEnv(noisy)
    try:
        a, a_info = force_only.reset(seed=61001)
        b, b_info = joint.reset(seed=61001)
        np.testing.assert_array_equal(a_info['goal_position'], b_info['goal_position'])
        np.testing.assert_array_equal(a_info['ee_position'], b_info['ee_position'])
        assert not np.array_equal(a, b)
        for _ in range(7):
            a = force_only.step(np.full(6, .1))
            b = joint.step(np.full(6, .1))
            # Disable precision scaling only in this interface test: identical
            # commands must produce identical physics and true-state rewards.
            np.testing.assert_array_equal(force_only.robot.joint_state_exact()[0],
                                          joint.robot.joint_state_exact()[0])
            assert a[1:5] == b[1:5]
            np.testing.assert_array_equal(a[5]['external_force_world_n'], b[5]['external_force_world_n'])
            np.testing.assert_array_equal(a[5]['ee_position'], b[5]['ee_position'])
            np.testing.assert_allclose(
                b[5]['measured_ee_position'] - b[5]['ee_position'],
                b[5]['position_sensor_noise_m'], atol=3e-8,
            )
            assert np.linalg.norm(b[5]['position_sensor_noise_m']) > 0
            assert not np.array_equal(a[0], b[0])
    finally:
        force_only.close()
        joint.close()
