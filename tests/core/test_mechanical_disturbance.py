import copy
from pathlib import Path

import numpy as np
import pytest

from rl_risk_sac.envs.mechanical_disturbance import MechanicalDisturbance
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.envs.thesis_homotopy_env_reduced_observation import ThesisHomotopyEnv as ReducedEnv
from rl_risk_sac.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def test_force_is_fixed_norm_piecewise_constant_paired_and_isotropic():
    low = MechanicalDisturbance({'force_n': 5}, .05)
    high = MechanicalDisturbance({'force_n': 20}, .05)
    low.reset(41001)
    high.reset(41001)
    samples = np.stack([low.force(5 * i) for i in range(3000)])
    np.testing.assert_allclose(np.linalg.norm(samples, axis=1), 5)
    np.testing.assert_allclose(samples.mean(axis=0), 0, atol=.15)
    np.testing.assert_allclose((samples ** 2).mean(axis=0), 25 / 3, rtol=.05)
    np.testing.assert_array_equal(low.force(11), low.force(14))
    assert not np.array_equal(low.force(14), low.force(15))
    np.testing.assert_allclose(high.force(12), 4 * low.force(12))
    low.reset(41001)
    np.testing.assert_array_equal(low.force(0), samples[0])


@pytest.mark.parametrize('env_class,name', [
    (ThesisHomotopyEnv, 'paper_full'), (ReducedEnv, 'paper_kp_sac'),
])
def test_force_changes_physics_but_not_initial_task_or_observation(env_class, name):
    cfg = load_config(ROOT / f'configs/experiments/{name}.yaml')
    zero_cfg = copy.deepcopy(cfg)
    zero_cfg['thesis']['evaluation_mechanical_disturbance'] = {'force_n': 0}
    noisy_cfg = copy.deepcopy(cfg)
    noisy_cfg['thesis']['evaluation_mechanical_disturbance'] = {'force_n': 20}
    clean, zero, forced = [env_class(c) for c in (cfg, zero_cfg, noisy_cfg)]
    try:
        observations = [env.reset(seed=41001)[0] for env in (clean, zero, forced)]
        np.testing.assert_array_equal(observations[0], observations[1])
        np.testing.assert_array_equal(observations[0], observations[2])
        for action in (np.zeros(6), np.full(6, .1), np.full(6, -.1)):
            state = copy.deepcopy(forced.rng.bit_generator.state)
            forced.mechanical_disturbance.force(forced.step_count)
            assert state == forced.rng.bit_generator.state
            a, b, c = [env.step(action) for env in (clean, zero, forced)]
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1:5] == b[1:5]
            np.testing.assert_allclose(np.linalg.norm(c[5]['external_force_world_n']), 20)
        q_clean, _ = clean.robot.joint_state_exact()
        q_forced, _ = forced.robot.joint_state_exact()
        assert np.linalg.norm(q_clean - q_forced) > 1e-9
    finally:
        for env in (clean, zero, forced):
            env.close()


@pytest.mark.parametrize('config', [
    {'force_n': -1}, {'force_n': float('nan')}, {'force_n': float('inf')},
    {'interval_s': 0}, {'interval_s': .075}, {'seed': -1},
])
def test_invalid_disturbance_rejected(config):
    with pytest.raises(ValueError):
        MechanicalDisturbance(config, .05)


def test_force_reapplied_each_physics_substep_at_current_world_tool(monkeypatch):
    import pybullet as p
    cfg = load_config(ROOT / 'configs/experiments/paper_full.yaml')
    cfg['thesis']['evaluation_mechanical_disturbance'] = {'force_n': 10}
    env = ThesisHomotopyEnv(cfg)
    calls = []
    original = p.applyExternalForce

    def record(body, link, force, position, frame, **kwargs):
        actual = p.getLinkState(body, link, computeForwardKinematics=True, **kwargs)[4]
        np.testing.assert_allclose(position, actual, atol=1e-12)
        assert frame == p.WORLD_FRAME
        calls.append(np.array(force))
        return original(body, link, force, position, frame, **kwargs)

    monkeypatch.setattr(p, 'applyExternalForce', record)
    try:
        env.reset(seed=41001)
        _, _, _, _, _, info = env.step(np.zeros(6))
        assert len(calls) == 12
        np.testing.assert_allclose(calls, np.tile(info['external_force_world_n'], (12, 1)))
        np.testing.assert_array_equal(info['measured_ee_position'], info['ee_position'])
    finally:
        env.close()
