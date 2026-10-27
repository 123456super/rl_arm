"""Formula, termination, replay parity and physical plumbing, not performance claims."""
import copy

import numpy as np
import pytest

from rl_risk_sac.tasks.goal_obstacle_coupling import (
    COUPLING_KEYS, coupling_parameters, coupling_potential, coupling_shaping,
)
from rl_risk_sac.tasks.thesis_reaching import homotopy_reward
from rl_risk_sac.tasks.obstacle_observation import obstacle_set
from rl_risk_sac.algorithms.homotopy_replay import _vectorized_replay_rewards, HomotopyReplayBuffer
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config
from scripts.core.train_thesis_homotopy import apply_stage_profile, reconcile_coupling_resume


@pytest.fixture
def config():
    c = load_config("configs/base/thesis_common.yaml")
    # The retired experiment is no longer a runnable config. Keep explicit
    # regression coverage for saved reward contracts.
    c["thesis"]["reward"]["goal_obstacle_coupling"] = dict(
        enabled=True, beta=1.0, position_scale_m=.30, orientation_scale_rad=1.0)
    apply_stage_profile(c, "s1")
    return c


def test_formula_and_safe_detour(config):
    p = coupling_parameters(config)
    assert float(coupling_potential(0., 0., .1, True, p)) == 1.
    assert float(coupling_potential(0., 0., .025, True, p)) == 0.
    f, a, b = coupling_shaping(.3, .35, .4, .4, .03, .05, True, True, False, p, .99)
    assert f == pytest.approx(.09469135802469135, abs=1e-12)
    assert b > a
    assert float(coupling_potential(.1, 1., .1, True, p)) < float(coupling_potential(.1, .1, .1, True, p))


@pytest.mark.parametrize("stage", ["s0", "s2", "s3"])
def test_only_s1_enables_the_term(config, stage):
    apply_stage_profile(config, stage)
    assert coupling_parameters(config) == {}


@pytest.mark.parametrize("key,value", [("beta", -1.), ("beta", float("nan")),
                                      ("position_scale_m", 0.), ("orientation_scale_rad", float("inf"))])
def test_invalid_settings_rejected(config, key, value):
    config["thesis"]["reward"]["goal_obstacle_coupling"][key] = value
    with pytest.raises(ValueError, match="invalid"):
        coupling_parameters(config)


def test_no_obstacle_wait_and_terminal(config):
    p = coupling_parameters(config)
    assert float(coupling_potential(.1, .2, .8, False, p)) == 0.
    f, a, b = coupling_shaping(.3, .3, .4, .4, .05, .05, True, True, False, p, .99)
    assert f == pytest.approx(-.01 * a)
    assert f < 0
    f, a, b = coupling_shaping(.3, 0., .4, 0., .05, .10, True, True, True, p, .99)
    assert b == 0.
    assert f == -a


@pytest.mark.parametrize("steps", [1, 10, 185, 429, 500])
def test_telescoping_all_lengths_and_no_discounted_cycle_gain(config, steps):
    p = coupling_parameters(config)
    rng = np.random.default_rng(steps)
    pos, ori, dist = rng.random((3, steps + 1))
    dist = .025 + .075 * dist
    done = np.arange(steps) == steps - 1
    f, a, b = coupling_shaping(pos[:-1], pos[1:], ori[:-1], ori[1:], dist[:-1], dist[1:],
                             True, True, done, p, .99)
    assert np.dot(.99 ** np.arange(steps), f) == pytest.approx(-a[0], abs=1e-12)


@pytest.mark.parametrize("cause", ["task_reached", "hard_failure", "terminal_obstacle_collision", "timeout"])
def test_scalar_terminal_is_negative_current_potential(config, cause):
    values = dict(rho_position=.3, next_rho_position=.1, rho_orientation=.4,
                  next_rho_orientation=.2, smooth_velocity=0., velocity_magnitude=0.,
                  orientation_scale=0., task_reached=False, hard_failure=False,
                  obstacle_collision=False, risk_max=.1, distance_min=.08, xi=1.,
                  previous_external_distance=.05, obstacle_present=True)
    values[cause] = True
    _, fields = homotopy_reward(**values, **coupling_parameters(config))
    assert fields["goal_obstacle_coupling_reward"] == -fields["goal_obstacle_potential"]
    assert fields["next_goal_obstacle_potential"] == 0.


def test_vector_scalar_random_parity_and_disabled_exactness(config):
    rng = np.random.default_rng(114)
    n = 80
    raw = np.zeros((n, 29))
    raw[:, :4] = rng.random((n, 4))
    raw[:, 24:26] = [.1, .3]
    raw[:, 26:29] = rng.random((n, 3))
    raw[:, 12] = rng.uniform(-.02, .2, n)
    raw[:, 18] = .25
    for i in range(n):
        if i % 5 < 3:
            raw[i, 5 + i % 5] = 1
        elif i % 5 == 3:
            raw[i, 22] = 1
    done = (raw[:, 5:8].sum(1) + raw[:, 22]) > 0
    sets, nxt = [], []
    for i in range(n):
        a = np.zeros((int(i % 3 != 0), 37))
        b = np.zeros_like(a)
        a[:, :4] = rng.uniform(-.02, .2)
        b[:, :4] = raw[i, 12]
        sets.append(a)
        nxt.append(b)
    p = dict(coupling_parameters(config), joint_position_tolerance=.1, joint_orientation_tolerance=.3)
    v, _ = _vectorized_replay_rewards(raw, done, np.ones(n), .2, .99, p,
                                     obstacle_sets=sets, next_obstacle_sets=nxt)
    for i, r in enumerate(raw):
        scalar, _ = homotopy_reward(
            rho_position=r[0], next_rho_position=r[1], rho_orientation=r[2], next_rho_orientation=r[3],
            smooth_velocity=0., velocity_magnitude=0., orientation_scale=0., task_reached=bool(r[5]),
            hard_failure=bool(r[6]), obstacle_collision=bool(r[7]),
            terminal_obstacle_collision=bool(r[7] and not r[5] and not r[6]), timeout=bool(r[22]),
            risk_max=0., distance_min=r[12], xi=1., self_distance_min=.25, lambda_self=.2,
            keypoint_distance=r[26], next_keypoint_distance=r[27], keypoint_tracking_quality=r[28],
            previous_external_distance=float(sets[i][:, :4].min()) if len(sets[i]) else .8,
            obstacle_present=bool(len(sets[i])), **p)
        assert scalar == pytest.approx(v[i], abs=1e-12)
    baseline, _ = _vectorized_replay_rewards(raw, done, np.ones(n), .2, .99, {})
    disabled, _ = _vectorized_replay_rewards(raw, done, np.ones(n), .2, .99,
                                           {"goal_obstacle_coupling_beta": 0.})
    np.testing.assert_array_equal(disabled, baseline)
    with pytest.raises(ValueError, match="raw obstacle"):
        _vectorized_replay_rewards(raw, done, np.ones(n), .2, .99, p)


def test_resume_requires_explicit_change_and_rejects_base_reward_change(config):
    from types import SimpleNamespace
    old = copy.deepcopy(config)
    old["thesis"]["reward"].pop("goal_obstacle_coupling")
    replay = SimpleNamespace(reward_parameters={})
    with pytest.raises(ValueError, match="allow-coupling"):
        reconcile_coupling_resume(replay, config, old)
    assert reconcile_coupling_resume(replay, config, old, allow_change=True)["changed"]
    assert replay.reward_parameters[COUPLING_KEYS[0]] == 1.
    assert not reconcile_coupling_resume(replay, config, config)["changed"]
    changed = copy.deepcopy(config)
    changed["thesis"]["reward"]["success_bonus"] += 1
    with pytest.raises(ValueError, match="base reward"):
        reconcile_coupling_resume(replay, changed, old, allow_change=True)


def test_physical_step_replay_parity(config):
    config["thesis"]["obstacles"]["path_sampling"]["enabled"] = False
    config["thesis"]["obstacles"]["count_probabilities"] = [0., 1., 0., 0.]
    env = ThesisHomotopyEnv(config)
    try:
        env.configure_episode("none", xi=1., strict=True, goal_scale=1., orientation_scale=0.,
                              position_tolerance=.1, orientation_tolerance=.3, lambda_self=.2)
        obs, _ = env.reset(seed=37)
        env.horizon = 4
        for _ in range(4):
            nxt, reward, _, term, trunc, info = env.step(np.zeros(6))
            raw = HomotopyReplayBuffer.transition_features(info, curriculum_level=0,
                                                          reward_parameters=env.reward_parameters)
            replay, _ = _vectorized_replay_rewards(np.asarray([raw]), [term or trunc], [1.], .2, .99,
                env.reward_parameters, obstacle_sets=[obstacle_set(obs)], next_obstacle_sets=[obstacle_set(nxt)])
            assert reward == pytest.approx(replay[0], abs=2e-6)
            if term or trunc:
                assert info["next_goal_obstacle_potential"] == 0.
                break
            obs = nxt
    finally:
        env.close()
