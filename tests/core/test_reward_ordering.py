"""Regression checks for the restored dense reward and online/replay parity."""
import numpy as np
import pytest

from rl_risk_sac.utils.config import load_config
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.thesis_reaching import homotopy_reward
from rl_risk_sac.algorithms.homotopy_replay import _vectorized_replay_rewards
from scripts.calibration.audit_reward_ordering import audit, physical_audit


@pytest.fixture
def parameters():
    env = ThesisHomotopyEnv(load_config("configs/base/thesis_common.yaml"))
    try:
        return dict(env.reward_parameters)
    finally:
        env.close()


def evaluate(parameters, **kwargs):
    values = dict(rho_position=.01, next_rho_position=.01,
                  rho_orientation=.04, next_rho_orientation=.04,
                  smooth_velocity=0., velocity_magnitude=0., orientation_scale=1.,
                  task_reached=False, hard_failure=False, obstacle_collision=False,
                  risk_max=0., distance_min=.2, xi=1., keypoint_distance=.02,
                  next_keypoint_distance=.02, keypoint_tracking_quality=1., gamma=.99)
    values.update(kwargs)
    return homotopy_reward(**values, **parameters)


def test_dense_reward_components_are_restored(parameters):
    _, fields = evaluate(parameters, next_keypoint_distance=.019)
    assert fields["keypoint_tracking_reward"] == pytest.approx(.05)
    assert fields["keypoint_progress_reward"] == pytest.approx(.02)
    assert fields["keypoint_precision_reward"] > 0
    assert fields["hold_reward"] == pytest.approx(.15)


def test_timeout_has_no_new_partial_credit(parameters):
    _, timeout = evaluate(parameters, timeout=True)
    assert timeout["timeout_penalty"] == pytest.approx(2.)
    assert "timeout_progress_reward" not in timeout
    assert "task_step_cost" not in timeout
    assert "task_motion_cost" not in timeout


@pytest.mark.parametrize("cause", ["task_reached", "hard_failure", "terminal_obstacle_collision", "timeout"])
def test_terminal_bottleneck_shaping_is_zero_as_before(parameters, cause):
    _, fields = evaluate(parameters, **{cause: True})
    assert fields["joint_bottleneck_shaping_reward"] == 0


def test_restored_reward_scalar_vector_parity(parameters):
    rng = np.random.default_rng(101)
    raw = np.zeros((100, 29))
    raw[:, :4] = rng.uniform(0, .2, (100, 4))
    raw[:, 4] = rng.uniform(0, 2, 100)
    raw[:, 5:8] = rng.integers(0, 2, (100, 3))
    raw[:, 11] = rng.uniform(0, 1, 100)
    raw[:, 12] = rng.uniform(-.02, .2, 100)
    raw[:, 15] = rng.uniform(0, 2, 100)
    raw[:, 17] = rng.uniform(0, 1, 100)
    raw[:, 18] = rng.uniform(-.01, .25, 100)
    raw[:, 22] = rng.integers(0, 2, 100)
    raw[:, 24] = .01
    raw[:, 25] = .03
    raw[:, 26:28] = rng.uniform(0, .2, (100, 2))
    raw[:, 28] = rng.uniform(0, 1, 100)
    dones = ((raw[:, 5:8].sum(1) + raw[:, 22]) > 0).astype(float)
    vector, _ = _vectorized_replay_rewards(raw, dones, np.ones(100), .37, .99, parameters)
    scalar = []
    for row in raw:
        p = dict(parameters, joint_position_tolerance=.01, joint_orientation_tolerance=.03)
        r, _ = evaluate(p, rho_position=row[0], next_rho_position=row[1],
                        rho_orientation=row[2], next_rho_orientation=row[3], smooth_velocity=row[4],
                        task_reached=bool(row[5]), hard_failure=bool(row[6]),
                        obstacle_collision=bool(row[7]),
                        terminal_obstacle_collision=bool(row[7] and not row[5] and not row[6]),
                        risk_max=row[11], distance_min=row[12], velocity_magnitude=row[15],
                        self_risk_max=row[17], self_distance_min=row[18], timeout=bool(row[22]),
                        keypoint_distance=row[26], next_keypoint_distance=row[27],
                        keypoint_tracking_quality=row[28], lambda_self=.37)
        scalar.append(r)
    np.testing.assert_allclose(scalar, vector, atol=1e-12, rtol=1e-12)


def test_original_reward_audit_baseline_is_restored():
    result = audit(load_config("configs/base/thesis_common.yaml"))
    assert result["gamma"] == .99
    assert result["reward_scheme"] == "legacy_dense"
    for level, values in result["levels"].items():
        # The original ranking limitations remain; rollback is not a new fix.
        assert any(not c["passed"] for c in values["comparisons"]), level


def test_real_same_state_original_returns_are_restored():
    config = load_config("configs/base/thesis_common.yaml")
    # Isolate the restored formula from the later success_bonus=50 experiment.
    config["thesis"]["reward"]["success_bonus"] = 20.
    rows = physical_audit(config)["results"]
    assert rows["wait"]["timeout"] and not rows["wait"]["success"]
    assert rows["retreat_then_success"]["success"] and not rows["retreat_then_success"]["collision"]
    assert rows["prompt_success"]["discounted_return"] == pytest.approx(21.973, abs=.01)
    assert rows["wait"]["discounted_return"] == pytest.approx(4.701, abs=.01)
    assert rows["retreat_then_success"]["discounted_return"] == pytest.approx(2.041, abs=.01)
