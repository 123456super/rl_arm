from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
import torch

from rl_risk_sac.envs.obstacle_scene import MovingObstacle, create_body
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.local_avoidance import arm_vector, contract_digest, validate_local_config
from rl_risk_sac.tasks.local_path_verification import (
    LocalPathVerifier, validate_path_certificate, verification_settings,
)
from rl_risk_sac.tasks.local_samples import simulation_config
from rl_risk_sac.utils.config import load_config
from scripts.core.train_thesis_homotopy import apply_stage_profile


@pytest.fixture
def setup(tmp_path):
    config = load_config("configs/experiments/paper_local_avoidance_geometry.yaml")
    apply_stage_profile(config, "s1")
    path = tmp_path / "scale.pt"
    torch.save(dict(format_version=1, accepted=True, feature_scale=np.ones(37, np.float32)), path)
    config["thesis"]["obstacles"]["predictor_checkpoint"] = str(path)
    env = ThesisHomotopyEnv(simulation_config(config), compute_external_features=False)
    env.configure_episode("none", xi=1., strict=True, terminate_on_success=False)
    env.reset(seed=1234)
    state = copy.deepcopy(env.episode_state_dict())
    entry = arm_vector(env)
    goal = state["q"].copy()
    goal[0] += .15
    env._set_joint_state(goal)
    delta = arm_vector(env) - entry
    state["goal_position"], state["goal_quaternion"] = env.robot.end_effector_pose()
    state["goal_joint_positions"] = goal.copy()
    obj = MovingObstacle(0, "sphere", [.04], "small", [3., 3., 3.], [0., 0., 0., 1.],
                         [1., 0., 0.], "constant", .1, 1., 0., -1., 1.)
    state["obstacles"] = [obj.metadata()]
    sample = dict(sample_id="real-kinematic-test", state=state, entry_arm=entry,
                  direction=delta / np.linalg.norm(delta), length=float(np.linalg.norm(delta) / 2))
    reaching = SimpleNamespace(select_action=lambda obs, deterministic: np.zeros(6))
    verifier = LocalPathVerifier(config, reaching)
    try:
        yield config, env, sample, goal, verifier
    finally:
        verifier.close()
        env.close()


def test_real_witness_checks_edges_and_handoff_without_physics_or_live_mutation(setup, monkeypatch):
    config, env, sample, goal, verifier = setup
    before = copy.deepcopy(env.episode_state_dict())
    monkeypatch.setattr(p, "stepSimulation", lambda **kw: pytest.fail("geometric witness must not step physics"))
    record = verifier.verify(sample, goal)
    assert record["status"] == "verified"
    assert record["method"] == "direct"
    assert record["verification_checks"] >= 31
    assert record["global_path_verified"] and record["local_handoff_passed"]
    assert record["local_exit_progress"] >= 1.
    assert record["local_exit_external_m"] >= .06
    assert verifier.query.physics_step_count == sample["state"]["physics_step_count"]
    validate_path_certificate(dict(sample, path_verification=record), config, task_digest=verifier.task_digest)
    after = env.episode_state_dict()
    for key in ("q", "qdot", "goal_position", "goal_quaternion"):
        np.testing.assert_array_equal(before[key], after[key])
    for key in ("rng_state", "step_count", "physics_step_count"):
        assert before[key] == after[key]


def test_global_route_does_not_imply_geometric_local_exit(setup, monkeypatch):
    _, _, sample, goal, verifier = setup
    sample["direction"] *= -1
    called = []
    def fallback(*args):
        called.append(True)
        return dict(args[-1], status="local_exit_unconfirmed")
    monkeypatch.setattr(verifier, "_search_local_exit", fallback)
    record = verifier.verify(sample, goal)
    assert record["global_path_verified"]
    assert record["status"] == "local_exit_unconfirmed"
    assert record["exit_checks"] == 0
    assert called


def test_plane_crossing_alone_does_not_pass_failed_handoff(setup, monkeypatch):
    _, _, sample, goal, verifier = setup
    monkeypatch.setattr(verifier.checker, "handoff", lambda *args: False)
    record = verifier.verify(sample, goal)
    assert record["status"] == "local_exit_unconfirmed"
    assert record["global_path_verified"] and record["exit_checks"] > 0


def test_exit_region_can_be_reached_on_a_different_branch_than_global_path(setup, monkeypatch):
    config, _, sample, goal, verifier = setup
    sample["direction"] *= -1
    target = sample["state"]["q"].copy()
    target[0] -= .15
    from rl_risk_sac.tasks.static_scene_planning import rrt_goal_region
    def directed(*args, **kwargs):
        kwargs["sample_target"] = lambda: target
        return rrt_goal_region(*args, **kwargs)
    monkeypatch.setattr("rl_risk_sac.tasks.local_path_verification.rrt_goal_region", directed)
    record = verifier.verify(sample, goal)
    assert record["status"] == "verified"
    assert record["global_path_max_progress"] < 1.
    assert record["local_search_checks"] > 0
    assert record["local_exit_progress"] >= 1.
    validate_path_certificate(dict(sample, path_verification=record), config, task_digest=verifier.task_digest)


def test_whole_arm_checks_second_obstacle_and_limits(setup):
    _, _, sample, _, verifier = setup
    verifier.query.restore_episode_state(copy.deepcopy(sample["state"]))
    q = sample["state"]["q"]
    assert verifier._valid(q)
    position = verifier.query.robot.end_effector_pose()[0]
    obj = MovingObstacle(1, "sphere", [.1], "small", position.tolist(), [0., 0., 0., 1.],
                         [1., 0., 0.], "constant", .1, 1., 0., -1., 1.)
    create_body(obj, verifier.query.client_id)
    verifier.query.obstacles.append(obj)
    assert not verifier._valid(q)
    assert not verifier._valid(np.full(6, np.nan))
    assert not verifier._valid(verifier.query.robot.joint_upper_limits + .1)


def test_fine_recheck_rejects_an_interior_state_missed_by_coarse_route(setup, monkeypatch):
    _, _, sample, goal, verifier = setup
    def coarse(start, end, *args, statistics, **kwargs):
        statistics.update(checks=2, method="rrt_connect")
        return [start.copy(), end.copy()]
    monkeypatch.setattr("rl_risk_sac.tasks.local_path_verification.rrt_connect", coarse)
    original = verifier._valid
    midpoint = (sample["state"]["q"] + goal) / 2
    monkeypatch.setattr(verifier, "_valid", lambda q: original(q) and abs(q[0] - midpoint[0]) > .003)
    record = verifier.verify(sample, goal)
    assert record["status"] == "fine_check_rejected"
    assert not record["global_path_verified"]


def test_validity_rejects_real_nonadjacent_link_self_collision(setup):
    _, _, sample, _, verifier = setup
    env = verifier.query
    env.restore_episode_state(copy.deepcopy(sample["state"]))
    rng = np.random.default_rng(77)
    for _ in range(200):
        q = rng.uniform(env.robot.joint_lower_limits, env.robot.joint_upper_limits)
        env._set_joint_state(q)
        if env._self_geometry().distance_min < env.d_self_safe:
            assert not verifier._valid(q)
            break
    else:
        pytest.fail("expected a self-colliding configuration in the deterministic probe")


def test_search_and_verification_exhaustion_stay_unconfirmed(setup, monkeypatch):
    _, _, sample, goal, verifier = setup
    def exhausted(*args, statistics, **kwargs):
        statistics.update(checks=4000, method="rrt_connect")
        return None
    with monkeypatch.context() as patch:
        patch.setattr("rl_risk_sac.tasks.local_path_verification.rrt_connect", exhausted)
        result = verifier.verify(sample, goal)
    assert result["status"] == "search_unconfirmed"
    assert not result["global_path_verified"] and result["search_checks"] == 4000
    verifier.settings["max_verification_checks"] = 2
    result = verifier.verify(sample, goal)
    assert result["status"] == "verification_budget_unconfirmed"
    assert result["verification_checks"] == 2 and not result["global_path_verified"]


def test_witness_bound_to_scene_reference_task_and_record(setup):
    config, _, sample, goal, verifier = setup
    valid = dict(sample, path_verification=verifier.verify(sample, goal))
    validate_path_certificate(valid, config, task_digest=verifier.task_digest)
    for kind in ("obstacle", "reference", "route"):
        changed = copy.deepcopy(valid)
        if kind == "obstacle":
            changed["state"]["obstacles"][0]["initial_position"][0] += .1
        elif kind == "reference":
            changed["length"] += .1
        else:
            changed["path_verification"]["waypoints"][0][0] += .1
        with pytest.raises(ValueError, match="witness differs"):
            validate_path_certificate(changed, config, task_digest=verifier.task_digest)
    with pytest.raises(ValueError, match="witness differs"):
        validate_path_certificate(valid, config, task_digest="different-task")
    with pytest.raises(ValueError, match="audit_local"):
        validate_path_certificate(sample, config, task_digest=verifier.task_digest)


@pytest.mark.parametrize("key,value", [("rrt_max_checks", 1), ("rrt_iterations", 1.5),
                                      ("verification_resolution_rad", .03), ("max_exit_checks", 0)])
def test_invalid_search_settings_fail_early(setup, key, value):
    config, *_ = setup
    config["local_path_verification"][key] = value
    with pytest.raises(ValueError, match="path_verification|path verification"):
        validate_local_config(config)


def test_offline_budgets_do_not_change_policy_or_replay_contract(setup):
    config, *_ = setup
    digest = contract_digest(config)
    config["local_path_verification"]["rrt_max_checks"] = 100
    assert verification_settings(config)["rrt_max_checks"] == 100
    assert contract_digest(config) == digest
