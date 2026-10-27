from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
import torch

from rl_risk_sac.algorithms.dual_sac import build_thesis_sac
from rl_risk_sac.algorithms.local_replay import LocalReplayBuffer
from rl_risk_sac.envs.local_avoidance_env import LocalAvoidanceEnv
from rl_risk_sac.envs.obstacle_scene import MovingObstacle
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.robots.ur5_capsules import CapsuleState
from rl_risk_sac.tasks.convex_sweep import ARM_GROUPS, ConvexQuery
from rl_risk_sac.tasks.geometric_passage import (
    GeometricPassageReference, NominalIntent, make_geometric_reference,
    support_plane, validate_geometric_reference,
)
from rl_risk_sac.tasks.local_avoidance import (
    GEOMETRIC_PROTOCOL, PROTOCOL, arm_vector, contract_digest, local_protocol,
    passage_progress, passage_candidate, validate_local_config,
)
from rl_risk_sac.tasks.local_samples import simulation_config
from rl_risk_sac.utils.config import load_config
from scripts.core.train_thesis_homotopy import apply_stage_profile


@pytest.fixture
def config(tmp_path):
    cfg = load_config("configs/experiments/paper_local_avoidance_geometry.yaml")
    apply_stage_profile(cfg, "s1")
    cfg["device"] = "cpu"
    cfg["sac"]["hidden_dims"] = [16, 16]
    path = tmp_path / "scale.pt"
    torch.save({"format_version": 1, "accepted": True, "feature_scale": np.ones(37, np.float32)}, path)
    cfg["thesis"]["obstacles"]["predictor_checkpoint"] = str(path)
    return cfg


def obstacle(identity=0, shape="sphere", size=None, quaternion=None):
    return MovingObstacle(identity, shape, size or [.05], "small", [0., 0., 0.],
                          quaternion or [0., 0., 0., 1.], [1., 0., 0.], "constant", .1, 1., 0., -1., 1.)


def entry_arm():
    return np.tile([[-.2, -.1, 0.], [-.2, .1, 0.]], (4, 1)).reshape(24)


@pytest.mark.parametrize("shape,size,axis,expected", [
    ("sphere", [.05], [1., 0., 0.], .05),
    ("box", [.04, .08, .12], [1., 0., 0.], .04),
    ("cylinder", [.04, .12], [0., 0., 1.], .06),
    ("cylinder", [.04, .12], [1., 0., 0.], .04),
])
def test_shape_support_uses_real_shape_extent(shape, size, axis, expected):
    assert support_plane(obstacle(shape=shape, size=size), np.asarray(axis), .08, .06) == pytest.approx(expected + .14)


def test_rotated_box_plane_uses_orientation():
    obj = obstacle(shape="box", size=[.04, .08, .12], quaternion=list(p.getQuaternionFromEuler([0., 0., np.pi / 2])))
    assert support_plane(obj, np.array([1., 0., 0.]), .08, .06) == pytest.approx(.22)


@pytest.mark.parametrize("fraction", [0., .3, 1.])
def test_projection_progress_is_exact_material_point_plane_and_retreat_cannot_pass(fraction):
    entry = entry_arm()
    sample = make_geometric_reference(entry, obstacle(), "forearm", fraction,
                                       [1., 0., 0.], .081, .06)
    moved = entry.reshape(4, 2, 3).copy()
    # Moving unrelated links cannot harvest the primary-link progress reward.
    moved[0, :, 0] += 2.
    assert passage_progress(moved.reshape(-1), sample) == pytest.approx(0.)
    moved[1, :, 0] -= .1
    assert passage_progress(moved.reshape(-1), sample) < 0
    moved[1, :, 0] = .05 + .081 + .06
    assert passage_progress(moved.reshape(-1), sample) == pytest.approx(1.)
    assert np.linalg.norm(sample["direction"]) == pytest.approx(1.)


def test_bank_reference_cannot_hide_different_plane_or_material_point(config):
    sample = make_geometric_reference(entry_arm(), obstacle(), "upper_arm", .3, [1., 0., 0.], .088, .06)
    sample["state"] = {"obstacles": [obstacle().metadata()]}
    validate_geometric_reference(sample, config)
    altered = copy.deepcopy(sample)
    altered["geometry"]["plane_offset"] += .01
    with pytest.raises(ValueError, match="plane"):
        validate_geometric_reference(altered, config)
    altered = copy.deepcopy(sample)
    altered["geometry"]["fraction"] = .7
    with pytest.raises(ValueError, match="plane"):
        validate_geometric_reference(altered, config)
    with pytest.raises(ValueError, match="geometric reference"):
        validate_geometric_reference(dict(sample, reference_protocol=PROTOCOL), config)


def fake_base(objects):
    capsules = [CapsuleState(np.array([-.2, -.1, 0.]), np.array([-.2, .1, 0.]), .088, name)
                for name in ARM_GROUPS]
    queries = [[ConvexQuery(.05, np.array([-.112, 0., 0.]), np.array([-.05, 0., 0.]), np.array([-1., 0., 0.]))
                for _ in ARM_GROUPS] for _ in objects]
    return SimpleNamespace(obstacles=objects, robot=SimpleNamespace(capsules=lambda: capsules),
                           _base_frame=lambda: (np.zeros(3), np.eye(3)), _external_queries=queries,
                           _static_clearance=lambda obj: .09)


def test_multi_obstacle_primary_choice_is_order_independent_and_zero_intent_is_clear(config):
    guide = GeometricPassageReference.__new__(GeometricPassageReference)
    guide.config, guide.local = config, config["local_avoidance"]
    guide.settings = guide.local["geometry"]
    intent = NominalIntent(np.ones(6), entry_arm(), np.tile([1., 0., 0.], 8), entry_arm(),
                           {1: .08, 2: .08}, .01, True)
    a = guide.build(fake_base([obstacle(2), obstacle(1)]), intent=intent)
    b = guide.build(fake_base([obstacle(1), obstacle(2)]), intent=intent)
    assert a.status == b.status == "blocked"
    assert a.sample["geometry"]["obstacle_identity"] == b.sample["geometry"]["obstacle_identity"] == 1
    np.testing.assert_array_equal(a.sample["direction"], b.sample["direction"])
    intent.velocity[:] = 0.
    assert guide.build(fake_base([obstacle(1)]), intent=intent).status == "clear"


@pytest.fixture
def base(config):
    env = ThesisHomotopyEnv(simulation_config(config), cache_geometry_queries=True)
    env.configure_episode("none", xi=1., strict=True, terminate_on_success=False)
    env.reset(seed=1234)
    try:
        yield env
    finally:
        env.close()


def test_geometry_prepare_and_handoff_do_not_step_any_physics_or_change_live_state(config, base, monkeypatch):
    reaching = SimpleNamespace(select_action=lambda observation, deterministic: np.array([.05, 0., 0., 0., 0., 0.]))
    guide = GeometricPassageReference(config, reaching)
    try:
        before = copy.deepcopy(base.episode_state_dict())
        monkeypatch.setattr(p, "stepSimulation", lambda **kwargs: pytest.fail("geometry must not advance physics"))
        observation = base._observe()[0]
        intent = guide.prepare(base, observation)
        guide.build(base, intent=intent)
        # A frozen plane aligned to this intent gives positive continuation.
        delta = intent.predicted_arm - intent.entry_arm
        sample = dict(entry_arm=intent.entry_arm, direction=delta / np.linalg.norm(delta), length=.03)
        assert guide.handoff(base, observation, sample)
        after = base.episode_state_dict()
        for key in ("q", "qdot", "goal_position", "goal_quaternion"):
            np.testing.assert_array_equal(before[key], after[key])
        for key in ("step_count", "physics_step_count", "success_hold_count", "rng_state"):
            assert before[key] == after[key]
        assert guide.query.physics_step_count == 0
    finally:
        guide.close()


def test_instantaneous_handoff_checks_all_objects_self_and_joint_limits(config, monkeypatch):
    guide = GeometricPassageReference.__new__(GeometricPassageReference)
    guide.config, guide.local = config, config["local_avoidance"]
    monkeypatch.setattr("rl_risk_sac.tasks.geometric_passage.MeshClearance",
                        lambda base: SimpleNamespace(external=.08, self_distance=.01))
    base = SimpleNamespace(d_self_safe=.005)
    intent = NominalIntent(np.zeros(6), np.zeros(24), np.ones(24), np.ones(24), {1: .08, 2: .01}, .01, True)
    guide.prepare = lambda *args: intent
    sample = dict(entry_arm=np.zeros(24), direction=np.ones(24) / np.sqrt(24), length=1.)
    assert not guide.handoff(base, None, sample)  # The second object blocks handoff.
    intent.predicted_external[2] = .08
    assert guide.handoff(base, None, sample)
    intent.predicted_self = .001
    assert not guide.handoff(base, None, sample)
    intent.predicted_self, intent.within_limits = .01, False
    assert not guide.handoff(base, None, sample)
    intent.within_limits, intent.predicted_arm = True, -np.ones(24)
    assert not guide.handoff(base, None, sample)  # S0 turns back into the passed plane.


def test_task_protocol_separates_same_shape_models_replay_and_bank_signatures(config):
    old = copy.deepcopy(config)
    old["local_avoidance"].pop("reference_mode")
    old["local_avoidance"].pop("geometry")
    assert local_protocol(config) == GEOMETRIC_PROTOCOL
    assert contract_digest(config) != contract_digest(old)
    legacy = build_thesis_sac(268, 6, old)
    geometric = build_thesis_sac(268, 6, config)
    with pytest.raises(ValueError, match="task protocol"):
        geometric.load_state_dict(legacy.state_dict())
    replay = LocalReplayBuffer(4, "cpu", 1, protocol=GEOMETRIC_PROTOCOL)
    with pytest.raises(ValueError, match="protocol"):
        replay.load_state_dict(LocalReplayBuffer(4, "cpu", 1).state_dict())
    assert not passage_candidate(-1., .5, config["local_avoidance"])


@pytest.mark.parametrize("field,value", [("prediction_dt", .1), ("kinematic_dt", .1),
                                         ("minimum_point_speed", 0.), ("minimum_closing_cosine", 1.)])
def test_invalid_geometry_budget_or_direction_threshold_rejected(config, field, value):
    config["local_avoidance"]["geometry"][field] = value
    with pytest.raises(ValueError, match="geometr"):
        validate_local_config(config)
