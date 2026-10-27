from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl_risk_sac.algorithms.dual_sac import build_thesis_sac
import rl_risk_sac.control.joint_policy as joint
from rl_risk_sac.control.joint_policy import (
    JointPolicyController, OnlinePassageReference, PassageReference,
)
from rl_risk_sac.envs.obstacle_scene import MovingObstacle
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.local_avoidance import LOCAL_OBSERVATION_DIM, actor_digest, arm_vector
from rl_risk_sac.tasks.local_samples import configure_nominal_episode, simulation_config
from rl_risk_sac.tasks.obstacle_observation import obstacle_set
from rl_risk_sac.utils.config import load_config
from scripts.core.evaluate_joint_policy import evaluate_episode, restore_full_scene, validate_evaluation_bank
from scripts.core.train_thesis_homotopy import apply_stage_profile


@pytest.fixture
def config(tmp_path):
    cfg = load_config("configs/experiments/paper_local_avoidance_geometry.yaml")
    # Reconstruct the trajectory task explicitly for compatibility tests.
    cfg["local_avoidance"].pop("reference_mode")
    cfg["local_avoidance"].pop("geometry")
    apply_stage_profile(cfg, "s1")
    cfg["device"] = "cpu"
    cfg["sac"]["hidden_dims"] = [16, 16]
    scale = tmp_path / "scale.pt"
    torch.save({"format_version": 1, "accepted": True,
                "feature_scale": np.ones(37, np.float32)}, scale)
    cfg["thesis"]["obstacles"]["predictor_checkpoint"] = str(scale)
    return cfg


@pytest.fixture
def env(config):
    base = ThesisHomotopyEnv(simulation_config(config), cache_geometry_queries=True)
    base.configure_episode("none", xi=1., strict=True, position_tolerance=.01,
                           orientation_tolerance=.01)
    base.reset(seed=1234)
    state = copy.deepcopy(base.episode_state_dict())
    state["obstacles"] = [MovingObstacle(
        identity, "sphere", [.04], "small", [-.6, -.6 + identity * 1.2, 1.],
        [0., 0., 0., 1.], [1., 0., 0.], "constant", .1, 1., 0., -1., 1.).metadata()
        for identity in range(2)]
    base.restore_episode_state(state)
    try:
        yield base
    finally:
        base.close()


def fixed_agent():
    seen = []

    def select_action(observation, deterministic, *, policy="reaching"):
        seen.append((policy, np.asarray(observation).shape, obstacle_set(observation).shape))
        return np.zeros(6, np.float32)

    return SimpleNamespace(select_action=select_action,
                           reaching=SimpleNamespace(select_action=select_action)), seen


def sample_for(base):
    return dict(entry_arm=arm_vector(base), direction=np.ones(24) / np.sqrt(24), length=.2)


def assert_state_unchanged(before, after):
    # Covers RNG, goals, velocities, histories, contracts, all physical clocks.
    def equal(a, b):
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        else:
            assert a == b
    equal(before, after)


@pytest.mark.parametrize("kind", ["blocked", "clear", "no_local_exit", "no_continuation_budget", "short"])
def test_reference_uses_last_unsafe_region_and_validated_exit(config, kind):
    builder = OnlinePassageReference.__new__(OnlinePassageReference)
    builder.config = config
    vectors = np.arange(40)[:, None] * np.ones((1, 24)) * .01
    distances = np.full(40, .2)
    distances[3:5] = -.01
    distances[12:14] = .01  # The second blocking region must also be passed.
    if kind == "clear":
        distances[:] = .08  # Inside trigger radius, but still safe to reach.
    elif kind == "no_local_exit":
        distances[14:] = .04
    elif kind == "no_continuation_budget":
        distances[30:35] = .01
    elif kind == "short":
        vectors[:] *= .0001
    result = builder.from_trace(vectors, distances)
    assert result.status == ("reference_motion_too_small" if kind == "short" else kind)
    if kind == "blocked":
        np.testing.assert_array_equal(result.sample["reference_exit_arm"], vectors[14])
        np.testing.assert_allclose(result.sample["direction"] * result.sample["length"], vectors[14] - vectors[0])


def test_online_reference_does_not_modify_live_physics_or_rng(config, env):
    config["local_avoidance"]["nominal_max_steps"] = 3
    agent, _ = fixed_agent()
    builder = OnlinePassageReference(config, agent.reaching)
    try:
        before = copy.deepcopy(env.episode_state_dict())
        result = builder.build(env)
        assert result.status == "nominal_budget"
        assert not builder.nominal.obstacles
        assert len(builder.probe.obstacles) == 2
        builder.next_clearance(env, env._observe()[0])
        assert_state_unchanged(before, env.episode_state_dict())
    finally:
        builder.close()


def test_handoffs_are_continuous_and_overlap_does_not_toggle(config, env, monkeypatch):
    agent, seen = fixed_agent()
    controller = JointPolicyController(config, agent)
    try:
        monkeypatch.setattr(env, "_static_clearance", lambda obstacle: .08)
        references = iter([PassageReference("blocked", sample_for(env)),
                           PassageReference("clear"), PassageReference("blocked", sample_for(env))])
        monkeypatch.setattr(controller.references, "build", lambda base: next(references))
        monkeypatch.setattr(joint, "passage_candidate", lambda *args: True)
        monkeypatch.setattr(joint, "verify_reaching_recovery", lambda *args: True)
        observation = env._observe()[0]
        initial = copy.deepcopy(env.episode_state_dict())
        monkeypatch.setattr(env, "reset", lambda *args, **kwargs: pytest.fail("handoff reset the live simulation"))
        monkeypatch.setattr(env, "restore_episode_state", lambda *args: pytest.fail("handoff replaced the live state"))
        observation, terminated, truncated, info = controller.step(env, observation)
        assert not terminated and not truncated
        assert info["action_policy"] == "avoidance" and info["next_policy"] == "reaching"
        assert seen[0] == ("avoidance", (268,), (2, 37))
        assert env.step_count == initial["step_count"] + 1
        assert env.physics_step_count == initial["physics_step_count"] + env.substeps
        assert env.precision_control_enabled
        np.testing.assert_array_equal(env.goal_position, initial["goal_position"])
        assert controller.passages == 1
        # Same 8 cm distance after handoff: a clear nominal continuation stays reaching.
        observation, _, _, info = controller.step(env, observation)
        assert info["action_policy"] == "reaching"
        # A genuinely blocked continuation can be taken over again, without reset.
        monkeypatch.setattr(joint, "verify_reaching_recovery", lambda *args: False)
        observation, _, _, info = controller.step(env, observation)
        assert info["action_policy"] == "avoidance"
        assert env.step_count == initial["step_count"] + 3
        assert controller.attempts == 2 and controller.passages == 1
        assert [(row["source"], row["destination"]) for row in controller.events] == [
            ("reaching", "avoidance"), ("avoidance", "reaching"), ("reaching", "avoidance")]
    finally:
        controller.close()


def test_switch_before_reaching_action_crosses_trigger_boundary(config, env, monkeypatch):
    agent, seen = fixed_agent()
    controller = JointPolicyController(config, agent)
    try:
        monkeypatch.setattr(env, "_static_clearance", lambda obstacle: .105)
        monkeypatch.setattr(controller.references, "next_clearance", lambda base, observation: .095)
        monkeypatch.setattr(controller.references, "build", lambda base: PassageReference("blocked", sample_for(base)))
        observation = env._observe()[0]
        observation, _, _, info = controller.step(env, observation)
        assert info["action_policy"] == "avoidance"
        assert seen[0][0] == "avoidance"
        assert controller.events[0]["clearance_m"] == .105
    finally:
        controller.close()


def test_failed_recovery_cannot_handoff_and_local_budget_is_enforced(config, env, monkeypatch):
    config["local_avoidance"]["horizon"] = 2
    agent, _ = fixed_agent()
    controller = JointPolicyController(config, agent)
    try:
        monkeypatch.setattr(env, "_static_clearance", lambda obstacle: .08)
        monkeypatch.setattr(controller.references, "build", lambda base: PassageReference("blocked", sample_for(base)))
        monkeypatch.setattr(joint, "passage_candidate", lambda *args: True)
        monkeypatch.setattr(joint, "verify_reaching_recovery", lambda *args: False)
        outcome = evaluate_episode(env, controller, env._observe()[0])
        assert outcome["status"] == "local_timeout"
        assert outcome["length"] == 2 and outcome["verified_passages"] == 0
        assert outcome["policy_steps"] == {"reaching": 0, "avoidance": 2}
        assert not outcome["success"]
        with pytest.raises(RuntimeError, match="reset"):
            controller.step(env, env._observe()[0])
    finally:
        controller.close()


def test_reference_failure_is_reported_without_stepping_or_falling_back(config, env, monkeypatch):
    agent, seen = fixed_agent()
    controller = JointPolicyController(config, agent)
    try:
        monkeypatch.setattr(env, "_static_clearance", lambda obstacle: .08)
        monkeypatch.setattr(controller.references, "build", lambda base: PassageReference("no_local_exit"))
        before = copy.deepcopy(env.episode_state_dict())
        outcome = evaluate_episode(env, controller, env._observe()[0])
        assert outcome["status"] == "reference_unavailable:no_local_exit"
        assert outcome["length"] == 0 and not outcome["success"] and not seen
        assert_state_unchanged(before, env.episode_state_dict())
    finally:
        controller.close()


def test_collision_overrides_candidate_and_goal_and_prevents_handoff(config, env, monkeypatch):
    agent, _ = fixed_agent()
    controller = JointPolicyController(config, agent)
    try:
        monkeypatch.setattr(env, "_static_clearance", lambda obstacle: .08)
        monkeypatch.setattr(controller.references, "build", lambda base: PassageReference("blocked", sample_for(base)))
        monkeypatch.setattr(joint, "passage_candidate", lambda *args: True)
        monkeypatch.setattr(joint, "verify_reaching_recovery", lambda *args: pytest.fail("collision cannot hand off"))
        monkeypatch.setattr(env, "_collision_events", lambda: dict(
            obstacle_collision=True, self_collision=False, environment_collision=False))
        outcome = evaluate_episode(env, controller, env._observe()[0])
        assert outcome["status"] == "collision" and not outcome["success"]
        assert outcome["verified_passages"] == 0
        assert outcome["length"] == 1
        assert outcome["simulated_seconds"] == pytest.approx(env.physics_dt)
    finally:
        controller.close()


def test_global_budget_is_shared_across_policies(config, env, monkeypatch):
    agent, _ = fixed_agent()
    controller = JointPolicyController(config, agent)
    try:
        env.horizon = 2
        monkeypatch.setattr(env, "_static_clearance", lambda obstacle: .08)
        references = iter([PassageReference("blocked", sample_for(env)), PassageReference("clear")])
        monkeypatch.setattr(controller.references, "build", lambda base: next(references))
        monkeypatch.setattr(joint, "passage_candidate", lambda *args: True)
        monkeypatch.setattr(joint, "verify_reaching_recovery", lambda *args: True)
        outcome = evaluate_episode(env, controller, env._observe()[0])
        assert outcome["status"] == "global_timeout" and not outcome["success"]
        assert outcome["policy_steps"] == {"reaching": 1, "avoidance": 1}
        assert outcome["verified_passages"] == 1
    finally:
        controller.close()


@pytest.mark.parametrize("overlap", ["identity", "trace", "none"])
def test_joint_evaluation_rejects_training_scene_or_source_trace(overlap, monkeypatch):
    import scripts.core.evaluate_joint_policy as evaluation
    # Full bank geometry/provenance validation is shared with local evaluation.
    def validate(bank, config, reaching, *, split):
        assert split == ("validation", "test")
        return {"evaluation-scene"}
    monkeypatch.setattr(evaluation, "validate_bank", validate)
    bank = {"samples": [{"provenance": {"seed": 123}}]}
    state = {"config": {}, "train_sample_ids": {"evaluation-scene"} if overlap == "identity" else set(),
             "train_trace_seeds": {123} if overlap == "trace" else set()}
    if overlap == "none":
        validate_evaluation_bank(bank, state, None)
    else:
        with pytest.raises(ValueError, match="overlap"):
            validate_evaluation_bank(bank, state, None)


def test_final_reaching_not_local_passage_determines_joint_success(config, env, monkeypatch):
    agent, _ = fixed_agent()
    controller = JointPolicyController(config, agent)
    try:
        position, quaternion = env.robot.end_effector_pose()
        state = copy.deepcopy(env.episode_state_dict())
        state["goal_position"], state["goal_quaternion"] = position, quaternion
        observation = env.restore_episode_state(state)
        env.success_hold_steps = 2
        monkeypatch.setattr(controller.references, "build", lambda base: PassageReference("clear"))
        outcome = evaluate_episode(env, controller, observation)
        assert outcome["success"] and outcome["status"] == "success"
        assert outcome["length"] == 2 and outcome["verified_passages"] == 0
    finally:
        controller.close()


def test_full_scene_starts_before_approach_and_ignores_saved_reference(config, env):
    configure_nominal_episode(env, config)
    env.reset(seed=321)
    initial = copy.deepcopy(env.episode_state_dict())
    captured = copy.deepcopy(initial)
    captured["obstacles"] = [MovingObstacle(
        0, "sphere", [.04], "small", [-.6, -.6, 1.], [0., 0., 0., 1.],
        [1., 0., 0.], "constant", .1, 1., 0., -1., 1.).metadata()]
    captured["q"] = captured["q"] + .02
    sample = dict(state=captured, provenance={"seed": 321}, direction="must not be used")
    observation = restore_full_scene(env, sample, config)
    np.testing.assert_array_equal(env.robot.joint_state_exact()[0], initial["q"])
    assert env.step_count == 0 and len(env.obstacles) == 1
    assert observation.shape == (241,) and env.contract.terminate_on_success
    sample["state"]["goal_position"] = sample["state"]["goal_position"] + .1
    with pytest.raises(ValueError, match="reproduces"):
        restore_full_scene(env, sample, config)


def test_reaching_and_avoidance_inputs_preserve_actor_weights(config, env):
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    before = {role: actor_digest(agent.policy(role).actor) for role in ("reaching", "avoidance")}
    observation = env._observe()[0]
    context = joint.local_context(arm_vector(env), sample_for(env), 0, 160)
    local = joint.ObstacleObservation(np.r_[np.asarray(observation), context], obstacle_set(observation))
    assert agent.select_action(observation, True, policy="reaching").shape == (6,)
    assert agent.select_action(local, True, policy="avoidance").shape == (6,)
    assert all(actor_digest(agent.policy(role).actor) == digest for role, digest in before.items())
