from __future__ import annotations

import copy
import csv
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_risk_sac.algorithms.obstacle_encoder import ObstacleSetEncoder
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.algorithms.homotopy_replay import HomotopyReplayBuffer
from rl_risk_sac.algorithms.replay_buffer import Batch
from rl_risk_sac.tasks.obstacle_evaluation import ObstacleEvaluation
from scripts.core.train_thesis_homotopy import (
    ORIENTATION_PROBE_FIELDS, load_reference_actor, scalar_probe_metrics,
    summarize_pose_probe_outcomes,
)
from rl_risk_sac.envs.obstacle_scene import sample_obstacle, sample_shape, bounding_radius
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.robots.ur5_capsules import CapsuleState
from rl_risk_sac.tasks.convex_sweep import ConvexSweep, signed_capsule_distance, swept_risk
from rl_risk_sac.tasks.obstacle_observation import ObstacleObservation, stack_observations, pad_obstacle_sets
from rl_risk_sac.tasks.obstacle_prediction import (
    ObstaclePredictor, PositionLSTM, history_features, line_segment, percentile_scales,
)
from rl_risk_sac.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def test_s1_promotion_excludes_empty_scenes(monkeypatch):
    import scripts.core.train_thesis_homotopy as trainer
    config = load_config(ROOT / "configs/experiments/paper_full.yaml")
    trainer.apply_stage_profile(config, "s1")
    assert config["thesis"]["obstacles"]["count_probabilities"] == [.2, .8, 0., 0.]
    assert not trainer.promotion_episode_eligible(config, {"obstacle_count": 0})
    assert trainer.promotion_episode_eligible(config, {"obstacle_count": 1})

    class Curriculum:
        def pose_contract(self):
            return {}

    class ProbeReached(Exception):
        pass

    def probe(agent, probe_config, **kwargs):
        assert probe_config["thesis"]["obstacles"]["count_probabilities"] == [0., 1., 0., 0.]
        np.testing.assert_allclose(probe_config["thesis"]["obstacles"]["path_sampling"]["probabilities"], [0, .1875, .1875, .625])
        raise ProbeReached

    monkeypatch.setattr(trainer, "run_deterministic_pose_probe", probe)
    with pytest.raises(ProbeReached):
        trainer.run_pose_promotion_probe(object(), config, Curriculum(), {})
    assert config["thesis"]["obstacles"]["count_probabilities"] == [.2, .8, 0., 0.]


@pytest.fixture
def checkpoint(tmp_path):
    # Synthetic zero weights isolate transport/history tests; never a trained artifact.
    model = PositionLSTM()
    for parameter in model.parameters():
        parameter.data.zero_()
    path = tmp_path / "test_only_predictor.pt"
    torch.save(dict(format_version=1, accepted=True, prediction_dt=1 / 20,
                    model=model.state_dict(), input_scale=np.ones(7), output_scale=1.,
                    feature_scale=np.ones(37), acceptance=dict(epsilon_line=.005, epsilon_pos=.001)), path)
    return path


@pytest.fixture
def config(checkpoint):
    config = load_config(ROOT / "configs/experiments/paper_full.yaml")
    config["thesis"]["obstacles"].update(
        enabled=True, dynamic=True, predictor_checkpoint=str(checkpoint),
        count_probabilities=[0., 1., 0., 0.], motion_modes=["constant"],
    )
    config["sac"]["hidden_dims"] = [32, 32]
    return config


def test_static_agent_accepts_geometry_scale_from_old_horizon(config, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["prediction_dt"] = 1 / 30
    torch.save(state, checkpoint)
    config["thesis"]["obstacles"]["dynamic"] = False
    agent = ThesisSACAgent(241, 6, config)
    torch.testing.assert_close(agent.actor.obstacle_encoder.scale, torch.ones(37, device=agent.device))
    # A static scale import must not make an incompatible dynamic model usable.
    with pytest.raises(ValueError, match="horizon mismatch"):
        ObstaclePredictor(checkpoint)


def test_timestamp_history_and_frozen_lstm(checkpoint):
    predictor = ObstaclePredictor(checkpoint)
    record = dict(id=4, timestamp=0., position=np.zeros(3))
    np.testing.assert_array_equal(predictor.predict([record])[4], 0.)
    predictor.predict([record])
    assert len(predictor.histories[4]) == 1
    record.update(timestamp=.05, position=np.array([.01, 0., 0.]))
    np.testing.assert_allclose(predictor.predict([record])[4][:, 0], .01 + np.arange(1, 4) / 20 * .2)
    for i in range(2, 10):
        record.update(timestamp=i * .05, position=np.array([i * .01, 0., 0.]))
        result = predictor.predict([record])[4]
    np.testing.assert_allclose(result, np.repeat(record["position"][None], 3, axis=0))
    assert all(not p.requires_grad for p in predictor.model.parameters())
    predictor.predict([])
    assert not predictor.histories
    predictor.predict([record])
    assert len(predictor.histories[4]) == 1
    predictor.predict([dict(record, timestamp=.01)])
    assert len(predictor.histories[4]) == 1


def test_irregular_history_and_scales():
    times = [0., .02, .07]
    features = history_features([(t, np.array([t * 2, 0., 0.])) for t in times])
    np.testing.assert_allclose(features[:, 3], 2.)
    np.testing.assert_allclose(features[:, 6], [.07, .05, 0.])
    scale, output_scale = percentile_scales(np.zeros((2, 10, 7)), np.zeros((2, 3, 3)))
    np.testing.assert_array_equal(scale, 1.)
    assert output_scale == 1.


def test_reverse_sweep_contains_intermediate_extrema():
    low, high, _ = line_segment([[.1, 0, 0], [-.2, 0, 0], [.02, 0, 0]], 1e-6, 1e-8)
    np.testing.assert_allclose(low, [-.2, 0, 0])
    np.testing.assert_allclose(high, [.1, 0, 0])
    with pytest.raises(ValueError, match="straight-line"):
        line_segment([[.1, 0, 0], [0, .1, 0], [.3, 0, 0]], .001, 1e-8)


@pytest.mark.parametrize("shape,size,extent", [("sphere", [.2], .2), ("box", [.2, .2, .2], .2),
                                              ("cylinder", [.2, .4], .2)])
def test_convex_signed_distance_witnesses(shape, size, extent):
    capsule = CapsuleState(np.array([0., 0., -.05]), np.array([0., 0., .05]), .1, "test")
    for center, expected in ((1., 1. - extent - .1), (.25, .25 - extent - .1)):
        sweep = ConvexSweep(shape, size, [center, 0., 0.], np.eye(3))
        query = signed_capsule_distance(sweep, capsule)
        assert query.distance == pytest.approx(expected, abs=1e-5)
        np.testing.assert_allclose(query.normal, [-1., 0., 0.], atol=.01)
        np.testing.assert_allclose(query.robot_point - query.obstacle_point,
                                   query.distance * query.normal, atol=1e-5)


def test_empty_set_permutation_masks_and_gradients():
    encoder = ObstacleSetEncoder(np.ones(37))
    features = torch.randn(3, 5, 37, requires_grad=True)
    mask = torch.tensor([[False] * 5, [True, True, False, False, False], [True] * 5])
    output = encoder(features, mask)
    np.testing.assert_array_equal(output[0].detach(), 0.)
    permutation = torch.tensor([4, 2, 0, 3, 1])
    torch.testing.assert_close(output, encoder(features[:, permutation], mask[:, permutation]))
    output.sum().backward()
    assert features.grad[1, :2].abs().sum() > 0
    assert features.grad[0].abs().sum() == 0
    assert features.grad[1, 2:].abs().sum() == 0
    assert encoder(features[:, :0], mask[:, :0]).shape == (3, 128)


def test_ragged_transport_and_no_global_object_cap():
    observations = [ObstacleObservation(np.zeros(241), np.full((n, 37), n)) for n in (0, 1, 7)]
    batch = pickle.loads(pickle.dumps(stack_observations(observations)))
    features, mask = pad_obstacle_sets(batch.obstacle_sets, "cpu")
    assert features.shape == (3, 7, 37)
    assert mask.sum(1).tolist() == [0, 1, 7]
    restored = pickle.loads(pickle.dumps(observations[-1]))
    np.testing.assert_array_equal(restored.obstacle_features, 7.)


def test_shape_size_marginals_and_motion_continuity():
    rng = np.random.default_rng(4)
    samples = [sample_shape(rng) for _ in range(3000)]
    for shape in ("sphere", "box", "cylinder"):
        assert abs(sum(s[0] == shape for s in samples) / len(samples) - 1 / 3) < .04
    for name, target in (("small", .3), ("medium", .5), ("large", .2)):
        assert abs(sum(s[2] == name for s in samples) / len(samples) - target) < .04
    obstacle = sample_obstacle(rng, 0, [-1, -1, -1], [1, 1, 1])
    for mode in ("constant", "accelerate", "decelerate", "stop", "reverse"):
        obstacle.speed_mode = mode
        before = obstacle.position(obstacle.period - 1e-6)
        after = obstacle.position(obstacle.period + 1e-6)
        assert np.linalg.norm(after - before) < 1e-5
        np.testing.assert_allclose(np.cross(obstacle.position(.2) - obstacle.initial_position, obstacle.direction), 0, atol=1e-8)


def test_risk_uses_nearest_group_and_is_bounded():
    settings = dict(d_safe=.05, d_cut=.2, tau=.02, phi_max=5.)
    assert swept_risk(np.empty((0, 37)), **settings) == 0.
    features = np.ones((2, 37))
    assert swept_risk(features, **settings) == 0.
    features[1, 2] = -.1
    assert .9 < swept_risk(features, **settings) <= 1.


def test_environment_actor_restore_and_real_contact(config):
    torch.set_num_threads(1)
    env = ThesisHomotopyEnv(config)
    try:
        observation, info = env.reset(seed=110)
        assert observation.shape == (241,)
        assert observation.obstacle_features.shape == (1, 37)
        assert info["obstacle_count"] == 1
        agent = ThesisSACAgent(241, 6, config)
        assert agent.critic_obs_dim == 187
        action = agent.select_action(observation, deterministic=True)
        assert np.isfinite(action).all()
        for _ in range(12):
            observation, *_ = env.step(np.zeros(6))
        state = env.episode_state_dict()
        restored = env.restore_episode_state(state)
        np.testing.assert_allclose(restored, observation, atol=1e-5)
        np.testing.assert_allclose(restored.obstacle_features, observation.obstacle_features, atol=1e-5)
        assert len(env.predictor.histories[0]) == 10
        # Moving a predicted sweep cannot create a true simulator collision.
        assert not env._collision_events()["obstacle_collision"]
    finally:
        env.close()


def test_full_sac_updates_independent_encoders_and_polyak_targets(config):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config["device"] = "cpu"
    config["sac"].update(actor_update_interval=1, chain_pcr_enabled=True,
                          chain_pcr_coefficient=.1)
    agent = ThesisSACAgent(241, 6, config)
    encoders = [network.obstacle_encoder for network in (agent.actor, agent.q1, agent.q2)]
    assert len({next(e.parameters()).data_ptr() for e in encoders}) == 3
    before = [copy.deepcopy(e.state_dict()) for e in encoders]
    target_before = copy.deepcopy(agent.target_q1.state_dict())
    batch = Batch(torch.randn(8, 241), torch.randn(8, 6).tanh(),
                  torch.randn(8, 1), torch.zeros(8, 1), torch.randn(8, 241),
                  torch.zeros(8, 1), obstacle_features=torch.randn(8, 3, 37),
                  obstacle_mask=torch.ones(8, 3, dtype=torch.bool),
                  next_obstacle_features=torch.randn(8, 2, 37),
                  next_obstacle_mask=torch.ones(8, 2, dtype=torch.bool))
    for _ in range(2):
        metrics = agent.update(batch, reference_observations=batch)
    assert all(np.isfinite(list(metrics.values())))
    assert metrics["policy_churn_kl"] > 0
    for encoder, previous in zip(encoders, before):
        assert any(not torch.equal(previous[k], v) for k, v in encoder.state_dict().items())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    assert any(not torch.equal(target_before[k], v) for k, v in agent.target_q1.state_dict().items())
    # One further update must apply Polyak averaging to encoder weights too.
    target_before = copy.deepcopy(agent.target_q1.state_dict())
    agent.update(batch, reference_observations=batch)
    for name, value in agent.target_q1.named_parameters():
        torch.testing.assert_close(value, target_before[name] * (1 - agent.tau)
                                   + agent.q1.state_dict()[name] * agent.tau)


def test_raw_sets_survive_replay_restore_snapshot_and_checksum():
    def make_replay():
        return HomotopyReplayBuffer(241, 6, "cpu", s0_anchor_capacity=16,
            s0_current_capacity=16, s0_success_capacity=16,
            s0_history_capacity_per_level=16, s0_anchor_fraction=0,
            s0_current_fraction=1, s0_success_fraction=0)
    replay = make_replay()
    info = dict(rho_position=.4, next_rho_position=.3, rho_orientation=.2,
                next_rho_orientation=.1, smooth_velocity=.5, task_reached=False,
                hard_failure=False, obstacle_collision=False, self_collision=False,
                environment_collision=False, joint_limit=False, control_max_risk=.2,
                control_min_distance=.1, obstacle_contact_seen_before=False,
                obstacle_contact_seen=False, velocity_magnitude=.25, orientation_scale=1.)
    for i, n in enumerate((0, 1, 7, 2)):
        replay.add("none", ObstacleObservation(np.full(241, i), np.full((n, 37), i + 1)),
                   np.zeros(6), ObstacleObservation(np.full(241, i + 1), np.full((n + 1, 37), i + 2)),
                   False, info, 0, i)
    restored = make_replay()
    restored.load_state_dict(pickle.loads(pickle.dumps(replay.state_dict())))
    assert replay.partition_sha256("none") == restored.partition_sha256("none")
    snapshot = restored.s0_current.snapshot(16, np.random.default_rng(0))
    for i in range(4):
        np.testing.assert_array_equal(snapshot.obstacle_sets[i], replay.s0_current.obstacle_sets[i])
        np.testing.assert_array_equal(snapshot.next_obstacle_sets[i], replay.s0_current.next_obstacle_sets[i])
    batch = restored.sample("s0", {}, 4)
    for row in range(4):
        i = int(batch.observations[row, 0])
        assert batch.obstacle_mask[row].sum() == (0, 1, 7, 2)[i]
        assert batch.next_obstacle_mask[row].sum() == (1, 2, 8, 3)[i]
        torch.testing.assert_close(batch.next_obstacle_features[row, batch.next_obstacle_mask[row]],
                                   torch.full(((1, 2, 8, 3)[i], 37), float(i + 2)))
    restored.s0_current.next_obstacle_sets[0][0, 0] += 1
    assert replay.partition_sha256("none") != restored.partition_sha256("none")


def test_reaching_actor_migration_is_explicit_and_resume_rejected(config, tmp_path):
    config["device"] = "cpu"
    old = ThesisSACAgent(162, 6, config)
    new = ThesisSACAgent(241, 6, config)
    path = tmp_path / "old.pt"
    torch.save(old.state_dict(), path)
    fresh = copy.deepcopy(new.actor.state_dict())
    migrated = load_reference_actor(new, path).state_dict()
    torch.testing.assert_close(migrated["backbone.0.weight"][:, :89], old.actor.state_dict()["backbone.0.weight"][:, :89])
    torch.testing.assert_close(migrated["backbone.0.weight"][:, 217:], old.actor.state_dict()["backbone.0.weight"][:, 138:])
    torch.testing.assert_close(migrated["backbone.0.weight"][:, 89:217], fresh["backbone.0.weight"][:, 89:217])
    for key in fresh:
        if key.startswith("obstacle_encoder."):
            torch.testing.assert_close(migrated[key], fresh[key])
    with pytest.raises(ValueError, match="explicit actor initialization"):
        new.load_state_dict(old.state_dict())


def test_small_evaluation_has_valid_json_and_action_aligned_timing():
    values = np.zeros((2, 13))
    values[:, 9] = [0, 1]
    result = summarize_pose_probe_outcomes(values, has_bins=True, orientation_bins=10,
        large_angle_steps=0, orientation_improvements=0, orientation_rebound_sum=0.)
    assert result["success_rate_o5_o6"] is None
    assert result["timeout_rate_o7_o9"] is None
    json.dumps(result, allow_nan=False)
    evaluation = ObstacleEvaluation()
    evaluation.record_step(0, dict(obstacle_processing_ms=7., action_obstacle_processing_ms=3.))
    assert evaluation.steps[0][0]["obstacle_processing_ms"] == 3.


def test_probe_csv_accepts_obstacle_reports_and_legacy_metrics():
    evaluation = ObstacleEvaluation()
    evaluation.record_step(0, dict(action_execution_s=.1))
    evaluation.finish_episode(0, dict(obstacle_count=1, obstacle_metadata=[
        dict(shape="sphere", size_class="small"),
    ]), success=True, collision=False)
    legacy = dict(success_rate=1., collision_rate=0.)
    metrics = {**legacy, **evaluation.report()}
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=ORIENTATION_PROBE_FIELDS)
    writer.writeheader()
    writer.writerow(scalar_probe_metrics(metrics))
    writer.writerow(scalar_probe_metrics(legacy))
    stream.seek(0)
    reader = csv.DictReader(stream)
    current, old = list(reader)
    assert reader.fieldnames[-3:] == [
        "obstacle_groups", "obstacle_episodes", "timing_contract",
    ]
    assert json.loads(current["obstacle_groups"]) == metrics["obstacle_groups"]
    assert json.loads(current["obstacle_episodes"]) == metrics["obstacle_episodes"]
    assert current["timing_contract"] == metrics["timing_contract"]
    assert current["success_rate"] == old["success_rate"] == "1.0"
    assert all(old[key] == "" for key in reader.fieldnames[-3:])
    assert None not in current  # Nested JSON stays inside its quoted CSV cell.
