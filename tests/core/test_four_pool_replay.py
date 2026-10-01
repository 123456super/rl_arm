from __future__ import annotations

import numpy as np

from rl_risk_sac.algorithms.homotopy_replay import HomotopyReplayBuffer, _Partition


def _fill(part: _Partition, count: int, episode: int = 1) -> None:
    raw = np.zeros(part.raw.shape[1])
    raw[24:26] = [.1, .3]
    for step in range(count):
        part.add(np.full(2, step), np.zeros(1), np.full(2, step + 1),
                 False, raw, episode, step)


def test_four_pool_exact_quotas_and_checkpoint_restore():
    replay = HomotopyReplayBuffer(2, 1, "cpu", s0_joint_pose=True)
    replay.s0_current_scale = 1.0
    anchors = {tier: _Partition(2, 1, 300) for tier in ("o0_o4", "o5_o6", "o7_o9")}
    for part in anchors.values():
        _fill(part, 300)
    frontier = {i: _Partition(2, 1, 5) for i in range(85)}
    for part in frontier.values():
        _fill(part, 5)
    replay.configure_four_pool({"anchor": anchors, "frontier": frontier}, source="frozen-1m")
    _fill(replay.s0_current, 1200)
    semantic = _Partition(2, 1, 300)
    _fill(semantic, 300)
    replay.semantic_enabled = True
    replay.semantic_slots = {0: [semantic]}

    restored = _restored(replay)
    for subject in (replay, restored):
        batch = subject.sample("s0", {}, 1024, orientation_scale=1.0)
        assert len(batch.observations) == 1024
        assert {key: subject.last_orientation_sample_counts[key] for key in (
            "current_recent", "stability_anchor", "frontier", "semantic_long_term",
            "current_success", "previous",
        )} == {
            "current_recent": 384, "stability_anchor": 256, "frontier": 192,
            "semantic_long_term": 192, "current_success": 0, "previous": 0,
        }
        assert int(batch.protected_mask.sum()) == 256
        assert sum(part.size for part in subject.stability_anchor.values()) == 900
        assert subject.s0_current_success.size == 0


def _restored(replay):
    result = HomotopyReplayBuffer(2, 1, "cpu", s0_joint_pose=True)
    result.load_state_dict(replay.state_dict())
    result.semantic_enabled = True
    result.semantic_slots = replay.semantic_slots
    return result


def test_frontier_keeps_early_and_success_recovery_steps():
    replay = HomotopyReplayBuffer(2, 1, "cpu", s0_joint_pose=True)
    replay.s0_current_scale = 1.0
    seed = {"anchor": {tier: _Partition(2, 1, 300)
                       for tier in ("o0_o4", "o5_o6", "o7_o9")},
            "frontier": {i: _Partition(2, 1, 176) for i in range(85)}}
    for part in seed["anchor"].values():
        _fill(part, 300)
    for part in seed["frontier"].values():
        _fill(part, 1)
    replay.configure_four_pool(seed, source="frozen-1m")
    for step in range(70):
        replay.s0_current.add(np.zeros(2), np.zeros(1), np.ones(2), False,
                              np.zeros(29), 100, 60 + step)
        replay.add_frontier_from_recent(100, 0, 60, recovered=False)
    replay.add_frontier_from_recent(100, 0, 60, recovered=True)
    steps = set(replay.frontier[0].step[:replay.frontier[0].size].tolist())
    assert set(range(60, 100)).issubset(steps)
    assert set(range(110, 130)).issubset(steps)
    assert 105 not in steps


def test_four_pool_promotion_inherits_history_and_starts_new_recent():
    replay = HomotopyReplayBuffer(2, 1, "cpu", s0_joint_pose=True)
    replay.s0_current_scale = 0.0
    anchors = {tier: _Partition(2, 1, 300) for tier in ("o0_o4", "o5_o6", "o7_o9")}
    for part in anchors.values():
        _fill(part, 300)
    frontier = {i: _Partition(2, 1, 5) for i in range(85)}
    for part in frontier.values():
        _fill(part, 5)
    replay.configure_four_pool({"anchor": anchors, "frontier": frontier}, source="frozen-1m")
    _fill(replay.s0_current, 1200)
    semantic = _Partition(2, 1, 10)
    _fill(semantic, 10)
    replay._loaded_semantic_state = {"enabled": True, "slots": {0: [semantic]}}

    replay.promote_four_pool_to_history(.25)
    assert not replay.four_pool_enabled
    assert replay.four_pool_history_mode
    assert replay.s0_current.size == 0
    assert replay.s0_history[0.0].size == 1200 + 900 + 425 + 10
    assert np.allclose(replay.s0_history[0.0].raw[:1200, 24], .1)

    batch = replay.sample("s0", {}, 1024, orientation_scale=.25)
    assert replay.last_orientation_sample_counts["previous"] == 1024
    assert int(batch.protected_mask.sum()) == 1024
    _fill(replay.s0_current, 700, episode=4)
    for subject in (replay, _restored(replay)):
        batch = subject.sample("s0", {}, 1024, orientation_scale=.25)
        assert subject.last_orientation_sample_counts["current_recent"] == 512
        assert subject.last_orientation_sample_counts["previous"] == 512
        assert int(batch.protected_mask.sum()) == 512


def test_four_pool_promotion_current_only_keeps_history_without_sampling_it():
    replay = HomotopyReplayBuffer(2, 1, "cpu", s0_joint_pose=True)
    replay.s0_current_scale = 0.0
    anchors = {tier: _Partition(2, 1, 300) for tier in ("o0_o4", "o5_o6", "o7_o9")}
    for part in anchors.values():
        _fill(part, 300)
    frontier = {i: _Partition(2, 1, 5) for i in range(85)}
    for part in frontier.values():
        _fill(part, 5)
    replay.configure_four_pool({"anchor": anchors, "frontier": frontier}, source="frozen-1m")
    _fill(replay.s0_current, 1200)

    replay.promote_four_pool_to_history(.25, current_fraction=1.0)
    assert replay.s0_history[0.0].size == 1200 + 900 + 425
    _fill(replay.s0_current, 1023, episode=4)
    restored = _restored(replay)
    for subject in (replay, restored):
        assert subject.four_pool_current_fraction == 1.0
        assert not subject.has_training_batch("s0", 1024, 0)
    _fill(replay.s0_current, 1, episode=5)
    for subject in (replay, restored):
        assert subject.has_training_batch("s0", 1024, 0)
        batch = subject.sample("s0", {}, 1024, orientation_scale=.25)
        assert subject.last_orientation_sample_counts["current_recent"] == 1024
        assert subject.last_orientation_sample_counts["previous"] == 0
        assert int(batch.protected_mask.sum()) == 0
        assert subject.s0_history[0.0].size == 1200 + 900 + 425
