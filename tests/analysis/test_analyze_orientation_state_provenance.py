from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.analyze_orientation_state_provenance import (
    _finish_actor_row,
    _landmark_bucket,
    action_orientation_metrics,
    is_first_descending_landmark,
    nearest_state_matches,
    orientation_dls_actions,
    orientation_origin,
    select_matching_records,
)


def test_orientation_origin_uses_actual_native_band_and_large_bins():
    assert orientation_origin(.8, 2) == "native_band"
    assert orientation_origin(1.4, 4) == "medium_descent"
    assert orientation_origin(1.8, 5) == "medium_descent"
    assert orientation_origin(2.4, 7) == "large_descent"
    assert orientation_origin(.4, 1) == "outside_control"


def test_orientation_dls_separates_direction_and_speed():
    jacobian = np.zeros((6, 6), dtype=np.float64)
    jacobian[3:, :3] = np.eye(3)
    error = np.asarray([1.0, 0.0, 0.0])
    actor = np.asarray([0.0, 0.35, 0.0, 0.0, 0.0, 0.0])
    actions = orientation_dls_actions(
        error, jacobian, actor, action_scale=.7, damping=.05,
    )
    np.testing.assert_allclose(
        actions["full_bound"], [.7, 0., 0., 0., 0., 0.], atol=1e-10,
    )
    np.testing.assert_allclose(
        actions["same_norm"], [.35, 0., 0., 0., 0., 0.], atol=1e-10,
    )
    assert actions["actor_action_cosine"] == pytest.approx(0.0)
    metrics = action_orientation_metrics(
        error, jacobian, actions["same_norm"], action_scale=.7,
    )
    assert metrics["angular_alignment"] == pytest.approx(1.0)
    assert metrics["effective_capacity_fraction"] == pytest.approx(.5)


def _match_row(episode: int, origin: str, outcome: float, offset: float = 0.0):
    observation = np.zeros(8, dtype=np.float64)
    observation[0] = offset
    return {
        "episode": episode,
        "step": 10,
        "origin": origin,
        "orientation_band_bucket": 2,
        "rho_orientation": .8 + offset,
        "rho_position": .2,
        "remaining_time_fraction": .8,
        "q_normalized": np.zeros(6),
        "qdot_normalized": np.zeros(6),
        "orientation_error_direction": np.asarray([1., 0., 0.]),
        "position_error_direction": np.asarray([0., 1., 0.]),
        "jacobian_directional_row": np.ones(6),
        "angular_jacobian": np.zeros((3, 6)),
        "jacobian_angular_singular_values": np.ones(3),
        "orientation_capacity_rad_s": 2.0,
        "observation": observation,
        "actor_command_alignment": outcome,
        "actor_measured_alignment": outcome,
        "orientation_progress_rad_s": outcome,
        "actor_command_effective_capacity_fraction": outcome,
        "actor_dls_action_cosine": outcome,
        "actor_dls_same_norm_effective_ratio": outcome,
        "orientation_reversal": float(outcome < 0.0),
        "future_cross_0p6": float(outcome > 0.0),
        "future_success": float(outcome > 0.0),
    }


def test_nearest_state_matches_reports_paired_origin_gap():
    queries = [
        _match_row(1, "large_descent", .2, 0.001),
        _match_row(2, "large_descent", .4, 0.002),
    ]
    controls = [
        _match_row(3, "native_band", .8, 0.0),
        _match_row(4, "native_band", .9, 0.01),
    ]
    result = nearest_state_matches(
        queries, controls, feature_set="rho_only", caliper=2.0, seed=1,
    )
    assert result["matched_pairs"] == 2
    assert result["outcomes"]["actor_command_alignment"][
        "paired_difference_query_minus_control"
    ] == pytest.approx(-.5)


def test_matching_records_keep_stalled_states_across_time_windows():
    base = _match_row(1, "large_descent", .2)
    rows = []
    for step in (10, 11, 20):
        row = dict(base)
        row["step"] = step
        row["time_bucket_10_steps"] = step // 10
        rows.append(row)
    selected = select_matching_records(rows)
    assert [row["step"] for row in selected] == [10, 20]


def test_measured_alignment_uses_pre_action_jacobian_and_error():
    pre_jacobian = np.zeros((3, 6))
    pre_jacobian[:3, :3] = np.eye(3)
    row = {
        "jacobian_directional_row": np.asarray([1., 0., 0., 0., 0., 0.]),
        "angular_jacobian": pre_jacobian,
        "actor_joint_velocity": np.zeros(6),
    }
    post_jacobian = np.zeros((6, 6))
    post_jacobian[3, :3] = [0., 1., 0.]
    info = {
        "ee_jacobian": post_jacobian,
        "orientation_error_vector": np.asarray([0., 1., 0.]),
        "joint_velocity": np.asarray([1., 0., 0., 0., 0., 0.]),
        "commanded_joint_velocity": np.zeros(6),
        "rho_orientation": .8,
        "next_rho_orientation": .79,
        "rho_position": .2,
        "next_rho_position": .19,
        "next_keypoint_distance": .1,
    }
    from scripts.analysis.analyze_orientation_state_provenance import REWARD_INFO_KEYS
    info.update({key: 0. for key in REWARD_INFO_KEYS})
    _finish_actor_row(row, info, control_dt=.05)
    assert row["actor_measured_alignment"] == pytest.approx(1.)
    assert row["actor_measured_effective_angular_speed_rad_s"] == pytest.approx(1.)


def test_landmarks_require_first_descending_boundary_crossing():
    visited: set[int] = set()
    sequence = [1.05, .96, .94, 1.02, .97, .87, .92, .89, .78, .68]
    chosen = []
    for step, (previous, rho) in enumerate(zip(sequence, sequence[1:]), 1):
        if not .6 <= rho < 1.0:
            continue
        bucket = min(int((rho - .6 + 1e-10) / .1), 3)
        if is_first_descending_landmark(
            rho, previous, bucket, visited, band_low=.6, band_high=1.,
        ):
            chosen.append((step, bucket))
            visited.add(bucket)
    assert chosen == [(1, 3), (5, 2), (8, 1), (9, 0)]
    assert is_first_descending_landmark(
        .8, None, 2, set(), band_low=.6, band_high=1., native_initial=True,
    )


def test_landmark_bucket_handles_exact_decimal_edges():
    assert [_landmark_bucket(rho, .6, 1.) for rho in (.6, .7, .8, .9, 1.)] == [0, 1, 2, 3, 3]
