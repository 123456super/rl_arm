import numpy as np
import pytest

from scripts.core.evaluate_thesis_homotopy import (
    MOTION_METRICS,
    automatic_num_envs,
    motion_metrics,
)
from scripts.analysis.analyze_orientation_crossing import (
    CROSSING_METRICS,
    infer_mechanism,
    nearest_matched_analysis,
    summarize_group,
)
from scripts.analysis.analyze_large_angle_transient import (
    _transition_row,
    summarize_transitions,
)


def test_motion_metrics_compute_velocity_acceleration_and_jerk():
    command = np.asarray([[0.0, 0.0], [1.0, -1.0], [3.0, -3.0]])
    measured = 0.5 * command
    metrics = motion_metrics(command, measured, control_dt=0.5)

    assert set(metrics) == set(MOTION_METRICS)
    assert metrics["command_velocity_rms_rad_s"] == pytest.approx(np.sqrt(20.0 / 6.0))
    assert metrics["command_velocity_peak_rad_s"] == 3.0
    assert metrics["command_acceleration_rms_rad_s2"] == pytest.approx(np.sqrt(10.0))
    assert metrics["command_acceleration_peak_rad_s2"] == 4.0
    assert metrics["command_jerk_rms_rad_s3"] == 4.0
    assert metrics["command_jerk_peak_rad_s3"] == 4.0
    assert metrics["measured_velocity_rms_rad_s"] == pytest.approx(
        0.5 * metrics["command_velocity_rms_rad_s"]
    )
    assert metrics["measured_acceleration_rms_rad_s2"] == pytest.approx(
        0.5 * metrics["command_acceleration_rms_rad_s2"]
    )
    assert metrics["measured_jerk_rms_rad_s3"] == 2.0


def test_motion_metrics_handle_short_episode_and_validate_inputs():
    one_step = np.zeros((1, 6), dtype=np.float64)
    metrics = motion_metrics(one_step, one_step, control_dt=0.05)
    assert metrics["command_acceleration_rms_rad_s2"] == 0.0
    assert metrics["measured_jerk_peak_rad_s3"] == 0.0

    with pytest.raises(ValueError, match="matching 2D"):
        motion_metrics(one_step, np.zeros((2, 6)), control_dt=0.05)
    with pytest.raises(ValueError, match="at least one"):
        motion_metrics(np.empty((0, 6)), np.empty((0, 6)), control_dt=0.05)
    with pytest.raises(ValueError, match="positive"):
        motion_metrics(one_step, one_step, control_dt=0.0)


def test_automatic_num_envs_never_exceeds_episode_count():
    assert automatic_num_envs(1) == 1
    assert 1 <= automatic_num_envs(7) <= 7


def _crossing_record(episode, group, rho_position, remaining, success, offset=0.0):
    crossing = {metric: 1.0 + offset for metric in CROSSING_METRICS}
    crossing.update({
        "rho_position": rho_position,
        "remaining_time_fraction": remaining,
        "remaining_steps": int(240 * remaining),
        "step": int(240 * (1.0 - remaining)),
    })
    return {
        "episode": episode,
        "orientation_group": group,
        "crossed": True,
        "crossing": crossing,
        "success": success,
        "timeout": not success,
        "collision": False,
    }


def test_crossing_analysis_matches_position_and_remaining_time():
    records = [
        _crossing_record(0, "o7_o9", .10, .60, True, .1),
        _crossing_record(1, "o7_o9", .20, .40, False, .2),
        _crossing_record(2, "o0_o4", .101, .601, True, .3),
        _crossing_record(3, "o0_o4", .199, .399, True, .4),
    ]
    matched = nearest_matched_analysis(records, caliper=.1)
    assert matched["matched_pairs"] == 2
    assert matched["unique_control_episodes"] == 2
    assert matched["large_success_rate"] == .5
    assert matched["matched_control_success_rate"] == 1.0
    assert matched["success_rate_difference_large_minus_control"] == -.5
    groups = {
        "o7_o9": summarize_group(records[:2]),
        "o0_o4": summarize_group(records[2:]),
    }
    assessment = infer_mechanism(groups, matched)
    assert assessment["label"] == "B_path_dependence_supported"


def test_crossing_group_reports_episodes_that_never_cross():
    crossed = _crossing_record(0, "o7_o9", .1, .5, True)
    never = {
        "episode": 1,
        "orientation_group": "o7_o9",
        "crossed": False,
        "crossing": None,
        "success": False,
        "timeout": True,
        "collision": False,
    }
    summary = summarize_group([crossed, never])
    assert summary["crossing_rate"] == .5
    assert summary["success_rate_given_crossing"] == 1.0
    assert summary["success_rate_without_crossing"] == 0.0


def test_large_angle_transition_metrics_separate_speed_and_alignment():
    info = {
        "ee_jacobian": np.eye(6),
        "joint_velocity": np.asarray([0., 0., 0., .35, 0., 0.]),
        "commanded_joint_velocity": np.asarray([0., 0., 0., .35, 0., 0.]),
        "policy_joint_velocity": np.asarray([0., 0., 0., .35, 0., 0.]),
        "ee_angular_velocity": np.asarray([.35, 0., 0.]),
        "orientation_error_vector": np.asarray([1., 0., 0.]),
        "rho_orientation": 1.0,
        "next_rho_orientation": .98,
        "rho_position": .2,
        "next_rho_position": .19,
    }
    row = _transition_row(
        info, control_dt=.05, action_scale=np.full(6, .7),
    )
    assert row["orientation_progress_rad_s"] == pytest.approx(.4)
    assert row["position_progress_m_s"] == pytest.approx(.2)
    assert row["angular_speed_rad_s"] == pytest.approx(.35)
    assert row["effective_angular_speed_rad_s"] == pytest.approx(.35)
    assert row["angular_alignment"] == pytest.approx(1.0)
    assert row["effective_capacity_fraction"] == pytest.approx(.5)
    assert row["jacobian_omega_relative_error"] == pytest.approx(0.0)
    summary = summarize_transitions([row])
    assert summary["positive_orientation_progress_rate"] == 1.0
    assert summary["both_progress_positive_rate"] == 1.0
