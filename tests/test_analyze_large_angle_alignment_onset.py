from __future__ import annotations

from scripts.analyze_large_angle_alignment_onset import (
    METRICS, compare, contiguous_collapse, phase, select_success_controls,
)


def rows(steps, alignments=None, rho=.8):
    if alignments is None:
        alignments = [-.2] * len(steps)
    return [{"step": step, "actor_command_alignment": alignment,
             "rho_orientation": rho}
            for step, alignment in zip(steps, alignments)]


def test_collapse_requires_actual_consecutive_steps_not_band_visits():
    assert contiguous_collapse(rows([10, 11, 14, 15, 16])) is None
    assert contiguous_collapse(rows([10, 11, 12, 13, 14]))["onset_step"] == 10
    gap = rows(range(10, 16))
    gap[2]["rho_orientation"] = .59
    assert contiguous_collapse(gap) is None


def test_collapse_reports_first_qualifying_window():
    event = contiguous_collapse(rows(range(8), [1., 1., -.2, -.2, .1, .1, .1, -.4]))
    assert event["onset_step"] == 2
    assert event["detection_step"] == 5
    assert event["negative_steps"] == 2
    assert contiguous_collapse(rows(range(5), [.9, -.2, .2, .2, .2])) is None


def test_phase_is_based_on_pre_action_state_step():
    assert phase(50, 50) == "after_crossing"
    assert phase(49, 50) == "before_crossing"
    assert phase(100, None) == "before_crossing"


def test_controls_require_initial_bins_phase_and_calipers():
    event = {"episode": 7, "step": 40, "rho_orientation": .8, "rho_position": .1,
             "initial_orientation_bin": 7, "initial_position_bin": 2,
             "crossing_phase": "before_crossing"}
    valid = {**event, "episode": 107, "step": 45}
    candidates = [
        {**event, "episode": 1, "initial_orientation_bin": 8},
        {**event, "episode": 2, "initial_position_bin": 3},
        {**event, "episode": 3, "crossing_phase": "after_crossing"},
        {**event, "episode": 4, "step": 70}, valid,
    ]
    selected, balance = select_success_controls(event, candidates)
    assert selected["episode"] == 107
    assert balance["delta_step"] == 5
    assert select_success_controls(event, candidates[:-1])[0] is None


def test_compare_bootstraps_paired_joint_coordinates():
    pair = {"failure": {"q_rad": [1.] * 6,
                        "q_normalized": [.1] * 6,
                        "qdot_normalized": [.2] * 6,
                        "actor_joint_velocity_rad_s": [.3] * 6,
                        "angular_jacobian_singular_values": [1., .9, .8],
                        "episode": 1},
            "success": {"q_rad": [0.] * 6,
                        "q_normalized": [0.] * 6,
                        "qdot_normalized": [0.] * 6,
                        "actor_joint_velocity_rad_s": [0.] * 6,
                        "angular_jacobian_singular_values": [1., .9, .8],
                        "episode": 2}}
    for row in pair.values():
        row.update({metric: 0. for metric in METRICS})
    result = compare([pair, pair], seed=1)
    assert result["metrics"]["q_rad"]["paired_delta_mean"] == [1.] * 6
    assert result["metrics"]["q_rad"]["paired_delta_bootstrap_95pct_ci"] == [[1., 1.]] * 6
