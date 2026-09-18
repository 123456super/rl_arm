import numpy as np
import pytest

from scripts.evaluate_thesis_homotopy import MOTION_METRICS, motion_metrics


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
