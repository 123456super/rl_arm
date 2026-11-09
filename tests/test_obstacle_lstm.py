from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from rl_risk_sac.prediction.obstacle_lstm import (
    CHECKPOINT_FORMAT,
    ObstacleMotionLSTM,
    build_history_features,
    load_lstm_predictor,
    resample_position_history,
)


def test_history_features_are_causal_and_translation_invariant():
    times = np.asarray([-0.2, -0.1, 0.0], dtype=np.float32)
    positions = np.asarray([[3.0, 1.0, 0.0], [3.1, 1.0, 0.0], [3.2, 1.0, 0.0]])
    features = build_history_features(times, positions)
    shifted = build_history_features(times + 7.0, positions + [9.0, -4.0, 2.0])
    np.testing.assert_allclose(features, shifted, atol=1.0e-5)
    np.testing.assert_allclose(
        features[:, :3], [[-0.2, 0, 0], [-0.1, 0, 0], [0, 0, 0]], atol=1.0e-6
    )
    np.testing.assert_allclose(features[:, 3:6], [[1, 0, 0]] * 3, atol=2.0e-6)
    np.testing.assert_allclose(features[:, 6], [0.2, 0.1, 0.0], atol=1.0e-6)


def test_lstm_checkpoint_contract_rejects_wrong_horizon():
    model_config = {"input_size": 7, "hidden_size": 4, "num_layers": 1, "horizon_steps": 3}
    model = ObstacleMotionLSTM(**model_config)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        torch.save({
            "format": CHECKPOINT_FORMAT,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "history_length": 10,
            "history_hz": 30.0,
            "horizon_times_s": [0.1, 0.2, 0.3],
            "input_scale": [1.0] * 7,
            "output_scale": 1.0,
        }, path)
        with pytest.raises(ValueError, match="horizons"):
            load_lstm_predictor(
                path,
                expected_history_length=10,
                expected_history_hz=30.0,
                expected_horizon_times=np.asarray([0.1, 0.2, 0.4]),
            )


def test_irregular_history_is_resampled_to_training_grid():
    times = np.asarray([-0.3, -0.2, -0.1, -0.05, 0.0])
    positions = np.stack([times, 2.0 * times, np.zeros_like(times)], axis=1)
    target_times, target_positions = resample_position_history(
        times, positions, history_length=4, sample_hz=10.0
    )
    np.testing.assert_allclose(target_times, [-0.3, -0.2, -0.1, 0.0], atol=1.0e-7)
    np.testing.assert_allclose(
        target_positions,
        np.stack([target_times, 2.0 * target_times, np.zeros(4)], axis=1),
        atol=1.0e-7,
    )
