from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


CHECKPOINT_FORMAT = "rl_risk_sac.obstacle_lstm.v1"


def build_history_features(
    timestamps: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Convert a causal position history to translation-invariant LSTM input.

    Each row is ``[relative_position(3), velocity(3), age(1)]``.  Positions are
    relative to the latest observation and age is zero at the latest sample.
    The same function is used by offline training and online inference.
    """
    times = np.asarray(timestamps, dtype=np.float32)
    points = np.asarray(positions, dtype=np.float32)
    if times.ndim != 1 or points.shape != (len(times), 3) or len(times) < 2:
        raise ValueError("history must contain at least two timestamped 3D positions")
    if not np.isfinite(times).all() or not np.isfinite(points).all():
        raise ValueError("history timestamps and positions must be finite")
    intervals = np.diff(times)
    if np.any(intervals <= 0.0):
        raise ValueError("history timestamps must be strictly increasing")
    velocities = np.empty_like(points)
    velocities[1:] = np.diff(points, axis=0) / intervals[:, None]
    velocities[0] = velocities[1]
    relative_positions = points - points[-1]
    ages = times[-1] - times
    return np.concatenate(
        [relative_positions, velocities, ages[:, None]], axis=1
    ).astype(np.float32)


def resample_position_history(
    timestamps: np.ndarray,
    positions: np.ndarray,
    *,
    history_length: int,
    sample_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate tracker data onto the exact window used during training."""
    times = np.asarray(timestamps, dtype=np.float64)
    points = np.asarray(positions, dtype=np.float64)
    if times.ndim != 1 or points.shape != (len(times), 3) or len(times) < 2:
        raise ValueError("raw history must contain timestamped 3D positions")
    if history_length < 2 or not np.isfinite(sample_hz) or sample_hz <= 0.0:
        raise ValueError("invalid resampling window")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("raw history timestamps must be strictly increasing")
    target_times = times[-1] + (
        np.arange(-(history_length - 1), 1, dtype=np.float64) / sample_hz
    )
    if target_times[0] < times[0] - 1.0e-7:
        raise ValueError("raw obstacle history does not cover the LSTM window")
    resampled = np.stack(
        [np.interp(target_times, times, points[:, axis]) for axis in range(3)],
        axis=1,
    )
    return target_times.astype(np.float32), resampled.astype(np.float32)


class ObstacleMotionLSTM(nn.Module):
    """Predict future displacement from a causal obstacle-position history."""

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 2,
        horizon_steps: int = 3,
    ) -> None:
        super().__init__()
        if input_size != 7:
            raise ValueError("obstacle LSTM input_size must be 7")
        if hidden_size < 1 or num_layers < 1 or horizon_steps < 1:
            raise ValueError("invalid obstacle LSTM dimensions")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon_steps = horizon_steps
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, horizon_steps * 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(features)
        return self.head(sequence[:, -1]).reshape(-1, self.horizon_steps, 3)


@dataclass(frozen=True)
class LSTMObstaclePredictor:
    """CPU/GPU inference wrapper used directly by the PyBullet environment."""

    model: ObstacleMotionLSTM
    history_length: int
    history_hz: float
    horizon_times: np.ndarray
    input_scale: np.ndarray
    output_scale: float
    device: torch.device
    checkpoint_path: Path
    validation_metrics: dict[str, float]

    def predict(
        self,
        timestamps: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        times, points = resample_position_history(
            timestamps,
            positions,
            history_length=self.history_length,
            sample_hz=self.history_hz,
        )
        features = build_history_features(times, points)
        normalized = features / self.input_scale
        tensor = torch.from_numpy(normalized[None]).to(self.device)
        with torch.inference_mode():
            displacement = self.model(tensor)[0].cpu().numpy() * self.output_scale
        if not np.isfinite(displacement).all():
            raise FloatingPointError("obstacle LSTM returned non-finite displacement")
        return (points[-1][None, :] + displacement).astype(np.float32)


def load_lstm_predictor(
    checkpoint_path: str | Path,
    *,
    expected_history_length: int,
    expected_history_hz: float,
    expected_horizon_times: np.ndarray,
    device: str = "cpu",
) -> LSTMObstaclePredictor:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"obstacle LSTM checkpoint does not exist: {path}")
    checkpoint: dict[str, Any] = torch.load(
        path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported obstacle LSTM checkpoint format: {path}")
    history_length = int(checkpoint["history_length"])
    history_hz = float(checkpoint["history_hz"])
    horizon_times = np.asarray(checkpoint["horizon_times_s"], dtype=np.float32)
    expected_times = np.asarray(expected_horizon_times, dtype=np.float32)
    if history_length != expected_history_length:
        raise ValueError(
            f"LSTM history length mismatch: checkpoint={history_length}, "
            f"environment={expected_history_length}"
        )
    if not np.isclose(history_hz, expected_history_hz, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            f"LSTM history frequency mismatch: checkpoint={history_hz}, "
            f"environment={expected_history_hz}"
        )
    if horizon_times.shape != expected_times.shape or not np.allclose(
        horizon_times, expected_times, rtol=0.0, atol=1.0e-6
    ):
        raise ValueError(
            "LSTM prediction horizons do not match the environment control period"
        )
    input_scale = np.asarray(checkpoint["input_scale"], dtype=np.float32)
    output_scale = float(checkpoint["output_scale"])
    if input_scale.shape != (7,) or np.any(input_scale <= 0.0):
        raise ValueError("invalid LSTM input scale in checkpoint")
    if not np.isfinite(output_scale) or output_scale <= 0.0:
        raise ValueError("invalid LSTM output scale in checkpoint")
    model_config = checkpoint["model_config"]
    model = ObstacleMotionLSTM(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    target_device = torch.device(device)
    model.to(target_device).eval()
    return LSTMObstaclePredictor(
        model=model,
        history_length=history_length,
        history_hz=history_hz,
        horizon_times=horizon_times,
        input_scale=input_scale,
        output_scale=output_scale,
        device=target_device,
        checkpoint_path=path,
        validation_metrics={
            str(key): float(value)
            for key, value in checkpoint.get("validation_metrics", {}).items()
        },
    )
