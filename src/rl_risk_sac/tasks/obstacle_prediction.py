"""Timestamped object tracking and frozen, offline-trained 10x7 -> 3x3 LSTM."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

PREDICTION_DT = 1.0 / 20.0
HORIZONS = np.arange(1, 4, dtype=np.float64) * PREDICTION_DT


def history_features(history):
    times = np.asarray([row[0] for row in history], dtype=np.float64)
    positions = np.asarray([row[1] for row in history], dtype=np.float64)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("prediction history requires distinct increasing timestamps")
    velocities = np.diff(positions, axis=0) / np.diff(times)[:, None]
    velocities = np.concatenate((velocities[:1], velocities), axis=0)
    return np.concatenate((positions - positions[-1], velocities,
                           (times[-1] - times)[:, None]), axis=1).astype(np.float32)


class PositionLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(7, 128, num_layers=2, batch_first=True)
        self.output = nn.Linear(128, 9)

    def forward(self, history):
        # Omitted h/c explicitly initializes a fresh zero state per window.
        _, (hidden, _) = self.lstm(history)
        return self.output(hidden[-1]).reshape(-1, 3, 3)


def percentile_scales(inputs, targets):
    def percentile(a):
        value = np.percentile(np.abs(a), 95)
        return float(value) if value > 0 else 1.0
    scales = [percentile(inputs[..., :3]), percentile(inputs[..., 3:6]), percentile(inputs[..., 6])]
    return np.asarray([scales[0]] * 3 + [scales[1]] * 3 + [scales[2]], np.float32), percentile(targets)


def line_segment(displacements, epsilon_line, epsilon_pos, *, validate=True):
    a = np.asarray(displacements, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(a).all():
        raise ValueError("non-finite predicted positions")
    nonzero = np.flatnonzero(np.linalg.norm(a, axis=1) > epsilon_pos)
    if not len(nonzero):
        return np.zeros(3), np.zeros(3), float(np.linalg.norm(a, axis=1).max())
    u = a[nonzero[0]] / np.linalg.norm(a[nonzero[0]])
    scalar = a @ u
    residual = float(np.linalg.norm(a - scalar[:, None] * u, axis=1).max())
    low, high = min(0.0, float(scalar.min())), max(0.0, float(scalar.max()))
    outside = np.maximum(low - scalar, scalar - high).max()
    if validate and (residual > epsilon_line or outside > epsilon_pos):
        raise ValueError(f"LSTM prediction failed straight-line acceptance: residual={residual:.6g}")
    return low * u, high * u, residual


def load_obstacle_feature_scale(checkpoint):
    """Read geometry normalization without constructing a temporal predictor.

    Static scenes do not use prediction horizons; dynamic inference still
    validates the checkpoint horizon in ObstaclePredictor.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1 or not state.get("accepted", False):
        raise ValueError("obstacle normalization requires an accepted pretrained checkpoint")
    scale = np.asarray(state["feature_scale"], np.float32)
    if scale.shape != (37,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("invalid frozen obstacle feature normalization")
    return scale


class ObstaclePredictor:
    FORMAT_VERSION = 1

    def __init__(self, checkpoint):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Pretrain obstacle LSTM first: checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format_version") != self.FORMAT_VERSION or not state.get("accepted", False):
            raise ValueError("obstacle predictor requires an accepted pretrained checkpoint")
        if not np.isclose(state["prediction_dt"], PREDICTION_DT):
            raise ValueError("predictor/control horizon mismatch")
        self.model = PositionLSTM().eval()
        self.model.load_state_dict(state["model"])
        self.model.requires_grad_(False)
        self.input_scale = np.asarray(state["input_scale"], np.float32)
        self.output_scale = float(state["output_scale"])
        self.feature_scale = np.asarray(state["feature_scale"], np.float32)
        self.epsilon_line = float(state["acceptance"]["epsilon_line"])
        self.epsilon_pos = float(state["acceptance"]["epsilon_pos"])
        if (self.input_scale.shape != (7,) or self.feature_scale.shape != (37,)
                or not np.isfinite(self.input_scale).all() or not np.isfinite(self.feature_scale).all()
                or np.any(self.input_scale <= 0) or np.any(self.feature_scale <= 0)
                or not np.isfinite(self.output_scale) or self.output_scale <= 0
                or not np.isfinite([self.epsilon_line, self.epsilon_pos]).all()
                or self.epsilon_line <= 0 or self.epsilon_pos <= 0):
            raise ValueError("invalid frozen training normalization")
        self.histories = {}

    def reset(self):
        self.histories.clear()

    def predict(self, records):
        """Records contain only identity, timestamp, current pose and geometry."""
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("duplicate obstacle identities in one observation")
        valid = {r["id"] for r in records if r.get("valid", True)
                 and np.asarray(r["position"]).shape == (3,) and np.isfinite(r["position"]).all()}
        for identity in set(self.histories) - valid:
            del self.histories[identity]
        if not valid:
            return {}
        predictions, windows, pending = {}, [], []
        for record in records:
            identity = record["id"]
            if identity not in valid:
                continue
            timestamp = float(record["timestamp"])
            position = np.asarray(record["position"], dtype=np.float64)
            if not np.isfinite(timestamp):
                self.histories.pop(identity, None)
                continue
            history = self.histories.setdefault(identity, deque(maxlen=10))
            if history and timestamp < history[-1][0]:
                history.clear()
            if not history or timestamp > history[-1][0]:
                history.append((timestamp, position.copy()))
            elif not np.array_equal(position, history[-1][1]):
                raise ValueError("same obstacle timestamp has inconsistent positions")
            if len(history) == 1:
                predictions[identity] = np.repeat(position[None], 3, axis=0)
            elif len(history) < 10:
                velocity = (history[-1][1] - history[-2][1]) / (history[-1][0] - history[-2][0])
                predictions[identity] = position + HORIZONS[:, None] * velocity
            else:
                windows.append(history_features(history) / self.input_scale)
                pending.append((identity, position))
        if windows:
            with torch.inference_mode():
                output = self.model(torch.as_tensor(np.stack(windows))).numpy() * self.output_scale
            for (identity, position), displacement in zip(pending, output):
                line_segment(displacement, self.epsilon_line, self.epsilon_pos)
                predictions[identity] = position + displacement
        return predictions

    def state_dict(self):
        return {identity: list(history) for identity, history in self.histories.items()}

    def load_state_dict(self, state):
        self.histories = {identity: deque(history, maxlen=10) for identity, history in state.items()}
