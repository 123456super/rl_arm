#!/usr/bin/env python3
"""Train the S2 obstacle LSTM on independent smooth-motion trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.prediction.obstacle_lstm import (  # noqa: E402
    CHECKPOINT_FORMAT,
    ObstacleMotionLSTM,
)
from rl_risk_sac.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/paper_s2_lstm_sweep_single.yaml"
    )
    parser.add_argument("--output-dir", help="Optional operational output override")
    parser.add_argument("--device", help="Optional operational device override")
    return parser.parse_args()


def make_dataset(
    sample_count: int,
    seed: int,
    history_times: np.ndarray,
    horizon_times: np.ndarray,
    obstacle_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one causal window per independently sampled sinusoid."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(sample_count, 3)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # Matches and slightly surrounds the amplitudes produced by the S2 sampler.
    amplitudes = rng.uniform(0.30, 0.65, size=sample_count).astype(np.float32)
    phases = rng.uniform(-math.pi, math.pi, size=sample_count).astype(np.float32)
    frequencies = obstacle_speed / amplitudes
    all_times = np.concatenate([history_times, horizon_times]).astype(np.float32)
    angles = phases[:, None] + frequencies[:, None] * all_times[None, :]
    scalar_positions = amplitudes[:, None] * np.cos(angles)
    positions = scalar_positions[:, :, None] * directions[:, None, :]
    history = positions[:, : len(history_times)]
    current = history[:, -1:, :]
    relative = history - current
    intervals = np.diff(history_times)
    velocities = np.empty_like(history)
    velocities[:, 1:] = np.diff(history, axis=1) / intervals[None, :, None]
    velocities[:, 0] = velocities[:, 1]
    ages = history_times[-1] - history_times
    age_features = np.broadcast_to(
        ages[None, :, None], (sample_count, len(history_times), 1)
    )
    features = np.concatenate([relative, velocities, age_features], axis=2)
    future = positions[:, len(history_times) :]
    targets = future - current
    return features.astype(np.float32), targets.astype(np.float32)


def displacement_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    errors = np.linalg.norm(prediction - target, axis=2)
    final = errors[:, -1]
    return {
        "ade_m": float(errors.mean()),
        "fde_m": float(final.mean()),
        "fde_p95_m": float(np.quantile(final, 0.95)),
        "fde_max_m": float(final.max()),
    }


def main() -> None:
    args = parse_args()
    config = load_config(ROOT / args.config)
    thesis = config["thesis"]
    prediction = thesis["dynamic_prediction"]
    training = prediction["training"]
    seed = int(config["seed"])
    train_samples = int(training["train_samples"])
    validation_samples = int(training["validation_samples"])
    test_samples = int(training["test_samples"])
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    learning_rate = float(training["learning_rate"])
    hidden_size = int(training["hidden_size"])
    num_layers = int(training["num_layers"])
    history_length = int(prediction["history_length"])
    history_hz = float(prediction["history_sample_hz"])
    control_dt = float(thesis["control_dt"])
    obstacle_speed = float(thesis["obstacle_speed"])
    device_name = args.device or str(training["device"])
    output_name = args.output_dir or str(training["output_dir"])
    if min(train_samples, validation_samples, test_samples) < 1:
        raise ValueError("all dataset splits must be non-empty")
    torch.manual_seed(seed)
    np.random.seed(seed)
    output = (ROOT / output_name).resolve()
    output.mkdir(parents=True, exist_ok=True)
    history_times = (
        np.arange(-(history_length - 1), 1, dtype=np.float32) / history_hz
    )
    horizon_times = (
        np.asarray([1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float32)
        * control_dt
    )
    train_x, train_y = make_dataset(
        train_samples, seed, history_times, horizon_times, obstacle_speed
    )
    validation_x, validation_y = make_dataset(
        validation_samples,
        seed + 1,
        history_times,
        horizon_times,
        obstacle_speed,
    )
    test_x, test_y = make_dataset(
        test_samples,
        seed + 2,
        history_times,
        horizon_times,
        obstacle_speed,
    )
    # Scalar scales per physical quantity retain rotational symmetry.
    input_scale = np.asarray(
        [
            max(float(np.quantile(np.abs(train_x[:, :, :3]), 0.995)), 1.0e-4),
        ] * 3
        + [
            max(float(np.quantile(np.abs(train_x[:, :, 3:6]), 0.995)), 1.0e-4),
        ] * 3
        + [max(float(np.max(train_x[:, :, 6])), 1.0e-4)],
        dtype=np.float32,
    )
    output_scale = max(float(np.quantile(np.abs(train_y), 0.995)), 1.0e-4)
    train_x /= input_scale
    validation_x /= input_scale
    test_x /= input_scale
    train_y /= output_scale
    validation_y /= output_scale
    test_y /= output_scale
    target_device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model_config = {
        "input_size": 7,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "horizon_steps": len(horizon_times),
    }
    model = ObstacleMotionLSTM(**model_config).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=target_device.type == "cuda",
    )
    validation_inputs = torch.from_numpy(validation_x).to(target_device)
    validation_targets = torch.from_numpy(validation_y).to(target_device)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for features, targets in loader:
            features = features.to(target_device, non_blocking=True)
            targets = targets.to(target_device, non_blocking=True)
            loss = torch.mean((model(features) - targets) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(features)
            total_count += len(features)
        model.eval()
        with torch.inference_mode():
            validation_loss = float(torch.mean(
                (model(validation_inputs) - validation_targets) ** 2
            ))
        if validation_loss < best_loss - 1.0e-8:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch}/{epochs} train={total_loss / total_count:.8f} "
                f"validation={validation_loss:.8f} best={best_loss:.8f}",
                flush=True,
            )
        if stale_epochs >= 12:
            print(f"early stop at epoch {epoch}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("LSTM training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        validation_prediction = (
            model(validation_inputs).cpu().numpy() * output_scale
        )
        test_prediction = (
            model(torch.from_numpy(test_x).to(target_device)).cpu().numpy()
            * output_scale
        )
    validation_metrics = displacement_metrics(
        validation_prediction, validation_y * output_scale
    )
    test_metrics = displacement_metrics(test_prediction, test_y * output_scale)
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "model_config": model_config,
        "model_state_dict": best_state,
        "history_length": history_length,
        "history_hz": history_hz,
        "horizon_times_s": horizon_times.tolist(),
        "input_scale": input_scale.tolist(),
        "output_scale": output_scale,
        "training_seed": seed,
        "training_samples": train_samples,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    torch.save(checkpoint, output / "best.pt")
    report = {
        "checkpoint": str(output / "best.pt"),
        "device": str(target_device),
        "validation": validation_metrics,
        "test": test_metrics,
        "input_scale": input_scale.tolist(),
        "output_scale": output_scale,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
