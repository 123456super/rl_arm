#!/usr/bin/env python3
"""Generate timestamped simulation supervision and train the frozen Full LSTM."""
from __future__ import annotations

import argparse
from collections import deque
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pybullet as p
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.envs.obstacle_scene import create_body, raw_record, sample_obstacle
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.convex_sweep import sweep_features
from rl_risk_sac.tasks.obstacle_prediction import (
    PREDICTION_DT, PositionLSTM, history_features, line_segment, percentile_scales,
)
from rl_risk_sac.utils.config import load_config


def generate_dataset(config, *, episodes, steps, intervals, seed):
    """Advance every 240 Hz step; requests and pending labels are separate events."""
    intervals = tuple(int(interval) for interval in intervals)
    if (episodes < 10 or not intervals or min(intervals) < 1
            or steps < 10 * max(intervals) + 24):
        raise ValueError("need >=10 episodes, positive request intervals and sufficient physical steps")
    rng = np.random.default_rng(seed)
    order = rng.permutation(episodes)
    split = {int(e): (0 if i < int(.8 * episodes) else 1 if i < int(.9 * episodes) else 2)
             for i, e in enumerate(order)}
    env_config = copy.deepcopy(config)
    env_config["thesis"]["obstacles"]["enabled"] = False
    env = ThesisHomotopyEnv(env_config)
    if not np.isclose(env.physics_dt, 1 / 240):
        env.close()
        raise ValueError("offline labels require the configured 240 Hz physics clock")
    xs, ys, sources, feature_rows, metadata = [], [], [], [], []
    settings = config["thesis"]["obstacles"]
    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            base, rotation = env._base_frame()
            capsules = env.robot.capsules()
            # Geometry-scale fitting uses training episodes only, in the same base frame.
            for capsule in capsules:
                capsule.start = rotation.T @ (capsule.start - base)
                capsule.end = rotation.T @ (capsule.end - base)
            objects = [sample_obstacle(rng, i, settings["workspace_low"], settings["workspace_high"])
                       for i in range(int(rng.choice(4, p=settings["count_probabilities"]))) ]
            for obj in objects:
                create_body(obj, env.client_id)
            metadata.append(dict(episode=episode, split=split[episode],
                                 obstacles=[o.metadata() for o in objects],
                                 request_intervals_steps=list(intervals)))
            histories = {o.identity: deque(maxlen=10) for o in objects}
            window_counts = {o.identity: 0 for o in objects}
            pending = {}
            request_step, request_index = 0, 0
            for step in range(steps):
                time = step * env.physics_dt
                for obj in objects:
                    p.resetBasePositionAndOrientation(obj.body_id, obj.position(time), obj.quaternion,
                                                     physicsClientId=env.client_id)
                p.stepSimulation(physicsClientId=env.client_id)
                records = [raw_record(o, time, env.client_id, base, rotation) for o in objects]
                by_id = {r["id"]: r for r in records}
                if step == request_step:
                    for record in records:
                        history = histories[record["id"]]
                        history.append((time, record["position"].copy()))
                        # Once the history is full, every subsequent successful
                        # request creates a new overlapping window.  Using == 10
                        # here silently limited each object to one training
                        # sample because deque(maxlen=10) never grows beyond 10.
                        if len(history) >= 10 and step + 24 < steps:
                            sample = dict(x=history_features(history), record=record, y=np.zeros((3, 3)), episode=episode)
                            window_counts[record["id"]] += 1
                            for horizon, offset in enumerate((12, 24, 36)):
                                pending.setdefault(step + offset, []).append((sample, horizon))
                    request_step += intervals[request_index % len(intervals)]
                    request_index += 1
                for sample, horizon in pending.pop(step, []):
                    record = sample["record"]
                    sample["y"][horizon] = by_id[record["id"]]["position"] - record["position"]
                    if horizon == 2:
                        xs.append(sample["x"])
                        ys.append(sample["y"].copy())
                        sources.append((episode, record["id"], record["timestamp"], split[episode]))
                        if split[episode] == 0:
                            future = {record["id"]: record["position"] + sample["y"]}
                            features, _ = sweep_features([record], future, capsules, 1e-6, 1e-8)
                            feature_rows.extend(features)
            metadata[-1]["window_counts"] = window_counts
            for obj in objects:
                p.removeBody(obj.body_id, physicsClientId=env.client_id)
    finally:
        env.close()
    if not xs or any(not any(int(s[3]) == split_id for s in sources) for split_id in range(3)):
        raise ValueError("empty split: increase episode count or use a different seed")
    if not feature_rows:
        raise ValueError("no training-episode geometry features were generated; increase episodes or obstacle count")
    features = np.asarray(feature_rows)
    feature_scale = np.percentile(np.abs(features), 95, axis=0).astype(np.float32)
    feature_scale[feature_scale == 0] = 1
    return dict(inputs=np.asarray(xs, np.float32), targets=np.asarray(ys, np.float32),
                sources=np.asarray(sources), feature_scale=feature_scale,
                metadata=json.dumps(metadata), intervals=np.asarray(intervals), physics_dt=1 / 240)


def prediction_metrics(prediction, targets, epsilon_pos=.001):
    error = np.linalg.norm(prediction - targets, axis=-1)
    residual = [line_segment(row, np.inf, epsilon_pos, validate=False)[2] for row in prediction]
    return dict(ade_m=float(error.mean()), fde_m=float(error[:, -1].mean()),
                rmse_m=float(np.sqrt(np.square(prediction - targets).mean())),
                horizon_error_m=error.mean(0).tolist(), max_line_residual_m=float(max(residual)))


def train_predictor(data, *, epochs, batch_size, learning_rate, seed, acceptance):
    torch.manual_seed(seed)
    train, val, test = [data["sources"][:, 3] == i for i in range(3)]
    input_scale, output_scale = percentile_scales(data["inputs"][train], data["targets"][train])
    x = torch.tensor(data["inputs"] / input_scale)
    y = torch.tensor(data["targets"])
    model = PositionLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    indices = torch.tensor(np.flatnonzero(train))
    best, best_loss = None, float("inf")
    history = []
    def predict(mask):
        with torch.inference_mode():
            return torch.cat([model(part) * output_scale for part in x[mask].split(batch_size)]).numpy()
    for epoch in range(epochs):
        model.train()
        total = 0.
        for batch in indices[torch.randperm(len(indices))].split(batch_size):
            prediction = model(x[batch]) * output_scale
            loss = (prediction - y[batch]).square().sum() / (3 * len(batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        model.eval()
        metrics = prediction_metrics(predict(val), data["targets"][val], acceptance["epsilon_pos"])
        history.append(dict(epoch=epoch + 1, train_loss_m2=total / len(indices), **metrics))
        if metrics["ade_m"] < best_loss:
            best_loss = metrics["ade_m"]
            best = copy.deepcopy(model.state_dict())
    if best is None:
        raise ValueError("epochs must be positive")
    model.load_state_dict(best)
    model.eval()
    metrics = {name: prediction_metrics(predict(mask), data["targets"][mask], acceptance["epsilon_pos"])
               for name, mask in (("validation", val), ("test", test))}
    # Test is reported once after validation selects weights; it never selects weights/scales.
    accepted = all(m["ade_m"] <= acceptance["max_ade_m"] and m["fde_m"] <= acceptance["max_fde_m"]
                   and m["max_line_residual_m"] <= acceptance["epsilon_line"] for m in metrics.values())
    return dict(format_version=1, accepted=accepted, prediction_dt=PREDICTION_DT,
                model=best, input_scale=input_scale, output_scale=output_scale,
                feature_scale=data["feature_scale"], acceptance=acceptance, metrics=metrics,
                training_history=history, request_intervals_steps=data["intervals"].tolist(),
                physics_dt=float(data["physics_dt"]), seed=seed,
                split_episode_ids={name: np.unique(data["sources"][mask, 0]).astype(int).tolist()
                                   for name, mask in (("train", train), ("validation", val), ("test", test))},
                feature_scale_source="training episodes: true convex sweeps and sampled robot configurations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/paper_full.yaml")
    parser.add_argument("--output", default="outputs/obstacle_predictor")
    parser.add_argument("--dataset", help="Reuse an existing generated .npz dataset")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--physics-steps", type=int, default=1200)
    parser.add_argument("--request-intervals", type=int, nargs="+", default=[8, 8, 12, 4, 8, 16, 8, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epsilon-line", type=float, default=.005)
    parser.add_argument("--epsilon-pos", type=float, default=.001)
    parser.add_argument("--max-ade", type=float, default=.03)
    parser.add_argument("--max-fde", type=float, default=.05)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.dataset:
        with np.load(args.dataset, allow_pickle=False) as archive:
            data = dict(archive)
    else:
        data = generate_dataset(load_config(args.config), episodes=args.episodes, steps=args.physics_steps,
                                intervals=args.request_intervals, seed=args.seed)
        np.savez_compressed(output / "dataset.npz", **data)
    acceptance = dict(epsilon_line=args.epsilon_line, epsilon_pos=args.epsilon_pos,
                      max_ade_m=args.max_ade, max_fde_m=args.max_fde)
    if any(not np.isfinite(v) or v <= 0 for v in acceptance.values()):
        raise ValueError("acceptance tolerances must be finite and positive")
    state = train_predictor(data, epochs=args.epochs, batch_size=args.batch_size,
                            learning_rate=args.learning_rate, seed=args.seed, acceptance=acceptance)
    torch.save(state, output / "predictor.pt")
    report = {k: v for k, v in state.items() if k not in {"model", "input_scale", "feature_scale"}}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(output / "predictor.pt"), "accepted": state["accepted"],
                      "metrics": state["metrics"]}, indent=2))
    return 0 if state["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
