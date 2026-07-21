from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


GROUP_COLUMNS = ["scenario", "method"]
METRIC_COLUMNS = [
    "success",
    "collision",
    "non_end_link_collision",
    "final_position_error",
    "completion_time",
    "min_distance",
    "mean_cost_per_step",
    "mean_risk",
    "max_risk",
    "safety_violation_rate",
    "mean_action_variation",
    "rms_acceleration",
    "rms_jerk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="outputs/formal/eval")
    parser.add_argument("--train-root", default="outputs/formal/train")
    parser.add_argument("--output-dir", default="outputs/formal/summary")
    return parser.parse_args()


def load_eval_rows(eval_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(eval_root.glob("*/*/train_seed_*/eval_seed_*.csv")):
        scenario, method, train_seed_dir, eval_seed_file = path.parts[-4:]
        frame = pd.read_csv(path)
        frame.insert(0, "scenario", scenario)
        frame.insert(1, "method", method)
        frame.insert(2, "train_seed", train_seed_dir.replace("train_seed_", ""))
        frame.insert(3, "eval_seed", eval_seed_file.removesuffix(".csv").replace("eval_seed_", ""))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_eval(frame: pd.DataFrame) -> pd.DataFrame:
    available_metrics = [column for column in METRIC_COLUMNS if column in frame.columns]
    link_metric_prefixes = ("min_distance_", "mean_risk_", "closest_steps_", "violation_steps_")
    available_metrics.extend(
        column
        for column in frame.columns
        if column.startswith(link_metric_prefixes) and column not in available_metrics
    )
    summary = frame.groupby(GROUP_COLUMNS)[available_metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def load_train_metrics(train_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(train_root.glob("*/seed_*/*/train_metrics.csv")):
        method = path.parts[-4]
        seed = path.parts[-3].replace("seed_", "")
        frame = pd.read_csv(path)
        frame.insert(0, "method", method)
        frame.insert(1, "seed", seed)
        if "episode_cost_mean" not in frame.columns and {"episode_cost", "episode_length"}.issubset(frame.columns):
            frame["episode_cost_mean"] = frame["episode_cost"] / frame["episode_length"].clip(lower=1)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_training_curves(train_frame: pd.DataFrame, output_dir: Path) -> None:
    if train_frame.empty:
        return
    curve_specs = [
        ("episode_reward", "training_reward.png"),
        ("episode_cost_mean", "training_mean_cost.png"),
        ("safety_violation_rate", "training_safety_violation.png"),
        ("lambda", "training_lambda.png"),
    ]
    for metric, filename in curve_specs:
        if metric not in train_frame.columns:
            continue
        plt.figure(figsize=(8, 4.5))
        for method, method_frame in train_frame.groupby("method"):
            curve = method_frame.groupby("episode")[metric].mean().rolling(10, min_periods=1).mean()
            plt.plot(curve.index, curve.values, label=method)
        plt.xlabel("Episode")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_frame = load_eval_rows(Path(args.eval_root))
    train_frame = load_train_metrics(Path(args.train_root))

    manifest = {
        "eval_root": args.eval_root,
        "train_root": args.train_root,
        "eval_files": int(eval_frame["eval_seed"].count()) if not eval_frame.empty else 0,
        "train_rows": int(len(train_frame)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not eval_frame.empty:
        eval_frame.to_csv(output_dir / "all_eval_episodes.csv", index=False)
        summarize_eval(eval_frame).to_csv(output_dir / "eval_summary_mean_std.csv", index=False)

    if not train_frame.empty:
        train_frame.to_csv(output_dir / "all_train_metrics.csv", index=False)
        plot_training_curves(train_frame, output_dir)

    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
