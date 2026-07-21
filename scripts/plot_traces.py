from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode", default=None)
    return parser.parse_args()


def select_trace(trace_dir: Path, episode: str | None) -> Path:
    if episode is not None:
        episode_index = int(episode)
        path = trace_dir / f"episode_{episode_index:04d}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    traces = sorted(trace_dir.glob("episode_*.csv"))
    if not traces:
        raise FileNotFoundError(f"No episode_*.csv files found in {trace_dir}")
    return traces[0]


def save_risk_plot(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
    axes[0].plot(frame["time"], frame["d_min"], label="d_min")
    axes[0].set_ylabel("distance (m)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(frame["time"], frame["risk_global"], color="tab:red", label="Risk_global")
    axes[1].set_ylabel("risk")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(frame["time"], frame["beta"], color="tab:green", label="beta")
    axes[2].set_ylabel("beta")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(True, alpha=0.25)

    for axis in axes:
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_motion_plot(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
    axes[0].plot(frame["time"], frame["qdot_norm"], label="||q_dot_cmd||")
    axes[0].set_ylabel("velocity")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(frame["time"], frame["acc_norm"], color="tab:orange", label="||acc||")
    axes[1].set_ylabel("acceleration")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(frame["time"], frame["jerk_norm"], color="tab:purple", label="||jerk||")
    axes[2].set_ylabel("jerk")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(True, alpha=0.25)

    for axis in axes:
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    trace_path = select_trace(Path(args.trace_dir), args.episode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(trace_path)
    stem = trace_path.stem

    save_risk_plot(frame, output_dir / f"{stem}_risk_beta.png")
    save_motion_plot(frame, output_dir / f"{stem}_motion_smoothness.png")
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
