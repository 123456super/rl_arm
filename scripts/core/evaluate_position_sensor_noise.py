#!/usr/bin/env python3
"""Run paired frozen-policy position-noise sweeps; save metrics and plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    "policy", "training_seed", "test_seed", "level_index", "position_noise_std_mm",
    "episodes", "success_rate", "collision_rate", "timeout_rate",
    "mean_episode_steps", "p95_position_error_mm", "p95_orientation_error_rad",
    "position_tolerance_mm", "orientation_tolerance_rad", "checkpoint_sha256",
)


def write_summary(output: Path, rows: list[dict]) -> None:
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(output: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = sorted({row["level_index"] for row in rows})
    fig, axes = plt.subplots(len(levels), 2, figsize=(10, 3.5 * len(levels)), squeeze=False)
    for index, level in enumerate(levels):
        selected = [row for row in rows if row["level_index"] == level]
        for policy_index, policy in enumerate(sorted({row["policy"] for row in selected})):
            color = f"C{policy_index % 10}"
            group = [row for row in selected if row["policy"] == policy]
            sigmas = sorted({row["position_noise_std_mm"] for row in group})
            success, error = [], []
            for sigma in sigmas:
                repeats = [row for row in group if row["position_noise_std_mm"] == sigma]
                success.append(100 * sum(row["success_rate"] for row in repeats) / len(repeats))
                error.append(sum(row["p95_position_error_mm"] for row in repeats) / len(repeats))
                if len(repeats) > 1:
                    axes[index, 0].scatter([sigma] * len(repeats),
                                           [100 * row["success_rate"] for row in repeats],
                                           color=color, alpha=.35)
            axes[index, 0].plot(sigmas, success, "o-", label=policy, color=color)
            axes[index, 1].plot(sigmas, error, "o-", label=policy, color=color)
        for axis in axes[index]:
            axis.set_xlabel("Position measurement noise: per-axis std (mm)")
            axis.grid(alpha=.25)
            axis.legend()
        axes[index, 0].set_ylim(0, 102)
        axes[index, 0].set_ylabel("True coupled-pose success (%)")
        axes[index, 1].set_ylabel("Terminal position error P95 (mm)")
        tolerance = selected[0]["position_tolerance_mm"]
        orientation = selected[0]["orientation_tolerance_rad"]
        axes[index, 1].axhline(tolerance, ls="--", c="gray", label="position tolerance")
        axes[index, 1].legend()
        axes[index, 0].set_title(f"L{level}: {tolerance:g} mm / {orientation:g} rad")
        axes[index, 1].set_title(f"L{level}: all episodes, including failures")
    fig.suptitle("Frozen-policy position sensor noise evaluation")
    fig.tight_layout()
    fig.savefig(output / "robustness.png", dpi=180)
    fig.savefig(output / "robustness.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", help="unique labels matching checkpoint order")
    parser.add_argument("--config", help="optional config override; default uses checkpoint config")
    parser.add_argument("--levels", nargs="+", type=int, default=[6, 13])
    parser.add_argument("--noise-std-mm", nargs="+", type=float, default=[0, .1, .5, 1, 2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[41001, 51001, 61001])
    parser.add_argument("--sensor-noise-seed", type=int, default=71001)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--output-dir", required=True, help="new directory, never overwrite")
    args = parser.parse_args()
    import math
    if args.episodes < 1 or args.num_envs < 1:
        parser.error("episodes and num-envs must be positive")
    if any(not math.isfinite(value) or value < 0 for value in args.noise_std_mm):
        parser.error("noise standard deviations must be finite and nonnegative")
    if any(seed < 0 for seed in args.seeds) or args.sensor_noise_seed < 0:
        parser.error("seeds must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or any(
        abs(a - b) < args.episodes
        for i, a in enumerate(args.seeds) for b in args.seeds[i + 1:]
    ):
        parser.error("test seed blocks must not overlap: separate starts by at least episodes")
    if len(set(args.noise_std_mm)) != len(args.noise_std_mm):
        parser.error("noise levels must be unique")
    if len(set(args.levels)) != len(args.levels):
        parser.error("evaluation levels must be unique")
    labels = args.labels or [f"policy_{i + 1}" for i in range(len(args.checkpoint))]
    if len(labels) != len(args.checkpoint) or len(set(labels)) != len(labels):
        parser.error("labels must be unique and match checkpoints")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(json.dumps({
        **vars(args), "labels": labels,
        "protocol": "paired_position_noise_v1",
        "noise_model": "world-frame Gaussian, per-axis std, independent per control sample",
        "task_pairing": "same reset seed and task-space bin at every noise level",
        "noise_pairing": "same standard-normal draw keyed by noise seed, reset seed and step",
        "precision_scaling": "measured position error and exact orientation error",
        "evaluation": "ground truth, original horizon and continuous-success contract",
        "seed_warning": "test-seed repeats are not independent training seeds",
    }, indent=2) + "\n")
    rows = []
    total = len(labels) * len(args.levels) * len(args.noise_std_mm) * len(args.seeds)
    completed = 0
    for policy_index, (label, checkpoint) in enumerate(zip(labels, args.checkpoint)):
        for level in args.levels:
            for sigma in args.noise_std_mm:
                for seed in args.seeds:
                    stem = f"policy{policy_index + 1}_L{level:02d}_sigma{sigma:g}_seed{seed}"
                    result = output / f"{stem}.json"
                    command = [
                        sys.executable, str(ROOT / "scripts/core/evaluate_thesis_homotopy.py"),
                        "--checkpoint", checkpoint, "--level-index", str(level),
                        "--episodes", str(args.episodes), "--seed", str(seed),
                        "--num-envs", str(args.num_envs), "--no-cpu-affinity",
                        "--position-noise-std-mm", str(sigma),
                        "--sensor-noise-seed", str(args.sensor_noise_seed),
                        "--output", str(result),
                    ]
                    command += ["--config", args.config] if args.config else ["--checkpoint-config"]
                    print(f"[{completed + 1}/{total}] {label} L{level} sigma={sigma:g} mm seed={seed}", flush=True)
                    with (output / f"{stem}.log").open("w") as log:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    metrics = json.loads(result.read_text())
                    row = {
                        "policy": label, "training_seed": metrics["training_seed"],
                        "test_seed": seed, "level_index": level,
                        "position_noise_std_mm": sigma, "episodes": args.episodes,
                        "p95_position_error_mm": 1000 * metrics["p95_position_error_m"],
                        "position_tolerance_mm": 1000 * metrics["position_tolerance_m"],
                        **{key: metrics[key] for key in (
                            "success_rate", "collision_rate", "timeout_rate", "mean_episode_steps",
                            "p95_orientation_error_rad", "orientation_tolerance_rad", "checkpoint_sha256",
                        )},
                    }
                    rows.append(row)
                    write_summary(output, rows)
                    completed += 1
                    print(f"  success={100 * row['success_rate']:.1f}% position P95={row['p95_position_error_mm']:.3f} mm", flush=True)
    plot_summary(output, rows)
    print(f"Saved {len(rows)} conditions to {output}", flush=True)


if __name__ == "__main__":
    main()
