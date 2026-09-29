#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, .95)),
        "max": float(np.max(array)),
    }


def _empty_block(start: int, end: int) -> dict[str, Any]:
    return {
        "start_exclusive": start,
        "end_inclusive": end,
        "distances": [],
        "next_distances": [],
        "qualities": [],
        "precision_qualities": [],
        "precision_rewards": [],
        "progress": [],
        "clips": [],
        "episodes": {},
        "policy_ratios": [],
        "policy_churn_kl": [],
        "pcr_coefficients": [],
        "pcr_sac_loss_ema": [],
        "pcr_loss_ema": [],
    }


def _finalize(block: dict[str, Any]) -> dict[str, Any]:
    progress = np.asarray(block["progress"], dtype=np.float64)
    episodes = block["episodes"]
    episode_reductions = np.asarray(
        [values[0] - values[1] for values in episodes.values()], dtype=np.float64,
    )
    return {
        "start_step_exclusive": block["start_exclusive"],
        "end_step_inclusive": block["end_inclusive"],
        "transitions": len(progress),
        "keypoint_distance_mean_m": _summary(block["distances"])["mean"],
        "keypoint_next_distance_mean_m": _summary(block["next_distances"])["mean"],
        "keypoint_tracking_quality_mean": _summary(block["qualities"])["mean"],
        "keypoint_precision_quality_mean": _summary(
            block["precision_qualities"]
        )["mean"],
        "keypoint_precision_reward_mean": _summary(
            block["precision_rewards"]
        )["mean"],
        "keypoint_progress_mean_m": (
            None if not len(progress) else float(np.mean(progress))
        ),
        "keypoint_progress_positive_ratio": (
            None if not len(progress) else float(np.mean(progress > 0.0))
        ),
        "jacobian_clip_ratio_mean": _summary(block["clips"])["mean"],
        "episodes_observed": len(episodes),
        "episode_keypoint_distance_reduction_mean_m": (
            None if not len(episode_reductions) else float(np.mean(episode_reductions))
        ),
        "episode_keypoint_improvement_ratio": (
            None if not len(episode_reductions)
            else float(np.mean(episode_reductions > 0.0))
        ),
        "policy_churn_to_sac_ratio": _summary(block["policy_ratios"]),
        "policy_churn_kl": _summary(block["policy_churn_kl"]),
        "chain_pcr_effective_coefficient": _summary(block["pcr_coefficients"]),
        "chain_pcr_sac_loss_ema": _summary(block["pcr_sac_loss_ema"]),
        "chain_pcr_loss_ema": _summary(block["pcr_loss_ema"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize keypoint and CHAIN metrics in fixed training windows."
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", required=True)
    parser.add_argument("--window-steps", type=int, default=25_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoints = sorted(set(args.checkpoint_steps))
    if checkpoints[0] < 1 or args.window_steps < 1:
        parser.error("checkpoint and window steps must be positive")
    maximum = checkpoints[-1]
    boundaries = list(range(args.window_steps, maximum + 1, args.window_steps))
    if not boundaries or boundaries[-1] != maximum:
        boundaries.append(maximum)
    blocks = {
        end: _empty_block(max(0, end - args.window_steps), end)
        for end in boundaries
    }

    run = _path(args.run)
    with (run / "transitions.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                step = int(row["stage_total_step"])
            except (KeyError, TypeError, ValueError):
                continue
            if step > maximum:
                break
            end = min(((step - 1) // args.window_steps + 1) * args.window_steps, maximum)
            block = blocks.get(end)
            if block is None or step <= block["start_exclusive"]:
                continue
            try:
                distance = float(row["keypoint_distance"])
                next_distance = float(row["next_keypoint_distance"])
                quality = float(row["keypoint_tracking_quality"])
                progress = float(row["keypoint_progress"])
                clip = float(row["jacobian_clip_ratio"])
                episode = int(row["episode"])
            except (KeyError, TypeError, ValueError):
                continue
            block["distances"].append(distance)
            block["next_distances"].append(next_distance)
            block["qualities"].append(quality)
            for target, source in (
                ("precision_qualities", "keypoint_precision_quality"),
                ("precision_rewards", "keypoint_precision_reward"),
            ):
                value = row.get(source, "")
                if value != "":
                    block[target].append(float(value))
            block["progress"].append(progress)
            block["clips"].append(clip)
            if episode not in block["episodes"]:
                block["episodes"][episode] = [distance, next_distance]
            else:
                block["episodes"][episode][1] = next_distance

    with (run / "updates.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                step = int(row["stage_total_step"])
            except (KeyError, TypeError, ValueError):
                continue
            if step > maximum:
                break
            end = min(((step - 1) // args.window_steps + 1) * args.window_steps, maximum)
            block = blocks.get(end)
            if block is None or step <= block["start_exclusive"]:
                continue
            for target, source in (
                ("policy_ratios", "policy_churn_to_sac_ratio"),
                ("policy_churn_kl", "policy_churn_kl"),
                ("pcr_coefficients", "chain_pcr_effective_coefficient"),
                ("pcr_sac_loss_ema", "chain_pcr_sac_loss_ema"),
                ("pcr_loss_ema", "chain_pcr_loss_ema"),
            ):
                value = row.get(source, "")
                if value != "":
                    block[target].append(float(value))

    windows = [_finalize(blocks[end]) for end in boundaries]
    selected = {
        str(step): next(window for window in windows if window["end_step_inclusive"] == step)
        for step in checkpoints
    }
    distance_means = np.asarray([
        window["keypoint_next_distance_mean_m"] for window in windows
    ], dtype=np.float64)
    quality_means = np.asarray([
        window["keypoint_tracking_quality_mean"] for window in windows
    ], dtype=np.float64)
    x = np.asarray([window["end_step_inclusive"] for window in windows], dtype=np.float64)
    result = {
        "run": str(run.relative_to(ROOT) if run.is_relative_to(ROOT) else run),
        "window_steps": args.window_steps,
        "checkpoint_steps": checkpoints,
        "windows": windows,
        "checkpoint_windows": selected,
        "trend": {
            "keypoint_next_distance_slope_per_100k": float(
                np.polyfit(x, distance_means, 1)[0] * 100_000
            ) if len(x) > 1 else None,
            "keypoint_tracking_quality_slope_per_100k": float(
                np.polyfit(x, quality_means, 1)[0] * 100_000
            ) if len(x) > 1 else None,
        },
    }
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint_windows": selected, "trend": result["trend"]}, indent=2))


if __name__ == "__main__":
    main()
