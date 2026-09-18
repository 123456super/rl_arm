#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_risk_sac.utils.config import load_config


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing {key!r} in plateau input")
    return float(value)


def _rate(rows: list[dict[str, str]], key: str) -> float:
    return float(np.mean([_number(row, key) for row in rows]))


def _update_diagnostics(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("plateau analysis requires a non-empty updates.csv")
    fields = ("critic_loss", "q1_mean", "q2_mean", "target_mean", "alpha")
    values = {
        key: np.asarray([_number(row, key) for row in rows], dtype=np.float64)
        for key in fields
    }
    finite = bool(all(np.all(np.isfinite(array)) for array in values.values()))
    tail_size = max(1, min(len(rows), max(1000, len(rows) // 10)))
    q_tail = np.concatenate([
        np.abs(values["q1_mean"][-tail_size:]),
        np.abs(values["q2_mean"][-tail_size:]),
        np.abs(values["target_mean"][-tail_size:]),
    ])
    return {
        "finite": finite,
        "updates": len(rows),
        "tail_updates": tail_size,
        "q_abs_p95": float(np.quantile(q_tail, 0.95)),
        "critic_loss_p95": float(np.quantile(values["critic_loss"][-tail_size:], 0.95)),
        "alpha_min": float(np.min(values["alpha"])),
        "alpha_max": float(np.max(values["alpha"])),
    }


def _block_snapshot(run: Path, current_eta: float, block_steps: int) -> dict[str, Any]:
    with (run / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("stage") != "s0":
        raise ValueError(f"plateau analysis only accepts S0 runs: {run}")
    if int(summary["stage_step"]) != int(block_steps):
        raise ValueError(
            f"plateau analysis requires a complete {block_steps}-step block: "
            f"{run} has {summary['stage_step']}"
        )
    episodes = _rows(run / "episodes.csv")
    eligible = [
        row for row in episodes
        if int(_number(row, "curriculum_eligible")) == 1
        and int(_number(row, "orientation_anchor")) == 0
        and np.isclose(_number(row, "orientation_scale"), current_eta, atol=1e-9)
    ]
    anchors = [
        row for row in episodes
        if int(_number(row, "curriculum_eligible")) == 1
        and int(_number(row, "orientation_anchor")) == 1
    ]
    all_pose_scales = {
        round(_number(row, "orientation_scale"), 9)
        for row in episodes
        if int(_number(row, "curriculum_eligible")) == 1
        and int(_number(row, "orientation_anchor")) == 0
    }
    probes = [
        row for row in _rows(run / "orientation_probes.csv")
        if np.isclose(_number(row, "orientation_scale"), current_eta, atol=1e-9)
    ]
    return {
        "run": str(run),
        "stage_total_step": int(summary.get("stage_total_step", summary["global_step"])),
        "orientation_level_index": int(summary["orientation_level_index"]),
        "orientation_scale": float(summary["orientation_scale"]),
        "s0_goal_gate_eligible": bool(summary["s0_goal_gate_eligible"]),
        "pose_scales_seen": sorted(all_pose_scales),
        "current_episodes": len(eligible),
        "anchor_episodes": len(anchors),
        "safe_success_rate": _rate(eligible, "safe_success") if eligible else None,
        "self_collision_rate": _rate(eligible, "self_collision") if eligible else None,
        "timeout_rate": _rate(eligible, "timeout") if eligible else None,
        "anchor_success_rate": _rate(anchors, "position_reached") if anchors else None,
        "probe": (
            {
                "success_rate": _number(probes[-1], "success_rate"),
                "collision_rate": _number(probes[-1], "collision_rate"),
                "timeout_rate": _number(probes[-1], "timeout_rate"),
                "passed": bool(int(_number(probes[-1], "passed"))),
            }
            if probes else None
        ),
        "updates": _update_diagnostics(_rows(run / "updates.csv")),
    }


def decide_s0_stop(
    previous: dict[str, Any], current: dict[str, Any], criteria: dict[str, Any],
    max_stage_steps: int, block_steps: int = 100000,
) -> dict[str, Any]:
    step_delta = int(current["stage_total_step"]) - int(previous["stage_total_step"])
    if step_delta != int(block_steps):
        raise ValueError(
            "plateau analysis requires consecutive block endpoints: "
            f"stage_total_step delta is {step_delta}, expected {block_steps}"
        )
    previous_updates = previous["updates"]
    current_updates = current["updates"]
    nonfinite = not previous_updates["finite"] or not current_updates["finite"]
    ratio = float(criteria["divergence_ratio"])
    q_diverged = bool(
        current_updates["q_abs_p95"] > float(criteria["q_abs_limit"])
        and current_updates["q_abs_p95"]
        > ratio * max(previous_updates["q_abs_p95"], 1e-12)
    )
    loss_diverged = bool(
        current_updates["critic_loss_p95"] > float(criteria["critic_loss_limit"])
        and current_updates["critic_loss_p95"]
        > ratio * max(previous_updates["critic_loss_p95"], 1e-12)
    )
    numerical_failure = bool(nonfinite or (q_diverged and loss_diverged))

    same_level = bool(
        previous["orientation_level_index"] == current["orientation_level_index"]
        and np.isclose(previous["orientation_scale"], current["orientation_scale"], atol=1e-9)
    )
    no_level_progress = bool(
        same_level
        and previous["pose_scales_seen"] == [round(current["orientation_scale"], 9)]
        and current["pose_scales_seen"] == [round(current["orientation_scale"], 9)]
    )
    sufficient = bool(
        previous["current_episodes"] >= int(criteria["min_current_episodes_per_block"])
        and current["current_episodes"] >= int(criteria["min_current_episodes_per_block"])
        and previous["anchor_episodes"] >= int(criteria["min_anchor_episodes_per_block"])
        and current["anchor_episodes"] >= int(criteria["min_anchor_episodes_per_block"])
        and previous["probe"] is not None
        and current["probe"] is not None
    )
    improvements: dict[str, bool] = {}
    if sufficient:
        improvements = {
            "safe_success": current["safe_success_rate"] - previous["safe_success_rate"]
            >= float(criteria["safe_success_delta"]),
            "self_collision": previous["self_collision_rate"] - current["self_collision_rate"]
            >= float(criteria["self_collision_delta"]),
            "timeout": previous["timeout_rate"] - current["timeout_rate"]
            >= float(criteria["timeout_delta"]),
            "anchor_success": current["anchor_success_rate"] - previous["anchor_success_rate"]
            >= float(criteria["anchor_success_delta"]),
            "probe_success": current["probe"]["success_rate"] - previous["probe"]["success_rate"]
            >= float(criteria["probe_success_delta"]),
            "probe_collision": previous["probe"]["collision_rate"] - current["probe"]["collision_rate"]
            >= float(criteria["probe_collision_delta"]),
            "probe_timeout": previous["probe"]["timeout_rate"] - current["probe"]["timeout_rate"]
            >= float(criteria["probe_timeout_delta"]),
        }
    plateau = bool(no_level_progress and sufficient and not any(improvements.values()))
    budget_exhausted = current["stage_total_step"] >= int(max_stage_steps)

    if numerical_failure:
        decision = "numerical_failure"
    elif current["s0_goal_gate_eligible"]:
        decision = "run_validation"
    elif budget_exhausted:
        decision = "hard_budget_stop"
    elif plateau:
        decision = "plateau_stop"
    else:
        decision = "continue"
    return {
        "decision": decision,
        "consecutive_block_step_delta": step_delta,
        "no_level_progress_for_two_blocks": no_level_progress,
        "comparison_data_sufficient": sufficient,
        "substantial_improvements": improvements,
        "plateau_detected": plateau,
        "numerical_failure": numerical_failure,
        "q_diverged": q_diverged,
        "critic_loss_diverged": loss_diverged,
        "hard_budget_exhausted": budget_exhausted,
        "previous": previous,
        "current": current,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-registered S0 two-block stop analysis")
    parser.add_argument("--config", default="configs/experiments/thesis_homotopy.yaml")
    parser.add_argument("--blocks", nargs=2, required=True, metavar=("PREVIOUS", "CURRENT"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    criteria = config["train"]["plateau"]
    if int(criteria["comparison_blocks"]) != 2:
        raise ValueError("this analyzer requires plateau.comparison_blocks=2")
    block_steps = int(config["train"]["total_steps"])
    max_stage_steps = int(config["train"]["max_stage_steps"])
    block_paths = [
        path.resolve() if path.is_absolute() else (ROOT / path).resolve()
        for value in args.blocks
        for path in [Path(value)]
    ]
    with (block_paths[-1] / "summary.json").open(encoding="utf-8") as handle:
        current_summary = json.load(handle)
    current_eta = float(current_summary["orientation_scale"])
    snapshots = [
        _block_snapshot(path, current_eta=current_eta, block_steps=block_steps)
        for path in block_paths
    ]
    result = decide_s0_stop(
        snapshots[0], snapshots[1], criteria, max_stage_steps, block_steps
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
