from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from rl_risk_sac.utils.provenance import _source_hashes, sha256_file


METRICS = [
    "success",
    "collision",
    "non_end_link_collision",
    "final_position_error",
    "completion_time",
    "min_distance",
    "safety_violation_rate",
    "mean_action_variation",
    "rms_acceleration",
    "rms_jerk",
    "peak_acceleration",
    "peak_jerk",
    "command_rms_acceleration",
    "command_rms_jerk",
    "command_peak_acceleration",
    "command_peak_jerk",
    "measured_rms_acceleration",
    "measured_rms_jerk",
    "measured_peak_acceleration",
    "measured_peak_jerk",
    "safety_qp_intervention_rate",
    "safety_qp_infeasible_rate",
    "mean_safety_qp_correction_norm",
    "max_safety_qp_correction_norm",
    "mean_safety_qp_solve_time_ms",
    "max_safety_qp_solve_time_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a fixed-actor evaluation matrix.")
    parser.add_argument(
        "--matrix",
        default="configs/experiments/minimal_qp_heldout_counterfactual.yaml",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame.reset_index()


def main() -> None:
    matrix_path = Path(parse_args().matrix)
    matrix = load_yaml(matrix_path)
    evaluation = matrix["evaluation"]
    root = Path(evaluation["output_root"])
    expected = set()
    for variant, variant_cfg in matrix["variants"].items():
        checkpoints = variant_cfg.get("checkpoints", matrix.get("checkpoints"))
        if checkpoints is None:
            raise KeyError(
                f"Variant {variant!r} must define checkpoints or inherit top-level checkpoints"
            )
        expected.update(
            (
                str(scenario),
                str(variant),
                int(train_seed),
                int(eval_seed),
            )
            for scenario in evaluation["scenarios"]
            for train_seed in checkpoints
            for eval_seed in evaluation["seeds"]
        )

    frames: list[pd.DataFrame] = []
    found: set[tuple[str, str, int, int]] = set()
    malformed_files: list[str] = []
    episodes_per_seed = int(evaluation["episodes_per_seed"])
    for scenario, variant, train_seed, eval_seed in sorted(expected):
        path = root / scenario / variant / f"train_seed_{train_seed}" / f"eval_seed_{eval_seed}.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        required_columns = {"episode", "episode_seed"}
        if len(frame) != episodes_per_seed or not required_columns.issubset(frame.columns):
            malformed_files.append(str(path))
            continue
        if frame["episode"].nunique() != episodes_per_seed or frame["episode_seed"].nunique() != episodes_per_seed:
            malformed_files.append(str(path))
            continue
        frame.insert(0, "eval_seed", eval_seed)
        frame.insert(0, "train_seed", train_seed)
        frame.insert(0, "variant", variant)
        frame.insert(0, "scenario", scenario)
        frames.append(frame)
        found.add((scenario, variant, train_seed, eval_seed))

    missing = sorted(expected - found)
    episodes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    expected_rows = len(expected) * int(evaluation["episodes_per_seed"])
    duplicate_episode_units = 0
    if not episodes.empty and "episode_seed" in episodes.columns:
        duplicate_episode_units = int(
            episodes.duplicated(["scenario", "variant", "train_seed", "episode_seed"]).sum()
        )
    plan_path = root / "evaluation_plan_manifest.json"
    plan_errors: list[str] = []
    if not plan_path.is_file():
        plan_errors.append(f"missing plan manifest: {plan_path}")
        plan: dict[str, Any] = {}
    else:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("matrix_sha256") != sha256_file(matrix_path):
            plan_errors.append("matrix hash changed after evaluation plan generation")
        repo_root = Path(__file__).resolve().parents[1]
        if plan.get("source_files_sha256") != _source_hashes(repo_root):
            plan_errors.append("source/config snapshot changed after evaluation plan generation")
        for section in ("generated_config_sha256", "checkpoint_sha256"):
            for raw_path, expected_hash in plan.get(section, {}).items():
                path = Path(raw_path)
                if not path.is_file() or sha256_file(path) != expected_hash:
                    plan_errors.append(f"hash mismatch in {section}: {path}")

    manifest = {
        "schema_version": "restart_eval_summary_manifest_v1",
        "plan_manifest": str(plan_path),
        "plan_verified": not plan_errors,
        "plan_errors": plan_errors,
        "expected_files": len(expected),
        "found_files": len(found),
        "expected_rows": expected_rows,
        "found_rows": len(episodes),
        "duplicate_episode_units": duplicate_episode_units,
        "episode_seed_recorded": "episode_seed" in episodes.columns,
        "malformed_files": malformed_files,
        "complete": (
            not missing
            and not malformed_files
            and not plan_errors
            and len(episodes) == expected_rows
            and "episode_seed" in episodes.columns
            and duplicate_episode_units == 0
        ),
        "missing": [list(item) for item in missing],
    }
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not frames:
        raise FileNotFoundError(f"No evaluation CSV files found below {root}")

    available_metrics = [metric for metric in METRICS if metric in episodes.columns]
    episodes.to_csv(summary_dir / "all_eval_episodes.csv", index=False)

    by_train_seed = (
        episodes.groupby(["scenario", "variant", "train_seed"], sort=True)[available_metrics]
        .mean()
        .reset_index()
    )
    by_train_seed.to_csv(summary_dir / "eval_summary_by_train_seed.csv", index=False)

    across_train_seeds = flatten_columns(
        by_train_seed.groupby(["scenario", "variant"], sort=True)[available_metrics].agg(["mean", "std"])
    )
    across_train_seeds.to_csv(summary_dir / "eval_summary_across_train_seeds.csv", index=False)

    macro_by_train_seed = (
        by_train_seed.groupby(["variant", "train_seed"], sort=True)[available_metrics].mean().reset_index()
    )
    macro = flatten_columns(
        macro_by_train_seed.groupby("variant", sort=True)[available_metrics].agg(["mean", "std"])
    )
    macro.to_csv(summary_dir / "eval_summary_macro_across_train_seeds.csv", index=False)
    print(f"saved summary: {summary_dir}")
    print(f"complete: {manifest['complete']} ({len(found)}/{len(expected)} files)")
    if not manifest["complete"]:
        raise RuntimeError(f"Evaluation matrix is incomplete or contains duplicate episode seeds: {manifest}")


if __name__ == "__main__":
    main()
