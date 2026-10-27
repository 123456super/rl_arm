#!/usr/bin/env python3
"""Explicit, non-destructive reward-only migration of a trusted local S1 bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from rl_risk_sac.algorithms.s1_expert_replay import file_sha256, relabel_coupling_bank
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config
from scripts.core.train_thesis_homotopy import apply_stage_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", required=True, help="explicit config containing the target S1 coupling reward")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output.with_suffix(".relabel.json")
    if args.output.exists() or report_path.exists():
        parser.error("output/report already exists; choose new paths, never overwrite source data")
    config = load_config(args.config)
    apply_stage_profile(config, "s1")
    env = ThesisHomotopyEnv(config)
    try:
        parameters = dict(env.reward_parameters)
    finally:
        env.close()
    bank = torch.load(args.source, map_location="cpu", weights_only=False)
    migrated = relabel_coupling_bank(bank, config, parameters)
    migrated["coupling_relabel"].update(source=str(args.source.resolve()), source_sha256=file_sha256(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        torch.save(migrated, handle)
    report = dict(migrated["coupling_relabel"], episodes=len(migrated["episodes"]),
                  transitions=sum(part.size for part in migrated["partitions"].values()),
                  output=str(args.output.resolve()), output_sha256=file_sha256(args.output))
    with report_path.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "initial_potential_delta"}, indent=2))


if __name__ == "__main__":
    main()
