#!/usr/bin/env python3
"""Capture S0 starts and filter with bounded whole-arm/local-exit witnesses."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch

from rl_risk_sac.algorithms.dual_sac import build_thesis_sac
from rl_risk_sac.tasks.local_avoidance import (
    PROTOCOL, LOCAL_OBSERVATION_DIM, validate_local_config, contract_digest,
    local_protocol,
)
from rl_risk_sac.tasks.local_samples import ReachingSampleGenerator, validate_bank
from rl_risk_sac.envs.thesis_homotopy_env import ResetSamplingError
from rl_risk_sac.utils.config import load_config
from scripts.core.train_thesis_homotopy import apply_stage_profile, initialize_frozen_reaching


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/paper_local_avoidance_geometry.yaml")
    parser.add_argument("--reaching-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--max-traces", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    args = parser.parse_args()
    if min(args.samples, args.max_traces) < 1 or args.seed < 0:
        parser.error("sample/trace budgets must be positive and seed nonnegative")
    config = load_config(args.config)
    apply_stage_profile(config, "s1")
    validate_local_config(config)
    config["device"] = "cpu"
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    initialization = initialize_frozen_reaching(agent, Path(args.reaching_checkpoint))
    split_id = ("train", "validation", "test").index(args.split)
    seeds = np.random.SeedSequence([args.seed, split_id]).generate_state(args.max_traces)
    generator = ReachingSampleGenerator(config, agent.reaching, int(args.seed) + split_id)
    samples, traces = [], 0
    last_progress = perf_counter()
    try:
        for seed in seeds:
            traces += 1
            try:
                sample = generator.sample(int(seed))
            except ResetSamplingError:
                generator.rejections["reset_budget"] += 1
                sample = None
            if sample is not None:
                samples.append(sample)
                print(f"captured {len(samples)}/{args.samples}: trace={traces} approach_steps={sample['provenance']['approach_steps']}", flush=True)
                last_progress = perf_counter()
            elif perf_counter() - last_progress >= 10.:
                print(f"sampling trace={traces}/{args.max_traces} captured={len(samples)}/{args.samples} "
                      f"path_statuses={generator.path_verifier.status_counts}", flush=True)
                last_progress = perf_counter()
            if len(samples) >= args.samples:
                break
    finally:
        generator.close()
    report = dict(protocol=local_protocol(config), split=args.split, requested=args.samples,
                  captured=len(samples), traces=traces, rejections=dict(generator.rejections),
                  config=str(Path(args.config).resolve()),
                  obstacle_counts=dict(Counter(len(sample["state"]["obstacles"]) for sample in samples)),
                  path_verification=generator.path_verifier.summary(),
                  reaching_initialization=initialization)
    (output / "generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if len(samples) != args.samples:
        raise RuntimeError("sample budget exhausted; report saved, no incomplete training bank written")
    bank = dict(format_version=1, protocol=local_protocol(config), split=args.split,
                seed=args.seed, contract_sha256=contract_digest(config),
                policy_sha256=generator.policy_sha256, reaching_initialization=initialization,
                samples=samples)
    bank["path_verification_summary"] = generator.path_verifier.summary()
    validate_bank(bank, config, agent.reaching, require_path_verification=True)
    torch.save(bank, output / "bank.pt")
    print(f"saved {output / 'bank.pt'}", flush=True)


if __name__ == "__main__":
    main()
