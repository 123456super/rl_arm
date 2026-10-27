#!/usr/bin/env python3
"""Audit an existing bank and write a separate, whole-arm-verified subset."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch

from rl_risk_sac.algorithms.dual_sac import build_thesis_sac
from rl_risk_sac.tasks.local_avoidance import LOCAL_OBSERVATION_DIM, validate_local_config
from rl_risk_sac.tasks.local_samples import ReachingSampleGenerator, validate_bank
from rl_risk_sac.tasks.local_path_verification import LocalPathVerifier, nominal_goal_for_sample
from rl_risk_sac.envs.thesis_homotopy_env import ResetSamplingError
from rl_risk_sac.utils.config import load_config
from scripts.core.train_thesis_homotopy import apply_stage_profile, initialize_frozen_reaching


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--config", default="configs/experiments/paper_local_avoidance_geometry.yaml")
    parser.add_argument("--reaching-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, help="new directory; source bank is never overwritten")
    parser.add_argument("--limit", type=int, help="audit only the first N starts for a timing/diagnostic run")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    config = load_config(args.config)
    apply_stage_profile(config, "s1")
    validate_local_config(config)
    config["device"] = "cpu"
    torch.set_num_threads(1)
    torch.manual_seed(int(config["seed"]))
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    initialization = initialize_frozen_reaching(agent, Path(args.reaching_checkpoint))
    source = Path(args.bank)
    bank = torch.load(source, map_location="cpu", weights_only=False)
    validate_bank(bank, config, agent.reaching, split=("train", "validation", "test"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    samples = bank["samples"][:args.limit]
    generator = ReachingSampleGenerator(config, agent.reaching, 0, verify_paths=False)
    verifier = LocalPathVerifier(config, agent.reaching)
    accepted, rows, statuses = [], [], Counter()
    started = perf_counter()
    try:
        for index, sample in enumerate(samples, 1):
            item_started = perf_counter()
            try:
                goal = nominal_goal_for_sample(generator, sample)
            except ResetSamplingError:
                goal = None
            if goal is None:
                record = dict(status="nominal_endpoint_unconfirmed", global_path_verified=False)
            else:
                record = verifier.verify(sample, goal)
            statuses[record["status"]] += 1
            if record["status"] == "verified":
                accepted.append(dict(sample, path_verification=record))
            rows.append(dict(sample_id=sample["sample_id"], source_seed=int(sample["provenance"]["seed"]),
                             obstacle_count=len(sample["state"]["obstacles"]),
                             **{key: value for key, value in record.items()
                                if key in ("status", "global_path_verified", "method", "search_checks",
                                           "verification_checks", "exit_checks", "min_external_m", "min_self_m",
                                           "global_path_max_progress", "global_path_exit_candidates", "local_search_checks")},
                             total_seconds=perf_counter() - item_started))
            print(f"path audit {index}/{len(samples)}: {record['status']} "
                  f"accepted={len(accepted)} item={rows[-1]['total_seconds']:.2f}s "
                  f"elapsed={perf_counter() - started:.1f}s", flush=True)
    finally:
        verifier.close()
        generator.close()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    lineage = dict(path=str(source.resolve()), sha256=digest.hexdigest())
    timings = [row["total_seconds"] for row in rows]
    report = dict(source_bank=lineage, split=bank["split"], source_samples=len(bank["samples"]),
                  audited=len(rows), limited=len(samples) < len(bank["samples"]), accepted=len(accepted), statuses=dict(statuses),
                  global_path_verified=sum(row["global_path_verified"] for row in rows),
                  path_verification=verifier.summary(), reaching_initialization=initialization,
                  nominal_rejections=dict(generator.rejections),
                  wall_seconds=perf_counter() - started,
                  median_seconds=float(np.median(timings)), p95_seconds=float(np.percentile(timings, 95)),
                  continuous_collision_proof=False, dynamic_tracking_proof=False,
                  unsuccessful_search_means="unconfirmed; not proof of no solution", outcomes=rows)
    (output / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if not accepted:
        raise RuntimeError("no samples confirmed; report saved, no empty bank written")
    filtered = dict(bank, samples=accepted, path_verification_source=lineage,
                    path_verification_summary=verifier.summary())
    validate_bank(filtered, config, agent.reaching, split=(bank["split"],), require_path_verification=True)
    torch.save(filtered, output / "bank.pt")
    print(f"saved {output / 'bank.pt'} ({len(accepted)}/{len(samples)} verified starts)", flush=True)


if __name__ == "__main__":
    main()
