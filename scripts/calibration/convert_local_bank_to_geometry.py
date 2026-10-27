#!/usr/bin/env python3
"""Relabel existing captured S0 starts with the deployed geometric task."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from rl_risk_sac.algorithms.dual_sac import build_thesis_sac
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv, ResetSamplingError
from rl_risk_sac.tasks.geometric_passage import GeometricPassageReference
from rl_risk_sac.tasks.local_avoidance import (
    LOCAL_OBSERVATION_DIM, PROTOCOL, contract_digest, geometric_mode, local_protocol, validate_local_config,
)
from rl_risk_sac.tasks.local_samples import ReachingSampleGenerator, relabel_geometric_sample, simulation_config, validate_bank
from rl_risk_sac.tasks.local_path_verification import LocalPathVerifier, nominal_goal_for_sample
from rl_risk_sac.utils.config import load_config
from scripts.core.train_thesis_homotopy import apply_stage_profile, initialize_frozen_reaching


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--source-config", help="optional explicit trajectory-task config; default derives it from the geometric destination")
    parser.add_argument("--config", default="configs/experiments/paper_local_avoidance_geometry.yaml")
    parser.add_argument("--reaching-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.config)
    if args.source_config:
        source = load_config(args.source_config)
    else:
        # Reconstruct only the source task label. Scene, robot and physical
        # parameters remain identical; the bank digest below verifies this.
        source = copy.deepcopy(config)
        source["local_avoidance"].pop("reference_mode", None)
        source["local_avoidance"].pop("geometry", None)
    for cfg in (source, config):
        apply_stage_profile(cfg, "s1")
        validate_local_config(cfg)
        cfg["device"] = "cpu"
    if geometric_mode(source) or not geometric_mode(config):
        parser.error("conversion requires a legacy source config and a geometric destination config")
    # Scene distribution, geometry, normalization and mature S0 must all agree.
    comparable = dict(config["local_avoidance"])
    comparable.pop("reference_mode")
    comparable.pop("geometry")
    if comparable != source["local_avoidance"] or source["robot"] != config["robot"] or source["thesis"] != config["thesis"]:
        parser.error("conversion may only change the local reference method, not the scene/robot/task distribution")
    agent = build_thesis_sac(LOCAL_OBSERVATION_DIM, 6, config)
    initialization = initialize_frozen_reaching(agent, Path(args.reaching_checkpoint))
    path = Path(args.bank)
    bank = torch.load(path, map_location="cpu", weights_only=False)
    validate_bank(bank, source, agent.reaching, split=("train", "validation", "test"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    env = ThesisHomotopyEnv(simulation_config(config), cache_geometry_queries=True)
    generator = ReachingSampleGenerator(config, agent.reaching, 0, verify_paths=False)
    verifier = LocalPathVerifier(config, agent.reaching)
    accepted, rejected = [], Counter()
    try:
        guide = GeometricPassageReference(config, agent.reaching)
        try:
            for index, sample in enumerate(bank["samples"], 1):
                result, status = relabel_geometric_sample(sample, env, guide)
                if result is None:
                    rejected[status] += 1
                else:
                    try:
                        goal = nominal_goal_for_sample(generator, result)
                    except ResetSamplingError:
                        goal = None
                    if goal is None:
                        rejected["nominal_endpoint_unconfirmed"] += 1
                    else:
                        record = verifier.verify(result, goal)
                        if record["status"] == "verified":
                            result["path_verification"] = record
                            accepted.append(result)
                        else:
                            rejected[record["status"]] += 1
                if index % 10 == 0 or index == len(bank["samples"]):
                    print(f"geometry {index}/{len(bank['samples'])}: accepted={len(accepted)} rejected={sum(rejected.values())}", flush=True)
        finally:
            guide.close()
    finally:
        env.close()
        generator.close()
        verifier.close()
    lineage = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), protocol=PROTOCOL)
    report = dict(protocol=local_protocol(config), split=bank["split"], source_samples=len(bank["samples"]),
                  accepted=len(accepted), rejected=dict(rejected), source_bank=lineage,
                  path_verification=verifier.summary(),
                  reaching_initialization=initialization)
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not accepted:
        raise RuntimeError("no captured starts have a valid geometric trigger; no empty bank written")
    converted = dict(bank, protocol=local_protocol(config), contract_sha256=contract_digest(config),
                     samples=accepted, source_bank=lineage, reaching_initialization=initialization)
    converted["path_verification_summary"] = verifier.summary()
    validate_bank(converted, config, agent.reaching, split=(bank["split"],), require_path_verification=True)
    torch.save(converted, output / "bank.pt")
    print(f"saved {output / 'bank.pt'} ({len(accepted)} starts; rejected starts were not copied)", flush=True)


if __name__ == "__main__":
    main()
