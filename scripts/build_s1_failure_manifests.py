from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.diagnostics).open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError("diagnostics CSV is empty")
    candidate = sorted({int(r["reset_seed"]) for r in rows if r["success"] == "0" and r["static_candidate_path_status"] == "candidate_path_found"})
    hard_statuses = {"not_found", "not_checked_due_to_ik"}
    hard = sorted(
        {
            int(r["reset_seed"])
            for r in rows
            if r["success"] == "0" and r["static_candidate_path_status"] in hard_statuses
        }
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = [
        ("s1_r3_candidate_failure_v2.json", candidate, "candidate_path_found failures"),
        ("s1_r4_hard_case_failure_v1.json", hard, "not_found/not_checked_due_to_ik failures"),
    ]
    for name, seeds, rule in specs:
        if not seeds:
            raise ValueError(f"empty failure bucket: {name}")
        payload = {
            "manifest_version": 1,
            "protocol": name.removesuffix(".json"),
            "role": "focused_training_resets",
            "source": args.diagnostics,
            "selection_rule": rule,
            "not_for_checkpoint_selection": True,
            "not_a_final_evaluation_manifest": True,
            "seeds": seeds,
        }
        (out / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {len(seeds)} seeds")


if __name__ == "__main__":
    main()
