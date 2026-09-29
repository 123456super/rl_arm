#!/usr/bin/env python3
"""Measure policy retention and forgetting on paired deterministic episodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def exact_mcnemar_p_value(lost: int, gained: int) -> float:
    """Two-sided exact McNemar test for discordant paired outcomes."""
    discordant = lost + gained
    if discordant == 0:
        return 1.0
    smaller = min(lost, gained)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
    return min(1.0, 2.0 * lower_tail)


def load_rows(path: Path, contract_name: str) -> tuple[list[str], dict[str, dict[int, dict[str, str]]]]:
    by_checkpoint: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    order: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["contract_name"] != contract_name:
                continue
            checkpoint = row["checkpoint"]
            if checkpoint not in by_checkpoint:
                order.append(checkpoint)
            episode = int(row["episode"])
            if episode in by_checkpoint[checkpoint]:
                raise ValueError(f"duplicate row for {checkpoint=} {episode=}")
            by_checkpoint[checkpoint][episode] = row
    if not by_checkpoint:
        raise ValueError(f"no rows found for contract {contract_name!r}")
    return order, dict(by_checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--contract", default="L15")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    order, rows = load_rows(args.episodes, args.contract)
    if args.reference not in rows:
        parser.error(f"reference {args.reference!r} is absent; choices: {', '.join(order)}")
    reference = rows[args.reference]
    reference_episodes = set(reference)
    result: dict[str, object] = {
        "episodes_csv": str(args.episodes),
        "contract_name": args.contract,
        "reference": args.reference,
        "comparisons": {},
    }

    for checkpoint in order:
        if checkpoint == args.reference:
            continue
        candidate = rows[checkpoint]
        if set(candidate) != reference_episodes:
            missing = sorted(reference_episodes - set(candidate))[:10]
            extra = sorted(set(candidate) - reference_episodes)[:10]
            raise ValueError(f"unpaired episodes for {checkpoint}: {missing=} {extra=}")

        counts: Counter[str] = Counter()
        lost_reasons: Counter[str] = Counter()
        for episode in sorted(reference_episodes):
            old = int(reference[episode]["safe_success"])
            new = int(candidate[episode]["safe_success"])
            category = {
                (1, 1): "retained_success",
                (1, 0): "lost_success",
                (0, 1): "gained_success",
                (0, 0): "persistent_failure",
            }[(old, new)]
            counts[category] += 1
            if category == "lost_success":
                row = candidate[episode]
                reason = next(
                    (name for name in ("collision", "joint_limit", "timeout") if int(row[name])),
                    "other",
                )
                lost_reasons[reason] += 1

        total = len(reference_episodes)
        old_successes = counts["retained_success"] + counts["lost_success"]
        new_successes = counts["retained_success"] + counts["gained_success"]
        lost = counts["lost_success"]
        gained = counts["gained_success"]
        result["comparisons"][checkpoint] = {
            **dict(counts),
            "episodes": total,
            "reference_success_rate": old_successes / total,
            "candidate_success_rate": new_successes / total,
            "net_success_rate_change": (new_successes - old_successes) / total,
            "old_success_retention_rate": (
                counts["retained_success"] / old_successes if old_successes else None
            ),
            "old_success_forgetting_rate": lost / old_successes if old_successes else None,
            "outcome_churn_rate": (lost + gained) / total,
            "lost_to_gained_ratio": lost / gained if gained else None,
            "mcnemar_exact_two_sided_p": exact_mcnemar_p_value(lost, gained),
            "lost_success_failure_reasons": dict(lost_reasons),
        }

    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
