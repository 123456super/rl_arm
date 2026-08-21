"""Compare two final hierarchical evaluation CSVs episode by episode."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.old, newline="", encoding="utf-8") as file:
        old = {(int(row["seed"]), int(row["episode"])): row for row in csv.DictReader(file)}
    with open(args.new, newline="", encoding="utf-8") as file:
        new = {(int(row["seed"]), int(row["episode"])): row for row in csv.DictReader(file)}
    if set(old) != set(new):
        raise ValueError("old/new CSVs do not contain identical seed/episode keys")
    transitions = {"old_success_new_success": 0, "old_success_new_failure": 0, "old_failure_new_success": 0, "old_failure_new_failure": 0}
    rows = []
    for key in sorted(old):
        old_success = int(float(old[key]["success"])) == 1
        new_success = int(float(new[key]["success"])) == 1
        if old_success and new_success:
            transition = "old_success_new_success"
        elif old_success:
            transition = "old_success_new_failure"
        elif new_success:
            transition = "old_failure_new_success"
        else:
            transition = "old_failure_new_failure"
        transitions[transition] += 1
        rows.append({"seed": key[0], "episode": key[1], "old_success": int(old_success), "new_success": int(new_success), "transition": transition, "old_failure_mode": old[key].get("hierarchical_failure_mode", ""), "new_failure_mode": new[key].get("hierarchical_failure_mode", "")})
    report = {
        "old": str(args.old),
        "new": str(args.new),
        "episodes": len(rows),
        "old_success": sum(row["old_success"] for row in rows),
        "new_success": sum(row["new_success"] for row in rows),
        "success_delta": sum(row["new_success"] for row in rows) - sum(row["old_success"] for row in rows),
        "transitions": transitions,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
