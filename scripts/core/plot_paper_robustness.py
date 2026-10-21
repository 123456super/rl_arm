#!/usr/bin/env python3
"""Plot existing frozen-policy evaluations; never train or rerun evaluation.

Run from the project root:
    conda run -n rl python scripts/core/plot_paper_robustness.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "sensor": ROOT / "outputs/sensor_noise/compact_L13_20261020",
    "force": ROOT / "outputs/mechanical_disturbance/compact_L13_20261020",
    "joint": ROOT / "outputs/joint_robustness/regular_L13_20261020",
}
SEEDS = (11001, 11002)
COLORS = ("#2F5D8C", "#A33D3D")


def load_verified(directory: Path) -> list[dict]:
    rows = list(csv.DictReader((directory / "summary.csv").open()))
    for row in rows:
        result = json.loads((directory / row["result_file"]).read_text())
        for key in ("success_rate", "training_seed", "episodes", "checkpoint_sha256"):
            if str(result[key]) != row[key]:
                raise ValueError(f"CSV/JSON mismatch: {directory}, {key}")
        if int(row["episodes"]) != 1000:
            raise ValueError("Expected 1000 episodes per policy and condition")
        if not (np.isclose(result["position_tolerance_m"], .001)
                and np.isclose(result["orientation_tolerance_rad"], .025)):
            raise ValueError("Expected strict L13 tolerance")
    return rows


def panel(ax, rows, field, levels, title, xlabel):
    rates = []
    for seed, color, marker in zip(SEEDS, COLORS, ("o", "s")):
        selected = [r for r in rows if int(r["training_seed"]) == seed]
        if sorted(float(r[field]) for r in selected) != levels:
            raise ValueError(f"Unexpected or duplicate conditions for seed {seed}")
        y = [100 * float(next(r for r in selected if float(r[field]) == x)["success_rate"])
             for x in levels]
        rates.append(y)
        ax.plot(levels, y, color=color, marker=marker, ms=3.8, lw=1.1,
                markerfacecolor="white", markeredgewidth=.85, zorder=3)
    mean = np.mean(rates, axis=0)
    ax.plot(levels, mean, color="#222222", ls="--", lw=1, marker="D", ms=2.6, zorder=4)
    ax.set(title=title, xlabel=xlabel, ylabel="Coupled-pose success (%)",
           ylim=(-2, 103), yticks=[0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#DEE2E5", lw=.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=.6, length=3)
    return {str(x): round(float(y), 2) for x, y in zip(levels, mean)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "docs/paper/manuscript_en/figures")
    args = parser.parse_args()
    data = {name: load_verified(path) for name, path in SOURCES.items()}
    # Policy identity must match across all three experiments.
    for seed in SEEDS:
        hashes = {r["checkpoint_sha256"] for rows in data.values() for r in rows
                  if int(r["training_seed"]) == seed}
        if len(hashes) != 1:
            raise ValueError("Different checkpoints across experiments")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.05))
    means = {}
    means["position"] = panel(
        axes[0, 0], [r for r in data["sensor"] if r["noise_type"] == "position"],
        "noise_std", [0., .1, .5, 1., 2.], "(a) Position measurement noise",
        r"Per-axis standard deviation $\sigma_p$ (mm)")
    axes[0, 0].set_xticks([0, .5, 1, 1.5, 2])
    means["orientation"] = panel(
        axes[0, 1], [r for r in data["sensor"] if r["noise_type"] == "orientation"],
        "noise_std", [0., .0025, .005, .01, .02], "(b) Orientation measurement noise",
        r"Rotation-vector component std. $\sigma_R$ (rad)")
    axes[0, 1].set_xticks([0, .005, .01, .015, .02], ["0", "0.005", "0.010", "0.015", "0.020"])
    means["force"] = panel(
        axes[1, 0], data["force"], "force_n", [0., 5., 10., 20.],
        "(c) External tool force", "Force magnitude (N)")
    axes[1, 0].set_xticks([0, 5, 10, 20])
    means["joint"] = panel(
        axes[1, 1], data["joint"], "position_std_mm", [.75, .85, .90, .95, 1.],
        "(d) Position noise + 20 N force",
        r"Per-axis standard deviation $\sigma_p$ (mm)")
    axes[1, 1].set_xticks([.75, .85, .90, .95, 1.], ["0.75", "0.85", "0.90", "0.95", "1.00"])
    handles = [Line2D([], [], color=c, marker=m, markerfacecolor="white", lw=1.1,
                      ms=3.8, label=f"Training seed {s}")
               for s, c, m in zip(SEEDS, COLORS, ("o", "s"))]
    handles.append(Line2D([], [], color="#222222", ls="--", marker="D", ms=2.6,
                          lw=1, label="Two-seed mean"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.52, 1), handlelength=2.3, columnspacing=2)
    fig.subplots_adjust(left=.085, right=.985, bottom=.115, top=.865,
                        hspace=.66, wspace=.32)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(out / f"fig_robustness.{extension}", dpi=220)
    plt.close(fig)
    manifest = {
        "training_seeds": SEEDS, "mean_success_percent": means,
        "sources": {name: {"csv": str(path / "summary.csv"),
                            "sha256": hashlib.sha256((path / "summary.csv").read_bytes()).hexdigest()}
                    for name, path in SOURCES.items()},
        "joint_display_range_mm": [.75, 1.],
        "joint_range_note": "Display restricted after evaluation; higher levels remain archived.",
        "uncertainty_note": "Individual training seeds shown; no confidence band estimated from n=2.",
    }
    (out / "fig_robustness_data.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(means, indent=2))
    print(f"Saved vector PDF/SVG and PNG preview to {out}")


if __name__ == "__main__":
    main()
