from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_LABELS = {
    "ee_fixed": "EE risk\nfixed penalty",
    "link_fixed_penalty1": "Link risk\nfixed penalty ($w_R=1.0$)",
    "ldrc_fixed": "Link risk\nconstrained SAC",
}
METHOD_ORDER = ["ee_fixed", "link_fixed_penalty1", "ldrc_fixed"]
SCENARIO_ORDER = [
    "random_crossing",
    "upper_arm_crossing",
    "elbow_crossing",
    "forearm_crossing",
    "wrist_crossing",
]
SCENARIO_LABELS = {
    "random_crossing": "Random",
    "upper_arm_crossing": "Upper arm",
    "elbow_crossing": "Elbow",
    "forearm_crossing": "Forearm",
    "wrist_crossing": "Wrist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reproducible paper tables and figures.")
    parser.add_argument(
        "--heldout-summary",
        default="outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_across_train_seeds.csv",
    )
    parser.add_argument(
        "--heldout-by-train-seed",
        default="outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_by_train_seed.csv",
    )
    parser.add_argument(
        "--macro-summary",
        default="outputs/rechecks/heldout_1004_1006/final_3methods/eval_summary_macro_across_train_seeds.csv",
    )
    parser.add_argument("--sweep-summary", default="outputs/sweeps/summary/eval_summary_by_variant.csv")
    parser.add_argument("--ldrc-train-root", default="outputs/formal/train/ldrc_fixed")
    parser.add_argument("--output-dir", default="outputs/paper/final_materials")
    return parser.parse_args()


def format_mean_std(mean: float, std: float, scale: float = 1.0, digits: int = 2) -> str:
    return f"{mean * scale:.{digits}f} +/- {std * scale:.{digits}f}"


def write_main_tables(summary: pd.DataFrame, output_dir: Path) -> None:
    rows = summary.set_index(["scenario", "method"])
    main_rows = []
    for method in METHOD_ORDER:
        row = rows.loc[("random_crossing", method)]
        main_rows.append(
            {
                "method": method,
                "success_rate": format_mean_std(row.success_mean, row.success_std, 100.0),
                "collision_rate": format_mean_std(row.collision_mean, row.collision_std, 100.0),
                "non_end_link_collision_rate": format_mean_std(
                    row.non_end_link_collision_mean, row.non_end_link_collision_std, 100.0
                ),
                "final_position_error_m": format_mean_std(
                    row.final_position_error_mean, row.final_position_error_std, digits=3
                ),
                "min_distance_m": format_mean_std(row.min_distance_mean, row.min_distance_std, digits=3),
                "safety_violation_rate": format_mean_std(
                    row.safety_violation_rate_mean, row.safety_violation_rate_std, 100.0
                ),
                "rms_jerk": format_mean_std(row.rms_jerk_mean, row.rms_jerk_std, digits=1),
            }
        )
    pd.DataFrame(main_rows).to_csv(output_dir / "table_1_random_crossing.csv", index=False)

    stress_rows = []
    for scenario in SCENARIO_ORDER[1:]:
        for method in METHOD_ORDER:
            row = rows.loc[(scenario, method)]
            stress_rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "success_rate": format_mean_std(row.success_mean, row.success_std, 100.0),
                    "collision_rate": format_mean_std(row.collision_mean, row.collision_std, 100.0),
                    "final_position_error_m": format_mean_std(
                        row.final_position_error_mean, row.final_position_error_std, digits=3
                    ),
                    "rms_jerk": format_mean_std(row.rms_jerk_mean, row.rms_jerk_std, digits=1),
                }
            )
    pd.DataFrame(stress_rows).to_csv(output_dir / "table_2_non_end_link_stress.csv", index=False)


def write_appendix_table(by_train_seed: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        "scenario",
        "method",
        "train_seed",
        "episodes",
        "success",
        "collision",
        "non_end_link_collision",
        "final_position_error",
        "min_distance",
        "safety_violation_rate",
        "rms_jerk",
        "mean_cost_per_step",
    ]
    result = by_train_seed.loc[:, columns].copy()
    result.to_csv(output_dir / "appendix_table_a1_by_train_seed.csv", index=False)


def plot_macro(macro: pd.DataFrame, output_dir: Path) -> None:
    frame = macro.set_index("method").loc[METHOD_ORDER]
    labels = [METHOD_LABELS[method] for method in METHOD_ORDER]
    metrics = [
        ("success_mean", "success_std", "Success rate (%)", 100.0),
        ("collision_mean", "collision_std", "Collision rate (%)", 100.0),
        ("final_position_error_mean", "final_position_error_std", "Final position error (m)", 1.0),
        ("safety_violation_rate_mean", "safety_violation_rate_std", "Safety violation rate (%)", 100.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for axis, (mean_col, std_col, title, scale) in zip(axes.flat, metrics):
        axis.bar(labels, frame[mean_col] * scale, yerr=frame[std_col] * scale, capsize=4)
        axis.set_title(title)
        axis.tick_params(axis="x", labelsize=9)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Held-out macro average across five scenarios (mean +/- std, n=3 train seeds)")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_heldout_macro_comparison.png", dpi=220)
    plt.close(fig)


def plot_stress(summary: pd.DataFrame, output_dir: Path) -> None:
    frame = summary.set_index(["scenario", "method"])
    scenarios = SCENARIO_ORDER
    x = range(len(scenarios))
    width = 0.24
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
    for method_index, method in enumerate(METHOD_ORDER):
        success = [frame.loc[(scenario, method), "success_mean"] * 100.0 for scenario in scenarios]
        success_std = [frame.loc[(scenario, method), "success_std"] * 100.0 for scenario in scenarios]
        collision = [frame.loc[(scenario, method), "collision_mean"] * 100.0 for scenario in scenarios]
        collision_std = [frame.loc[(scenario, method), "collision_std"] * 100.0 for scenario in scenarios]
        offset = [value + (method_index - 1) * width for value in x]
        axes[0].bar(offset, success, width, yerr=success_std, capsize=3, label=METHOD_LABELS[method])
        axes[1].bar(offset, collision, width, yerr=collision_std, capsize=3, label=METHOD_LABELS[method])
    axes[0].set_ylabel("Success rate (%)")
    axes[1].set_ylabel("Collision rate (%)")
    axes[1].set_xticks(list(x), [SCENARIO_LABELS[scenario] for scenario in scenarios])
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Held-out evaluation by scenario (mean +/- std, n=3 train seeds)")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_2_heldout_by_scenario.png", dpi=220)
    plt.close(fig)


def plot_penalty_sweep(sweep: pd.DataFrame, output_dir: Path) -> None:
    frame = sweep[sweep["sweep"] == "fixed_risk_penalty"].copy()
    frame["penalty"] = frame["variant"].str.rsplit("_", n=1).str[-1].astype(float)
    frame = frame.sort_values("penalty")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for axis, mean_col, std_col, label, scale in [
        (axes[0], "success_mean", "success_std", "Success rate (%)", 100.0),
        (axes[1], "collision_mean", "collision_std", "Collision rate (%)", 100.0),
    ]:
        axis.errorbar(frame["penalty"], frame[mean_col] * scale, yerr=frame[std_col] * scale, marker="o", capsize=4)
        axis.set_xlabel("Fixed risk penalty $w_R$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    fig.suptitle("Fixed-risk-penalty screening (two train seeds; exploratory only)")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_a1_fixed_penalty_sensitivity.png", dpi=220)
    plt.close(fig)


def plot_ldrc_training(train_root: Path, output_dir: Path) -> None:
    paths = sorted(train_root.glob("seed_*/**/train_metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No ldrc_fixed train_metrics.csv below {train_root}")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    columns = [
        ("episode_reward", "Episode reward"),
        ("episode_cost", "Episode cost"),
        ("lambda", "Lagrange multiplier"),
        ("safety_violation_rate", "Safety violation rate"),
    ]
    for path in paths:
        frame = pd.read_csv(path)
        seed = path.parts[-3].replace("seed_", "seed ")
        for axis, (column, ylabel) in zip(axes.flat, columns):
            axis.plot(frame["episode"], frame[column].rolling(20, min_periods=1).mean(), label=seed, alpha=0.9)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Episode")
    axes[0, 0].legend()
    fig.suptitle("LDRC fixed training diagnostics (20-episode rolling mean; historical mechanism analysis)")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_a2_ldrc_training_diagnostics.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.heldout_summary)
    by_train_seed = pd.read_csv(args.heldout_by_train_seed)
    macro = pd.read_csv(args.macro_summary)
    sweep = pd.read_csv(args.sweep_summary)

    write_main_tables(summary, output_dir)
    write_appendix_table(by_train_seed, output_dir)
    plot_macro(macro, output_dir)
    plot_stress(summary, output_dir)
    plot_penalty_sweep(sweep, output_dir)
    plot_ldrc_training(Path(args.ldrc_train_root), output_dir)
    print(f"saved paper materials: {output_dir}")


if __name__ == "__main__":
    main()
