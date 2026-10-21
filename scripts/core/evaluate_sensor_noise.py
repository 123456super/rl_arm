#!/usr/bin/env python3
"""Compact L13 benchmark: separate position and SO(3) sensor-noise sweeps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "scripts/core/evaluate_sensor_noise.py",
    "scripts/core/evaluate_thesis_homotopy.py",
    "src/rl_risk_sac/envs/position_sensor_noise.py",
    "src/rl_risk_sac/envs/orientation_sensor_noise.py",
    "src/rl_risk_sac/envs/thesis_homotopy_env.py",
    "src/rl_risk_sac/envs/thesis_homotopy_env_reduced_observation.py",
)
FIELDS = (
    "policy", "training_seed", "test_seed", "noise_type", "noise_std", "noise_unit",
    "episodes", "success_rate", "collision_rate", "timeout_rate", "mean_episode_steps",
    "p95_position_error_mm", "p95_orientation_error_rad", "checkpoint_sha256",
    "source_config_sha256", "result_file",
)


def save_csv(output: Path, rows: list[dict]) -> None:
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict], field: str) -> float:
    return sum(row[field] for row in rows) / len(rows)


def plot_results(output: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = sorted({row["policy"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for axis, kind, unit in zip(axes, ("position", "orientation"), ("mm", "rad")):
        selected = [row for row in rows if row["noise_type"] == kind]
        for index, policy in enumerate(policies):
            group = [row for row in selected if row["policy"] == policy]
            sigmas = sorted({row["noise_std"] for row in group})
            rates = [100 * mean([row for row in group if row["noise_std"] == sigma],
                                "success_rate") for sigma in sigmas]
            training_seeds = {row["training_seed"] for row in group}
            if len(training_seeds) != 1:
                raise ValueError("each policy label must identify one training seed")
            label = f"Full, train seed {next(iter(training_seeds))}"
            axis.plot(sigmas, rates, marker=("o", "s", "^")[index % 3],
                      color=f"C{index % 10}", lw=1.6, ms=4.5, label=label)
        maximum = max(row["noise_std"] for row in selected)
        ticks = [maximum * index / 4 for index in range(5)] if maximum else [0]
        axis.set_xticks(ticks, [f"{sigma:g}" for sigma in ticks])
        axis.set_xlabel(f"Noise std per component ({unit})")
        axis.set_title(f"({'a' if kind == 'position' else 'b'}) {kind.capitalize()} measurement")
        axis.grid(alpha=.25)
        axis.set_ylim(-2, 103)
        axis.tick_params(labelsize=9)
        axis.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("Coupled-pose success (%)")
    fig.suptitle("Frozen policies at 1 mm / 0.025 rad", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "sensor_noise_compact.png", dpi=300)
    fig.savefig(output / "sensor_noise_compact.svg")
    plt.close(fig)


def write_report(output: Path, rows: list[dict], run: dict) -> None:
    unique_results = {row["result_file"] for row in rows}
    episodes = run["episodes"]
    lines = [
        "# L13 位置与姿态观测噪声：精简实验结果", "",
        f"完成 {len(unique_results)} 个独立评估条件，共 {len(unique_results) * episodes} 回合。"
        f"每个冻结策略、每个条件 {episodes} 回合；零噪声对照在两类曲线间共用。",
        f"测试任务块起点：{run['seeds']}；噪声 seed：{run['sensor_noise_seed']}。", "",
        "仅测试 L13（1 mm / 0.025 rad）。策略、原成功保持条件 H=5、原 horizon=500 保持冻结。"
        "每个控制周期独立扰动一个末端测量通道，另一通道保持准确。"
        "位置采用世界坐标加性高斯噪声，姿态采用左乘 SO(3) 小旋转；"
        "噪声等级均为三维向量每个分量的标准差。", "",
        "同一带噪位姿生成显式误差、关键点误差和精度动作缩放信号。"
        "关节状态、速度和基于关节模型的 Jacobian 不加噪。成功和误差统计使用真实状态。"
        "各条件共用配对的初始状态和目标，噪声 RNG 与任务 RNG 独立。", "",
    ]
    for kind, title, unit in (("position", "位置", "mm"), ("orientation", "姿态", "rad")):
        group = [row for row in rows if row["noise_type"] == kind]
        lines += [f"## {title}噪声", "",
                  f"| 分量标准差 ({unit}) | 两策略平均成功率 (%) | seed 11001 (%) | seed 11002 (%) | 位置 P95 (mm) | 姿态 P95 (rad) | 超时率 (%) |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for sigma in sorted({row["noise_std"] for row in group}):
            selected = [row for row in group if row["noise_std"] == sigma]
            seed_rates = [100 * mean([row for row in selected if row["training_seed"] == seed],
                                     "success_rate") if any(row["training_seed"] == seed for row in selected)
                          else float("nan") for seed in (11001, 11002)]
            lines.append(f"| {sigma:g} | {100 * mean(selected, 'success_rate'):.2f} | "
                         f"{seed_rates[0]:.2f} | {seed_rates[1]:.2f} | "
                         f"{mean(selected, 'p95_position_error_mm'):.3f} | "
                         f"{mean(selected, 'p95_orientation_error_rad'):.4f} | "
                         f"{100 * mean(selected, 'timeout_rate'):.2f} |")
        lines.append("")
    lines += [
        "P95 包含失败回合；表中 P95 是各策略/任务集 P95 的算术平均，"
        "不是把所有回合合并后重新计算的分位数。", "",
        "## 正文使用方式", "",
        "只需一张双子图、一段方法和一段结果讨论。图中分别保留两个训练种子，"
        "不把测试回合当作训练重复。原有 L6 先导结果可放补充材料，不与这里的独立测试任务混合。", "",
        "英文方法段：", "",
        f"We evaluated two frozen Full policies at the strict tolerance of 1 mm and 0.025 rad. "
        f"Each policy was tested on {episodes * len(run['seeds'])} paired episodes per noise condition. "
        "Position measurements were perturbed by zero-mean isotropic Gaussian translation noise, "
        "whereas orientation measurements were perturbed separately by left-multiplicative SO(3) "
        "rotations generated from isotropic Gaussian rotation vectors. The noise levels denote "
        "the standard deviation of each vector component. Perturbed measurements were used "
        "consistently for pose and keypoint features and precision-dependent action scaling. "
        "Success and terminal errors were determined from the ground-truth simulated state. "
        "Joint measurements and model-derived Jacobians remained unperturbed.", "",
        "图注建议：", "",
        "Robustness of frozen Full policies to position (a) and orientation (b) measurement noise "
        "at 1 mm / 0.025 rad. Curves correspond to independently trained policies. "
        "Noise is applied to one measurement channel at a time; the zero-noise evaluation "
        "is shared by both panels. Success is measured using the ground-truth state.", "",
        "## 限制", "",
        "只含两个训练种子及指定的测量噪声模型。未测试机械扰动、固定标定偏差、延迟、"
        "长期保持或真机；没有 Base 噪声对照，不能据此声称相较 Base 的鲁棒性增益。"
        "独立测试任务增多不等于增加独立训练种子。", "",
        "`summary.csv` 保存逐策略指标，逐条件 JSON 保存 checkpoint/config 哈希及噪声设置。"
        "图为 `sensor_noise_compact.png` / `.svg`，原始运行参数与代码哈希在 `run.json`。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--position-std-mm", nargs="+", type=float, default=[0, .1, .5, 1, 2])
    parser.add_argument("--orientation-std-rad", nargs="+", type=float, default=[0, .0025, .005, .01, .02])
    parser.add_argument("--seeds", nargs="+", type=int, default=[41001])
    parser.add_argument("--sensor-noise-seed", type=int, default=71001)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=40)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if len(args.labels) != len(args.checkpoint) or len(set(args.labels)) != len(args.labels):
        parser.error("unique labels must match checkpoint order")
    if args.episodes < 1 or args.num_envs < 1:
        parser.error("episodes and num-envs must be positive")
    for values in (args.position_std_mm, args.orientation_std_rad):
        if (not values or len(set(values)) != len(values)
                or any(not math.isfinite(value) or value < 0 for value in values) or 0 not in values):
            parser.error("noise levels must be finite, nonnegative, unique and include zero")
    if args.sensor_noise_seed < 0 or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or any(
        abs(a - b) < args.episodes
        for i, a in enumerate(args.seeds) for b in args.seeds[i + 1:]
    ):
        parser.error("test seed blocks must not overlap")
    for checkpoint in args.checkpoint:
        if not (ROOT / checkpoint).is_file():
            parser.error(f"checkpoint not found: {checkpoint}")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = {
        **vars(args), "protocol": "compact_separate_pose_sensor_noise_L13_v1",
        "status": "running", "level_index": 13,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in SOURCE_FILES},
        "task_pairing": "same test_seed + episode and same task-space bin",
        "noise_pairing": "same standard normal draw per channel, reset seed and control sample",
        "jacobian_source": "unperturbed joint/model state",
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    conditions = [("baseline", 0.0)]
    conditions += [("orientation", sigma) for sigma in args.orientation_std_rad if sigma > 0]
    conditions += [("position", sigma) for sigma in args.position_std_mm if sigma > 0]
    total = len(args.checkpoint) * len(args.seeds) * len(conditions)
    completed, rows = 0, []
    started = time.perf_counter()
    for index, (label, checkpoint) in enumerate(zip(args.labels, args.checkpoint)):
        for seed in args.seeds:
            for kind, sigma in conditions:
                stem = f"policy{index + 1}_{kind}_sigma{sigma:g}_seed{seed}"
                result = output / f"{stem}.json"
                command = [
                    sys.executable, str(ROOT / "scripts/core/evaluate_thesis_homotopy.py"),
                    "--checkpoint", checkpoint, "--checkpoint-config", "--level-index", "13",
                    "--episodes", str(args.episodes), "--seed", str(seed),
                    "--sensor-noise-seed", str(args.sensor_noise_seed),
                    "--num-envs", str(args.num_envs), "--no-cpu-affinity", "--output", str(result),
                ]
                if kind != "baseline":
                    flag = "--position-noise-std-mm" if kind == "position" else "--orientation-noise-std-rad"
                    command += [flag, str(sigma)]
                print(f"[{completed + 1}/{total}] {label} {kind} sigma={sigma:g}, seed={seed}", flush=True)
                with (output / f"{stem}.log").open("w") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                metrics = json.loads(result.read_text())
                if (metrics["position_tolerance_m"] != .001
                        or metrics["orientation_tolerance_rad"] != .025):
                    raise ValueError("compact benchmark requires the 1 mm / 0.025 rad L13 contract")
                for row_kind in (("position", "orientation") if kind == "baseline" else (kind,)):
                    rows.append({
                        "policy": label, "training_seed": metrics["training_seed"], "test_seed": seed,
                        "noise_type": row_kind, "noise_std": sigma,
                        "noise_unit": "mm" if row_kind == "position" else "rad", "episodes": args.episodes,
                        **{key: metrics[key] for key in (
                            "success_rate", "collision_rate", "timeout_rate", "mean_episode_steps",
                            "p95_orientation_error_rad", "checkpoint_sha256", "source_config_sha256",
                        )},
                        "p95_position_error_mm": 1000 * metrics["p95_position_error_m"],
                        "result_file": result.name,
                    })
                completed += 1
                save_csv(output, rows)
                print(f"  success={100 * metrics['success_rate']:.2f}% "
                      f"P95={1000 * metrics['p95_position_error_m']:.3f} mm / "
                      f"{metrics['p95_orientation_error_rad']:.4f} rad "
                      f"elapsed={metrics['elapsed_seconds']:.1f}s", flush=True)
    plot_results(output, rows)
    write_report(output, rows, run)
    run.update(status="complete", unique_conditions=completed,
               total_episodes=completed * args.episodes, elapsed_seconds=time.perf_counter() - started)
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"Completed {completed} unique conditions. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
