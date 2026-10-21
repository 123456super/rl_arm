#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as p
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def pose_tolerances(config: dict, eta: float) -> tuple[float, float]:
    joint = config["thesis"].get("joint_pose_curriculum", {}).get("levels")
    if joint:
        scales = np.asarray(
            config["thesis"]["orientation_curriculum"]["levels"], dtype=np.float64
        )
        index = int(np.argmin(np.abs(scales - float(eta))))
        if not np.isclose(scales[index], float(eta), rtol=0.0, atol=1e-7):
            raise ValueError(f"V12 orientation scale must be one of {scales.tolist()}")
        return (
            float(joint[index]["position_tolerance_m"]),
            float(joint[index]["orientation_tolerance_rad"]),
        )
    curriculum = config["thesis"]["orientation_curriculum"]
    start = float(curriculum["tolerance_start"])
    end = float(curriculum["tolerance_end"])
    return (
        float(config["thesis"]["position_tolerance"]),
        start + float(np.clip(eta, 0.0, 1.0)) * (end - start),
    )


def quaternion_from_z_axis(direction: np.ndarray) -> list[float]:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    z_axis = np.asarray([0.0, 0.0, 1.0])
    dot = float(np.dot(z_axis, direction))
    if dot < -1.0 + 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    xyz = np.cross(z_axis, direction)
    quaternion = np.asarray([xyz[0], xyz[1], xyz[2], 1.0 + dot])
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.tolist()


def add_visual_segment(
    client: int,
    start: np.ndarray,
    end: np.ndarray,
    color: list[float],
    radius: float,
) -> None:
    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        return
    shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=length,
        rgbaColor=color,
        physicsClientId=client,
    )
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=shape,
        baseCollisionShapeIndex=-1,
        basePosition=((np.asarray(start) + np.asarray(end)) * 0.5).tolist(),
        baseOrientation=quaternion_from_z_axis(vector),
        physicsClientId=client,
    )


def add_scene_visuals(env: ThesisHomotopyEnv) -> int:
    client = env.client_id
    floor_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.70, 0.70, 0.01],
        rgbaColor=[0.82, 0.84, 0.86, 1.0],
        physicsClientId=client,
    )
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=floor_shape,
        baseCollisionShapeIndex=-1,
        basePosition=[0.20, 0.0, -0.015],
        physicsClientId=client,
    )
    shape = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.025,
        rgbaColor=[0.1, 0.8, 0.2, 0.55],
        physicsClientId=client,
    )
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=shape,
        baseCollisionShapeIndex=-1,
        basePosition=env.goal_position.tolist(),
        physicsClientId=client,
    )
    rotation = np.asarray(
        p.getMatrixFromQuaternion(env.goal_quaternion.tolist()), dtype=np.float64
    ).reshape(3, 3)
    colors = ([0.9, 0.1, 0.1, 1.0], [0.1, 0.75, 0.2, 1.0], [0.1, 0.25, 0.95, 1.0])
    for axis, color in enumerate(colors):
        endpoint = env.goal_position + 0.10 * rotation[:, axis]
        add_visual_segment(client, env.goal_position, endpoint, list(color), 0.005)
    return p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.006,
        rgbaColor=[0.98, 0.55, 0.05, 0.9],
        physicsClientId=client,
    )


def add_trace_point(client: int, shape: int, position: np.ndarray) -> None:
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=shape,
        baseCollisionShapeIndex=-1,
        basePosition=np.asarray(position, dtype=np.float64).tolist(),
        physicsClientId=client,
    )


def capture_rgb(env: ThesisHomotopyEnv, args: argparse.Namespace) -> np.ndarray:
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=args.camera_target,
        distance=args.camera_distance,
        yaw=args.camera_yaw,
        pitch=args.camera_pitch,
        roll=0.0,
        upAxisIndex=2,
    )
    projection = p.computeProjectionMatrixFOV(
        fov=50.0,
        aspect=float(args.width) / float(args.height),
        nearVal=0.05,
        farVal=4.0,
    )
    renderer = p.ER_BULLET_HARDWARE_OPENGL if args.gui else p.ER_TINY_RENDERER
    _, _, rgba, _, _ = p.getCameraImage(
        width=args.width,
        height=args.height,
        viewMatrix=view,
        projectionMatrix=projection,
        shadow=1,
        renderer=renderer,
        physicsClientId=env.client_id,
    )
    return np.asarray(rgba, dtype=np.uint8).reshape(args.height, args.width, 4)[:, :, :3]


def save_annotated(
    path: Path,
    rgb: np.ndarray,
    *,
    episode: int,
    step: int,
    eta: float,
    info: dict,
    label: str,
) -> None:
    image = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_size = max(14, int(image.height / 36))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    text = (
        f"{label} | episode {episode} | step {step}\n"
        f"eta={eta:.2f}  position={float(info['goal_error_norm']):.4f} m  "
        f"orientation={float(info['orientation_error_norm']):.3f} rad\n"
        f"self clearance={float(info['self_d_min']):.4f} m"
    )
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=6)
    width = box[2] - box[0] + 24
    height = box[3] - box[1] + 20
    draw.rectangle((12, 12, 12 + width, 12 + height), fill=(250, 250, 250, 220))
    draw.multiline_text(
        (24, 22), text, fill=(20, 20, 20, 255), font=font, spacing=6
    )
    Image.alpha_composite(image, overlay).convert("RGB").save(path, quality=95)


def outcome(info: dict, terminated: bool, truncated: bool, capture_limited: bool) -> str:
    if info.get("obstacle_collision"):
        return "obstacle_collision"
    if info.get("self_collision"):
        return "self_collision"
    if info.get("environment_collision"):
        return "environment_collision"
    if info.get("joint_limit"):
        return "joint_limit"
    if info.get("safe_success"):
        return "safe_success"
    if truncated:
        return "timeout"
    if capture_limited and not terminated:
        return "capture_limit"
    return "terminated"


def encode_video(frame_dir: Path, destination: Path, fps: int) -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%06d.png"), "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    subprocess.run(command, check=True)


def generate_framework_figure(output: Path) -> None:
    """Generate the complete Full learning loop."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from matplotlib.path import Path as MplPath

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    ink = "#20262c"
    muted = "#5f6870"
    neutral_edge = "#7a838a"
    neutral_fill = "#f1f3f4"
    scaling_fill = "#f8f9f9"
    accent, accent_fill = "#2468a2", "#eaf2f8"
    update_red = "#b23a32"

    fig, axis = plt.subplots(figsize=(7.15, 2.48))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    def box(x, y, width, height, *, face, edge, radius=.018, linewidth=.9):
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=face, edgecolor=edge, linewidth=linewidth,
            zorder=2,
        )
        axis.add_patch(patch)
        return patch

    def arrow(
        start, end, *, color=ink, linewidth=1.0, connection="arc3",
        linestyle="solid",
    ):
        patch = FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=8.5,
            linewidth=linewidth, color=color, connectionstyle=connection,
            linestyle=linestyle, shrinkA=0, shrinkB=0, zorder=5,
        )
        axis.add_patch(patch)
        return patch

    def orthogonal_arrow(
        points, *, color=ink, linewidth=1.0, linestyle="solid", radius=.012,
    ):
        """Draw a continuous right-angle connector with gently rounded bends."""
        pts = np.asarray(points, dtype=float)
        vertices = [tuple(pts[0])]
        codes = [MplPath.MOVETO]
        for index in range(1, len(pts) - 1):
            previous, corner, following = pts[index - 1:index + 2]
            incoming = corner - previous
            outgoing = following - corner
            incoming_length = float(np.linalg.norm(incoming))
            outgoing_length = float(np.linalg.norm(outgoing))
            bend = min(radius, incoming_length / 2.0, outgoing_length / 2.0)
            entry = corner - bend * incoming / incoming_length
            exit_ = corner + bend * outgoing / outgoing_length
            vertices.extend([tuple(entry), tuple(corner), tuple(exit_)])
            codes.extend([MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3])
        vertices.append(tuple(pts[-1]))
        codes.append(MplPath.LINETO)
        patch = FancyArrowPatch(
            path=MplPath(vertices, codes), arrowstyle="-|>", mutation_scale=8.5,
            linewidth=linewidth, color=color, linestyle=linestyle,
            capstyle="round", joinstyle="round", zorder=5,
        )
        axis.add_patch(patch)
        return patch

    def connector_label(x, y, text, *, color=muted, size=5.8, rotation=0):
        return axis.text(
            x, y, text, ha="center", va="center", fontsize=size, color=color,
            rotation=rotation, zorder=7,
        )

    top_nodes = [
        (.035, .59, .225, .22, accent_fill, accent,
         "Full observation (162-D)", None),
        (.315, .61, .13, .18, neutral_fill, neutral_edge,
         "SAC actor", r"$a_t\sim\pi_\theta(\cdot\mid s_t)$"),
        (.50, .61, .15, .18, scaling_fill, neutral_edge,
         "Precision scaling", r"$\dot q_{\rm cmd}=g(\rho_t)(0.7a_t)$"),
        (.705, .59, .23, .22, neutral_fill, neutral_edge,
         "PyBullet UR5", "joint-velocity control\nstate transition"),
    ]
    for x, y, width, height, face, edge, title, body in top_nodes:
        box(x, y, width, height, face=face, edge=edge, linewidth=1.0)
        axis.text(x + width / 2, y + height - .052, title, ha="center", va="center",
                  fontsize=7.15, fontweight="bold",
                  color=accent if edge == accent else ink)
        if body is not None:
            axis.text(x + width / 2, y + .060, body, ha="center", va="center",
                      fontsize=6.05, color=muted, linespacing=1.22)

    observation_center = .035 + .225 / 2
    axis.text(observation_center, .696, "pose + keypoint errors",
              ha="center", va="center", fontsize=6.0, color=ink)
    axis.text(observation_center, .659, r"Keypoint Jacobians $J_{1:3}(q)$",
              ha="center", va="center", fontsize=6.1, color=accent,
              fontweight="bold")
    axis.text(observation_center, .622, "joint / velocity / safety",
              ha="center", va="center", fontsize=6.0, color=ink)

    for start, end in [
        ((.26, .70), (.315, .70)),
        ((.445, .70), (.50, .70)),
        ((.65, .70), (.705, .70)),
    ]:
        arrow(start, end)
    connector_label(.2875, .731, r"$s_t$", size=6.1)
    connector_label(.4725, .731, r"$a_t$", size=6.1)
    connector_label(.6775, .731, r"$\dot q_{\rm cmd}$", size=6.1)

    orthogonal_arrow(
        [(.82, .81), (.82, .90), (.1475, .90), (.1475, .81)],
        color=neutral_edge, linewidth=1.1,
    )
    connector_label(.485, .932, r"state feedback $s_{t+1}$", size=6.15)

    bottom_nodes = [
        (.705, .13, .23, .25, accent_fill, accent,
         "Coupled precision objective",
         "Pose / keypoint errors\n" + r"$\downarrow$" +
         "\nPrecision-quality + bottleneck"),
        (.39, .13, .25, .25, accent_fill, accent,
         "Precision-region replay",
         "Uniform / Success / Boundary\n0.6 / 0.2 / 0.2\n" +
         r"Boundary: $0.8\leq\rho_t^l\leq1.5$"),
        (.065, .13, .255, .25, neutral_fill, neutral_edge,
         "SAC optimization",
         "twin critics + entropy objective\ntarget-network update"),
    ]
    for x, y, width, height, face, edge, title, body in bottom_nodes:
        box(x, y, width, height, face=face, edge=edge, linewidth=1.0)
        axis.text(x + width / 2, y + height - .054, title, ha="center", va="center",
                  fontsize=7.05, fontweight="bold",
                  color=accent if edge == accent else ink)
        axis.text(x + width / 2, y + .077, body, ha="center", va="center",
                  fontsize=6.05, color=ink if edge == accent else muted,
                  linespacing=1.25)

    for start, end in [
        ((.705, .255), (.64, .255)),
        ((.39, .255), (.32, .255)),
    ]:
        arrow(start, end)
    connector_label(.6725, .286, r"$r_t$", size=6.2)
    connector_label(.355, .313, "stratified\nminibatch", size=5.55)
    arrow((.82, .59), (.82, .38), linewidth=1.05)
    connector_label(.85, .485, "errors", size=5.8, rotation=90)
    orthogonal_arrow(
        [(.1925, .38), (.1925, .475), (.38, .475), (.38, .61)],
        color=update_red, linewidth=1.1, radius=0,
    )
    connector_label(.286, .503, "parameter update", color=update_red,
                    size=5.8)

    fig.subplots_adjust(left=.008, right=.992, bottom=.025, top=.975)
    fig.savefig(output / "fig1_framework.pdf", bbox_inches="tight", pad_inches=.015)
    plt.close(fig)


def generate_paper_summary_figures(output: Path) -> None:
    """Generate compact vector framework, curriculum, and result figures."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

    output.mkdir(parents=True, exist_ok=True)
    generate_framework_figure(output)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 7.6,
        "legend.fontsize": 6.4,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    levels = np.arange(14)
    position_mm = np.asarray(
        [100, 80, 64, 51.2, 50, 25, 10, 5, 2.5, 2, 1.5, 1, 1, 1],
        dtype=np.float64,
    )
    orientation_rad = np.asarray(
        [.30, .24, .192, .1536, .10, .10, .10, .10, .10, .10, .10, .10, .05, .025],
        dtype=np.float64,
    )
    red = "#b22d2d"
    blue = "#2468a2"
    gray = "#697077"
    grid = "#d7dadd"

    fig, axes = plt.subplots(2, 1, figsize=(3.48, 2.55), sharex=True)
    axes[0].step(levels, position_mm, where="mid", color=blue, linewidth=1.6)
    axes[0].scatter(levels, position_mm, s=13, color=blue, edgecolor="white", linewidth=.35, zorder=3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Position tol. (mm)")
    axes[0].set_ylim(.75, 135)
    axes[0].yaxis.set_major_locator(FixedLocator([1, 2.5, 5, 10, 25, 50, 100]))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[0].yaxis.set_minor_locator(NullLocator())
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].text(11.95, 1.23, "1 mm plateau", color=blue, ha="center", va="bottom", fontsize=6.2)

    axes[1].step(levels, orientation_rad, where="mid", color=red, linewidth=1.6)
    axes[1].scatter(levels, orientation_rad, s=13, color=red, edgecolor="white", linewidth=.35, zorder=3)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Orientation tol. (rad)")
    axes[1].set_xlabel("Curriculum level")
    axes[1].set_ylim(.019, .38)
    axes[1].yaxis.set_major_locator(FixedLocator([.025, .05, .10, .20, .30]))
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[1].yaxis.set_minor_locator(NullLocator())
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xticks(levels)
    axes[1].set_xticklabels([f"L{level}" for level in levels], rotation=45, ha="right")
    axes[1].annotate(
        "orientation tightening",
        xy=(13, .025), xytext=(9.0, .037), color=red,
        arrowprops={"arrowstyle": "->", "color": red, "lw": .8},
        fontsize=6.2, ha="center",
    )
    for axis in axes:
        axis.grid(axis="y", color=grid, linewidth=.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.5, width=.6)
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=.18, right=.98, top=.98, bottom=.24, hspace=.18)
    fig.savefig(output / "fig_curriculum_schedule.pdf", bbox_inches="tight", pad_inches=.02)
    plt.close(fig)

    methods = ["Base", "Full", "M1", "M2", "M3"]
    common_success = np.asarray([84.75, 96.35, 95.90, 97.50, 91.13])
    common_std = np.asarray([.62, .36, .40, .26, .95])
    highest_level = np.asarray([6.5, 13, 13, 13, 11])
    highest_success = [None, 97.20, 97.60, 89.23, 77.13]
    colors = [gray, red, blue, blue, blue]
    markers = ["s", "o", "^", "D", "v"]
    y = np.arange(len(methods))[::-1]

    fig, axes = plt.subplots(2, 1, figsize=(3.48, 3.45))
    for index, (value, deviation, color, marker) in enumerate(
        zip(common_success, common_std, colors, markers)
    ):
        axes[0].errorbar(
            value, y[index], xerr=deviation, fmt=marker, color=color,
            markersize=4.6, capsize=2.2, elinewidth=.9, markeredgewidth=.5,
        )
        axes[0].text(value + 1.0, y[index], f"{value:.2f}", va="center", fontsize=6.4)
    axes[0].axvline(85, color=gray, linestyle="--", linewidth=.8)
    axes[0].text(85.3, 4.38, "promotion 85%", color=gray, fontsize=6.0, va="bottom")
    axes[0].set_xlim(82, 101)
    axes[0].set_yticks(y, methods)
    axes[0].set_xlabel("Success at common L6 (%)")
    axes[0].set_title("(a) Common-tolerance comparison", loc="left", pad=3)

    for index, (level, color, marker, success) in enumerate(
        zip(highest_level, colors, markers, highest_success)
    ):
        if index == 0:
            axes[1].plot([6, 7], [y[index], y[index]], color=color, linewidth=1.1)
            axes[1].plot([6, 7], [y[index], y[index]], "|", color=color, markersize=6)
            label = "L6-L7"
        else:
            axes[1].plot(level, y[index], marker=marker, color=color, markersize=4.8)
            label = f"L{int(level)}  {success:.2f}%"
        axes[1].text(level + .35, y[index], label, va="center", fontsize=6.4)
    axes[1].axvline(13, color=grid, linewidth=.7, zorder=0)
    axes[1].set_xlim(5.5, 15.2)
    axes[1].set_xticks([6, 7, 8, 9, 10, 11, 12, 13])
    axes[1].set_yticks(y, methods)
    axes[1].set_xlabel("Highest curriculum level reached")
    axes[1].set_title("(b) Fixed-budget progression", loc="left", pad=3)
    axes[1].text(
        .99, .03, "labels show success at the highest level",
        transform=axes[1].transAxes, ha="right", va="bottom", color=gray, fontsize=5.9,
    )
    for axis in axes:
        axis.grid(axis="x", color=grid, linewidth=.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.5, width=.6)
        axis.set_ylim(-.65, 4.65)
    fig.subplots_adjust(left=.19, right=.98, top=.97, bottom=.15, hspace=.46)
    fig.savefig(output / "fig_result_overview.pdf", bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a thesis checkpoint in PyBullet and capture report figures"
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--config", default="configs/experiments/paper_full.yaml")
    parser.add_argument("--scene", choices=("none", "static", "dynamic"), default="none")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=61001)
    parser.add_argument("--goal-scale", type=float)
    parser.add_argument("--orientation-scale", type=float)
    parser.add_argument("--lambda-self", type=float)
    parser.add_argument("--output")
    parser.add_argument("--gui", action="store_true", help="open the interactive PyBullet GUI")
    parser.add_argument("--video", action="store_true", help="also encode an annotated MP4")
    parser.add_argument("--hold", action="store_true", help="keep the GUI open until Enter is pressed")
    parser.add_argument("--no-realtime", action="store_true", help="do not pace GUI playback at 20 Hz")
    parser.add_argument("--max-steps", type=int, help="presentation/debug capture limit")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera-target", nargs=3, type=float, default=[0.35, 0.0, 0.32])
    parser.add_argument("--camera-distance", type=float, default=1.05)
    parser.add_argument("--camera-yaw", type=float, default=50.0)
    parser.add_argument("--camera-pitch", type=float, default=-25.0)
    parser.add_argument(
        "--paper-summary-figures", action="store_true",
        help="generate vector curriculum and result-summary figures without a checkpoint",
    )
    args = parser.parse_args()

    if args.paper_summary_figures:
        destination = resolve_path(args.output) if args.output else (
            ROOT / "docs/paper/manuscript_en/figures"
        )
        generate_paper_summary_figures(destination)
        print(f"saved paper summary figures to {destination}")
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --paper-summary-figures is used")

    if args.episodes < 1 or args.width < 16 or args.height < 16 or args.fps < 1:
        parser.error("episodes, width, height, and fps must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")

    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    protocol = str(config["thesis"]["protocol"])
    if checkpoint.get("protocol") != protocol:
        raise ValueError(
            f"checkpoint protocol {checkpoint.get('protocol')!r} does not match {protocol!r}"
        )

    curriculum_state = checkpoint["curriculum"]
    checkpoint_goal_scale = float(curriculum_state["goal"].scale)
    checkpoint_eta = float(curriculum_state["orientation"].scale)
    self_state = curriculum_state.get("self_safety")
    checkpoint_lambda_self = float(self_state.weight) if self_state is not None else 1.0
    goal_scale = checkpoint_goal_scale if args.goal_scale is None else float(args.goal_scale)
    eta = checkpoint_eta if args.orientation_scale is None else float(args.orientation_scale)
    lambda_self = (
        checkpoint_lambda_self if args.lambda_self is None else float(args.lambda_self)
    )
    position_tolerance, current_orientation_tolerance = pose_tolerances(config, eta)
    if not 0.0 < goal_scale <= 1.0:
        parser.error("--goal-scale must be in (0, 1]")
    if not 0.0 <= eta <= 1.0 or not 0.0 <= lambda_self <= 1.0:
        parser.error("--orientation-scale and --lambda-self must be in [0, 1]")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = resolve_path(args.output) if args.output else (
        ROOT / config["train"]["output_dir"] / "visualizations"
        / f"{checkpoint_path.parent.parent.name}_{args.scene}_{stamp}"
    )
    output.mkdir(parents=True, exist_ok=False)

    env = ThesisHomotopyEnv(config, render_mode="human" if args.gui else "rgb_array")
    agent = ThesisSACAgent(
        int(env.observation_space.shape[0]), int(env.action_space.shape[0]), config
    )
    agent.load_state_dict(checkpoint["agent"])
    if args.gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.client_id)
        p.resetDebugVisualizerCamera(
            args.camera_distance, args.camera_yaw, args.camera_pitch, args.camera_target,
            physicsClientId=env.client_id,
        )

    rows: list[dict] = []
    try:
        for episode in range(args.episodes):
            env.configure_episode(
                args.scene,
                xi=1.0,
                strict=True,
                goal_scale=goal_scale,
                orientation_scale=eta,
                position_tolerance=position_tolerance,
                orientation_tolerance=current_orientation_tolerance,
                lambda_self=lambda_self,
            )
            observation, info = env.reset(seed=args.seed + episode)
            trace_shape = add_scene_visuals(env)
            frame_dir = output / f"episode_{episode:03d}_frames"
            if args.video:
                frame_dir.mkdir()
            frame_index = 0
            rgb = capture_rgb(env, args)
            save_annotated(
                output / f"episode_{episode:03d}_initial.png", rgb,
                episode=episode, step=0, eta=eta, info=info, label="initial",
            )
            if args.video:
                save_annotated(
                    frame_dir / f"frame_{frame_index:06d}.png", rgb,
                    episode=episode, step=0, eta=eta, info=info, label="running",
                )
                frame_index += 1

            terminated = truncated = False
            step = 0
            best_score = float("inf")
            total_reward = 0.0
            previous_ee = np.asarray(info["ee_position"], dtype=np.float64)
            while not (terminated or truncated):
                if args.max_steps is not None and step >= args.max_steps:
                    break
                started = time.perf_counter()
                action = agent.select_action(observation, deterministic=True)
                observation, reward, _, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                step += 1
                current_ee = np.asarray(info["ee_position"], dtype=np.float64)
                p.addUserDebugLine(
                    previous_ee.tolist(), current_ee.tolist(), [0.95, 0.65, 0.05], 2.0,
                    physicsClientId=env.client_id,
                )
                previous_ee = current_ee
                if step == 1 or step % 5 == 0 or terminated or truncated:
                    add_trace_point(env.client_id, trace_shape, current_ee)
                score = (
                    float(info["goal_error_norm"]) / float(config["thesis"]["position_tolerance"])
                    + eta * float(info["orientation_error_norm"])
                    / max(current_orientation_tolerance, 1e-9)
                )
                need_rgb = args.video or score < best_score or terminated or truncated
                if need_rgb:
                    rgb = capture_rgb(env, args)
                if score < best_score:
                    best_score = score
                    save_annotated(
                        output / f"episode_{episode:03d}_best.png", rgb,
                        episode=episode, step=step, eta=eta, info=info, label="best so far",
                    )
                if args.video:
                    save_annotated(
                        frame_dir / f"frame_{frame_index:06d}.png", rgb,
                        episode=episode, step=step, eta=eta, info=info, label="running",
                    )
                    frame_index += 1
                if args.gui and not args.no_realtime:
                    elapsed = time.perf_counter() - started
                    time.sleep(max(0.0, env.control_dt - elapsed))

            capture_limited = bool(
                args.max_steps is not None and step >= args.max_steps
                and not (terminated or truncated)
            )
            result = outcome(info, terminated, truncated, capture_limited)
            terminal_rgb = capture_rgb(env, args)
            save_annotated(
                output / f"episode_{episode:03d}_terminal_{result}.png", terminal_rgb,
                episode=episode, step=step, eta=eta, info=info, label=result,
            )
            if args.video:
                encode_video(frame_dir, output / f"episode_{episode:03d}.mp4", args.fps)

            rows.append({
                "episode": episode,
                "seed": args.seed + episode,
                "scene": args.scene,
                "goal_scale": goal_scale,
                "orientation_scale": eta,
                "orientation_tolerance": orientation_tolerance(config, eta),
                "lambda_self": lambda_self,
                "steps": step,
                "return": total_reward,
                "outcome": result,
                "safe_success": int(bool(info.get("safe_success"))),
                "self_collision": int(bool(info.get("self_collision"))),
                "joint_limit": int(bool(info.get("joint_limit"))),
                "timeout": int(bool(truncated)),
                "position_error_m": float(info["goal_error_norm"]),
                "orientation_error_rad": float(info["orientation_error_norm"]),
                "minimum_self_clearance_m": float(info["control_self_min_distance"]),
            })
    finally:
        if args.gui and args.hold:
            input("Press Enter to close the PyBullet GUI...")
        env.close()

    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "purpose": "presentation visualization only; not validation or Gate evidence",
        "protocol": protocol,
        "checkpoint": str(checkpoint_path.relative_to(ROOT) if checkpoint_path.is_relative_to(ROOT) else checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config": str(config_path.relative_to(ROOT) if config_path.is_relative_to(ROOT) else config_path),
        "config_sha256": file_sha256(config_path),
        "scene": args.scene,
        "episodes": args.episodes,
        "seed": args.seed,
        "goal_scale": goal_scale,
        "orientation_scale": eta,
        "orientation_tolerance": orientation_tolerance(config, eta),
        "lambda_self": lambda_self,
        "deterministic_actor": True,
        "strict_collision_semantics": True,
        "gui": args.gui,
        "video": args.video,
        "camera": {
            "target": args.camera_target,
            "distance": args.camera_distance,
            "yaw": args.camera_yaw,
            "pitch": args.camera_pitch,
            "width": args.width,
            "height": args.height,
        },
        "results": rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved visualization to {output}")


if __name__ == "__main__":
    main()
