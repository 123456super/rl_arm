"""Generate publication-ready vector figures for the English manuscript.

Run with the project environment:
    conda run -n rl python docs/paper/figures_src/make_vector_figures.py

Figure 1 reads the actual UR5 pose and target from ThesisHomotopyEnv. All
figures are saved as PDF, so LaTeX embeds vector paths and text.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from PIL import Image
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/paper/manuscript_en/figures"
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.thesis_reaching import pose_keypoint_offsets_world
from rl_risk_sac.utils.config import load_config

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.65,
})

INK = "#222222"
MID = "#5D6872"
GRID = "#D9DEE2"
BLUE = "#2F5D8C"
RED = "#A33D3D"
GREEN = "#3E7650"
GOLD = "#A06A20"
PURPLE = "#6D5A86"


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def quat_matrix(q):
    return np.asarray(p.getMatrixFromQuaternion(np.asarray(q).tolist()), dtype=float).reshape(3, 3)


def project(points):
    """Oblique scientific projection: horizontal x/y, vertical z."""
    points = np.asarray(points)
    return np.c_[points[:, 0] - 0.62 * points[:, 1], points[:, 2] + 0.24 * points[:, 1]]


def actual_scene():
    config = load_config(ROOT / "configs/experiments/paper_full.yaml")
    env = ThesisHomotopyEnv(config, render_mode="rgb_array")
    env.configure_episode("none", xi=1.0, strict=True, goal_scale=1.0,
                         orientation_scale=1.0, position_tolerance=0.001,
                         orientation_tolerance=0.025, lambda_self=0.2)
    _, info = env.reset(seed=31001)
    client = env.client_id
    robot = env.robot
    link_ids = list(robot.joint_ids) + [robot.tool_link_id]
    link_points = [np.asarray(p.getLinkState(robot.robot_id, lid, computeForwardKinematics=True,
                                               physicsClientId=client)[4], dtype=float)
                   for lid in link_ids]
    ee_pos, ee_quat = robot.end_effector_pose()
    goal_pos = np.asarray(env.goal_position, dtype=float)
    goal_quat = np.asarray(env.goal_quaternion, dtype=float)
    key_current = np.asarray(ee_pos) + pose_keypoint_offsets_world(ee_quat, cube_side_m=env.keypoint_cube_side_m)
    key_goal = goal_pos + pose_keypoint_offsets_world(goal_quat, cube_side_m=env.keypoint_cube_side_m)
    env.close()
    return np.asarray(link_points), np.asarray(ee_pos), np.asarray(ee_quat), goal_pos, goal_quat, key_current, key_goal


def capture_pybullet_scene():
    """Capture a clean, unannotated render from the actual project environment."""
    config = load_config(ROOT / "configs/experiments/paper_full.yaml")
    env = ThesisHomotopyEnv(config, render_mode="rgb_array")
    env.configure_episode("none", xi=1.0, strict=True, goal_scale=1.0,
                         orientation_scale=1.0, position_tolerance=0.001,
                         orientation_tolerance=0.025, lambda_self=0.2)
    env.reset(seed=31001)
    client = env.client_id
    # Flat ground and target pose, using the same world state as the evaluation environment.
    floor_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.70, 0.70, 0.01],
                                      rgbaColor=[0.86, 0.87, 0.88, 1.0], physicsClientId=client)
    p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=floor_shape,
                      basePosition=[0.20, 0.0, -0.015], physicsClientId=client)
    sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.025,
                                 rgbaColor=[0.20, 0.55, 0.28, 1.0], physicsClientId=client)
    p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=sphere,
                      basePosition=env.goal_position.tolist(), physicsClientId=client)
    R = quat_matrix(env.goal_quaternion)
    for axis, color in enumerate(([0.72, 0.16, 0.16, 1.0], [0.18, 0.48, 0.25, 1.0], [0.15, 0.28, 0.65, 1.0])):
        start = env.goal_position
        end = env.goal_position + 0.11 * R[:, axis]
        vec = end - start; length = float(np.linalg.norm(vec)); mid = (start + end) / 2
        z = np.array([0.0, 0.0, 1.0]); v = vec / length; dot = float(np.dot(z, v))
        if dot < -0.999999: q = [1, 0, 0, 0]
        else:
            xyz = np.cross(z, v); q = np.r_[xyz, 1 + dot]; q = (q / np.linalg.norm(q)).tolist()
        shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.0045, length=length,
                                    rgbaColor=color, physicsClientId=client)
        p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=shape,
                          basePosition=mid.tolist(), baseOrientation=q, physicsClientId=client)
    view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.37, -0.03, 0.39],
                                               distance=1.15, yaw=48, pitch=-27, roll=0, upAxisIndex=2)
    projection = p.computeProjectionMatrixFOV(fov=48, aspect=1.48, nearVal=0.05, farVal=4.0)
    _, _, rgba, _, _ = p.getCameraImage(width=1480, height=1000, viewMatrix=view,
                                        projectionMatrix=projection, shadow=1,
                                        renderer=p.ER_TINY_RENDERER, physicsClientId=client)
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(1000, 1480, 4)[:, :, :3]
    env.close()
    path = OUT / "fig1_pybullet_render.png"
    Image.fromarray(rgb).save(path)
    return path


def fig1():
    links, ee, ee_q, goal, goal_q, key_cur, key_goal = actual_scene()
    render_path = capture_pybullet_scene()
    fig = plt.figure(figsize=(7.0, 3.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 0.88])
    ax_render = fig.add_subplot(gs[0, 0]); ax = fig.add_subplot(gs[0, 1])
    ax_render.imshow(Image.open(render_path))
    ax_render.set_axis_off()
    ax_render.text(0.02, 0.96, "(a) PyBullet scene", transform=ax_render.transAxes,
                   ha="left", va="top", fontsize=9.5, color="white",
                   bbox={"facecolor": "black", "alpha": 0.42, "pad": 2.5, "edgecolor": "none"})
    ax.set_aspect("equal")
    ax.set_axis_off()
    # The right panel is deliberately abstract: two pose frames and three pairs of points.
    kc, kg = project(key_cur), project(key_goal)
    def frame(pos, quat, scale, color, linestyle="-"):
        o = project(np.asarray(pos)[None, :])[0]
        R = quat_matrix(quat)
        ends = np.asarray(pos) + scale * R.T
        ep = project(ends)
        for endpoint in ep:
            ax.plot([o[0], endpoint[0]], [o[1], endpoint[1]], color=color,
                    linewidth=0.85, linestyle=linestyle, zorder=4)
    frame(ee, ee_q, 0.095, INK, "-")
    frame(goal, goal_q, 0.12, BLUE, (0, (3, 2)))
    ax.plot(*np.vstack([kc, kc[0]]).T, color=INK, linewidth=1.15, zorder=5)
    ax.plot(*np.vstack([kg, kg[0]]).T, color=BLUE, linewidth=1.0, linestyle=(0, (3, 2)), zorder=5)
    for i, (a, b) in enumerate(zip(kc, kg), start=1):
        ax.annotate("", xy=b, xytext=a,
                    arrowprops={"arrowstyle": "-|>", "lw": 0.85, "color": GOLD})
        ax.scatter(*a, s=22, color=INK, zorder=7)
        ax.scatter(*b, s=25, facecolor="white", edgecolor=BLUE, linewidth=1.0, zorder=7)
        ax.text(a[0] - 0.02, a[1] - 0.025, "$x_{%d}$" % i, fontsize=8, color=INK)
        ax.text(b[0] + 0.012, b[1] + 0.012, "$x^{\\mathrm{g}}_{%d}$" % i, fontsize=8, color=BLUE)
    ax.text(0.02, 0.97, "(b) Pose and keypoint geometry", transform=ax.transAxes,
            ha="left", va="top", fontsize=10, color=INK)
    ax.text(0.02, 0.03, "solid: current pose    dashed: target pose    arrows: $e_i=x_i^g-x_i$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=MID)
    all_xy = np.vstack([kc, kg, project(np.asarray(goal)[None, :]), project(np.asarray(ee)[None, :])])
    xmin, ymin = all_xy.min(axis=0) - np.array([0.16, 0.14])
    xmax, ymax = all_xy.max(axis=0) + np.array([0.16, 0.14])
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    save(fig, "fig1_ur5_task")


def box(ax, xy, wh, title, body, face):
    x, y = xy; w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                           facecolor=face, edgecolor=MID, linewidth=0.75)
    ax.add_patch(patch)
    ax.text(x + 0.02, y + h - 0.040, title, fontsize=8.8, fontweight="bold", color=INK, va="top")
    ax.text(x + 0.02, y + h - 0.092, body, fontsize=7.1, color=INK, va="top", linespacing=1.25)


def framework():
    """Jiang-style training architecture: innovation modules are the visual focus."""
    fig, ax = plt.subplots(figsize=(7.0, 3.55))
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    def rounded(x, y, w, h, title, body, face, edge=MID, lw=0.8, title_fs=8.0, body_fs=6.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                    facecolor=face, edgecolor=edge, linewidth=lw))
        ax.text(x+w/2, y+h-0.030, title, ha="center", va="top", fontsize=title_fs,
                fontweight="bold", color=INK)
        if body:
            ax.text(x+w/2, y+0.025, body, ha="center", va="bottom", fontsize=body_fs,
                    color=INK, linespacing=1.15)
    def network(x, y, w, h, title, body):
        # Use the same rounded-rectangle grammar for networks and process modules.
        rounded(x, y, w, h, title, body, "#F1F2F3", MID, title_fs=7.3, body_fs=6.0)
    def arrow(points, color=INK, style="-", mutation=9):
        pts = list(points)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=0.85,
                linestyle=style, solid_capstyle="butt", zorder=3)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=mutation,
                                     linewidth=0.85, color=color, linestyle=style, zorder=4))

    # Central contribution block, using one consistent accent as requested.
    ax.add_patch(FancyBboxPatch((0.13, 0.59), 0.74, 0.30,
                                boxstyle="round,pad=0.018,rounding_size=0.025",
                                facecolor="#FFF9FA", edgecolor=RED, linewidth=1.0,
                                linestyle=(0, (3, 2))))
    ax.text(0.50, 0.855, "Local Convergence Enhancement", ha="center", va="center",
            fontsize=10.0, fontweight="bold", color=RED)
    rounded(0.18, 0.655, 0.19, 0.145, "Keypoint Jacobian", "$J_k(q) \u2192 s_t", "#F4D8DE", RED, title_fs=6.9, body_fs=6.0)
    rounded(0.405, 0.655, 0.19, 0.145, "Coupled objective", "$e_p,e_R \u2192 r_t", "#F4D8DE", RED, title_fs=6.9, body_fs=6.0)
    rounded(0.63, 0.655, 0.19, 0.145, "Precision-boundary replay", "sampling strategy", "#F4D8DE", RED, title_fs=6.1, body_fs=5.8)

    # Base SAC data path.
    ax.text(0.012, 0.46, "Joint state\nPose errors\nKeypoint errors\nLocal kinematics", ha="left", va="center", fontsize=6.5, color=INK)
    network(0.22, 0.37, 0.16, 0.15, "Policy network", "$\\pi_\\theta(s_t)$")
    rounded(0.46, 0.385, 0.16, 0.13, "PyBullet UR5", "state transition", "#F1F2F3", MID)
    rounded(0.70, 0.385, 0.16, 0.13, "Replay buffer", "transition store", "#DDEFEA", "#3D8274", title_fs=7.0, body_fs=5.8)
    ax.text(0.78, 0.355, "uniform / success / boundary", ha="center", va="top", fontsize=5.7, color="#3D8274")
    network(0.40, 0.10, 0.18, 0.15, "Twin critics", "$Q_1,Q_2$")
    rounded(0.65, 0.12, 0.18, 0.15, "SAC Update", "actor + critic", "#DDEFEA", "#3D8274", title_fs=7.2, body_fs=6.0)

    arrow([(0.18, 0.46), (0.22, 0.46)])
    arrow([(0.38, 0.445), (0.46, 0.445)])
    arrow([(0.62, 0.45), (0.70, 0.45)])
    arrow([(0.78, 0.37), (0.78, 0.285), (0.49, 0.285), (0.49, 0.25)], MID)
    ax.text(0.64, 0.30, "boundary-aware minibatch", ha="center", va="bottom", fontsize=5.6, color="#3D8274")
    arrow([(0.58, 0.175), (0.65, 0.175)], INK)
    arrow([(0.74, 0.12), (0.74, 0.06), (0.19, 0.06), (0.19, 0.34), (0.30, 0.37)], RED, ":")
    arrow([(0.54, 0.515), (0.54, 0.61), (0.50, 0.61), (0.50, 0.655)], RED, ":")
    ax.text(0.555, 0.615, "$e_p,e_R$", ha="left", va="center", fontsize=5.8, color=RED)
    arrow([(0.50, 0.655), (0.50, 0.55), (0.78, 0.55), (0.78, 0.515)], RED, ":")
    ax.text(0.64, 0.535, "$r_t$", ha="center", va="bottom", fontsize=5.8, color=RED)
    arrow([(0.275, 0.655), (0.275, 0.515), (0.30, 0.515)], RED, ":")
    arrow([(0.725, 0.655), (0.725, 0.57), (0.86, 0.57), (0.86, 0.515)], RED, ":")
    save(fig, "fig1_framework")


def rows(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def milestone_plot():
    """Show when each run reaches the three scientifically meaningful tolerances."""
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    runs = [
        ("Base-1", "Base", "kp_sac_weak/base_seed11001_t6"),
        ("Base-2", "Base", "kp_sac_weak/base_seed11002_t6"),
        ("Full-1", "Full", "full/full_seed11001_6m_l13_resume"),
        ("Full-2", "Full", "full/full_seed11002_6m_l13_resume"),
        ("M1", "M1", "m1_no_bottleneck/m1_seed11001_6m"),
        ("M2", "M2", "m2_no_boundary_replay/m2_seed11001_6m"),
        ("M3", "M3", "m3_no_jacobian/m3_seed11001_6m"),
    ]
    levels = [(6, "10 mm / 0.10 rad", "o"), (11, "1 mm / 0.10 rad", "^"),
              (13, "1 mm / 0.025 rad", "D")]
    yloc = np.arange(len(runs))[::-1]
    for yi, (label, method, d) in zip(yloc, runs):
        path = ROOT / "outputs/retrain_20261013" / d / "progress.csv"
        x = np.array([]); lev = np.array([])
        if path.exists():
            data = rows(path)
            x = np.array([float(r["global_step"]) for r in data]) / 1e6
            lev = np.array([float(r.get("orientation_level_index", r.get("goal_level_index", 0))) for r in data])
        ax.hlines(yi, 0, 6.0, color="#D3D7DA", linewidth=0.7, zorder=0)
        ax.text(-0.06, yi, label, ha="right", va="center", fontsize=7.0, color=INK,
                fontweight="bold" if method == "Full" else "normal")
        if x.size:
            for j, (level, _, marker) in enumerate(levels):
                hit = np.flatnonzero(lev >= level)
                if hit.size:
                    # Separate coincident milestones by a tiny display-only offset.
                    x_display = x[hit[0]] + (j - 1) * 0.100
                    ax.plot(x_display, yi, marker=marker, markersize=6.2 if marker != "D" else 6.4,
                            markerfacecolor="white" if marker != "D" else INK,
                            markeredgecolor=INK, markeredgewidth=0.9, color=INK, zorder=4)
    for val in range(1, 7):
        ax.axvline(val, color="#E3E6E8", linewidth=0.55, zorder=0)
    ax.set_xlim(0, 6.35); ax.set_ylim(-0.65, len(runs)-0.35)
    ax.set_xticks(range(7)); ax.set_xlabel("Training transitions (million)"); ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=8, length=2)
    save(fig, "fig2_milestones")


def heat_values(method, level):
    paths = sorted((ROOT / "outputs/evaluation_final").glob(f"{method}/L{level:02d}/eval_seed*.json"))
    arrays = [json.loads(pth.read_text())["bin_success_rates"] for pth in paths]
    return np.mean(np.asarray(arrays), axis=0).reshape(10, 10) if arrays else np.zeros((10, 10))


def profiles():
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.5), sharey=True)
    profiles = {
        "Base, L6": heat_values("base_s11001", 6),
        "Full, L6": heat_values("full_s11001", 6),
        "Full, L13": heat_values("full_s11001", 13),
        "M3, L11": heat_values("m3_s11001", 11),
    }
    distance_x = np.linspace(0.03, 0.50, 10)
    orientation_x = np.linspace(0.03, np.pi, 10)
    def line(ax, x, y, label, color, marker):
        ax.plot(x, y, color=color, linewidth=1.25, marker=marker, markersize=3.3,
                markerfacecolor="white", markeredgewidth=0.8, label=label)
    # Directly show marginal task-space profiles instead of a dense cell matrix.
    line(axes[0,0], distance_x, profiles["Base, L6"].mean(axis=1), "Base, 10 mm / 0.10 rad", "#555555", "o")
    line(axes[0,0], distance_x, profiles["Full, L6"].mean(axis=1), "Full, 10 mm / 0.10 rad", BLUE, "s")
    line(axes[0,1], orientation_x, profiles["Base, L6"].mean(axis=0), "Base, 10 mm / 0.10 rad", "#555555", "o")
    line(axes[0,1], orientation_x, profiles["Full, L6"].mean(axis=0), "Full, 10 mm / 0.10 rad", BLUE, "s")
    for title, color, marker in [("Full, L6", BLUE, "s"), ("Full, L13", RED, "o"), ("M3, L11", PURPLE, "^")]:
        names = {"Full, L6": "Full, 10 mm / 0.10 rad", "Full, L13": "Full, 1 mm / 0.025 rad", "M3, L11": "M3, 1 mm / 0.10 rad"}
        line(axes[1,0], distance_x, profiles[title].mean(axis=1), names[title], color, marker)
        line(axes[1,1], orientation_x, profiles[title].mean(axis=0), names[title], color, marker)
    titles = [("(a) Common level", "initial position error (m)"), ("(b) Common level", "initial orientation error (rad)"),
              ("(c) Strict levels", "initial position error (m)"), ("(d) Strict levels", "initial orientation error (rad)")]
    for ax, (title, xlabel) in zip(axes.flat, titles):
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
        ax.set_ylim(0, 1.05); ax.set_xlabel(xlabel, fontsize=7.5)
        ax.set_ylabel("Success rate", fontsize=7.5); ax.grid(axis="y", color="#E6E8EA", linewidth=0.35)
        ax.axhline(0.85, color="#777777", linestyle="--", linewidth=0.65)
        ax.tick_params(labelsize=7.5, length=2)
    for ax in axes.flat:
        ax.text(0.04, 0.89, "promotion = 0.85", transform=ax.transAxes, ha="left", va="center", fontsize=6.3, color="#666666")
        ax.legend(frameon=False, loc="lower left", fontsize=6.2, handlelength=1.4)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig3_task_space_profiles")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    framework(); milestone_plot(); profiles()
    print(f"wrote vector figures to {OUT}")
