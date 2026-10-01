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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a thesis checkpoint in PyBullet and capture report figures"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
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
    args = parser.parse_args()

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
                    / max(orientation_tolerance(config, eta), 1e-9)
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
