#!/usr/bin/env python3
"""Replay one thesis PPO episode deterministically in the PyBullet GUI."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pybullet as p
import torch

from rl_risk_sac.algorithms.thesis_ppo import ThesisPPOAgent
from rl_risk_sac.tasks.reference_reaching import residual_action_penalty
from rl_risk_sac.utils.config import deep_update, load_config


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CameraInputState:
    """Persistent mouse state for smooth rotate and pan controls."""

    drag_mode: str | None = None
    drag_button_index: int | None = None
    last_mouse_x: float | None = None
    last_mouse_y: float | None = None
    last_poll_time: float | None = None
    close_requested: bool = False


def load_evaluation_config(
    payload: dict,
    stage: str,
    seed: int,
    device: str | None,
    override_config_path: Path | None = None,
) -> dict:
    """Rebuild the evaluator config and restore checkpoint-time overrides."""
    saved_config = deepcopy(payload["config"])
    base_path = (
        "configs/experiments/paper_s2_lstm_sweep_single.yaml"
        if stage == "s2"
        else "configs/experiments/paper_full.yaml"
    )
    config = load_config(ROOT / base_path)
    deep_update(config, saved_config)
    if override_config_path is not None:
        path = (
            override_config_path
            if override_config_path.is_absolute()
            else ROOT / override_config_path
        )
        # An explicit evaluation config intentionally takes precedence over
        # checkpoint-time environment/control values. Network dimensions must
        # remain compatible and are checked when the state dict is loaded.
        deep_update(config, load_config(path))
    # Checkpoints produced before the execution-route switch did not store an
    # explicit ``enabled`` flag: the presence of joint_motion_controller meant
    # that the CB3.1 bounded controller was unconditionally active.  Do not let
    # the restored legacy POSITION_CONTROL default leak into those replays.
    saved_motion = saved_config.get("thesis", {}).get("joint_motion_controller")
    if isinstance(saved_motion, dict) and "enabled" not in saved_motion:
        config["thesis"]["joint_motion_controller"]["enabled"] = True
    config["seed"] = seed
    if device is not None:
        config["device"] = device
    config["thesis"]["stage"] = stage
    config["thesis"]["simple_reaching_mode"] = stage == "s0"
    config["thesis"]["blocking_obstacle_mode"] = stage != "s0"
    return config


def draw_goal(client_id: int, position: np.ndarray, quaternion: np.ndarray) -> None:
    """Draw a target sphere and RGB orientation axes without collision geometry."""
    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.015,
        rgbaColor=[0.1, 1.0, 0.1, 0.65],
        physicsClientId=client_id,
    )
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual,
        basePosition=position.tolist(),
        physicsClientId=client_id,
    )
    rotation = np.asarray(
        p.getMatrixFromQuaternion(quaternion.tolist()), dtype=np.float64
    ).reshape(3, 3)
    colors = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.4, 1.0])
    for axis, color in enumerate(colors):
        endpoint = position + 0.08 * rotation[:, axis]
        p.addUserDebugLine(
            position.tolist(), endpoint.tolist(), color, lineWidth=3.0,
            physicsClientId=client_id,
        )


def vector_json(value: np.ndarray) -> str:
    return json.dumps(np.asarray(value, dtype=np.float64).tolist())


def update_camera_from_input(
    client_id: int,
    state: CameraInputState,
    *,
    initial_distance: float,
    initial_yaw: float,
    initial_pitch: float,
    initial_target: tuple[float, float, float],
) -> None:
    """Apply smooth keyboard, middle-button rotate, and right-button pan."""
    now = time.monotonic()
    elapsed = 0.0 if state.last_poll_time is None else now - state.last_poll_time
    state.last_poll_time = now
    elapsed = float(np.clip(elapsed, 0.0, 0.05))
    events = p.getKeyboardEvents(physicsClientId=client_id)
    active_mask = p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED

    def active(key: int) -> bool:
        return bool(events.get(key, 0) & active_mask)

    if active(ord("x")) or active(p.B3G_RETURN):
        state.close_requested = True

    controlled_keys = (
        p.B3G_LEFT_ARROW,
        p.B3G_RIGHT_ARROW,
        p.B3G_UP_ARROW,
        p.B3G_DOWN_ARROW,
        ord("w"), ord("s"), ord("a"), ord("d"), ord("q"), ord("e"),
        ord("r"),
    )
    rotate_delta_x = 0.0
    rotate_delta_y = 0.0
    pan_delta_x = 0.0
    pan_delta_y = 0.0
    # PyBullet mouse event layout:
    # (event_type, x, y, button_index, button_state). Bullet normalizes the
    # buttons to 0=left, 1=middle/wheel, 2=right.
    for event in p.getMouseEvents(physicsClientId=client_id):
        event_type, mouse_x, mouse_y, button_index, button_state = event
        if event_type == 2 and button_index in (1, 2):
            if (
                button_state & p.KEY_WAS_TRIGGERED
                and state.drag_button_index is None
            ):
                state.drag_button_index = int(button_index)
                state.drag_mode = "rotate" if button_index == 1 else "pan"
                state.last_mouse_x = float(mouse_x)
                state.last_mouse_y = float(mouse_y)
            if (
                button_state & p.KEY_WAS_RELEASED
                and button_index == state.drag_button_index
            ):
                state.drag_mode = None
                state.drag_button_index = None
                state.last_mouse_x = None
                state.last_mouse_y = None
        elif event_type == 1 and state.drag_mode is not None:
            if state.last_mouse_x is not None and state.last_mouse_y is not None:
                delta_x = float(mouse_x) - state.last_mouse_x
                delta_y = float(mouse_y) - state.last_mouse_y
                if state.drag_mode == "rotate":
                    rotate_delta_x += delta_x
                    rotate_delta_y += delta_y
                else:
                    pan_delta_x += delta_x
                    pan_delta_y += delta_y
            state.last_mouse_x = float(mouse_x)
            state.last_mouse_y = float(mouse_y)

    keyboard_active = any(active(key) for key in controlled_keys)
    mouse_active = any(
        delta != 0.0
        for delta in (
            rotate_delta_x, rotate_delta_y, pan_delta_x, pan_delta_y,
        )
    )
    if not keyboard_active and not mouse_active:
        return
    if active(ord("r")):
        distance = initial_distance
        yaw = initial_yaw
        pitch = initial_pitch
        target = np.asarray(initial_target, dtype=np.float64)
    else:
        camera = p.getDebugVisualizerCamera(physicsClientId=client_id)
        yaw = float(camera[8])
        pitch = float(camera[9])
        distance = float(camera[10])
        target = np.asarray(camera[11], dtype=np.float64)
        yaw += 90.0 * elapsed * (
            float(active(p.B3G_RIGHT_ARROW))
            - float(active(p.B3G_LEFT_ARROW))
        )
        pitch += 90.0 * elapsed * (
            float(active(p.B3G_UP_ARROW)) - float(active(p.B3G_DOWN_ARROW))
        )
        yaw += 0.25 * rotate_delta_x
        pitch -= 0.25 * rotate_delta_y
        target[2] += 0.30 * elapsed * (
            float(active(ord("w"))) - float(active(ord("s")))
        )
        target[0] += 0.30 * elapsed * (
            float(active(ord("d"))) - float(active(ord("a")))
        )
        distance += 1.0 * elapsed * (
            float(active(ord("q"))) - float(active(ord("e")))
        )
        if pan_delta_x != 0.0 or pan_delta_y != 0.0:
            horizontal = np.asarray(camera[6], dtype=np.float64)
            vertical = np.asarray(camera[7], dtype=np.float64)
            horizontal /= max(float(np.linalg.norm(horizontal)), 1e-12)
            vertical /= max(float(np.linalg.norm(vertical)), 1e-12)
            pan_scale = max(distance, 0.25) * 0.0015
            target -= pan_scale * pan_delta_x * horizontal
            target += pan_scale * pan_delta_y * vertical
        pitch = float(np.clip(pitch, -89.0, 89.0))
        distance = float(np.clip(distance, 0.25, 5.0))
    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target.tolist(),
        physicsClientId=client_id,
    )


def keep_camera_interactive(
    client_id: int,
    state: CameraInputState,
    *,
    initial_distance: float,
    initial_yaw: float,
    initial_pitch: float,
    initial_target: tuple[float, float, float],
    hold_seconds: float | None,
) -> None:
    """Keep polling GUI input after physics playback has stopped."""
    deadline = (
        None if hold_seconds is None else time.monotonic() + hold_seconds
    )
    while not state.close_requested:
        if deadline is not None and time.monotonic() >= deadline:
            break
        if not p.isConnected(client_id):
            break
        try:
            update_camera_from_input(
                client_id,
                state,
                initial_distance=initial_distance,
                initial_yaw=initial_yaw,
                initial_pitch=initial_pitch,
                initial_target=initial_target,
            )
        except p.error:
            break
        time.sleep(1.0 / 240.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        help=(
            "Explicit evaluation config applied after checkpoint settings; "
            "use this for controlled execution-contract ablations."
        ),
    )
    parser.add_argument(
        "--scene", choices=("none", "static", "dynamic", "hybrid"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=41001)
    parser.add_argument("--stage", choices=("s0", "s1", "s2"))
    parser.add_argument("--device")
    parser.add_argument(
        "--static-obstacle-count", type=int, choices=range(1, 9),
        help="Override the checkpoint's static-scene obstacle-count mixture.",
    )
    parser.add_argument(
        "--static-obstacle-layout",
        choices=("legacy_blocking", "path_offset", "mixed", "structured"),
        help="Override the checkpoint's static obstacle layout distribution.",
    )
    parser.add_argument(
        "--playback-speed", type=float, default=1.0,
        help="Wall-clock playback multiplier; 1.0 shows the 60 s episode in real time.",
    )
    parser.add_argument(
        "--residual-cutoff-position-m", type=float, default=0.025,
        help=(
            "Deployment Base handoff at or below this position error "
            "(default: 0.025 m)."
        ),
    )
    parser.add_argument(
        "--no-residual-cutoff",
        action="store_const",
        const=None,
        dest="residual_cutoff_position_m",
        help="Disable the default 25 mm residual cutoff for an ablation run.",
    )
    parser.add_argument(
        "--trace-output", type=Path,
        help="Optional CSV path for the per-policy-step trace.",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument(
        "--wait-on-exit", action="store_true",
        help="Keep the GUI open until Enter is pressed after the episode.",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run the same single-environment path without a GUI (for testing).",
    )
    parser.add_argument("--camera-distance", type=float, default=1.45)
    parser.add_argument("--camera-yaw", type=float, default=45.0)
    parser.add_argument("--camera-pitch", type=float, default=-30.0)
    parser.add_argument(
        "--camera-target", type=float, nargs=3,
        default=(0.0, 0.35, 0.35), metavar=("X", "Y", "Z"),
    )
    args = parser.parse_args()
    if not np.isfinite(args.playback_speed) or args.playback_speed <= 0.0:
        parser.error("--playback-speed must be finite and positive")
    if args.residual_cutoff_position_m is not None and (
        not np.isfinite(args.residual_cutoff_position_m)
        or args.residual_cutoff_position_m <= 0.0
    ):
        parser.error("--residual-cutoff-position-m must be finite and positive")
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    if not np.isfinite(args.hold_seconds) or args.hold_seconds < 0.0:
        parser.error("--hold-seconds must be finite and non-negative")
    return args


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "config" not in payload or "agent" not in payload:
        raise ValueError("checkpoint must contain config and agent state")
    saved_config = payload["config"]
    checkpoint_stage = str(
        payload.get("stage", saved_config["thesis"].get("stage", ""))
    ).lower()
    stage = args.stage or checkpoint_stage
    if stage not in {"s0", "s1", "s2"}:
        raise ValueError(f"invalid checkpoint stage {stage!r}")
    if stage == "s0" and args.scene != "none":
        raise ValueError("S0 replay supports only --scene none")
    if args.scene in {"dynamic", "hybrid"} and stage != "s2":
        raise ValueError("dynamic and hybrid replay require an S2 checkpoint")

    config = load_evaluation_config(
        payload, stage, args.seed, args.device, args.config
    )
    if args.static_obstacle_count is not None:
        if args.scene != "static":
            raise ValueError(
                "--static-obstacle-count currently applies only to --scene static"
            )
        config["thesis"]["static_obstacle_count_probabilities"] = {
            args.static_obstacle_count: 1.0,
        }
    if args.static_obstacle_layout is not None:
        if args.scene not in {"static", "hybrid"}:
            raise ValueError(
                "--static-obstacle-layout requires --scene static or hybrid"
            )
        config["thesis"]["blocking_obstacle_mode"] = True
        config["thesis"]["static_obstacle_layout_probabilities"] = {
            args.static_obstacle_layout: 1.0,
        }
    if args.scene in {"dynamic", "hybrid"}:
        prediction = config["thesis"]["dynamic_prediction"]
        if not prediction["enabled"] or not prediction["checkpoint"]:
            raise ValueError("dynamic replay requires an enabled LSTM checkpoint")

    from rl_risk_sac.envs.thesis_homotopy_env_reduced_observation import (
        ThesisHomotopyEnv,
    )

    render_mode = None if args.headless else "human"
    env = ThesisHomotopyEnv(config, render_mode=render_mode)
    camera_input = CameraInputState()
    trace: list[dict[str, object]] = []
    try:
        if not args.headless:
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI, 1, physicsClientId=env.client_id
            )
            p.resetDebugVisualizerCamera(
                cameraDistance=args.camera_distance,
                cameraYaw=args.camera_yaw,
                cameraPitch=args.camera_pitch,
                cameraTargetPosition=list(args.camera_target),
                physicsClientId=env.client_id,
            )
            env.configure_human_playback(
                args.playback_speed,
                input_callback=lambda: update_camera_from_input(
                    env.client_id,
                    camera_input,
                    initial_distance=args.camera_distance,
                    initial_yaw=args.camera_yaw,
                    initial_pitch=args.camera_pitch,
                    initial_target=tuple(args.camera_target),
                ),
            )
        env.configure_episode(args.scene, terminate_on_success=False)
        observation, reset_info = env.reset(seed=args.seed)
        if not args.headless:
            draw_goal(
                env.client_id,
                np.asarray(reset_info["goal_position"], dtype=np.float64),
                np.asarray(reset_info["goal_quaternion"], dtype=np.float64),
            )

        obs_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.shape[0])
        agent = ThesisPPOAgent(obs_dim, action_dim, config)
        agent.load_state_dict(payload["agent"])
        reward_config = config["thesis"]["reward"]
        current_position_error = float(reset_info["goal_error_norm"])
        previous_ee_position = np.asarray(
            reset_info["ee_position"], dtype=np.float64
        )
        episode_return = 0.0
        collision_seen = {
            "collision": False,
            "obstacle_collision": False,
            "self_collision": False,
            "environment_collision": False,
        }

        print(
            f"checkpoint={checkpoint}\n"
            f"stage={stage} scene={args.scene} seed={args.seed} "
            f"playback_speed={args.playback_speed} "
            f"residual_cutoff={args.residual_cutoff_position_m}\n"
            f"obstacles={reset_info.get('obstacle_count', 0)} "
            f"layout={reset_info.get('obstacle_layout', 'none')} "
            f"initial_position={1000.0 * current_position_error:.2f}mm "
            f"initial_orientation={float(reset_info['orientation_error_norm']):.5f}rad\n"
            "camera: middle-drag=rotate, right-drag=pan, arrows=rotate, "
            "W/S=target up/down, A/D=target left/right, Q/E=zoom, R=reset; "
            "after playback press X or Enter in the GUI to close",
            flush=True,
        )

        terminated = truncated = False
        while not (terminated or truncated):
            actions, _logp, _values, policy = agent.act_for_deployment(
                observation[None, :],
                np.asarray([current_position_error], dtype=np.float32),
                residual_cutoff_position_m=args.residual_cutoff_position_m,
            )
            executed_action = actions[0].copy()
            total_action = policy["policy_actions"][0].copy()
            residual_action = policy["residual_actions"][0].copy()
            base_action = policy["base_actions"][0].copy()
            residual_authority = float(policy["residual_authority"][0])
            residual_disabled = residual_authority == 0.0
            (
                next_observation,
                environment_reward,
                _cost,
                terminated,
                truncated,
                info,
            ) = env.step(executed_action)
            residual_penalty, _ = residual_action_penalty(
                residual_action,
                float(info["goal_error_norm"]),
                float(info["d_min"]),
                penalty_weight=float(
                    reward_config["reference_residual_penalty_weight"]
                ),
                goal_start_m=float(
                    reward_config["reference_residual_penalty_goal_start_m"]
                ),
                goal_full_m=float(
                    reward_config["reference_residual_penalty_goal_full_m"]
                ),
                clearance_start_m=float(
                    reward_config["reference_residual_penalty_clearance_start_m"]
                ),
                clearance_full_m=float(
                    reward_config["reference_residual_penalty_clearance_full_m"]
                ),
            )
            reward = float(environment_reward) - residual_penalty
            episode_return += reward
            current_position_error = float(info["goal_error_norm"])
            ee_position = np.asarray(info["ee_position"], dtype=np.float64)
            if not args.headless:
                p.addUserDebugLine(
                    previous_ee_position.tolist(),
                    ee_position.tolist(),
                    [0.0, 0.8, 1.0],
                    lineWidth=2.0,
                    physicsClientId=env.client_id,
                )
            previous_ee_position = ee_position
            for key in collision_seen:
                collision_seen[key] = collision_seen[key] or bool(info[key])

            step = int(info.get("step", len(trace) + 1))
            trace.append({
                "step": step,
                "reward": reward,
                "environment_reward": float(environment_reward),
                "residual_penalty": float(residual_penalty),
                "return": episode_return,
                "position_error_m": current_position_error,
                "position_error_vector": vector_json(
                    np.asarray(info["position_error_vector"], dtype=np.float64)
                ),
                "orientation_error_rad": float(info["orientation_error_norm"]),
                "reference_error": float(info["reference_error"]),
                "r_goal": float(info["r_goal"]),
                "c_proximity": float(info["c_proximity"]),
                "obstacle_penalty": float(info["reference_obstacle_penalty"]),
                "d_min_m": float(info["d_min"]),
                "collision": int(bool(info["collision"])),
                "obstacle_collision": int(bool(info["obstacle_collision"])),
                "self_collision": int(bool(info["self_collision"])),
                "environment_collision": int(bool(info["environment_collision"])),
                "residual_disabled": int(residual_disabled),
                "residual_authority": residual_authority,
                "base_action": vector_json(base_action),
                "residual_action": vector_json(residual_action),
                "policy_total_action": vector_json(total_action),
                "executed_action": vector_json(executed_action),
                "joint_speed_peak_rad_s": float(info["joint_speed_peak_rad_s"]),
                "joint_acceleration_peak_rad_s2": float(
                    info["joint_acceleration_peak_rad_s2"]
                ),
                "joint_velocity_limiter_fraction": float(
                    info["joint_velocity_limiter_fraction"]
                ),
                "joint_acceleration_limiter_fraction": float(
                    info["joint_acceleration_limiter_fraction"]
                ),
            })
            if step == 1 or step % args.log_every == 0 or terminated or truncated:
                flags = "".join((
                    "O" if info["obstacle_collision"] else "-",
                    "S" if info["self_collision"] else "-",
                    "P" if info["environment_collision"] else "-",
                ))
                print(
                    f"step={step:03d}/{env.horizon} reward={reward:8.3f} "
                    f"return={episode_return:9.2f} "
                    f"pos={1000.0 * current_position_error:7.2f}mm "
                    f"ori={float(info['orientation_error_norm']):.5f}rad "
                    f"dmin={1000.0 * float(info['d_min']):7.2f}mm "
                    f"action=(base={np.mean(np.abs(base_action)):.4f},"
                    f"res={np.mean(np.abs(residual_action)):.4f},"
                    f"exec={np.mean(np.abs(executed_action)):.4f}) "
                    f"residual_authority={residual_authority:.1f} "
                    f"handoff={int(residual_disabled)} collision={flags}",
                    flush=True,
                )
            observation = next_observation

        summary = {
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(
                payload.get("global_step", payload.get("step", 0))
            ),
            "stage": stage,
            "scene": args.scene,
            "seed": args.seed,
            "length": len(trace),
            "return": episode_return,
            "final_position_m": current_position_error,
            "final_orientation_rad": float(info["orientation_error_norm"]),
            "best_position_m": min(row["position_error_m"] for row in trace),
            "collision": collision_seen,
            "residual_cutoff_position_m": args.residual_cutoff_position_m,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

        if args.trace_output is not None:
            output = args.trace_output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() or output.with_suffix(".json").exists():
                raise FileExistsError(output)
            with output.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
                writer.writeheader()
                writer.writerows(trace)
            output.with_suffix(".json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"saved trace: {output}", flush=True)
            print(f"saved summary: {output.with_suffix('.json')}", flush=True)

        if not args.headless:
            if args.wait_on_exit:
                print(
                    "回放完成：视角控制仍然有效；在 PyBullet 窗口按 X 或 "
                    "Enter 关闭。",
                    flush=True,
                )
                keep_camera_interactive(
                    env.client_id,
                    camera_input,
                    initial_distance=args.camera_distance,
                    initial_yaw=args.camera_yaw,
                    initial_pitch=args.camera_pitch,
                    initial_target=tuple(args.camera_target),
                    hold_seconds=None,
                )
            elif args.hold_seconds > 0.0:
                keep_camera_interactive(
                    env.client_id,
                    camera_input,
                    initial_distance=args.camera_distance,
                    initial_yaw=args.camera_yaw,
                    initial_pitch=args.camera_pitch,
                    initial_target=tuple(args.camera_target),
                    hold_seconds=args.hold_seconds,
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
