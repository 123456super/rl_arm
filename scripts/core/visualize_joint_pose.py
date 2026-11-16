#!/usr/bin/env python3
"""Display one UR5 joint configuration in the interactive PyBullet GUI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pybullet as p

from rl_risk_sac.envs.thesis_homotopy_env_reduced_observation import (
    ThesisHomotopyEnv,
)
from rl_risk_sac.utils.config import load_config
from visualize_thesis_ppo import (
    CameraInputState,
    draw_goal,
    keep_camera_interactive,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/paper_full.yaml"),
    )
    parser.add_argument(
        "--joint-degrees",
        type=float,
        nargs=6,
        required=True,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
    )
    parser.add_argument(
        "--random-goal",
        action="store_true",
        help="Display one collision-free random IK goal instead of the initial pose.",
    )
    parser.add_argument(
        "--goal-euler-degrees",
        type=float,
        nargs=3,
        default=(-170.0, 30.0, 95.0),
        metavar=("ROLL", "PITCH", "YAW"),
    )
    parser.add_argument("--seed", type=int, default=41001)
    parser.add_argument("--camera-distance", type=float, default=1.35)
    parser.add_argument("--camera-yaw", type=float, default=45.0)
    parser.add_argument("--camera-pitch", type=float, default=-25.0)
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=(0.0, 0.30, 0.30),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        help="Close automatically after this many seconds; default waits for X or Enter.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run geometry checks without opening the GUI.",
    )
    args = parser.parse_args()
    if args.hold_seconds is not None and (
        not np.isfinite(args.hold_seconds) or args.hold_seconds < 0.0
    ):
        parser.error("--hold-seconds must be finite and non-negative")
    return args


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    env = ThesisHomotopyEnv(
        config,
        render_mode=None if args.headless else "human",
    )
    try:
        env._initialize_world()
        initial_joint_positions = np.deg2rad(
            np.asarray(args.joint_degrees, dtype=np.float64)
        )
        joint_positions = initial_joint_positions.copy()
        goal_position = None
        goal_quaternion = None
        if args.random_goal:
            env._set_joint_state(initial_joint_positions)
            env.rng = np.random.default_rng(args.seed)
            goal_quaternion = np.asarray(
                p.getQuaternionFromEuler(
                    np.deg2rad(args.goal_euler_degrees).tolist()
                ),
                dtype=np.float32,
            )
            for _ in range(env.goal_sample_max_attempts):
                proposal = env._propose_reference_goal_position()
                if proposal is None:
                    continue
                validation = env.robot.check_pose_ik(
                    proposal,
                    goal_quaternion,
                    position_tolerance=env.position_tolerance,
                    orientation_tolerance=env.orientation_tolerance,
                    rest_pose=initial_joint_positions,
                )
                if not validation["reachable"]:
                    continue
                candidate = np.asarray(validation["q_ik"], dtype=np.float64)
                candidate = (candidate + np.pi) % (2.0 * np.pi) - np.pi
                env._set_joint_state(candidate)
                p.performCollisionDetection(physicsClientId=env.client_id)
                if (
                    not any(env._collision_events().values())
                    and env._self_geometry().distance_min >= env.d_self_safe
                ):
                    goal_position = proposal
                    joint_positions = candidate
                    break
            else:
                raise RuntimeError(
                    "failed to sample a collision-free random goal pose"
                )

        env._set_joint_state(joint_positions)
        p.performCollisionDetection(physicsClientId=env.client_id)

        tcp_position, tcp_quaternion = env.robot.end_effector_pose()
        plane_clearances = env._plane_link_clearances()[1:]
        self_clearance = env._self_geometry().distance_min
        collisions = env._collision_events()
        print(f"initial_joint_degrees={np.asarray(args.joint_degrees).tolist()}")
        print(f"display_joint_degrees={np.rad2deg(joint_positions).tolist()}")
        print(f"joint_radians={joint_positions.tolist()}")
        if goal_position is not None and goal_quaternion is not None:
            print(f"goal_position_m={goal_position.tolist()}")
            print(f"goal_euler_degrees={list(args.goal_euler_degrees)}")
        print(f"tcp_position_m={tcp_position.tolist()}")
        print(f"tcp_quaternion_xyzw={tcp_quaternion.tolist()}")
        print(f"plane_clearances_m={plane_clearances.tolist()}")
        print(f"self_clearance_m={self_clearance:.9f}")
        print(f"collisions={collisions}", flush=True)

        if args.headless:
            return
        p.configureDebugVisualizer(
            p.COV_ENABLE_GUI, 1, physicsClientId=env.client_id
        )
        p.resetDebugVisualizerCamera(
            cameraDistance=args.camera_distance,
            cameraYaw=args.camera_yaw,
            cameraPitch=args.camera_pitch,
            cameraTargetPosition=args.camera_target,
            physicsClientId=env.client_id,
        )
        if goal_position is not None and goal_quaternion is not None:
            draw_goal(env.client_id, goal_position, goal_quaternion)
        p.addUserDebugText(
            (
                "Random goal pose" if args.random_goal else "Static initial pose"
            ) + " | middle-drag: rotate | right-drag: pan | X/Enter: close",
            [0.0, 0.0, 0.85],
            textColorRGB=[0.1, 0.1, 0.1],
            textSize=1.2,
            physicsClientId=env.client_id,
        )
        keep_camera_interactive(
            env.client_id,
            CameraInputState(),
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
