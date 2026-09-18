from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

from rl_risk_sac.robots.pybullet_robot import PyBulletRobot
from rl_risk_sac.tasks.thesis_reaching import (
    THESIS_OBSERVATION_DIM,
    SelfCollisionGeometry,
    ThesisGeometry,
    build_thesis_observation,
    compute_self_collision_geometry,
    compute_thesis_geometry,
    homotopy_reward,
    pose_error,
)
from rl_risk_sac.utils.risk import closest_point_on_segment
from rl_risk_sac.utils.runtime_config import RuntimeConfig


@dataclass
class EpisodeContract:
    scene: str = "none"
    xi: float = 1.0
    strict_obstacle_collision: bool = True
    goal_scale: float = 1.0
    orientation_scale: float = 1.0
    orientation_tolerance: float = 0.10
    lambda_self: float = 1.0


class ThesisHomotopyEnv(gym.Env):
    """PyBullet environment implementing Chapters 1--3 of the thesis protocol.

    The legacy environment remains available for historical checkpoints.  This
    class uses continuous 6D rotation observations, explicit task-space twist,
    constant joint-velocity commands, and the task-first collision homotopy.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: dict[str, Any], render_mode: str | None = None) -> None:
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        self.render_mode = render_mode
        self.config = config
        self.runtime = RuntimeConfig.from_mapping(config)
        thesis = config["thesis"]
        self.physics_dt = float(thesis.get("physics_dt", 1.0 / 240.0))
        self.control_dt = float(thesis.get("control_dt", 0.05))
        self.substeps = int(round(self.control_dt / self.physics_dt))
        if self.substeps != 12 or not np.isclose(self.substeps * self.physics_dt, self.control_dt):
            raise ValueError("thesis protocol requires 12 substeps at 1/240 s for a 0.05 s control period")
        self.horizon = int(thesis.get("horizon", 240))
        self.action_scale = np.full(6, float(thesis.get("action_scale", 0.7)), dtype=np.float32)
        self.velocity_limits = np.full(6, np.pi, dtype=np.float32)
        self.ee_linear_velocity_scale = float(thesis.get("ee_linear_velocity_scale", 1.0))
        self.ee_angular_velocity_scale = float(thesis.get("ee_angular_velocity_scale", 4.2))
        if self.ee_linear_velocity_scale <= 0.0 or self.ee_angular_velocity_scale <= 0.0:
            raise ValueError("end-effector velocity scales must be positive")
        self.motor_force = float(thesis.get("joint_motor_force", 90.0))
        self.obstacle_radius = float(thesis.get("obstacle_radius", 0.075))
        self.obstacle_speed = float(thesis.get("obstacle_speed", 0.1))
        self.d_safe = float(thesis.get("d_safe", 0.12))
        self_collision_config = thesis.get("self_collision", {})
        self.d_self_safe = float(self_collision_config.get("safe_distance_m", 0.005))
        self.self_query_distance = float(self_collision_config.get("query_distance_m", 0.25))
        self.self_ttc_max = float(self_collision_config.get("ttc_max_s", 3.0))
        self.self_approach_velocity_scale = float(
            self_collision_config.get("approach_velocity_scale", 0.7)
        )
        self.self_ttc_tau = float(self_collision_config.get("ttc_tau_s", 1.0))
        if not 0.0 < self.d_self_safe < self.self_query_distance:
            raise ValueError("self collision distances must satisfy 0 < safe < query")
        self.position_tolerance = float(thesis.get("position_tolerance", 0.055))
        self.orientation_tolerance = float(thesis.get("orientation_tolerance", 0.10))
        reward_config = thesis.get("reward", {})
        self.reward_parameters = {
            "d_safe": self.d_safe,
            "d_self_safe": self.d_self_safe,
            "position_sigma": float(reward_config.get("position_sigma_m", 0.20)),
            "orientation_sigma": float(reward_config.get("orientation_sigma_rad", 1.0)),
            "micro_power": float(reward_config.get("micro_power", 4.0)),
            "orientation_priority_weight": float(
                reward_config.get("orientation_priority_weight", 1.0)
            ),
            "potential_scale": float(reward_config.get("potential_scale", 20.0)),
            "success_bonus": float(reward_config.get("success_bonus", 20.0)),
            "velocity_cost_weight": float(reward_config.get("velocity_cost_weight", 0.04)),
            "smooth_cost_weight": float(reward_config.get("smooth_cost_weight", 0.01)),
            "hard_failure_penalty": float(
                reward_config.get("hard_failure_penalty", 10.0)
            ),
            "safety_risk_weight": float(reward_config.get("safety_risk_weight", 2.0)),
            "safety_clearance_weight": float(
                reward_config.get("safety_clearance_weight", 8.0)
            ),
            "self_risk_weight": float(reward_config.get("self_risk_weight", 2.0)),
            "self_clearance_weight": float(
                reward_config.get("self_clearance_weight", 8.0)
            ),
            "external_safety_scale": float(
                reward_config.get("external_safety_scale", 1.0)
            ),
            "self_safety_scale": float(reward_config.get("self_safety_scale", 1.0)),
        }
        self.goal_sample_max_attempts = int(thesis.get("goal_sample_max_attempts", 10000))
        if self.goal_sample_max_attempts < 1:
            raise ValueError("goal_sample_max_attempts must be positive")
        self.contract = EpisodeContract()
        self.rng = np.random.default_rng(int(config["seed"]))

        self.client_id = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError(
                "failed to connect to PyBullet; human rendering requires an available display"
            )
        self.robot = PyBulletRobot(
            self.runtime.robot,
            Path(config["robot"]["urdf"]),
            self.client_id,
        )
        self.robot_id = -1
        self.obstacle_id: int | None = None
        self.obstacle_position = np.zeros(3, dtype=np.float32)
        self.obstacle_velocity = np.zeros(3, dtype=np.float32)
        self.goal_position = np.zeros(3, dtype=np.float32)
        self.goal_quaternion = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.goal_joint_positions = np.zeros(6, dtype=np.float32)
        self.goal_ik_validation: dict[str, Any] = {}
        self.goal_sample_attempts_last = 0
        self.step_count = 0
        self.obstacle_contact_seen = False
        self.previous_rho_position = 0.0
        self.previous_rho_orientation = 0.0
        self._link_name_to_id: dict[str, int] = {}
        self._collision_link_ids: set[int] = set()
        self._self_link_order: list[int] = []
        self._self_pair_link_ids: list[tuple[int, int]] = []
        self._self_pair_slots: list[tuple[int, int]] = []

        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(THESIS_OBSERVATION_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

    def configure_episode(
        self, scene: str, *, xi: float, strict: bool, goal_scale: float = 1.0,
        orientation_scale: float = 1.0, orientation_tolerance: float | None = None,
        lambda_self: float = 1.0,
    ) -> None:
        if scene not in {"none", "static", "dynamic"}:
            raise ValueError(f"unknown thesis scene {scene!r}")
        if not 0.02 <= float(xi) <= 1.0:
            raise ValueError("xi must be in [0.02, 1]")
        if not 0.0 < float(goal_scale) <= 1.0:
            raise ValueError("goal_scale must be in (0, 1]")
        if not 0.0 <= float(orientation_scale) <= 1.0:
            raise ValueError("orientation_scale must be in [0, 1]")
        if not 0.0 <= float(lambda_self) <= 1.0:
            raise ValueError("lambda_self must be in [0, 1]")
        effective_orientation_tolerance = (
            self.orientation_tolerance if orientation_tolerance is None else float(orientation_tolerance)
        )
        if not 0.0 < effective_orientation_tolerance <= np.pi:
            raise ValueError("orientation_tolerance must be in (0, pi]")
        self.contract = EpisodeContract(
            scene=scene,
            xi=float(xi),
            strict_obstacle_collision=bool(strict),
            goal_scale=float(goal_scale),
            orientation_scale=float(orientation_scale),
            orientation_tolerance=effective_orientation_tolerance,
            lambda_self=float(lambda_self),
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        p.resetSimulation(physicsClientId=self.client_id)
        p.setTimeStep(self.physics_dt, physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, 0.0, physicsClientId=self.client_id)
        self.robot_id = self.robot.load(self.rng, enable_self_collision=True)
        self._resolve_collision_links()
        self._sample_valid_reset()
        self._sample_goal_pose()
        self._create_obstacle()
        self.step_count = 0
        self.obstacle_contact_seen = False
        observation, info, geometry, self_geometry, position_error, orientation_error = self._observe()
        self.previous_rho_position = float(np.linalg.norm(position_error))
        self.previous_rho_orientation = float(np.linalg.norm(orientation_error))
        info.update(self._contract_info())
        info.update(self._geometry_info(geometry))
        info.update(self._self_geometry_info(self_geometry))
        return observation, info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        command = np.clip(action * self.action_scale, -self.velocity_limits, self.velocity_limits)
        q_before, qdot_before = self.robot.joint_state()
        contact_seen_before = self.obstacle_contact_seen
        obstacle_collision = False
        self_collision = False
        environment_collision = False
        joint_limit = False
        max_risk = 0.0
        min_distance = float("inf")
        self_max_risk = 0.0
        self_min_distance = self.self_query_distance
        self_min_ttc = self.self_ttc_max
        self_max_approach = 0.0
        executed_substeps = 0

        for _ in range(self.substeps):
            self._advance_obstacle()
            p.setJointMotorControlArray(
                self.robot_id,
                self.robot.joint_ids,
                p.VELOCITY_CONTROL,
                targetVelocities=command.tolist(),
                forces=[self.motor_force] * 6,
                physicsClientId=self.client_id,
            )
            p.stepSimulation(physicsClientId=self.client_id)
            executed_substeps += 1
            geometry = self._geometry()
            if geometry is not None:
                max_risk = max(max_risk, geometry.risk_max)
                min_distance = min(min_distance, geometry.distance_min)
            self_geometry = self._self_geometry()
            self_max_risk = max(self_max_risk, self_geometry.risk_max)
            self_min_distance = min(self_min_distance, self_geometry.distance_min)
            self_min_ttc = min(self_min_ttc, self_geometry.ttc_min)
            self_max_approach = max(self_max_approach, self_geometry.approach_max)
            events = self._collision_events()
            obstacle_collision = obstacle_collision or events["obstacle_collision"]
            self_collision = self_collision or events["self_collision"]
            environment_collision = environment_collision or events["environment_collision"]
            q_now, _ = self.robot.joint_state()
            lower, upper = self.robot.joint_lower_limits, self.robot.joint_upper_limits
            joint_limit = joint_limit or bool(np.any(q_now < lower - 1e-6) or np.any(q_now > upper + 1e-6))
            hard_failure = self_collision or environment_collision or joint_limit
            strict_obstacle_failure = obstacle_collision and self.contract.strict_obstacle_collision
            if hard_failure or strict_obstacle_failure:
                break

        self.obstacle_contact_seen = self.obstacle_contact_seen or obstacle_collision
        self.step_count += 1
        observation, info, endpoint_geometry, endpoint_self_geometry, position_error, orientation_error = self._observe()
        next_rho_position = float(np.linalg.norm(position_error))
        next_rho_orientation = float(np.linalg.norm(orientation_error))
        hard_failure = bool(self_collision or environment_collision or joint_limit)
        obstacle_failure = bool(obstacle_collision and self.contract.strict_obstacle_collision and not hard_failure)
        position_reached = bool(
            not hard_failure
            and not obstacle_failure
            and next_rho_position <= self.position_tolerance
        )
        task_reached = bool(
            position_reached
            and next_rho_orientation <= self.contract.orientation_tolerance
        )
        terminated = bool(hard_failure or obstacle_failure or task_reached)
        truncated = bool(self.step_count >= self.horizon and not terminated)
        _, qdot_after = self.robot.joint_state()
        # Average across joints before the bounded reward mapping. Summing six
        # normalized squared changes made the smoothness term saturate on
        # nearly every transition.
        smooth_velocity = float(
            np.mean(np.square((qdot_after - qdot_before) / self.action_scale))
        )
        velocity_magnitude = float(np.mean(np.square(qdot_after / self.action_scale)))
        if endpoint_geometry is None:
            min_distance = 0.80
            max_risk = 0.0
        else:
            min_distance = min(min_distance, endpoint_geometry.distance_min)
            max_risk = max(max_risk, endpoint_geometry.risk_max)
        self_min_distance = min(self_min_distance, endpoint_self_geometry.distance_min)
        self_max_risk = max(self_max_risk, endpoint_self_geometry.risk_max)
        self_min_ttc = min(self_min_ttc, endpoint_self_geometry.ttc_min)
        self_max_approach = max(self_max_approach, endpoint_self_geometry.approach_max)
        reward, reward_fields = homotopy_reward(
            rho_position=self.previous_rho_position,
            next_rho_position=next_rho_position,
            rho_orientation=self.previous_rho_orientation,
            next_rho_orientation=next_rho_orientation,
            smooth_velocity=smooth_velocity,
            velocity_magnitude=velocity_magnitude,
            orientation_scale=self.contract.orientation_scale,
            task_reached=task_reached,
            hard_failure=hard_failure,
            obstacle_collision=obstacle_collision,
            terminal_obstacle_collision=obstacle_failure,
            risk_max=max_risk,
            distance_min=min_distance,
            xi=self.contract.xi if self.contract.scene != "none" else 1.0,
            self_risk_max=self_max_risk,
            self_distance_min=self_min_distance,
            lambda_self=self.contract.lambda_self,
            gamma=float(self.config["sac"]["gamma"]),
            horizon=self.horizon,
            **self.reward_parameters,
        )
        collision_assisted_reach = bool(task_reached and self.obstacle_contact_seen)
        safe_success = bool(task_reached and not self.obstacle_contact_seen and not hard_failure)
        cost = float(
            max_risk
            + self_max_risk
            + 3.0 * (min_distance < self.d_safe)
            + 3.0 * (self_min_distance < self.d_self_safe)
            + 8.0 * (obstacle_collision or hard_failure)
        )
        info.update(
            {
                **self._contract_info(),
                **self._geometry_info(endpoint_geometry),
                **self._self_geometry_info(endpoint_self_geometry),
                **reward_fields,
                "reward": reward,
                "hard_penalty": reward_fields["hard_penalty"],
                "safety_penalty": reward_fields["safety_penalty"],
                "terminal_guard_penalty": reward_fields["terminal_guard_penalty"],
                "cost": cost,
                "task_reached": task_reached,
                "position_reached": position_reached,
                "collision_assisted_reach": collision_assisted_reach,
                "safe_success": safe_success,
                "success": safe_success,
                "collision": bool(obstacle_collision or self_collision or environment_collision),
                "obstacle_collision": obstacle_collision,
                "self_collision": self_collision,
                "environment_collision": environment_collision,
                "joint_limit": joint_limit,
                "hard_failure": hard_failure,
                "obstacle_contact_seen_before": contact_seen_before,
                "obstacle_contact_seen": self.obstacle_contact_seen,
                "control_max_risk": max_risk,
                "control_min_distance": min_distance,
                "control_safety_violation": bool(min_distance < self.d_safe),
                "control_self_max_risk": self_max_risk,
                "control_self_min_distance": self_min_distance,
                "control_self_min_ttc": self_min_ttc,
                "control_self_max_approach": self_max_approach,
                "control_self_safety_violation": bool(
                    self_min_distance < self.d_self_safe
                ),
                "executed_substeps": executed_substeps,
                "qdot_cmd": command.copy(),
                "qdot_measured": qdot_after.copy(),
                "smooth_velocity": smooth_velocity,
                "velocity_magnitude": velocity_magnitude,
                "rho_position": self.previous_rho_position,
                "next_rho_position": next_rho_position,
                "rho_orientation": self.previous_rho_orientation,
                "next_rho_orientation": next_rho_orientation,
            }
        )
        self.previous_rho_position = next_rho_position
        self.previous_rho_orientation = next_rho_orientation
        return observation, reward, cost, terminated, truncated, info

    def _sample_valid_reset(self) -> None:
        for _ in range(100):
            self.robot.reset_joints(self.rng)
            p.performCollisionDetection(physicsClientId=self.client_id)
            events = self._collision_events()
            self_geometry = self._self_geometry()
            q, _ = self.robot.joint_state()
            if (
                not events["self_collision"]
                and self_geometry.distance_min >= self.d_self_safe
                and np.all(q >= self.robot.joint_lower_limits)
                and np.all(q <= self.robot.joint_upper_limits)
            ):
                return
        raise RuntimeError("failed to sample a valid initial joint configuration in 100 attempts")

    def _sample_goal_pose(self) -> None:
        initial_q, initial_qdot = self.robot.joint_state()
        lower = self.robot.joint_lower_limits + 0.10
        upper = self.robot.joint_upper_limits - 0.10
        for attempt in range(1, self.goal_sample_max_attempts + 1):
            full_candidate = self.rng.uniform(lower, upper).astype(np.float32)
            candidate = initial_q + self.contract.goal_scale * (full_candidate - initial_q)
            self._set_joint_state(candidate)
            p.performCollisionDetection(physicsClientId=self.client_id)
            position, quaternion = self.robot.end_effector_pose()
            within_workspace = bool(
                0.25 <= position[0] <= 0.78
                and -0.45 <= position[1] <= 0.45
                and 0.18 <= position[2] <= 0.78
            )
            target_self_collision = self._collision_events()["self_collision"]
            target_self_clearance = self._self_geometry().distance_min
            ik_validation = self.robot.check_pose_ik(
                position,
                quaternion,
                position_tolerance=0.01,
                orientation_tolerance=0.05,
                rest_pose=candidate,
            )
            self._set_joint_state(initial_q, initial_qdot)
            current_position, current_quaternion = self.robot.end_effector_pose()
            initial_position_error, initial_orientation_error = pose_error(position, quaternion, current_position, current_quaternion)
            already_reached = (
                np.linalg.norm(initial_position_error) <= self.position_tolerance
                and np.linalg.norm(initial_orientation_error) <= self.contract.orientation_tolerance
            )
            if (
                within_workspace
                and not target_self_collision
                and target_self_clearance >= self.d_self_safe
                and bool(ik_validation["reachable"])
                and not already_reached
                and np.linalg.norm(initial_orientation_error) < np.pi - 1e-3
            ):
                self.goal_position = position.copy()
                self.goal_quaternion = quaternion.copy()
                self.goal_joint_positions = candidate.copy()
                self.goal_ik_validation = {
                    **ik_validation,
                    "q_ik": np.asarray(ik_validation["q_ik"], dtype=np.float32),
                }
                self.goal_sample_attempts_last = attempt
                return
        self._set_joint_state(initial_q, initial_qdot)
        raise RuntimeError(
            "failed to sample a valid pose goal in "
            f"{self.goal_sample_max_attempts} attempts"
        )

    def _create_obstacle(self) -> None:
        self.obstacle_id = None
        if self.contract.scene == "none":
            self.obstacle_position.fill(0.0)
            self.obstacle_velocity.fill(0.0)
            return
        for _ in range(100):
            if self.contract.scene == "static":
                position = np.asarray(
                    [
                        self.rng.uniform(0.22, 0.72),
                        self.rng.uniform(-0.48, 0.48),
                        self.rng.uniform(0.18, 0.62),
                    ],
                    dtype=np.float32,
                )
                velocity = np.zeros(3, dtype=np.float32)
                if np.linalg.norm(position - self.goal_position) < 0.15:
                    continue
            else:
                side = -1.0 if self.rng.random() < 0.5 else 1.0
                position = np.asarray(
                    [self.rng.uniform(0.22, 0.72), side * self.rng.uniform(0.42, 0.62), self.rng.uniform(0.18, 0.62)],
                    dtype=np.float32,
                )
                waypoint = np.asarray(
                    [self.rng.uniform(0.22, 0.72), -side * self.rng.uniform(0.24, 0.48), self.rng.uniform(0.18, 0.62)],
                    dtype=np.float32,
                )
                direction = waypoint - position
                velocity = (self.obstacle_speed * direction / np.linalg.norm(direction)).astype(np.float32)
            geometry = compute_thesis_geometry(
                self.robot.capsules(), position, velocity, self._closest_point_velocities(position),
                obstacle_radius=self.obstacle_radius, d_safe=self.d_safe,
            )
            if geometry.distance_min >= self.d_safe:
                self.obstacle_position = position
                self.obstacle_velocity = velocity
                self._create_obstacle_body()
                p.performCollisionDetection(physicsClientId=self.client_id)
                if not self._collision_events()["obstacle_collision"]:
                    return
                p.removeBody(self.obstacle_id, physicsClientId=self.client_id)
                self.obstacle_id = None
        else:
            raise RuntimeError("failed to sample a valid obstacle in 100 attempts")

    def _create_obstacle_body(self) -> None:
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.obstacle_radius,
            physicsClientId=self.client_id,
        )
        self.obstacle_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            basePosition=self.obstacle_position.tolist(),
            physicsClientId=self.client_id,
        )

    def episode_state_dict(self) -> dict[str, Any]:
        q, qdot = self.robot.joint_state_exact()
        return {
            "contract": self.contract, "q": q, "qdot": qdot,
            "goal_position": self.goal_position.copy(), "goal_quaternion": self.goal_quaternion.copy(),
            "goal_joint_positions": self.goal_joint_positions.copy(),
            "goal_ik_validation": self.goal_ik_validation.copy(),
            "goal_sample_attempts_last": self.goal_sample_attempts_last,
            "obstacle_position": self.obstacle_position.copy(), "obstacle_velocity": self.obstacle_velocity.copy(),
            "step_count": self.step_count, "obstacle_contact_seen": self.obstacle_contact_seen,
            "previous_rho_position": self.previous_rho_position,
            "previous_rho_orientation": self.previous_rho_orientation,
            "rng_state": self.rng.bit_generator.state,
        }

    def restore_episode_state(self, state: dict[str, Any]) -> np.ndarray:
        """Rebuild a boundary state saved after a complete control transition."""
        self.contract = state["contract"]
        p.resetSimulation(physicsClientId=self.client_id)
        p.setTimeStep(self.physics_dt, physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, 0.0, physicsClientId=self.client_id)
        self.robot_id = self.robot.load(self.rng, enable_self_collision=True)
        self._resolve_collision_links(); self._set_joint_state(state["q"], state["qdot"])
        self.goal_position = np.asarray(state["goal_position"], dtype=np.float32)
        self.goal_quaternion = np.asarray(state["goal_quaternion"], dtype=np.float32)
        self.goal_joint_positions = np.asarray(
            state.get("goal_joint_positions", np.zeros(6)), dtype=np.float32
        )
        self.goal_ik_validation = state.get("goal_ik_validation", {}).copy()
        self.goal_sample_attempts_last = int(state.get("goal_sample_attempts_last", 0))
        self.obstacle_position = np.asarray(state["obstacle_position"], dtype=np.float32)
        self.obstacle_velocity = np.asarray(state["obstacle_velocity"], dtype=np.float32)
        self.obstacle_id = None
        if self.contract.scene != "none": self._create_obstacle_body()
        self.step_count = int(state["step_count"])
        self.obstacle_contact_seen = bool(state["obstacle_contact_seen"])
        self.previous_rho_position = float(state["previous_rho_position"])
        self.previous_rho_orientation = float(state["previous_rho_orientation"])
        self.rng.bit_generator.state = state["rng_state"]
        p.performCollisionDetection(physicsClientId=self.client_id)
        return self._observe()[0]

    def _advance_obstacle(self) -> None:
        if self.obstacle_id is None or self.contract.scene != "dynamic":
            return
        position = (self.obstacle_position + self.obstacle_velocity * self.physics_dt).astype(np.float32)
        velocity = self.obstacle_velocity.copy()
        lows = np.asarray([0.22, -0.62, 0.18], dtype=np.float32)
        highs = np.asarray([0.72, 0.62, 0.62], dtype=np.float32)
        for axis in range(3):
            if position[axis] < lows[axis]:
                position[axis] = 2.0 * lows[axis] - position[axis]
                velocity[axis] = abs(velocity[axis])
            elif position[axis] > highs[axis]:
                position[axis] = 2.0 * highs[axis] - position[axis]
                velocity[axis] = -abs(velocity[axis])
        self.obstacle_position, self.obstacle_velocity = position, velocity
        p.resetBasePositionAndOrientation(
            self.obstacle_id,
            position.tolist(),
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        p.resetBaseVelocity(
            self.obstacle_id,
            linearVelocity=velocity.tolist(),
            physicsClientId=self.client_id,
        )

    def _observe(self):
        q, qdot = self.robot.joint_state()
        ee_position, ee_quaternion, ee_linear_velocity, ee_angular_velocity = (
            self.robot.end_effector_kinematics()
        )
        position_error, orientation_error = pose_error(
            self.goal_position, self.goal_quaternion, ee_position, ee_quaternion
        )
        geometry = self._geometry()
        self_geometry = self._self_geometry()
        observation = build_thesis_observation(
            q=q,
            qdot=qdot,
            joint_lower=self.robot.joint_lower_limits,
            joint_upper=self.robot.joint_upper_limits,
            joint_velocity_scale=self.action_scale,
            ee_position=ee_position,
            ee_quaternion=ee_quaternion,
            ee_linear_velocity=ee_linear_velocity,
            ee_angular_velocity=ee_angular_velocity,
            ee_linear_velocity_scale=self.ee_linear_velocity_scale,
            ee_angular_velocity_scale=self.ee_angular_velocity_scale,
            goal_position=self.goal_position,
            goal_quaternion=self.goal_quaternion,
            position_error=position_error,
            orientation_error=orientation_error,
            orientation_scale=self.contract.orientation_scale,
            remaining_time_fraction=max(0.0, (self.horizon - self.step_count) / self.horizon),
            obstacle_present=self.obstacle_id is not None,
            obstacle_position=self.obstacle_position,
            obstacle_velocity=self.obstacle_velocity,
            geometry=geometry,
            self_geometry=self_geometry,
        )
        info = {
            "goal_position": self.goal_position.copy(),
            "goal_quaternion": self.goal_quaternion.copy(),
            "goal_joint_positions": self.goal_joint_positions.copy(),
            "goal_ik_validation": self.goal_ik_validation.copy(),
            "goal_sample_attempts": self.goal_sample_attempts_last,
            "ee_position": ee_position.copy(),
            "ee_quaternion": ee_quaternion.copy(),
            "ee_linear_velocity": ee_linear_velocity.copy(),
            "ee_angular_velocity": ee_angular_velocity.copy(),
            "goal_error_norm": float(np.linalg.norm(position_error)),
            "orientation_error_norm": float(np.linalg.norm(orientation_error)),
        }
        return observation, info, geometry, self_geometry, position_error, orientation_error

    def _geometry(self) -> ThesisGeometry | None:
        if self.obstacle_id is None:
            return None
        return compute_thesis_geometry(
            self.robot.capsules(),
            self.obstacle_position,
            self.obstacle_velocity,
            self._closest_point_velocities(self.obstacle_position),
            obstacle_radius=self.obstacle_radius,
            d_safe=self.d_safe,
        )

    def _self_geometry(self) -> SelfCollisionGeometry:
        pair_distances = np.full(
            len(self._self_pair_link_ids), self.self_query_distance, dtype=np.float32
        )
        pair_approaches = np.zeros(len(self._self_pair_link_ids), dtype=np.float32)
        pair_lookup = {
            pair: index for index, pair in enumerate(self._self_pair_link_ids)
        }
        for point in p.getClosestPoints(
            bodyA=self.robot_id,
            bodyB=self.robot_id,
            distance=self.self_query_distance,
            physicsClientId=self.client_id,
        ):
            first, second = int(point[3]), int(point[4])
            if first == second:
                continue
            if first < second:
                pair = (first, second)
                point_first = np.asarray(point[5], dtype=np.float32)
                point_second = np.asarray(point[6], dtype=np.float32)
            else:
                pair = (second, first)
                point_first = np.asarray(point[6], dtype=np.float32)
                point_second = np.asarray(point[5], dtype=np.float32)
            pair_index = pair_lookup.get(pair)
            if pair_index is None or float(point[8]) >= pair_distances[pair_index]:
                continue
            vector = point_second - point_first
            norm = float(np.linalg.norm(vector))
            if norm > 1e-8:
                normal = vector / norm
            else:
                normal = -np.asarray(point[7], dtype=np.float32)
            velocity_first = self._link_world_point_velocity(first, point_first)
            velocity_second = self._link_world_point_velocity(second, point_second)
            distance_rate = float(np.dot(normal, velocity_second - velocity_first))
            pair_distances[pair_index] = float(point[8])
            pair_approaches[pair_index] = max(-distance_rate, 0.0)
        return compute_self_collision_geometry(
            pair_distances,
            pair_approaches,
            self._self_pair_slots,
            d_safe=self.d_self_safe,
            query_distance=self.self_query_distance,
            ttc_max=self.self_ttc_max,
            approach_velocity_scale=self.self_approach_velocity_scale,
            ttc_tau=self.self_ttc_tau,
        )

    def _link_world_point_velocity(
        self, link_id: int, world_point: np.ndarray
    ) -> np.ndarray:
        state = p.getLinkState(
            self.robot_id,
            link_id,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        center_of_mass = np.asarray(state[0], dtype=np.float32)
        linear = np.asarray(state[6], dtype=np.float32)
        angular = np.asarray(state[7], dtype=np.float32)
        return linear + np.cross(angular, np.asarray(world_point) - center_of_mass)

    def _closest_point_velocities(self, obstacle_position: np.ndarray) -> np.ndarray:
        capsules = self.robot.capsules()
        velocities = []
        for capsule, spec in zip(capsules, self.robot.capsule_model.specs):
            _, fraction = closest_point_on_segment(obstacle_position, capsule.start, capsule.end)
            start_velocity = self._link_point_velocity(spec.parent_link_name, spec.parent_offset)
            end_velocity = self._link_point_velocity(spec.child_link_name, spec.child_offset)
            velocities.append((1.0 - fraction) * start_velocity + fraction * end_velocity)
        return np.asarray(velocities, dtype=np.float32)

    def _link_point_velocity(self, link_name: str, local_offset: np.ndarray) -> np.ndarray:
        link_id = self.robot.capsule_model.link_name_to_id[link_name]
        if link_id < 0:
            return np.zeros(3, dtype=np.float32)
        state = p.getLinkState(
            self.robot_id,
            link_id,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        rotation = np.asarray(p.getMatrixFromQuaternion(state[5]), dtype=np.float32).reshape(3, 3)
        radius = rotation @ np.asarray(local_offset, dtype=np.float32)
        linear = np.asarray(state[6], dtype=np.float32)
        angular = np.asarray(state[7], dtype=np.float32)
        return linear + np.cross(angular, radius)

    def _resolve_collision_links(self) -> None:
        self._link_name_to_id = {}
        for joint_id in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.robot_id, joint_id, physicsClientId=self.client_id)
            self._link_name_to_id[info[12].decode("utf-8")] = joint_id
        collision_link_names = {
            link_name
            for spec in self.robot.capsule_model.specs
            for link_name in (spec.parent_link_name, spec.child_link_name)
        }
        self._collision_link_ids = {
            self._link_name_to_id[link_name]
            for link_name in collision_link_names
            if self._link_name_to_id[link_name] >= 0
        }
        names = [
            "base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link",
            "wrist_1_link", "wrist_2_link", "wrist_3_link",
        ]
        self._self_link_order = [self._link_name_to_id[name] for name in names]
        self._self_pair_link_ids = []
        self._self_pair_slots = []
        for first_index, first_link in enumerate(self._self_link_order):
            for second_index in range(first_index + 2, len(self._self_link_order)):
                second_link = self._self_link_order[second_index]
                self._self_pair_link_ids.append(tuple(sorted((first_link, second_link))))
                self._self_pair_slots.append(
                    (first_index - 1, second_index - 1)
                )

    def _collision_events(self) -> dict[str, bool]:
        obstacle_collision = False
        if self.obstacle_id is not None:
            obstacle_collision = any(
                point[8] <= 0.0 and point[3] in self._collision_link_ids
                for point in p.getContactPoints(
                    bodyA=self.robot_id,
                    bodyB=self.obstacle_id,
                    physicsClientId=self.client_id,
                )
            )
        self_collision = False
        order = {link_id: index for index, link_id in enumerate(self._self_link_order)}
        for point in p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id, physicsClientId=self.client_id):
            first, second = int(point[3]), int(point[4])
            if point[8] > 0.0 or first not in order or second not in order:
                continue
            if abs(order[first] - order[second]) > 1:
                self_collision = True
                break
        return {
            "obstacle_collision": bool(obstacle_collision),
            "self_collision": bool(self_collision),
            "environment_collision": False,
        }

    def _set_joint_state(self, q: np.ndarray, qdot: np.ndarray | None = None) -> None:
        velocity = np.zeros(6, dtype=np.float64) if qdot is None else np.asarray(qdot, dtype=np.float64)
        for joint_id, position, speed in zip(self.robot.joint_ids, q, velocity):
            p.resetJointState(
                self.robot_id,
                joint_id,
                float(position),
                targetVelocity=float(speed),
                physicsClientId=self.client_id,
            )

    def _contract_info(self) -> dict[str, Any]:
        return {
            "scene": self.contract.scene,
            "xi_scene": self.contract.xi,
            "strict_obstacle_collision": self.contract.strict_obstacle_collision,
            "goal_scale": self.contract.goal_scale,
            "orientation_scale": self.contract.orientation_scale,
            "orientation_tolerance": self.contract.orientation_tolerance,
            "lambda_self": self.contract.lambda_self,
            "obstacle_enabled": self.obstacle_id is not None,
        }

    @staticmethod
    def _geometry_info(geometry: ThesisGeometry | None) -> dict[str, Any]:
        if geometry is None:
            return {"risk_body": np.zeros(6, dtype=np.float32), "d_min": 0.80, "risk_global": 0.0}
        return {
            "risk_body": geometry.risk_per_link.copy(),
            "distance_body": geometry.distances.copy(),
            "ttc_body": geometry.ttc.copy(),
            "d_min": geometry.distance_min,
            "risk_global": geometry.risk_max,
        }

    @staticmethod
    def _self_geometry_info(geometry: SelfCollisionGeometry) -> dict[str, Any]:
        return {
            "self_distance_body": geometry.distances.copy(),
            "self_ttc_body": geometry.ttc.copy(),
            "self_approach_body": geometry.approach_velocities.copy(),
            "self_risk_body": geometry.risk_per_link.copy(),
            "self_d_min": geometry.distance_min,
            "self_risk_global": geometry.risk_max,
            "self_ttc_min": geometry.ttc_min,
            "self_approach_max": geometry.approach_max,
            "self_closest_pair": geometry.closest_pair,
        }

    def close(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)
