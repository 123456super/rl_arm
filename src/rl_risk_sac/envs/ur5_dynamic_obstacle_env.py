from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from rl_risk_sac.robots.ur5_capsules import CapsuleState, UR5CapsuleModel
from rl_risk_sac.utils.risk import RiskConfig, compute_link_risk


METHODS = {"ee_fixed", "link_fixed", "ldrc_fixed", "ldrc_adaptive"}


class UR5DynamicObstacleEnv(gym.Env):
    """UR5-like joint-velocity reaching task with one dynamic spherical obstacle."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: dict[str, Any], method: str = "ldrc_adaptive", render_mode: str | None = None):
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {sorted(METHODS)}")
        self.config = config
        self.method = method
        self.render_mode = render_mode

        env_cfg = config["env"]
        risk_cfg = config["risk"]
        smoothing_cfg = config["smoothing"]
        self.reward_cfg = config["reward"]

        self.time_step = float(env_cfg["time_step"])
        self.control_dt = float(env_cfg["control_dt"])
        self.sim_substeps = max(1, int(round(self.control_dt / self.time_step)))
        self.max_episode_steps = int(env_cfg["max_episode_steps"])
        self.action_scale = float(env_cfg["action_scale"])
        self.fixed_beta = float(env_cfg["fixed_beta"])
        self.success_tolerance = float(env_cfg["success_tolerance"])
        self.workspace = env_cfg["workspace"]
        self.obstacle_cfg = env_cfg["obstacle"]
        self.goal_cfg = env_cfg["goal"]

        self.beta_min = float(smoothing_cfg["beta_min"])
        self.beta_max = float(smoothing_cfg["beta_max"])
        self.risk_high = float(smoothing_cfg["risk_high"])
        self.lambda_beta = float(smoothing_cfg["lambda_beta"])

        weights = risk_cfg["weights"]
        self.risk_config = RiskConfig(
            d_safe=float(risk_cfg["d_safe"]),
            sigma_d=float(risk_cfg["sigma_d"]),
            v_max=float(risk_cfg["v_max"]),
            ttc_max=float(risk_cfg["ttc_max"]),
            tau=float(risk_cfg["tau"]),
            eps=float(risk_cfg["eps"]),
            eps_v=float(risk_cfg["eps_v"]),
            w_distance=float(weights["distance"]),
            w_velocity=float(weights["velocity"]),
            w_ttc=float(weights["ttc"]),
        )
        self.cost_cfg = risk_cfg["cost"]

        self.rng = np.random.default_rng(int(config.get("seed", 42)))
        self.physics_client_id = p.connect(p.GUI if render_mode == "human" or env_cfg.get("gui") else p.DIRECT)
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client_id)
        p.setGravity(0, 0, 0, physicsClientId=self.physics_client_id)

        self.repo_root = Path(__file__).resolve().parents[3]
        self.robot_urdf = self.repo_root / "assets" / "urdf" / "ur5_like.urdf"
        self.capsule_model = UR5CapsuleModel()
        self.joint_ids = list(range(6))
        self.tool_link_id = 6

        self.robot_id: int | None = None
        self.obstacle_id: int | None = None
        self.goal_marker_id: int | None = None
        self.prev_capsules: list[CapsuleState] | None = None
        self.prev_goal_error_norm = 0.0
        self.prev_qdot_cmd = np.zeros(6, dtype=np.float32)
        self.prev_joint_acc = np.zeros(6, dtype=np.float32)
        self.beta = self.fixed_beta
        self.step_count = 0
        self.goal = np.zeros(3, dtype=np.float32)
        self.obstacle_center = np.zeros(3, dtype=np.float32)
        self.obstacle_velocity = np.zeros(3, dtype=np.float32)
        self.last_risk = None
        self.last_info: dict[str, Any] = {}

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(67,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        p.resetSimulation(physicsClientId=self.physics_client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client_id)
        p.setGravity(0, 0, 0, physicsClientId=self.physics_client_id)
        self._create_floor()
        self.robot_id = p.loadURDF(
            str(self.robot_urdf),
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        self._reset_robot()
        self.goal = self._sample_goal()
        self.obstacle_center, self.obstacle_velocity = self._sample_obstacle()
        self.obstacle_id = self._create_obstacle(self.obstacle_center)
        self.goal_marker_id = self._create_goal_marker(self.goal)

        self.prev_qdot_cmd = np.zeros(6, dtype=np.float32)
        self.prev_joint_acc = np.zeros(6, dtype=np.float32)
        self.beta = self.fixed_beta if self.method.endswith("fixed") else self.beta_min
        self.step_count = 0
        self.prev_capsules = self._capsules()
        self.prev_goal_error_norm = float(np.linalg.norm(self._goal_error()))
        obs, info = self._get_obs_and_info()
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        pre_risk = self._compute_risk()
        qdot_policy = self.action_scale * action
        if self.method.endswith("adaptive"):
            beta = self._adaptive_beta(pre_risk.risk_global)
        else:
            beta = self.fixed_beta
        qdot_cmd = beta * qdot_policy + (1.0 - beta) * self.prev_qdot_cmd
        qdot_cmd = np.clip(qdot_cmd, -self.action_scale, self.action_scale).astype(np.float32)

        for _ in range(self.sim_substeps):
            self._move_obstacle(self.time_step)
            p.setJointMotorControlArray(
                self.robot_id,
                self.joint_ids,
                p.VELOCITY_CONTROL,
                targetVelocities=qdot_cmd.tolist(),
                forces=[90.0] * 6,
                physicsClientId=self.physics_client_id,
            )
            p.stepSimulation(physicsClientId=self.physics_client_id)

        self.step_count += 1
        obs, info = self._get_obs_and_info()
        risk = self.last_risk
        collision = self._has_collision(risk.d_min)
        success = info["goal_error_norm"] < self.success_tolerance and risk.d_min > self.risk_config.d_safe
        terminated = bool(collision or success)
        truncated = self.step_count >= self.max_episode_steps

        reward = self._reward(qdot_cmd, success, collision)
        cost = self._cost(risk, collision)
        if self.method in {"ee_fixed", "link_fixed"}:
            reward -= float(self.config["sac"]["fixed_risk_penalty"]) * cost

        joint_acc = (qdot_cmd - self.prev_qdot_cmd) / self.control_dt
        jerk = (joint_acc - self.prev_joint_acc) / self.control_dt
        info.update(
            {
                "reward": float(reward),
                "cost": float(cost),
                "collision": bool(collision),
                "success": bool(success),
                "qdot_cmd": qdot_cmd.copy(),
                "joint_acc": joint_acc.copy(),
                "joint_jerk": jerk.copy(),
                "beta": float(beta),
            }
        )
        self.prev_qdot_cmd = qdot_cmd
        self.prev_joint_acc = joint_acc
        self.prev_capsules = self._capsules()
        self.prev_goal_error_norm = info["goal_error_norm"]
        self.beta = float(beta)
        self.last_info = info
        return obs, float(reward), float(cost), terminated, truncated, info

    def close(self) -> None:
        if p.isConnected(self.physics_client_id):
            p.disconnect(self.physics_client_id)

    def get_metrics(self) -> dict[str, Any]:
        return dict(self.last_info)

    def _get_obs_and_info(self) -> tuple[np.ndarray, dict[str, Any]]:
        q, q_dot = self._joint_state()
        ee_pos, ee_vel = self._end_effector_state()
        goal_error = self.goal - ee_pos
        goal_velocity_error = -ee_vel
        risk = self._compute_risk()
        self.last_risk = risk

        obs = np.concatenate(
            [
                q / np.pi,
                q_dot / max(self.action_scale, 1e-6),
                goal_error,
                goal_velocity_error,
                np.clip(risk.distances, -1.0, 1.5),
                risk.directions.reshape(-1),
                np.clip(risk.approach_velocities / max(self.risk_config.v_max, 1e-6), 0.0, 1.0),
                risk.ttc / max(self.risk_config.ttc_max, 1e-6),
                risk.risks,
                self.prev_qdot_cmd / max(self.action_scale, 1e-6),
                np.array([self.beta], dtype=np.float32),
            ]
        ).astype(np.float32)
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(f"Observation shape {obs.shape} does not match {self.observation_space.shape}")

        info = {
            "goal": self.goal.copy(),
            "ee_pos": ee_pos.copy(),
            "goal_error_norm": float(np.linalg.norm(goal_error)),
            "risk_global": float(risk.risk_global),
            "d_min": float(risk.d_min),
            "closest_link": int(risk.closest_link),
            "safety_violation": bool(risk.d_min < self.risk_config.d_safe),
            "risk_body": risk.risks.copy(),
        }
        return obs, info

    def _compute_risk(self):
        use_ee_only = self.method == "ee_fixed"
        return compute_link_risk(
            capsules=self._capsules(),
            prev_capsules=self.prev_capsules,
            obstacle_center=self.obstacle_center,
            obstacle_velocity=self.obstacle_velocity,
            obstacle_radius=float(self.obstacle_cfg["radius"]),
            dt=self.control_dt,
            config=self.risk_config,
            use_end_effector_only=use_ee_only,
        )

    def _reward(self, qdot_cmd: np.ndarray, success: bool, collision: bool) -> float:
        goal_error_norm = float(np.linalg.norm(self._goal_error()))
        progress = self.prev_goal_error_norm - goal_error_norm
        smooth = float(np.sum(np.square(qdot_cmd - self.prev_qdot_cmd)))
        reward = (
            -float(self.reward_cfg["w_position"]) * goal_error_norm**2
            + float(self.reward_cfg["w_progress"]) * progress
            - float(self.reward_cfg["w_smooth"]) * smooth
        )
        if success:
            reward += float(self.reward_cfg["success_bonus"])
        if collision:
            reward -= float(self.reward_cfg["collision_penalty"])
        return float(reward)

    def _cost(self, risk, collision: bool) -> float:
        violation = 1.0 if risk.d_min < self.risk_config.d_safe else 0.0
        collision_f = 1.0 if collision else 0.0
        return float(
            float(self.cost_cfg["k_risk"]) * risk.risk_global
            + float(self.cost_cfg["k_violation"]) * violation
            + float(self.cost_cfg["k_collision"]) * collision_f
        )

    def _adaptive_beta(self, risk_global: float) -> float:
        ratio = np.clip(risk_global / max(self.risk_high, 1e-6), 0.0, 1.0)
        beta_raw = self.beta_min + (self.beta_max - self.beta_min) * ratio
        return float(self.lambda_beta * beta_raw + (1.0 - self.lambda_beta) * self.beta)

    def _joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        states = p.getJointStates(self.robot_id, self.joint_ids, physicsClientId=self.physics_client_id)
        q = np.asarray([s[0] for s in states], dtype=np.float32)
        q_dot = np.asarray([s[1] for s in states], dtype=np.float32)
        return q, q_dot

    def _end_effector_state(self) -> tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(
            self.robot_id,
            self.tool_link_id,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
            physicsClientId=self.physics_client_id,
        )
        return np.asarray(state[4], dtype=np.float32), np.asarray(state[6], dtype=np.float32)

    def _goal_error(self) -> np.ndarray:
        ee_pos, _ = self._end_effector_state()
        return self.goal - ee_pos

    def _capsules(self) -> list[CapsuleState]:
        return self.capsule_model.states(self.robot_id, self.physics_client_id)

    def _has_collision(self, d_min: float) -> bool:
        if d_min <= 0.0:
            return True
        contacts = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.obstacle_id,
            physicsClientId=self.physics_client_id,
        )
        return len(contacts) > 0

    def _reset_robot(self) -> None:
        default = np.array([0.0, -1.25, 1.35, -0.95, -1.1, 0.0], dtype=np.float32)
        noise = self.rng.uniform(-0.22, 0.22, size=6).astype(np.float32)
        q0 = default + noise
        for joint_id, joint_value in zip(self.joint_ids, q0):
            p.resetJointState(
                self.robot_id,
                joint_id,
                float(joint_value),
                targetVelocity=0.0,
                physicsClientId=self.physics_client_id,
            )

    def _sample_goal(self) -> np.ndarray:
        if self.goal_cfg.get("fixed", False):
            return np.asarray(self.goal_cfg["position"], dtype=np.float32)
        return np.array(
            [
                self.rng.uniform(*self.workspace["x"]),
                self.rng.uniform(*self.workspace["y"]),
                self.rng.uniform(*self.workspace["z"]),
            ],
            dtype=np.float32,
        )

    def _sample_obstacle(self) -> tuple[np.ndarray, np.ndarray]:
        target_band_z = self.rng.uniform(0.18, 0.62)
        side = -1.0 if self.rng.random() < 0.5 else 1.0
        center = np.array(
            [
                self.rng.uniform(0.22, 0.72),
                side * self.rng.uniform(0.42, 0.62),
                target_band_z,
            ],
            dtype=np.float32,
        )
        target = np.array(
            [
                self.rng.uniform(0.22, 0.72),
                -side * self.rng.uniform(0.24, 0.48),
                self.rng.uniform(0.18, 0.62),
            ],
            dtype=np.float32,
        )
        direction = target - center
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        speed = self.rng.uniform(*self.obstacle_cfg["speed_range"])
        return center, (direction * speed).astype(np.float32)

    def _move_obstacle(self, dt: float) -> None:
        self.obstacle_center = (self.obstacle_center + self.obstacle_velocity * dt).astype(np.float32)
        x_low, x_high = self.workspace["x"]
        y_low, y_high = -0.7, 0.7
        z_low, z_high = 0.08, 0.85
        for axis, (low, high) in enumerate([(x_low, x_high), (y_low, y_high), (z_low, z_high)]):
            if self.obstacle_center[axis] < low or self.obstacle_center[axis] > high:
                self.obstacle_velocity[axis] *= -1.0
                self.obstacle_center[axis] = np.clip(self.obstacle_center[axis], low, high)
        p.resetBasePositionAndOrientation(
            self.obstacle_id,
            self.obstacle_center.tolist(),
            [0, 0, 0, 1],
            physicsClientId=self.physics_client_id,
        )

    def _create_floor(self) -> None:
        collision = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.physics_client_id)
        visual = p.createVisualShape(
            p.GEOM_PLANE,
            rgbaColor=[0.82, 0.84, 0.86, 1.0],
            physicsClientId=self.physics_client_id,
        )
        p.createMultiBody(0, collision, visual, [0, 0, -0.005], physicsClientId=self.physics_client_id)

    def _create_obstacle(self, center: np.ndarray) -> int:
        radius = float(self.obstacle_cfg["radius"])
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.physics_client_id)
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[0.95, 0.16, 0.12, 1.0],
            physicsClientId=self.physics_client_id,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=center.tolist(),
            physicsClientId=self.physics_client_id,
        )

    def _create_goal_marker(self, goal: np.ndarray) -> int:
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.035,
            rgbaColor=[0.12, 0.64, 0.28, 0.75],
            physicsClientId=self.physics_client_id,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=goal.tolist(),
            physicsClientId=self.physics_client_id,
        )
