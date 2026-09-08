from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from rl_risk_sac.collision import LinkRiskDetector, NullRiskDetector, RiskDetector
from rl_risk_sac.control import ExecutionPipeline
from rl_risk_sac.robots.ur5_capsules import CapsuleState, UR5CapsuleModel
from rl_risk_sac.scene import ObstacleState, SphericalObstacleProvider
from rl_risk_sac.tasks import ReachingObservationBuilder, ReachingObjective, WorkspaceTargetProvider
from rl_risk_sac.utils.risk import RiskConfig


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
        self.robot_cfg = config.get("robot", {})
        self.observation_cfg = env_cfg.get("observation", {})
        self.visual_cfg = env_cfg.get("visual", {})
        self.execution_cfg = env_cfg.get("execution", {})
        self.fixed_smoothing_mode = str(self.execution_cfg["fixed_smoothing_mode"])
        self.obstacle_enabled = bool(self.obstacle_cfg.get("enabled", True))

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
        p.setGravity(*env_cfg["gravity"], physicsClientId=self.physics_client_id)

        self.repo_root = Path(__file__).resolve().parents[3]
        robot_urdf = Path(self.robot_cfg["urdf"])
        self.robot_urdf = robot_urdf if robot_urdf.is_absolute() else self.repo_root / robot_urdf
        self.capsule_model = UR5CapsuleModel(self.robot_cfg.get("capsules"))
        self.joint_ids: list[int] = []
        self.tool_link_id = -1
        self.joint_count = len(self.robot_cfg["joint_names"])
        self.execution_pipeline = ExecutionPipeline(
            joint_count=self.joint_count,
            action_scale=self.action_scale,
            control_dt=self.control_dt,
            smoothing_mode=self.fixed_smoothing_mode,
            fixed_beta=self.fixed_beta,
            cutoff_angular_frequency=float(self.execution_cfg["rtb"]["cutoff_angular_frequency"]),
            max_policy_velocity_delta=self.execution_cfg.get("max_policy_velocity_delta"),
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            risk_high=self.risk_high,
            lambda_beta=self.lambda_beta,
        )
        # Kept as a compatibility alias for analysis scripts that inspected the
        # RTB filter state directly before the execution pipeline was extracted.
        self.fixed_rtb = self.execution_pipeline.rtb
        distance_clip = self.observation_cfg["distance_clip"]
        self.observation_builder = ReachingObservationBuilder(
            action_scale=self.action_scale,
            distance_clip=(float(distance_clip[0]), float(distance_clip[1])),
            v_max=self.risk_config.v_max,
            ttc_max=self.risk_config.ttc_max,
        )
        self.objective = ReachingObjective(self.reward_cfg, self.cost_cfg, self.risk_config.d_safe)
        self.target_provider = WorkspaceTargetProvider(self.goal_cfg, self.workspace)
        self.obstacle_provider = SphericalObstacleProvider(self.obstacle_cfg)
        detector_args = {
            "config": self.risk_config,
            "obstacle_radius": float(self.obstacle_cfg["radius"]),
            "dt": self.control_dt,
            "end_effector_only": self.method == "ee_fixed",
            "no_obstacle_distance": float(self.observation_cfg["no_obstacle_distance"]),
        }
        self.risk_detector: RiskDetector = (
            LinkRiskDetector(**detector_args)
            if self.obstacle_enabled
            else NullRiskDetector(
                config=self.risk_config,
                no_obstacle_distance=float(self.observation_cfg["no_obstacle_distance"]),
            )
        )

        self.robot_id: int | None = None
        self.obstacle_id: int | None = None
        self.obstacle_ids: list[int] = []
        self.goal_marker_id: int | None = None
        self.prev_capsules: list[CapsuleState] | None = None
        self.prev_goal_error_norm = 0.0
        self.prev_qdot_cmd = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_joint_acc = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_physics_qdot_cmd = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_physics_joint_acc = np.zeros(self.joint_count, dtype=np.float32)
        self.beta = self.fixed_beta
        self.step_count = 0
        self.goal = np.zeros(3, dtype=np.float32)
        self.goal_velocity = np.zeros(3, dtype=np.float32)
        self.obstacle_center = np.zeros(3, dtype=np.float32)
        self.obstacle_velocity = np.zeros(3, dtype=np.float32)
        self.obstacle_states: tuple[ObstacleState, ...] = ()
        self.last_risk = None
        self.last_info: dict[str, Any] = {}

        obs_dim = self.observation_builder.dimension(self.joint_count, self.capsule_model.count)
        obs_bound = float(self.observation_cfg["space_bound"])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.joint_count,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_bound, high=obs_bound, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        p.resetSimulation(physicsClientId=self.physics_client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client_id)
        p.setGravity(*self.config["env"]["gravity"], physicsClientId=self.physics_client_id)
        self._create_floor()
        self.robot_id = p.loadURDF(
            str(self.robot_urdf),
            basePosition=self.robot_cfg["base_position"],
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        self._resolve_robot_references()
        self._reset_robot()
        target_state = self.target_provider.reset(self.rng)
        self.goal = target_state.position
        self.goal_velocity = target_state.velocity
        self.obstacle_states = self.obstacle_provider.reset(self.rng)
        self._sync_legacy_obstacle_state()
        self.obstacle_ids = (
            [self._create_obstacle(obstacle.center) for obstacle in self.obstacle_states if obstacle.enabled]
            if self.obstacle_enabled
            else []
        )
        self.obstacle_id = self.obstacle_ids[0] if self.obstacle_ids else None
        self.goal_marker_id = self._create_goal_marker(self.goal)

        self.prev_qdot_cmd = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_joint_acc = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_physics_qdot_cmd = np.zeros(self.joint_count, dtype=np.float32)
        self.prev_physics_joint_acc = np.zeros(self.joint_count, dtype=np.float32)
        self.execution_pipeline.reset(adaptive=self.method.endswith("adaptive"))
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
        execution = self.execution_pipeline.process(
            normalized_action=action,
            previous_command=self.prev_qdot_cmd,
            risk_global=pre_risk.risk_global,
            sample_count=self.sim_substeps,
            adaptive=self.method.endswith("adaptive"),
        )
        qdot_trajectory = execution.trajectory
        qdot_cmd = execution.command
        beta = execution.beta

        physics_accelerations = []
        physics_jerks = []
        for qdot_substep in qdot_trajectory:
            self._advance_obstacle(self.time_step)
            self._move_target(self.time_step)
            p.setJointMotorControlArray(
                self.robot_id,
                self.joint_ids,
                p.VELOCITY_CONTROL,
                targetVelocities=qdot_substep.tolist(),
                forces=[float(self.execution_cfg["joint_motor_force"])] * self.joint_count,
                physicsClientId=self.physics_client_id,
            )
            p.stepSimulation(physicsClientId=self.physics_client_id)
            physics_acc = (qdot_substep - self.prev_physics_qdot_cmd) / self.time_step
            physics_jerk = (physics_acc - self.prev_physics_joint_acc) / self.time_step
            physics_accelerations.append(physics_acc.copy())
            physics_jerks.append(physics_jerk.copy())
            self.prev_physics_qdot_cmd = qdot_substep.copy()
            self.prev_physics_joint_acc = physics_acc.copy()

        self.step_count += 1
        obs, info = self._get_obs_and_info()
        risk = self.last_risk
        collision = self._has_collision(risk.d_min)
        success = info["goal_error_norm"] < self.success_tolerance and risk.d_min > self.risk_config.d_safe
        terminated = bool(collision or success)
        truncated = self.step_count >= self.max_episode_steps

        reward = self.objective.reward(
            goal_error_norm=float(info["goal_error_norm"]),
            previous_goal_error_norm=self.prev_goal_error_norm,
            command=qdot_cmd,
            previous_command=self.prev_qdot_cmd,
            success=success,
            collision=collision,
        )
        cost = self.objective.cost(risk, collision)
        if self.method in {"ee_fixed", "link_fixed"}:
            reward -= float(self.config["sac"]["fixed_risk_penalty"]) * cost

        joint_acc = (qdot_cmd - self.prev_qdot_cmd) / self.control_dt
        jerk = (joint_acc - self.prev_joint_acc) / self.control_dt
        physics_acceleration = np.asarray(physics_accelerations, dtype=np.float32)
        physics_jerk = np.asarray(physics_jerks, dtype=np.float32)
        info.update(
            {
                "reward": float(reward),
                "cost": float(cost),
                "collision": bool(collision),
                "success": bool(success),
                "qdot_cmd": qdot_cmd.copy(),
                "qdot_policy": execution.policy_velocity.copy(),
                "qdot_policy_limited": execution.limited_policy_velocity.copy(),
                "policy_rate_limited": bool(execution.rate_limited),
                "joint_acc": joint_acc.copy(),
                "joint_jerk": jerk.copy(),
                "physics_rms_acceleration": float(np.sqrt(np.mean(np.square(physics_acceleration)))),
                "physics_rms_jerk": float(np.sqrt(np.mean(np.square(physics_jerk)))),
                "qdot_substeps": qdot_trajectory.copy(),
                "joint_acc_substeps": physics_acceleration.copy(),
                "joint_jerk_substeps": physics_jerk.copy(),
                "beta": float(beta),
                "fixed_smoothing_mode": self.fixed_smoothing_mode,
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
        goal_velocity_error = self.goal_velocity - ee_vel
        risk = self._compute_risk()
        self.last_risk = risk

        obs = self.observation_builder.build(
            q=q,
            q_dot=q_dot,
            goal_error=goal_error,
            goal_velocity_error=goal_velocity_error,
            risk=risk,
            previous_command=self.prev_qdot_cmd,
            beta=self.beta,
        )
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(f"Observation shape {obs.shape} does not match {self.observation_space.shape}")

        info = {
            "observation_schema": str(self.observation_cfg["schema_version"]),
            "goal": self.goal.copy(),
            "goal_velocity": self.goal_velocity.copy(),
            "ee_pos": ee_pos.copy(),
            "goal_error_norm": float(np.linalg.norm(goal_error)),
            "obstacle_enabled": bool(self.obstacle_enabled),
            "obstacle_center": self.obstacle_center.copy(),
            "obstacle_velocity": self.obstacle_velocity.copy(),
            "obstacle_centers": np.asarray([state.center for state in self.obstacle_states], dtype=np.float32),
            "obstacle_velocities": np.asarray([state.velocity for state in self.obstacle_states], dtype=np.float32),
            "obstacle_count": int(sum(state.enabled for state in self.obstacle_states)),
            "risk_global": float(risk.risk_global),
            "d_min": float(risk.d_min),
            "closest_link": int(risk.closest_link),
            "safety_violation": bool(risk.d_min < self.risk_config.d_safe),
            "risk_body": risk.risks.copy(),
        }
        return obs, info

    def _compute_risk(self):
        return self.risk_detector.detect(
            capsules=self._capsules(),
            previous_capsules=self.prev_capsules,
            obstacles=self.obstacle_states,
        )

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
        if not self.obstacle_enabled or not self.obstacle_ids:
            return False
        if d_min <= 0.0:
            return True
        return any(
            p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=obstacle_id,
                physicsClientId=self.physics_client_id,
            )
            for obstacle_id in self.obstacle_ids
        )

    def _reset_robot(self) -> None:
        reset_cfg = self.robot_cfg["reset"]
        default = np.asarray(reset_cfg["default_joint_positions"], dtype=np.float32)
        noise_range = float(reset_cfg["joint_noise_range"])
        noise = self.rng.uniform(-noise_range, noise_range, size=self.joint_count).astype(np.float32)
        q0 = default + noise
        for joint_id, joint_value in zip(self.joint_ids, q0):
            p.resetJointState(
                self.robot_id,
                joint_id,
                float(joint_value),
                targetVelocity=0.0,
                physicsClientId=self.physics_client_id,
            )

    def _resolve_robot_references(self) -> None:
        joint_name_to_id, link_name_to_id = self._robot_name_maps()
        self.joint_ids = [self._resolve_name(name, joint_name_to_id, "joint") for name in self.robot_cfg["joint_names"]]
        self.tool_link_id = self._resolve_name(self.robot_cfg["tool_link_name"], link_name_to_id, "link")
        self.joint_count = len(self.joint_ids)
        self.capsule_model.resolve_link_names(link_name_to_id)

    def _robot_name_maps(self) -> tuple[dict[str, int], dict[str, int]]:
        joint_name_to_id: dict[str, int] = {}
        link_name_to_id = {"base": -1}
        for joint_id in range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client_id)):
            info = p.getJointInfo(self.robot_id, joint_id, physicsClientId=self.physics_client_id)
            joint_name_to_id[info[1].decode("utf-8")] = joint_id
            link_name_to_id[info[12].decode("utf-8")] = joint_id
        return joint_name_to_id, link_name_to_id

    @staticmethod
    def _resolve_name(name: str, name_to_id: dict[str, int], kind: str) -> int:
        if name not in name_to_id:
            available = ", ".join(sorted(name_to_id))
            raise KeyError(f"Unknown {kind} name {name!r}; available {kind}s: {available}")
        return name_to_id[name]

    def _move_target(self, dt: float) -> None:
        target_state = self.target_provider.advance(dt)
        self.goal = target_state.position
        self.goal_velocity = target_state.velocity
        if self.goal_marker_id is not None:
            p.resetBasePositionAndOrientation(
                self.goal_marker_id,
                self.goal.tolist(),
                [0, 0, 0, 1],
                physicsClientId=self.physics_client_id,
            )

    def _advance_obstacle(self, dt: float) -> None:
        self.obstacle_states = self.obstacle_provider.advance(dt)
        self._sync_legacy_obstacle_state()
        if not self.obstacle_enabled or not self.obstacle_ids:
            return
        active_states = [state for state in self.obstacle_states if state.enabled]
        for obstacle_id, obstacle_state in zip(self.obstacle_ids, active_states):
            p.resetBasePositionAndOrientation(
                obstacle_id,
                obstacle_state.center.tolist(),
                [0, 0, 0, 1],
                physicsClientId=self.physics_client_id,
            )

    def _sync_legacy_obstacle_state(self) -> None:
        if not self.obstacle_states:
            self.obstacle_center = np.asarray(self.obstacle_cfg["disabled_position"], dtype=np.float32)
            self.obstacle_velocity = np.zeros(3, dtype=np.float32)
            return
        self.obstacle_center = self.obstacle_states[0].center
        self.obstacle_velocity = self.obstacle_states[0].velocity

    def _create_floor(self) -> None:
        collision = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.physics_client_id)
        visual = p.createVisualShape(
            p.GEOM_PLANE,
            rgbaColor=self.visual_cfg["floor_rgba"],
            physicsClientId=self.physics_client_id,
        )
        p.createMultiBody(
            0,
            collision,
            visual,
            self.visual_cfg["floor_position"],
            physicsClientId=self.physics_client_id,
        )

    def _create_obstacle(self, center: np.ndarray) -> int:
        radius = float(self.obstacle_cfg["radius"])
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.physics_client_id)
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=self.visual_cfg["obstacle_rgba"],
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
            radius=float(self.visual_cfg["goal_marker_radius"]),
            rgbaColor=self.visual_cfg["goal_rgba"],
            physicsClientId=self.physics_client_id,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=goal.tolist(),
            physicsClientId=self.physics_client_id,
        )
