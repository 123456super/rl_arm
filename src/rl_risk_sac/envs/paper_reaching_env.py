"""Independent static-obstacle reaching task with the reference's 25D state."""
from __future__ import annotations

import copy

import numpy as np
import pybullet as p
from gymnasium import spaces

from rl_risk_sac.robots.pybullet_robot import PyBulletRobot
from rl_risk_sac.tasks.paper_reaching import (
    PAPER_OBSERVATION_DIM, PAPER_POSITION_THRESHOLDS_MM,
    combined_pose_error, paper_observation, paper_reward,
    point_segment_distances, pose_errors, reference_error_components,
    sample_goal_position, sample_obstacle_center,
)
from rl_risk_sac.tasks.thesis_reaching import pose_error
from rl_risk_sac.utils.paper_config import PAPER_PROTOCOL, experiment_signature, robot_runtime, robot_urdf


class PaperReachingEnv:
    def __init__(self, config, *, gui=False):
        self.config = copy.deepcopy(config)
        self.task = self.config["paper"]
        self.rng = np.random.default_rng(config["seed"])
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        self.observation_space = spaces.Box(-np.inf, np.inf, (PAPER_OBSERVATION_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., (6,), dtype=np.float32)
        self.initialized = False
        p.setTimeStep(self.task["physics_dt"], physicsClientId=self.client_id)
        p.setGravity(0, 0, self.task["gravity"], physicsClientId=self.client_id)
        # No point-cloud/vision model or rendering work in the training workers.
        self.robot = PyBulletRobot(robot_runtime(config), robot_urdf(config), self.client_id)
        self.robot_id = self.robot.load(self.rng, enable_self_collision=True)
        self.initial_q = np.asarray(config["robot"]["reset"]["default_joint_positions"], dtype=np.float64)
        self.control_lower = np.maximum(self.robot.joint_lower_limits, self.initial_q - self.task["action_limit"])
        self.control_upper = np.minimum(self.robot.joint_upper_limits, self.initial_q + self.task["action_limit"])
        force = self.task["joint_motor_force"]
        self.motor_forces = ([float(p.getJointInfo(self.robot_id, joint, physicsClientId=self.client_id)[10])
                              for joint in self.robot.joint_ids] if force is None else [float(force)] * 6)
        if min(self.motor_forces) <= 0:
            raise ValueError("the robot URDF must supply positive per-joint effort limits")
        floor_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client_id)
        self.floor = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_shape,
                                       physicsClientId=self.client_id)
        self._configure_contact_filters()
        radius = self.task["obstacles"]["radius"]
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.client_id)
        visual = (p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[.15, .3, .9, 1],
                                     physicsClientId=self.client_id) if gui else -1)
        self.obstacle_ids = [p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual, basePosition=[5 + i, 5, 5], physicsClientId=self.client_id)
            for i in range(3)]
        self.goal_marker = None
        if gui:
            visual = p.createVisualShape(p.GEOM_SPHERE, radius=.015, rgbaColor=[.1, .9, .2, 1],
                                        physicsClientId=self.client_id)
            self.goal_marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
                                                 physicsClientId=self.client_id)
        self.signature = experiment_signature(config)

    def _configure_contact_filters(self):
        count = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        infos = [p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id) for i in range(count)]
        parent = {i: i for i in range(-1, count)}

        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i

        for i, info in enumerate(infos):
            if info[2] == p.JOINT_FIXED:
                parent[root(i)] = root(info[16])
        adjacent = {frozenset((root(i), root(info[16]))) for i, info in enumerate(infos)
                    if info[2] != p.JOINT_FIXED}
        for a in range(-1, count):
            for b in range(a + 1, count):
                if root(a) == root(b) or frozenset((root(a), root(b))) in adjacent:
                    p.setCollisionFilterPair(self.robot_id, self.robot_id, a, b, 0,
                                             physicsClientId=self.client_id)
            if root(a) == root(-1):
                p.setCollisionFilterPair(self.robot_id, self.floor, a, -1, 0,
                                         physicsClientId=self.client_id)

    def _set_q(self, q, qdot=None):
        velocities = np.zeros(6) if qdot is None else qdot
        for joint, value, velocity in zip(self.robot.joint_ids, q, velocities):
            p.resetJointState(self.robot_id, joint, float(value), float(velocity),
                              physicsClientId=self.client_id)

    def _servo(self, target):
        p.setJointMotorControlArray(self.robot_id, self.robot.joint_ids, p.POSITION_CONTROL,
            targetPositions=np.asarray(target).tolist(), targetVelocities=[0.] * 6,
            positionGains=[self.task["position_gain"]] * 6,
            velocityGains=[self.task["velocity_gain"]] * 6,
            forces=self.motor_forces, physicsClientId=self.client_id)

    def _contacts(self):
        obstacle, self_contact, floor = False, False, False
        for contact in p.getContactPoints(bodyA=self.robot_id, physicsClientId=self.client_id):
            if contact[8] > 0:
                continue
            obstacle |= contact[2] in self.obstacle_ids
            self_contact |= contact[2] == self.robot_id
            floor |= contact[2] == self.floor
        return bool(obstacle), bool(self_contact), bool(floor)

    def _capsules(self):
        capsules = self.robot.capsules()
        return (np.asarray([c.start for c in capsules]), np.asarray([c.end for c in capsules]),
                np.asarray([c.radius for c in capsules]), [c.name for c in capsules])

    def _move_obstacles(self, centers):
        for body, center in zip(self.obstacle_ids, centers):
            p.resetBasePositionAndOrientation(body, np.asarray(center).tolist(), [0, 0, 0, 1],
                                              physicsClientId=self.client_id)

    def _goal_solution(self, target, quaternion):
        """Refine Bullet's coarse IK before certifying a strict reachable goal.

        This runs only at reset. No IK action or Jacobian is provided to SAC.
        """
        goal_cfg = self.task["goal"]
        result = self.robot.check_pose_ik(target, quaternion,
            position_tolerance=.03, orientation_tolerance=.1,
            max_iterations=goal_cfg.get("ik_max_iterations", 200), residual_threshold=1e-8)
        if not result["reachable"]:
            return None
        old_q, old_qdot = self.robot.joint_state_exact()
        # Select a revolute-equivalent solution inside the same command range
        # available to the policy. A URDF-reachable goal alone is insufficient.
        q = self.initial_q + (result["q_ik"] - self.initial_q + np.pi) % (2 * np.pi) - np.pi
        lower, upper = self.control_lower, self.control_upper
        q = np.clip(q, lower, upper)
        try:
            for _ in range(goal_cfg.get("ik_refine_iterations", 12)):
                self._set_q(q)
                position, orientation = self.robot.end_effector_pose()
                distance, angle = pose_errors(position, orientation, target, quaternion)
                combined = combined_pose_error(reference_error_components(distance, angle), self.task["reward"])
                if (distance <= goal_cfg["ik_position_tolerance"]
                        and angle <= goal_cfg["ik_orientation_tolerance"]
                        and combined <= self.task["success"]["strict_combined_error_tolerance"]):
                    return q.copy()
                position_error, orientation_error = pose_error(target, quaternion, position, orientation)
                error = np.concatenate((position_error, orientation_error)).astype(np.float64)
                jacobian = self.robot.end_effector_jacobian().astype(np.float64)
                delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-8 * np.eye(6), error)
                delta *= min(1., .2 / max(np.max(np.abs(delta)), 1e-12))
                improved = False
                for scale in (1., .5, .25):
                    candidate = np.clip(q + scale * delta, lower, upper)
                    self._set_q(candidate)
                    d, a = pose_errors(*self.robot.end_effector_pose(), target, quaternion)
                    score = combined_pose_error(reference_error_components(d, a), self.task["reward"])
                    if score < combined:
                        q, improved = candidate, True
                        break
                if not improved:
                    break
            return None
        finally:
            self._set_q(old_q, old_qdot)

    def _sample_scene(self):
        self._move_obstacles([[5 + i, 5, 5] for i in range(3)])
        self.robot.reset_joints(self.rng)
        q_start, _ = self.robot.joint_state_exact()
        p.performCollisionDetection(physicsClientId=self.client_id)
        if any(self._contacts()):
            raise RuntimeError("configured initial joint pose is in collision")
        start_capsules = self._capsules()
        goal_cfg, obstacle_cfg = self.task["goal"], self.task["obstacles"]
        base = self.config["robot"]["base_position"]
        quaternion = np.asarray(p.getQuaternionFromEuler(goal_cfg["rpy"]))
        for _ in range(int(goal_cfg["max_attempts"])):
            target = sample_goal_position(self.rng, goal_cfg, base)
            goal_q = self._goal_solution(target, quaternion)
            if goal_q is None:
                continue
            self._set_q(goal_q)
            p.performCollisionDetection(physicsClientId=self.client_id)
            errors = reference_error_components(*pose_errors(*self.robot.end_effector_pose(), target, quaternion))
            target_valid = (not any(self._contacts()) and combined_pose_error(errors, self.task["reward"])
                            <= self.task["success"]["strict_combined_error_tolerance"])
            goal_capsules = self._capsules()
            self._set_q(q_start)
            if not target_valid:
                continue
            centers = []
            for _ in range(int(obstacle_cfg["max_attempts"])):
                center = sample_obstacle_center(self.rng, target, base, obstacle_cfg)
                radius, clearance = obstacle_cfg["radius"], obstacle_cfg["endpoint_clearance"]
                if center[2] < radius + clearance or np.linalg.norm(center - base) > 1.3:
                    continue
                if centers and np.min(np.linalg.norm(np.asarray(centers) - center, axis=1)) < 2 * radius + clearance:
                    continue
                if any(np.min(point_segment_distances([center], caps[0], caps[1]) - caps[2])
                       < radius + clearance for caps in (start_capsules, goal_capsules)):
                    continue
                centers.append(center)
                if len(centers) == 3:
                    return dict(start_q=q_start, goal_position=target, goal_quaternion=quaternion,
                                goal_q=goal_q, obstacle_centers=np.asarray(centers))
        raise RuntimeError("reference scene sampling exhausted its IK/endpoint-clearance budget")

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = {} if options is None else options
        scene = copy.deepcopy(options["scene"]) if "scene" in options else self._sample_scene()
        self._set_q(scene["start_q"])
        self.goal_position = np.asarray(scene["goal_position"], dtype=np.float64)
        self.goal_quaternion = np.asarray(scene["goal_quaternion"], dtype=np.float64)
        self.reference_position, self.reference_quaternion = self.goal_position.copy(), self.goal_quaternion.copy()
        self.obstacle_centers = np.asarray(scene["obstacle_centers"], dtype=np.float64)
        if self.obstacle_centers.shape != (3, 3) or not np.isfinite(self.obstacle_centers).all():
            raise ValueError("scene requires three finite obstacle centers")
        self._move_obstacles(self.obstacle_centers)
        self.scene = scene
        self.desired_q = np.asarray(scene["start_q"], dtype=np.float64).copy()
        self._servo(self.desired_q)
        self.step_count = 0
        self.collision_seen = self.joint_limit_seen = False
        self.return_sum = 0.
        self.pose_history = dict(report_reached_seen=False, strict_reached_seen=False,
            min_position_error_m=float("inf"), min_orientation_error_rad=float("inf"),
            min_combined_error=float("inf"),
            position_thresholds_seen=np.zeros(len(PAPER_POSITION_THRESHOLDS_MM), dtype=bool),
            pose_thresholds_seen=np.zeros(len(PAPER_POSITION_THRESHOLDS_MM), dtype=bool))
        self.initialized = True
        p.performCollisionDetection(physicsClientId=self.client_id)
        if any(self._contacts()):
            raise ValueError("scene starts in contact")
        if self.goal_marker is not None:
            p.resetBasePositionAndOrientation(self.goal_marker, self.goal_position, [0, 0, 0, 1],
                                              physicsClientId=self.client_id)
        obs, info = self._observe()
        self._update_pose_history(info)
        info["scene"] = copy.deepcopy(scene)
        return obs, info

    def _update_pose_history(self, info):
        tolerance, history = self.task["success"], self.pose_history
        position, orientation = info["position_error_m"], info["orientation_error_rad"]
        history["report_reached_seen"] |= (position <= tolerance["report_position_tolerance"]
                                           and orientation <= tolerance["report_orientation_tolerance"])
        history["strict_reached_seen"] |= info["final_combined_error"] <= tolerance["strict_combined_error_tolerance"]
        for name, value in (("min_position_error_m", position), ("min_orientation_error_rad", orientation),
                            ("min_combined_error", info["final_combined_error"])):
            history[name] = min(history[name], value)
        position_hits = position <= np.asarray(PAPER_POSITION_THRESHOLDS_MM) / 1000
        history["position_thresholds_seen"] |= position_hits
        history["pose_thresholds_seen"] |= position_hits & (orientation <= tolerance["report_orientation_tolerance"])

    def _observe(self):
        q, _ = self.robot.joint_state()
        position, quaternion = self.robot.end_effector_pose()
        starts, ends, radii, names = self._capsules()
        selected = [names.index(name) for name in self.task["distance_links"]]
        distances = point_segment_distances(self.obstacle_centers, starts[selected], ends[selected]).min(axis=1)
        obs = paper_observation(q, position, quaternion, self.reference_position, self.reference_quaternion, distances)
        position_error, orientation_error = pose_errors(position, quaternion, self.goal_position, self.goal_quaternion)
        final_error = combined_pose_error(reference_error_components(position_error, orientation_error), self.task["reward"])
        return obs, dict(position_error_m=position_error, orientation_error_rad=orientation_error,
            final_combined_error=final_error, ee_position=position.astype(float), ee_quaternion=quaternion.astype(float),
            capsule_clearance_m=float(np.min(point_segment_distances(self.obstacle_centers, starts, ends)
                                              - radii - self.task["obstacles"]["radius"])))

    def step(self, action):
        if not self.initialized:
            raise RuntimeError("reset the environment before stepping")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("action must contain six finite values")
        self.desired_q = np.clip(self.initial_q + np.clip(action, -1., 1.) * self.task["action_limit"],
                                self.robot.joint_lower_limits, self.robot.joint_upper_limits)
        self._servo(self.desired_q)
        contact_flags = np.zeros(3, dtype=bool)
        for _ in range(int(self.task["action_repeat"])):
            p.stepSimulation(physicsClientId=self.client_id)
            contact_flags |= self._contacts()
        collision = bool(contact_flags.any())
        self.collision_seen |= collision
        self.step_count += 1
        q_after, _ = self.robot.joint_state_exact()
        joint_limit = bool(np.any(q_after < self.robot.joint_lower_limits - .001)
                           or np.any(q_after > self.robot.joint_upper_limits + .001))
        self.joint_limit_seen |= joint_limit
        obs, info = self._observe()
        self._update_pose_history(info)
        reward, rewards = paper_reward(obs[20:22], obs[22:25], collision, self.task["reward"])
        self.return_sum += reward
        reached = info["final_combined_error"] <= self.task["success"]["combined_error_tolerance"]
        # Reference SAC ends on reaching or the time limit, rather than adding
        # an unpenalized joint-limit terminal state. Violations remain failures
        # in collision-free success, even if the arm subsequently recovers.
        terminated = bool(reached or (collision and self.task["terminate_on_collision"]))
        truncated = self.step_count >= self.task["horizon"] and not terminated
        safe = not self.collision_seen and not self.joint_limit_seen
        info.update(rewards, reached=bool(reached), safe_success=bool(reached and safe),
            report_success=bool(safe and self.pose_history["report_reached_seen"]),
            final_report_success=bool(safe and info["position_error_m"] <= self.task["success"]["report_position_tolerance"]
                                and info["orientation_error_rad"] <= self.task["success"]["report_orientation_tolerance"]),
            strict_success=bool(safe and self.pose_history["strict_reached_seen"]),
            **{key: copy.deepcopy(value) for key, value in self.pose_history.items()},
            collision=collision, collision_seen=self.collision_seen, joint_limit=joint_limit,
            joint_limit_seen=self.joint_limit_seen,
            obstacle_collision=bool(contact_flags[0]), self_collision=bool(contact_flags[1]),
            floor_collision=bool(contact_flags[2]), length=self.step_count, episode_return=self.return_sum)
        return obs, reward, float(collision), terminated, bool(truncated), info

    def set_reference(self, position, quaternion):
        self.reference_position = np.asarray(position, dtype=np.float64).copy()
        self.reference_quaternion = np.asarray(quaternion, dtype=np.float64).copy()
        return self._observe()[0]

    def episode_state_dict(self):
        if not self.initialized:
            raise RuntimeError("cannot snapshot an uninitialized environment")
        q, qdot = self.robot.joint_state_exact()
        return dict(protocol=PAPER_PROTOCOL, signature=self.signature, scene=copy.deepcopy(self.scene),
            q=q, qdot=qdot, desired_q=self.desired_q.copy(), step_count=self.step_count,
            collision_seen=self.collision_seen, joint_limit_seen=self.joint_limit_seen,
            return_sum=self.return_sum, pose_history=copy.deepcopy(self.pose_history),
            reference_position=self.reference_position.copy(),
            reference_quaternion=self.reference_quaternion.copy(), rng=copy.deepcopy(self.rng.bit_generator.state))

    def restore_episode_state(self, state):
        if state["protocol"] != PAPER_PROTOCOL or state["signature"] != self.signature:
            raise ValueError("environment snapshot belongs to a different paper task")
        self.reset(options={"scene": state["scene"]})
        self._set_q(state["q"], state["qdot"])
        for name in ("step_count", "collision_seen", "joint_limit_seen", "return_sum"):
            setattr(self, name, state[name])
        self.pose_history = copy.deepcopy(state["pose_history"])
        self.desired_q = np.asarray(state["desired_q"], dtype=np.float64).copy()
        self._servo(self.desired_q)
        self.rng.bit_generator.state = state["rng"]
        self.reference_position = np.asarray(state["reference_position"], dtype=np.float64).copy()
        self.reference_quaternion = np.asarray(state["reference_quaternion"], dtype=np.float64).copy()
        p.performCollisionDetection(physicsClientId=self.client_id)
        return self._observe()[0]

    def close(self):
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)
