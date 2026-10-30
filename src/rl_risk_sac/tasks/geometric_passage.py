"""One S0 inference, forward kinematics and a frozen obstacle support plane.

This module never advances physics or rolls a policy out to the final goal.
The 24D progress covector identifies a fixed material point on one link, so
the scalar progress is exactly its displacement through the frozen plane.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pybullet as p

from rl_risk_sac.envs.obstacle_scene import create_body, MovingObstacle
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.convex_sweep import ARM_GROUPS, ConvexSweep
from rl_risk_sac.tasks.local_avoidance import (
    GEOMETRIC_PROTOCOL, MeshClearance, arm_vector, passage_progress,
)


@dataclass
class PassageReference:
    status: str
    sample: dict | None = None
    nominal_steps: int = 0


@dataclass
class NominalIntent:
    action: np.ndarray
    entry_arm: np.ndarray
    velocity: np.ndarray
    predicted_arm: np.ndarray
    predicted_external: dict
    predicted_self: float
    within_limits: bool

    @property
    def next_clearance(self):
        return min(self.predicted_external.values(), default=float("inf"))


def scaled_reaching_command(base, action):
    command = np.clip(np.asarray(action, dtype=np.float64), -1., 1.) * base.action_scale
    if base.precision_control_enabled:
        z = max(base.previous_rho_position / base.contract.position_tolerance,
                base.previous_rho_orientation / base.contract.orientation_tolerance)
        if z <= 1.:
            scale = base.precision_min_action_scale + (base.precision_strict_action_scale
                    - base.precision_min_action_scale) * max(z, 0.)
        else:
            scale = base.precision_strict_action_scale + (1. - base.precision_strict_action_scale) * min(z - 1., 1.) ** 2
        command *= scale
    return command


def point_covector(group, fraction, axis):
    """Unit 24D weights; dot(v,w) = dot(material_point,axis)/normalizer."""
    weights = np.zeros((4, 2, 3), np.float64)
    weights[ARM_GROUPS.index(group)] = np.asarray([(1. - fraction) * axis, fraction * axis])
    normalizer = float(np.linalg.norm(weights))
    return weights.reshape(-1) / normalizer, normalizer


def support_plane(obstacle, axis, radius, release):
    rotation = np.asarray(p.getMatrixFromQuaternion(obstacle.quaternion)).reshape(3, 3)
    shape = ConvexSweep(obstacle.shape, obstacle.size, obstacle.initial_position, rotation)
    return float(axis @ shape.support(axis) + radius + release)


def make_geometric_reference(entry, obstacle, group, fraction, axis, radius, release):
    axis = np.asarray(axis, np.float64)
    axis = axis / np.linalg.norm(axis)
    direction, normalizer = point_covector(group, fraction, axis)
    endpoints = np.asarray(entry).reshape(4, 2, 3)[ARM_GROUPS.index(group)]
    point = (1. - fraction) * endpoints[0] + fraction * endpoints[1]
    plane = support_plane(obstacle, axis, radius, release)
    length = float((plane - axis @ point) / normalizer)
    return dict(reference_protocol=GEOMETRIC_PROTOCOL, entry_arm=np.asarray(entry).copy(),
                direction=direction, length=length,
                # Algebraic plane representative, never a requested robot pose.
                reference_exit_arm=np.asarray(entry) + length * direction,
                geometry=dict(obstacle_identity=obstacle.identity, group=group, fraction=float(fraction),
                              axis=axis.copy(), radius=float(radius), plane_offset=plane))


def validate_geometric_reference(sample, config):
    if sample.get("reference_protocol") != GEOMETRIC_PROTOCOL:
        raise ValueError("geometric task requires a geometric reference; regenerate the bank")
    geometry = sample["geometry"]
    group, fraction, axis = geometry["group"], geometry["fraction"], np.asarray(geometry["axis"])
    if (group not in ARM_GROUPS or not np.isfinite(fraction) or not 0 <= fraction <= 1
            or axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1.)):
        raise ValueError("invalid geometric material point/axis")
    specs = {spec["name"]: spec for spec in config["robot"]["capsules"]}
    radius = specs[group]["radius"]
    matches = [obj for obj in sample["state"]["obstacles"] if obj["identity"] == geometry["obstacle_identity"]]
    if len(matches) != 1 or not np.isclose(geometry["radius"], radius):
        raise ValueError("geometric blocker/radius differs from scene")
    expected = make_geometric_reference(sample["entry_arm"], MovingObstacle(**matches[0]), group,
                                        fraction, axis, radius, config["local_avoidance"]["release_distance_m"])
    if (not np.allclose(sample["direction"], expected["direction"], atol=1e-7)
            or not np.isfinite(sample["length"])
            or sample["length"] < config["local_avoidance"]["minimum_reference_motion_m"]
            or not np.isclose(sample["length"], expected["length"])
            or not np.isclose(geometry["plane_offset"], expected["geometry"]["plane_offset"])):
        raise ValueError("geometric progress does not match the obstacle support plane")


class GeometricPassageReference:
    def __init__(self, config, reaching):
        # Local import avoids a sampling/reference import cycle.
        from rl_risk_sac.tasks.local_samples import simulation_config
        self.config, self.local, self.reaching = config, config["local_avoidance"], reaching
        self.settings = self.local["geometry"]
        self.query = ThesisHomotopyEnv(simulation_config(config),
                                      compute_external_features=False, cache_geometry_queries=True)
        self._scene_key = None

    def _sync_scene(self, base):
        metadata = [obj.metadata() for obj in base.obstacles]
        key = json.dumps(metadata, sort_keys=True)
        if key != self._scene_key:
            self.query._disable_obstacle()
            for value in metadata:
                obj = MovingObstacle(**value)
                create_body(obj, self.query.client_id)
                self.query.obstacles.append(obj)
            self._scene_key = key

    def prepare(self, base, observation, *, action=None):
        self._sync_scene(base)
        action = (self.reaching.select_action(observation, deterministic=True) if action is None else
                  np.asarray(action, dtype=np.float32))
        command = scaled_reaching_command(base, action)
        q, qdot = base.robot.joint_state_exact()
        entry = arm_vector(base)
        self.query._set_joint_state(q + self.settings["kinematic_dt"] * command, qdot)
        velocity = (arm_vector(self.query) - entry) / self.settings["kinematic_dt"]
        predicted_q = q + self.settings["prediction_dt"] * command
        within_limits = bool(np.all(predicted_q >= base.robot.joint_lower_limits)
                             and np.all(predicted_q <= base.robot.joint_upper_limits))
        self.query._set_joint_state(predicted_q, qdot)
        with self.query._geometry_query_scope():
            margins = MeshClearance(self.query)
            distances = {obj.identity: self.query._static_clearance(obj) for obj in self.query.obstacles}
        return NominalIntent(action, entry, velocity, arm_vector(self.query), distances,
                             margins.self_distance, within_limits)

    def next_clearance(self, base, observation):
        return self.prepare(base, observation).next_clearance

    def build(self, base, *, intent=None, observation=None):
        if intent is None:
            if observation is None:
                observation = base._observe()[0]
            intent = self.prepare(base, observation)
        if base.obstacles and len(base._external_queries) != len(base.obstacles):
            raise ValueError("geometric guide requires a current multi-obstacle observation")
        groups = {capsule.name: capsule for capsule in base.robot.capsules()}
        origin, rotation = base._base_frame()
        candidates = []
        for index, obj in enumerate(base.obstacles):
            current = base._static_clearance(obj)
            if min(current, intent.predicted_external[obj.identity]) > self.local["trigger_distance_m"]:
                continue
            # Reuse the current observation's capsule queries; no GJK re-run.
            for group_index, query in enumerate(base._external_queries[index]):
                group = ARM_GROUPS[group_index]
                capsule = groups[group]
                if query.distance > self.local["trigger_distance_m"] + capsule.radius:
                    continue
                normal = rotation @ query.normal
                spine_point = rotation @ (query.robot_point + capsule.radius * query.normal)
                spine_point += origin
                edge = capsule.end - capsule.start
                fraction = float(np.clip((spine_point - capsule.start) @ edge / max(edge @ edge, 1e-12), 0., 1.))
                endpoints = intent.velocity.reshape(4, 2, 3)[group_index]
                velocity = (1. - fraction) * endpoints[0] + fraction * endpoints[1]
                speed = float(np.linalg.norm(velocity))
                if speed < self.settings["minimum_point_speed"]:
                    continue
                axis = velocity / speed
                closing = -float(axis @ normal)
                if closing < self.settings["minimum_closing_cosine"]:
                    continue
                sample = make_geometric_reference(intent.entry_arm, obj, group, fraction, axis,
                                                   capsule.radius, self.local["release_distance_m"])
                if sample["length"] < self.local["minimum_reference_motion_m"]:
                    continue
                key = (max(query.distance, 0.) / (speed * closing), query.distance, obj.identity, group_index)
                candidates.append((key, sample))
        if not candidates:
            return PassageReference("clear")
        _, sample = min(candidates, key=lambda row: row[0])
        return PassageReference("blocked", sample)

    def handoff(self, base, observation, sample):
        current = MeshClearance(base)
        safe = self.config["thesis"]["obstacles"]["risk"]["d_safe"]
        if current.external < self.local["release_distance_m"] or current.self_distance < base.d_self_safe:
            return False
        intent = self.prepare(base, observation)
        # Endpoint kinematics is a local heuristic, not a dynamic safety proof.
        return bool(intent.within_limits and intent.next_clearance >= safe
                    and intent.predicted_self >= base.d_self_safe
                    and passage_progress(intent.predicted_arm, sample)
                        >= passage_progress(intent.entry_arm, sample) - 1e-6)

    def close(self):
        self.query.close()
