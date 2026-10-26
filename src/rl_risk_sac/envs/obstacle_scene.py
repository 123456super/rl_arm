"""Original collision bodies with fixed orientation and straight-line motion."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pybullet as p


@dataclass
class MovingObstacle:
    identity: int
    shape: str
    size: list
    size_class: str
    initial_position: list
    quaternion: list
    direction: list
    speed_mode: str
    speed: float
    period: float
    phase: float
    scalar_low: float
    scalar_high: float
    body_id: int = -1

    @property
    def bound_radius(self):
        return bounding_radius(self.shape, self.size)

    def position(self, time):
        # All modes use one fixed line; wall reflections only reverse that line.
        t = max(0., float(time))
        if self.speed_mode == "constant":
            distance = self.speed * t
        elif self.speed_mode in {"accelerate", "decelerate"}:
            cycles, local = divmod(t, self.period)
            integral = 0.25 * local + 0.75 * local * local / self.period
            if self.speed_mode == "decelerate":
                integral = 1.75 * local - 0.75 * local * local / self.period
            distance = self.speed * (cycles * self.period + integral)
        elif self.speed_mode == "stop":
            cycles, local = divmod(t, self.period)
            distance = self.speed * (cycles * self.period * 0.5 + min(local, self.period * 0.5))
        else:
            omega = 2 * np.pi / self.period
            distance = self.speed / omega * (np.sin(omega * t + self.phase) - np.sin(self.phase))
        width = self.scalar_high - self.scalar_low
        folded = (distance - self.scalar_low) % (2 * width)
        scalar = self.scalar_low + (folded if folded <= width else 2 * width - folded)
        return np.asarray(self.initial_position) + scalar * np.asarray(self.direction)

    def metadata(self):
        state = asdict(self)
        state.pop("body_id")
        return state


def bounding_radius(shape, size):
    if shape == "sphere":
        return float(size[0])
    if shape == "box":
        return float(np.linalg.norm(size))
    return float(np.hypot(size[0], size[1] / 2))


def sample_shape(rng):
    shape = str(rng.choice(["sphere", "cylinder", "box"]))
    # The proposed independent shape/size marginals are not feasible: boxes
    # and cylinders cannot be small, spheres cannot be large. This conditional
    # distribution preserves both the equal shape marginal and 30/50/20 sizes.
    probabilities = [0.9, 0.1, 0.] if shape == "sphere" else [0., 0.7, 0.3]
    size_class = str(rng.choice(["small", "medium", "large"], p=probabilities))
    low, high = {"small": (.04, .06), "medium": (.06, .09), "large": (.09, .14)}[size_class]
    for _ in range(10000):
        size = ([rng.uniform(.04, .09)] if shape == "sphere" else
                [rng.uniform(.04, .08), rng.uniform(.10, .22)] if shape == "cylinder" else
                rng.uniform(.04, .08, size=3).tolist())
        if low <= bounding_radius(shape, size) <= high:
            return shape, size, size_class
    raise RuntimeError("failed to sample a feasible obstacle size")


def sample_obstacle(rng, identity, low, high, *, center=None,
                    motion_modes=None, speed_range=(.04, .22)):
    shape, size, size_class = sample_shape(rng)
    radius = bounding_radius(shape, size)
    low, high = np.asarray(low) + radius, np.asarray(high) - radius
    if np.any(low >= high):
        raise ValueError("obstacle workspace too small for configured shape bounds")
    position = rng.uniform(low, high) if center is None else np.clip(center, low, high)
    quaternion = p.getQuaternionFromEuler(rng.uniform(-np.pi / 6, np.pi / 6, 3))
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    nonzero = np.abs(direction) > 1e-10
    endpoints = np.stack(((low[nonzero] - position[nonzero]) / direction[nonzero],
                          (high[nonzero] - position[nonzero]) / direction[nonzero]))
    scalar_low = float(endpoints.min(axis=0).max())
    scalar_high = float(endpoints.max(axis=0).min())
    modes = motion_modes or ["constant", "accelerate", "decelerate", "stop", "reverse"]
    if len(modes) == 0 or speed_range[0] <= 0 or speed_range[1] < speed_range[0]:
        raise ValueError("invalid obstacle motion curriculum")
    return MovingObstacle(identity, shape, size, size_class, position.tolist(), list(quaternion),
                          direction.tolist(), str(rng.choice(modes)),
                          float(rng.uniform(*speed_range)), float(rng.uniform(.8, 3.0)),
                          float(rng.uniform(0, 2 * np.pi)), scalar_low, scalar_high)


def create_body(obstacle, client):
    if obstacle.shape == "sphere":
        kwargs = dict(shapeType=p.GEOM_SPHERE, radius=obstacle.size[0])
    elif obstacle.shape == "box":
        kwargs = dict(shapeType=p.GEOM_BOX, halfExtents=obstacle.size)
    else:
        kwargs = dict(shapeType=p.GEOM_CYLINDER, radius=obstacle.size[0], height=obstacle.size[1])
    collision = p.createCollisionShape(**kwargs, physicsClientId=client)
    visual_kwargs = dict(kwargs)
    if obstacle.shape == "cylinder":
        visual_kwargs["length"] = visual_kwargs.pop("height")
    visual = p.createVisualShape(**visual_kwargs, rgbaColor=[.8, .25, .15, 1.], physicsClientId=client)
    obstacle.body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                        baseVisualShapeIndex=visual, basePosition=obstacle.initial_position,
                                        baseOrientation=obstacle.quaternion,
                                        baseInertialFramePosition=[0, 0, 0],
                                        baseInertialFrameOrientation=[0, 0, 0, 1], physicsClientId=client)
    return obstacle.body_id


def raw_record(obstacle, time, client, base_position, base_rotation):
    # Never read getBaseVelocity or the motion generator in the predictor path.
    position, quaternion = p.getBasePositionAndOrientation(obstacle.body_id, physicsClientId=client)
    rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    return dict(id=obstacle.identity, timestamp=float(time), valid=True,
                position=base_rotation.T @ (np.asarray(position) - base_position),
                rotation=base_rotation.T @ rotation, shape=obstacle.shape, size=obstacle.size)
