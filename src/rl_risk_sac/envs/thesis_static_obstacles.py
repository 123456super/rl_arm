"""Static obstacle sampling, placement and PyBullet body lifecycle."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pybullet as p

from rl_risk_sac.envs.thesis_obstacles import (
    _horizontal_obb_surface_distance,
    _swept_obb_envelope,
)


class StaticObstacleMixin:
    def _disable_obstacle(self) -> None:
        self.obstacle_id = None
        self._obstacle_body_id = None
        self.obstacle_layout = "none"
        self.obstacle_positions = np.empty((0, 3), dtype=np.float32)
        self.obstacle_velocities = np.empty((0, 3), dtype=np.float32)
        self.obstacle_half_extents_all = np.empty((0, 3), dtype=np.float32)
        self.obstacle_quaternions = np.empty((0, 4), dtype=np.float32)
        self.dynamic_motion_centers = np.empty((0, 3), dtype=np.float32)
        self.dynamic_motion_directions = np.empty((0, 3), dtype=np.float32)
        self.dynamic_motion_amplitudes = np.empty(0, dtype=np.float32)
        self.dynamic_motion_angular_frequencies = np.empty(0, dtype=np.float32)
        self.dynamic_motion_phases = np.empty(0, dtype=np.float32)
        self.dynamic_motion_time = 0.0
        self.dynamic_history_substeps = 0
        self.dynamic_position_history = []
        self._dynamic_prediction_cache = {}
        self.obstacle_position.fill(0.0)
        self.obstacle_velocity.fill(0.0)
        for body_id in self._obstacle_body_ids:
            p.removeBody(body_id, physicsClientId=self.client_id)
        self._obstacle_body_ids = []

    def _create_obstacle(self) -> None:
        self._disable_obstacle()
        if self.contract.scene == "none":
            self.obstacle_sampling_proposals = 0
            self.obstacle_sampling_time_s = 0.0
            return
        sampling_started = perf_counter()
        self.obstacle_sampling_proposals = 0
        self.obstacle_sampling_time_s = 0.0
        obstacle_count = (
            int(self.rng.choice(
                self.static_obstacle_counts,
                p=self.static_obstacle_count_probabilities,
            ))
            if self.contract.scene == "static"
            else 1
        )
        layout = (
            str(self.rng.choice(
                self.static_obstacle_layouts,
                p=self.static_obstacle_layout_probabilities,
            ))
            if self.contract.scene == "static" and self.blocking_obstacle_mode
            else "legacy_blocking"
        )
        self.obstacle_layout = layout
        blocking_candidates: list[np.ndarray] = []
        blocking_candidates_by_state: list[list[np.ndarray]] = []
        if self.contract.scene == "static" and self.blocking_obstacle_mode:
            # Build witnesses on the straight joint-space path from reset to
            # the sampled goal.  An OBB centered on an intermediate capsule
            # blocks that nominal path while remaining clear of both endpoints.
            initial_q, initial_qdot = self.robot.joint_state_exact()
            path_states: list[list[Any]] = []
            for fraction in np.linspace(0.20, 0.80, 13):
                self._set_joint_state(
                    initial_q + float(fraction) * (self.goal_joint_positions - initial_q)
                )
                path_states.append(self.robot.capsules())
            self._set_joint_state(initial_q, initial_qdot)
            for capsules in path_states:
                state_candidates: list[np.ndarray] = []
                for capsule in capsules:
                    for candidate in (capsule.start, capsule.end,
                                      0.5 * (capsule.start + capsule.end)):
                        candidate = np.asarray(candidate, dtype=np.float32)
                        # The candidate is a point on this capsule's
                        # centreline and is also the OBB centre, so that
                        # capsule necessarily intersects the candidate OBB.
                        blocking_candidates.append(candidate)
                        state_candidates.append(candidate)
                blocking_candidates_by_state.append(state_candidates)
        initial_q, initial_qdot = self.robot.joint_state_exact()
        initial_capsules = self.robot.capsules()
        self._set_joint_state(self.goal_joint_positions)
        goal_capsules = self.robot.capsules()
        self._set_joint_state(initial_q, initial_qdot)
    
        # Dense five-obstacle layouts occasionally need several complete-set
        # retries even though every individual proposal is valid.  Retrying
        # the set is preferable to silently falling back to an easier layout.
        for _ in range(32):
            positions: list[np.ndarray] = []
            velocities: list[np.ndarray] = []
            half_extents_all: list[np.ndarray] = []
            quaternions: list[np.ndarray] = []
            motion_centers: list[np.ndarray] = []
            motion_directions: list[np.ndarray] = []
            motion_amplitudes: list[float] = []
            motion_angular_frequencies: list[float] = []
            motion_phases: list[float] = []
            legacy_candidates = [
                blocking_candidates[int(index)].copy()
                for index in self.rng.permutation(len(blocking_candidates))
            ]
            # A mixed scene contains one broad-workspace random obstacle and
            # keeps the remaining obstacles path-critical.  The random box is
            # sampled first so later blocking boxes can satisfy set clearance.
            structured_angle = float(self.rng.uniform(-np.pi, np.pi))
            structured_direction = np.asarray(
                [np.cos(structured_angle), np.sin(structured_angle), 0.0],
                dtype=np.float32,
            )
            structured_states = np.rint(np.linspace(
                0, max(0, len(blocking_candidates_by_state) - 1), obstacle_count
            )).astype(np.int64)
            valid_set = True
            for obstacle_index in range(obstacle_count):
                mixed_is_random = bool(
                    layout == "mixed"
                    and obstacle_count > 1
                    and obstacle_index == 0
                )
                half_extents = (
                    self.rng.uniform(
                        self.obstacle_half_extents_min,
                        self.obstacle_half_extents_max,
                    ).astype(np.float32)
                    if self.contract.scene == "static"
                    else self.dynamic_obstacle_half_extents.copy()
                )
                obstacle_yaw = float(self.rng.uniform(-np.pi, np.pi))
                quaternion = np.asarray(
                    p.getQuaternionFromEuler([0.0, 0.0, obstacle_yaw]),
                    dtype=np.float32,
                )
                accepted = False
                for _ in range(500):
                    self.obstacle_sampling_proposals += 1
                    if self.contract.scene == "static":
                        if layout == "legacy_blocking" and legacy_candidates:
                            position = legacy_candidates.pop()
                        elif layout == "path_offset":
                            position = np.asarray(
                                [
                                    self.rng.uniform(0.22, 0.72),
                                    self.rng.uniform(-0.48, 0.48),
                                    self.rng.uniform(0.18, 0.62),
                                ],
                                dtype=np.float32,
                            )
                        elif (
                            layout == "mixed"
                            and not mixed_is_random
                            and legacy_candidates
                        ):
                            position = legacy_candidates.pop()
                        elif (
                            layout == "structured"
                            and blocking_candidates_by_state
                        ):
                            state_index = int(np.clip(
                                structured_states[obstacle_index]
                                + int(self.rng.integers(-1, 2)),
                                0,
                                len(blocking_candidates_by_state) - 1,
                            ))
                            state_candidates = blocking_candidates_by_state[state_index]
                            position = state_candidates[int(self.rng.integers(
                                0, len(state_candidates)
                            ))].copy()
                            side = -1.0 if obstacle_index % 2 == 0 else 1.0
                            position += side * structured_direction * float(
                                self.rng.uniform(
                                    self.structured_offset_min,
                                    self.structured_offset_max,
                                )
                            )
                            position[2] += float(self.rng.uniform(-0.04, 0.04))
                        elif mixed_is_random:
                            # Sample the workspace-random obstacle from a broader
                            # reachable-workspace envelope.  Restricting it to
                            # the blocking corridor made dense five-box sets
                            # unnecessarily hard to construct.
                            position = np.asarray(
                                [
                                    self.rng.uniform(-0.70, 0.78),
                                    self.rng.uniform(-0.70, 0.70),
                                    self.rng.uniform(0.18, 0.78),
                                ],
                                dtype=np.float32,
                            )
                        else:
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
                        path_start = np.asarray(
                            [self.rng.uniform(0.22, 0.72), side * self.rng.uniform(0.42, 0.62), self.rng.uniform(0.18, 0.62)],
                            dtype=np.float32,
                        )
                        path_end = np.asarray(
                            [self.rng.uniform(0.22, 0.72), -side * self.rng.uniform(0.24, 0.48), self.rng.uniform(0.18, 0.62)],
                            dtype=np.float32,
                        )
                        path_delta = path_end - path_start
                        path_length = float(np.linalg.norm(path_delta))
                        if path_length <= 1.0e-6:
                            continue
                        motion_center = 0.5 * (path_start + path_end)
                        motion_direction = (path_delta / path_length).astype(
                            np.float32
                        )
                        motion_amplitude = 0.5 * path_length
                        motion_angular_frequency = (
                            self.obstacle_speed / motion_amplitude
                        )
                        motion_phase = float(self.rng.uniform(-np.pi, np.pi))
                        position = (
                            motion_center
                            + motion_direction
                            * motion_amplitude
                            * np.cos(motion_phase)
                        ).astype(np.float32)
                        velocity = (
                            -motion_direction
                            * motion_amplitude
                            * motion_angular_frequency
                            * np.sin(motion_phase)
                        ).astype(np.float32)
                        path_envelope_position, path_envelope_half_extents = (
                            _swept_obb_envelope(
                                np.stack([path_start, path_end]),
                                half_extents,
                                quaternion,
                            )
                        )
                    if (
                        self.mounting_plane_enabled
                        and (
                            position[2] - half_extents[2]
                            if self.contract.scene == "static"
                            else path_envelope_position[2]
                            - path_envelope_half_extents[2]
                        )
                        < self.mounting_plane_height + 0.01
                    ):
                        continue
                    # Reject overlapping obstacle sets before the much more
                    # expensive capsule--OBB endpoint/path queries. This is an
                    # exact OBB test, so it changes only computation order,
                    # never the accepted scene distribution.
                    separated = all(
                        _horizontal_obb_surface_distance(
                            position,
                            half_extents,
                            quaternion,
                            other_position,
                            other_half_extents,
                            other_quaternion,
                        ) >= self.obstacle_surface_clearance_min
                        for other_position, other_half_extents, other_quaternion in zip(
                            positions,
                            half_extents_all,
                            quaternions,
                            strict=True,
                        )
                    )
                    if not separated:
                        continue
                    query_position = (
                        position
                        if self.contract.scene == "static"
                        else path_envelope_position
                    )
                    query_half_extents = (
                        half_extents
                        if self.contract.scene == "static"
                        else path_envelope_half_extents
                    )
                    initial_distance = self._minimum_obstacle_clearance_for(
                        initial_capsules,
                        query_position,
                        query_half_extents,
                        quaternion,
                    )
                    if initial_distance < self.d_safe:
                        continue
                    goal_distance = self._minimum_obstacle_clearance_for(
                        goal_capsules,
                        query_position,
                        query_half_extents,
                        quaternion,
                    )
                    if goal_distance < self.d_safe:
                        continue
                    if (
                        layout == "path_offset"
                        and not self.path_offset_min <= min(
                            self._minimum_obstacle_clearance_for(
                                capsules, position, half_extents, quaternion
                            )
                            for capsules in path_states
                        ) <= self.path_offset_max
                    ):
                        continue
                    if initial_distance >= self.d_safe and goal_distance >= self.d_safe:
                        positions.append(position.copy())
                        velocities.append(velocity.copy())
                        half_extents_all.append(half_extents.copy())
                        quaternions.append(quaternion.copy())
                        if self.contract.scene == "dynamic":
                            motion_centers.append(motion_center.copy())
                            motion_directions.append(motion_direction.copy())
                            motion_amplitudes.append(motion_amplitude)
                            motion_angular_frequencies.append(
                                motion_angular_frequency
                            )
                            motion_phases.append(motion_phase)
                        accepted = True
                        break
                if not accepted:
                    valid_set = False
                    break
            if not valid_set:
                continue
            self.obstacle_positions = np.asarray(positions, dtype=np.float32)
            self.obstacle_velocities = np.asarray(velocities, dtype=np.float32)
            self.obstacle_half_extents_all = np.asarray(
                half_extents_all, dtype=np.float32
            )
            self.obstacle_quaternions = np.asarray(quaternions, dtype=np.float32)
            if self.contract.scene == "dynamic":
                self.dynamic_motion_centers = np.asarray(
                    motion_centers, dtype=np.float32
                )
                self.dynamic_motion_directions = np.asarray(
                    motion_directions, dtype=np.float32
                )
                self.dynamic_motion_amplitudes = np.asarray(
                    motion_amplitudes, dtype=np.float32
                )
                self.dynamic_motion_angular_frequencies = np.asarray(
                    motion_angular_frequencies, dtype=np.float32
                )
                self.dynamic_motion_phases = np.asarray(
                    motion_phases, dtype=np.float32
                )
                self.dynamic_motion_time = 0.0
                self.dynamic_history_substeps = 0
                history_dt = 1.0 / self.dynamic_history_sample_hz
                self.dynamic_position_history = []
                for center, direction, amplitude, frequency, phase in zip(
                    self.dynamic_motion_centers,
                    self.dynamic_motion_directions,
                    self.dynamic_motion_amplitudes,
                    self.dynamic_motion_angular_frequencies,
                    self.dynamic_motion_phases,
                    strict=True,
                ):
                    history: list[tuple[float, np.ndarray]] = []
                    for history_index in range(
                        self.dynamic_history_length - 1, -1, -1
                    ):
                        timestamp = -history_index * history_dt
                        angle = phase + frequency * timestamp
                        historical_position = (
                            center
                            + direction * amplitude * np.cos(angle)
                        ).astype(np.float32)
                        history.append((timestamp, historical_position))
                    self.dynamic_position_history.append(history)
            self._sync_primary_obstacle()
            self.obstacle_layout = layout
            self._create_obstacle_bodies()
            p.performCollisionDetection(physicsClientId=self.client_id)
            if not self._collision_events()["obstacle_collision"]:
                self.obstacle_sampling_time_s = perf_counter() - sampling_started
                return
            self._disable_obstacle()
        raise RuntimeError(
            f"failed to sample a valid set of {obstacle_count} obstacles "
            f"after {self.obstacle_sampling_proposals} proposals"
        )

    def _sync_primary_obstacle(self) -> None:
        if len(self.obstacle_positions) == 0:
            self.obstacle_position.fill(0.0)
            self.obstacle_velocity.fill(0.0)
            return
        self.obstacle_position = self.obstacle_positions[0].copy()
        self.obstacle_velocity = self.obstacle_velocities[0].copy()
        self.obstacle_half_extents = self.obstacle_half_extents_all[0].copy()
        self.obstacle_quaternion = self.obstacle_quaternions[0].copy()

    def _create_obstacle_bodies(self) -> None:
        for body_id in self._obstacle_body_ids:
            p.removeBody(body_id, physicsClientId=self.client_id)
        self._obstacle_body_ids = []
        for position, velocity, half_extents, quaternion in zip(
            self.obstacle_positions,
            self.obstacle_velocities,
            self.obstacle_half_extents_all,
            self.obstacle_quaternions,
            strict=True,
        ):
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents.tolist(),
                physicsClientId=self.client_id,
            )
            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision_shape,
                basePosition=position.tolist(),
                baseOrientation=quaternion.tolist(),
                physicsClientId=self.client_id,
            )
            self._obstacle_body_ids.append(body_id)
            p.resetBaseVelocity(
                body_id,
                linearVelocity=velocity.tolist(),
                angularVelocity=[0.0, 0.0, 0.0],
                physicsClientId=self.client_id,
            )
        self._obstacle_body_id = (
            self._obstacle_body_ids[0] if self._obstacle_body_ids else None
        )
        self.obstacle_id = self._obstacle_body_id

    def _create_obstacle_body(self) -> None:
        """Compatibility wrapper for legacy single-obstacle state restores."""
        if len(self.obstacle_positions) == 0:
            self.obstacle_positions = self.obstacle_position.reshape(1, 3).copy()
            self.obstacle_velocities = self.obstacle_velocity.reshape(1, 3).copy()
            self.obstacle_half_extents_all = (
                self.obstacle_half_extents.reshape(1, 3).copy()
            )
            self.obstacle_quaternions = self.obstacle_quaternion.reshape(1, 4).copy()
        self._sync_primary_obstacle()
        self._create_obstacle_bodies()

