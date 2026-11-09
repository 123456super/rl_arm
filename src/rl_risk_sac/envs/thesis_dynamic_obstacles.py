"""Dynamic obstacle tracking, LSTM prediction and swept-envelope updates."""

from __future__ import annotations

import numpy as np
import pybullet as p

from rl_risk_sac.envs.thesis_obstacles import _swept_obb_envelope


class DynamicObstacleMixin:
    def _advance_obstacle(self) -> None:
        if self.obstacle_id is None or self.contract.scene != "dynamic":
            return
        if len(self.dynamic_motion_centers) != len(self.obstacle_positions):
            raise RuntimeError("dynamic obstacle motion state is incomplete")
        self.dynamic_motion_time += self.physics_dt
        angles = (
            self.dynamic_motion_phases
            + self.dynamic_motion_angular_frequencies * self.dynamic_motion_time
        )
        offsets = (
            self.dynamic_motion_directions
            * self.dynamic_motion_amplitudes[:, None]
            * np.cos(angles)[:, None]
        )
        velocities = (
            -self.dynamic_motion_directions
            * (
                self.dynamic_motion_amplitudes
                * self.dynamic_motion_angular_frequencies
                * np.sin(angles)
            )[:, None]
        )
        self.obstacle_positions = (
            self.dynamic_motion_centers + offsets
        ).astype(np.float32)
        self.obstacle_velocities = velocities.astype(np.float32)
        self._sync_primary_obstacle()
        for body_id, position, velocity, quaternion in zip(
            self._obstacle_body_ids,
            self.obstacle_positions,
            self.obstacle_velocities,
            self.obstacle_quaternions,
            strict=True,
        ):
            p.resetBasePositionAndOrientation(
                body_id,
                position.tolist(),
                quaternion.tolist(),
                physicsClientId=self.client_id,
            )
            p.resetBaseVelocity(
                body_id,
                linearVelocity=velocity.tolist(),
                angularVelocity=[0.0, 0.0, 0.0],
                physicsClientId=self.client_id,
            )
        self.dynamic_history_substeps += 1
        if (
            self.dynamic_history_substeps
            % self.dynamic_history_interval_substeps
            == 0
        ):
            self._record_dynamic_history()

    def _record_dynamic_history(self) -> None:
        """Store one timestamped position per dynamic obstacle, without duplicates."""
        if self.contract.scene != "dynamic" or not len(self.obstacle_positions):
            return
        if len(self.dynamic_position_history) != len(self.obstacle_positions):
            self.dynamic_position_history = [
                [] for _ in range(len(self.obstacle_positions))
            ]
        timestamp = float(self.dynamic_motion_time)
        for history, position in zip(
            self.dynamic_position_history,
            self.obstacle_positions,
            strict=True,
        ):
            if history and np.isclose(history[-1][0], timestamp):
                history[-1] = (timestamp, position.copy())
            else:
                history.append((timestamp, position.copy()))
                # Two extra raw samples guarantee that inserting an exact
                # policy-decision timestamp does not shorten the 30 Hz span
                # needed by the LSTM resampler.
                del history[:-(self.dynamic_history_length + 2)]

    def _tracked_obstacle_velocity(self, obstacle_index: int) -> np.ndarray:
        """Estimate velocity causally from the two latest recorded positions."""
        if obstacle_index >= len(self.dynamic_position_history):
            return np.zeros(3, dtype=np.float32)
        history = self.dynamic_position_history[obstacle_index]
        if len(history) < 2:
            return np.zeros(3, dtype=np.float32)
        earlier_time, earlier_position = history[-2]
        later_time, later_position = history[-1]
        elapsed = float(later_time - earlier_time)
        if elapsed <= 0.0:
            return np.zeros(3, dtype=np.float32)
        return (
            (np.asarray(later_position) - np.asarray(earlier_position)) / elapsed
        ).astype(np.float32)

    def _predicted_obstacle_positions(self, obstacle_index: int) -> np.ndarray:
        """Predict current plus three future centres for one logical obstacle."""
        current = self.obstacle_positions[obstacle_index]
        if not self.dynamic_prediction_enabled:
            return current.reshape(1, 3).copy()
        if self.dynamic_predictor is None:
            raise RuntimeError("LSTM obstacle predictor is not initialized")
        history = self.dynamic_position_history[obstacle_index]
        timestamps = np.asarray([item[0] for item in history], dtype=np.float32)
        positions = np.asarray([item[1] for item in history], dtype=np.float32)
        history_timestamp = float(timestamps[-1])
        cached = self._dynamic_prediction_cache.get(obstacle_index)
        if cached is None or cached[0] != history_timestamp:
            future = self.dynamic_predictor.predict(timestamps, positions)
            self._dynamic_prediction_cache[obstacle_index] = (
                history_timestamp, future.copy()
            )
        else:
            future = cached[1]
        return np.concatenate([current[None, :], future], axis=0).astype(
            np.float32
        )

    def _predicted_obstacle_envelope(
        self, obstacle_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = self._predicted_obstacle_positions(obstacle_index)
        center, half_extents = _swept_obb_envelope(
            positions,
            self.obstacle_half_extents_all[obstacle_index],
            self.obstacle_quaternions[obstacle_index],
        )
        half_extents = (
            half_extents + self.dynamic_prediction_margin
        ).astype(np.float32)
        return center, half_extents, positions

