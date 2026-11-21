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


class ObstacleSamplingError(RuntimeError):
    """Raised when no valid obstacle set exists for the sampled goal."""


_PHYSICAL_STATIC_LAYOUTS = {
    "grounded_path",
    "grounded_random",
    "suspended_path",
    "suspended_random",
}


def _sample_physical_layout_plan(
    obstacle_count: int,
    *,
    challenging_probability: float,
    grounded_probability: float,
    extra_path_probability: float,
    max_path_obstacles: int,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """Sample independent support and placement labels for one obstacle set.

    A challenging episode always starts with one path-relevant object. Extra
    objects may add at most one further path constraint under the formal
    configuration; all remaining objects provide random scene context.
    """
    if obstacle_count < 0:
        raise ValueError("obstacle_count must be non-negative")
    if obstacle_count == 0:
        return ()
    path_flags = np.zeros(obstacle_count, dtype=bool)
    if rng.random() < challenging_probability:
        path_flags[0] = True
        path_count = 1
        for index in range(1, obstacle_count):
            if (
                path_count < max_path_obstacles
                and rng.random() < extra_path_probability
            ):
                path_flags[index] = True
                path_count += 1
    grounded_flags = rng.random(obstacle_count) < grounded_probability
    return tuple(
        f"{'grounded' if grounded else 'suspended'}_"
        f"{'path' if path_relevant else 'random'}"
        for grounded, path_relevant in zip(
            grounded_flags, path_flags, strict=True
        )
    )


def _obb_vertical_half_extent(
    half_extents: np.ndarray, quaternion: np.ndarray
) -> float:
    """Return the world-Z half extent of an arbitrarily oriented OBB."""
    rotation = np.asarray(
        p.getMatrixFromQuaternion(np.asarray(quaternion, dtype=np.float64)),
        dtype=np.float64,
    ).reshape(3, 3)
    return float(
        np.abs(rotation[2]) @ np.asarray(half_extents, dtype=np.float64)
    )


class StaticObstacleMixin:
    _COUNTERFACTUAL_VARIANTS = (
        "low_wide", "tall_narrow", "offset_left", "offset_right"
    )

    def configure_evaluation_low_overpass(
        self,
        *,
        top_height_m: float = 0.375,
        path_half_depth_m: float = 0.075,
        cross_path_half_width_m: float = 0.28,
        plane_gap_m: float = 0.012,
    ) -> None:
        """Enable a single low wall for deterministic visualization only.

        The wall is centred on the nominal Base TCP path in the horizontal
        plane, aligned with that path, and widened across it.  Training never
        calls this method, so the sampled training distribution is unchanged.
        """
        values = np.asarray(
            [
                top_height_m,
                path_half_depth_m,
                cross_path_half_width_m,
                plane_gap_m,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("low-overpass dimensions must be finite and positive")
        bottom_height = float(self.mounting_plane_height + plane_gap_m)
        if top_height_m <= bottom_height:
            raise ValueError("low-overpass top must be above its plane clearance")
        self._evaluation_low_overpass = {
            "top_height_m": float(top_height_m),
            "bottom_height_m": bottom_height,
            "path_half_depth_m": float(path_half_depth_m),
            "cross_path_half_width_m": float(cross_path_half_width_m),
        }

    def _disable_obstacle(self) -> None:
        self.obstacle_id = None
        self._obstacle_body_id = None
        self.obstacle_layout = "none"
        self.obstacle_layouts: tuple[str, ...] = ()
        self.obstacle_positions = np.empty((0, 3), dtype=np.float32)
        self.obstacle_velocities = np.empty((0, 3), dtype=np.float32)
        self.obstacle_half_extents_all = np.empty((0, 3), dtype=np.float32)
        self.obstacle_quaternions = np.empty((0, 4), dtype=np.float32)
        self.obstacle_dynamic_mask = np.empty(0, dtype=bool)
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
        if self.contract.scene == "static":
            static_obstacle_count = (
                self._requested_static_obstacle_count
                if self._requested_static_obstacle_count is not None
                else int(self.rng.choice(
                    self.static_obstacle_counts,
                    p=self.static_obstacle_count_probabilities,
                ))
            )
            obstacle_count = static_obstacle_count
        elif self.contract.scene == "dynamic":
            static_obstacle_count = 0
            obstacle_count = 1
        elif self.contract.scene == "hybrid":
            static_obstacle_count = self.hybrid_static_obstacle_count
            obstacle_count = 1 + static_obstacle_count
        else:
            raise ValueError(f"unsupported obstacle scene {self.contract.scene!r}")
        low_overpass = getattr(self, "_evaluation_low_overpass", None)
        counterfactual_variant = getattr(
            self, "_evaluation_counterfactual_variant", None
        )
        if counterfactual_variant is not None and (
            self.contract.scene != "static" or obstacle_count != 1
        ):
            raise ValueError("counterfactual evaluation requires one static obstacle")
        if low_overpass is not None:
            if self.contract.scene != "static" or obstacle_count != 1:
                raise ValueError(
                    "evaluation low_overpass requires one static obstacle"
                )
            if not self.blocking_obstacle_mode:
                raise ValueError("evaluation low_overpass requires blocking mode")
            layout = "low_overpass"
        elif (
            self.contract.scene in {"static", "hybrid"}
            and self.blocking_obstacle_mode
        ):
            layout = self._requested_obstacle_layout or str(self.rng.choice(
                self.static_obstacle_layouts,
                p=self.static_obstacle_layout_probabilities,
            ))
        else:
            layout = "legacy_blocking"
        self.obstacle_layout = layout
        physical_layout_plan = (
            self._requested_physical_layout_plan
            or _sample_physical_layout_plan(
                    static_obstacle_count,
                    challenging_probability=(
                        self.physical_challenging_episode_probability
                    ),
                    grounded_probability=self.physical_grounded_probability,
                    extra_path_probability=self.physical_extra_path_probability,
                    max_path_obstacles=self.physical_max_path_obstacles,
                    rng=self.rng,
                )
            if layout == "physical_mixed"
            else ()
        )
        blocking_candidates: list[np.ndarray] = []
        secondary_link_candidates: list[np.ndarray] = []
        blocking_candidates_by_state: list[list[np.ndarray]] = []
        low_overpass_candidates: list[tuple[np.ndarray, float]] = []
        if (
            self.contract.scene in {"static", "hybrid"}
            and self.blocking_obstacle_mode
        ):
            # The target pose has already been converted to an IK-valid goal
            # configuration.  Interpolate from reset to that configuration,
            # use FK to trace the tool centre, and place path-critical boxes
            # only on this nominal end-effector trajectory.  Link capsules are
            # retained separately for clearance checks and offset layouts.
            initial_q, initial_qdot = self.robot.joint_state_exact()
            path_states: list[list[Any]] = []
            for fraction in np.linspace(0.20, 0.80, 13):
                self._set_joint_state(
                    initial_q + float(fraction) * (self.goal_joint_positions - initial_q)
                )
                path_states.append(self.robot.capsules())
                end_effector_position, _ = self.robot.end_effector_pose()
                candidate = np.asarray(
                    end_effector_position, dtype=np.float32
                ).copy()
                blocking_candidates.append(candidate)
                blocking_candidates_by_state.append([candidate])
                for capsule in path_states[-1]:
                    secondary_link_candidates.extend((
                        np.asarray(capsule.start, dtype=np.float32).copy(),
                        np.asarray(capsule.end, dtype=np.float32).copy(),
                        np.asarray(
                            0.5 * (capsule.start + capsule.end),
                            dtype=np.float32,
                        ).copy(),
                    ))
            self._set_joint_state(initial_q, initial_qdot)
            if low_overpass is not None:
                top_height = float(low_overpass["top_height_m"])
                start = np.asarray(
                    self.episode_start_ee_position, dtype=np.float64
                )
                goal = np.asarray(self.goal_position, dtype=np.float64)
                tangent = goal[:2] - start[:2]
                if float(np.linalg.norm(tangent)) <= 1.0e-8:
                    tangent = np.asarray([1.0, 0.0], dtype=np.float64)
                yaw = float(np.arctan2(tangent[1], tangent[0]))
                # The wall is centred on the actual straight start--goal
                # route, not on a joint-interpolation TCP trace.  This makes
                # its finite width symmetric about the route and prevents an
                # accidentally easy side exit caused by an offset target.
                # Reverse the preference because candidates are consumed via
                # pop() below: try the midpoint first, then nearby anchors.
                for fraction in (0.70, 0.30, 0.60, 0.40, 0.50):
                    position = np.asarray(
                        [
                            start[0] + fraction * (goal[0] - start[0]),
                            start[1] + fraction * (goal[1] - start[1]),
                            0.5 * (
                                float(low_overpass["bottom_height_m"])
                                + top_height
                            ),
                        ],
                        dtype=np.float32,
                    )
                    low_overpass_candidates.append((position, yaw))
        initial_q, initial_qdot = self.robot.joint_state_exact()
        initial_capsules = self.robot.capsules()
        self._set_joint_state(self.goal_joint_positions)
        goal_capsules = self.robot.capsules()
        self._set_joint_state(initial_q, initial_qdot)

        if counterfactual_variant is not None:
            self._create_counterfactual_obstacle(
                counterfactual_variant,
                blocking_candidates,
                initial_capsules,
                goal_capsules,
                sampling_started,
            )
            return
    
        # Dense multi-obstacle layouts occasionally need several complete-set
        # retries even though every individual proposal is valid.
        # Physical single-obstacle layouts already contain a large, shuffled
        # set of swept-link anchors. Rebuilding that same set 32 times only
        # repeats expensive capsule queries; reset() can instead sample a new
        # feasible goal when the current geometry admits no valid object.
        set_attempts = (
            (4 if static_obstacle_count > 1 else 1)
            if layout == "physical_mixed"
            else (1 if layout in _PHYSICAL_STATIC_LAYOUTS else 32)
        )
        for _ in range(set_attempts):
            positions: list[np.ndarray] = []
            velocities: list[np.ndarray] = []
            half_extents_all: list[np.ndarray] = []
            quaternions: list[np.ndarray] = []
            motion_centers: list[np.ndarray] = []
            motion_directions: list[np.ndarray] = []
            motion_amplitudes: list[float] = []
            motion_angular_frequencies: list[float] = []
            motion_phases: list[float] = []
            dynamic_mask: list[bool] = []
            obstacle_layouts: list[str] = []
            sampling_positions: list[np.ndarray] = []
            sampling_half_extents: list[np.ndarray] = []
            legacy_candidates = [
                blocking_candidates[int(index)].copy()
                for index in self.rng.permutation(len(blocking_candidates))
            ]
            secondary_candidates = [
                secondary_link_candidates[int(index)].copy()
                for index in self.rng.permutation(
                    len(secondary_link_candidates)
                )
            ]
            tcp_blocker_placed = False
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
                is_dynamic = bool(
                    self.contract.scene == "dynamic"
                    or (
                        self.contract.scene == "hybrid"
                        and obstacle_index == 0
                    )
                )
                mixed_is_random = bool(
                    layout == "mixed"
                    and static_obstacle_count > 1
                    and obstacle_index == (1 if self.contract.scene == "hybrid" else 0)
                )
                static_index = (
                    obstacle_index - 1
                    if self.contract.scene == "hybrid"
                    else obstacle_index
                )
                obstacle_layout = (
                    physical_layout_plan[static_index]
                    if layout == "physical_mixed" and not is_dynamic
                    else ("dynamic" if is_dynamic else layout)
                )
                half_extents = (
                    self.dynamic_obstacle_half_extents.copy()
                    if is_dynamic
                    else
                    self.rng.uniform(
                        self.obstacle_half_extents_min,
                        self.obstacle_half_extents_max,
                    ).astype(np.float32)
                )
                if low_overpass is not None and not is_dynamic:
                    half_extents = np.asarray(
                        [
                            low_overpass["path_half_depth_m"],
                            low_overpass["cross_path_half_width_m"],
                            0.5 * (
                                low_overpass["top_height_m"]
                                - low_overpass["bottom_height_m"]
                            ),
                        ],
                        dtype=np.float32,
                    )
                physical_layout = bool(
                    not is_dynamic
                    and obstacle_layout in _PHYSICAL_STATIC_LAYOUTS
                )
                grounded_layout = bool(
                    physical_layout and obstacle_layout.startswith("grounded_")
                )
                path_relevant_layout = bool(
                    physical_layout and obstacle_layout.endswith("_path")
                )
                if physical_layout and not grounded_layout:
                    obstacle_euler = self.rng.uniform(-np.pi, np.pi, size=3)
                else:
                    obstacle_euler = np.asarray(
                        [0.0, 0.0, self.rng.uniform(-np.pi, np.pi)]
                    )
                obstacle_yaw = float(obstacle_euler[2])
                quaternion = np.asarray(
                    p.getQuaternionFromEuler(obstacle_euler.tolist()),
                    dtype=np.float32,
                )
                vertical_half_extent = (
                    _obb_vertical_half_extent(half_extents, quaternion)
                    if physical_layout else 0.0
                )
                if grounded_layout and path_relevant_layout:
                    # Only swept-link points low enough to intersect this
                    # ground-supported box can possibly satisfy the path test.
                    # Filtering them before capsule--OBB queries removes many
                    # guaranteed rejections without changing accepted scenes.
                    candidate_height_limit = (
                        self.mounting_plane_height
                        + 2.0 * vertical_half_extent
                        + 0.11
                    )
                    secondary_candidates = [
                        candidate for candidate in secondary_candidates
                        if float(candidate[2]) <= candidate_height_limit
                    ]
                accepted = False
                proposal_attempts = 500
                for _ in range(proposal_attempts):
                    self.obstacle_sampling_proposals += 1
                    candidate_is_tcp_blocker = False
                    if low_overpass is not None and not is_dynamic:
                        if not low_overpass_candidates:
                            break
                        position, obstacle_yaw = low_overpass_candidates.pop()
                        quaternion = np.asarray(
                            p.getQuaternionFromEuler([0.0, 0.0, obstacle_yaw]),
                            dtype=np.float32,
                        )
                        candidate_is_tcp_blocker = True
                        velocity = np.zeros(3, dtype=np.float32)
                        if np.linalg.norm(position - self.goal_position) < 0.15:
                            continue
                    elif not is_dynamic:
                        requires_path_candidate = bool(
                            self.blocking_obstacle_mode
                            and (
                                layout == "legacy_blocking"
                                or (layout == "mixed" and not mixed_is_random)
                                or path_relevant_layout
                            )
                        )
                        primary_path_candidates = (
                            secondary_candidates
                            if grounded_layout
                            else legacy_candidates
                        )
                        if (
                            requires_path_candidate
                            and not tcp_blocker_placed
                            and primary_path_candidates
                        ):
                            # Ground-supported boxes cannot be translated to a
                            # high TCP waypoint without ceasing to be grounded.
                            # Anchor their XY coordinates to the swept robot
                            # links instead. Suspended boxes use the TCP trace.
                            position = primary_path_candidates.pop()
                            candidate_is_tcp_blocker = True
                        elif (
                            requires_path_candidate
                            and not tcp_blocker_placed
                        ):
                            # The mandatory TCP blocker could not satisfy the
                            # endpoint/plane constraints for this box sample.
                            break
                        elif requires_path_candidate and secondary_candidates:
                            # Once the direct TCP route is blocked, additional
                            # boxes may occupy other link swept volumes. This
                            # keeps dense 2--5 obstacle scenes geometrically
                            # feasible without weakening the primary blocker.
                            position = secondary_candidates.pop()
                        elif requires_path_candidate:
                            # The direct route is already blocked. If all
                            # secondary link-sweep candidates conflict with the
                            # existing set, use the broad reachable workspace
                            # for the remaining context obstacles only.
                            position = np.asarray(
                                [
                                    self.rng.uniform(-0.70, 0.78),
                                    self.rng.uniform(-0.70, 0.70),
                                    self.rng.uniform(0.18, 0.78),
                                ],
                                dtype=np.float32,
                            )
                        elif (
                            layout == "workspace_random"
                            or (physical_layout and not path_relevant_layout)
                        ):
                            # Broad workspace sampling for non-path-critical
                            # scenes. The common endpoint capsule checks below
                            # guarantee clearance from both the initial and
                            # target robot configurations.
                            position = np.asarray(
                                [
                                    self.rng.uniform(-0.70, 0.78),
                                    self.rng.uniform(-0.70, 0.70),
                                    self.rng.uniform(0.18, 0.78),
                                ],
                                dtype=np.float32,
                            )
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
                            layout == "structured"
                            and blocking_candidates_by_state
                        ):
                            static_index = (
                                obstacle_index - 1
                                if self.contract.scene == "hybrid"
                                else obstacle_index
                            )
                            state_index = int(np.clip(
                                structured_states[static_index]
                                + int(self.rng.integers(-1, 2)),
                                0,
                                len(blocking_candidates_by_state) - 1,
                            ))
                            state_candidates = blocking_candidates_by_state[state_index]
                            position = state_candidates[int(self.rng.integers(
                                0, len(state_candidates)
                            ))].copy()
                            side = -1.0 if static_index % 2 == 0 else 1.0
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
                        if physical_layout:
                            if grounded_layout:
                                # A supported obstacle rests exactly on the
                                # mounting plane. The validation below exempts
                                # only this intentional obstacle--plane contact;
                                # robot--plane collision handling is unchanged.
                                position[2] = (
                                    self.mounting_plane_height
                                    + vertical_half_extent
                                )
                            elif not path_relevant_layout:
                                min_center_z = (
                                    self.mounting_plane_height
                                    + self.physical_suspended_bottom_min
                                    + vertical_half_extent
                                )
                                max_center_z = (
                                    self.physical_obstacle_top_max
                                    - vertical_half_extent
                                )
                                if min_center_z >= max_center_z:
                                    continue
                                position[2] = float(
                                    self.rng.uniform(min_center_z, max_center_z)
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
                    obstacle_bottom = (
                        float(position[2])
                        - _obb_vertical_half_extent(half_extents, quaternion)
                    ) if not is_dynamic else (
                        float(path_envelope_position[2])
                        - float(path_envelope_half_extents[2])
                    )
                    obstacle_top = (
                        float(position[2])
                        + _obb_vertical_half_extent(half_extents, quaternion)
                    ) if not is_dynamic else (
                        float(path_envelope_position[2])
                        + float(path_envelope_half_extents[2])
                    )
                    if (
                        self.mounting_plane_enabled
                        and not grounded_layout
                        and obstacle_bottom
                        < self.mounting_plane_height + 0.01
                    ):
                        continue
                    if (
                        physical_layout
                        and not grounded_layout
                        and (
                            obstacle_bottom
                            < self.mounting_plane_height
                            + self.physical_suspended_bottom_min
                            or obstacle_top > self.physical_obstacle_top_max
                        )
                    ):
                        continue
                    # Reject overlapping obstacle sets before the much more
                    # expensive capsule--OBB endpoint/path queries. This is an
                    # exact OBB test, so it changes only computation order,
                    # never the accepted scene distribution.
                    separated = all(
                        _horizontal_obb_surface_distance(
                            (
                                path_envelope_position
                                if is_dynamic else position
                            ),
                            (
                                path_envelope_half_extents
                                if is_dynamic else half_extents
                            ),
                            quaternion,
                            other_position,
                            other_half_extents,
                            other_quaternion,
                        ) >= self.obstacle_surface_clearance_min
                        for other_position, other_half_extents, other_quaternion in zip(
                            sampling_positions,
                            sampling_half_extents,
                            quaternions,
                            strict=True,
                        )
                    )
                    if not separated:
                        continue
                    query_position = (
                        path_envelope_position if is_dynamic else position
                    )
                    query_half_extents = (
                        path_envelope_half_extents if is_dynamic else half_extents
                    )
                    initial_distance = self._minimum_obstacle_clearance_for(
                        initial_capsules,
                        query_position,
                        query_half_extents,
                        quaternion,
                    )
                    endpoint_surface_clearance = (
                        0.01 if low_overpass is not None
                        else (
                            0.01 if path_relevant_layout
                            else self.obstacle_surface_clearance_min
                        )
                    )
                    if initial_distance < endpoint_surface_clearance:
                        continue
                    goal_distance = self._minimum_obstacle_clearance_for(
                        goal_capsules,
                        query_position,
                        query_half_extents,
                        quaternion,
                    )
                    if goal_distance < endpoint_surface_clearance:
                        continue
                    if path_relevant_layout and not is_dynamic:
                        # This only states where the object is placed. It does
                        # not label or prescribe an over/side avoidance route.
                        nominal_path_clearance = min(
                            self._minimum_obstacle_clearance_for(
                                capsules, position, half_extents, quaternion
                            )
                            for capsules in path_states
                        )
                        if nominal_path_clearance > self.physical_path_clearance_max:
                            continue
                    if (
                        layout == "path_offset"
                        and not is_dynamic
                        and not self.path_offset_min <= min(
                            self._minimum_obstacle_clearance_for(
                                capsules, position, half_extents, quaternion
                            )
                            for capsules in path_states
                        ) <= self.path_offset_max
                    ):
                        continue
                    if (
                        initial_distance >= endpoint_surface_clearance
                        and goal_distance >= endpoint_surface_clearance
                    ):
                        positions.append(position.copy())
                        velocities.append(velocity.copy())
                        half_extents_all.append(half_extents.copy())
                        quaternions.append(quaternion.copy())
                        sampling_positions.append(query_position.copy())
                        sampling_half_extents.append(query_half_extents.copy())
                        dynamic_mask.append(is_dynamic)
                        obstacle_layouts.append(obstacle_layout)
                        if candidate_is_tcp_blocker:
                            tcp_blocker_placed = True
                        if is_dynamic:
                            motion_centers.append(motion_center.copy())
                            motion_directions.append(motion_direction.copy())
                            motion_amplitudes.append(motion_amplitude)
                            motion_angular_frequencies.append(
                                motion_angular_frequency
                            )
                            motion_phases.append(motion_phase)
                        else:
                            motion_centers.append(position.copy())
                            motion_directions.append(np.zeros(3, dtype=np.float32))
                            motion_amplitudes.append(0.0)
                            motion_angular_frequencies.append(0.0)
                            motion_phases.append(0.0)
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
            self.obstacle_dynamic_mask = np.asarray(dynamic_mask, dtype=bool)
            self.obstacle_layouts = tuple(obstacle_layouts)
            if np.any(self.obstacle_dynamic_mask):
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
                self.dynamic_position_history = [
                    [] for _ in range(len(self.obstacle_positions))
                ]
                for obstacle_index, (
                    center, direction, amplitude, frequency, phase,
                ) in enumerate(zip(
                    self.dynamic_motion_centers,
                    self.dynamic_motion_directions,
                    self.dynamic_motion_amplitudes,
                    self.dynamic_motion_angular_frequencies,
                    self.dynamic_motion_phases,
                    strict=True,
                )):
                    if not self.obstacle_dynamic_mask[obstacle_index]:
                        continue
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
                    self.dynamic_position_history[obstacle_index] = history
            self._sync_primary_obstacle()
            self.obstacle_layout = layout
            self._create_obstacle_bodies()
            p.performCollisionDetection(physicsClientId=self.client_id)
            if not self._collision_events()["obstacle_collision"]:
                self.obstacle_sampling_time_s = perf_counter() - sampling_started
                return
            self._disable_obstacle()
        self.obstacle_sampling_time_s = perf_counter() - sampling_started
        raise ObstacleSamplingError(
            f"failed to sample a valid set of {obstacle_count} obstacles "
            f"after {self.obstacle_sampling_proposals} proposals"
        )

    def _create_counterfactual_obstacle(
        self,
        variant: str,
        path: list[np.ndarray],
        initial_capsules: list[Any],
        goal_capsules: list[Any],
        sampling_started: float,
    ) -> None:
        """Place one of four paired, evaluation-only obstacle variants.

        Every environment evaluates all four geometries while selecting the
        anchor.  Consequently the chosen TCP-path anchor is identical across
        a quartet; only the requested shape or lateral offset changes.
        """
        if variant not in self._COUNTERFACTUAL_VARIANTS:
            raise ValueError(f"unknown counterfactual variant {variant!r}")
        if len(path) < 3:
            raise ObstacleSamplingError("counterfactual TCP path is unavailable")

        # Local dimensions are [along path, across path, vertical].  The
        # low-wide/tall-narrow pair shares a centre.  The offset pair shares a
        # balanced shape and moves equally to either side of the path.
        local_geometry = {
            "low_wide": (np.asarray([0.045, 0.120, 0.050]), 0.0),
            "tall_narrow": (np.asarray([0.045, 0.050, 0.120]), 0.0),
            "offset_left": (np.asarray([0.045, 0.075, 0.085]), 0.045),
            "offset_right": (np.asarray([0.045, 0.075, 0.085]), -0.045),
        }
        centre = (len(path) - 1) // 2
        anchor_order = sorted(range(1, len(path) - 1), key=lambda i: abs(i - centre))
        selected: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
        # Counterfactual endpoints must be collision-free, but they need not
        # inherit the training sampler's large scene-generation buffer.  That
        # buffer would reject many valid tests and bias the retained goals.
        endpoint_clearance = 0.01
        for index in anchor_order:
            before = np.asarray(path[index - 1], dtype=np.float64)
            after = np.asarray(path[index + 1], dtype=np.float64)
            tangent = after[:2] - before[:2]
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1.0e-8:
                continue
            tangent /= tangent_norm
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
            yaw = float(np.arctan2(tangent[1], tangent[0]))
            quaternion = np.asarray(
                p.getQuaternionFromEuler([0.0, 0.0, yaw]), dtype=np.float32
            )
            quartet: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            all_valid = True
            for name, (local_half_extents, lateral_offset) in local_geometry.items():
                position = np.asarray(path[index], dtype=np.float32).copy()
                position[:2] += (normal * lateral_offset).astype(np.float32)
                half_extents = np.asarray(local_half_extents, dtype=np.float32)
                self.obstacle_sampling_proposals += 1
                if np.linalg.norm(position - self.goal_position) < 0.15:
                    all_valid = False
                    break
                if (
                    self.mounting_plane_enabled
                    and position[2] - half_extents[2]
                    < self.mounting_plane_height + 0.01
                ):
                    all_valid = False
                    break
                if self._minimum_obstacle_clearance_for(
                    initial_capsules, position, half_extents, quaternion
                ) < endpoint_clearance:
                    all_valid = False
                    break
                if self._minimum_obstacle_clearance_for(
                    goal_capsules, position, half_extents, quaternion
                ) < endpoint_clearance:
                    all_valid = False
                    break
                quartet[name] = (position, half_extents, quaternion)
            if all_valid and len(quartet) == len(local_geometry):
                selected = quartet
                break
        if selected is None:
            self.obstacle_sampling_time_s = perf_counter() - sampling_started
            raise ObstacleSamplingError(
                "no TCP-path anchor is feasible for all counterfactual variants"
            )

        position, half_extents, quaternion = selected[variant]
        self.obstacle_positions = position.reshape(1, 3).astype(np.float32)
        self.obstacle_velocities = np.zeros((1, 3), dtype=np.float32)
        self.obstacle_half_extents_all = half_extents.reshape(1, 3).astype(np.float32)
        self.obstacle_quaternions = quaternion.reshape(1, 4).astype(np.float32)
        self.obstacle_dynamic_mask = np.zeros(1, dtype=bool)
        self.obstacle_layouts = (f"counterfactual_{variant}",)
        self.dynamic_motion_centers = self.obstacle_positions.copy()
        self.dynamic_motion_directions = np.zeros((1, 3), dtype=np.float32)
        self.dynamic_motion_amplitudes = np.zeros(1, dtype=np.float32)
        self.dynamic_motion_angular_frequencies = np.zeros(1, dtype=np.float32)
        self.dynamic_motion_phases = np.zeros(1, dtype=np.float32)
        self._sync_primary_obstacle()
        self.obstacle_layout = f"counterfactual_{variant}"
        self._create_obstacle_bodies()
        p.performCollisionDetection(physicsClientId=self.client_id)
        if self._collision_events()["obstacle_collision"]:
            self._disable_obstacle()
            raise ObstacleSamplingError("counterfactual obstacle collides at reset")
        self.obstacle_sampling_time_s = perf_counter() - sampling_started

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

