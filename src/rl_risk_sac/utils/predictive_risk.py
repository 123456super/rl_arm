"""Short-horizon link-level risk prediction with explicit uncertainty margins.

This module intentionally has no environment or controller dependency.  The
next-stage safety filter can consume :class:`PredictiveLinkRisk` directly and
must treat an unusable result as a safe-stop condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from rl_risk_sac.robots.ur5_capsules import CapsuleState
from rl_risk_sac.utils.risk import closest_point_on_segment


class PredictionStatus(str, Enum):
    """Whether an obstacle estimate can be used by the safety layer."""

    VALID = "valid"
    INVALID = "invalid"
    STALE = "stale"


@dataclass(frozen=True)
class ObstacleStateEstimate:
    """Timestamped obstacle state supplied by perception or simulation.

    Error bounds are Euclidean bounds in metres and metres per second.  They
    are deliberately required inputs: a missing bound must be represented by
    ``valid=False`` instead of silently assuming zero uncertainty.
    """

    position: np.ndarray
    velocity: np.ndarray
    radius_m: float
    timestamp_s: float
    position_error_bound_m: float
    velocity_error_bound_mps: float
    valid: bool = True


@dataclass(frozen=True)
class PredictiveRiskConfig:
    """Assumptions used to turn a prediction into a conservative clearance."""

    d_safe_m: float = 0.12
    prediction_horizon_s: float = 0.15
    max_observation_age_s: float = 0.10
    control_delay_s: float = 0.05
    max_link_speed_mps: float = 0.7
    tracking_error_bound_m: float = 0.01
    geometry_margin_m: float | tuple[float, ...] = 0.0


@dataclass(frozen=True)
class PredictiveLinkRisk:
    """Per-link prediction result used as the P2 safety-filter input."""

    status: PredictionStatus
    status_reason: str
    observation_age_s: float
    closest_prediction_times_s: np.ndarray
    predicted_distances_m: np.ndarray
    robust_distances_m: np.ndarray
    safety_functions_m: np.ndarray
    geometry_margins_m: np.ndarray
    perception_margin_m: float
    delay_margin_m: float
    tracking_margin_m: float

    @property
    def usable(self) -> bool:
        return self.status is PredictionStatus.VALID

    @property
    def requires_safe_stop(self) -> bool:
        """True when a filter must not issue a motion command from this state."""

        return not self.usable


def compute_predictive_link_risk(
    capsules: Sequence[CapsuleState],
    link_velocities_mps: np.ndarray,
    obstacle: ObstacleStateEstimate,
    now_s: float,
    config: PredictiveRiskConfig,
) -> PredictiveLinkRisk:
    """Predict robust surface clearance for each link over a short horizon.

    Capsule endpoints and obstacle centre are propagated with their supplied
    first-order velocities over ``prediction_horizon_s``.  The robust distance
    then subtracts independent geometry, perception, delay, and tracking
    margins.  Observation age and control delay are not assumed to be known
    perfectly; their possible relative displacement is covered by
    ``delay_margin_m``.
    """

    _validate_config(config)
    count = len(capsules)
    if count == 0:
        raise ValueError("capsules must not be empty")
    velocities = _as_link_velocities(link_velocities_mps, count)
    now = float(now_s)
    if not np.isfinite(now):
        raise ValueError("now_s must be finite")

    estimate, reason = _validate_estimate(obstacle)
    if reason is not None:
        return _unusable_result(count, PredictionStatus.INVALID, reason)

    position, velocity = estimate
    age_s = now - float(obstacle.timestamp_s)
    if age_s < 0.0:
        return _unusable_result(count, PredictionStatus.INVALID, "observation timestamp is in the future", age_s)
    if age_s > config.max_observation_age_s:
        return _unusable_result(count, PredictionStatus.STALE, "observation exceeded max_observation_age_s", age_s)

    geometry_margins = _geometry_margins(config.geometry_margin_m, count)
    horizon = config.prediction_horizon_s
    predicted_distances = np.zeros(count, dtype=np.float64)
    closest_prediction_times = np.zeros(count, dtype=np.float64)

    for index, capsule in enumerate(capsules):
        center_distance, closest_time = _minimum_center_distance_over_horizon(
            position,
            velocity,
            np.asarray(capsule.start, dtype=np.float64),
            np.asarray(capsule.end, dtype=np.float64),
            velocities[index],
            horizon,
        )
        predicted_distances[index] = center_distance - float(capsule.radius) - obstacle.radius_m
        closest_prediction_times[index] = closest_time

    perception_margin = float(
        obstacle.position_error_bound_m + obstacle.velocity_error_bound_mps * horizon
    )
    delay_window = age_s + config.control_delay_s
    delay_margin = float(
        (np.linalg.norm(velocity) + obstacle.velocity_error_bound_mps + config.max_link_speed_mps) * delay_window
    )
    tracking_margin = config.tracking_error_bound_m
    robust_distances = predicted_distances - geometry_margins - perception_margin - delay_margin - tracking_margin
    safety_functions = robust_distances - config.d_safe_m
    return PredictiveLinkRisk(
        status=PredictionStatus.VALID,
        status_reason="",
        observation_age_s=age_s,
        closest_prediction_times_s=closest_prediction_times,
        predicted_distances_m=predicted_distances,
        robust_distances_m=robust_distances,
        safety_functions_m=safety_functions,
        geometry_margins_m=geometry_margins,
        perception_margin_m=perception_margin,
        delay_margin_m=delay_margin,
        tracking_margin_m=tracking_margin,
    )


def _validate_config(config: PredictiveRiskConfig) -> None:
    non_negative = {
        "prediction_horizon_s": config.prediction_horizon_s,
        "max_observation_age_s": config.max_observation_age_s,
        "control_delay_s": config.control_delay_s,
        "max_link_speed_mps": config.max_link_speed_mps,
        "tracking_error_bound_m": config.tracking_error_bound_m,
    }
    if not np.isfinite(config.d_safe_m) or config.d_safe_m <= 0.0:
        raise ValueError("d_safe_m must be finite and positive")
    for name, value in non_negative.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")


def _as_link_velocities(velocities: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(velocities, dtype=np.float64)
    if values.shape != (count, 3):
        raise ValueError(f"link_velocities_mps must have shape ({count}, 3)")
    if not np.isfinite(values).all():
        raise ValueError("link_velocities_mps must be finite")
    return values


def _validate_estimate(obstacle: ObstacleStateEstimate) -> tuple[tuple[np.ndarray, np.ndarray] | None, str | None]:
    if not obstacle.valid:
        return None, "obstacle estimate is marked invalid"
    position = np.asarray(obstacle.position, dtype=np.float64)
    velocity = np.asarray(obstacle.velocity, dtype=np.float64)
    if position.shape != (3,) or velocity.shape != (3,):
        return None, "obstacle position and velocity must each have shape (3,)"
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        return None, "obstacle position and velocity must be finite"
    scalar_values = {
        "radius_m": obstacle.radius_m,
        "timestamp_s": obstacle.timestamp_s,
        "position_error_bound_m": obstacle.position_error_bound_m,
        "velocity_error_bound_mps": obstacle.velocity_error_bound_mps,
    }
    for name, value in scalar_values.items():
        if not np.isfinite(value):
            return None, f"{name} must be finite"
        if name != "timestamp_s" and value < 0.0:
            return None, f"{name} must be non-negative"
    return (position, velocity), None


def _minimum_center_distance_over_horizon(
    obstacle_position: np.ndarray,
    obstacle_velocity: np.ndarray,
    capsule_start: np.ndarray,
    capsule_end: np.ndarray,
    link_velocity: np.ndarray,
    horizon_s: float,
) -> tuple[float, float]:
    """Return the exact closest centre distance for a translating capsule.

    In the capsule frame the obstacle is a point travelling on a line.  The
    minimisation over time and segment coordinate is a two-variable bounded
    least-squares problem.  Its optimum is either interior or on one of the
    four boundaries, all of which are evaluated below.
    """

    segment = capsule_end - capsule_start
    relative_position = obstacle_position - capsule_start
    relative_velocity = obstacle_velocity - link_velocity
    candidates: list[tuple[float, float]] = []

    for time_s in (0.0, horizon_s):
        point = relative_position + relative_velocity * time_s
        _, rho = closest_point_on_segment(point, np.zeros(3), segment)
        candidates.append((time_s, rho))

    velocity_sq = float(np.dot(relative_velocity, relative_velocity))
    if velocity_sq > 1e-12:
        for rho in (0.0, 1.0):
            offset = relative_position - rho * segment
            time_s = float(np.clip(-np.dot(offset, relative_velocity) / velocity_sq, 0.0, horizon_s))
            candidates.append((time_s, rho))

    matrix = np.column_stack((relative_velocity, -segment))
    interior, _, _, _ = np.linalg.lstsq(matrix, -relative_position, rcond=None)
    time_s, rho = float(interior[0]), float(interior[1])
    if 0.0 <= time_s <= horizon_s and 0.0 <= rho <= 1.0:
        candidates.append((time_s, rho))

    best_time, best_rho = min(
        candidates,
        key=lambda candidate: np.linalg.norm(
            relative_position + relative_velocity * candidate[0] - segment * candidate[1]
        ),
    )
    distance = float(np.linalg.norm(relative_position + relative_velocity * best_time - segment * best_rho))
    return distance, best_time


def _geometry_margins(value: float | tuple[float, ...], count: int) -> np.ndarray:
    margins = np.asarray(value, dtype=np.float64)
    if margins.ndim == 0:
        margins = np.full(count, float(margins), dtype=np.float64)
    if margins.shape != (count,):
        raise ValueError(f"geometry_margin_m must be a scalar or contain {count} values")
    if not np.isfinite(margins).all() or (margins < 0.0).any():
        raise ValueError("geometry_margin_m must be finite and non-negative")
    return margins


def _unusable_result(
    count: int,
    status: PredictionStatus,
    reason: str,
    observation_age_s: float = float("nan"),
) -> PredictiveLinkRisk:
    unavailable = np.full(count, np.nan, dtype=np.float64)
    return PredictiveLinkRisk(
        status=status,
        status_reason=reason,
        observation_age_s=observation_age_s,
        closest_prediction_times_s=unavailable.copy(),
        predicted_distances_m=unavailable.copy(),
        robust_distances_m=unavailable.copy(),
        safety_functions_m=np.full(count, -np.inf, dtype=np.float64),
        geometry_margins_m=unavailable.copy(),
        perception_margin_m=float("nan"),
        delay_margin_m=float("nan"),
        tracking_margin_m=float("nan"),
    )
