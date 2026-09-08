from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_risk_sac.utils.trajectory_blending import ButterworthQuinticRTB


class JointVelocityRateLimiter:
    """Limit policy-command changes before smoothing, as in the reference paper."""

    def __init__(self, joint_count: int, max_delta: float | None) -> None:
        if max_delta is not None and max_delta <= 0.0:
            raise ValueError("max_delta must be positive or null")
        self.max_delta = max_delta
        self.previous = np.zeros(joint_count, dtype=np.float32)

    def reset(self, velocity: np.ndarray | None = None) -> None:
        if velocity is None:
            self.previous.fill(0.0)
        else:
            self.previous = np.asarray(velocity, dtype=np.float32).copy()

    def apply(self, velocity: np.ndarray) -> tuple[np.ndarray, bool]:
        value = np.asarray(velocity, dtype=np.float32)
        if self.max_delta is None:
            limited = value.copy()
        else:
            limited = self.previous + np.clip(value - self.previous, -self.max_delta, self.max_delta)
        was_limited = not np.allclose(limited, value)
        self.previous = limited.astype(np.float32)
        return self.previous.copy(), was_limited


@dataclass(frozen=True)
class ExecutionResult:
    policy_velocity: np.ndarray
    limited_policy_velocity: np.ndarray
    trajectory: np.ndarray
    command: np.ndarray
    beta: float
    rate_limited: bool


class ExecutionPipeline:
    """Policy-rate safety limiting followed by the configured smoothing stage."""

    def __init__(
        self,
        joint_count: int,
        action_scale: float,
        control_dt: float,
        smoothing_mode: str,
        fixed_beta: float,
        cutoff_angular_frequency: float,
        max_policy_velocity_delta: float | None,
        beta_min: float,
        beta_max: float,
        risk_high: float,
        lambda_beta: float,
    ) -> None:
        self.joint_count = joint_count
        self.action_scale = float(action_scale)
        self.smoothing_mode = smoothing_mode
        self.fixed_beta = float(fixed_beta)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.risk_high = float(risk_high)
        self.lambda_beta = float(lambda_beta)
        self.beta = self.fixed_beta
        self.rate_limiter = JointVelocityRateLimiter(joint_count, max_policy_velocity_delta)
        self.rtb = ButterworthQuinticRTB(joint_count, control_dt, cutoff_angular_frequency)

    def reset(self, adaptive: bool = False) -> None:
        self.beta = self.beta_min if adaptive else self.fixed_beta
        self.rate_limiter.reset()
        self.rtb.reset()

    def process(
        self,
        normalized_action: np.ndarray,
        previous_command: np.ndarray,
        risk_global: float,
        sample_count: int,
        adaptive: bool,
    ) -> ExecutionResult:
        action = np.clip(np.asarray(normalized_action, dtype=np.float32), -1.0, 1.0)
        raw_policy_velocity = self.action_scale * action
        policy_velocity, rate_limited = self.rate_limiter.apply(raw_policy_velocity)

        if adaptive:
            beta = self._adaptive_beta(risk_global)
            command = beta * policy_velocity + (1.0 - beta) * previous_command
            trajectory = np.repeat(command[None, :], sample_count, axis=0)
        elif self.smoothing_mode == "ema":
            beta = self.fixed_beta
            command = beta * policy_velocity + (1.0 - beta) * previous_command
            trajectory = np.repeat(command[None, :], sample_count, axis=0)
        elif self.smoothing_mode == "butterworth_quintic":
            beta = self.fixed_beta
            trajectory = self.rtb.trajectory(previous_command, policy_velocity, sample_count)
        else:
            raise ValueError(f"Unknown smoothing mode: {self.smoothing_mode}")

        trajectory = np.clip(trajectory, -self.action_scale, self.action_scale).astype(np.float32)
        self.beta = float(beta)
        return ExecutionResult(
            policy_velocity=raw_policy_velocity.astype(np.float32),
            limited_policy_velocity=policy_velocity,
            trajectory=trajectory,
            command=trajectory[-1].copy(),
            beta=self.beta,
            rate_limited=rate_limited,
        )

    def _adaptive_beta(self, risk_global: float) -> float:
        ratio = np.clip(risk_global / max(self.risk_high, 1e-6), 0.0, 1.0)
        beta_raw = self.beta_min + (self.beta_max - self.beta_min) * ratio
        return float(self.lambda_beta * beta_raw + (1.0 - self.lambda_beta) * self.beta)
