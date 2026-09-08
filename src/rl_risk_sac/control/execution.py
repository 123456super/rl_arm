from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_risk_sac.utils.trajectory_blending import ButterworthQuinticRTB


class JointVelocityRateLimiter:
    """Limit policy-command changes before smoothing, as in the reference paper.

    actor 可以突然改变关节速度目标；这个限制器先约束相邻策略步的速度差，
    后面的 EMA/RTB 再负责生成更平滑的执行轨迹。
    """

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
        """Clip the per-joint velocity change relative to the previous policy step."""
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
    """All action signals that are useful for control, logging, and evaluation."""

    policy_velocity: np.ndarray
    limited_policy_velocity: np.ndarray
    trajectory: np.ndarray
    command: np.ndarray
    beta: float
    rate_limited: bool


class ExecutionPipeline:
    """Policy-rate safety limiting followed by the configured smoothing stage.

    输入是 actor 的归一化动作；输出是 PyBullet 每个物理子步要执行的关节速度。
    这样训练算法不需要知道底层执行器是 EMA、Butterworth+五次插值还是别的实现。
    """

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
        """Convert a normalized SAC action into a substep velocity trajectory."""
        raw_policy_velocity, policy_velocity, rate_limited = self.prepare_policy_velocity(normalized_action)
        return self.smooth_prepared_velocity(
            raw_policy_velocity=raw_policy_velocity,
            limited_policy_velocity=policy_velocity,
            smoothing_target_velocity=policy_velocity,
            previous_command=previous_command,
            risk_global=risk_global,
            sample_count=sample_count,
            adaptive=adaptive,
            rate_limited=rate_limited,
        )

    def prepare_policy_velocity(self, normalized_action: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Scale and policy-rate-limit an actor action without smoothing it yet."""
        action = np.clip(np.asarray(normalized_action, dtype=np.float32), -1.0, 1.0)
        raw_policy_velocity = self.action_scale * action
        policy_velocity, rate_limited = self.rate_limiter.apply(raw_policy_velocity)
        return raw_policy_velocity.astype(np.float32), policy_velocity.astype(np.float32), rate_limited

    def smooth_prepared_velocity(
        self,
        raw_policy_velocity: np.ndarray,
        limited_policy_velocity: np.ndarray,
        smoothing_target_velocity: np.ndarray,
        previous_command: np.ndarray,
        risk_global: float,
        sample_count: int,
        adaptive: bool,
        rate_limited: bool = False,
    ) -> ExecutionResult:
        """Generate substep commands from an already prepared policy target.

        ``limited_policy_velocity`` remains the pre-safety policy target for
        logging. ``smoothing_target_velocity`` is the target actually passed to
        the smoothing stage, which lets safety filters intervene before RTB
        creates the physical substep trajectory.
        """
        target_velocity = np.asarray(smoothing_target_velocity, dtype=np.float32)
        if adaptive:
            # LDRC adaptive: 风险越高，beta 越靠近 beta_max，当前策略命令占比越大。
            # 风险低时 beta 较小，更多沿用上一条命令，动作更平滑。
            beta = self._adaptive_beta(risk_global)
            command = beta * target_velocity + (1.0 - beta) * previous_command
            trajectory = np.repeat(command[None, :], sample_count, axis=0)
        elif self.smoothing_mode == "ema":
            beta = self.fixed_beta
            command = beta * target_velocity + (1.0 - beta) * previous_command
            trajectory = np.repeat(command[None, :], sample_count, axis=0)
        elif self.smoothing_mode == "butterworth_quintic":
            beta = self.fixed_beta
            # RTB 分支直接生成 sample_count 个子步速度，command 取最后一个子步。
            trajectory = self.rtb.trajectory(previous_command, target_velocity, sample_count)
        else:
            raise ValueError(f"Unknown smoothing mode: {self.smoothing_mode}")

        trajectory = np.clip(trajectory, -self.action_scale, self.action_scale).astype(np.float32)
        self.beta = float(beta)
        return ExecutionResult(
            policy_velocity=np.asarray(raw_policy_velocity, dtype=np.float32),
            limited_policy_velocity=np.asarray(limited_policy_velocity, dtype=np.float32),
            trajectory=trajectory,
            command=trajectory[-1].copy(),
            beta=self.beta,
            rate_limited=rate_limited,
        )

    def _adaptive_beta(self, risk_global: float) -> float:
        ratio = np.clip(risk_global / max(self.risk_high, 1e-6), 0.0, 1.0)
        beta_raw = self.beta_min + (self.beta_max - self.beta_min) * ratio
        return float(self.lambda_beta * beta_raw + (1.0 - self.lambda_beta) * self.beta)
