from __future__ import annotations

import numpy as np


def quintic_smoothstep(s: float | np.ndarray) -> float | np.ndarray:
    """五次 smoothstep：端点的一阶、二阶导数都为 0。

    用在速度插值时，这能让子步轨迹在起点和终点处更平滑，减少加速度
    和 jerk 的突变。
    """
    clipped = np.clip(s, 0.0, 1.0)
    return 6.0 * clipped**5 - 15.0 * clipped**4 + 10.0 * clipped**3


class ButterworthQuinticRTB:
    """Fixed two-stage real-time trajectory blender for joint velocities.

    A first-order Butterworth low-pass filter is followed by a quintic
    interpolation from the currently executed velocity to the filtered target.

    中文理解：先用一阶低通滤波削弱策略速度命令的高频抖动，再在一个
    control_dt 内用五次曲线生成多个物理子步速度。
    """

    def __init__(self, joint_count: int, control_dt: float, cutoff_angular_frequency: float) -> None:
        if joint_count <= 0:
            raise ValueError("joint_count must be positive")
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if cutoff_angular_frequency <= 0.0:
            raise ValueError("cutoff_angular_frequency must be positive")
        self.joint_count = int(joint_count)
        self.control_dt = float(control_dt)
        self.cutoff_angular_frequency = float(cutoff_angular_frequency)
        self.previous_policy_velocity = np.zeros(self.joint_count, dtype=np.float32)
        self.previous_filtered_velocity = np.zeros(self.joint_count, dtype=np.float32)

    def reset(self, velocity: np.ndarray | None = None) -> None:
        """重置滤波器记忆。

        如果给定 velocity，就把上一策略速度和上一滤波速度都设为它；
        否则从全零速度开始。
        """
        initial = (
            np.zeros(self.joint_count, dtype=np.float32)
            if velocity is None
            else self._as_velocity(velocity)
        )
        self.previous_policy_velocity = initial.copy()
        self.previous_filtered_velocity = initial.copy()

    def filter(self, policy_velocity: np.ndarray) -> np.ndarray:
        """对策略速度做一阶 Butterworth 低通滤波。"""
        policy_velocity = self._as_velocity(policy_velocity)
        # 双线性变换形式的一阶低通系数。cutoff 越小，k 越大，输出越平滑。
        k = 2.0 / (self.cutoff_angular_frequency * self.control_dt)
        filtered = (
            policy_velocity
            + self.previous_policy_velocity
            - (1.0 - k) * self.previous_filtered_velocity
        ) / (1.0 + k)
        self.previous_policy_velocity = policy_velocity.copy()
        self.previous_filtered_velocity = filtered.astype(np.float32)
        return self.previous_filtered_velocity.copy()

    def trajectory(
        self,
        start_velocity: np.ndarray,
        policy_velocity: np.ndarray,
        sample_count: int,
    ) -> np.ndarray:
        """生成一个控制周期内的物理子步速度轨迹。

        start_velocity 是上一控制周期最后实际执行的速度；policy_velocity
        是本周期策略给出的目标速度。返回形状为 [sample_count, joint_count]。
        """
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        start = self._as_velocity(start_velocity)
        target = self.filter(policy_velocity)
        phase = np.arange(1, sample_count + 1, dtype=np.float32) / float(sample_count)
        blend = np.asarray(quintic_smoothstep(phase), dtype=np.float32)[:, None]
        return (start[None, :] + blend * (target - start)[None, :]).astype(np.float32)

    def _as_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """把输入转成 joint_count 维 float32 速度向量，并检查维度。"""
        value = np.asarray(velocity, dtype=np.float32)
        if value.shape != (self.joint_count,):
            raise ValueError(f"Expected velocity shape {(self.joint_count,)}, got {value.shape}")
        return value
