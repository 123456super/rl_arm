from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class TargetState:
    """目标点的状态：位置和速度都在世界坐标系下。"""

    position: np.ndarray
    velocity: np.ndarray


class TargetProvider(Protocol):
    """环境使用的目标状态机接口。"""

    def reset(self, rng: np.random.Generator) -> TargetState: ...

    def advance(self, dt: float) -> TargetState: ...


class WorkspaceTargetProvider:
    """Static or linearly moving target independent from simulator state.

    目标点不依赖 PyBullet 物体本身；provider 只负责给出数学上的目标状态，
    环境再把可视化 marker 移到同一个位置。
    """

    def __init__(self, config: dict[str, Any], workspace: dict[str, list[float]]) -> None:
        self.config = config
        self.workspace = workspace
        self.mode = str(config.get("mode", "static"))
        self._position = np.zeros(3, dtype=np.float32)
        self._velocity = np.zeros(3, dtype=np.float32)

    def reset(self, rng: np.random.Generator) -> TargetState:
        """为新 episode 采样或设置目标初始位置和速度。"""
        if self.config.get("fixed", False):
            self._position = np.asarray(self.config["position"], dtype=np.float32).copy()
        else:
            self._position = np.asarray(
                [rng.uniform(*self.workspace[axis]) for axis in ("x", "y", "z")],
                dtype=np.float32,
            )

        if self.mode == "static":
            self._velocity = np.zeros(3, dtype=np.float32)
        elif self.mode == "linear_bounce":
            # 动态目标从随机方向和随机速度开始，在 workspace 边界反弹。
            direction = rng.normal(size=3).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-8
            speed = rng.uniform(*self.config["speed_range"])
            self._velocity = (direction * speed).astype(np.float32)
        else:
            raise ValueError(f"Unknown goal mode: {self.mode}")
        return self._state()

    def advance(self, dt: float) -> TargetState:
        """推进目标运动 dt 秒；静态目标会原地不动。"""
        if self.mode == "linear_bounce":
            self._position = (self._position + self._velocity * dt).astype(np.float32)
            for axis_index, axis in enumerate(("x", "y", "z")):
                low, high = self.workspace[axis]
                if self._position[axis_index] < low or self._position[axis_index] > high:
                    # 越界时反转该轴速度并夹回边界内，形成“线性反弹”。
                    self._velocity[axis_index] *= -1.0
                    self._position[axis_index] = np.clip(self._position[axis_index], low, high)
        return self._state()

    def _state(self) -> TargetState:
        """返回状态副本，避免调用方意外修改 provider 内部数组。"""
        return TargetState(self._position.copy(), self._velocity.copy())
