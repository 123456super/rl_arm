"""Immutable runtime objects used by the active thesis PPO pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeAlias

Vector3: TypeAlias = tuple[float, float, float]


def _vector3(value: Any) -> Vector3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("expected a three-element vector")
    return float(value[0]), float(value[1]), float(value[2])


@dataclass(frozen=True)
class CapsuleSpec:
    name: str
    parent_link_name: str
    child_link_name: str
    radius: float
    parent_offset: Vector3
    child_offset: Vector3
    allow_degenerate: bool


@dataclass(frozen=True)
class RobotResetConfig:
    default_joint_positions: tuple[float, ...]


@dataclass(frozen=True)
class RobotRuntimeConfig:
    urdf: str
    base_position: Vector3
    joint_names: tuple[str, ...]
    tool_link_name: str
    reset: RobotResetConfig
    capsules: tuple[CapsuleSpec, ...]


@dataclass(frozen=True)
class DeviceSelectionConfig:
    fallback_to_cpu: bool
    candidate_ids: tuple[int, ...] | None
    min_free_memory_gb: float
    memory_weight: float
    compute_weight: float
    print_summary: bool


@dataclass(frozen=True)
class RiskConfig:
    """Geometry-risk parameters retained as a reusable numerical API type."""

    d_safe: float
    geometry_margin: float
    sigma_d: float
    v_max: float
    ttc_max: float
    tau: float
    eps: float
    eps_v: float
    w_distance: float
    w_velocity: float
    w_ttc: float


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    device: str
    device_selection: DeviceSelectionConfig
    robot: RobotRuntimeConfig

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "RuntimeConfig":
        robot = config["robot"]
        reset = robot["reset"]
        capsules = tuple(
            CapsuleSpec(
                name=str(item["name"]),
                parent_link_name=str(item["parent_link_name"]),
                child_link_name=str(item["child_link_name"]),
                radius=float(item["radius"]),
                parent_offset=_vector3(item["parent_offset"]),
                child_offset=_vector3(item["child_offset"]),
                allow_degenerate=bool(item["allow_degenerate"]),
            )
            for item in robot["capsules"]
        )
        selection = config["device_selection"]
        candidate_ids = selection["candidate_ids"]
        return cls(
            seed=int(config["seed"]),
            device=str(config["device"]),
            device_selection=DeviceSelectionConfig(
                fallback_to_cpu=bool(selection["fallback_to_cpu"]),
                candidate_ids=(
                    None
                    if candidate_ids is None
                    else tuple(int(value) for value in candidate_ids)
                ),
                min_free_memory_gb=float(selection["min_free_memory_gb"]),
                memory_weight=float(selection["memory_weight"]),
                compute_weight=float(selection["compute_weight"]),
                print_summary=bool(selection["print_summary"]),
            ),
            robot=RobotRuntimeConfig(
                urdf=str(robot["urdf"]),
                base_position=_vector3(robot["base_position"]),
                joint_names=tuple(str(name) for name in robot["joint_names"]),
                tool_link_name=str(robot["tool_link_name"]),
                reset=RobotResetConfig(
                    default_joint_positions=tuple(
                        float(value) for value in reset["default_joint_positions"]
                    )
                ),
                capsules=capsules,
            ),
        )


def resolve_robot_urdf(repo_root: Path, robot: RobotRuntimeConfig) -> Path:
    path = Path(robot.urdf)
    return path if path.is_absolute() else repo_root / path
