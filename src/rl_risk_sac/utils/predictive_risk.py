from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_risk_sac.robots.ur5_capsules import CapsuleState
from rl_risk_sac.utils.risk import RiskConfig, closest_point_on_segment
from rl_risk_sac.utils.runtime_config import PredictiveRiskConfig


@dataclass
class PredictiveLinkRisk:
    """预测风险结果。

    d_pred: 每根连杆在预测窗内的最小表面距离。
    t_enter: 首次进入安全距离 d_safe 的时间；未进入则为 inf。
    risk_pred_per_link: 每根连杆的预测风险分数。
    risk_pred_body: 整个机械臂的最大预测风险。
    """

    d_pred: np.ndarray
    t_enter: np.ndarray
    risk_pred_per_link: np.ndarray
    risk_pred_body: float
    critical_link: int
    closest_points_pred: np.ndarray
    closest_times: np.ndarray
    approach_velocities: np.ndarray
    d_pred_raw: np.ndarray | None = None


def config_from_current_risk(
    config: RiskConfig,
    horizon: float,
    step: float,
    w_min_distance: float,
    w_enter_time: float,
    w_approach: float,
) -> PredictiveRiskConfig:
    """从当前帧风险配置派生预测风险配置。

    这样离线预测风险可以复用论文中已有的 d_safe、sigma_d、tau、v_max
    等尺度参数，只额外指定预测时间窗和离散步长。
    """
    return PredictiveRiskConfig(
        horizon=horizon,
        step=step,
        d_safe=config.d_safe,
        geometry_margin=config.geometry_margin,
        sigma_d=config.sigma_d,
        tau_enter=config.tau,
        v_max=config.v_max,
        eps=config.eps,
        w_min_distance=w_min_distance,
        w_enter_time=w_enter_time,
        w_approach=w_approach,
    )


def compute_predictive_link_risk(
    capsules: list[CapsuleState],
    prev_capsules: list[CapsuleState] | None,
    obstacle_center: np.ndarray,
    obstacle_velocity: np.ndarray,
    obstacle_radius: float,
    dt: float,
    config: PredictiveRiskConfig,
    use_end_effector_only: bool = False,
) -> PredictiveLinkRisk:
    """计算未来时间窗内的连杆级预测风险。

    算法假设在短 horizon 内连杆端点和障碍物都做匀速运动：
    - 连杆端点速度由当前 capsule 与 prev_capsules 差分得到；
    - 障碍物速度直接来自 ObstacleState；
    - 在每个预测时间点重新计算球心到胶囊线段的最近距离；
    - 最后用最小距离、首次进入安全距离时间和最大接近速度融合风险。
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if config.horizon < 0:
        raise ValueError("horizon must be non-negative")
    if config.step <= 0:
        raise ValueError("step must be positive")

    count = len(capsules)
    times = _prediction_times(config.horizon, config.step)
    # 末端基线只让最后一根胶囊参与风险打分；其他连杆仍计算几何量，
    # 但 risk_pred_per_link 会保持 0。
    active_indices = {count - 1} if use_end_effector_only else set(range(count))

    d_pred = np.full(count, np.inf, dtype=np.float32)
    d_pred_raw = np.full(count, np.inf, dtype=np.float32)
    t_enter = np.full(count, np.inf, dtype=np.float32)
    closest_points_pred = np.zeros((count, 3), dtype=np.float32)
    closest_times = np.zeros(count, dtype=np.float32)
    approach_velocities = np.zeros(count, dtype=np.float32)
    risks = np.zeros(count, dtype=np.float32)

    for i, capsule in enumerate(capsules):
        start_velocity, end_velocity = _capsule_endpoint_velocities(capsule, prev_capsules, i, dt)
        link_approach = 0.0

        for t in times:
            # 对连杆胶囊两端和障碍物球心做线性外推，得到未来 t 时刻的几何状态。
            start_t = capsule.start + start_velocity * t
            end_t = capsule.end + end_velocity * t
            obstacle_t = obstacle_center + obstacle_velocity * t
            closest, _ = closest_point_on_segment(obstacle_t, start_t, end_t)
            delta = obstacle_t - closest
            center_distance = float(np.linalg.norm(delta))
            raw_surface_distance = center_distance - capsule.radius - obstacle_radius
            surface_distance = raw_surface_distance - config.geometry_margin

            if surface_distance < d_pred[i]:
                # 记录预测窗内最近的一次距离和对应最近点，用于诊断/画图。
                d_pred[i] = surface_distance
                d_pred_raw[i] = raw_surface_distance
                closest_points_pred[i] = closest
                closest_times[i] = t

            if not np.isfinite(t_enter[i]) and surface_distance <= config.d_safe:
                # 只记录首次进入安全距离的时间，越早进入风险越高。
                t_enter[i] = t

            direction = delta / (center_distance + config.eps)
            # 这里用胶囊两端速度的平均值近似最近点速度；比固定取 start/end
            # 更稳定，但仍是短时线性预测的近似。
            link_velocity_t = start_velocity + (end_velocity - start_velocity) * 0.5
            distance_rate = float(np.dot(direction, obstacle_velocity - link_velocity_t))
            link_approach = max(link_approach, max(0.0, -distance_rate))

        approach_velocities[i] = link_approach
        if i in active_indices:
            # 只有 active link 参与风险打分；ee-only 基线借此屏蔽非末端连杆。
            risks[i] = _predictive_risk_score(
                min_distance=float(d_pred[i]),
                enter_time=float(t_enter[i]),
                approach_velocity=link_approach,
                config=config,
            )

    if count == 0:
        critical_link = -1
        risk_body = 0.0
    else:
        critical_link = int(np.argmax(risks))
        risk_body = float(np.max(risks))

    return PredictiveLinkRisk(
        d_pred=d_pred,
        t_enter=t_enter,
        risk_pred_per_link=risks,
        risk_pred_body=risk_body,
        critical_link=critical_link,
        closest_points_pred=closest_points_pred,
        closest_times=closest_times,
        approach_velocities=approach_velocities,
        d_pred_raw=d_pred_raw,
    )


def _prediction_times(horizon: float, step: float) -> np.ndarray:
    """生成从 0 到 horizon 的预测采样时间点。"""
    if horizon == 0:
        return np.asarray([0.0], dtype=np.float32)
    # ceil 保证最后覆盖 horizon；随后把最后一个点精确设成 horizon，
    # 避免浮点误差或 step 不能整除 horizon 带来的越界。
    count = int(np.ceil(horizon / step))
    times = np.linspace(0.0, count * step, count + 1, dtype=np.float32)
    times[-1] = horizon
    return times


def _capsule_endpoint_velocities(
    capsule: CapsuleState,
    prev_capsules: list[CapsuleState] | None,
    index: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """用相邻两帧 capsule 端点差分估计连杆端点速度。"""
    if prev_capsules is None or index >= len(prev_capsules):
        # reset 后第一帧没有上一帧几何，保守地认为连杆当前静止。
        zero = np.zeros(3, dtype=np.float32)
        return zero, zero

    previous = prev_capsules[index]
    start_velocity = (capsule.start - previous.start) / dt
    end_velocity = (capsule.end - previous.end) / dt
    return start_velocity.astype(np.float32), end_velocity.astype(np.float32)


def _predictive_risk_score(
    min_distance: float,
    enter_time: float,
    approach_velocity: float,
    config: PredictiveRiskConfig,
) -> float:
    """把预测窗内的距离、进入时间和接近速度融合成 [0, 1] 风险。"""
    distance_risk = float(np.clip(np.exp(-(min_distance - config.d_safe) / config.sigma_d), 0.0, 1.0))
    if np.isfinite(enter_time):
        # 越早进入安全距离，enter_risk 越接近 1；未进入则该项为 0。
        enter_risk = float(np.exp(-enter_time / config.tau_enter))
    else:
        enter_risk = 0.0
    approach_risk = float(np.clip(approach_velocity / config.v_max, 0.0, 1.0))
    return float(
        np.clip(
            config.w_min_distance * distance_risk
            + config.w_enter_time * enter_risk
            + config.w_approach * approach_risk,
            0.0,
            1.0,
        )
    )
