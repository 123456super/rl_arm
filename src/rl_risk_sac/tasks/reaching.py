from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_risk_sac.utils.predictive_risk import PredictiveLinkRisk
from rl_risk_sac.utils.risk import LinkRisk


@dataclass(frozen=True)
class ReachingObservationBuilder:
    """Build the fixed observation vector consumed by SAC.

    observation 的顺序就是模型输入 schema。只要 checkpoint 还要复用，
    这里的字段顺序和维度就不能随意改。
    """
    action_scale: float
    distance_clip: tuple[float, float]
    v_max: float
    ttc_max: float
    schema_version: str = "link_risk_v1"
    prediction_horizon: float = 1.0
    include_predictive_per_link_score: bool = True

    def dimension(self, joint_count: int, link_count: int) -> int:
        # q, qdot, previous command, goal position/velocity errors,
        # and distance/direction/approach/TTC/risk for every link.
        base_dim = joint_count * 3 + link_count * 7 + 7
        if self.schema_version == "link_risk_v1":
            return base_dim
        if self.schema_version == "link_risk_pred_v1":
            # d_pred, T_enter, optional risk_pred, critical-link one-hot, plus
            # body-level summaries: min d_pred, min T_enter, max risk_pred.
            per_link_score_dim = link_count if self.include_predictive_per_link_score else 0
            return base_dim + link_count * 3 + per_link_score_dim + 3
        raise ValueError(f"Unknown observation schema {self.schema_version!r}")

    def build(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        goal_error: np.ndarray,
        goal_velocity_error: np.ndarray,
        risk: LinkRisk,
        previous_command: np.ndarray,
        beta: float,
        predictive_risk: PredictiveLinkRisk | None = None,
    ) -> np.ndarray:
        # 归一化的关节角/速度让神经网络输入处在较稳定的数值范围；
        # 风险相关项保留 per-link 结构，再展平成一维向量。
        parts = [
            q / np.pi,
            q_dot / max(self.action_scale, 1e-6),
            goal_error,
            goal_velocity_error,
            np.clip(risk.distances, *self.distance_clip),
            risk.directions.reshape(-1),
            np.clip(risk.approach_velocities / max(self.v_max, 1e-6), 0.0, 1.0),
            risk.ttc / max(self.ttc_max, 1e-6),
            risk.risks,
            previous_command / max(self.action_scale, 1e-6),
            np.array([beta], dtype=np.float32),
        ]

        if self.schema_version == "link_risk_pred_v1":
            if predictive_risk is None:
                raise ValueError("predictive_risk is required for link_risk_pred_v1")
            link_count = len(risk.risks)
            critical_link = np.zeros(link_count, dtype=np.float32)
            if 0 <= predictive_risk.critical_link < link_count:
                critical_link[predictive_risk.critical_link] = 1.0
            finite_t_enter = np.where(
                np.isfinite(predictive_risk.t_enter),
                predictive_risk.t_enter,
                self.prediction_horizon,
            )
            t_enter_norm = np.clip(finite_t_enter / max(self.prediction_horizon, 1e-6), 0.0, 1.0)
            min_t_enter = float(np.min(t_enter_norm)) if len(t_enter_norm) else 1.0
            min_d_pred = float(np.min(predictive_risk.d_pred)) if len(predictive_risk.d_pred) else self.distance_clip[1]
            parts.extend(
                [
                    np.clip(predictive_risk.d_pred, *self.distance_clip),
                    t_enter_norm,
                ]
            )
            if self.include_predictive_per_link_score:
                parts.append(predictive_risk.risk_pred_per_link)
            parts.extend(
                [
                    critical_link,
                    np.array(
                        [
                            np.clip(min_d_pred, *self.distance_clip),
                            min_t_enter,
                            predictive_risk.risk_pred_body,
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
        elif self.schema_version != "link_risk_v1":
            raise ValueError(f"Unknown observation schema {self.schema_version!r}")

        return np.concatenate(parts).astype(np.float32)


@dataclass(frozen=True)
class ReachingObjective:
    """Reward/cost definition for the reaching-with-obstacle task."""
    reward_config: dict[str, Any]
    cost_config: dict[str, Any]
    safe_distance: float

    def reward(
        self,
        goal_error_norm: float,
        previous_goal_error_norm: float,
        command: np.ndarray,
        previous_command: np.ndarray,
        success: bool,
        collision: bool,
    ) -> float:
        # reward 主要鼓励靠近目标，同时惩罚速度命令跳变；安全相关信号
        # 通过 cost 返回，固定惩罚方法才会在环境里把 cost 扣进 reward。
        progress = previous_goal_error_norm - goal_error_norm
        smooth = float(np.sum(np.square(command - previous_command)))
        value = (
            -float(self.reward_config["w_position"]) * goal_error_norm**2
            + float(self.reward_config["w_progress"]) * progress
            - float(self.reward_config["w_smooth"]) * smooth
        )
        if success:
            value += float(self.reward_config["success_bonus"])
        if collision:
            value -= float(self.reward_config["collision_penalty"])
        return float(value)

    def cost(self, risk: LinkRisk, collision: bool) -> float:
        # cost 是约束 SAC 关注的安全代价：连续风险 + 是否进入安全距离
        # + 是否真实碰撞。它和 reward 分开记录，便于 LDRC 使用 cost critic。
        violation = float(risk.d_min < self.safe_distance)
        return float(
            float(self.cost_config["k_risk"]) * risk.risk_global
            + float(self.cost_config["k_violation"]) * violation
            + float(self.cost_config["k_collision"]) * float(collision)
        )
