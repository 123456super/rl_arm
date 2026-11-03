"""Compact-keypoint task definitions for the serial thesis protocol."""

from rl_risk_sac.tasks.thesis_reaching_reduced_observation import (
    build_thesis_observation,
    homotopy_reward,
)

__all__ = ["build_thesis_observation", "homotopy_reward"]
