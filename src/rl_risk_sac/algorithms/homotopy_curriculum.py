from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SceneHomotopy:
    xi: float
    floor: float
    strict: bool = False
    eligible_steps: int = 0
    strict_steps: int = 0
    replay_strictified: bool = False
    outcomes: deque = field(default_factory=lambda: deque(maxlen=100))


class HomotopyCurriculum:
    """Episode-level S0/S1/S2 mixture and gated safety continuation."""

    def __init__(self, stage: str, seed: int = 0, ramp_steps: int = 50000) -> None:
        if stage not in {"s0", "s1", "s2"}:
            raise ValueError("stage must be s0, s1 or s2")
        self.stage, self.ramp_steps = stage, int(ramp_steps)
        self.rng = np.random.default_rng(seed)
        self.probabilities = {"s0": [1, 0, 0], "s1": [.25, .75, 0], "s2": [.20, .30, .50]}[stage]
        self.states = {
            "static": SceneHomotopy(1.0 if stage == "s2" else .02, .90, strict=stage == "s2", replay_strictified=stage == "s2"),
            "dynamic": SceneHomotopy(.02, .80),
        }

    def choose_scene(self) -> str:
        return str(self.rng.choice(["none", "static", "dynamic"], p=self.probabilities))

    def contract(self, scene: str) -> tuple[float, bool]:
        return (1.0, True) if scene == "none" else (self.states[scene].xi, self.states[scene].strict)

    def finish_episode(self, scene: str, task_reached: bool, transitions: int) -> dict[str, Any]:
        if scene == "none": return {"became_strict": False, "rolling_task_reach_rate": None}
        state = self.states[scene]; was_strict = state.strict; state.outcomes.append(bool(task_reached))
        rate = float(np.mean(state.outcomes))
        became = False
        if not state.strict and len(state.outcomes) == 100 and rate >= state.floor:
            state.eligible_steps += int(transitions)
            old = state.xi
            state.xi = min(1.0, .02 + .98 * state.eligible_steps / self.ramp_steps)
            if old < 1.0 and state.xi >= 1.0:
                state.strict = True; became = True
        if was_strict:
            state.strict_steps += int(transitions)
        return {"became_strict": became, "rolling_task_reach_rate": rate, "xi": state.xi, "strict": state.strict}

    def xi_map(self) -> dict[str, float]:
        return {name: state.xi for name, state in self.states.items()}

    def state_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "ramp_steps": self.ramp_steps, "probabilities": self.probabilities,
                "states": self.states, "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["stage"] != self.stage: raise ValueError("checkpoint curriculum stage mismatch")
        self.ramp_steps = state["ramp_steps"]; self.probabilities = state["probabilities"]
        self.states = state["states"]; self.rng.bit_generator.state = state["rng_state"]
