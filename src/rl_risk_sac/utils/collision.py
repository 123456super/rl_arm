from __future__ import annotations

from dataclasses import dataclass


COLLISION_TERMINATION_MODES = {"any", "physical_contact"}


@dataclass(frozen=True)
class CollisionEvents:
    capsule_overlap: bool
    pybullet_contact: bool
    collision_any: bool
    termination_collision: bool
    termination_reason: str


def classify_collision_events(
    capsule_overlap: bool,
    pybullet_contact: bool,
    termination_mode: str,
) -> CollisionEvents:
    if termination_mode not in COLLISION_TERMINATION_MODES:
        raise ValueError(
            f"Unknown collision termination mode {termination_mode!r}; "
            f"expected one of {sorted(COLLISION_TERMINATION_MODES)}"
        )

    collision_any = bool(capsule_overlap or pybullet_contact)
    termination_collision = collision_any if termination_mode == "any" else bool(pybullet_contact)
    if not termination_collision:
        termination_reason = ""
    elif pybullet_contact:
        termination_reason = "pybullet_contact"
    else:
        termination_reason = "capsule_overlap"
    return CollisionEvents(
        capsule_overlap=bool(capsule_overlap),
        pybullet_contact=bool(pybullet_contact),
        collision_any=collision_any,
        termination_collision=termination_collision,
        termination_reason=termination_reason,
    )
