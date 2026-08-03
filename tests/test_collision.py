import pytest

from rl_risk_sac.utils.collision import classify_collision_events


def test_any_mode_terminates_on_capsule_overlap() -> None:
    events = classify_collision_events(True, False, "any")

    assert events.collision_any
    assert events.termination_collision
    assert events.termination_reason == "capsule_overlap"


def test_physical_mode_keeps_capsule_overlap_as_nonterminating_event() -> None:
    events = classify_collision_events(True, False, "physical_contact")

    assert events.collision_any
    assert not events.termination_collision
    assert events.termination_reason == ""


def test_physical_contact_is_always_reported_and_can_terminate() -> None:
    events = classify_collision_events(False, True, "physical_contact")

    assert events.pybullet_contact
    assert events.collision_any
    assert events.termination_collision
    assert events.termination_reason == "pybullet_contact"


def test_unknown_collision_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown collision termination mode"):
        classify_collision_events(False, False, "unknown")
