"""Local learner scheduling, separate from the captured-start task contract."""
from __future__ import annotations

import math


def local_training_schedule(config, *, validation_mode=False):
    local, sac, train = config["local_avoidance"], config["sac"], config["train"]
    schedule = dict(format_version=1,
                    batch_size=8 if validation_mode else sac["batch_size"],
                    updates_per_transition=float(sac.get("updates_per_transition", 1.)),
                    update_after=8 if validation_mode else local["update_after"],
                    warmup_steps=0 if validation_mode else local["warmup_steps"],
                    diagnostic_interval=train.get("update_log_interval", 10))
    for name in ("batch_size", "update_after", "diagnostic_interval"):
        value = schedule[name]
        if int(value) != value or value < 1:
            raise ValueError(f"local training {name} must be a positive integer")
        schedule[name] = int(value)
    rate = schedule["updates_per_transition"]
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("sac.updates_per_transition must be finite and positive")
    return schedule


def legacy_local_training_schedule(checkpoint):
    """Before schedule metadata, local fields overrode the SAC batch/UTD."""
    local = checkpoint["config"]["local_avoidance"]
    validation = checkpoint["validation_mode"]
    return dict(format_version=1, batch_size=8 if validation else local["batch_size"],
                updates_per_transition=float(local["updates_per_step"]),
                update_after=8 if validation else local["update_after"],
                warmup_steps=0 if validation else local["warmup_steps"], diagnostic_interval=1)


def sac_resume_contract(config):
    """Batch and update frequency may migrate; model/optimizer math stays strict."""
    return {key: value for key, value in config["sac"].items()
            if key not in ("batch_size", "updates_per_transition")}


def schedule_migrations(checkpoint, schedule):
    history = list(checkpoint.get("training_schedule_migrations", [])) if checkpoint else []
    if checkpoint:
        previous = checkpoint.get("training_schedule") or legacy_local_training_schedule(checkpoint)
        if previous != schedule:
            history.append(dict(step=checkpoint["counters"]["steps"],
                                previous=previous, current=dict(schedule)))
    return history
