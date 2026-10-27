"""Optional CSV diagnostics without creating disabled log files."""
from __future__ import annotations

import csv
from pathlib import Path


PROBE_DETAIL_FIELDS = frozenset(("seed_results", "obstacle_episodes"))
LEARNING_DIAGNOSTIC_FIELDS = (
    "alpha_loss", "policy_entropy", "critic_gradient_norm", "actor_gradient_norm",
    "actor_updated",
)


def detailed_log_enabled(config, name):
    return bool(config.get("train", {}).get("logging", {}).get(name, False))


def learning_diagnostics(metrics):
    return {key: metrics.get(key, "") for key in LEARNING_DIAGNOSTIC_FIELDS}


def record_online_success(counters, success):
    outcomes = counters.setdefault("recent_online_successes", [])
    outcomes.append(bool(success))
    del outcomes[:-100]


def format_training_progress(step, total, elapsed_s, successes, *, label="训练", completed_steps=None):
    """One flushed terminal line; success uses completed online episodes only."""
    successes = list(successes)[-100:]
    rate = f"{sum(successes) / len(successes):.1%}" if successes else "--"
    fraction = min(1., max(0., step / total))
    filled = int(fraction * 20)
    bar = "#" * filled + "-" * (20 - filled)
    completed = step if completed_steps is None else completed_steps
    speed = completed / max(elapsed_s, 1e-9)
    if speed > 0:
        seconds = int(max(0, total - step) / speed)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        eta = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        eta = "--"
    return (f"{label} [{bar}] {step:,}/{total:,} ({fraction:.2%}) | "
            f"在线成功率(最近{len(successes)}回合)={rate} | "
            f"速度={speed:.1f}步/秒 | 剩余={eta}")


class CsvLog:
    """CSV writer/context manager; disabled diagnostics perform no file I/O."""

    def __init__(self, path, fieldnames=None, *, enabled=True, exclude=()):
        self.enabled = enabled
        self.exclude = frozenset(exclude)
        self.handle = Path(path).open("w", newline="", encoding="utf-8") if enabled else None
        self.writer = None
        self.header_written = False
        if fieldnames is not None and enabled:
            self.writer = csv.DictWriter(self.handle, fieldnames=[
                key for key in fieldnames if key not in self.exclude])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def writeheader(self):
        if self.writer is not None and not self.header_written:
            self.writer.writeheader()
            self.header_written = True

    def writerow(self, row):
        if not self.enabled:
            return
        row = {key: value for key, value in row.items() if key not in self.exclude}
        if self.writer is None:
            self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
        self.writeheader()
        self.writer.writerow(row)

    def flush(self):
        if self.handle is not None:
            self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close()
