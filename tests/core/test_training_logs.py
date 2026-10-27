from __future__ import annotations

import csv
import json

from rl_risk_sac.utils.training_logs import (
    CsvLog, PROBE_DETAIL_FIELDS, detailed_log_enabled, format_training_progress, record_online_success,
)
from scripts.core.train_thesis_homotopy import ORIENTATION_PROBE_FIELDS, scalar_probe_metrics


def test_disabled_diagnostic_does_not_create_file(tmp_path):
    path = tmp_path / "updates.csv"
    with CsvLog(path, ["steps"], enabled=False) as log:
        log.writeheader()
        log.writerow({"steps": 1000})
        log.flush()
    assert not path.exists()
    assert not detailed_log_enabled({}, "updates")


def test_opt_in_diagnostic_remains_readable_and_flushes(tmp_path):
    config = {"train": {"logging": {"updates": True}}}
    path = tmp_path / "updates.csv"
    with CsvLog(path, enabled=detailed_log_enabled(config, "updates")) as log:
        log.writerow({"steps": 1000, "actor_loss": -.25})
        log.flush()
        with path.open() as handle:
            assert list(csv.DictReader(handle)) == [{"steps": "1000", "actor_loss": "-0.25"}]


def test_compact_probe_preserves_results_and_skips_nested_detail_serialization(tmp_path):
    # These details cannot be JSON encoded; compact output must never encode them.
    metrics = dict(success_rate=.75, collision_rate=.1, timeout_rate=.15,
                   seed_success_rates=[.7, .8],
                   obstacle_groups={"count/2": {"success_rate": .75, "episodes": 4}},
                   seed_results=object(), obstacle_episodes=object())
    row = scalar_probe_metrics(metrics, include_details=False)
    path = tmp_path / "orientation_probes.csv"
    with CsvLog(path, ORIENTATION_PROBE_FIELDS, exclude=PROBE_DETAIL_FIELDS) as log:
        log.writeheader()
        log.writerow(row)
    with path.open() as handle:
        reader = csv.DictReader(handle)
        saved = next(reader)
        assert PROBE_DETAIL_FIELDS.isdisjoint(reader.fieldnames)
    assert saved["success_rate"] == "0.75"
    assert saved["seed_success_rates"] == "[0.7, 0.8]"
    assert json.loads(saved["obstacle_groups"]) == metrics["obstacle_groups"]
    assert metrics["seed_results"] is not None


def test_detailed_probe_can_be_enabled_explicitly(tmp_path):
    metrics = dict(success_rate=1., seed_results=[{"success_rate": 1.}],
                   obstacle_episodes=[{"episode": 0, "timings": [{"action_execution_s": .05}]}])
    path = tmp_path / "orientation_probes.csv"
    with CsvLog(path, ORIENTATION_PROBE_FIELDS) as log:
        log.writerow(scalar_probe_metrics(metrics, include_details=True))
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["obstacle_episodes"]) == metrics["obstacle_episodes"]
    assert row["seed_results"]


def test_online_success_window_keeps_latest_100_outcomes_and_resumes():
    counters = {"recent_online_successes": [False] * 100}
    for _ in range(57):
        record_online_success(counters, True)
    outcomes = counters["recent_online_successes"]
    assert len(outcomes) == 100
    assert sum(outcomes) == 57
    line = format_training_progress(1000, 1000000, 100., outcomes, label="越障训练")
    assert "1,000/1,000,000 (0.10%)" in line
    assert "在线成功率(最近100回合)=57.0%" in line
    assert "速度=10.0步/秒" in line
    assert "剩余=27:45:00" in line


def test_progress_reports_empty_window_and_resume_speed_without_counting_old_steps():
    initial = format_training_progress(10000, 1000000, 0., [], completed_steps=0)
    assert "在线成功率(最近0回合)=--" in initial
    assert "剩余=--" in initial
    resumed = format_training_progress(11000, 1000000, 100., [True, False], completed_steps=1000)
    assert "在线成功率(最近2回合)=50.0%" in resumed
    assert "速度=10.0步/秒" in resumed
