from __future__ import annotations

import json

from scripts.smoke_test import main
from scripts.train import git_revision


def test_smoke_test_script_runs() -> None:
    main([])


def test_p3_filter_smoke_test_checks_predictive_speed_diagnostics() -> None:
    main(["--config", "configs/experiments/p3/b4_predictive_nonrobust_filter.yaml"])


def test_p3_current_risk_smoke_test_checks_predictive_speed_diagnostics() -> None:
    main(["--config", "configs/experiments/p3/b2_link_current.yaml"])


def test_training_config_can_record_git_revision() -> None:
    revision = git_revision()
    assert revision is None or len(revision) == 40
    json.dumps({"git_revision": revision})
