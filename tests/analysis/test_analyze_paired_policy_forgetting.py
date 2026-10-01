import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "analyze_paired_policy_forgetting.py"
SPEC = importlib.util.spec_from_file_location("analyze_paired_policy_forgetting", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_exact_mcnemar_is_symmetric_and_handles_no_churn():
    assert MODULE.exact_mcnemar_p_value(0, 0) == 1.0
    assert MODULE.exact_mcnemar_p_value(7, 2) == MODULE.exact_mcnemar_p_value(2, 7)
    assert 0.0 <= MODULE.exact_mcnemar_p_value(7, 2) <= 1.0


def test_load_rows_preserves_checkpoint_order(tmp_path):
    path = tmp_path / "episodes.csv"
    path.write_text(
        "checkpoint,contract_name,episode,safe_success,collision,joint_limit,timeout\n"
        "source,L15,0,1,0,0,0\n"
        "later,L15,0,0,0,0,1\n"
        "source,L14,0,0,0,0,1\n",
        encoding="utf-8",
    )
    order, rows = MODULE.load_rows(path, "L15")
    assert order == ["source", "later"]
    assert rows["source"][0]["safe_success"] == "1"
