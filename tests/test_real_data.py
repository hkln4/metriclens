"""Golden expectations on real data (downloaded once into ~/.cache/prefit).

These are the cases that drove 0.3 and 0.4: Wine Quality (duplicates, an ordered
score read as classes) and Telco churn (numbers stored as text, a structural gap).
Skipped when the data cannot be downloaded."""
import pytest

import prefit as pf
from prefit.enums import ColumnKind

pytestmark = [pytest.mark.network, pytest.mark.slow]


def _load(name):
    from real_datasets import load

    try:
        return load(name)
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"{name} not available: {exc}")


def test_wine_red():
    ds = _load("wine_red")
    rep = pf.analyze(ds.frame, ds.target)
    assert rep.task.value == "classification" and rep.details.profile.task_inferred
    assert "Ordered scores are usually regression" in next(q.text for q in rep.questions if q.id == "q.task")
    d = rep.details.profile.dataset
    assert d.duplicate_rows == 240 and d.duplicate_adjacency["p_value"] < 0.05
    assert "d.duplicate_rows" in {x.id for x in rep.decisions}
    reg = rep.answer(task="regression", metric="mae")
    assert reg.task.value == "regression" and reg.metric_plan.dev_metric == "neg_mean_absolute_error"


def test_telco():
    ds = _load("telco")
    rep = pf.analyze(ds.frame, ds.target)
    cols = rep.details.profile.columns
    assert cols["TotalCharges"].kind is ColumnKind.NUMERIC
    assert cols["TotalCharges"].missingness["n_missing"] == 11
    sv = cols["TotalCharges"].missingness["structural"]
    assert (sv["column"], sv["value"], sv["coverage"], sv["precision"]) == ("tenure", 0, 1.0, 1.0)
    ids = {d.id for d in rep.decisions}
    assert "d.fill_zero.TotalCharges" in ids and "d.target_encode" not in ids
    assert cols["customerID"].kind is ColumnKind.ID_LIKE
    assert cols["MonthlyCharges"].kind is ColumnKind.NUMERIC
