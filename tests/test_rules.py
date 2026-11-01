"""Rule behaviour. The benchmark cases give every rule fire / no-fire / uncertain
coverage; the remaining tests pin the design constraints."""
import numpy as np
import pandas as pd
import pytest
from fixtures import CASES
from pydantic import ValidationError

import metriclens as ml
from metriclens.enums import DecisionClass, DecisionStatus, Severity, Trigger
from metriclens.report.model import Finding
from metriclens.rules.registry import trigger


def _findings_by_rule(rep):
    out = {}
    for f in rep.details.findings:
        out.setdefault(f.rule_id, set()).add(f.trigger.value)
    return out


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_benchmark_case(case):
    rep = ml.analyze(case.make(), case.target, task=case.task, **case.kwargs)
    got = _findings_by_rule(rep)
    for rid in case.must_fire:
        assert "fired" in got.get(rid, set()), f"{rid} should fire"
    for rid in case.must_not_fire:
        assert rid not in got, f"{rid} should not fire (got {got.get(rid)})"
    for rid in case.uncertain:
        assert got.get(rid) == {"uncertain"}, f"{rid} should be near threshold"


def test_trigger_bands():
    assert trigger(5.0, (4.0, 6.0), 3.0) is Trigger.FIRED
    assert trigger(3.2, (2.5, 3.9), 3.0) is Trigger.UNCERTAIN
    assert trigger(2.8, (2.5, 3.1), 3.0) is Trigger.UNCERTAIN
    assert trigger(1.0, (0.5, 1.5), 3.0) is None
    assert trigger(None, None, 3.0) is None


def test_finding_requires_evidence():
    with pytest.raises(ValidationError):
        Finding(id="f.x", rule_id="x", rule_version="1", severity=Severity.INFO, trigger=Trigger.FIRED,
                message="m", evidence_ids=[])


def test_every_finding_and_decision_cites_existing_evidence(messy_df):
    rep = ml.analyze(messy_df, "y")
    for f in rep.details.findings:
        assert set(f.evidence_ids) <= set(rep.details.evidence)
    for d in rep.decisions:
        assert set(d.evidence_ids) <= set(rep.details.evidence)
        assert set(d.finding_ids) <= {f.id for f in rep.details.findings}


def test_procedural_decisions_never_modify_data_content(messy_df):
    rep = ml.analyze(messy_df, "y")
    allowed = {"d.cv", "d.holdout", "d.fit_in_fold", "d.drop_constant"}
    for d in rep.decisions:
        if d.cls is DecisionClass.PROCEDURAL:
            assert d.id in allowed, d.id


def test_no_group_split_without_user_groups():
    from fixtures import duplicates, id_column

    for df in (duplicates(), id_column()):
        rep = ml.analyze(df, "y")
        assert "Group" not in rep.decision("d.cv").step.params["name"]


def test_group_split_with_user_groups(reg_df):
    rep = ml.analyze(reg_df.assign(g=np.arange(len(reg_df)) // 5), "y", groups="g")
    assert rep.decision("d.cv").step.params["name"] == "GroupKFold"
    assert "g" not in rep.details.profile.columns


def test_time_col_switches_to_time_series_split(reg_df):
    rep = ml.analyze(reg_df.assign(t=pd.date_range("2020", periods=len(reg_df), freq="h")), "y", time_col="t")
    assert rep.decision("d.cv").step.params["name"] == "TimeSeriesSplit"
    assert not any(d.id == "d.temporal" for d in rep.decisions)


def test_leakage_is_only_a_hint(messy_df):
    rep = ml.analyze(messy_df, "y")
    f = rep.details.finding("f.leakage_hint.leak")
    assert "hint" in f.message.lower() and "manual review" in f.message.lower()
    assert "detected" not in f.message.lower()
    d = rep.decision("d.drop_leak.leak")
    assert d.cls is DecisionClass.CONTEXTUAL and d.status is DecisionStatus.AWAITING_USER


def test_id_like_is_a_review_question_not_a_verdict():
    from fixtures import id_column

    rep = ml.analyze(id_column(), "y")
    f = rep.details.finding("f.id_like.customer_id")
    assert "not a leakage verdict" in f.message
    assert rep.decision("d.drop_id.customer_id").status is DecisionStatus.AWAITING_USER


def test_duplicates_are_review_items():
    from fixtures import duplicates

    rep = ml.analyze(duplicates(), "y")
    assert rep.decision("d.duplicate_rows").cls is DecisionClass.REVIEW


def test_class_weight_not_applicable_for_estimator_without_it(imb_df):
    from sklearn.neighbors import KNeighborsClassifier

    rep = ml.analyze(imb_df, "y", estimator=KNeighborsClassifier())
    assert rep.decision("d.class_weight").status is DecisionStatus.NOT_APPLICABLE


def test_negative_target_gets_no_log_transform():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=3000)})
    df["y"] = rng.standard_t(2, 3000) * 10
    rep = ml.analyze(df, "y", task="regression")
    d = rep.decision("d.log_target")
    assert d.cls is DecisionClass.REVIEW and d.step is None


def test_findings_are_ordered_critical_first(messy_df):
    rep = ml.analyze(messy_df, "y")
    order = {Severity.CRITICAL: 0, Severity.WARN: 1, Severity.INFO: 2}
    sev = [order[f.severity] for f in rep.details.findings]
    assert sev == sorted(sev)


def test_deterministic(messy_df):
    a = ml.analyze(messy_df, "y").to_json()
    b = ml.analyze(messy_df, "y").to_json()
    assert a == b


def test_datetime_and_text_are_review_items_not_dropped_as_uninformative(messy_df):
    rep = ml.analyze(messy_df, "y")
    d = rep.decision("d.unsupported_columns")
    assert d.cls is DecisionClass.REVIEW and d.step is None
    assert "not a judgement that these columns are uninformative" in d.rationale
    ex = rep.export()
    assert {"ts", "note"} <= {u["column"] for u in ex.unsupported}
    assert "d.unsupported_columns" not in {x["id"] for x in ex.included}


def test_groups_and_time_col_together_are_rejected(reg_df):
    df = reg_df.assign(g=np.arange(len(reg_df)) // 5, t=pd.date_range("2020", periods=len(reg_df), freq="h"))
    with pytest.raises(ValueError, match="not supported in v1"):
        ml.analyze(df, "y", groups="g", time_col="t")


def test_time_split_states_that_stratification_is_off(imb_df):
    df = imb_df.assign(t=pd.date_range("2020", periods=len(imb_df), freq="h"))
    rep = ml.analyze(df, "y", time_col="t")
    assert "stratification is not applied" in rep.decision("d.cv").rationale


def test_leakage_hint_is_named_single_feature():
    from fixtures import leaky_numeric_regression

    rep = ml.analyze(leaky_numeric_regression(), "y", task="regression")
    f = rep.details.finding("f.leakage_hint.leak")
    assert f.message.startswith("Single-feature predictive leakage hint")
    assert "not proof of leakage" in f.message
