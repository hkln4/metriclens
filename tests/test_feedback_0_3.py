"""0.3.0: equivalence, test-score intervals, task framing, duplicate evidence,
review notes, change_task, cv_estimate, standard practice, LLM audit fields."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import prefit as pf
from prefit.enums import DecisionStatus
from prefit.verify import classify_outcome, equivalence_test, status_from_experiments


def _ordinal_score(n=1200, seed=0):
    """An integer score 3..8 (inferred as classification) with adjacent exact copies."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.lognormal(0, 1.2, n)})
    df["score"] = np.clip(np.rint(5.5 + df.a + 0.5 * rng.normal(size=n)), 3, 8).astype(int)
    copies = np.sort(rng.choice(n, 150, replace=False))
    rows = []
    for i in range(n):  # each copy right after its original, as a re-entered record would be
        rows.append(i)
        if i in set(copies):
            rows.append(i)
    return df.iloc[rows].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 1. equivalence (TOST)
# --------------------------------------------------------------------------- #

def test_equivalence_test_separates_absence_from_noise():
    base = np.array([0.50, 0.51, 0.49, 0.50, 0.52, 0.50, 0.51, 0.49, 0.50, 0.51])
    tight = base + np.array([1, -1, 1, -1, 0, 1, -1, 0, 1, -1]) * 1e-4
    noisy = base + np.array([3, -4, 5, -3, 4, -5, 3, -4, 5, -4]) * 1e-2
    assert equivalence_test(base, tight, 0.25, 0.005) < 0.01
    assert equivalence_test(base, noisy, 0.25, 0.005) > 0.05
    assert equivalence_test(base, tight, 0.25, 0.0) == 1.0
    assert classify_outcome(0.0001, 0.9, 0.005, 0.05, p_equivalence=0.01) == "no_practical_effect"
    assert classify_outcome(0.0001, 0.9, 0.005, 0.05, p_equivalence=0.4) == "no_difference"
    assert classify_outcome(0.02, 0.01, 0.005, 0.05, p_equivalence=0.9) == "improved"


def test_no_practical_effect_status_and_export_reason():
    df = _ordinal_score()
    rep = pf.analyze(df, "score", task="regression", answers={"metric_priority": "typical_error"})
    rep = rep.verify(df, estimator=RandomForestRegressor(n_estimators=60, random_state=0, n_jobs=-1),
                     budget=pf.Budget(n_repeats=3))
    e = rep.experiments[0]
    assert e.p_equivalence is not None
    d = rep.decision("d.power_transform")
    assert e.outcome == "no_practical_effect" and e.p_equivalence < 0.05 <= e.p_adjusted
    assert d.status is DecisionStatus.NO_PRACTICAL_EFFECT and "Equivalent" in d.note
    reason = next(x for x in rep.export().excluded if x["id"] == "d.power_transform")["reason"]
    assert "no practical effect" in reason
    # natural units: an error metric reads "lower = better"
    assert "lower = better" in rep.to_text() and "MAE change" not in rep.to_text()


def test_status_from_experiments_for_proxies():
    from prefit.report.model import Experiment

    def exp(fam, outcome):
        return Experiment(id=f"exp.x.{fam}", decision_id="d.x", family=fam, estimator="E", metric="r2", n_rows=10,
                          n_splits=5, n_repeats=1, baseline_scores=[0.0], candidate_scores=[0.0], mean_diff=0.0,
                          outcome=outcome)

    d = pf.analyze(pd.DataFrame({"a": np.arange(300.0), "y": np.arange(300.0)}), "y").decisions[0]
    st, _ = status_from_experiments(d, [exp("linear_proxy", "no_practical_effect"),
                                        exp("tree_proxy", "no_practical_effect")], user=False)
    assert st is DecisionStatus.NO_PRACTICAL_EFFECT
    st, note = status_from_experiments(d, [exp("user", "no_difference")], user=True)
    assert st is DecisionStatus.INCONCLUSIVE and "equivalence" in note


# --------------------------------------------------------------------------- #
# 2. intervals on the test score, CV consistency
# --------------------------------------------------------------------------- #

def test_review_intervals_and_cv_check():
    df = _ordinal_score()
    dev, test = pf.split(df, "score", random_state=1)
    est = RandomForestRegressor(n_estimators=60, random_state=0, n_jobs=-1)
    rep = pf.analyze(dev, "score", task="regression", answers={"metric_priority": "typical_error"}, estimator=est)
    rep = rep.verify(dev, budget=pf.Budget(max_experiments=0))
    cv = rep.cv_estimate()
    assert cv["label"] == "MAE" and cv["mean"] > 0 and cv["family"] == "user"
    feats = [c for c in df.columns if c != "score"]
    pred = est.fit(dev[feats], dev.score).predict(test[feats])
    rv = pf.review(test.score, pred, rep, X=test[feats])
    lo, hi = rv.intervals["MAE"]
    assert lo <= rv.metrics["MAE"] <= hi and set(rv.intervals) <= set(rv.metrics)
    check = next(c for c in rv.checks if "CV estimate" in c.hypothesis)
    assert check.held is (lo <= cv["mean"] <= hi)
    assert "duplicates resampled together" in " ".join(rv.notes)
    assert "95% CI" in rv.to_text() and "95% interval" in rv.to_html()
    off = pf.review(test.score, pred + 2.0, rep)
    assert next(c for c in off.checks if "CV estimate" in c.hypothesis).held is False


def test_review_intervals_for_classification_and_opt_out(imb_df):
    rng = np.random.default_rng(0)
    y = imb_df.y.to_numpy()
    pred = np.where(rng.random(len(y)) < 0.9, y, 1 - y)
    rv = pf.review(y, pred, task="classification")
    assert {"accuracy", "balanced_accuracy", "macro_f1"} <= set(rv.intervals)
    assert pf.review(y, pred, task="classification", n_bootstrap=0).intervals == {}


def test_cv_estimate_is_none_before_verify(imb_df):
    assert pf.analyze(imb_df, "y").cv_estimate() is None


# --------------------------------------------------------------------------- #
# 3. an unconfirmed task frames nothing silently
# --------------------------------------------------------------------------- #

def test_task_dependent_items_are_flagged_while_task_is_open():
    df = _ordinal_score()
    rep = pf.analyze(df, "score")
    assert rep.task.value == "classification" and rep.details.profile.task_inferred
    assert rep.details.finding("f.class_imbalance").task_dependent
    assert rep.decision("d.class_weight").task_dependent and not rep.decision("d.duplicate_rows").task_dependent
    pl = rep.llm_payload()
    assert pl["task_confirmed"] is False
    assert next(f for f in pl["findings"] if f["rule"] == "class_imbalance")["task_dependent"] is True
    assert "task_dependent" not in next(f for f in pl["findings"] if f["rule"] == "duplicate_rows")
    assert "Assumes the inferred task" in rep.to_text()
    q = rep.questions[0]
    assert "Ordered scores are usually regression" in q.text
    assert "ordered score" in next(o.label for o in q.options if o.value == "regression")
    confirmed = rep.answer({"task": "classification"})
    assert confirmed.llm_payload()["task_confirmed"] is True
    assert "Assumes the inferred task" not in confirmed.to_text()


def test_change_task_keeps_what_still_applies():
    df = _ordinal_score()
    dev, test = pf.split(df, "score", random_state=0)
    rep = pf.analyze(dev, "score", holdout=None)
    rep = rep.register_external_holdout(test, dev_df=dev, name="test set")
    rep = rep.resolve("d.duplicate_rows", "re-entered records; keep them, the split keeps copies together")
    with pytest.raises(ValueError, match="change_task"):
        rep.answer({"task": "regression"})
    reg = rep.change_task(dev, "regression")
    assert reg.task.value == "regression" and not reg.details.profile.task_inferred
    assert reg.details.manifest.external_holdout["name"] == "test set"
    assert reg.decision("d.duplicate_rows").status is DecisionStatus.REVIEWED
    h = reg.details.manifest.history[-1]
    assert h["action"] == "change_task" and h["from"] == "classification"
    assert reg.details.manifest.revision == rep.details.manifest.revision + 1
    assert not any(d.id == "d.class_weight" for d in reg.decisions)
    from prefit.fingerprint import FingerprintMismatch

    with pytest.raises(FingerprintMismatch):
        rep.change_task(test, "regression")


# --------------------------------------------------------------------------- #
# 5. duplicate evidence
# --------------------------------------------------------------------------- #

def test_duplicate_groups_and_adjacency_evidence():
    rep = pf.analyze(_ordinal_score(), "score", task="regression")
    ds = rep.details.profile.dataset
    assert ds.duplicate_rows == 150 and ds.duplicate_groups == 150 and ds.duplicate_rows_in_groups == 300
    assert ds.duplicate_adjacency["median_gap"] == 1 and ds.duplicate_adjacency["p_value"] < 0.05
    d = rep.decision("d.duplicate_rows")
    assert "sit close together" in d.action and "report.resolve" in d.action
    assert "300 rows in 150 groups" in rep.details.finding("f.duplicate_rows").message


def test_appended_copies_give_no_order_hint():
    from fixtures import duplicates

    rep = pf.analyze(duplicates(), "y")
    adj = rep.details.profile.dataset.duplicate_adjacency
    assert adj["p_value"] > 0.5
    assert "row order gives no hint" in rep.decision("d.duplicate_rows").action


# --------------------------------------------------------------------------- #
# 6. review items closed with a note
# --------------------------------------------------------------------------- #

def test_resolve_closes_a_review_item_and_survives_later_steps(tmp_path):
    from fixtures import duplicates

    df = duplicates()
    rep = pf.analyze(df, "y")
    r1 = rep.resolve("duplicate_rows", "same records entered twice; dropping them upstream")
    d = r1.decision("d.duplicate_rows")
    assert d.status is DecisionStatus.REVIEWED and d.review_note.startswith("same records")
    assert rep.decision("d.duplicate_rows").status is DecisionStatus.NEEDS_REVIEW   # old revision untouched
    assert r1.details.manifest.history[-1]["action"] == "resolve"
    r2 = r1.answer({"metric_priority": "symmetric"}).verify(df, budget=pf.Budget(max_experiments=0))
    assert r2.decision("d.duplicate_rows").status is DecisionStatus.REVIEWED
    ex = r2.export()
    assert not any("duplicates" in a for a in ex.assumptions)
    assert ex.reviewed[0]["id"] == "d.duplicate_rows" and "review items you closed" in ex.describe()
    assert "your note: same records" in r2.to_text()
    rv = pf.review(df.y, df.y, r2)
    assert rv.reviewed[0]["note"].startswith("same records")
    assert not any(u["id"] == "d.duplicate_rows" for u in rv.unverified)
    path = tmp_path / "r.json"
    r2.to_json(str(path))
    assert pf.load_report(str(path)).decision("d.duplicate_rows").status is DecisionStatus.REVIEWED


def test_resolve_rejects_non_review_items_and_empty_notes():
    from fixtures import duplicates

    rep = pf.analyze(duplicates(), "y")
    with pytest.raises(ValueError, match="only review items"):
        rep.resolve("d.cv", "ok")
    with pytest.raises(ValueError, match="needs a note"):
        rep.resolve("d.duplicate_rows", "  ")
    with pytest.raises(KeyError, match="report.resolve"):
        rep.answer({"duplicate_rows": "keep"})


# --------------------------------------------------------------------------- #
# 4 + 10. text fixes and standard practice
# --------------------------------------------------------------------------- #

def test_report_text_fixes(imb_df):
    from prefit.render.common import partition_note

    dev, test = pf.split(imb_df, "y")
    rep = pf.analyze(dev, "y").register_external_holdout(test, name="test set")
    assert "the test set (" in partition_note(rep) and "an test" not in partition_note(rep)
    near = pd.DataFrame({"a": np.random.default_rng(0).lognormal(0, 1.5, 3000)})
    near["b"] = np.random.default_rng(1).gamma(1.0, size=3000)        # skew ~2: interval straddles 2
    near["y"] = near.a + np.random.default_rng(2).normal(size=3000)
    rep = pf.analyze(near, "y", task="regression")
    d = rep.decision("d.power_transform")
    assert "across their whole interval" in d.why_short
    assert "1 skewed + 1 near-threshold" in d.title
    assert "and 1 more has an interval that straddles 2" in d.why_short


def test_baseline_row_names_the_dummy(reg_df, imb_df):
    reg = pf.analyze(reg_df, "y", verify=True, budget=pf.Budget(max_experiments=0)).to_html()
    cls = pf.analyze(imb_df, "y", verify=True, budget=pf.Budget(max_experiments=0)).to_html()
    assert "prior / median" not in reg + cls
    assert ">median<" in reg and ">class prior<" in cls


def test_standard_practice_is_grouped(reg_df):
    from fixtures import duplicates

    rep = pf.analyze(reg_df, "y")
    std = {d.id for d in rep.decisions if d.standard}
    assert {"d.fit_in_fold", "d.holdout", "d.cv"} <= std
    assert "standard practice (applied on every dataset)" in rep.to_text()
    assert "Standard practice (" in rep.to_html()
    assert not pf.analyze(duplicates(), "y").decision("d.cv").standard   # duplicate-aware CV is data-specific


# --------------------------------------------------------------------------- #
# 9. LLM audit trail
# --------------------------------------------------------------------------- #

def test_llm_keeps_raw_response_and_what_was_dropped(messy_df):
    from test_llm import FakeProvider, _good

    rep = pf.analyze(messy_df, "y", context="churn screening")
    resp = _good(rep)
    out = rep.interpret(pf.LLMConfig(provider=FakeProvider(resp))).details.llm
    assert json.loads(out.raw_response) == resp
    assert out.dropped_claims == len(out.dropped) == 3
    assert any("unknown ids" in x for x in out.dropped)
    assert any("not an option" in x for x in out.dropped)
    assert any("unknown column 'ghost'" in x for x in out.dropped)
    system = FakeProvider(resp)
    rep.interpret(pf.LLMConfig(provider=system))
    assert "task_dependent" in system.calls[0][0]
