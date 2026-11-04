"""The 0.4 flow: the frame kept in memory, keyword answers, three states,
runnable next steps, export().fit() and report.review(model)."""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

import prefit as pf
from prefit.enums import DecisionStatus
from prefit.fingerprint import FingerprintMismatch


def _ordinal(n=1200, seed=0):
    """An ordered 0-6 score that the task inference reads as classification, with exact duplicates."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    score = np.clip(np.round(3 + X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.7, n)), 0, 6).astype(int)
    df = pd.DataFrame(X, columns=list("abcd")).assign(score=score)
    return pd.concat([df, df.iloc[:100]], ignore_index=True)


def test_four_line_flow():
    df = _ordinal()
    model = HistGradientBoostingRegressor(random_state=0, max_iter=60)
    report = pf.analyze(df, "score", estimator=model, holdout=0.2)
    report = report.answer(task="regression", metric="mae").verify(budget=pf.Budget(n_repeats=1))
    fitted = report.export().fit()
    review = report.review(fitted)
    assert report.task.value == "regression" and report.metric_plan.dev_metric == "neg_mean_absolute_error"
    assert review.n == report.details.profile.holdout["n_holdout"]
    assert any("CV estimate" in c.hypothesis for c in review.checks)
    # change_task kept the holdout rows that analyze set aside under the inferred task
    dev, test = report.holdout()
    first = pf.analyze(df, "score", holdout=0.2)
    assert test.index.equals(first.holdout(df)[1].index)


def test_answer_keywords_and_aliases(imb_df):
    rep = pf.analyze(imb_df, "y")
    assert not any(q.id == "q.task" for q in rep.questions)      # a 0/1 target leaves no doubt: nobody is asked
    a = rep.answer(task="classification", metric="recall")       # confirming it anyway is accepted
    assert a.metric_plan.dev_metric == "average_precision" and not a.metric_plan.dev_metric_provisional
    assert rep.answer({"metric_priority": "miss_costly"}).decisions == rep.answer(metric="miss_costly").decisions
    with pytest.raises(ValueError, match="Options: miss_costly"):
        rep.answer(metric="mae")                                  # a regression metric for a classifier
    with pytest.raises(KeyError, match="This report takes"):
        rep.answer(colour="blue")


def test_column_questions_take_true_false(messy_df):
    rep = pf.analyze(messy_df, "y")
    a = rep.answer(known_at_prediction={"leak": False}, identifier=["customer_id"])
    assert a.decision("d.drop_leak.leak").status is DecisionStatus.CONFIRMED
    assert a.decision("d.drop_id.customer_id").status is DecisionStatus.CONFIRMED
    b = rep.answer(known_at_prediction={"leak": True})
    assert b.decision("d.drop_leak.leak").status is DecisionStatus.DECLINED
    with pytest.raises(KeyError, match="no open question about column 'age'"):
        rep.answer(known_at_prediction={"age": True})


def test_the_suggested_lines_run_to_the_end():
    """Execute what the report says to run, line by line, until only the review is left."""
    df = _ordinal()
    report = pf.analyze(df, "score", estimator=HistGradientBoostingRegressor(random_state=0, max_iter=40),
                        holdout=0.2, budget=pf.Budget(n_repeats=1, max_experiments=2))
    env = {"report": report, "pf": pf}
    for _ in range(10):
        line = env["report"].next_steps()[0]                      # the first open item, every time
        exec(line.split("  #")[0], env)
        if line.startswith("fitted ="):
            review_line = env["report"].next_steps()[-1]
            assert review_line.startswith("review = report.review(fitted)")
            exec(review_line.split("  #")[0], env)
            break
    else:
        pytest.fail("the suggested lines never reached the review")
    assert env["review"].n > 0


def test_three_states():
    df = _ordinal()
    rep = pf.analyze(df, "score", task="regression", estimator=HistGradientBoostingRegressor(max_iter=30))
    states = {d.id: d.state for d in rep.decisions}
    assert states["d.metric"] == "needs_you" and states["d.cv"] == "done"
    assert "needs you" in str(rep) and "Status:" in str(rep)
    v = rep.answer(metric="mae").verify(budget=pf.Budget(n_repeats=1))
    assert all(d.state == "tested" for d in v.decisions if d.experiment_ids)


def test_data_in_memory_is_still_fingerprint_checked():
    df = _ordinal()
    rep = pf.analyze(df, "score", task="regression")
    df.loc[0, "a"] = 99.0                                      # an in-place edit after analyze
    with pytest.raises(FingerprintMismatch):
        rep.verify()
    kept = pf.analyze(_ordinal(), "score", task="regression", keep_data=False)
    with pytest.raises(ValueError, match="keeps none"):
        kept.verify()
    assert '"_data"' not in kept.to_json() and '"_session"' not in kept.to_json()


def test_review_twice_warns_and_uses_the_model_columns():
    df = _ordinal()
    rep = pf.analyze(df, "score", task="regression", holdout=0.2, estimator=HistGradientBoostingRegressor(max_iter=30))
    fitted = rep.export().fit()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rep.review(fitted)
    with pytest.warns(UserWarning, match="reviewed 2 times"):
        rep.answer(metric="mae").review(fitted)                 # later revisions share the session


def test_review_needs_a_test_frame_without_internal_holdout(imb_df):
    rep = pf.analyze(imb_df, "y", estimator=LogisticRegression())
    fitted = rep.export().fit()
    with pytest.raises(ValueError, match="no internal holdout"):
        rep.review(fitted)
    rv = rep.review(fitted, imb_df.iloc[:500])
    assert rv.n == 500


def test_classification_review_gets_scores_for_the_positive_class(imb_df):
    rep = pf.analyze(imb_df, "y", holdout=0.2, estimator=HistGradientBoostingClassifier(max_iter=30))
    rv = rep.review(rep.export().fit())
    assert rv.thresholds  # needs y_score


def test_first_look_and_show_without_ipython(messy_df, capsys):
    rep = pf.analyze(messy_df, "y")
    fl = rep.first_look()
    assert list(fl.columns) == ["column", "kind", "missing %", "distinct", "skew", "signal", "leak score",
                                "decisions"]
    assert "d.drop_leak.leak" in fl.set_index("column").loc["leak", "decisions"]
    rep.show()
    out = capsys.readouterr().out
    assert "FIRST LOOK" in out and "Run next:" in out
    rep.show("d.drop_leak.leak")
    assert "Drop 'leak'" in capsys.readouterr().out


def test_every_color_token_resolves(messy_df):
    from prefit.render.notebook import figure, resolve

    rep = pf.analyze(messy_df, "y")
    for f in rep.details.figures:
        text = repr(resolve(f.plot))
        assert "var:" not in text and "scale:" not in text, f.id
        fig = figure(f)
        assert fig is None or len(fig.data) >= 1


def test_duplicate_gap_figure(messy_df):
    df = pd.concat([messy_df, messy_df.iloc[:200]], ignore_index=True)
    rep = pf.analyze(df, "y")
    assert "fig.duplicate_gaps" in rep.decision("d.duplicate_rows").figure_ids


def test_change_task_accepts_both_argument_orders(imb_df):
    rep = pf.analyze(imb_df, "y")
    assert rep.change_task("regression").task.value == "regression"
    assert rep.change_task(imb_df, "regression").task.value == "regression"


def test_show_in_a_notebook_displays_card_table_and_figures(messy_df, monkeypatch):
    import prefit.render.notebook as nb

    shown = []
    monkeypatch.setattr(nb, "_ipython", lambda: (shown.append, lambda h: ("html", h), lambda m: ("md", m)))
    rep = pf.analyze(messy_df, "y")
    rep.show()
    kinds = [type(x).__name__ if not isinstance(x, tuple) else x[0] for x in shown]
    assert kinds[0] == "html" and "Run next" in shown[0][1]
    assert "DataFrame" in kinds and any(k in ("Figure", "DataFrame") for k in kinds[3:])
    shown.clear()
    rep.show("drop_leak.leak")
    assert shown[0][0] == "md" and "Drop 'leak'" in shown[0][1]


def test_figures_fall_back_to_their_table_without_plotly(messy_df, monkeypatch, capsys):
    import builtins

    real_import = builtins.__import__

    def no_plotly(name, *a, **k):
        if name.startswith("plotly"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_plotly)
    from prefit.render.notebook import figure, figure_table

    rep = pf.analyze(messy_df, "y")
    f = rep.details.figures[0]
    assert figure(f) is None and len(figure_table(f)) > 0
