import pytest

import metriclens as ml
from metriclens.enums import DecisionStatus


def test_answer_confirms_contextual_and_export_includes_it(messy_df):
    rep = ml.analyze(messy_df, "y")
    assert any(q.id == "q.leakage.leak" for q in rep.pending_questions)
    rep2 = rep.answer({"leakage.leak": "no"})
    assert rep2.decision("d.drop_leak.leak").status is DecisionStatus.CONFIRMED
    ex = rep2.export()
    assert "leak" not in ex.feature_columns
    assert "leak" in rep.export().feature_columns  # unanswered -> status quo, nothing silently dropped
    assert rep2._holdout_positions is not None


def test_declined_answer_keeps_column(messy_df):
    rep = ml.analyze(messy_df, "y").answer({"leakage.leak": "yes"})
    assert rep.decision("d.drop_leak.leak").status is DecisionStatus.DECLINED
    assert "leak" in rep.export().feature_columns


def test_metric_answer_sets_plan(imb_df):
    rep = ml.analyze(imb_df, "y", answers={"metric_priority": "miss_costly"})
    assert rep.metric_plan.dev_metric == "average_precision"
    assert not rep.metric_plan.dev_metric_provisional
    assert rep.decision("d.metric").status is DecisionStatus.CONFIRMED
    assert "accuracy (alone)" in rep.metric_plan.avoid


def test_unknown_question_and_option(messy_df):
    rep = ml.analyze(messy_df, "y")
    with pytest.raises(KeyError):
        rep.answer({"nope": "x"})
    with pytest.raises(ValueError):
        rep.answer({"metric_priority": "whatever"})


def test_changing_the_task_requires_rerun(messy_df):
    rep = ml.analyze(messy_df, "y")
    with pytest.raises(ValueError, match="re-run"):
        rep.answer({"task": "regression"})


def test_answers_do_not_mutate_original(messy_df):
    rep = ml.analyze(messy_df, "y")
    rep.answer({"metric_priority": "symmetric"})
    assert rep.decision("d.metric").status is DecisionStatus.AWAITING_USER
