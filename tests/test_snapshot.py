"""Golden snapshots of the decision structure (ids, classes, statuses, finding
ids). Numeric values are excluded so the snapshot is stable across numpy/pandas
versions. Regenerate with: UPDATE_GOLDEN=1 pytest tests/test_snapshot.py"""
import json
import os
from pathlib import Path

import pytest
from fixtures import CASES

import metriclens as ml

GOLDEN = Path(__file__).parent / "golden"
NAMES = ["imbalanced", "skewed_target", "leaky_categorical_classification", "datetime_column", "high_missing"]


def _snapshot(rep):
    return {
        "task": rep.task.value,
        "findings": [[f.id, f.severity.value, f.trigger.value] for f in rep.findings],
        "decisions": [[d.id, d.cls.value, d.status.value, d.stage] for d in rep.decisions],
        "questions": [q.id for q in rep.questions],
        "metric_plan": [rep.metric_plan.primary, rep.metric_plan.dev_metric],
        "figures": [f.id for f in rep.figures],
        "roadmap": [[s.stage, s.decision_ids] for s in rep.roadmap],
    }


@pytest.mark.parametrize("name", NAMES)
def test_golden(name):
    case = next(c for c in CASES if c.name == name)
    snap = _snapshot(ml.analyze(case.make(), case.target, task=case.task))
    path = GOLDEN / f"{name}.json"
    if os.environ.get("UPDATE_GOLDEN") or not path.exists():
        path.write_text(json.dumps(snap, indent=1) + "\n")
    assert json.loads(path.read_text()) == json.loads(json.dumps(snap))
