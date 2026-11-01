# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # tests (includes the benchmark cases)
ruff check . && mypy metriclens
python benchmarks/run.py       # refreshes benchmarks/results and the track record
```

## Adding or changing a rule

1. Write the rule in `metriclens/rules/catalog.py` with
   `@rule("id", "1", thresholds={...})`. Compare statistics with `trigger()` and
   pass the statistic's interval so near-threshold cases become `uncertain`.
2. Choose the decision class honestly:
   - procedural only if it is about *how to evaluate* and is right for any data;
   - anything that changes the data's content is review, conditional or contextual;
   - anything that needs knowledge outside the data is contextual (ask a question).
3. Never phrase a hint as a verdict ("leakage hint — manual review required",
   not "leakage detected").
4. If the decision has a pipeline step, make sure `export.build_model` supports
   its `Step.kind`.
5. Add benchmark cases in `benchmarks/fixtures.py`: one where it must fire, one
   where it must not, and a near-threshold one where that is meaningful.
6. Bump the rule's version when its thresholds or logic change, and bump
   `MANIFEST_VERSION` when cases or expectations change. Re-run
   `python benchmarks/run.py` and commit the results, including failures.
7. If decision structure changes, refresh the golden files:
   `UPDATE_GOLDEN=1 pytest tests/test_snapshot.py`, and review the diff.

## Versioning

Semantic versioning; before 1.0, minor versions may change the API. Every
release gets a CHANGELOG entry. Rule versions are independent and appear in
every report's manifest.

## Pull requests

Keep the deterministic core free of new hard dependencies. Tests, ruff and mypy
must pass; coverage must stay above the configured floor.
