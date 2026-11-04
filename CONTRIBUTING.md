# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q -n auto              # all tests in parallel (~20 s); -m "not slow" skips simulations,
                               # timing and real-data tests (those download into ~/.cache/prefit)
ruff check . && mypy prefit
python benchmarks/run.py       # refreshes benchmarks/results and the track record
python benchmarks/check_manifest.py   # what CI checks: results match the shipped rule versions
python benchmarks/real_data.py # plain flow vs prefit on real data (~15 min, downloads once)
```

## Adding or changing a rule

1. Write the rule in `prefit/rules/catalog.py` with
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
6. Bump the rule's version when its thresholds, logic or emitted text change,
   and bump `MANIFEST_VERSION` when rules, cases or expectations change. Re-run
   `python benchmarks/run.py` and commit the results, including failures. CI
   (`benchmarks/check_manifest.py`) fails when the committed results were
   produced by other rule versions.
7. Give every conditional decision its `alternatives` (what else, and when it
   is right), and every question a `keyword` (and `column` for per-column
   questions, with `flag` on the options) so `answer(**kw)` and the suggested
   lines work.
8. If decision structure changes, refresh the golden files:
   `UPDATE_GOLDEN=1 pytest tests/test_snapshot.py`, and review the diff.

## Versioning

Semantic versioning; before 1.0, minor versions may change the API. Every
release gets a CHANGELOG entry. Rule versions are independent and appear in
every report's manifest.

## Pull requests

Keep the deterministic core free of new hard dependencies. Tests, ruff and mypy
must pass; coverage must stay above the configured floor.

## Releasing

Publishing runs in GitHub Actions with PyPI Trusted Publishing (no tokens).

1. Bump `version` in `pyproject.toml` and `prefit/_version.py`, add a CHANGELOG entry, merge to `main`.
2. Optional dry run: Actions → Release → Run workflow → `target=testpypi`, then
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ prefit`.
3. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`. The workflow checks that the tag matches the
   package version, builds, runs `twine check` and a wheel smoke test, then publishes to PyPI.
