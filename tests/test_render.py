import re

import numpy as np

import metriclens as ml


def test_plain_text_has_no_markdown(messy_df):
    text = str(ml.analyze(messy_df, "y"))
    assert "**" not in text and "##" not in text
    for s in ("METRICLENS REPORT", "ROADMAP", "QUESTIONS FOR YOU", "METRICS", "FINDINGS", "LIMITATIONS"):
        assert s in text


def test_markdown(messy_df):
    md = ml.analyze(messy_df, "y").to_markdown()
    assert md.startswith("# metriclens report") and "## Roadmap" in md


def _html(df, **kw):
    return ml.analyze(df, "y", **kw).to_html()


def test_html_sections_and_anchors(messy_df):
    h = _html(messy_df)
    for sec in ("summary", "roadmap", "questions", "metrics", "decisions", "figures", "findings", "evidence",
                "profile", "limitations", "manifest"):
        assert f'id="{sec}"' in h
    ids = set(re.findall(r'id="([^"]+)"', h))
    hrefs = set(re.findall(r'href="#([^"]+)"', h))
    assert hrefs <= ids, sorted(hrefs - ids)


def test_html_contains_no_raw_rows(messy_df):
    df = messy_df.copy()
    secret = [f"SECRET-{i:05d}" for i in range(len(df))]
    df["account"] = secret                                  # id-like strings
    df["zip"] = [f"ZIPSECRET{i % 300}" for i in range(len(df))]  # high-cardinality labels
    h = _html(df)
    assert "SECRET-" not in h and "ZIPSECRET" not in h
    assert "free text comment number" not in h


def test_html_escapes_column_names():
    rng = np.random.default_rng(0)
    import pandas as pd

    df = pd.DataFrame({"<script>alert(1)</script>": rng.normal(size=600), "y": rng.normal(size=600)})
    h = _html(df)
    assert "<script>alert(1)</script>" not in h


def test_html_redaction_hides_small_category_labels():
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"city": rng.choice(["Ankara", "Izmir"], 600), "y": rng.normal(size=600)})
    rep = ml.analyze(df, "y", profile_options=ml.ProfileOptions(redact_values=True))
    assert "Ankara" not in rep.to_html() and "Ankara" not in rep.to_json()


def test_review_section_is_embedded(imb_df):
    rep = ml.analyze(imb_df, "y")
    y = imb_df.y.to_numpy()
    h = rep.to_html(review=ml.review(y, y, rep))
    assert 'id="review"' in h and h.index('id="review"') < h.index('id="limitations"')


def test_plotlyjs_modes(reg_df):
    rep = ml.analyze(reg_df, "y")
    assert "cdn.plot.ly" in rep.to_html()
    assert "cdn.plot.ly" not in rep.to_html(plotlyjs="none")
    assert len(rep.to_html(plotlyjs="inline")) > 1_000_000


def test_to_html_writes_file(tmp_path, reg_df):
    p = tmp_path / "r.html"
    ml.analyze(reg_df, "y").to_html(str(p))
    assert p.read_text().startswith("<!doctype html>")


def test_v0_compat_properties(imb_df):
    rep = ml.analyze(imb_df, "y")
    assert rep.recommended_metrics["primary"]
    assert "accuracy (alone)" in rep.avoid
    assert rep.validation_strategy
