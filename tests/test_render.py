import re

import numpy as np

import prefit as pf


def test_plain_text_has_no_markdown(messy_df):
    text = str(pf.analyze(messy_df, "y"))
    assert "**" not in text and "##" not in text
    for s in ("PREFIT REPORT", "Next step:", "ROADMAP", "QUESTIONS FOR YOU", "METRICS", "LIMITATIONS"):
        assert s in text


def test_markdown(messy_df):
    md = pf.analyze(messy_df, "y").to_markdown()
    assert md.startswith("# prefit report") and "## Roadmap" in md


def _html(df, **kw):
    return pf.analyze(df, "y", **kw).to_html()


def test_html_sections_and_anchors(messy_df):
    h = _html(messy_df)
    for sec in ("summary", "actions", "roadmap", "details", "findings", "evidence", "profile", "manifest"):
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
    rep = pf.analyze(df, "y", profile_options=pf.ProfileOptions(redact_values=True))
    assert "Ankara" not in rep.to_html() and "Ankara" not in rep.to_json()


def test_review_section_is_embedded(imb_df):
    rep = pf.analyze(imb_df, "y")
    y = imb_df.y.to_numpy()
    h = rep.to_html(review=pf.review(y, y, rep))
    assert 'id="review"' in h and h.index('id="review"') < h.index('id="details"')


def test_plotlyjs_modes(reg_df):
    rep = pf.analyze(reg_df, "y")
    assert "cdn.plot.ly" in rep.to_html()
    assert "cdn.plot.ly" not in rep.to_html(plotlyjs="none")
    assert len(rep.to_html(plotlyjs="inline")) > 1_000_000


def test_to_html_writes_file(tmp_path, reg_df):
    p = tmp_path / "r.html"
    pf.analyze(reg_df, "y").to_html(str(p))
    assert p.read_text().startswith("<!doctype html>")


# ---- HTML structure snapshot ------------------------------------------------ #

def _cards(h):
    return re.findall(r'<article class="card" id="([^"]+)" data-decision="([^"]+)">(.*?)</article>', h, re.S)


def test_navigation_has_only_four_sections(messy_df):
    h = pf.analyze(messy_df, "y").to_html()
    nav = re.search(r'<nav aria-label="sections">(.*?)</nav>', h, re.S).group(1)
    assert re.findall(r">([^<]+)</a>", nav) == ["Summary", "Actions", "Roadmap", "Details"]


def test_short_card_is_open_and_details_are_closed(messy_df):
    h = pf.analyze(messy_df, "y").to_html()
    cards = _cards(h)
    assert cards
    for cid, did, body in cards:
        assert cid == did
        head = body.split("<details", 1)[0]
        assert 'class="why"' in head and 'class="do"' in head      # short card visible by default
        assert "<table" not in head                                 # evidence tables are not open
        for det in re.findall(r"<details[^>]*>", body):
            assert " open" not in det
    summaries = set(re.findall(r"<summary>([^<]+)</summary>", h))
    assert {"Show evidence", "Show rule and benchmark details", "Show related figures"} <= summaries


def test_experiment_details_are_collapsed(imb_df):
    h = pf.analyze(imb_df, "y", verify=True, budget=pf.Budget(max_experiments=1)).to_html()
    assert "<summary>Show experiment details</summary>" in h
    card = next(b for c, _, b in _cards(h) if c == "d.class_weight")
    assert 'class="exp"' in card.split("<details", 1)[0]          # one-line summary stays open


def test_details_sections_are_collapsed_at_the_end(messy_df):
    h = pf.analyze(messy_df, "y").to_html()
    for sec in ("all-diagnostics", "findings", "evidence", "profile", "manifest"):
        tag = re.search(rf'<details id="{sec}"[^>]*>', h).group(0)
        assert " open" not in tag
        assert h.index(tag) > h.index('id="roadmap"')
    assert h.index('id="details"') > h.index('id="roadmap"') > h.index('id="actions"') > h.index('id="summary"')


def test_at_most_one_main_figure_per_card_bound_to_its_decision(messy_df):
    h = pf.analyze(messy_df, "y").to_html()
    shown = []
    for cid, _, body in _cards(h):
        plots = re.findall(r'<div class="plot" data-fig="([^"]+)" data-decision="([^"]+)"', body)
        assert len(plots) <= 1
        for fig, did in plots:
            assert did == cid
            shown.append(fig)
    assert shown and len(shown) == len(set(shown))                  # a figure is inlined once


def test_roadmap_stages_are_numbered_consecutively(messy_df):
    rep = pf.analyze(messy_df, "y")
    nums = [int(s.title.split(".")[0]) for s in rep.roadmap]
    assert nums == list(range(1, len(nums) + 1))


def test_first_screen_has_the_essentials(messy_df):
    h = pf.analyze(messy_df, "y").to_html()
    summary = h[h.index('id="summary"'):h.index('id="actions"')]
    for s in ("development rows", "questions awaiting your answer", "critical or review decisions", "Next step:"):
        assert s in summary
