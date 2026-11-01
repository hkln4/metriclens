import json
import warnings

import pytest

import metriclens as ml
from metriclens import LLMConfig, Provider


class FakeProvider(Provider):
    model = "fake-1"

    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.response if isinstance(self.response, str) else json.dumps(self.response)


def _good(rep):
    return {
        "summary": "Leakage hint on one column; metric unanswered.",
        "priorities": [
            {"text": "Check the leak column first.", "evidence_ids": ["ev.col.leak.association"],
             "decision_ids": ["d.drop_leak.leak"]},
            {"text": "Invented claim.", "evidence_ids": ["ev.made.up"], "decision_ids": []},
        ],
        "suggested_answers": [{"question_id": "q.metric_priority", "option": "miss_costly", "reason": "screening"},
                              {"question_id": "q.metric_priority", "option": "not-an-option", "reason": "x"}],
        "questions_for_user": ["Is 'leak' recorded after the outcome?"],
        "semantic_flags": [{"column": "leak", "concern": "name suggests post-outcome", "reason": "naming"},
                           {"column": "ghost", "concern": "x", "reason": "y"}],
    }


def test_report_is_complete_without_llm(messy_df):
    rep = ml.analyze(messy_df, "y")
    assert rep.details.llm is None and rep.decisions and rep.metric_plan.primary


def test_llm_claims_must_cite_known_ids(messy_df):
    rep = ml.analyze(messy_df, "y", context="churn screening")
    fake = FakeProvider(_good(rep))
    out = rep.interpret(LLMConfig(provider=fake))
    assert out.details.llm is not None
    assert [c.text for c in out.details.llm.priorities] == ["Check the leak column first."]
    assert [f.column for f in out.details.llm.semantic_flags] == ["leak"]
    assert out.details.llm.dropped_claims == 3
    q = next(q for q in out.questions if q.id == "q.metric_priority")
    assert q.suggestion == "miss_costly" and q.answer is None          # suggested, never applied
    assert out.decision("d.metric").status is rep.decision("d.metric").status
    assert [d.status for d in out.decisions] == [d.status for d in rep.decisions]
    assert out._holdout_positions is not None


def test_payload_never_contains_raw_rows_and_can_hide_names(messy_df):
    rep = ml.analyze(messy_df, "y", context="private context")
    shared = rep.llm_payload()
    assert "free text comment" not in json.dumps(shared, default=str)
    hidden = rep.llm_payload(LLMConfig(share_column_names=False, include_context=False))
    text = json.dumps(hidden, default=str)
    for col in ("customer_id", "leak", "zip", "city"):
        assert f'"{col}"' not in text and f"'{col}'" not in text and f".{col}" not in text
    assert "private context" not in text


def test_aliases_are_mapped_back(messy_df):
    rep = ml.analyze(messy_df, "y")
    cfg = LLMConfig(share_column_names=False)
    payload = rep.llm_payload(cfg)
    leak_alias = f"col_{list(rep.details.profile.columns).index('leak') + 1}"
    assert leak_alias in [c["name"] for c in payload["columns"]]
    dec = next(d["id"] for d in payload["decisions"] if d["rule"] == "leakage_hint")
    fake = FakeProvider({"summary": "s", "priorities": [{"text": "t", "decision_ids": [dec]}],
                         "semantic_flags": [{"column": leak_alias, "concern": "c", "reason": "r"}]})
    out = rep.interpret(LLMConfig(provider=fake, share_column_names=False))
    assert out.details.llm.priorities[0].decision_ids == ("d.drop_leak.leak",)
    assert out.details.llm.semantic_flags[0].column in rep.details.profile.columns


def test_malformed_output_is_rejected(messy_df):
    rep = ml.analyze(messy_df, "y")
    with pytest.warns(UserWarning, match="rejected"):
        out = rep.interpret(LLMConfig(provider=FakeProvider("not json at all")))
    assert out.details.llm is None


def test_provider_failure_does_not_break(messy_df):
    class Boom(Provider):
        def complete(self, s, u):
            raise RuntimeError("network down")

    rep = ml.analyze(messy_df, "y")
    with pytest.warns(UserWarning, match="failed"):
        assert rep.interpret(Boom()).details.llm is None


def test_missing_key_falls_back(messy_df, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rep = ml.analyze(messy_df, "y")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = rep.interpret("anthropic")
    assert out.details.llm is None


def test_llm_section_rendered_and_marked_optional(messy_df):
    rep = ml.analyze(messy_df, "y")
    out = rep.interpret(FakeProvider(_good(rep)))
    h = out.to_html()
    assert "Optional interpretation" in h and "manual review required" in h
    assert h.index("Optional interpretation") < h.index('id="roadmap"')


def test_local_provider_against_a_local_server(messy_df):
    import http.server
    import threading

    from metriclens.llm.local import LocalProvider

    rep = ml.analyze(messy_df, "y")
    answer = json.dumps(_good(rep))
    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = rep.interpret(LLMConfig(provider="local", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                                      model="tiny"))
    finally:
        srv.shutdown()
    assert out.details.llm is not None and out.details.llm.model == "tiny"
    assert seen["body"]["temperature"] == 0
    with pytest.raises(ValueError):
        LocalProvider(base_url="file:///etc/passwd")
