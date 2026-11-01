"""Optional LLM interpretation, constrained to the computed report.

The LLM can: summarize, prioritize existing decisions, suggest answers to open
questions, suggest further questions, and raise *semantic* flags (e.g. a column
name that sounds like it is recorded after the outcome). It cannot: compute or
invent statistics, create findings, or change any decision. Every claim must
cite ids that exist in the payload; claims that cite unknown ids, and flags on
unknown columns, are dropped and counted. Suggestions are stored next to the
question and applied only through `Report.answer()`.
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..frozen import evolve
from ..report.model import LLMClaim, LLMInterpretation, SemanticFlag
from .config import LLMConfig, coerce_config
from .payload import build_payload, disclosure
from .provider import Provider

SYSTEM = """You review a PRECOMPUTED, deterministic data-analysis report for a supervised ML project.
Every number was computed by code and is correct. Rules:
- Never invent, recompute or estimate a statistic. Never introduce metrics, findings or decisions that are not in the payload.
- Every claim must cite ids that exist in the payload (evidence ids, decision ids).
- You may raise semantic flags about COLUMN NAMES only (e.g. a name suggesting the value is recorded after the outcome).
  These are hints that require manual review; never state that leakage was detected.
- You may suggest an answer to an open question only if user_context clearly supports it; otherwise omit it.
Respond with ONE JSON object and nothing else:
{"summary": str (<= 120 words),
 "priorities": [{"text": str, "evidence_ids": [str], "decision_ids": [str]}],
 "suggested_answers": [{"question_id": str, "option": str, "reason": str}],
 "questions_for_user": [str],
 "semantic_flags": [{"column": str, "concern": str, "reason": str}]}"""


class _Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    decision_ids: list[str] = Field(default_factory=list)


class _Suggestion(BaseModel):
    question_id: str
    option: str
    reason: str = ""


class _Flag(BaseModel):
    column: str
    concern: str
    reason: str = ""


class _Output(BaseModel):
    summary: str
    priorities: list[_Claim] = Field(default_factory=list)
    suggested_answers: list[_Suggestion] = Field(default_factory=list)
    questions_for_user: list[str] = Field(default_factory=list)
    semantic_flags: list[_Flag] = Field(default_factory=list)


def resolve_provider(cfg: LLMConfig) -> Provider:
    if isinstance(cfg.provider, Provider):
        return cfg.provider
    name = str(cfg.provider).lower()
    kw: dict[str, Any] = {"api_key": cfg.api_key, "max_tokens": cfg.max_tokens}
    if cfg.model:
        kw["model"] = cfg.model
    if name == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(**kw)
    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(**kw)
    if name == "local":
        from .local import LocalProvider

        return LocalProvider(base_url=cfg.base_url, **kw)
    raise ValueError(f"Unknown provider '{cfg.provider}'. Known: 'anthropic', 'gemini', 'local', or a Provider.")


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in the response")
    return json.loads(text[start:end + 1])


def interpret_report(report: Any, llm: Any) -> Any:
    cfg = coerce_config(llm)
    try:
        provider = resolve_provider(cfg)
    except (ImportError, ValueError) as exc:
        warnings.warn(f"LLM interpretation skipped: {exc}", stacklevel=3)
        return report
    payload, reverse = build_payload(report, cfg)
    user = "Report payload:\n" + json.dumps(payload, default=str) + "\n\nReturn the JSON object."
    try:
        raw = provider.complete(SYSTEM, user)
        out = _Output.model_validate(_extract_json(raw))
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        warnings.warn(f"LLM output rejected (does not match the schema): {exc}", stacklevel=3)
        return report
    except Exception as exc:  # network / provider errors must not break analysis
        warnings.warn(f"LLM call failed: {exc}", stacklevel=3)
        return report

    def real(x: str) -> str:
        return reverse.get(x, x)

    known_ev, known_dec = set(report.details.evidence), {d.id for d in report.decisions}
    known_cols = set(report.details.profile.columns) | {report.target}
    dropped = 0
    claims = []
    for c in out.priorities:
        ev = [real(x) for x in c.evidence_ids]
        dec = [real(x) for x in c.decision_ids]
        if not (ev or dec) or not set(ev) <= known_ev or not set(dec) <= known_dec:
            dropped += 1
            continue
        claims.append(LLMClaim(text=c.text, evidence_ids=ev, decision_ids=dec))
    flags = []
    for f in out.semantic_flags:
        col = real(f.column)
        if col not in known_cols:
            dropped += 1
            continue
        flags.append(SemanticFlag(column=col, concern=f.concern, reason=f.reason))

    suggestions: dict[str, tuple[str, str]] = {}
    qmap = {q.id: q for q in report.questions}
    for s in out.suggested_answers:
        q = qmap.get(real(s.question_id))
        if q is None or q.answer is not None or s.option not in [o.value for o in q.options]:
            dropped += 1
            continue
        suggestions[q.id] = (s.option, s.reason)
    questions = [evolve(q, suggestion=suggestions[q.id][0], suggestion_reason=suggestions[q.id][1])
                 if q.id in suggestions else q for q in report.questions]
    model = getattr(provider, "model", None) or cfg.model or "?"
    from ..api import _carry, _next_revision

    llm_out = LLMInterpretation(
        provider=type(provider).__name__, model=str(model), summary=out.summary, priorities=claims,
        questions_for_user=out.questions_for_user, semantic_flags=flags, dropped_claims=dropped,
        disclosure=disclosure(cfg, type(provider).__name__, str(model), payload))
    manifest = _next_revision(report.details.manifest, "interpret", provider=type(provider).__name__,
                              model=str(model), column_names_sent=cfg.share_column_names, dropped_claims=dropped)
    details = evolve(report.details, manifest=manifest, llm=llm_out)
    return _carry(report, evolve(report, questions=questions, details=details))
