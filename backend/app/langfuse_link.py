"""Langfuse: LLM tracing and evaluation scores. Optional, off by default.

Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (plus LANGFUSE_BASE_URL for
a self-hosted Langfuse; Langfuse Cloud otherwise) and the API, the worker
and the eval scripts send Langfuse two things:

- Traces. Langfuse's SDK is built on OpenTelemetry, so it is one more span
  processor on doc-pilot's own tracer provider (app/telemetry.py) -- the
  spans doc-pilot already emits, not a second instrumentation. Its default
  filter forwards only spans with gen_ai.* attributes: every model call
  (a generation, with doc-pilot's own cost and token counts), the agent
  run and its tool calls, retrieval and embedding, and the online grader
  -- not SQL or HTTP spans. A document chat's conversation id is its
  Langfuse session, so a conversation's turns read as one thread.
- Scores, on the trace of the answer they grade (upserted by id, so a
  changed rating replaces the old one): the online groundedness grader's
  verdict (app/evals/online.py), a person's thumbs up / down
  (POST /ask/runs/{id}/feedback), and the offline agent eval's rubric
  (evals/run_agent.py).

The tracing content rule holds here too: no prompts, documents,
questions, answers or grader explanations go to Langfuse -- receipts carry
personal data. What goes is models, token counts, cost, latency, ids and
verdicts. Score failures are logged and dropped: observability must never
fail the request or job it observes.
"""

import json
import logging
import os
from typing import Any

from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

# Langfuse's OpenTelemetry attribute names (langfuse._client.attributes).
OBSERVATION_TYPE = "langfuse.observation.type"
COST_DETAILS = "langfuse.observation.cost_details"
USAGE_DETAILS = "langfuse.observation.usage_details"
SESSION_ID = "session.id"

_client: Any = None


def langfuse_configured() -> bool:
    return all(
        os.environ.get(key, "").strip() for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    )


def attach(provider: Any) -> Any:
    """Add Langfuse's span processor to `provider` and keep the client for
    scores. Called by app.telemetry.configure_tracing()."""
    global _client
    from langfuse import Langfuse

    # An empty LANGFUSE_HOST (compose passes unset variables as "") would
    # otherwise win over the SDK's default; resolve the URL here.
    base_url = (
        os.environ.get("LANGFUSE_BASE_URL", "").strip()
        or os.environ.get("LANGFUSE_HOST", "").strip()
        or "https://cloud.langfuse.com"
    )
    _client = Langfuse(base_url=base_url, tracer_provider=provider, should_export_span=should_export)
    return _client


def should_export(span: Any) -> bool:
    """Langfuse's default (spans with gen_ai.* attributes, and its own),
    plus any span doc-pilot typed for Langfuse -- the grader's `evaluate`
    span carries no gen_ai.* attribute of its own."""
    from langfuse.span_filter import is_default_export_span

    return is_default_export_span(span) or OBSERVATION_TYPE in (span.attributes or {})


def active() -> bool:
    return _client is not None


def shutdown() -> None:
    """Flush queued spans and scores and stop Langfuse's threads."""
    global _client
    client, _client = _client, None
    if client is not None:
        client.shutdown()


# ---- span attributes ---------------------------------------------------------------


def observation_attributes(kind: str, session_id: str | None = None) -> dict[str, str]:
    """Attributes that tell Langfuse what a span is ("agent", "tool",
    "evaluator", ...) -- empty when Langfuse is off, so plain OTLP traces
    stay vendor-neutral."""
    if _client is None:
        return {}
    attributes = {OBSERVATION_TYPE: kind}
    if session_id:
        attributes[SESSION_ID] = session_id
    return attributes


def record_generation(span: Span, *, cost_usd: float, usage: dict[str, int]) -> None:
    """Mark a model-call span as a Langfuse generation with doc-pilot's own
    cost: Langfuse would otherwise price it from its model table, which
    lags new models and knows nothing of cache pricing."""
    if _client is None:
        return
    span.set_attribute(OBSERVATION_TYPE, "generation")
    span.set_attribute(COST_DETAILS, json.dumps({"total": cost_usd}))
    span.set_attribute(USAGE_DETAILS, json.dumps(usage))


# ---- scores -----------------------------------------------------------------------------


def _score(*, trace_id: str | None, **fields: Any) -> None:
    if _client is None or not trace_id:
        return
    try:
        _client.create_score(trace_id=trace_id, data_type="BOOLEAN", **fields)
    except Exception:
        logger.exception("could not queue Langfuse score %s", fields.get("name"))


def score_judgment(run: Any) -> None:
    """The online grader's verdict on a stored answer (an AskRun). An
    unreadable verdict (judge_error) has nothing to score."""
    if run.judge_grounded is None:
        return
    metadata = {
        "ask_run_id": str(run.id),
        "grader_model": run.judge_model,
        "grader_prompt_version": run.judge_prompt_version,
    }
    _score(
        trace_id=run.trace_id,
        name="grounded",
        value=1.0 if run.judge_grounded else 0.0,
        score_id=f"{run.id}-grounded",
        metadata=metadata,
    )
    if run.judge_answers_question is not None:
        _score(
            trace_id=run.trace_id,
            name="answers_question",
            value=1.0 if run.judge_answers_question else 0.0,
            score_id=f"{run.id}-answers-question",
            metadata=metadata,
        )


def score_feedback(run: Any) -> None:
    """A person's thumbs up / down. Upserted: changing the rating replaces
    the score rather than adding a second one."""
    if run.feedback not in ("up", "down"):
        return
    _score(
        trace_id=run.trace_id,
        name="user_feedback",
        value=1.0 if run.feedback == "up" else 0.0,
        score_id=f"{run.id}-feedback",
        metadata={"ask_run_id": str(run.id)},
    )


def score_eval_run(result: dict[str, Any]) -> int:
    """The offline agent eval's rubric, one score per question, on the
    trace of the run that answered it. Returns how many were queued."""
    if _client is None:
        return 0
    queued = 0
    for entry in result.get("per_question", []):
        scores = entry.get("scores") or {}
        if "correct" not in scores or not entry.get("trace_id"):
            continue
        metadata = {
            "question_id": entry.get("id"),
            "question_type": entry.get("type"),
            "question_set_version": result.get("question_set_version"),
            "prompt_version": result.get("prompt_version"),
            "eval_run": result.get("started_at_utc"),
        }
        for name in ("correct", "cites_relevant"):
            if scores.get(name) is None:
                continue
            _score(
                trace_id=entry["trace_id"],
                name=f"eval_{name}",
                value=1.0 if scores[name] else 0.0,
                score_id=f"{result.get('started_at_utc')}-{entry.get('id')}-{name}",
                metadata=metadata,
            )
            queued += 1
    return queued
