"""POST /ask -- answer a question about the documents with the tool-use
agent (app/agent/), with citations back to pages and extracted fields.

This is the one API route that calls the model, so unlike the rest of the
API it needs ANTHROPIC_API_KEY (see docker-compose.yml). Every call is
billed; the per-question step and dollar caps live in Settings.

Every answer is stored (app.models.AskRun) with the evidence the agent
saw, so it can be audited and graded after the fact: GET /ask/runs lists
them, POST /ask/runs/{id}/feedback records a person's thumbs up / down,
and a sampled share gets a background groundedness grade
(app/evals/online.py).
"""

import uuid
from datetime import UTC, datetime

import anthropic
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.loop import answer_question
from app.agent.tools import ToolContext
from app.config import Settings, get_settings
from app.db import get_session
from app.evals.online import record_ask_run
from app.extraction import ExtractionError, NonRetryableExtractionError, _build_client
from app.models import AskRun
from app.retrieval.embeddings import Embedder, get_embedder
from app.schemas import (
    AskFeedbackRequest,
    AskJudgmentOut,
    AskRequest,
    AskResponse,
    AskRunSummary,
)

router = APIRouter()

_client: anthropic.AsyncAnthropic | None = None


def get_anthropic_client(
    settings: Settings = Depends(get_settings),
) -> anthropic.AsyncAnthropic | None:
    """One client for the API process: a client owns a connection pool,
    and building one per request would pay the TLS setup every time and
    leave the pools to the garbage collector. None when no API key is
    configured (the route answers 503, after request validation). A FastAPI
    dependency so tests can substitute a fake."""
    global _client
    if not settings.anthropic_api_key:
        return None
    if _client is None:
        _client = _build_client(settings)
    return _client


def ask_response(run: AskRun) -> AskResponse:
    judgment = None
    if run.judged_at is not None:
        judgment = AskJudgmentOut(
            grounded=run.judge_grounded,
            answers_question=run.judge_answers_question,
            unsupported_claims=run.judge_claims or [],
            explanation=run.judge_explanation,
            model=run.judge_model,
            prompt_version=run.judge_prompt_version,
            cost_usd=float(run.judge_cost_usd or 0),
            error=run.judge_error,
            judged_at=run.judged_at,
        )
    return AskResponse(
        id=run.id,
        question=run.question,
        created_at=run.created_at,
        status=run.status,
        answer=run.answer,
        citations=run.citations,
        tool_calls=run.tool_calls,
        steps=run.steps,
        model=run.model,
        prompt_version=run.prompt_version,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        cache_read_input_tokens=run.cache_read_input_tokens,
        cost_usd=float(run.cost_usd),
        latency_ms=run.latency_ms,
        refusal_category=run.refusal_category,
        trace_id=run.trace_id,
        feedback=run.feedback,
        feedback_note=run.feedback_note,
        judge_sampled=run.judge_sampled,
        judgment=judgment,
    )


@router.post("", response_model=AskResponse)
async def ask(
    request: AskRequest,
    session: AsyncSession = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
    settings: Settings = Depends(get_settings),
    client: anthropic.AsyncAnthropic | None = Depends(get_anthropic_client),
) -> AskResponse:
    if client is None:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured")
    try:
        result = await answer_question(
            ToolContext(session=session, embedder=embedder), request.question, settings, client
        )
    except NonRetryableExtractionError as exc:
        raise HTTPException(status_code=502, detail=f"model request rejected: {exc}") from exc
    except ExtractionError as exc:
        # Transient (rate limit, overload, network): the client may retry.
        headers = (
            {"Retry-After": str(int(exc.retry_after_seconds))} if exc.retry_after_seconds else None
        )
        raise HTTPException(status_code=503, detail=f"model unavailable: {exc}", headers=headers) from exc
    run = await record_ask_run(session, request.question, result, settings)
    return ask_response(run)


@router.get("/runs", response_model=list[AskRunSummary])
async def list_runs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[AskRun]:
    stmt = (
        select(AskRun)
        .order_by(AskRun.created_at.desc(), AskRun.id)
        .limit(limit)
        .offset(offset)
    )
    return list((await session.execute(stmt)).scalars())


async def _get_run(session: AsyncSession, run_id: uuid.UUID) -> AskRun:
    run = await session.get(AskRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="ask run not found")
    return run


@router.get("/runs/{run_id}", response_model=AskResponse)
async def get_run(run_id: uuid.UUID, session: AsyncSession = Depends(get_session)) -> AskResponse:
    return ask_response(await _get_run(session, run_id))


@router.post("/runs/{run_id}/feedback", response_model=AskResponse)
async def give_feedback(
    run_id: uuid.UUID,
    request: AskFeedbackRequest,
    session: AsyncSession = Depends(get_session),
) -> AskResponse:
    """Record (or change) a person's verdict on an answer. The latest
    rating wins; the note is replaced with it (null clears it)."""
    run = await _get_run(session, run_id)
    run.feedback = request.rating
    run.feedback_note = request.note
    run.feedback_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(run)
    return ask_response(run)
