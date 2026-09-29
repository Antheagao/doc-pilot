"""Aggregate stats endpoint: one GET /stats snapshot of document/extraction
volume, spend (every model-calling stage: extraction, transcription, the
/ask agent, its online grader), latency, review-queue state, /ask quality
signals, and the most recent eval run.

No caching -- every field is computed fresh from the DB (and, for
last_eval, the eval-results directory) on each request. This endpoint is
a dashboard/ops read, not a hot path, so the extra round-trips are an
acceptable trade for never serving stale numbers.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import Integer, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.budget import seconds_until_reset, spent_today_usd
from app.config import Settings, get_settings
from app.db import get_session
from app.evals.report import load_results
from app.models import AskRun, Document, DocumentPage, ExtractedField, Extraction
from app.routers.review import _PENDING
from app.schemas import (
    StatsAsk,
    StatsBudget,
    StatsLastEval,
    StatsOut,
    StatsReview,
    StatsSpend,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Zero-filled in documents_by_status so the response shape is stable even
# before a status (e.g. the future "refused") has ever been written --
# an unexpected status found in the DB still appears in the dict rather
# than being dropped, it just isn't guaranteed to be present up front.
KNOWN_STATUSES = ("uploaded", "processing", "extracted", "failed", "refused")

# Test seam for load_results' results_dir: None defers to that function's
# own default (EVALS_DIR/"results"). A test pointing this at a tmp_path
# with a junk artifact can exercise the corrupt-artifact path without
# touching the real evals/results/ directory.
_RESULTS_DIR: Path | None = None


def _load_last_eval_results() -> list[dict[str, Any]]:
    return load_results(_RESULTS_DIR)


@router.get("", response_model=StatsOut)
async def get_stats(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> StatsOut:
    documents_total = (
        await session.execute(select(func.count()).select_from(Document))
    ).scalar_one()

    status_rows = (
        await session.execute(select(Document.status, func.count()).group_by(Document.status))
    ).all()
    documents_by_status = dict.fromkeys(KNOWN_STATUSES, 0)
    for status, count in status_rows:
        documents_by_status[status] = count

    extraction_row = (
        await session.execute(
            select(
                func.count(func.distinct(Extraction.document_id)),
                func.count(),
                func.sum(Extraction.cost_usd),
                func.sum(Extraction.input_tokens),
                func.sum(Extraction.output_tokens),
                func.percentile_cont(0.5).within_group(Extraction.latency_ms.asc()),
                func.percentile_cont(0.95).within_group(Extraction.latency_ms.asc()),
            ).select_from(Extraction)
        )
    ).one()
    (
        documents_processed,
        extractions_total,
        total_cost_usd_raw,
        total_input_tokens_raw,
        total_output_tokens_raw,
        latency_p50_ms,
        latency_p95_ms,
    ) = extraction_row

    # SUM(Numeric) comes back as Decimal via asyncpg -- cast before it
    # reaches the (float) pydantic schema. SUM over zero rows is NULL for
    # every aggregate here; documents_processed/extractions_total are
    # COUNTs so they're already 0, not None.
    total_cost_usd = float(total_cost_usd_raw) if total_cost_usd_raw is not None else 0.0
    total_input_tokens = total_input_tokens_raw or 0
    total_output_tokens = total_output_tokens_raw or 0
    mean_cost_per_doc = (
        total_cost_usd / documents_processed if documents_processed else None
    )

    pending = (
        await session.execute(
            select(func.count()).select_from(ExtractedField).where(*_PENDING)
        )
    ).scalar_one()
    action_rows = (
        await session.execute(
            select(ExtractedField.review_action, func.count()).group_by(
                ExtractedField.review_action
            )
        )
    ).all()
    review_counts = {"approved": 0, "corrected": 0}
    for action, count in action_rows:
        if action in review_counts:
            review_counts[action] = count

    transcription_cost_raw = (
        await session.execute(
            select(func.sum(DocumentPage.cost_usd)).where(DocumentPage.source == "transcription")
        )
    ).scalar_one()
    transcription_cost = float(transcription_cost_raw or 0)
    mean_pipeline_cost_per_doc = (
        (total_cost_usd + transcription_cost) / documents_processed if documents_processed else None
    )

    ask = await _ask_stats(session)
    agent_cost, judge_cost = ask.pop("agent_usd"), ask.pop("judge_usd")
    spend = StatsSpend(
        extraction_usd=total_cost_usd,
        transcription_usd=transcription_cost,
        agent_usd=agent_cost,
        judge_usd=judge_cost,
        total_usd=total_cost_usd + transcription_cost + agent_cost + judge_cost,
    )

    now = datetime.now(UTC)

    # The whole load-and-project step is one try/except: a loadable-but-
    # malformed artifact (e.g. summary: {} instead of a full dict, or a
    # wrong-typed top-level value) can raise just as easily during
    # projection into StatsLastEval as during load_results itself, and
    # either way a corrupt artifact must never 500 this endpoint.
    last_eval = None
    try:
        eval_results = await run_in_threadpool(_load_last_eval_results)
        if eval_results:
            last = eval_results[-1]
            summary = last["summary"]
            last_eval = StatsLastEval(
                model=last["model"],
                prompt_version=last["prompt_version"],
                dataset_version=last["dataset_version"],
                started_at_utc=last["started_at_utc"],
                overall_accuracy=summary["overall_accuracy"] if summary is not None else None,
                caught_by_review=summary["caught_by_review"] if summary is not None else None,
                mean_cost_per_doc=last["mean_cost_per_doc"],
                n_scored=last["n_scored"],
            )
    except Exception:
        logger.warning("failed to load eval results for /stats", exc_info=True)
        last_eval = None

    return StatsOut(
        documents_total=documents_total,
        documents_by_status=documents_by_status,
        documents_processed=documents_processed,
        extractions_total=extractions_total,
        total_cost_usd=total_cost_usd,
        mean_cost_per_doc=mean_cost_per_doc,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms,
        review=StatsReview(
            pending=pending, approved=review_counts["approved"], corrected=review_counts["corrected"]
        ),
        last_eval=last_eval,
        mean_pipeline_cost_per_doc=mean_pipeline_cost_per_doc,
        spend=spend,
        budget=StatsBudget(
            daily_budget_usd=settings.daily_budget_usd or None,
            spent_today_usd=await spent_today_usd(session, now),
            resets_in_seconds=seconds_until_reset(now),
        ),
        ask=StatsAsk(**ask),
    )


async def _ask_stats(session: AsyncSession) -> dict[str, Any]:
    readable = AskRun.judged_at.is_not(None) & AskRun.judge_error.is_(None)
    row = (
        await session.execute(
            select(
                func.count(),
                func.count().filter(AskRun.status == "answered"),
                func.sum(AskRun.cost_usd),
                func.percentile_cont(0.5).within_group(AskRun.latency_ms.asc()),
                func.percentile_cont(0.95).within_group(AskRun.latency_ms.asc()),
                func.count().filter(AskRun.feedback == "up"),
                func.count().filter(AskRun.feedback == "down"),
                func.count().filter(AskRun.judge_sampled),
                func.count().filter(readable),
                func.avg(cast(AskRun.judge_grounded, Integer)).filter(readable),
                func.avg(cast(AskRun.judge_answers_question, Integer)).filter(readable),
                func.sum(AskRun.judge_cost_usd),
            ).select_from(AskRun)
        )
    ).one()
    (runs, answered, cost, p50, p95, up, down, sampled, judged, grounded, answers, judge_cost) = row
    agent_usd = float(cost or 0)
    return {
        "runs": runs,
        "answered": answered,
        "mean_cost_usd": agent_usd / runs if runs else None,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "feedback_up": up,
        "feedback_down": down,
        "judge_sampled": sampled,
        "judged": judged,
        "judge_grounded_rate": float(grounded) if grounded is not None else None,
        "judge_answers_rate": float(answers) if answers is not None else None,
        "agent_usd": agent_usd,
        "judge_usd": float(judge_cost or 0),
    }
