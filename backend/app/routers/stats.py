"""Aggregate stats endpoint: one GET /stats snapshot of document/extraction
volume, spend, latency, review-queue state, and the most recent eval run.

No caching -- every field is computed fresh from the DB (and, for
last_eval, the eval-results directory) on each request. This endpoint is
a dashboard/ops read, not a hot path, so the extra round-trips are an
acceptable trade for never serving stale numbers.
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.db import get_session
from app.evals.report import load_results
from app.models import Document, ExtractedField, Extraction
from app.routers.review import _PENDING
from app.schemas import StatsLastEval, StatsOut, StatsReview

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
async def get_stats(session: AsyncSession = Depends(get_session)) -> StatsOut:
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
    )
