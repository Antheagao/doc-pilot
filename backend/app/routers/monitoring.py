"""GET /monitoring/* -- the data behind the monitoring dashboard
(frontend /dashboard): how good, how expensive and how slow the AI parts
are, offline and in production.

- /evals: every committed eval run (extraction, the /ask agent,
  retrieval), each normalized to accuracy, cost per item and latency per
  item (app/evals/history.py). Offline, against known answers.
- /daily: the same three questions about live traffic, per UTC day (the
  daily budget's day): spend by stage, answer quality (the sampled
  groundedness grader and people's feedback), cost and latency per answer
  and per extracted document.

Computed fresh on every request, like GET /stats.
"""

from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Date, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.db import get_session
from app.evals.history import eval_history
from app.models import AskRun, DocumentPage, Extraction
from app.retrieval.indexing import ChunkingConfig
from app.schemas import (
    DailyAnswers,
    DailyExtractions,
    DailyPoint,
    DailySpend,
    EvalHistoryOut,
    EvalRunOut,
)

router = APIRouter()


@router.get("/evals", response_model=EvalHistoryOut)
async def evals(settings: Settings = Depends(get_settings)) -> EvalHistoryOut:
    history = await run_in_threadpool(eval_history, ChunkingConfig.from_settings(settings))
    return EvalHistoryOut(
        **{
            suite: [EvalRunOut.model_validate(point) for point in points]
            for suite, points in history.items()
        }
    )


def _utc_day(column: Any) -> Any:
    """The UTC calendar day of a timestamptz column."""
    return cast(func.timezone("UTC", column), Date)


def _p(fraction: float, column: Any) -> Any:
    return func.percentile_cont(fraction).within_group(column)


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


@router.get("/daily", response_model=list[DailyPoint])
async def daily(
    days: int = Query(30, ge=1, le=366),
    session: AsyncSession = Depends(get_session),
) -> list[DailyPoint]:
    """The last `days` UTC days, oldest first, today included; a day with
    no traffic is present with zero counts and null rates."""
    today = datetime.now(UTC).date()
    first = today - timedelta(days=days - 1)
    since = datetime.combine(first, time.min, tzinfo=UTC)

    ask_day = _utc_day(AskRun.created_at)
    answers = {
        row.day: row
        for row in await session.execute(
            select(
                ask_day.label("day"),
                func.count().label("runs"),
                func.count().filter(AskRun.status == "answered").label("answered"),
                func.count().filter(AskRun.judge_grounded.is_not(None)).label("judged"),
                func.count().filter(AskRun.judge_grounded.is_(True)).label("grounded"),
                func.count().filter(AskRun.feedback == "up").label("up"),
                func.count().filter(AskRun.feedback == "down").label("down"),
                func.sum(AskRun.cost_usd).label("cost"),
                func.avg(AskRun.cost_usd).label("mean_cost"),
                _p(0.5, AskRun.latency_ms).label("p50"),
                _p(0.95, AskRun.latency_ms).label("p95"),
            )
            .where(AskRun.created_at >= since)
            .group_by(ask_day)
        )
    }
    # Grading is billed when it happens, not when the answer was given --
    # the same attribution as the daily budget (app/budget.py).
    judge_day = _utc_day(AskRun.judged_at)
    grading = dict(
        (
            await session.execute(
                select(judge_day, func.sum(AskRun.judge_cost_usd))
                .where(AskRun.judged_at >= since)
                .group_by(judge_day)
            )
        ).all()
    )
    extraction_day = _utc_day(Extraction.created_at)
    extractions = {
        row.day: row
        for row in await session.execute(
            select(
                extraction_day.label("day"),
                func.count().label("documents"),
                func.sum(Extraction.cost_usd).label("cost"),
                func.avg(Extraction.cost_usd).label("mean_cost"),
                _p(0.5, Extraction.latency_ms).label("p50"),
                _p(0.95, Extraction.latency_ms).label("p95"),
            )
            .where(Extraction.created_at >= since)
            .group_by(extraction_day)
        )
    }
    page_day = _utc_day(DocumentPage.created_at)
    transcription = dict(
        (
            await session.execute(
                select(page_day, func.sum(DocumentPage.cost_usd))
                .where(DocumentPage.created_at >= since, DocumentPage.source == "transcription")
                .group_by(page_day)
            )
        ).all()
    )

    points = []
    for offset in range(days):
        day: date = first + timedelta(days=offset)
        ask = answers.get(day)
        ext = extractions.get(day)
        documents_usd = float(ext.cost if ext else 0) + float(transcription.get(day) or 0)
        answers_usd = float(ask.cost if ask else 0)
        grading_usd = float(grading.get(day) or 0)
        points.append(
            DailyPoint(
                date=day,
                spend=DailySpend(
                    documents_usd=round(documents_usd, 6),
                    answers_usd=round(answers_usd, 6),
                    grading_usd=round(grading_usd, 6),
                    total_usd=round(documents_usd + answers_usd + grading_usd, 6),
                ),
                answers=DailyAnswers(
                    runs=ask.runs if ask else 0,
                    answered=ask.answered if ask else 0,
                    judged=ask.judged if ask else 0,
                    grounded_rate=ask.grounded / ask.judged if ask and ask.judged else None,
                    feedback_up=ask.up if ask else 0,
                    feedback_down=ask.down if ask else 0,
                    mean_cost_usd=_float(ask.mean_cost) if ask else None,
                    latency_p50_ms=_float(ask.p50) if ask else None,
                    latency_p95_ms=_float(ask.p95) if ask else None,
                ),
                extractions=DailyExtractions(
                    documents=ext.documents if ext else 0,
                    mean_cost_usd=_float(ext.mean_cost) if ext else None,
                    latency_p50_ms=_float(ext.p50) if ext else None,
                    latency_p95_ms=_float(ext.p95) if ext else None,
                ),
            )
        )
    return points
