"""A daily cap on model spend, checked where spending starts.

Two routes start billed model calls: POST /documents (each upload queues
an extraction and a transcription) and POST /ask (an agent run, plus a
sampled grading job). Both refuse new work with 429 once the spend
recorded since midnight UTC reaches DAILY_BUDGET_USD, before anything is
stored or called. Retry-After says when the day resets.

It is a soft cap by construction. Work admitted under the cap still
finishes: jobs already queued, and questions already in flight (each
capped at AGENT_MAX_COST_USD). So the day can end above the budget by
at most that in-flight work. "Recorded spend" is what the database
holds: extraction and transcription rows, stored /ask runs and their
grades. A call that failed after being billed shows up only in its job's
last_error, so it isn't counted.
"""

from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import AskRun, DocumentPage, Extraction


def day_start(now: datetime) -> datetime:
    return now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def seconds_until_reset(now: datetime) -> int:
    return max(1, int((day_start(now) + timedelta(days=1) - now).total_seconds()))


async def spent_today_usd(session: AsyncSession, now: datetime | None = None) -> float:
    since = day_start(now or datetime.now(UTC))
    extraction, transcription, agent, judge = (
        await session.execute(
            select(
                select(func.sum(Extraction.cost_usd))
                .where(Extraction.created_at >= since)
                .scalar_subquery(),
                select(func.sum(DocumentPage.cost_usd))
                .where(DocumentPage.source == "transcription", DocumentPage.created_at >= since)
                .scalar_subquery(),
                select(func.sum(AskRun.cost_usd)).where(AskRun.created_at >= since).scalar_subquery(),
                select(func.sum(AskRun.judge_cost_usd))
                .where(AskRun.judged_at >= since)
                .scalar_subquery(),
            )
        )
    ).one()
    return float(sum(value or 0 for value in (extraction, transcription, agent, judge)))


async def enforce_daily_budget(session: AsyncSession, settings: Settings) -> None:
    """Raise 429 if today's recorded spend has reached the budget.
    DAILY_BUDGET_USD = 0 turns the cap off."""
    budget = settings.daily_budget_usd
    if budget <= 0:
        return
    now = datetime.now(UTC)
    spent = await spent_today_usd(session, now)
    if spent >= budget:
        raise HTTPException(
            status_code=429,
            detail=(
                f"daily model budget reached: ${spent:.2f} of ${budget:.2f} spent today (UTC); "
                "resets at midnight UTC"
            ),
            headers={"Retry-After": str(seconds_until_reset(now))},
        )
