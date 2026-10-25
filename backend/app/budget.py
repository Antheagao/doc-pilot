"""A daily cap on model spend, checked where spending starts.

Two routes start billed model calls: POST /documents (each upload queues
an extraction and a transcription) and POST /ask (an agent run, plus a
sampled grading job). Both refuse new work with 429 once the spend
recorded since midnight UTC reaches a cap, before anything is stored or
called. Retry-After says when the day resets.

There are two caps. DAILY_BUDGET_USD is the installation's: it guards the one
Anthropic key, so it counts every organization's spend, through
docpilot_spend_since (a SECURITY DEFINER sum that returns a number and no rows,
because row-level security hides other organizations' rows from the
application role), and its refusal never prints a figure, which would be other
tenants' spend. ORGANIZATION_DAILY_BUDGET_USD is the optional noisy-neighbor
cap: it counts the session's own organization only (spent_today_usd, scoped by
the ORM layer and by row-level security) and is checked first, so a tenant that
hit its own cap is told so rather than blamed on the installation.

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
from sqlalchemy import DateTime, func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import AskRun, DocumentPage, Extraction


def day_start(now: datetime) -> datetime:
    return now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def seconds_until_reset(now: datetime) -> int:
    return max(1, int((day_start(now) + timedelta(days=1) - now).total_seconds()))


async def spent_today_usd(session: AsyncSession, now: datetime | None = None) -> float:
    """Today's recorded spend of the session's organization."""
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


async def installation_spent_today_usd(session: AsyncSession, now: datetime | None = None) -> float:
    """Today's recorded spend across every organization, from the database
    function that sums past row-level security (migration 73a7d8691932)."""
    since = day_start(now or datetime.now(UTC))
    spent = (
        await session.execute(select(func.docpilot_spend_since(literal(since, DateTime(timezone=True)))))
    ).scalar_one()
    return float(spent)


async def enforce_daily_budget(session: AsyncSession, settings: Settings) -> None:
    """Raise 429 if today's recorded spend has reached a cap: the
    organization's own (ORGANIZATION_DAILY_BUDGET_USD) first, then the
    installation's (DAILY_BUDGET_USD). A cap of 0 is off."""
    organization_budget = settings.organization_daily_budget_usd
    installation_budget = settings.daily_budget_usd
    if organization_budget <= 0 and installation_budget <= 0:
        return
    now = datetime.now(UTC)
    retry_after = {"Retry-After": str(seconds_until_reset(now))}
    if organization_budget > 0:
        spent = await spent_today_usd(session, now)
        if spent >= organization_budget:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"daily model budget reached: your organization has spent ${spent:.2f} of "
                    f"${organization_budget:.2f} today (UTC); resets at midnight UTC"
                ),
                headers=retry_after,
            )
    if installation_budget > 0 and await installation_spent_today_usd(session, now) >= installation_budget:
        # No figure: the installation's total is every organization's spend.
        raise HTTPException(
            status_code=429,
            detail=(
                "daily model budget reached: the installation's budget is spent for today (UTC); "
                "resets at midnight UTC"
            ),
            headers=retry_after,
        )
