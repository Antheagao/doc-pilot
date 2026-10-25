"""The daily spend cap (app/budget.py): what counts as today's spend, and
the two routes that refuse new billed work once it's reached."""

import base64
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from fastapi import HTTPException
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.budget import (
    day_start,
    enforce_daily_budget,
    installation_spent_today_usd,
    seconds_until_reset,
    spent_today_usd,
)
from app.config import Settings, get_settings
from app.main import app
from app.models import AskRun, Document, DocumentPage, Extraction, Organization
from app.routers.ask import get_anthropic_client
from app.tenancy import bind_tenant

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _run(**overrides) -> AskRun:
    fields = {
        "question": "q", "status": "answered", "answer": "a", "citations": [], "tool_calls": [],
        "evidence": "e", "steps": 1, "model": "claude-opus-5-5", "prompt_version": "agent_v1",
    }
    return AskRun(**{**fields, **overrides})


def _cap(tmp_path, budget: float, organization_budget: float = 0) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        upload_dir=str(tmp_path), daily_budget_usd=budget, organization_daily_budget_usd=organization_budget,
        anthropic_api_key="test", ask_rate_limit_per_minute=0,
    )


def test_the_day_is_a_utc_day() -> None:
    late = datetime(2026, 9, 29, 23, 59, 30, tzinfo=UTC)

    assert day_start(late) == datetime(2026, 9, 29, tzinfo=UTC)
    assert seconds_until_reset(late) == 30
    assert seconds_until_reset(datetime(2026, 9, 30, tzinfo=UTC)) == 86400
    # A non-UTC clock reading is the same instant.
    pacific = late.astimezone(ZoneInfo("America/Los_Angeles"))
    assert day_start(pacific) == datetime(2026, 9, 29, tzinfo=UTC)


async def test_todays_spend_counts_every_billed_stage_and_nothing_older(
    db_session: AsyncSession,
) -> None:
    before = await spent_today_usd(db_session)
    yesterday = datetime.now(UTC) - timedelta(days=1, hours=1)
    document = Document(filename="r.png", mime_type="image/png", storage_path="/tmp/r", status="extracted")
    db_session.add(document)
    await db_session.flush()
    db_session.add_all([
        Extraction(document_id=document.id, prompt_version="extract_v1", model="claude-sonnet-5",
                   raw_response={}, input_tokens=1, output_tokens=1, latency_ms=1, cost_usd=0.010),
        Extraction(document_id=document.id, prompt_version="extract_v1", model="claude-sonnet-5",
                   raw_response={}, input_tokens=1, output_tokens=1, latency_ms=1, cost_usd=9.0,
                   created_at=yesterday),
        DocumentPage(document_id=document.id, page_number=1, text="t", source="transcription", cost_usd=0.004),
        DocumentPage(document_id=document.id, page_number=2, text="t", source="gold", cost_usd=0),
        _run(cost_usd=0.050, judge_cost_usd=0.003, judged_at=datetime.now(UTC)),
        _run(cost_usd=9.0, created_at=yesterday),
    ])
    await db_session.flush()

    assert await spent_today_usd(db_session) == pytest.approx(before + 0.067, abs=1e-6)


async def test_uploads_are_refused_once_the_budget_is_spent(
    client: AsyncClient, db_session: AsyncSession, tmp_path
) -> None:
    spent = await installation_spent_today_usd(db_session)
    documents_before = (await db_session.execute(select(func.count()).select_from(Document))).scalar_one()
    db_session.add(_run(cost_usd=0.25))
    await db_session.commit()
    _cap(tmp_path, spent + 0.20)

    response = await client.post("/documents", files={"file": ("r.png", TINY_PNG, "image/png")})

    assert response.status_code == 429
    assert "daily model budget reached" in response.json()["detail"]
    assert "$" not in response.json()["detail"]  # the installation's total is every tenant's spend
    assert 1 <= int(response.headers["retry-after"]) <= 86400
    # Refused before anything was stored: no row, no file.
    assert (await db_session.execute(select(func.count()).select_from(Document))).scalar_one() == documents_before
    assert list(tmp_path.iterdir()) == []


async def test_uploads_under_the_budget_or_with_the_cap_off_go_through(
    client: AsyncClient, db_session: AsyncSession, tmp_path
) -> None:
    spent = await installation_spent_today_usd(db_session)
    _cap(tmp_path, spent + 1.0)
    assert (await client.post("/documents", files={"file": ("a.png", TINY_PNG, "image/png")})).status_code == 201

    db_session.add(_run(cost_usd=2.0))
    await db_session.commit()
    _cap(tmp_path, 0)  # off
    assert (await client.post("/documents", files={"file": ("b.png", TINY_PNG, "image/png")})).status_code == 201


async def test_ask_is_refused_before_any_model_call(
    client: AsyncClient, db_session: AsyncSession, tmp_path
) -> None:
    spent = await installation_spent_today_usd(db_session)
    db_session.add(_run(cost_usd=0.25))
    await db_session.commit()
    _cap(tmp_path, spent + 0.20)
    create = AsyncMock()
    app.dependency_overrides[get_anthropic_client] = lambda: SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=create))
    )

    for route in ("/ask", "/ask/stream"):
        response = await client.post(route, json={"question": "How much did I spend?"})

        assert response.status_code == 429 and "retry-after" in response.headers
    create.assert_not_awaited()


async def test_stats_report_the_budget(client: AsyncClient, db_session: AsyncSession, tmp_path) -> None:
    db_session.add(_run(cost_usd=0.125))
    await db_session.commit()
    _cap(tmp_path, 7.5)

    budget = (await client.get("/stats")).json()["budget"]

    assert budget["daily_budget_usd"] == 7.5
    assert budget["organization_daily_budget_usd"] is None
    assert budget["installation_budget_reached"] is False
    assert budget["spent_today_usd"] >= 0.125
    assert 1 <= budget["resets_in_seconds"] <= 86400

    _cap(tmp_path, 0)
    off = (await client.get("/stats")).json()["budget"]
    assert off["daily_budget_usd"] is None and off["installation_budget_reached"] is False


async def test_stats_report_the_organization_cap_and_whether_the_installation_cap_is_reached(
    client: AsyncClient, db_session: AsyncSession, tmp_path
) -> None:
    db_session.add(_run(cost_usd=0.5))
    await db_session.commit()
    spent = await installation_spent_today_usd(db_session)

    _cap(tmp_path, spent - 0.25, organization_budget=3.5)
    reached = (await client.get("/stats")).json()["budget"]
    _cap(tmp_path, spent + 100)
    clear = (await client.get("/stats")).json()["budget"]

    assert reached["organization_daily_budget_usd"] == 3.5
    assert reached["installation_budget_reached"] is True
    assert reached["daily_budget_usd"] == pytest.approx(spent - 0.25)
    assert clear["installation_budget_reached"] is False


async def test_one_organizations_spend_counts_toward_the_installation_cap_and_not_toward_anothers(
    db_session: AsyncSession, second_organization: Organization
) -> None:
    spent_here = await spent_today_usd(db_session)
    db_session.add(_run(cost_usd=0.25))
    await db_session.flush()
    installation = await installation_spent_today_usd(db_session)
    await bind_tenant(db_session, second_organization.id)

    # The second organization has spent nothing, so its own cap is clear...
    assert await spent_today_usd(db_session) == 0
    await enforce_daily_budget(
        db_session, Settings(daily_budget_usd=0, organization_daily_budget_usd=0.20)
    )
    # ...while the installation's cap, the sum over both, is reached.
    assert await installation_spent_today_usd(db_session) == pytest.approx(installation)
    with pytest.raises(HTTPException) as refused:
        await enforce_daily_budget(
            db_session, Settings(daily_budget_usd=installation - 0.01, organization_daily_budget_usd=0.20)
        )

    assert refused.value.status_code == 429
    assert "installation's budget" in refused.value.detail
    assert "$" not in refused.value.detail  # no figure: the sum is other tenants' spend too
    assert "retry-after" in {name.lower() for name in refused.value.headers}
    assert installation >= spent_here + 0.25


async def test_the_organization_cap_is_checked_first_and_names_the_organizations_own_spend(
    db_session: AsyncSession
) -> None:
    db_session.add(_run(cost_usd=0.25))
    await db_session.flush()
    mine = await spent_today_usd(db_session)

    # Both caps are reached: the organization's is the one named.
    with pytest.raises(HTTPException) as refused:
        await enforce_daily_budget(
            db_session, Settings(daily_budget_usd=0.01, organization_daily_budget_usd=mine - 0.01)
        )

    assert refused.value.status_code == 429
    assert "your organization" in refused.value.detail
    assert f"${mine:.2f}" in refused.value.detail


async def test_each_cap_is_enforced_on_its_own_and_both_off_enforces_nothing(
    db_session: AsyncSession
) -> None:
    db_session.add(_run(cost_usd=50.0))
    await db_session.flush()

    await enforce_daily_budget(db_session, Settings(daily_budget_usd=0, organization_daily_budget_usd=0))
    with pytest.raises(HTTPException):
        await enforce_daily_budget(db_session, Settings(daily_budget_usd=0, organization_daily_budget_usd=49))
    with pytest.raises(HTTPException):
        await enforce_daily_budget(db_session, Settings(daily_budget_usd=49, organization_daily_budget_usd=0))


async def test_an_organization_over_its_cap_is_refused_before_any_work(
    client: AsyncClient, db_session: AsyncSession, tmp_path
) -> None:
    spent = await spent_today_usd(db_session)
    db_session.add(_run(cost_usd=0.25))
    await db_session.commit()
    _cap(tmp_path, 0, organization_budget=spent + 0.20)

    response = await client.post("/documents", files={"file": ("r.png", TINY_PNG, "image/png")})

    assert response.status_code == 429
    assert "your organization" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_the_organization_cap_must_not_be_negative_and_defaults_to_off() -> None:
    assert Settings().organization_daily_budget_usd == 0
    with pytest.raises(ValueError):
        Settings(organization_daily_budget_usd=-1)


def test_a_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError):
        Settings(daily_budget_usd=-1)
