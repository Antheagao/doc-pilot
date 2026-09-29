"""The daily spend cap (app/budget.py): what counts as today's spend, and
the two routes that refuse new billed work once it's reached."""

import base64
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.budget import day_start, seconds_until_reset, spent_today_usd
from app.config import Settings, get_settings
from app.main import app
from app.models import AskRun, Document, DocumentPage, Extraction
from app.routers.ask import get_anthropic_client

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


def _cap(tmp_path, budget: float) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        upload_dir=str(tmp_path), daily_budget_usd=budget, anthropic_api_key="test"
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
    spent = await spent_today_usd(db_session)
    documents_before = (await db_session.execute(select(func.count()).select_from(Document))).scalar_one()
    db_session.add(_run(cost_usd=0.25))
    await db_session.commit()
    _cap(tmp_path, spent + 0.20)

    response = await client.post("/documents", files={"file": ("r.png", TINY_PNG, "image/png")})

    assert response.status_code == 429
    assert "daily model budget reached" in response.json()["detail"]
    assert 1 <= int(response.headers["retry-after"]) <= 86400
    # Refused before anything was stored: no row, no file.
    assert (await db_session.execute(select(func.count()).select_from(Document))).scalar_one() == documents_before
    assert list(tmp_path.iterdir()) == []


async def test_uploads_under_the_budget_or_with_the_cap_off_go_through(
    client: AsyncClient, db_session: AsyncSession, tmp_path
) -> None:
    spent = await spent_today_usd(db_session)
    _cap(tmp_path, spent + 1.0)
    assert (await client.post("/documents", files={"file": ("a.png", TINY_PNG, "image/png")})).status_code == 201

    db_session.add(_run(cost_usd=2.0))
    await db_session.commit()
    _cap(tmp_path, 0)  # off
    assert (await client.post("/documents", files={"file": ("b.png", TINY_PNG, "image/png")})).status_code == 201


async def test_ask_is_refused_before_any_model_call(
    client: AsyncClient, db_session: AsyncSession, tmp_path
) -> None:
    spent = await spent_today_usd(db_session)
    db_session.add(_run(cost_usd=0.25))
    await db_session.commit()
    _cap(tmp_path, spent + 0.20)
    create = AsyncMock()
    app.dependency_overrides[get_anthropic_client] = lambda: SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=create))
    )

    response = await client.post("/ask", json={"question": "How much did I spend?"})

    assert response.status_code == 429 and "retry-after" in response.headers
    create.assert_not_awaited()


async def test_stats_report_the_budget(client: AsyncClient, db_session: AsyncSession, tmp_path) -> None:
    db_session.add(_run(cost_usd=0.125))
    await db_session.commit()
    _cap(tmp_path, 7.5)

    budget = (await client.get("/stats")).json()["budget"]

    assert budget["daily_budget_usd"] == 7.5
    assert budget["spent_today_usd"] >= 0.125
    assert 1 <= budget["resets_in_seconds"] <= 86400

    _cap(tmp_path, 0)
    assert (await client.get("/stats")).json()["budget"]["daily_budget_usd"] is None


def test_a_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError):
        Settings(daily_budget_usd=-1)
