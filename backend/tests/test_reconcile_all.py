"""scripts/reconcile_all.py: backfill and recompute every extraction's
reconciliation. The test database can hold rows other tests committed, so every
assertion is scoped to the document this test creates."""

import importlib.util
import uuid
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import BACKEND_DIR, Settings
from app.extraction import build_extracted_fields
from app.models import Document, ExtractedField, Extraction, Reconciliation


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "reconcile_all_cli", BACKEND_DIR / "scripts" / "reconcile_all.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _leaf(value, confidence=0.99):
    return {"value": value, "confidence": confidence}


async def _seed_unreconciled(db_session: AsyncSession, filename: str) -> uuid.UUID:
    """A staged-016 extraction stored the way it was before reconciliation
    existed: confidence flags only, no reconciliations row."""
    rows = [
        {
            "description": _leaf("a"),
            "quantity": _leaf(4),
            "unit_price": _leaf(9.15),
            "total": _leaf(36.6),
        },
        {
            "description": _leaf("b"),
            "quantity": _leaf(1),
            "unit_price": _leaf(8.65),
            "total": _leaf(8.65),
        },
    ]
    tool_input = {
        "vendor": _leaf("Bakery"),
        "document_date": _leaf("2025-03-01"),
        "line_items": _leaf(rows),
        "subtotal": _leaf(46.25, 0.68),
        "tax": _leaf(3.05),
        "total": _leaf(48.3),
        "currency": _leaf("USD"),
    }
    document = Document(
        filename=filename, mime_type="image/png", storage_path="/tmp/x.png", status="extracted"
    )
    db_session.add(document)
    await db_session.flush()
    extraction = Extraction(
        document_id=document.id,
        prompt_version="extract_v1",
        model="claude-sonnet-5",
        raw_response=tool_input,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0,
        latency_ms=1,
    )
    db_session.add(extraction)
    await db_session.flush()
    db_session.add_all(build_extracted_fields(extraction.id, tool_input, 0.8))
    await db_session.flush()
    return extraction.id


def _factory(session: AsyncSession):
    """The script opens one session per extraction; the tests hand it their
    own transaction every time (commit and rollback act on its savepoint)."""

    @asynccontextmanager
    async def open_session():
        yield session

    return open_session


async def test_backfills_a_missing_report_and_is_idempotent(db_session: AsyncSession) -> None:
    cli = _load_cli()
    filename = f"reconcile-all-{uuid.uuid4()}.png"
    extraction_id = await _seed_unreconciled(db_session, filename)
    settings = Settings(review_threshold=0.8)

    done, errors = await cli.reconcile_all(_factory(db_session), settings, dry_run=False)

    assert errors == []
    ours = [o for o in done if o.filename == filename]
    assert len(ours) == 1
    assert ours[0].created is True
    assert ours[0].changes == ("subtotal confidence -> reconciliation:lines_sum",)
    assert ours[0].failing == ("lines_sum", "total_math")
    text = cli.render(ours, errors, dry_run=True)
    assert "Dry run: nothing written." in text
    assert "subtotal confidence -> reconciliation:lines_sum" in text
    assert "lines_sum, total_math" in text

    again, errors = await cli.reconcile_all(_factory(db_session), settings, dry_run=False)

    assert errors == []
    second = [o for o in again if o.filename == filename]
    assert second[0].created is False
    assert second[0].changes == ()
    rows = (
        await db_session.execute(
            select(Reconciliation).where(Reconciliation.extraction_id == extraction_id)
        )
    ).scalars().all()
    assert [row.status for row in rows] == ["fail"]


async def test_a_dry_run_leaves_the_database_as_it_found_it(db_session: AsyncSession) -> None:
    cli = _load_cli()
    filename = f"reconcile-all-{uuid.uuid4()}.png"
    extraction_id = await _seed_unreconciled(db_session, filename)
    await db_session.commit()

    done, _ = await cli.reconcile_all(
        _factory(db_session), Settings(review_threshold=0.8), dry_run=True
    )
    assert any(o.filename == filename and o.created for o in done)

    assert (
        await db_session.execute(
            select(Reconciliation).where(Reconciliation.extraction_id == extraction_id)
        )
    ).scalar_one_or_none() is None
    reasons = {
        f.field_name: f.review_reason
        for f in (
            await db_session.execute(
                select(ExtractedField).where(ExtractedField.extraction_id == extraction_id)
            )
        ).scalars()
        if f.needs_review
    }
    assert reasons == {"subtotal": "confidence"}


async def test_one_failing_extraction_is_reported_and_the_rest_still_run(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    bad = f"reconcile-all-{uuid.uuid4()}.png"
    good = f"reconcile-all-{uuid.uuid4()}.png"
    await _seed_unreconciled(db_session, bad)
    good_id = await _seed_unreconciled(db_session, good)
    await db_session.commit()
    real = cli.refresh_reconciliation
    bad_id = (
        await db_session.execute(
            select(Extraction.id).join(Document, Extraction.document_id == Document.id)
            .where(Document.filename == bad)
        )
    ).scalar_one()

    async def refresh(session, extraction_id, settings):
        if extraction_id == bad_id:
            raise RuntimeError("rules bug")
        return await real(session, extraction_id, settings)

    monkeypatch.setattr(cli, "refresh_reconciliation", refresh)

    done, errors = await cli.reconcile_all(
        _factory(db_session), Settings(review_threshold=0.8), dry_run=False
    )

    assert [e for e in errors if bad in e and "RuntimeError: rules bug" in e]
    assert any(o.filename == good for o in done)
    assert not any(o.filename == bad for o in done)
    assert (
        await db_session.execute(
            select(Reconciliation).where(Reconciliation.extraction_id == good_id)
        )
    ).scalar_one().status == "fail"
