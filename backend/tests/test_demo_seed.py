"""Tests for the deterministic demo seed (app/demo_seed.py).

The seed runs inside db_session's rolled-back transaction with the hashing
embedder and a tmp upload dir. The test database can hold rows committed by
other tests, so every assertion is scoped to the seeded document ids (or is
a before/after delta), never a global count.

The guarded entry points refuse any database but the demo one by name, and
this is docpilot_test: tests that need to get past that check patch
demo_seed.DEMO_DATABASE_NAME, and the refusal tests do not.
"""

import importlib.util
import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import demo_seed
from app.budget import day_start, spent_today_usd
from app.config import BACKEND_DIR, Settings
from app.demo_seed import (
    HISTORY_DAYS,
    REPLAY_ARTIFACT,
    STAGED_REVIEW_CASES,
    SeedSummary,
    _wipe,
    demo_document_id,
    remove_stale_uploads,
    reset_demo_data,
    seed_demo_data,
)
from app.extraction import TOP_LEVEL_FIELDS
from app.models import Document, DocumentPage, ExtractedField, Extraction
from app.retrieval.embeddings import HashingEmbedder
from app.routers.documents import ALLOWED_MIME_TYPES

NOW = datetime(2026, 10, 10, 15, 30, tzinfo=UTC)
STAGED = STAGED_REVIEW_CASES[0]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(upload_dir=str(tmp_path), embedding_backend="hashing")


@pytest.fixture
def artifact() -> dict:
    return json.loads(REPLAY_ARTIFACT.read_text(encoding="utf-8"))


@pytest_asyncio.fixture
async def seeded(db_session: AsyncSession, settings: Settings) -> SeedSummary:
    return await seed_demo_data(db_session, settings, HashingEmbedder(), now=NOW)


@pytest.fixture
def demo_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the guarded entry points treat the test database as the demo one."""
    monkeypatch.setattr(demo_seed, "DEMO_DATABASE_NAME", "docpilot_test")
    monkeypatch.setenv("DOCPILOT_DEMO", "1")


async def _count_seeded(session: AsyncSession, seeded: SeedSummary) -> int:
    stmt = select(func.count()).select_from(Document).where(Document.id.in_(list(seeded.document_ids.values())))
    return (await session.execute(stmt)).scalar_one()


async def test_seeds_every_replayed_document_with_all_fields(
    db_session: AsyncSession, seeded: SeedSummary, artifact: dict
) -> None:
    ids = list(seeded.document_ids.values())
    assert len(ids) == len(artifact["per_doc"]) == 25
    assert seeded.document_ids == {e["doc_id"]: demo_document_id(e["doc_id"]) for e in artifact["per_doc"]}

    statuses = (await db_session.execute(select(Document.status).where(Document.id.in_(ids)))).scalars().all()
    assert statuses == ["extracted"] * 25

    rows = (
        await db_session.execute(
            select(Extraction.document_id, func.count(ExtractedField.id))
            .join(ExtractedField, ExtractedField.extraction_id == Extraction.id)
            .where(Extraction.document_id.in_(ids))
            .group_by(Extraction.document_id)
        )
    ).all()
    assert len(rows) == 25
    assert {count for _, count in rows} == {len(TOP_LEVEL_FIELDS)} == {7}


async def test_extraction_replays_the_artifact_with_provenance(
    db_session: AsyncSession, seeded: SeedSummary, artifact: dict
) -> None:
    entry = artifact["per_doc"][5]
    assert entry["doc_id"] != STAGED.doc_id
    extraction = (
        await db_session.execute(
            select(Extraction).where(Extraction.document_id == demo_document_id(entry["doc_id"]))
        )
    ).scalar_one()
    assert extraction.model == artifact["model"]
    assert extraction.prompt_version == artifact["prompt_version"]
    assert float(extraction.cost_usd) == entry["cost_usd"]
    assert extraction.latency_ms == entry["latency_ms"]
    assert (extraction.input_tokens, extraction.output_tokens) == (entry["input_tokens"], entry["output_tokens"])
    # Exactly the 7 model leaves plus the provenance key.
    assert set(extraction.raw_response) == {*TOP_LEVEL_FIELDS, "_demo_replay"}
    assert extraction.raw_response["_demo_replay"] == {
        "artifact": REPLAY_ARTIFACT.name,
        "doc_id": entry["doc_id"],
        "staged": {},
    }
    for name in TOP_LEVEL_FIELDS:
        assert extraction.raw_response[name] == {
            "value": entry["fields"][name]["actual"],
            "confidence": entry["fields"][name]["confidence"],
        }
    assert seeded.label_fallbacks == 0
    assert seeded.total_cost_usd == pytest.approx(artifact["total_cost_usd"])


async def test_pending_review_is_the_low_confidence_fields_plus_the_staged_ones(
    db_session: AsyncSession, seeded: SeedSummary, artifact: dict, settings: Settings
) -> None:
    staged = {(case.doc_id, case.field_name) for case in STAGED_REVIEW_CASES}
    expected = {
        (entry["doc_id"], name)
        for entry in artifact["per_doc"]
        for name, field in entry["fields"].items()
        if field["confidence"] < settings.review_threshold
    } | staged
    # The replayed run itself is uncertain about at least one field, so the
    # expectation is not just the staged cases.
    assert expected - staged

    pending = (
        await db_session.execute(
            select(Extraction.document_id, ExtractedField.field_name)
            .join(ExtractedField, ExtractedField.extraction_id == Extraction.id)
            .where(
                Extraction.document_id.in_(seeded.document_ids.values()),
                ExtractedField.needs_review.is_(True),
                ExtractedField.reviewed_at.is_(None),
            )
        )
    ).all()
    by_demo_id = {demo_id: doc_id for doc_id, demo_id in seeded.document_ids.items()}
    assert {(by_demo_id[document_id], name) for document_id, name in pending} == expected
    assert seeded.pending_review == len(expected)
    assert seeded.staged == len(STAGED_REVIEW_CASES)


async def test_staged_subtotal_for_016_is_pending_and_recorded_in_provenance(
    db_session: AsyncSession, seeded: SeedSummary, artifact: dict
) -> None:
    document_id = seeded.document_ids[STAGED.doc_id]
    field = (
        await db_session.execute(
            select(ExtractedField)
            .join(Extraction, ExtractedField.extraction_id == Extraction.id)
            .where(Extraction.document_id == document_id, ExtractedField.field_name == "subtotal")
        )
    ).scalar_one()
    assert field.value == {"value": 46.25, "confidence": 0.68}
    assert field.confidence == 0.68
    assert field.needs_review is True

    replayed = next(e for e in artifact["per_doc"] if e["doc_id"] == STAGED.doc_id)["fields"]["subtotal"]
    extraction = (
        await db_session.execute(select(Extraction).where(Extraction.document_id == document_id))
    ).scalar_one()
    # The model's own leaf is untouched; the override is recorded beside it.
    assert extraction.raw_response["subtotal"] == {
        "value": replayed["actual"],
        "confidence": replayed["confidence"],
    }
    assert extraction.raw_response["_demo_replay"]["staged"] == {
        "subtotal": {
            "replayed": {"value": replayed["actual"], "confidence": replayed["confidence"]},
            "staged": {"value": 46.25, "confidence": 0.68},
            "reason": STAGED.reason,
        }
    }


async def test_nothing_is_spent_today_and_history_is_staggered(
    db_session: AsyncSession, settings: Settings
) -> None:
    before = await spent_today_usd(db_session, NOW)
    seeded = await seed_demo_data(db_session, settings, HashingEmbedder(), now=NOW)
    assert seeded.total_cost_usd > 0
    assert await spent_today_usd(db_session, NOW) == pytest.approx(before, abs=1e-9)

    ids = list(seeded.document_ids.values())
    today = day_start(NOW)
    for model, id_column in (
        (Document, Document.id),
        (Extraction, Extraction.document_id),
        (DocumentPage, DocumentPage.document_id),
    ):
        stamps = (await db_session.execute(select(model.created_at).where(id_column.in_(ids)))).scalars().all()
        assert stamps
        assert all(today - timedelta(days=HISTORY_DAYS) <= stamp < today for stamp in stamps)

    created = (await db_session.execute(select(Document.created_at).where(Document.id.in_(ids)))).scalars().all()
    assert len(set(created)) == 25
    assert len({stamp.date() for stamp in created}) == HISTORY_DAYS


async def test_images_are_stored_where_the_file_endpoint_serves_them(
    db_session: AsyncSession, seeded: SeedSummary, tmp_path
) -> None:
    documents = (
        await db_session.execute(select(Document).where(Document.id.in_(list(seeded.document_ids.values()))))
    ).scalars().all()
    for document in documents:
        extension = ALLOWED_MIME_TYPES[document.mime_type]
        path = tmp_path / f"{document.id}{extension}"
        assert document.storage_path == str(path)
        assert path.is_file()
        assert document.filename.endswith(extension)


async def test_gold_pages_are_indexed(db_session: AsyncSession, seeded: SeedSummary) -> None:
    sources = (
        await db_session.execute(
            select(DocumentPage.source).where(DocumentPage.document_id.in_(list(seeded.document_ids.values())))
        )
    ).scalars().all()
    assert sources == ["gold"] * 25


async def test_ids_are_the_same_across_a_wipe_and_a_second_seed(
    db_session: AsyncSession, settings: Settings, seeded: SeedSummary, tmp_path
) -> None:
    await _wipe(db_session, tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert (await db_session.execute(select(func.count()).select_from(Document))).scalar_one() == 0

    again = await seed_demo_data(db_session, settings, HashingEmbedder(), now=NOW)
    assert again.document_ids == seeded.document_ids
    assert len(list(tmp_path.iterdir())) == 25


async def test_reset_refuses_the_wrong_database_or_a_missing_env_var(
    db_session: AsyncSession, settings: Settings, seeded: SeedSummary, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # The env var alone is not enough: this is docpilot_test, not the demo database.
    monkeypatch.setenv("DOCPILOT_DEMO", "1")
    with pytest.raises(RuntimeError, match="docpilot_test"):
        await reset_demo_data(db_session, settings)

    # Right database, but the env var is missing or not "1".
    monkeypatch.setattr(demo_seed, "DEMO_DATABASE_NAME", "docpilot_test")
    monkeypatch.delenv("DOCPILOT_DEMO", raising=False)
    with pytest.raises(RuntimeError, match="refusing"):
        await reset_demo_data(db_session, settings)
    monkeypatch.setenv("DOCPILOT_DEMO", "0")
    with pytest.raises(RuntimeError, match="refusing"):
        await reset_demo_data(db_session, settings)

    assert len(list(tmp_path.iterdir())) == 25
    assert await _count_seeded(db_session, seeded) == 25


@pytest.mark.usefixtures("demo_database")
async def test_reset_deletes_rows_and_leaves_the_files_for_after_the_commit(
    db_session: AsyncSession, settings: Settings, seeded: SeedSummary, tmp_path
) -> None:
    old_upload = tmp_path / f"{uuid.uuid4()}.jpg"
    old_upload.write_bytes(b"old upload")

    await reset_demo_data(db_session, settings)
    assert (await db_session.execute(select(func.count()).select_from(Document))).scalar_one() == 0
    assert len(list(tmp_path.iterdir())) == 26

    remove_stale_uploads(settings, seeded)
    assert not old_upload.exists()
    assert len(list(tmp_path.iterdir())) == 25


@pytest.mark.usefixtures("demo_database")
@pytest.mark.parametrize("foreign", ["notes.txt", "model.onnx", f"{uuid.uuid4()}.exe", "not-a-uuid.png", "subdir"])
async def test_an_upload_dir_holding_anything_but_uploads_is_never_touched(
    db_session: AsyncSession, settings: Settings, seeded: SeedSummary, tmp_path, foreign: str
) -> None:
    if foreign == "subdir":
        (tmp_path / foreign).mkdir()
    else:
        (tmp_path / foreign).write_text("not an upload")

    with pytest.raises(RuntimeError, match="not uploads"):
        await reset_demo_data(db_session, settings)
    with pytest.raises(RuntimeError, match="not uploads"):
        remove_stale_uploads(settings, seeded)
    with pytest.raises(RuntimeError, match="not uploads"):
        await _wipe(db_session, tmp_path)
    with pytest.raises(RuntimeError, match="not uploads"):
        await seed_demo_data(db_session, settings, HashingEmbedder(), now=NOW)

    assert len(list(tmp_path.iterdir())) == 26
    assert await _count_seeded(db_session, seeded) == 25


def _load_cli():
    spec = importlib.util.spec_from_file_location("seed_demo_cli", BACKEND_DIR / "scripts" / "seed_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_cli_refuses_a_database_that_is_not_the_demo_one(
    db_session: AsyncSession, settings: Settings, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCPILOT_DEMO", "1")
    cli = _load_cli()
    for reset in (False, True):
        with pytest.raises(RuntimeError, match="docpilot_test"):
            await cli.run(db_session, settings, reset=reset)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.usefixtures("demo_database")
async def test_cli_refuses_a_non_empty_database_without_reset_and_replaces_it_with_reset(
    db_session: AsyncSession, settings: Settings, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    db_session.add(Document(filename="x.png", mime_type="image/png", storage_path="/x", status="uploaded"))
    await db_session.flush()

    assert await cli.run(db_session, settings, reset=False) == 1
    assert "--reset" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []

    assert await cli.run(db_session, settings, reset=True) == 0
    assert "Seeded 25 documents, 4 fields pending review" in capsys.readouterr().out
    assert (await db_session.execute(select(func.count()).select_from(Document))).scalar_one() == 25
    assert len(list(tmp_path.iterdir())) == 25
