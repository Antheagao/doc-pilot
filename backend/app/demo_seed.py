"""Seed the demo database from the committed Sonnet extraction run.

The demo stack (docker-compose.demo.yml) needs a believable history without
spending a cent: every document is the eval corpus image, and its extraction
replays what claude-sonnet-5 really returned for it (values, confidences,
cost, latency, tokens) from the committed artifact in evals/results/. The
page text is the gold text, indexed with the same embedder the API queries
with. The only invented data is STAGED_REVIEW_CASES, marked as such.

Document ids are uuid5 of the doc_id, so recorded scenarios can deep-link a
document across resets, and created_at is staggered over the 14 days before
today so the dashboard has per-day history and today's recorded spend is 0.

Everything destructive is bound to the demo database by name
(DEMO_DATABASE_NAME, set only in docker-compose.demo.yml), because the dev
database lives on the same Docker host: a reset or seed against any other
database refuses.

Like app/evals/corpus.py, nothing here commits: the caller owns the
transaction. Unlike corpus.py the extraction is NOT label-perfect, which is
why the agent eval keeps using seed_labeled_corpus.
"""

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.budget import day_start
from app.config import Settings, resolve_upload_dir
from app.db import Base
from app.evals.dataset import EVALS_DIR, EvalCase, load_cases
from app.evals.retrieval import load_gold_corpus
from app.extraction import TOP_LEVEL_FIELDS, build_extracted_fields
from app.models import Document, DocumentPage, ExtractedField, Extraction
from app.retrieval.embeddings import Embedder, build_embedder
from app.retrieval.indexing import ChunkingConfig, PageText, index_document_pages
from app.routers.documents import ALLOWED_MIME_TYPES

REPLAY_ARTIFACT = EVALS_DIR / "results" / "20260731T064853Z_claude-sonnet-5_extract_v1.json"

# Random but fixed: changing it changes every demo document id.
DEMO_NAMESPACE = uuid.UUID("5f0c1d6e-7a3b-4c52-9e8d-2b4f6a1c3d70")

# Documents are spread over this many days ending yesterday, oldest first,
# so none of them lands in today's spend window (budget.spent_today_usd).
HISTORY_DAYS = 14

DEMO_ENV_VAR = "DOCPILOT_DEMO"
DEMO_DATABASE_NAME = "docpilot_demo"

# The only files the seed ever deletes in the upload directory:
# <uuid><extension of an allowed mime type>, exactly what uploads and the
# seed store. Anything else there belongs to someone else.
_UPLOAD_SUFFIXES = frozenset(ALLOWED_MIME_TYPES.values())


@dataclass(frozen=True)
class StagedReviewCase:
    """STAGED, not model output: a field forced to a low-confidence misread
    so the review queue and the correction scenario have a known case. The
    replayed Sonnet run is nearly perfect, so on its own the queue would
    hold only its three genuinely uncertain fields."""

    doc_id: str
    field_name: str
    value: float
    confidence: float
    reason: str


STAGED_REVIEW_CASES = (
    # The session-5 staged misread: the bakery note's subtotal read as 46.25
    # instead of 45.25, with the doubt a real model would show.
    StagedReviewCase(
        "016-handwriting-bakery-note", "subtotal", 46.25, 0.68, "review-correction scenario"
    ),
)


@dataclass
class SeedSummary:
    document_ids: dict[str, uuid.UUID]
    pending_review: int
    total_cost_usd: float
    # Entries whose artifact record had no "actual" and fell back to the label.
    label_fallbacks: int
    staged: int
    stored_files: list[Path]


def demo_document_id(doc_id: str) -> uuid.UUID:
    return uuid.uuid5(DEMO_NAMESPACE, doc_id)


def _leaf(value: Any) -> dict:
    return {"value": value, "confidence": 1.0}


def _label_tool_input(case: EvalCase) -> dict:
    """The label as a tool_input at confidence 1.0, shaped like
    seed_labeled_corpus's fields."""
    tool_input: dict[str, Any] = {}
    for name in TOP_LEVEL_FIELDS:
        value = case.fields[name]
        if name == "line_items":
            value = [{key: _leaf(cell) for key, cell in row.items()} for row in value or []]
        tool_input[name] = _leaf(value)
    return tool_input


def _replayed_tool_input(entry: dict, case: EvalCase) -> tuple[dict, bool]:
    """Rebuild the tool_input the model returned from an artifact per_doc
    entry: each field's `actual` and `confidence`. Returns (tool_input,
    used_label_fallback). An entry with no `actual` key at all (an older
    artifact shape) falls back to the label at confidence 1.0 for the whole
    document; the committed Sonnet run has none, and the seed counts them."""
    fields = entry["fields"]
    if any("actual" not in fields[name] for name in TOP_LEVEL_FIELDS):
        return _label_tool_input(case), True
    tool_input = {
        name: {"value": fields[name]["actual"], "confidence": fields[name]["confidence"]}
        for name in TOP_LEVEL_FIELDS
    }
    return tool_input, False


def _created_at(index: int, count: int, today: datetime) -> datetime:
    """Deterministic UTC timestamp for the index-th of `count` documents:
    oldest first, HISTORY_DAYS down to 1 days back, at a minute of the
    working day that no other document shares (97 is coprime with 480), so
    the document list order never depends on tie-breaking."""
    days_ago = HISTORY_DAYS - (index * HISTORY_DAYS) // count
    return today - timedelta(days=days_ago) + timedelta(hours=9, minutes=(index * 97) % 480)


def _apply_staged_cases(
    doc_id: str, fields: list[ExtractedField], review_threshold: float
) -> dict[str, dict]:
    """Override this document's staged fields in memory, before they are
    inserted, and return the provenance for raw_response: what the model
    returned and what the demo shows instead."""
    staged = {}
    for case in STAGED_REVIEW_CASES:
        if case.doc_id != doc_id:
            continue
        field = next(f for f in fields if f.field_name == case.field_name)
        replayed = {"value": field.value["value"], "confidence": field.confidence}
        field.value = {"value": case.value, "confidence": case.confidence}
        field.confidence = case.confidence
        field.needs_review = case.confidence < review_threshold
        staged[case.field_name] = {
            "replayed": replayed,
            "staged": field.value,
            "reason": case.reason,
        }
    return staged


async def seed_demo_data(
    session: AsyncSession,
    settings: Settings,
    embedder: Embedder | None = None,
    now: datetime | None = None,
) -> SeedSummary:
    """Add the demo documents, replayed extractions, copied images and gold
    page index. `now` only anchors the created_at stagger. Callers that seed
    a real database check ensure_demo_database first."""
    artifact = json.loads(REPLAY_ARTIFACT.read_text(encoding="utf-8"))
    cases = {case.doc_id: case for case in load_cases()}
    gold_text = {doc.doc_id: doc.text for doc in load_gold_corpus(list(cases.values()))}
    embedder = embedder or build_embedder(settings)
    chunking = ChunkingConfig.from_settings(settings)
    today = day_start(now or datetime.now(UTC))
    upload_dir = resolve_upload_dir(settings)
    _unreferenced_uploads(upload_dir, set())  # refuse before copying into a foreign directory
    upload_dir.mkdir(parents=True, exist_ok=True)

    entries = sorted(artifact["per_doc"], key=lambda e: e["doc_id"])
    unknown = {case.doc_id for case in STAGED_REVIEW_CASES} - {e["doc_id"] for e in entries}
    if unknown:
        raise ValueError(f"staged review cases name documents not in the replay: {sorted(unknown)}")

    document_ids: dict[str, uuid.UUID] = {}
    stored_files: list[Path] = []
    pending = 0
    staged_count = 0
    label_fallbacks = 0
    total_cost = 0.0
    for index, entry in enumerate(entries):
        doc_id = entry["doc_id"]
        if entry["status"] != "scored" or doc_id not in cases or doc_id not in gold_text:
            raise ValueError(f"{doc_id}: not a scored replay entry with a label and gold text")
        case = cases[doc_id]
        tool_input, fallback = _replayed_tool_input(entry, case)
        label_fallbacks += fallback

        document_id = demo_document_id(doc_id)
        extension = ALLOWED_MIME_TYPES[case.mime_type]
        stored = upload_dir / f"{document_id}{extension}"
        shutil.copyfile(case.image_path, stored)
        stored_files.append(stored)

        uploaded_at = _created_at(index, len(entries), today)
        extracted_at = uploaded_at + timedelta(milliseconds=entry["latency_ms"]) + timedelta(seconds=1)
        # Flushed parent first throughout: the models declare no
        # relationships, so a single flush would not order the inserts.
        session.add(
            Document(
                id=document_id,
                filename=f"{doc_id}{extension}",
                mime_type=case.mime_type,
                storage_path=str(stored),
                status="extracted",
                created_at=uploaded_at,
            )
        )
        await session.flush()
        extraction_id = uuid.uuid5(DEMO_NAMESPACE, f"extraction:{doc_id}")
        fields = build_extracted_fields(extraction_id, tool_input, settings.review_threshold)
        staged = _apply_staged_cases(doc_id, fields, settings.review_threshold)
        session.add(
            Extraction(
                id=extraction_id,
                document_id=document_id,
                prompt_version=artifact["prompt_version"],
                model=artifact["model"],
                # The model's leaves stay exactly as returned; a staged
                # override is recorded beside them, never merged into them.
                raw_response={
                    **tool_input,
                    "_demo_replay": {
                        "artifact": REPLAY_ARTIFACT.name,
                        "doc_id": doc_id,
                        "staged": staged,
                    },
                },
                input_tokens=entry["input_tokens"],
                output_tokens=entry["output_tokens"],
                cost_usd=entry["cost_usd"],
                latency_ms=entry["latency_ms"],
                created_at=extracted_at,
            )
        )
        await session.flush()
        session.add_all(fields)
        await session.flush()
        pending += sum(f.needs_review for f in fields)
        staged_count += len(staged)
        document_ids[doc_id] = document_id
        total_cost += entry["cost_usd"]

        await index_document_pages(
            session,
            document_id,
            [PageText(page_number=1, text=gold_text[doc_id], source="gold")],
            embedder,
            chunking,
        )
        await session.execute(
            update(DocumentPage)
            .where(DocumentPage.document_id == document_id)
            .values(created_at=extracted_at + timedelta(seconds=3))
        )

    await session.flush()
    return SeedSummary(
        document_ids, pending, round(total_cost, 6), label_fallbacks, staged_count, stored_files
    )


async def ensure_demo_database(session: AsyncSession) -> None:
    """Refuse unless the session is bound to the demo database. The check is
    the database's identity, not a flag: the dev database shares the host."""
    bind = await session.connection()
    database = bind.engine.url.database
    if database != DEMO_DATABASE_NAME:
        raise RuntimeError(f"refusing to touch database {database!r}")


def _is_upload_file(path: Path) -> bool:
    if path.is_symlink() or not path.is_file() or path.suffix not in _UPLOAD_SUFFIXES:
        return False
    try:
        uuid.UUID(path.stem)
    except ValueError:
        return False
    return True


def _unreferenced_uploads(upload_dir: Path, keep: set[Path]) -> list[Path]:
    """The upload-shaped files in upload_dir that are not in `keep`. Raises,
    before anything is deleted, if the directory holds anything else: a
    subdirectory, a symlink, or a file that is not <uuid><image extension>
    (an UPLOAD_DIR typo could otherwise point at the shared embedding
    cache)."""
    if not upload_dir.is_dir():
        return []
    stale = []
    foreign = []
    for child in sorted(upload_dir.iterdir()):
        if not _is_upload_file(child):
            foreign.append(child.name)
        elif child not in keep:
            stale.append(child)
    if foreign:
        raise RuntimeError(f"{upload_dir} holds files that are not uploads: {foreign[:5]}")
    return stale


async def _delete_rows(session: AsyncSession) -> None:
    # Row deletes, children first: unlike TRUNCATE they take only row locks,
    # so they cannot deadlock with a parallel session on the same database.
    for table in reversed(Base.metadata.sorted_tables):
        await session.execute(table.delete())


async def _wipe(session: AsyncSession, upload_dir: Path) -> None:
    """Delete every app row and every upload file, with no identity checks
    (reset_demo_data and remove_stale_uploads are the guarded entry points)."""
    stale = _unreferenced_uploads(upload_dir, set())
    await _delete_rows(session)
    for path in stale:
        path.unlink()


async def reset_demo_data(session: AsyncSession, settings: Settings) -> None:
    """Delete every app row, in the caller's transaction. Refuses unless the
    session is bound to the demo database and DOCPILOT_DEMO=1 (only the demo
    seed service sets it), and unless the upload directory holds nothing but
    uploads.

    The files are deliberately NOT deleted here: remove_stale_uploads does
    that after the caller commits the new rows, so a seed that fails leaves
    the old database and its images intact together.
    """
    await ensure_demo_database(session)
    if os.environ.get(DEMO_ENV_VAR) != "1":
        raise RuntimeError("refusing to reset this database")
    _unreferenced_uploads(resolve_upload_dir(settings), set())
    await _delete_rows(session)


def remove_stale_uploads(settings: Settings, summary: SeedSummary) -> None:
    """After the seed is committed: delete the uploads it did not write (the
    previous dataset's, and any real ones)."""
    for path in _unreferenced_uploads(resolve_upload_dir(settings), set(summary.stored_files)):
        path.unlink()
