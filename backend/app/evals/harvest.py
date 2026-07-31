"""Harvest human review resolutions into new eval label cases.

This closes the loop the review queue opened: once every flagged field on
a document has been resolved by a human, the document carries a complete
human-verified answer key -- exactly the shape of an evals/labels/ case.
Harvesting copies the uploaded image into evals/docs/ and writes a label
JSON whose `fields` are the human truth:

- review_action == 'corrected'  -> the human's corrected_value
- review_action == 'approved'   -> the model's extracted value, confirmed
- not flagged (high confidence) -> the model's extracted value as-is

Only *fully* reviewed documents are harvested: a document with any
needs_review field still pending has an incomplete answer key, and a
half-human, half-guessed label would poison the eval set. High-confidence
unflagged fields are accepted deliberately -- trusting them is the same
bet the whole pipeline makes by only routing low-confidence fields to
humans in the first place.

Every written label is validated with the same loud validator the eval
loader uses (app.evals.dataset._load_case) before it is kept; a label
that fails validation (e.g. a hand-typed date in the wrong format) is
deleted again and reported, never left half-broken in the dataset.

Idempotence: each harvested label records the source document's UUID in a
`source_document_id` top-level key (dataset.py tolerates extra top-level
keys), and the harvester skips any document whose UUID already appears in
an existing label -- re-running after new reviews only exports new
documents.
"""

import json
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.evals import dataset
from app.extraction import TOP_LEVEL_FIELDS
from app.models import Document, ExtractedField, Extraction

HARVEST_SOURCE = "human-review"
HARVEST_DIFFICULTY = "review"

_MIME_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

_DOC_ID_PREFIX_RE = re.compile(r"^(\d{3})-")
_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class HarvestOutcome:
    """One document's harvest result, for CLI reporting."""

    document_id: uuid.UUID
    doc_id: str | None  # the new label's doc_id when harvested
    status: str  # 'harvested' | 'skipped_existing' | 'skipped_pending' | 'invalid'
    detail: str = ""


def _slugify(filename: str) -> str:
    stem = Path(filename).stem.lower()
    slug = _SLUG_STRIP_RE.sub("-", stem).strip("-")
    return slug or "document"


def _next_doc_number(labels_dir: Path) -> int:
    """One past the highest NNN- prefix among existing labels, so
    harvested cases number on from the generated corpus."""
    highest = 0
    for path in labels_dir.glob("*.json"):
        match = _DOC_ID_PREFIX_RE.match(path.stem)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def _existing_source_ids(labels_dir: Path) -> set[str]:
    """Document UUIDs already harvested, read from labels'
    source_document_id keys. Unreadable/label-shaped-but-broken files are
    ignored here -- validation of *new* labels is loud, but a dedupe scan
    shouldn't crash on an unrelated bad file."""
    seen: set[str] = set()
    for path in labels_dir.glob("*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source_id = raw.get("source_document_id")
        if isinstance(source_id, str):
            seen.add(source_id)
    return seen


def _semantic_leaf_value(stored: Any) -> Any:
    """The plain value out of a stored {'value', 'confidence'} leaf."""
    if isinstance(stored, dict) and "value" in stored:
        return stored["value"]
    return None


def _semantic_line_items(stored: Any) -> list[dict[str, Any]] | None:
    """Convert the stored line_items array (leaf-per-cell, see
    app/extraction.py) into the plain rows the label format uses."""
    if not isinstance(stored, list):
        return None
    rows: list[dict[str, Any]] = []
    for item in stored:
        cells = item if isinstance(item, dict) else {}
        rows.append(
            {
                key: _semantic_leaf_value(cells.get(key))
                for key in ("description", "quantity", "unit_price", "total")
            }
        )
    return rows


def _human_truth(field: ExtractedField) -> Any:
    """The human-verified semantic value for one field."""
    if field.review_action == "corrected":
        return field.corrected_value
    if field.field_name == "line_items":
        return _semantic_line_items(field.value)
    return _semantic_leaf_value(field.value)


async def _reviewable_documents(
    session: AsyncSession,
) -> list[tuple[Document, Extraction, list[ExtractedField]]]:
    """Extracted documents with their latest extraction's fields."""
    documents = (
        (
            await session.execute(
                select(Document)
                .where(Document.status == "extracted")
                .order_by(Document.created_at)
            )
        )
        .scalars()
        .all()
    )

    results = []
    for document in documents:
        extraction = (
            (
                await session.execute(
                    select(Extraction)
                    .where(Extraction.document_id == document.id)
                    .order_by(Extraction.created_at.desc())
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )
        if extraction is None:
            continue
        fields = (
            (
                await session.execute(
                    select(ExtractedField).where(
                        ExtractedField.extraction_id == extraction.id
                    )
                )
            )
            .scalars()
            .all()
        )
        results.append((document, extraction, list(fields)))
    return results


async def harvest_corrections(
    session: AsyncSession,
    evals_dir: Path | None = None,
) -> list[HarvestOutcome]:
    """Export every fully-reviewed, not-yet-harvested document as an eval
    case under evals_dir (defaults to the repo's evals/). Returns one
    outcome per candidate document; the CLI prints them, tests assert on
    them."""
    evals_dir = evals_dir if evals_dir is not None else dataset.EVALS_DIR
    labels_dir = evals_dir / "labels"
    docs_dir = evals_dir / "docs"
    labels_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    already_harvested = _existing_source_ids(labels_dir)
    next_number = _next_doc_number(labels_dir)

    outcomes: list[HarvestOutcome] = []
    for document, extraction, fields in await _reviewable_documents(session):
        if str(document.id) in already_harvested:
            outcomes.append(
                HarvestOutcome(document.id, None, "skipped_existing")
            )
            continue

        pending = [
            f.field_name for f in fields if f.needs_review and f.reviewed_at is None
        ]
        if pending:
            outcomes.append(
                HarvestOutcome(
                    document.id,
                    None,
                    "skipped_pending",
                    f"unreviewed field(s): {', '.join(sorted(pending))}",
                )
            )
            continue

        ext = _MIME_EXT.get(document.mime_type)
        if ext is None:
            # PDFs and anything else the eval corpus doesn't cover yet.
            outcomes.append(
                HarvestOutcome(
                    document.id,
                    None,
                    "invalid",
                    f"unsupported mime type for eval corpus: {document.mime_type}",
                )
            )
            continue

        source_path = Path(document.storage_path)
        if not source_path.is_file():
            outcomes.append(
                HarvestOutcome(
                    document.id, None, "invalid", f"file missing: {source_path}"
                )
            )
            continue

        by_name = {f.field_name: f for f in fields}
        label_fields = {
            name: _human_truth(by_name[name]) if name in by_name else None
            for name in TOP_LEVEL_FIELDS
        }

        doc_id = f"{next_number:03d}-review-{_slugify(document.filename)}"
        label_path = labels_dir / f"{doc_id}.json"
        image_path = docs_dir / f"{doc_id}{ext}"

        shutil.copyfile(source_path, image_path)
        label_path.write_text(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "image": f"docs/{doc_id}{ext}",
                    "mime_type": document.mime_type,
                    "source": HARVEST_SOURCE,
                    "source_document_id": str(document.id),
                    "dataset_version": "v1",
                    "difficulty": HARVEST_DIFFICULTY,
                    "fields": label_fields,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        # Same loud validation the eval loader applies -- a label that
        # would break load_cases() must never be left in the dataset.
        try:
            dataset._load_case(label_path, evals_dir=evals_dir)
        except ValueError as exc:
            label_path.unlink(missing_ok=True)
            image_path.unlink(missing_ok=True)
            outcomes.append(
                HarvestOutcome(document.id, None, "invalid", str(exc))
            )
            continue

        next_number += 1
        outcomes.append(HarvestOutcome(document.id, doc_id, "harvested"))

    return outcomes
