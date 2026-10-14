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
A label must also add up: its values are run through the reconciliation
rules, and one that fails them is refused the same way. Labels are the
consistent baseline the reconciliation eval (app.evals.checks) breaks one
number at a time, and a label that disagrees with itself would score a
correct extraction wrong.

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
from app.evals.checks import reconcile_label
from app.extraction import LINE_ITEM_MONEY_KEYS, MONEY_FIELDS, TOP_LEVEL_FIELDS
from app.models import Document, ExtractedField, Extraction
from app.money import MoneyError, to_decimal, to_label_number
from app.reconcile import STATUS_FAIL
from app.records import field_value

HARVEST_SOURCE = "human-review"
HARVEST_DIFFICULTY = "review"

_MIME_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

# Quantities are numbers on a label too, and become canonical strings once a
# reviewer can correct them.
_LINE_ITEM_NUMBER_KEYS = ("quantity", *LINE_ITEM_MONEY_KEYS)

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


def _label_number(value: Any) -> Any:
    """A money value as the JSON number a label carries: a string correction
    such as "45.25" becomes 45.25, whole amounts become ints, and null stays
    null. Anything that is not a plain amount is returned unchanged so the
    dataset validator rejects the label loudly, with its own message."""
    if value is None:
        return None
    try:
        return to_label_number(to_decimal(value))
    except MoneyError:
        return value


def _label_fields(by_name: dict[str, ExtractedField]) -> dict[str, Any]:
    """The human-verified value of every top-level field, in label form."""
    label_fields = {
        name: field_value(by_name[name]) if name in by_name else None
        for name in TOP_LEVEL_FIELDS
    }
    for name in MONEY_FIELDS:
        label_fields[name] = _label_number(label_fields[name])
    if isinstance(label_fields["line_items"], list):
        # A row that is not an object (a typed correction can be any JSON
        # array) goes through unchanged for the validator to reject.
        label_fields["line_items"] = [
            {
                key: _label_number(cell) if key in _LINE_ITEM_NUMBER_KEYS else cell
                for key, cell in row.items()
            }
            if isinstance(row, dict)
            else row
            for row in label_fields["line_items"]
        ]
    return label_fields


def _label_problem(label_path: Path, evals_dir: Path) -> str | None:
    """Why a freshly written label must not be kept, or None when it is fine.

    The same loud validation the eval loader applies comes first (a label that
    would break load_cases() never stays in the dataset), then the
    reconciliation rules: a label whose own amounts do not add up is refused
    with the failing checks named.
    """
    try:
        case = dataset._load_case(label_path, evals_dir=evals_dir)
    except ValueError as exc:
        return str(exc)
    report = reconcile_label(case.fields)
    if report.status == STATUS_FAIL:
        failing = " ".join(f"{check.id}: {check.detail}" for check in report.failing_checks)
        return f"the reviewed values fail reconciliation ({report.rules_version}): {failing}"
    return None


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
        label_fields = _label_fields(by_name)

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

        problem = _label_problem(label_path, evals_dir)
        if problem is not None:
            label_path.unlink(missing_ok=True)
            image_path.unlink(missing_ok=True)
            outcomes.append(HarvestOutcome(document.id, None, "invalid", problem))
            continue

        next_number += 1
        outcomes.append(HarvestOutcome(document.id, doc_id, "harvested"))

    return outcomes
