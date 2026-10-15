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

A label has every field of the v2 extraction schema, so only an extraction
made with a schema that has them all (extract_v2) can be harvested: a v1
extraction lacks the new fields, and a label with them guessed as null would
state that the document shows none. It is skipped with the reason, and
re-extracting the document with extract_v2 makes it harvestable.

A harvested label also carries the evidence of the values nobody reviewed, as the
extraction stored it (a reviewed value has none: the person's word replaced the
model's quote), so the label's quotes are checked with its numbers.

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
from app.extraction_fields import ARRAY_CELLS, SCHEMA_V2
from app.models import Document, ExtractedField, Extraction
from app.money import MoneyError, to_decimal, to_label_number
from app.reconcile import STATUS_FAIL
from app.records import field_value, schema_for_stored

HARVEST_SOURCE = "human-review"
HARVEST_DIFFICULTY = "review"

_MIME_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

# The cells of an array field that are numbers on a label: quantities, rates,
# and amounts become canonical strings once a reviewer can correct them.
_ARRAY_NUMBER_KEYS = {
    name: tuple(key for key, kind in cells.items() if kind != "text")
    for name, cells in ARRAY_CELLS.items()
}

_DOC_ID_PREFIX_RE = re.compile(r"^(\d{3})-")
_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class HarvestOutcome:
    """One document's harvest result, for CLI reporting."""

    document_id: uuid.UUID
    doc_id: str | None  # the new label's doc_id when harvested
    # 'harvested' | 'skipped_existing' | 'skipped_pending' | 'skipped_schema' | 'invalid'
    status: str
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
    """The human-verified value of every top-level field, in label form. Labels
    carry the v2 field set."""
    label_fields = {
        name: field_value(by_name[name]) if name in by_name else None
        for name in SCHEMA_V2.field_names
    }
    for name in SCHEMA_V2.money_field_names:
        label_fields[name] = _label_number(label_fields[name])
    for name, number_keys in _ARRAY_NUMBER_KEYS.items():
        rows = label_fields[name]
        if isinstance(rows, list):
            # A row that is not an object (a typed correction can be any JSON
            # array) goes through unchanged for the validator to reject.
            label_fields[name] = [
                {
                    key: _label_number(cell) if key in number_keys else cell
                    for key, cell in row.items()
                }
                if isinstance(row, dict)
                else row
                for row in rows
            ]
    return label_fields


def _label_evidence(
    by_name: dict[str, ExtractedField], label_fields: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """The evidence of the label's values that nobody reviewed: the page and quote
    the extraction stored (already masked of card numbers), for each non-null
    value. A reviewed field, approved or corrected, has none: the person's word
    replaced the model's quote, which may be the very thing they fixed."""
    evidence: dict[str, dict[str, Any]] = {}

    def keep(path: str, stored: Any, value: Any) -> None:
        cited = stored.get("evidence") if isinstance(stored, dict) else None
        if value is None or not isinstance(cited, dict):
            return
        page, quote = cited.get("page"), cited.get("quote")
        if isinstance(page, int) and not isinstance(page, bool) and page >= 1 and isinstance(quote, str):
            evidence[path] = {"page": page, "quote": quote}

    for name in SCHEMA_V2.field_names:
        field = by_name.get(name)
        if field is None or field.reviewed_at is not None:
            continue
        if name in ARRAY_CELLS:
            stored_rows = field.value if isinstance(field.value, list) else []
            label_rows = label_fields[name] if isinstance(label_fields[name], list) else []
            for index, (stored, row) in enumerate(zip(stored_rows, label_rows, strict=False)):
                if not isinstance(stored, dict) or not isinstance(row, dict):
                    continue
                for cell in ARRAY_CELLS[name]:
                    keep(f"{name}[{index}].{cell}", stored.get(cell), row.get(cell))
        else:
            keep(name, field.value, label_fields[name])
    return evidence


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

        schema = schema_for_stored(extraction.prompt_version)
        missing_fields = [name for name in SCHEMA_V2.field_names if name not in schema.field_names]
        if missing_fields:
            outcomes.append(
                HarvestOutcome(
                    document.id,
                    None,
                    "skipped_schema",
                    f"extracted with {extraction.prompt_version}, which has no "
                    f"{', '.join(missing_fields)}; re-extract with {SCHEMA_V2.prompt_version} "
                    "to harvest it",
                )
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
        label_evidence = _label_evidence(by_name, label_fields)

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
                    "dataset_version": "v2",
                    "difficulty": HARVEST_DIFFICULTY,
                    "fields": label_fields,
                    "evidence": label_evidence,
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
