"""The current, human-verified view of each extracted document.

Stored ExtractedField rows keep the model's original answer and any human
correction side by side (see app/routers/review.py). Anything that reads
extracted data *as data* -- the eval harvester, the agent's structured
query tool -- wants one value per field: the correction if a reviewer
made one, otherwise the model's value. This module is that rule, in one
place.
"""

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, ExtractedField, Extraction


def leaf_value(stored: Any) -> Any:
    """The plain value out of a stored {'value', 'confidence'} leaf."""
    if isinstance(stored, dict) and "value" in stored:
        return stored["value"]
    return None


def line_item_rows(stored: Any) -> list[dict[str, Any]] | None:
    """Convert the stored line_items array (leaf-per-cell, see
    app/extraction.py) into plain rows."""
    if not isinstance(stored, list):
        return None
    rows: list[dict[str, Any]] = []
    for item in stored:
        cells = item if isinstance(item, dict) else {}
        rows.append(
            {key: leaf_value(cells.get(key)) for key in ("description", "quantity", "unit_price", "total")}
        )
    return rows


def field_value(field: ExtractedField) -> Any:
    """The human-verified semantic value for one field: a reviewer's
    correction wins over the model's answer."""
    if field.review_action == "corrected":
        return field.corrected_value
    if field.field_name == "line_items":
        return line_item_rows(field.value)
    return leaf_value(field.value)


@dataclass
class DocumentRecord:
    document_id: uuid.UUID
    filename: str
    fields: dict[str, Any]
    # Fields still flagged for review and not yet resolved: the value is
    # the model's low-confidence answer, and a consumer should say so.
    unreviewed_low_confidence: list[str]


async def load_records(
    session: AsyncSession, document_ids: list[uuid.UUID] | None = None
) -> list[DocumentRecord]:
    """One record per extracted document, from its latest extraction, in
    upload order. document_ids narrows the set (the agent eval scopes
    itself to its own corpus this way)."""
    latest = (
        select(Extraction.id)
        .where(Extraction.document_id == Document.id)
        .order_by(Extraction.created_at.desc())
        .limit(1)
        .correlate(Document)
        .scalar_subquery()
    )
    stmt = (
        select(Document, ExtractedField)
        .join(Extraction, Extraction.id == latest)
        .join(ExtractedField, ExtractedField.extraction_id == Extraction.id)
        .where(Document.status == "extracted")
        # filename breaks created_at ties (rows inserted in one transaction
        # share now()), so the same data always lists in the same order.
        .order_by(Document.created_at, Document.filename, Document.id)
    )
    if document_ids is not None:
        stmt = stmt.where(Document.id.in_(document_ids))

    records: dict[uuid.UUID, DocumentRecord] = {}
    for document, field in (await session.execute(stmt)).all():
        record = records.setdefault(
            document.id, DocumentRecord(document.id, document.filename, {}, [])
        )
        record.fields[field.field_name] = field_value(field)
        if field.needs_review and field.reviewed_at is None:
            record.unreviewed_low_confidence.append(field.field_name)
    return list(records.values())
