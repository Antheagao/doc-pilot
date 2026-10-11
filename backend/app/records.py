"""The current, human-verified view of each extracted document.

Stored ExtractedField rows keep the model's original answer and any human
correction side by side (see app/routers/review.py). Anything that reads
extracted data *as data* -- the eval harvester, the agent's structured
query tool -- wants one value per field: the correction if a reviewer
made one, otherwise the model's value. This module is that rule, in one
place.
"""

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.extraction import LINE_ITEM_MONEY_KEYS, MONEY_FIELDS
from app.models import Document, ExtractedField, Extraction, Reconciliation
from app.money import (
    CurrencyPrecision,
    MoneyError,
    currency_precision,
    parse_amount_input,
    parse_quantity_input,
    shown_value,
    to_api_string,
    to_decimal,
    to_plain_string,
)
from app.schemas import ExtractedFieldOut, ReconciliationOut

logger = logging.getLogger(__name__)

LINE_ITEM_KEYS = ("description", "quantity", *LINE_ITEM_MONEY_KEYS)

# A currency code as a person types it; case is fixed on the way in. Not
# checked against the ISO 4217 table: that table is a snapshot, and
# `currency_precision` already formats an unlisted code with the default
# digits and says so.
_CURRENCY_CODE_RE = re.compile(r"[A-Za-z]{3}", re.ASCII)


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


# ---- The API view: every number read from a document as a decimal string ----


def precision_for(currency_field: ExtractedField | None) -> CurrencyPrecision:
    """How a document's amounts are formatted and validated, from its
    currency field (a correction wins); the default of two decimal places,
    with the assumption recorded, when there is none."""
    code = field_value(currency_field) if currency_field is not None else None
    return currency_precision(code if isinstance(code, str) else None)


def minor_units_for(currency_field: ExtractedField | None) -> int:
    """The decimal places a document's amounts use (see precision_for)."""
    return precision_for(currency_field).minor_units


def currency_code(currency_field: ExtractedField | None) -> str | None:
    """The document's currency as an upper-case code, or None when unstated."""
    code = field_value(currency_field) if currency_field is not None else None
    return currency_precision(code if isinstance(code, str) else None).code


async def currency_fields(
    session: AsyncSession, extraction_ids: list[uuid.UUID]
) -> dict[uuid.UUID, ExtractedField]:
    """The currency field of each given extraction, keyed by extraction id."""
    if not extraction_ids:
        return {}
    stmt = select(ExtractedField).where(
        ExtractedField.extraction_id.in_(extraction_ids),
        ExtractedField.field_name == "currency",
    )
    return {f.extraction_id: f for f in (await session.execute(stmt)).scalars()}


def _api_cell(key: str, raw: Any, minor_units: int) -> Any:
    """One amount or quantity as its canonical string; null stays null. A
    value that is not a number (garbage the model stored) passes through
    untouched so a later format check can flag it."""
    if raw is None:
        return None
    try:
        d = to_decimal(raw)
    except MoneyError:
        return raw
    if key == "quantity":
        return to_plain_string(d)
    return to_api_string(d, minor_units)


def _api_row(row: Any, minor_units: int, *, leaves: bool) -> Any:
    """A line-item row with its quantity and money cells converted. With
    `leaves` a cell is a stored {value, confidence} leaf and only its
    "value" changes; otherwise it is a plain correction value."""
    if not isinstance(row, dict):
        return row
    out = dict(row)
    for key in LINE_ITEM_KEYS[1:]:
        cell = row.get(key)
        if not leaves:
            if key in row:
                out[key] = _api_cell(key, cell, minor_units)
        elif isinstance(cell, dict) and "value" in cell:
            out[key] = {**cell, "value": _api_cell(key, cell["value"], minor_units)}
    return out


def api_field_value(field_name: str, stored: Any, minor_units: int) -> Any:
    """A field's stored `value` as the API serves it: the subtotal, tax,
    total, and each line item's quantity, unit price, and total become
    decimal strings. Only `leaf["value"]` changes; the other leaf keys
    (confidence, and the evidence Batch B adds) are kept as they are."""
    if field_name == "line_items":
        if not isinstance(stored, list):
            return stored
        return [_api_row(row, minor_units, leaves=True) for row in stored]
    if field_name in MONEY_FIELDS and isinstance(stored, dict) and "value" in stored:
        return {**stored, "value": _api_cell(field_name, stored["value"], minor_units)}
    return stored


def api_corrected_value(field_name: str, corrected: Any, minor_units: int) -> Any:
    """`api_field_value` for a stored correction, which is a plain value, not
    a leaf. Older corrections were JSON numbers and are served as strings
    the same way."""
    if field_name == "line_items":
        if not isinstance(corrected, list):
            return corrected
        return [_api_row(row, minor_units, leaves=False) for row in corrected]
    if field_name in MONEY_FIELDS:
        return _api_cell(field_name, corrected, minor_units)
    return corrected


def field_to_out(field: ExtractedField, minor_units: int) -> ExtractedFieldOut:
    """An extracted field as the API serves it."""
    return ExtractedFieldOut(
        id=field.id,
        field_name=field.field_name,
        value=api_field_value(field.field_name, field.value, minor_units),
        confidence=field.confidence,
        needs_review=field.needs_review,
        review_reason=field.review_reason,
        reviewed_at=field.reviewed_at,
        review_action=field.review_action,
        corrected_value=api_corrected_value(
            field.field_name, field.corrected_value, minor_units
        ),
    )


def reconciliation_to_out(row: Reconciliation) -> ReconciliationOut | None:
    """A stored reconciliation as the API serves it: the report JSON,
    validated, plus when it was computed. A report that does not validate
    (written by a rules version this code does not know) is logged and served
    as no reconciliation, so one bad row never fails the whole document."""
    try:
        return ReconciliationOut.model_validate({**row.report, "computed_at": row.computed_at})
    except ValidationError:
        logger.exception("stored reconciliation %s does not validate; serving none", row.id)
        return None


def _normalize_line_item(
    index: int, row: Any, precision: CurrencyPrecision
) -> dict[str, Any]:
    minor_units = precision.minor_units
    label = f"Line item {index + 1}"
    if not isinstance(row, dict):
        raise MoneyError(f"{label} must be an object with {', '.join(LINE_ITEM_KEYS)}.")
    if set(row) != set(LINE_ITEM_KEYS):
        raise MoneyError(
            f"{label} must have exactly the keys {', '.join(LINE_ITEM_KEYS)}; "
            f"it has {', '.join(sorted(row)) or 'none'}."
        )
    description = row["description"]
    if description is not None and not isinstance(description, str):
        raise MoneyError(f"{label} description must be text or null.")
    out: dict[str, Any] = {"description": description}
    for key in LINE_ITEM_KEYS[1:]:
        raw = row[key]
        try:
            if raw is None:
                out[key] = None
            elif key == "quantity":
                out[key] = to_plain_string(parse_quantity_input(raw))
            else:
                amount = parse_amount_input(raw, minor_units, precision)
                out[key] = to_api_string(amount, minor_units)
        except MoneyError as exc:
            raise MoneyError(f"{label} {key}: {exc}") from exc
    return out


def normalize_correction(field_name: str, raw: Any, precision: CurrencyPrecision) -> Any:
    """Validate a human correction and return what gets stored.

    Money fields take a decimal string or number that fits the currency and
    are stored as the canonical string. line_items takes null or a list of
    objects with exactly description, quantity, unit_price, and total, each
    converted the same way. currency takes a three-letter code, stored in
    upper case. Every other field (vendor, document_date) is stored as sent,
    as it always was. A bad value raises MoneyError, whose message is a
    sentence to show the reviewer.
    """
    if field_name in MONEY_FIELDS:
        if raw is None:
            return None
        minor_units = precision.minor_units
        amount = parse_amount_input(raw, minor_units, precision)
        return to_api_string(amount, minor_units)
    if field_name == "line_items":
        if raw is None:
            return None
        if not isinstance(raw, list):
            raise MoneyError("Line items must be a list, or null for none.")
        return [_normalize_line_item(i, row, precision) for i, row in enumerate(raw)]
    if field_name == "currency":
        if raw is None:
            return None
        code = raw.strip() if isinstance(raw, str) else ""
        if not _CURRENCY_CODE_RE.fullmatch(code):
            raise MoneyError(f"{shown_value(raw)} is not a three-letter currency code, such as USD.")
        return code.upper()
    return raw
