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

from app.extraction_fields import ARRAY_CELLS, SCHEMA_V1, SCHEMAS, ExtractionSchema
from app.models import Document, ExtractedField, Extraction, Reconciliation
from app.money import (
    CurrencyPrecision,
    MoneyError,
    currency_precision,
    parse_amount_input,
    parse_quantity_input,
    parse_rate_input,
    shown_value,
    to_api_string,
    to_decimal,
    to_plain_string,
)
from app.pii import CARD_LAST4_RE, has_long_digit_run
from app.schemas import ExtractedFieldOut, ReconciliationOut

logger = logging.getLogger(__name__)

# What an array field's rows are called in a message to a reviewer: one row,
# and the whole list.
_ARRAY_NOUNS = {
    "line_items": ("Line item", "Line items"),
    "tax_lines": ("Tax line", "Tax lines"),
}

# The cell kinds that are numbers rather than text, and how a person's number
# for each is parsed. A quantity is a count, a rate a printed percent.
_NUMBER_PARSERS = {"quantity": parse_quantity_input, "rate": parse_rate_input}

# A currency code as a person types it; case is fixed on the way in. Not
# checked against the ISO 4217 table: that table is a snapshot, and
# `currency_precision` already formats an unlisted code with the default
# digits and says so.
_CURRENCY_CODE_RE = re.compile(r"[A-Za-z]{3}", re.ASCII)


def schema_for_stored(prompt_version: str) -> ExtractionSchema:
    """The schema to read a stored extraction with. A prompt version the code
    does not register (a row from a newer or removed version) reads as v1 with a
    warning, so one odd row cannot fail an endpoint or the agent's tools. The
    worker and the settings stay strict (`schema_for`): they decide what to
    write, and this only reads.
    """
    schema = SCHEMAS.get(prompt_version)
    if schema is None:
        logger.warning(
            "stored extraction has unknown prompt version %r; reading it with %s",
            prompt_version,
            SCHEMA_V1.prompt_version,
        )
        return SCHEMA_V1
    return schema


def leaf_value(stored: Any) -> Any:
    """The plain value out of a stored {'value', 'confidence'} leaf."""
    if isinstance(stored, dict) and "value" in stored:
        return stored["value"]
    return None


def array_rows(array_name: str, stored: Any) -> list[dict[str, Any]] | None:
    """Convert a stored array field (line_items, tax_lines: a leaf per cell,
    see app/extraction.py) into plain rows keyed by the field's cells."""
    if not isinstance(stored, list):
        return None
    rows: list[dict[str, Any]] = []
    for item in stored:
        cells = item if isinstance(item, dict) else {}
        rows.append({key: leaf_value(cells.get(key)) for key in ARRAY_CELLS[array_name]})
    return rows


def field_value(field: ExtractedField) -> Any:
    """The human-verified semantic value for one field: a reviewer's
    correction wins over the model's answer."""
    if field.review_action == "corrected":
        return field.corrected_value
    if field.field_name in ARRAY_CELLS:
        return array_rows(field.field_name, field.value)
    return leaf_value(field.value)


@dataclass
class DocumentRecord:
    document_id: uuid.UUID
    filename: str
    fields: dict[str, Any]
    # Fields still flagged for review and not yet resolved: the value is
    # the model's low-confidence answer, and a consumer should say so.
    unreviewed_low_confidence: list[str]
    # The prompt version the record was extracted with, which says which fields
    # it can have. The v1 default is for a record built without an extraction
    # (an eval label), which has only the v1 fields.
    prompt_version: str = SCHEMA_V1.prompt_version


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
        select(Document, ExtractedField, Extraction.prompt_version)
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
    for document, field, prompt_version in (await session.execute(stmt)).all():
        record = records.setdefault(
            document.id, DocumentRecord(document.id, document.filename, {}, [], prompt_version)
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


def _api_cell(kind: str, raw: Any, minor_units: int) -> Any:
    """One cell or amount as its canonical string: a "number" (a quantity or a
    rate) in plain form, anything else (an amount) padded to the currency's
    digits; null stays null. A value that is not a number (garbage the model
    stored) passes through untouched so a later format check can flag it."""
    if raw is None:
        return None
    try:
        d = to_decimal(raw)
    except MoneyError:
        return raw
    if kind == "number":
        return to_plain_string(d)
    return to_api_string(d, minor_units)


def _api_row(array_name: str, row: Any, minor_units: int, *, leaves: bool) -> Any:
    """A row of an array field with its number and money cells converted. With
    `leaves` a cell is a stored {value, confidence, ...} leaf and only its
    "value" changes (the evidence passes through); otherwise it is a plain
    correction value."""
    if not isinstance(row, dict):
        return row
    out = dict(row)
    for key, kind in ARRAY_CELLS[array_name].items():
        if kind not in ("number", "money"):
            continue
        cell = row.get(key)
        if not leaves:
            if key in row:
                out[key] = _api_cell(kind, cell, minor_units)
        elif isinstance(cell, dict) and "value" in cell:
            out[key] = {**cell, "value": _api_cell(kind, cell["value"], minor_units)}
    return out


def api_field_value(field_name: str, stored: Any, minor_units: int, schema: ExtractionSchema) -> Any:
    """A field's stored `value` as the API serves it: every amount (the money
    fields of `schema`) and each array row's number and money cells (a line
    item's quantity, unit price, and total; a tax line's rate, base, and
    amount) become decimal strings. Only `leaf["value"]` changes; the other
    leaf keys (confidence, and the evidence) are kept as they are."""
    if field_name in ARRAY_CELLS:
        if not isinstance(stored, list):
            return stored
        return [_api_row(field_name, row, minor_units, leaves=True) for row in stored]
    if field_name in schema.money_field_names and isinstance(stored, dict) and "value" in stored:
        return {**stored, "value": _api_cell("money", stored["value"], minor_units)}
    return stored


def api_corrected_value(
    field_name: str, corrected: Any, minor_units: int, schema: ExtractionSchema
) -> Any:
    """`api_field_value` for a stored correction, which is a plain value, not
    a leaf. Older corrections were JSON numbers and are served as strings
    the same way."""
    if field_name in ARRAY_CELLS:
        if not isinstance(corrected, list):
            return corrected
        return [_api_row(field_name, row, minor_units, leaves=False) for row in corrected]
    if field_name in schema.money_field_names:
        return _api_cell("money", corrected, minor_units)
    return corrected


def field_to_out(
    field: ExtractedField, minor_units: int, schema: ExtractionSchema
) -> ExtractedFieldOut:
    """An extracted field as the API serves it, under the schema of the
    extraction it belongs to."""
    return ExtractedFieldOut(
        id=field.id,
        field_name=field.field_name,
        value=api_field_value(field.field_name, field.value, minor_units, schema),
        confidence=field.confidence,
        needs_review=field.needs_review,
        review_reason=field.review_reason,
        reviewed_at=field.reviewed_at,
        review_action=field.review_action,
        corrected_value=api_corrected_value(
            field.field_name, field.corrected_value, minor_units, schema
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


def _normalize_array_row(
    array_name: str, index: int, row: Any, precision: CurrencyPrecision
) -> dict[str, Any]:
    minor_units = precision.minor_units
    cells = ARRAY_CELLS[array_name]
    keys = ", ".join(cells)
    label = f"{_ARRAY_NOUNS[array_name][0]} {index + 1}"
    if not isinstance(row, dict):
        raise MoneyError(f"{label} must be an object with {keys}.")
    if set(row) != set(cells):
        raise MoneyError(
            f"{label} must have exactly the keys {keys}; "
            f"it has {', '.join(sorted(row)) or 'none'}."
        )
    out: dict[str, Any] = {}
    for key, kind in cells.items():
        raw = row[key]
        try:
            if raw is None:
                out[key] = None
            elif kind == "text":
                if not isinstance(raw, str):
                    raise MoneyError(f"{label} {key} must be text or null.")
                out[key] = raw
            elif kind == "number":
                out[key] = to_plain_string(_NUMBER_PARSERS[key](raw))
            else:
                amount = parse_amount_input(raw, minor_units, precision)
                out[key] = to_api_string(amount, minor_units)
        except MoneyError as exc:
            if kind == "text":
                raise
            raise MoneyError(f"{label} {key}: {exc}") from exc
    return out


def normalize_correction(
    field_name: str, raw: Any, precision: CurrencyPrecision, schema: ExtractionSchema
) -> Any:
    """Validate a human correction and return what gets stored.

    Money fields (the amounts of `schema`) take a decimal string or number that
    fits the currency and are stored as the canonical string. An array field
    takes null or a list of objects with exactly its cells (line_items:
    description, quantity, unit_price, total; tax_lines: label, rate, base,
    amount), each converted the same way; a tax rate is the printed percent,
    at most six decimal places, between 0 and 100. currency takes a
    three-letter code, stored in upper case. payment_card_last4 takes exactly
    four digits and payment_method text without a card-number-like digit run
    (see app/pii.py), so a correction can never put a card number in the record. Every other
    field (vendor, the dates, the ids) is stored as sent, as it always was. A
    bad value raises MoneyError, whose message is a sentence to show the
    reviewer.
    """
    if field_name in schema.money_field_names:
        if raw is None:
            return None
        minor_units = precision.minor_units
        amount = parse_amount_input(raw, minor_units, precision)
        return to_api_string(amount, minor_units)
    if field_name in ARRAY_CELLS:
        if raw is None:
            return None
        if not isinstance(raw, list):
            raise MoneyError(f"{_ARRAY_NOUNS[field_name][1]} must be a list, or null for none.")
        return [_normalize_array_row(field_name, i, row, precision) for i, row in enumerate(raw)]
    if field_name == "currency":
        if raw is None:
            return None
        code = raw.strip() if isinstance(raw, str) else ""
        if not _CURRENCY_CODE_RE.fullmatch(code):
            raise MoneyError(f"{shown_value(raw)} is not a three-letter currency code, such as USD.")
        return code.upper()
    if field_name == "payment_card_last4":
        if raw is None:
            return None
        if not isinstance(raw, str) or not CARD_LAST4_RE.fullmatch(raw):
            raise MoneyError(
                f"{shown_value(raw)} is not the last four digits of a card; "
                "enter exactly four digits, such as 4242."
            )
        return raw
    if field_name == "payment_method":
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise MoneyError("A payment method is text or null, such as VISA or Cash.")
        if has_long_digit_run(raw):
            raise MoneyError(
                "A payment method is the brand or method only, such as VISA or Cash; "
                "do not enter a card or account number."
            )
        return raw
    return raw
