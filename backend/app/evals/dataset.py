"""Loader for the labeled eval dataset (repo-root evals/).

Pairs each label JSON in evals/labels/ with its image, keyed by the label
file's stem (the same string as its own doc_id, checked below), and
validates the label shape LOUDLY -- a malformed or missing field raises
ValueError naming the offending file/doc_id and key, rather than being
silently skipped, so a broken committed dataset fails CI instead of just
quietly shrinking the eval set.

Pure/offline: this module touches only the filesystem and the standard
library. It does not import app.worker or open a DB session, so it stays
importable with Postgres stopped -- see app/evals/scoring.py for the
sibling module that scores what this one loads.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pypdf

from app.config import BACKEND_DIR
from app.extraction_fields import ARRAY_CELLS, EVIDENCE_QUOTE_MAX_CHARS, SCHEMA_V2
from app.pii import (
    CARD_LAST4_RE,
    has_long_digit_run,
    mask_card_numbers,
    mask_digit_runs,
)

EVALS_DIR = BACKEND_DIR.parent / "evals"
LABELS_DIR = EVALS_DIR / "labels"

# The exact key set fields must have -- no more, no less. A dataset entry
# with a stray or missing key is a labeling bug worth failing loudly on,
# not silently tolerating. Dataset v2 labels carry every field of the v2
# extraction schema; a field the document does not show is null. Scoring still
# reads only the seven v1 fields of each label (app/evals/scoring.py).
FIELD_KEYS = frozenset(SCHEMA_V2.field_names)
LINE_ITEM_KEYS = frozenset(ARRAY_CELLS["line_items"])
TAX_LINE_KEYS = frozenset(ARRAY_CELLS["tax_lines"])

# The fields a label holds as free text, as ISO dates, and as amounts.
_TEXT_KEYS = ("invoice_number", "po_number", "vendor_tax_id", "payment_terms", "currency")
_DATE_KEYS = ("document_date", "due_date")
_MONEY_KEYS = tuple(SCHEMA_V2.money_field_names)

_LABEL_TOP_LEVEL_KEYS = (
    "image",
    "mime_type",
    "source",
    "dataset_version",
    "difficulty",
    "fields",
)

# Matches app.evals.scoring's own _ISO_DATE_RE. date.fromisoformat alone
# is too permissive (Python 3.11+ also accepts "20260101" and ISO week
# dates like "2026-W01-1"), so a label could pass fromisoformat but be a
# guaranteed permanent zero once match_date -- which only recognizes the
# dashed form -- gets hold of it. Validate against the same pattern
# scoring uses, not just "does date.fromisoformat accept it".
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# A non-null vendor with no word character at all (e.g. "***", "   ")
# normalizes to the empty string under scoring's fuzzy match -- reject it
# at load time rather than letting it silently become an unscoreable (or
# vacuously-matched) label.
_WORD_CHAR_RE = re.compile(r"\w")


@dataclass
class EvalCase:
    doc_id: str
    image_path: Path
    mime_type: str
    source: str
    dataset_version: str
    difficulty: str
    fields: dict[str, Any]
    # Where each non-null value is printed: {path: {"page", "quote"}}, the
    # gold counterpart of a v2 extraction's evidence. Optional per label.
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)


def _validate_document_date(doc_id: str, value: Any, key: str = "document_date") -> None:
    """A label's date field (the document date or the due date): null, or a
    dashed ISO calendar date."""
    if value is None:
        return
    if not isinstance(value, str):
        # ValueError (not TypeError, contra ruff's TRY004) is deliberate:
        # this is label-data validation, not a Python type-checking bug --
        # every other validator in this module raises ValueError for the
        # same "malformed dataset entry" class of problem.
        raise ValueError(  # noqa: TRY004
            f"{doc_id}: fields.{key} must be a string or null, got {value!r}"
        )
    if not _ISO_DATE_RE.match(value):
        raise ValueError(f"{doc_id}: fields.{key} {value!r} is not ISO YYYY-MM-DD (dashed)")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{doc_id}: fields.{key} {value!r} is not a valid calendar date"
        ) from exc


def _validate_numeric(doc_id: str, key: str, value: Any) -> None:
    # bool is a subclass of int -- isinstance(True, int | float) is True
    # -- so it must be excluded explicitly, or a label with
    # "subtotal": true would pass this check.
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int | float)
    ):
        raise ValueError(
            f"{doc_id}: fields.{key} must be numeric or null, got {value!r}"
        )


def _validate_string_or_null(doc_id: str, key: str, value: Any) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(
            f"{doc_id}: fields.{key} must be a string or null, got {value!r}"
        )


def _validate_vendor(doc_id: str, value: Any) -> None:
    _validate_string_or_null(doc_id, "vendor", value)
    if value is not None and not _WORD_CHAR_RE.search(value):
        raise ValueError(
            f"{doc_id}: fields.vendor {value!r} has no word characters -- empty or "
            "punctuation-only vendor strings can't be meaningfully fuzzy-matched"
        )


def _validate_line_item_scalar(doc_id: str, index: int, key: str, value: Any) -> None:
    if key == "description":
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"{doc_id}: fields.line_items[{index}].description must be a string "
                f"or null, got {value!r}"
            )
        return
    # quantity / unit_price / total: numeric or null, bool excluded (see
    # _validate_numeric).
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int | float)
    ):
        raise ValueError(
            f"{doc_id}: fields.line_items[{index}].{key} must be numeric or null, "
            f"got {value!r}"
        )


def _validate_line_items(doc_id: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        # See _validate_document_date: ValueError is deliberate here too.
        raise ValueError(  # noqa: TRY004
            f"{doc_id}: fields.line_items must be a list or null, got {type(value).__name__}"
        )
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(  # noqa: TRY004
                f"{doc_id}: fields.line_items[{i}] must be an object, got {type(item).__name__}"
            )
        missing = LINE_ITEM_KEYS - item.keys()
        if missing:
            raise ValueError(
                f"{doc_id}: fields.line_items[{i}] missing keys {sorted(missing)}"
            )
        for key in LINE_ITEM_KEYS:
            _validate_line_item_scalar(doc_id, i, key, item[key])


def _validate_payment_method(doc_id: str, value: Any) -> None:
    _validate_string_or_null(doc_id, "payment_method", value)
    if isinstance(value, str) and has_long_digit_run(value):
        raise ValueError(
            f"{doc_id}: fields.payment_method {value!r} holds a card or account number; "
            "a label keeps the brand only"
        )


def _validate_card_last4(doc_id: str, value: Any) -> None:
    if value is not None and not (isinstance(value, str) and CARD_LAST4_RE.fullmatch(value)):
        raise ValueError(
            f"{doc_id}: fields.payment_card_last4 must be exactly four digits or null, got {value!r}"
        )


def _validate_tax_lines(doc_id: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise ValueError(  # noqa: TRY004 -- label-data validation, see _validate_document_date
            f"{doc_id}: fields.tax_lines must be a list or null, got {type(value).__name__}"
        )
    for i, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(  # noqa: TRY004
                f"{doc_id}: fields.tax_lines[{i}] must be an object, got {type(row).__name__}"
            )
        if row.keys() != TAX_LINE_KEYS:
            raise ValueError(
                f"{doc_id}: fields.tax_lines[{i}] must have exactly keys "
                f"{sorted(TAX_LINE_KEYS)}, got {sorted(row)}"
            )
        _validate_string_or_null(doc_id, f"tax_lines[{i}].label", row["label"])
        for key in ("rate", "base", "amount"):
            _validate_numeric(doc_id, f"tax_lines[{i}].{key}", row[key])


def label_values(fields: dict[str, Any]) -> dict[str, Any]:
    """Every path a label has a value at, with that value: the scalar fields and
    each cell of each row of line_items and tax_lines. These are the paths an
    evidence key may name."""
    values = {name: value for name, value in fields.items() if name not in ARRAY_CELLS}
    for name, cells in ARRAY_CELLS.items():
        for index, row in enumerate(fields[name] or []):
            for cell in cells:
                values[f"{name}[{index}].{cell}"] = row[cell]
    return values


# The payment fields, whose quotes get the stricter digit-run mask the product
# applies to them (app/extraction.py, mask_payment_fields).
_PAYMENT_PATHS = ("payment_method", "payment_card_last4")


def _validate_evidence(
    doc_id: str, raw: Any, fields: dict[str, Any], page_count: int = 1
) -> dict[str, dict[str, Any]]:
    """The label's optional `evidence` key: {path: {page, quote}} for values the
    label gives, with a page the document has (1 to `page_count`) and a verbatim
    quote that is non-empty, short, and holds no card number: none that passes the
    Luhn check anywhere, and no long digit run at all in a payment field's quote,
    exactly what the product masks."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(  # noqa: TRY004
            f"{doc_id}: evidence must be an object, got {type(raw).__name__}"
        )
    values = label_values(fields)
    for path, entry in raw.items():
        if path not in values:
            raise ValueError(f"{doc_id}: evidence names unknown path {path!r}")
        if values[path] is None:
            raise ValueError(f"{doc_id}: evidence for {path!r}, whose value is null")
        if not isinstance(entry, dict) or entry.keys() != {"page", "quote"}:
            raise ValueError(f"{doc_id}: evidence[{path!r}] must be an object with page and quote")
        page, quote = entry["page"], entry["quote"]
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError(f"{doc_id}: evidence[{path!r}].page must be an integer from 1, got {page!r}")
        if page > page_count:
            raise ValueError(
                f"{doc_id}: evidence[{path!r}].page {page} is past the document's {page_count} page(s)"
            )
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError(f"{doc_id}: evidence[{path!r}].quote must be non-empty text")
        if len(quote) > EVIDENCE_QUOTE_MAX_CHARS:
            raise ValueError(
                f"{doc_id}: evidence[{path!r}].quote is over {EVIDENCE_QUOTE_MAX_CHARS} characters"
            )
        masked = mask_digit_runs(quote) if path in _PAYMENT_PATHS else mask_card_numbers(quote)
        if masked != quote:
            raise ValueError(f"{doc_id}: evidence[{path!r}].quote holds a card number; mask it")
    return raw


def _page_count(doc_id: str, raw: dict[str, Any], image_path: Path) -> int:
    """How many pages the label's document has: an image is one, a PDF as many
    as it holds."""
    if raw["mime_type"] != "application/pdf":
        return 1
    try:
        return len(pypdf.PdfReader(image_path).pages)
    except (pypdf.errors.PyPdfError, OSError) as exc:
        raise ValueError(f"{doc_id}: unreadable PDF {image_path}: {exc}") from exc


def _load_case(label_path: Path, evals_dir: Path = EVALS_DIR) -> EvalCase:
    """Load and loudly validate one label. evals_dir anchors the label's
    relative `image` path; the default is the repo corpus, and
    app.evals.harvest passes its target dir so freshly harvested labels
    validate against the directory they were just written to."""
    try:
        raw = json.loads(label_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label_path}: invalid JSON: {exc}") from exc

    doc_id = raw.get("doc_id")
    if not doc_id:
        raise ValueError(f"{label_path}: missing doc_id")
    if doc_id != label_path.stem:
        raise ValueError(
            f"{label_path}: doc_id {doc_id!r} does not match label filename stem "
            f"{label_path.stem!r} -- labels are paired to images by stem"
        )

    missing_top_level = [key for key in _LABEL_TOP_LEVEL_KEYS if key not in raw]
    if missing_top_level:
        raise ValueError(f"{doc_id}: missing top-level key(s) {missing_top_level}")

    if not raw["source"]:
        raise ValueError(f"{doc_id}: source must be present/non-empty")

    fields = raw["fields"]
    if not isinstance(fields, dict) or set(fields.keys()) != FIELD_KEYS:
        got = (
            sorted(fields.keys()) if isinstance(fields, dict) else type(fields).__name__
        )
        raise ValueError(
            f"{doc_id}: fields must have exactly keys {sorted(FIELD_KEYS)}, got {got}"
        )

    _validate_vendor(doc_id, fields["vendor"])
    for key in _DATE_KEYS:
        _validate_document_date(doc_id, fields[key], key)
    for key in _TEXT_KEYS:
        _validate_string_or_null(doc_id, key, fields[key])
    _validate_payment_method(doc_id, fields["payment_method"])
    _validate_card_last4(doc_id, fields["payment_card_last4"])
    for key in _MONEY_KEYS:
        _validate_numeric(doc_id, key, fields[key])
    _validate_line_items(doc_id, fields["line_items"])
    _validate_tax_lines(doc_id, fields["tax_lines"])

    image_path = evals_dir / raw["image"]
    if not image_path.is_file():
        raise ValueError(f"{doc_id}: image file not found: {image_path}")
    evidence = _validate_evidence(doc_id, raw.get("evidence"), fields, _page_count(doc_id, raw, image_path))

    return EvalCase(
        doc_id=doc_id,
        image_path=image_path,
        mime_type=raw["mime_type"],
        source=raw["source"],
        dataset_version=raw["dataset_version"],
        difficulty=raw["difficulty"],
        fields=fields,
        evidence=evidence,
    )


def load_cases(limit: int | None = None) -> list[EvalCase]:
    """Load every label in evals/labels/, sorted by doc_id.

    Each label JSON is validated loudly (ValueError naming the offending
    file/doc_id and key) before being paired with its image via the
    `image` path it records, relative to EVALS_DIR -- the doc_id ==
    label-filename-stem check above ensures that pairing lines up with
    the image's own `NNN-slug` filename rather than silently drifting.

    `limit` slices AFTER sorting, so `load_cases(limit=5)` deterministically
    returns the first 5 cases by doc_id regardless of directory iteration
    order.
    """
    label_paths = sorted(LABELS_DIR.glob("*.json"))
    cases = [_load_case(p) for p in label_paths]
    cases.sort(key=lambda c: c.doc_id)
    if limit is not None:
        cases = cases[:limit]
    return cases
