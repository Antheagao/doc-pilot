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
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from app.config import BACKEND_DIR

EVALS_DIR = BACKEND_DIR.parent / "evals"
LABELS_DIR = EVALS_DIR / "labels"

# The exact key set fields must have -- no more, no less. A dataset entry
# with a stray or missing key is a labeling bug worth failing loudly on,
# not silently tolerating.
FIELD_KEYS = frozenset(
    {"vendor", "document_date", "currency", "subtotal", "tax", "total", "line_items"}
)
LINE_ITEM_KEYS = frozenset({"description", "quantity", "unit_price", "total"})

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


def _validate_document_date(doc_id: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        # ValueError (not TypeError, contra ruff's TRY004) is deliberate:
        # this is label-data validation, not a Python type-checking bug --
        # every other validator in this module raises ValueError for the
        # same "malformed dataset entry" class of problem.
        raise ValueError(  # noqa: TRY004
            f"{doc_id}: fields.document_date must be a string or null, got {value!r}"
        )
    if not _ISO_DATE_RE.match(value):
        raise ValueError(
            f"{doc_id}: fields.document_date {value!r} is not ISO YYYY-MM-DD (dashed)"
        )
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{doc_id}: fields.document_date {value!r} is not a valid calendar date"
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


def _load_case(label_path: Path) -> EvalCase:
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
    _validate_document_date(doc_id, fields["document_date"])
    _validate_string_or_null(doc_id, "currency", fields["currency"])
    for key in ("subtotal", "tax", "total"):
        _validate_numeric(doc_id, key, fields[key])
    _validate_line_items(doc_id, fields["line_items"])

    image_path = EVALS_DIR / raw["image"]
    if not image_path.is_file():
        raise ValueError(f"{doc_id}: image file not found: {image_path}")

    return EvalCase(
        doc_id=doc_id,
        image_path=image_path,
        mime_type=raw["mime_type"],
        source=raw["source"],
        dataset_version=raw["dataset_version"],
        difficulty=raw["difficulty"],
        fields=fields,
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
