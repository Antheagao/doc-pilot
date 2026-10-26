"""query_extractions decides its filters, count, and sums in SQL.

The old implementation (load every record of the organization, filter and sum
in Python) is kept here as the oracle, verbatim, and a property test draws
tenants of documents with awkward vendors, items, dates, currencies, totals,
corrections, and extractions, and filters over them, and requires the tool's
whole output to equal the oracle's. The rest pins what the property cannot
say: that the work is bounded, that the SQL helpers agree with Python's own
tables, that text filters are not patterns, and that the statements stay
inside the organization.
"""

import string
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, Self

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import event as note
from hypothesis import strategies as st
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import tools as tools_module
from app.agent.extraction_query import (
    CASEFOLD_TO_ASCII,
    PYTHON_WHITESPACE,
    ExtractionFilters,
    MatchTotals,
    find_matches,
)
from app.agent.tools import (
    SOURCE_SCHEME,
    SUMMARY_SOURCE,
    Source,
    ToolContext,
    _parse_date,
    _search_result,
    _total_bounds,
    query_extractions,
    record_lines,
)
from app.db import engine
from app.extraction_fields import SCHEMA_V1
from app.models import (
    DEFAULT_ORGANIZATION_ID,
    Document,
    ExtractedField,
    Extraction,
    Organization,
)
from app.money import (
    MAX_AMOUNT,
    MoneyError,
    currency_precision,
    to_api_string,
    to_decimal,
)
from app.records import DocumentRecord, load_records
from app.retrieval.embeddings import HashingEmbedder
from app.tenancy import bind_tenant

EMBEDDER = HashingEmbedder()
BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


# --- the oracle: the implementation this replaced ----------------------------------


def _oracle_decimal_or_none(value: Any) -> Decimal | None:
    try:
        amount = to_decimal(value)
    except MoneyError:
        return None
    return amount if abs(amount) < MAX_AMOUNT else None


def _oracle_matches(
    record: DocumentRecord,
    tool_input: dict[str, Any],
    date_from: date | None,
    date_to: date | None,
    min_total: Decimal | None,
    max_total: Decimal | None,
) -> bool:
    fields = record.fields
    vendor = tool_input.get("vendor")
    if vendor and vendor.casefold() not in str(fields.get("vendor") or "").casefold():
        return False
    item = tool_input.get("item")
    if item and not any(
        item.casefold() in str(row.get("description") or "").casefold()
        for row in fields.get("line_items") or []
    ):
        return False
    currency = tool_input.get("currency")
    if currency and str(fields.get("currency") or "").upper() != currency.upper():
        return False
    if date_from or date_to:
        try:
            document_date = date.fromisoformat(str(fields.get("document_date")))
        except ValueError:
            return False
        if (date_from and document_date < date_from) or (date_to and document_date > date_to):
            return False
    if min_total is None and max_total is None:
        return True
    total = _oracle_decimal_or_none(fields.get("total"))
    if total is None:
        return False
    return (min_total is None or total >= min_total) and (max_total is None or total <= max_total)


def _oracle_summary(matches: list[DocumentRecord]) -> str:
    from decimal import Inexact, localcontext

    sums: dict[str, Decimal] = {}
    missing_total = 0
    with localcontext() as context:
        context.prec = 400
        context.traps[Inexact] = True
        for record in matches:
            total = _oracle_decimal_or_none(record.fields.get("total"))
            if total is None:
                missing_total += 1
                continue
            currency = currency_precision(record.fields.get("currency")).code or "unknown currency"
            sums[currency] = sums.get(currency, Decimal(0)) + total
    parts = [f"{len(matches)} matching document(s)."]
    if sums:
        parts.append(
            "Sum of totals: "
            + "; ".join(
                f"{to_api_string(amount, currency_precision(currency).minor_units)} {currency}"
                for currency, amount in sorted(sums.items())
            )
            + "."
        )
    if missing_total:
        parts.append(f"{missing_total} matching document(s) have no total.")
    if len(matches) > tools_module.RECORDS_MAX:
        parts.append(
            f"Listing the first {tools_module.RECORDS_MAX}; narrow the filters to see the rest."
        )
    return " ".join(parts)


async def oracle_query_extractions(ctx: ToolContext, tool_input: dict[str, Any]) -> list[dict]:
    date_from = _parse_date(tool_input.get("date_from"), "date_from")
    date_to = _parse_date(tool_input.get("date_to"), "date_to")
    records = await load_records(ctx.session, ctx.document_ids)
    min_total, max_total = _total_bounds(tool_input)
    matches = [
        r for r in records if _oracle_matches(r, tool_input, date_from, date_to, min_total, max_total)
    ]

    summary = _oracle_summary(matches)
    if not matches:
        return [{"type": "text", "text": summary}]
    content: list[dict[str, Any]] = [
        {
            "type": "search_result",
            "source": SUMMARY_SOURCE,
            "title": "Summary of the matches",
            "content": [{"type": "text", "text": summary}],
            "citations": {"enabled": True},
        }
    ]
    for record in matches[: tools_module.RECORDS_MAX]:
        lines = record_lines(record)
        source = Source(
            source=f"{SOURCE_SCHEME}/{record.document_id}/extraction",
            kind="record",
            document_id=record.document_id,
            filename=record.filename,
            page_number=None,
            blocks=[name for name, _ in lines],
            lines=[text for _, text in lines],
        )
        ctx.sources[source.source] = source
        content.append(
            _search_result(
                source,
                f"{record.filename} (extracted fields, document_id {record.document_id})",
                [{"type": "text", "text": text} for _, text in lines],
            )
        )
    return content


# --- building a tenant ----------------------------------------------------------------

UNSET = object()


@dataclass
class StoredField:
    """One extracted_fields row. `stored` is the value column as it sits (a
    leaf for a scalar, the list of cells for line_items); `action` and
    `corrected` are the review columns (`corrected` UNSET leaves the column SQL
    NULL, None stores a JSON null)."""

    stored: Any
    action: str | None = None
    corrected: Any = UNSET
    unreviewed_low_confidence: bool = False


@dataclass
class StoredExtraction:
    minutes: int
    fields: dict[str, StoredField]
    prompt_version: str = SCHEMA_V1.prompt_version


@dataclass
class StoredDocument:
    filename: str
    minutes: int
    extractions: list[StoredExtraction]
    status: str = "extracted"
    vendor_hint: str = field(default="", compare=False)


def leaf(value: Any) -> dict[str, Any]:
    return {"value": value, "confidence": 0.95}


def cell_row(description: Any, quantity: Any = 1, unit_price: Any = 2.5, total: Any = 2.5) -> dict:
    return {
        "description": leaf(description),
        "quantity": leaf(quantity),
        "unit_price": leaf(unit_price),
        "total": leaf(total),
    }


def plain_row(description: Any, quantity: Any = 1, unit_price: Any = "2.50", total: Any = "2.50") -> dict:
    return {
        "description": description,
        "quantity": quantity,
        "unit_price": unit_price,
        "total": total,
    }


def scalar(name: str, value: Any, **kwargs: Any) -> tuple[str, StoredField]:
    return name, StoredField(leaf(value), **kwargs)


def simple_document(
    name: str,
    *,
    vendor: Any = "Acme",
    total: Any = 10.0,
    currency: Any = "USD",
    document_date: Any = "2026-01-15",
    items: tuple[Any, ...] = (),
    minutes: int = 0,
    status: str = "extracted",
) -> StoredDocument:
    fields = dict(
        [
            scalar("vendor", vendor),
            scalar("document_date", document_date),
            scalar("currency", currency),
            scalar("total", total),
            ("line_items", StoredField([cell_row(item) for item in items])),
        ]
    )
    return StoredDocument(name, minutes, [StoredExtraction(0, fields)], status)


async def store(session: AsyncSession, documents: list[StoredDocument]) -> list[uuid.UUID]:
    """Insert `documents` for the organization the session is bound to; the
    ids come back in the order given."""
    ids = []
    for spec in documents:
        document = Document(
            filename=spec.filename,
            mime_type="image/png",
            storage_path="/tmp/extraction-query",
            status=spec.status,
            created_at=BASE_TIME + timedelta(minutes=spec.minutes),
        )
        session.add(document)
        await session.flush()
        ids.append(document.id)
        for extraction_spec in spec.extractions:
            extraction = Extraction(
                document_id=document.id,
                prompt_version=extraction_spec.prompt_version,
                model="extraction-query-test",
                raw_response={},
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
                created_at=BASE_TIME + timedelta(minutes=extraction_spec.minutes),
            )
            session.add(extraction)
            await session.flush()
            for name, stored in extraction_spec.fields.items():
                kwargs: dict[str, Any] = {}
                if stored.corrected is not UNSET:
                    kwargs["corrected_value"] = stored.corrected
                session.add(
                    ExtractedField(
                        extraction_id=extraction.id,
                        field_name=name,
                        value=stored.stored,
                        confidence=0.4 if stored.unreviewed_low_confidence else 0.95,
                        needs_review=stored.unreviewed_low_confidence,
                        review_reason="confidence" if stored.unreviewed_low_confidence else None,
                        review_action=stored.action,
                        **kwargs,
                    )
                )
    await session.flush()
    return ids


async def run_both(
    session: AsyncSession,
    tool_input: dict[str, Any],
    document_ids: list[uuid.UUID] | None = None,
) -> tuple[list[dict], list[dict]]:
    """The tool and the oracle on the same data: (new content, old content),
    and the sources each registered must be the same too."""
    new_ctx = ToolContext(session=session, embedder=EMBEDDER, document_ids=document_ids)
    old_ctx = ToolContext(session=session, embedder=EMBEDDER, document_ids=document_ids)
    old = await oracle_query_extractions(old_ctx, tool_input)
    new = await query_extractions(new_ctx, tool_input)
    assert canonical(new_ctx.sources) == canonical(old_ctx.sources)
    return canonical(new), canonical(old)


NOTE = "note: not yet human-reviewed and low-confidence: "


def canonical(value: Any) -> Any:
    """`value` with the field names of every review note sorted. A record lists
    them in the order the database returns a document's field rows, which no
    query orders (the old one did not either), so equality is up to that."""
    if isinstance(value, str) and value.startswith(NOTE):
        return NOTE + ", ".join(sorted(value[len(NOTE) :].split(", ")))
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, Source):
        return Source(**{name: canonical(getattr(value, name)) for name in value.__dataclass_fields__})
    return value


def summary_of(content: list[dict]) -> str:
    if content[0]["type"] == "text":
        return content[0]["text"]
    return content[0]["content"][0]["text"]


# --- strategies ---------------------------------------------------------------------------------------------------

# Characters that expose the differences between PostgreSQL and Python: case
# folds that grow (sharp s), fold to ASCII from elsewhere (Kelvin sign, long s,
# dotted capital I, ligatures), wildcards, spaces, and accents.
ALPHABET = "aAbBkKsSjJiI ß\u1e9e\u017f\u212a\u0130\u0131\ufb01\u01f0%_éÉç€\\'\"-"
NAMES = [
    "Acme Corp",
    "ACME",
    "Northgate Office Outfitters",
    "Café Roma",
    "Straße Bräu",
    "STRASSE",
    "\u0130stanbul Çay",
    "Kelvin Kiosk",
    "50%_off",
    "a_b",
    "Tea & Co",
    "",
    "  ",
    "Zürich Ltd",
]
FOLDING = "sSß\u1e9e\u017fkK\u212aiI\u0130\u0131\ufb01\u01f0jJ "
texts = st.one_of(
    st.sampled_from(NAMES),
    st.text(alphabet=ALPHABET, max_size=8),
    st.text(alphabet=FOLDING, max_size=6),
)
odd_values = st.sampled_from([0, 12, 3.5, True, False, [], ["x"], {}, {"a": 1}])
text_values = st.one_of(texts, texts, st.none(), odd_values)

CURRENCIES = ["USD", "usd", "Usd", "EUR", "eur", "JPY", "jpy", "KWD", "XYZ", " usd ", "\u00a0USD", "ß", "\u01f0", "", "\u0131"]
currency_values = st.one_of(st.sampled_from(CURRENCIES), st.none(), odd_values)

DATES = [
    "2026-01-15",
    "2026-02-28",
    "2026-02-30",
    "2026-12-31",
    "2025-06-01",
    "0001-01-01",
    "0000-01-01",
    "20260115",
    "2026-W03-4",
    "2026W034",
    "2026-W03",
    "2026-1-5",
    "2026-01-15\n",
    " 2026-01-15",
    "2026-01-15T00:00:00",
    "\uff12\uff10\uff12\uff16-\uff10\uff11-\uff11\uff15",
    "not a date",
    "",
    "2026-01-1\u0665",
]
date_values = st.one_of(
    st.sampled_from(DATES),
    st.dates(date(1990, 1, 1), date(2030, 12, 31)).map(lambda d: d.isoformat()),
    st.none(),
    st.sampled_from([20260115, 2026.0, True, ["2026-01-15"]]),
)

TOTAL_TEXTS = [
    "12.50",
    " 12.5 ",
    "12.5\u00a0",
    "\u30001.00",
    "1,234.50",
    "abc",
    "1e3",
    "-0",
    "-5.25",
    "0123.4500",
    "+7",
    ".5",
    "5.",
    "",
    "999999999999.99",
    "1000000000000",
    "\u0663.\u0665",
    "0." + "1" * 40,
    "9" * 30,
    "12.50\n",
]
total_values = st.one_of(
    st.integers(-50, 500).map(float),
    st.integers(-5000, 50000).map(lambda cents: cents / 100),
    st.floats(0, 1000, allow_nan=False, allow_infinity=False),
    st.integers(-10, 1000),
    st.sampled_from(TOTAL_TEXTS),
    st.sampled_from([0.1, 0.2, 0.3, 1e12, 999999999999.99, 1e-7, -0.0, 0, 10**15, 3 * 10**11]),
    st.none(),
    st.sampled_from([True, [], {}, [1]]),
)


@st.composite
def scalar_field(draw: st.DrawFn, values: st.SearchStrategy) -> StoredField:
    """A scalar field's row: usually a leaf, sometimes reviewed (approved or
    corrected, the correction any value the strategy yields), sometimes odd
    storage."""
    kind = draw(st.sampled_from(["leaf"] * 12 + ["approved", "corrected", "corrected", "odd"]))
    value = draw(values)
    flagged = draw(st.booleans()) and kind == "leaf"
    if kind == "odd":
        stored = draw(st.sampled_from([None, "text", ["a"], {"nokey": 1}, {"value": None}, 5]))
        return StoredField(stored)
    stored = leaf(value)
    if kind == "approved":
        return StoredField(stored, "approved")
    if kind == "corrected":
        corrected = draw(st.one_of(values, st.just(UNSET), st.just(None)))
        return StoredField(stored, "corrected", corrected)
    return StoredField(stored, unreviewed_low_confidence=flagged)


@st.composite
def line_items_field(draw: st.DrawFn) -> StoredField:
    descriptions = st.one_of(texts, st.none(), odd_values)
    kind = draw(st.sampled_from(["cells", "cells", "approved", "corrected", "empty", "odd"]))
    if kind == "odd":
        return StoredField(draw(st.sampled_from([None, {"value": []}, "x", [3, "a", None], [{"description": "Tea"}]])))
    rows = [cell_row(draw(descriptions)) for _ in range(draw(st.integers(0, 3)))]
    if kind == "empty":
        return StoredField([])
    if kind == "approved":
        return StoredField(rows, "approved")
    if kind == "corrected":
        corrected_rows = [plain_row(draw(descriptions)) for _ in range(draw(st.integers(0, 3)))]
        corrected = draw(st.sampled_from([corrected_rows, corrected_rows, None, UNSET]))
        return StoredField(rows, "corrected", corrected)
    return StoredField(rows)


@st.composite
def extraction(draw: st.DrawFn, minutes: int) -> StoredExtraction:
    fields: dict[str, StoredField] = {}
    if draw(st.integers(0, 9)):
        # An extraction with no field rows is rare; it hides its document.
        for name, values in (
            ("vendor", text_values),
            ("document_date", date_values),
            ("currency", currency_values),
            ("total", total_values),
        ):
            if draw(st.integers(0, 11)):
                fields[name] = draw(scalar_field(values))
        if draw(st.integers(0, 5)):
            fields["line_items"] = draw(line_items_field())
        if draw(st.booleans()):
            fields["subtotal"] = draw(scalar_field(total_values))
    version = draw(st.sampled_from(["extract_v1", "extract_v1", "extract_v2", "extract_v9"]))
    return StoredExtraction(minutes, fields, version)


@st.composite
def stored_document(draw: st.DrawFn, index: int) -> StoredDocument:
    count = draw(st.sampled_from([1, 1, 1, 2, 3]))
    minutes = draw(st.lists(st.integers(0, 5), min_size=count, max_size=count, unique=True))
    extractions = [draw(extraction(m)) for m in minutes]
    status = draw(st.sampled_from(["extracted"] * 12 + ["uploaded", "failed"]))
    return StoredDocument(
        filename=draw(st.sampled_from(["a.png", "b.png", "B.png", "é.png", f"{index}.png"])),
        minutes=draw(st.integers(0, 3)),
        extractions=extractions if draw(st.integers(0, 14)) else [],
        status=status,
    )


tenants = st.integers(1, 9).flatmap(
    lambda n: st.tuples(*[stored_document(i) for i in range(n)]).map(list)
)

dates_for_filters = st.dates(date(2025, 1, 1), date(2027, 1, 1)).map(lambda d: d.isoformat())
numbers_for_filters = st.one_of(
    st.integers(-10, 600),
    st.integers(-1000, 60000).map(lambda cents: cents / 100),
    st.sampled_from([0, 0.1, 12.5, 10.0, 1e12, 999999999999.99, 1e13]),
)


def held_values(documents: list[StoredDocument], name: str) -> list[Any]:
    """What the documents store (or are corrected to) for a field, whatever
    its type: the material the filters are cut from."""
    found = []
    for spec in documents:
        for extraction_spec in spec.extractions:
            stored = extraction_spec.fields.get(name)
            if stored is None:
                continue
            if isinstance(stored.stored, dict):
                found.append(stored.stored.get("value"))
            if stored.corrected is not UNSET:
                found.append(stored.corrected)
    return found


def held_descriptions(documents: list[StoredDocument]) -> list[str]:
    found = []
    for rows in held_values(documents, "line_items"):
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict):
                description = row.get("description")
                description = description.get("value") if isinstance(description, dict) else description
                if isinstance(description, str):
                    found.append(description)
    return found


@st.composite
def tool_inputs(draw: st.DrawFn, documents: list[StoredDocument]) -> dict[str, Any]:
    """One to three of the tool's filters, cut from what the documents hold so
    that they match often: a slice of a vendor or description in another case
    (or its case fold), a currency as held or re-cased, a date or total at or
    next to one that is stored, and sometimes a value from nowhere."""
    transforms = ["strip", "upper", "lower", "swapcase", "casefold", "casefold"]

    def text_filter(pool: list[Any], fallback: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
        strings = [value for value in pool if isinstance(value, str) and value]
        if not strings:
            return fallback
        sliced = st.tuples(
            st.sampled_from(strings), st.integers(0, 6), st.integers(1, 8), st.sampled_from(transforms)
        ).map(lambda t: getattr(t[0][t[1] : t[1] + t[2]], t[3])() or t[0])
        return st.one_of(fallback, sliced, sliced, sliced)

    def date_filter() -> st.SearchStrategy[str]:
        parsed = []
        for value in held_values(documents, "document_date"):
            try:
                parsed.append(date.fromisoformat(value))
            except (TypeError, ValueError):
                pass
        if not parsed:
            return dates_for_filters
        shifted = st.tuples(st.sampled_from(parsed), st.integers(-2, 2)).map(
            lambda t: (t[0] + timedelta(days=t[1])).isoformat()
        )
        return st.one_of(dates_for_filters, shifted, shifted, shifted)

    def total_filter() -> st.SearchStrategy[Any]:
        amounts = []
        for value in held_values(documents, "total"):
            try:
                amounts.append(to_decimal(value))
            except MoneyError:
                pass
        if not amounts:
            return numbers_for_filters
        shifted = st.tuples(st.sampled_from(amounts), st.sampled_from(["0", "0", "0.01", "-0.01"])).map(
            lambda t: float(t[0] + Decimal(t[1])) if abs(t[0]) < 10**9 else 0
        )
        return st.one_of(numbers_for_filters, shifted, shifted, shifted)

    builders: dict[str, st.SearchStrategy[Any]] = {
        "vendor": text_filter(held_values(documents, "vendor"), texts),
        "item": text_filter(held_descriptions(documents), texts),
        "currency": text_filter(held_values(documents, "currency"), st.sampled_from(CURRENCIES + ["Usd", "eUr"])),
        "date_from": date_filter(),
        "date_to": date_filter(),
        "min_total": total_filter(),
        "max_total": total_filter(),
    }
    first = draw(st.sampled_from(["vendor", "vendor", "item", "currency", "date_from", "min_total"]))
    more = draw(st.lists(st.sampled_from(sorted(builders)), max_size=2, unique=True))
    names = [first, *(name for name in more if name != first)]
    return {name: draw(builders[name]) for name in names}


async def fresh_tenant(session: AsyncSession) -> uuid.UUID:
    organization = Organization(name="Property", slug=f"property-{uuid.uuid4().hex[:12]}")
    session.add(organization)
    await session.flush()
    await bind_tenant(session, organization.id)
    return organization.id


async def check_against_oracle(
    session: AsyncSession,
    documents: list[StoredDocument],
    inputs: st.SearchStrategy[dict[str, Any]] | None,
    data: st.DataObject,
    filters_drawn: int = 3,
) -> None:
    """Store `documents` in a tenant of their own, then compare the tool with
    the oracle on `filters_drawn` random filter sets and a random scope. All of
    it is rolled back."""
    savepoint = await session.begin_nested()
    try:
        await fresh_tenant(session)
        ids = await store(session, documents)
        scope = data.draw(st.one_of(st.none(), st.none(), st.lists(st.sampled_from(ids), unique=True)))
        for _ in range(filters_drawn):
            tool_input = data.draw(inputs if inputs is not None else tool_inputs(documents))
            new, old = await run_both(session, tool_input, scope)
            assert new == old, tool_input
            note(f"matched={min(len(new) - 1, 3) if new[0]['type'] == 'search_result' else 0}")
    finally:
        await savepoint.rollback()
        await bind_tenant(session, DEFAULT_ORGANIZATION_ID)
        session.expunge_all()


@settings(
    max_examples=120,
    report_multiple_bugs=False,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(data=st.data(), documents=tenants)
async def test_the_tool_returns_what_the_python_implementation_returned(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    data: st.DataObject,
    documents: list[StoredDocument],
) -> None:
    """Over random tenants with every awkward value, and random filter
    combinations cut from them, `query_extractions` returns the same content,
    in the same order, with the same sources, as the implementation that loaded
    and filtered every record in Python. The listing limit is cut to 3 so that
    more than the limit match often."""
    monkeypatch.setattr(tools_module, "RECORDS_MAX", 3)
    await check_against_oracle(db_session, documents, None, data)


CLEAN_DATES = ["2026-03-01", "2026-03-15", "2026-03-31", "2026-04-01", "2026-02-28"]
# Stored dates also include a basic-format ISO date, which SQL leaves to Python.
STORED_DATES = [*CLEAN_DATES, "20260315"]
CLEAN_TOTALS = [0, 5, 9.99, 10, 10.01, "10.00", "10", " 10 ", "ten", None, 1e12, 999999999999.99, -3.5, "0.1", 0.2]
clean_fields = st.fixed_dictionaries(
    {
        "vendor": scalar_field(st.sampled_from(["Acme", "ACME Corp", "Beta", "gamma Ltd", "Café", "Straße"])),
        "document_date": scalar_field(st.sampled_from(STORED_DATES)),
        "currency": scalar_field(st.sampled_from(["USD", "usd", "EUR", "eur", " usd", None])),
        "total": scalar_field(st.sampled_from(CLEAN_TOTALS)),
        "line_items": st.builds(
            lambda descriptions: StoredField([cell_row(d) for d in descriptions]),
            st.lists(st.sampled_from(["Tea", "tea bag", "Cake", "TEA", "straße cake"]), max_size=3),
        ),
    }
)
clean_documents = st.lists(
    st.builds(
        lambda name, minutes, fields: StoredDocument(
            name, minutes, [StoredExtraction(0, fields)]
        ),
        st.sampled_from(["a.png", "b.png", "c.png"]),
        st.integers(0, 2),
        clean_fields,
    ),
    min_size=1,
    max_size=8,
)
clean_inputs = st.fixed_dictionaries(
    {},
    optional={
        "vendor": st.sampled_from(["acme", "Corp", "é", "gamma", "BETA", "SS", "strasse"]),
        "item": st.sampled_from(["tea", "CAKE", "ss", "bag"]),
        "currency": st.sampled_from(["usd", "EUR", "Usd", "eur"]),
        "date_from": st.sampled_from(CLEAN_DATES),
        "date_to": st.sampled_from(CLEAN_DATES),
        "min_total": st.sampled_from([0, 5, 9.99, 10, 10.01, 1e12, -3.5]),
        "max_total": st.sampled_from([0, 5, 9.99, 10, 10.01, 1e12, -3.5]),
    },
)


@settings(
    max_examples=120,
    report_multiple_bugs=False,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(data=st.data(), documents=clean_documents)
async def test_the_filter_edges_agree_with_the_python_implementation(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    data: st.DataObject,
    documents: list[StoredDocument],
) -> None:
    """The same comparison over tenants of ordinary values and filters on the
    very values stored, so inclusive bounds, case, and corrections are hit on
    their edges every time."""
    monkeypatch.setattr(tools_module, "RECORDS_MAX", 3)
    await check_against_oracle(db_session, documents, clean_inputs, data, filters_drawn=4)


# --- named cases the property cannot name ----------------------------------------------------------------------------


async def test_a_user_filter_is_text_not_a_pattern(db_session: AsyncSession) -> None:
    await store(
        db_session,
        [
            simple_document("a.png", vendor="50%_off Shop", items=("100% cotton",)),
            simple_document("b.png", vendor="500 off Shop", items=("1000 cotton",)),
            simple_document("c.png", vendor="Back\\slash", items=("it's",)),
        ],
    )

    for tool_input, filenames in (
        ({"vendor": "%"}, ["a.png"]),
        ({"vendor": "%o"}, []),
        ({"vendor": "50%_"}, ["a.png"]),
        ({"item": "0% c"}, ["a.png"]),
        ({"vendor": "\\"}, ["c.png"]),
        ({"item": "'"}, ["c.png"]),
        ({"vendor": ""}, ["a.png", "b.png", "c.png"]),
    ):
        new, old = await run_both(db_session, tool_input)
        assert new == old
        found = sorted(block["title"].split(" (")[0] for block in new[1:])
        assert found == filenames, tool_input


async def test_case_insensitive_matching_is_pythons_not_the_locales(db_session: AsyncSession) -> None:
    await store(
        db_session,
        [
            simple_document("sharp.png", vendor="Straße", currency="ß"),
            simple_document("caps.png", vendor="STRASSE", currency="SS"),
            simple_document("kelvin.png", vendor="K\u212a", currency="\u01f0"),
            simple_document("dotted.png", vendor="\u0130stanbul", currency="\u0131"),
            simple_document("plain.png", vendor="Cafe", currency="USD"),
            simple_document("accent.png", vendor="Café", currency="usd"),
        ],
    )

    for tool_input in (
        {"vendor": "strasse"},
        {"vendor": "STRASSE"},
        {"vendor": "ß"},
        {"vendor": "kk"},
        {"vendor": "istanbul"},
        {"vendor": "\u0130"},
        {"vendor": "i\u0307s"},
        {"vendor": "café"},
        {"vendor": "CAFÉ"},
        {"vendor": "caf"},
        {"currency": "ss"},
        {"currency": "ß"},
        {"currency": "I"},
        {"currency": "J\u030c"},
        {"currency": "Usd"},
        {"currency": "uSd"},
    ):
        new, old = await run_both(db_session, tool_input)
        assert new == old, tool_input


async def test_a_filter_no_stored_text_can_equal_matches_nothing_instead_of_failing(
    db_session: AsyncSession,
) -> None:
    await store(db_session, [simple_document("a.png", vendor="Acme", items=("Tea",))])

    for tool_input in (
        {"vendor": "ac\x00me"},
        {"item": "\x00"},
        {"currency": "US\x00D"},
        {"vendor": "\ud800"},
        {"vendor": "acme", "item": "te\udfffa"},
    ):
        new, old = await run_both(db_session, tool_input)
        assert new == old == [{"type": "text", "text": "0 matching document(s)."}], tool_input


async def test_python_whitespace_and_case_tables_are_the_interpreters() -> None:
    assert PYTHON_WHITESPACE == "".join(c for c in map(chr, range(0x110000)) if c.isspace())
    assert CASEFOLD_TO_ASCII == "".join(
        c for c in map(chr, range(0x110000)) if ord(c) > 127 and any(ord(x) < 128 for x in c.casefold())
    )
    assert all(c.casefold() == c.lower() for c in string.ascii_letters)


async def test_sums_are_exact_decimals_not_floats(db_session: AsyncSession) -> None:
    totals = [0.1, 0.2, 0.3, "0.30", "1234567.89", 7, 1e-7, "0.0000001", 999999999999.99]
    await store(
        db_session,
        [simple_document(f"{i}.png", total=total, currency="usd" if i % 2 else "USD") for i, total in enumerate(totals)],
    )

    new, old = await run_both(db_session, {})

    assert new == old
    assert "Sum of totals: 1000001234575.7800002 USD." in summary_of(new)
    # The SQL sum is the Decimal sum, digit for digit.
    outcome = await find_matches(db_session, ExtractionFilters(), None, 20)
    assert outcome.totals.sums["USD"] == sum((to_decimal(t) for t in totals), Decimal(0))


async def test_total_bounds_are_inclusive_and_a_bad_total_never_satisfies_one(
    db_session: AsyncSession,
) -> None:
    await store(
        db_session,
        [
            simple_document("low.png", total=9.99),
            simple_document("edge.png", total=10.0),
            simple_document("text.png", total="10.00"),
            simple_document("high.png", total=10.01),
            simple_document("bad.png", total="ten"),
            simple_document("none.png", total=None),
            simple_document("huge.png", total=1e12),
        ],
    )

    for tool_input in (
        {"min_total": 10},
        {"max_total": 10},
        {"min_total": 10, "max_total": 10},
        {"min_total": 0, "max_total": 1e13},
        {"max_total": -1},
    ):
        new, old = await run_both(db_session, tool_input)
        assert new == old, tool_input
    both = await run_both(db_session, {"min_total": 10, "max_total": 10})
    assert summary_of(both[0]) == "2 matching document(s). Sum of totals: 20.00 USD."


async def test_dates_are_inclusive_and_only_real_dates_count(db_session: AsyncSession) -> None:
    await store(
        db_session,
        [
            simple_document(f"{i}.png", document_date=value)
            for i, value in enumerate(
                ["2026-03-01", "2026-03-31", "2026-04-01", "2026-02-30", "20260315", "2026-W11-7", None, 20260315]
            )
        ],
    )

    for tool_input in (
        {"date_from": "2026-03-01", "date_to": "2026-03-31"},
        {"date_from": "2026-03-31"},
        {"date_to": "2026-03-01"},
        {"date_from": "2026-03-15", "date_to": "2026-03-15"},
    ):
        new, old = await run_both(db_session, tool_input)
        assert new == old, tool_input


async def test_the_latest_extraction_and_reviews_decide(db_session: AsyncSession) -> None:
    older = StoredExtraction(0, {"vendor": StoredField(leaf("Old Name")), "total": StoredField(leaf(1.0))})
    newer = StoredExtraction(
        5,
        {
            "vendor": StoredField(leaf("Model Name"), "corrected", "Corrected Name"),
            "total": StoredField(leaf(2.0), "corrected", "30.00"),
            "currency": StoredField(leaf("USD"), "approved"),
        },
    )
    hidden = StoredExtraction(9, {})
    await store(
        db_session,
        [
            StoredDocument("two.png", 0, [older, newer]),
            StoredDocument("empty-latest.png", 1, [older, hidden]),
            StoredDocument("never.png", 2, [], "extracted"),
            StoredDocument("failed.png", 3, [newer], "failed"),
        ],
    )

    for tool_input in ({}, {"vendor": "corrected"}, {"vendor": "old"}, {"min_total": 20}):
        new, old = await run_both(db_session, tool_input)
        assert new == old, tool_input
    only, _ = await run_both(db_session, {"vendor": "corrected"})
    assert summary_of(only) == "1 matching document(s). Sum of totals: 30.00 USD."


async def test_more_than_the_listing_limit_match_and_the_rest_are_counted(
    db_session: AsyncSession,
) -> None:
    await store(
        db_session,
        [
            simple_document(f"{i:03}.png", total=1.5, minutes=i, vendor="Acme" if i % 7 else "Other")
            for i in range(45)
        ],
    )

    new, old = await run_both(db_session, {"vendor": "acme"})

    assert new == old
    assert summary_of(new) == (
        "38 matching document(s). Sum of totals: 57.00 USD. "
        "Listing the first 20; narrow the filters to see the rest."
    )
    assert len(new) == 1 + tools_module.RECORDS_MAX


async def test_a_corrected_line_item_list_that_is_not_rows_fails_as_it_did(
    db_session: AsyncSession,
) -> None:
    await store(
        db_session,
        [
            StoredDocument(
                "odd.png",
                0,
                [StoredExtraction(0, {"line_items": StoredField([], "corrected", ["not a row"])})],
            )
        ],
    )
    ctx = ToolContext(session=db_session, embedder=EMBEDDER)

    with pytest.raises(AttributeError):
        await oracle_query_extractions(ctx, {"item": "tea"})
    with pytest.raises(AttributeError):
        await query_extractions(ctx, {"item": "tea"})
    # Nothing reads the items without an item filter.
    new, old = await run_both(db_session, {"vendor": "x"})
    assert new == old


# --- bounded work -----------------------------------------------------------------------------------------------------------


class Statements:
    """The SQL the engine runs, and the ExtractedField rows the session loads."""

    def __init__(self) -> None:
        self.sql: list[str] = []
        self.loaded = 0

    def _statement(self, conn, cursor, statement, parameters, context, executemany) -> None:
        self.sql.append(statement)

    def _loaded(self, target, context) -> None:
        self.loaded += 1

    def __enter__(self) -> Self:
        event.listen(engine.sync_engine, "before_cursor_execute", self._statement)
        event.listen(ExtractedField, "load", self._loaded)
        return self

    def __exit__(self, *exc: object) -> None:
        event.remove(engine.sync_engine, "before_cursor_execute", self._statement)
        event.remove(ExtractedField, "load", self._loaded)

    def reading_fields(self) -> list[str]:
        return [s for s in self.sql if "extracted_fields" in s]


async def test_the_work_is_bounded_by_the_listing_not_by_the_corpus(db_session: AsyncSession) -> None:
    corpus = 150
    await store(
        db_session,
        [
            simple_document(f"{i:03}.png", total=2.25, minutes=i, items=("Tea", "Cake"))
            for i in range(corpus)
        ],
    )
    ctx = ToolContext(session=db_session, embedder=EMBEDDER)
    db_session.expunge_all()

    with Statements() as seen:
        content = await query_extractions(ctx, {"currency": "usd", "item": "tea", "min_total": 1})

    assert summary_of(content).startswith(f"{corpus} matching document(s). Sum of totals: 337.50 USD.")
    assert len(content) == 1 + tools_module.RECORDS_MAX
    # Totals and listing ids in SQL, then the records of the listing alone.
    assert len(seen.reading_fields()) <= 3
    fields_per_document = 5
    assert seen.loaded == tools_module.RECORDS_MAX * fields_per_document

    # The implementation this replaced read every field of the organization.
    db_session.expunge_all()
    with Statements() as before:
        await load_records(db_session, None)
    assert before.loaded >= corpus * fields_per_document


async def test_currencies_in_any_script_or_case_are_grouped_without_leaving_sql(
    db_session: AsyncSession,
) -> None:
    currencies = ["\u20ac", " usd", "USD", "Usd\u00a0", "eur", "EUR", "\u00df", "\u20ac ", None, 5, "", "  "]
    await store(
        db_session,
        [
            simple_document(f"{i:02}.png", total=1.5 + i, currency=currencies[i % len(currencies)], minutes=i)
            for i in range(48)
        ],
    )
    ctx = ToolContext(session=db_session, embedder=EMBEDDER)

    # No filter: every document is settled, so only the listing is loaded.
    db_session.expunge_all()
    with Statements() as seen:
        content = await query_extractions(ctx, {})
    assert seen.loaded == tools_module.RECORDS_MAX * 5
    assert len(seen.reading_fields()) <= 2
    assert "USD" in summary_of(content) and "\u20ac" in summary_of(content)

    for tool_input in ({}, {"min_total": 10}, {"currency": "usd"}, {"currency": "\u20ac"}, {"vendor": "acme"}):
        new, old = await run_both(db_session, tool_input)
        assert new == old, tool_input


async def test_documents_sql_cannot_judge_are_loaded_and_judged_in_python(
    db_session: AsyncSession,
) -> None:
    await store(
        db_session,
        [simple_document(f"plain{i}.png", vendor="Acme") for i in range(40)]
        + [
            simple_document("accent.png", vendor="Café"),
            simple_document("number.png", vendor=7),
            simple_document("sharp.png", vendor="Straße"),
        ],
    )
    ctx = ToolContext(session=db_session, embedder=EMBEDDER)
    fields_per_document = 5

    # An accent cannot fold into an ASCII letter, so an ASCII needle is decided
    # in SQL for it. A number and a sharp s are not: those two are loaded and
    # judged by ExtractionFilters.matches, besides the listing.
    db_session.expunge_all()
    with Statements() as seen:
        await query_extractions(ctx, {"vendor": "acme"})
    assert seen.loaded == (tools_module.RECORDS_MAX + 2) * fields_per_document

    db_session.expunge_all()
    with Statements() as seen:
        content = await query_extractions(ctx, {"vendor": "cafe"})
    assert summary_of(content) == "0 matching document(s)."
    assert seen.loaded == 2 * fields_per_document

    for needle in ("ss", "7", "STRASSE", "café", "acme"):
        new, old = await run_both(db_session, {"vendor": needle})
        assert new == old, needle
    assert summary_of((await run_both(db_session, {"vendor": "ss"}))[0]).startswith("1 matching")


# --- tenancy ----------------------------------------------------------------------------------------------------------------------


async def test_the_statements_stay_inside_the_organization(owner_session: AsyncSession) -> None:
    """On the owner's connection row-level security does not apply, so this is
    the application layer alone: every statement of the tool, the CTE and the
    lateral subqueries included, is filtered to the session's organization."""
    session = owner_session
    other = Organization(name="Other", slug=f"other-{uuid.uuid4().hex[:8]}")
    session.add(other)
    await session.flush()
    await bind_tenant(session, other.id)
    theirs = await store(session, [simple_document("theirs.png", vendor="Acme", total=999.0)])
    await bind_tenant(session, DEFAULT_ORGANIZATION_ID)
    ours = await store(session, [simple_document("ours.png", vendor="Acme", total=1.0)])

    for filters in ({}, {"vendor": "acme"}, {"currency": "usd", "min_total": 0, "date_from": "2026-01-01"}):
        content = await query_extractions(ToolContext(session=session, embedder=EMBEDDER), filters)
        assert summary_of(content) == "1 matching document(s). Sum of totals: 1.00 USD."
        assert str(ours[0]) in str(content) and str(theirs[0]) not in str(content)


async def test_row_level_security_alone_keeps_the_tool_in_its_organization(
    db_session: AsyncSession, second_organization: Organization
) -> None:
    await bind_tenant(db_session, second_organization.id)
    await store(db_session, [simple_document("theirs.png", total=999.0)])
    await bind_tenant(db_session, DEFAULT_ORGANIZATION_ID)
    content = await query_extractions(ToolContext(session=db_session, embedder=EMBEDDER), {})
    assert "999" not in str(content)


# --- the summary pieces -------------------------------------------------------------------------------------------------------


def test_match_totals_group_by_normalized_currency_and_skip_missing_totals() -> None:
    totals = MatchTotals()
    totals.add_group("USD", 3, 1, Decimal("5.50"))
    totals.add_group("EUR", 1, 1, None)
    totals.add_record(DocumentRecord(uuid.uuid4(), "r.png", {"total": "4.5", "currency": " usd "}, []))
    totals.add_record(DocumentRecord(uuid.uuid4(), "r.png", {"total": None, "currency": 5}, []))

    assert totals.count == 6
    assert totals.missing_total == 3
    assert totals.sums == {"USD": Decimal("10.00")}
