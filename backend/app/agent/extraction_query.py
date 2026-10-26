"""Which documents a query_extractions call matches, decided by the database.

`ExtractionFilters.matches` is the definition: it takes the human-verified
record of one document (app/records.py) and says whether the call's filters
accept it. Running it over every record of an organization is what the tool
used to do, and it cost a row per extracted field in Python on the event
loop, whatever the filters. `find_matches` reaches the same answer while
loading at most `limit` records, besides the pending documents below: the
filters, the count, and the per-currency sums run in SQL over each document's
latest extraction, and only the listing loads full records.

SQL can decide a filter exactly only where it computes what Python computes.
Each filter is a three-valued SQL expression: TRUE or FALSE where the two
agree for certain, NULL (unknown) where they might not, and PostgreSQL's AND
combines them as a conjunction should. A document the SQL accepts for certain
is settled; a document it cannot rule out and cannot accept is pending, and
pending documents are loaded and judged by `matches` itself, so the answer is
the Python one whatever the data holds. What is unknown, and why:

- A text filter on a value that is not a JSON string or null. Python reads
  `str(value or "")` of it, which no SQL expression reproduces.
- A case-insensitive match against a value with a character that Python's
  case fold treats differently from PostgreSQL's lower() (a sharp s folds to
  "ss", a Kelvin sign to "k", and so on: CASEFOLD_TO_ASCII). Plain ASCII, and
  any other character that cannot fold into an ASCII letter, is decided in
  SQL, with an ASCII-only lower-casing that does not depend on the database's
  locale. A needle that does not case-fold to ASCII can match only a haystack
  with a non-ASCII character, and that is the haystack SQL leaves to Python.
- A currency filter, or a currency being matched, that is not ASCII. The
  summary's currency groups are never unknown: SQL groups by the stripped
  text and Python normalizes each group.
- A document date that is not `YYYY-MM-DD` but could still be one of the other
  ISO 8601 forms `date.fromisoformat` reads (a basic date, a week date), or
  that is not a JSON string. A string that cannot be any of them, and a null,
  is decided: it is not a date.
- A line-item correction that is not a list of objects, which Python iterates
  or fails on as the shape dictates.
- A total written as text of more than _AMOUNT_TEXT_LIMIT characters.

Everything else is exact. A total is a JSON number or a plain decimal string
(`to_decimal` reads the same two), read as `numeric`, so a sum is exact where
Python's is; a total that is missing, unparseable, or at or beyond MAX_AMOUNT
has no value and cannot satisfy a bound. User text is only ever a bound
parameter compared with strpos(), never a LIKE pattern, so `%` and `_` are
ordinary characters. A JSON number reads back exactly because SQLAlchemy
serializes it with json.dumps, which writes a float as repr(float), the
shortest text that round-trips; `to_decimal` reads that same float through
str(), so `numeric` and Decimal hold the same value. `pg_input_is_valid`
needs PostgreSQL 16 (the compose image, the CI service, and the test server
are 16.15).

Every statement is an ORM select on the session, so the organization filter
and row-level security apply to it exactly as to any query.
"""

import string
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, Inexact, localcontext
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Numeric,
    Text,
    Uuid,
    and_,
    case,
    cast,
    column,
    false,
    func,
    literal,
    null,
    select,
    true,
    union_all,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, distinct_on
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.models import Document, ExtractedField, Extraction
from app.money import MAX_AMOUNT, MoneyError, currency_precision, to_decimal
from app.records import (
    LATEST_EXTRACTION_ORDER,
    DocumentRecord,
    load_records,
    scalar_value_sql,
)

UNKNOWN_CURRENCY = "unknown currency"

# The characters Python's str.strip() removes (str.isspace, Unicode 15). The
# database strips the same set from a stored amount or currency, so " 5.00\u00a0"
# is read the way to_decimal reads it. tests/test_extraction_query.py pins the
# string against the interpreter's own table.
PYTHON_WHITESPACE = (
    "\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
)

# The non-ASCII characters whose case fold contains an ASCII letter: they can
# take part in a match against an ASCII needle, which SQL's lower-casing would
# miss. Pinned the same way.
CASEFOLD_TO_ASCII = (
    "\u00df\u0130\u0149\u017f\u01f0\u1e96\u1e97\u1e98\u1e99\u1e9a\u1e9e\u212a"
    "\ufb00\ufb01\ufb02\ufb03\ufb04\ufb05\ufb06"
)

# A total written as text longer than this is judged by Python: the database
# would refuse a numeric of unbounded size, and no stored amount is near it.
_AMOUNT_TEXT_LIMIT = 1000

# `to_decimal`'s plain decimal (ASCII digits only), and the other ISO 8601
# date forms that `date.fromisoformat` reads on top of YYYY-MM-DD: YYYYMMDD,
# YYYY-Www, YYYYWww, YYYY-Www-D, and YYYYWwwD. Loose on purpose: a string that
# matches is handed to Python, which says whether it really is a date.
_PLAIN_DECIMAL = r"^-?[0-9]+(\.[0-9]+)?$"
_ISO_DATE = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
_OTHER_ISO_DATE_FORMS = r"^[0-9]{4}-?(W[0-9]{2}(-?[0-9])?|[0-9]{4})$"


def usable_total(value: Any) -> Decimal | None:
    """A stored amount as a Decimal, or None when it is not a usable amount:
    unparseable, or at or above the bound a correction may carry, so sums
    stay inside the Decimal context's precision and never round silently."""
    try:
        amount = to_decimal(value)
    except MoneyError:
        return None
    return amount if abs(amount) < MAX_AMOUNT else None


def _is_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


@dataclass(frozen=True)
class ExtractionFilters:
    """The filters of one query_extractions call, parsed. A missing or empty
    text filter, and a missing date or total bound, filters nothing."""

    vendor: str | None = None
    item: str | None = None
    currency: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    min_total: Decimal | None = None
    max_total: Decimal | None = None

    @property
    def matches_nothing(self) -> bool:
        """Whether a text filter holds a character no stored text can hold (a
        NUL, or a lone surrogate that is not UTF-8), so that no document
        matches it. The database cannot take such a parameter, and Python's
        answer for it is always no."""
        for text in (self.vendor, self.item, self.currency):
            if text and ("\x00" in text or not _is_utf8(text)):
                return True
        return False

    def matches(self, record: DocumentRecord) -> bool:
        """Whether `record` passes every filter: vendor and item are
        case-insensitive substrings (an item of any line), the currency a
        case-insensitive equality, the dates and totals inclusive bounds. A
        document without a parseable date, or without a usable total, cannot
        satisfy a bound on it."""
        fields = record.fields
        if self.vendor and self.vendor.casefold() not in str(fields.get("vendor") or "").casefold():
            return False
        if self.item and not any(
            self.item.casefold() in str(row.get("description") or "").casefold()
            for row in fields.get("line_items") or []
        ):
            return False
        if self.currency and str(fields.get("currency") or "").upper() != self.currency.upper():
            return False
        if self.date_from or self.date_to:
            try:
                document_date = date.fromisoformat(str(fields.get("document_date")))
            except ValueError:
                return False
            if (self.date_from and document_date < self.date_from) or (
                self.date_to and document_date > self.date_to
            ):
                return False
        if self.min_total is None and self.max_total is None:
            return True
        # A total that is missing or unparseable cannot satisfy a bound.
        total = usable_total(fields.get("total"))
        if total is None:
            return False
        return (self.min_total is None or total >= self.min_total) and (
            self.max_total is None or total <= self.max_total
        )


@contextmanager
def _exact_sums() -> Iterator[None]:
    """A Decimal context wide enough that adding floats below MAX_AMOUNT is
    exact (a float can print down to 1e-308), and trapped so a sum that is not
    exact fails loudly instead of rounding."""
    with localcontext() as context:
        context.prec = 400
        context.traps[Inexact] = True
        yield


@dataclass
class MatchTotals:
    """What the summary line of a query_extractions result reports: how many
    documents matched, how many of them have no usable total, and the sum of
    the totals per currency (a currency with no usable total has no entry)."""

    count: int = 0
    missing_total: int = 0
    sums: dict[str, Decimal] = field(default_factory=dict)

    def add_group(self, currency: str, count: int, missing_total: int, amount: Decimal | None) -> None:
        """Add `count` documents of one currency, `missing_total` of them with
        no usable total, whose usable totals sum to `amount` (None when none
        has one)."""
        self.count += count
        self.missing_total += missing_total
        if amount is not None:
            with _exact_sums():
                self.sums[currency] = self.sums.get(currency, Decimal(0)) + amount

    def add_record(self, record: DocumentRecord) -> None:
        """Add one record, grouped under its currency normalized, so "usd" and
        "USD" are one sum (as ExtractionFilters.matches treats them)."""
        total = usable_total(record.fields.get("total"))
        currency = currency_precision(record.fields.get("currency")).code or UNKNOWN_CURRENCY
        self.add_group(currency, 1, 0 if total is not None else 1, total)


@dataclass
class ExtractionMatches:
    """The outcome of a query: totals over every matching document, and the
    first `limit` of them as full records, in upload order."""

    totals: MatchTotals
    records: list[DocumentRecord]


# --- the filters as three-valued SQL --------------------------------------------
#
# Each helper returns a SQL boolean that is TRUE or FALSE where the database is
# certain what the Python rule says, and NULL where it is not (the module
# docstring lists where).

_ASCII_UPPER = string.ascii_uppercase
_ASCII_LOWER = string.ascii_lowercase


def _unknown() -> ColumnElement[Any]:
    return cast(null(), Boolean)


def _json_type(value: Any) -> Any:
    """The JSON type name of `value`; SQL NULL (a field that is not there) is
    "null", like a JSON null."""
    return func.coalesce(func.jsonb_typeof(value), "null")


def _json_text(value: Any) -> Any:
    """The text of a JSON string (meaningless for any other type)."""
    return value.op("#>>", return_type=Text)(cast(literal("{}"), ARRAY(Text)))


def _json_get(value: Any, key: str) -> Any:
    """`value -> key`: the member of a JSON object, NULL for any other type."""
    return value.op("->", return_type=JSONB)(key)


def _is_ascii(text: Any) -> Any:
    return func.octet_length(text) == func.char_length(text)


def _string_state(value: Any, test: Callable[[Any], Any]) -> Any:
    """`test` of a JSON string's text. A null or missing value reads as the
    empty text, which no non-empty filter accepts; any other JSON type is
    unknown (Python takes `str()` of it)."""
    kind = _json_type(value)
    return case(
        (kind == "string", test(_json_text(value))),
        (kind == "null", false()),
        else_=_unknown(),
    )


def _contains_folded(needle: str) -> Callable[[Any], Any]:
    """The test "the text's case fold contains needle's" (needle not empty)."""
    folded = needle.casefold()
    if not folded.isascii():
        # An ASCII text folds to ASCII, so it cannot hold the needle.
        return lambda text: case((_is_ascii(text), false()), else_=_unknown())

    def test(text: Any) -> Any:
        lowered = func.translate(text, _ASCII_UPPER, _ASCII_LOWER)
        return case(
            # Found in the characters that fold alone: a match for Python too.
            (func.strpos(lowered, folded) > 0, true()),
            # Not found, but a character that folds to ASCII is there to match.
            (func.translate(text, CASEFOLD_TO_ASCII, "") != text, _unknown()),
            else_=false(),
        )

    return test


def _equals_upper(needle: str) -> Callable[[Any], Any]:
    """The test "the text, upper-cased, is needle upper-cased" (needle not
    empty). Only ASCII is decided here."""
    upper = needle.upper()
    if not upper.isascii():
        return lambda text: case((_is_ascii(text), false()), else_=_unknown())
    return lambda text: case(
        (_is_ascii(text), func.translate(text, _ASCII_LOWER, _ASCII_UPPER) == upper),
        else_=_unknown(),
    )


def _date_state(value: Any, date_from: date | None, date_to: date | None) -> Any:
    """Whether the value reads as a date within the bounds. A string that is
    exactly YYYY-MM-DD is a date when the calendar has it (pg_input_is_valid
    refuses February 30 without raising); one that can be another ISO form is
    unknown; any other string is not a date."""

    def test(text: Any) -> Any:
        parsed = cast(text, Date)
        bounds = []
        if date_from:
            bounds.append(parsed >= date_from)
        if date_to:
            bounds.append(parsed <= date_to)
        return case(
            (
                text.regexp_match(_ISO_DATE),
                case((func.pg_input_is_valid(text, "date"), and_(true(), *bounds)), else_=false()),
            ),
            (text.regexp_match(_OTHER_ISO_DATE_FORMS), _unknown()),
            else_=false(),
        )

    return _string_state(value, test)


def _items_state(array: Any, corrected: Any, test: Callable[[Any], Any]) -> Any:
    """Whether any line item's description passes `test`. `array` is the
    field's stored list of cells (a leaf per cell) or, when `corrected`, the
    reviewer's list of plain rows; anything else holds no items, except a
    correction that is not a list, which Python would iterate or fail on."""
    kind = _json_type(array)
    # Only a list is unnested: the function refuses any other JSON type.
    elements = func.jsonb_array_elements(
        case((kind == "array", array), else_=literal([], JSONB))
    ).table_valued(column("value", JSONB))
    element = elements.c.value
    state = case(
        (
            corrected,
            case(
                (_json_type(element) == "object", _string_state(_json_get(element, "description"), test)),
                else_=_unknown(),
            ),
        ),
        else_=_string_state(_json_get(_json_get(element, "description"), "value"), test),
    )
    # TRUE (2) beats unknown (1) beats FALSE (0), so one max() over the items says
    # whether any is TRUE, else whether any is unknown; a simple CASE evaluates
    # each item's state once.
    rank = case({True: 2, False: 0}, value=state, else_=1)
    best = select(func.max(rank)).select_from(elements).scalar_subquery()
    any_item = case({2: true(), 1: _unknown()}, value=best, else_=false())
    return case((kind == "array", any_item), (kind == "null", false()), (corrected, _unknown()), else_=false())


def _total_columns(value: Any) -> tuple[Any, Any]:
    """(the usable total as numeric or NULL, whether Python must read it) for a
    total's JSON value: a number, or a plain decimal string once stripped, below
    MAX_AMOUNT; anything else has no value, as `usable_total` finds."""
    kind = _json_type(value)
    raw = _json_text(value)
    stripped = func.btrim(raw, PYTHON_WHITESPACE)
    parsed = case(
        (kind == "number", cast(value, Numeric)),
        (
            and_(
                kind == "string",
                func.char_length(raw) <= _AMOUNT_TEXT_LIMIT,
                stripped.regexp_match(_PLAIN_DECIMAL),
            ),
            cast(stripped, Numeric),
        ),
        else_=cast(null(), Numeric),
    )
    usable = case((func.abs(parsed) < MAX_AMOUNT, parsed), else_=cast(null(), Numeric))
    return usable, and_(kind == "string", func.char_length(raw) > _AMOUNT_TEXT_LIMIT)


def _currency_text(value: Any) -> Any:
    """The summary's group key for a currency's JSON value: the text stripped of
    Python's whitespace, or NULL (the unknown currency) for a value that is not
    a string or is empty once stripped. Upper-casing is left to Python
    (`currency_precision`), which `find_matches` applies to each group, so any
    script is grouped exactly and equal codes merge there."""
    stripped = func.btrim(_json_text(value), PYTHON_WHITESPACE)
    return case(
        (and_(_json_type(value) == "string", stripped != ""), stripped),
        else_=cast(null(), Text),
    )


def _facts(filters: ExtractionFilters, document_ids: list[uuid.UUID] | None) -> Any:
    """One row per extracted document in scope that has a field row in its
    latest extraction, holding the JSON of the fields the filters and the
    summary read (as `field_value` reads them: a correction wins), and the sort
    key. The fields come from one pass over the extraction's rows, pivoted by
    name; an extraction with no rows has no row here, as it had no record."""
    latest = (
        select(Extraction.id, Extraction.document_id)
        .join(Document, Document.id == Extraction.document_id)
        .where(Document.status == "extracted")
        .ext(distinct_on(Extraction.document_id))
        .order_by(Extraction.document_id, *LATEST_EXTRACTION_ORDER)
    )
    if document_ids is not None:
        latest = latest.where(Document.id.in_(document_ids))
    latest = latest.subquery("latest")

    def pivot(name: str, value: Any) -> Any:
        return (func.array_agg(value).filter(ExtractedField.field_name == name))[1]

    wanted = ["total", "currency"]
    if filters.vendor:
        wanted.append("vendor")
    if filters.date_from or filters.date_to:
        wanted.append("document_date")
    columns: list[Any] = [Document.id, Document.created_at, Document.filename]
    columns.extend(pivot(name, scalar_value_sql(ExtractedField)).label(name) for name in wanted)
    if filters.item:
        # Line items are not a leaf: the stored list itself, or the correction.
        corrected = ExtractedField.review_action == "corrected"
        items = case((corrected, ExtractedField.corrected_value), else_=ExtractedField.value)
        columns.append(pivot("line_items", items).label("items"))
        columns.append(pivot("line_items", corrected).label("items_corrected"))
    return (
        select(*columns)
        .select_from(latest)
        .join(Document, Document.id == latest.c.document_id)
        .join(ExtractedField, ExtractedField.extraction_id == latest.c.id)
        .group_by(Document.id)
        .subquery("facts")
    )


def _scored(filters: ExtractionFilters, document_ids: list[uuid.UUID] | None) -> Any:
    """One row per extracted document in scope, from its latest extraction:
    its sort key, whether the filters accept it (`matched`: true, false, or
    null for unknown), the summary's currency group and usable total, and
    `settled`, whether SQL alone decides everything about it."""
    facts = _facts(filters, document_ids)
    total, total_unknown = _total_columns(facts.c.total)
    # A fence (OFFSET 0) keeps PostgreSQL from folding these back into every
    # place they are used, which would parse the total a dozen times per row.
    derived = (
        select(
            *[column_ for column_ in facts.c if column_.key != "total"],
            total.label("total"),
            total_unknown.label("total_unknown"),
            _currency_text(facts.c.currency).label("currency_text"),
        )
        .offset(0)
        .subquery("derived")
    )

    states = []
    if filters.vendor:
        states.append(_string_state(derived.c.vendor, _contains_folded(filters.vendor)))
    if filters.item:
        states.append(
            _items_state(derived.c["items"], derived.c.items_corrected, _contains_folded(filters.item))
        )
    if filters.currency:
        states.append(_string_state(derived.c.currency, _equals_upper(filters.currency)))
    if filters.date_from or filters.date_to:
        states.append(_date_state(derived.c.document_date, filters.date_from, filters.date_to))
    if filters.min_total is not None or filters.max_total is not None:
        bounds = [
            bound
            for bound in (
                derived.c.total >= filters.min_total if filters.min_total is not None else None,
                derived.c.total <= filters.max_total if filters.max_total is not None else None,
            )
            if bound is not None
        ]
        states.append(
            case(
                (derived.c.total_unknown, _unknown()),
                (derived.c.total.is_(None), false()),
                else_=and_(*bounds),
            )
        )
    judged = (
        select(
            derived.c.id,
            derived.c.created_at,
            derived.c.filename,
            and_(true(), *states).label("matched"),
            derived.c.currency_text,
            derived.c.total,
            derived.c.total_unknown,
        )
        .offset(0)
        .subquery("judged")
    )
    settled = and_(judged.c.matched.is_(True), judged.c.total_unknown.is_(False))
    return select(
        judged.c.id,
        judged.c.created_at,
        judged.c.filename,
        judged.c.matched,
        judged.c.currency_text.label("currency"),
        judged.c.total,
        settled.label("settled"),
    ).cte("scored")


def _answer_statement(
    filters: ExtractionFilters, document_ids: list[uuid.UUID] | None, limit: int
) -> Any:
    """The one statement of find_matches: per-currency totals of the settled
    matches (rank 0), then the first `limit` settled documents and every pending
    one (rank 1), in upload order."""
    scored = _scored(filters, document_ids)
    no_id, no_time, no_name = cast(null(), Uuid), cast(null(), DateTime(timezone=True)), cast(null(), Text)
    groups = (
        select(
            literal(0).label("rank"),
            scored.c.currency.label("currency"),
            func.count().label("documents"),
            func.count().filter(scored.c.total.is_(None)).label("missing_total"),
            func.sum(scored.c.total).label("amount"),
            no_id.label("id"),
            false().label("settled"),
            no_time.label("created_at"),
            no_name.label("filename"),
        )
        .where(scored.c.settled)
        .group_by(scored.c.currency)
    )

    def listed(settled: bool) -> Any:
        return select(
            literal(1).label("rank"),
            cast(null(), Text).label("currency"),
            cast(null(), BigInteger).label("documents"),
            cast(null(), BigInteger).label("missing_total"),
            cast(null(), Numeric).label("amount"),
            scored.c.id.label("id"),
            literal(settled).label("settled"),
            scored.c.created_at.label("created_at"),
            scored.c.filename.label("filename"),
        )

    settled_first = (
        listed(True)
        .where(scored.c.settled)
        .order_by(scored.c.created_at, scored.c.filename, scored.c.id)
        .limit(limit)
    )
    pending = listed(False).where(scored.c.matched.is_not(False), scored.c.settled.is_(False))
    return union_all(groups, settled_first, pending).order_by("rank", "created_at", "filename", "id")


async def find_matches(
    session: AsyncSession,
    filters: ExtractionFilters,
    document_ids: list[uuid.UUID] | None,
    limit: int,
) -> ExtractionMatches:
    """The documents `filters` match among the extracted ones of the session's
    organization (narrowed to `document_ids` when given): totals over all of
    them, and the first `limit` as records in upload order.

    One statement answers everything SQL can: the totals of the settled
    matches per currency, the first `limit` settled documents, and every
    pending one, in upload order (created_at, filename, id, the order
    load_records uses). Then the records of the listing are loaded, with the
    pending documents (the module docstring says which) before them, to be
    judged by `ExtractionFilters.matches`."""
    totals = MatchTotals()
    if filters.matches_nothing:
        return ExtractionMatches(totals, [])
    rows = (await session.execute(_answer_statement(filters, document_ids, limit))).all()

    listing = []
    for row in rows:
        if row.rank == 0:
            code = currency_precision(row.currency).code if row.currency is not None else None
            totals.add_group(code or UNKNOWN_CURRENCY, row.documents, row.missing_total, row.amount)
        else:
            listing.append(row)

    loaded: dict[uuid.UUID, DocumentRecord] = {}
    undecided = [row.id for row in listing if not row.settled]
    if undecided:
        for record in await load_records(session, undecided):
            if filters.matches(record):
                loaded[record.document_id] = record
                totals.add_record(record)
    chosen = [row.id for row in listing if row.settled or row.id in loaded][:limit]
    missing = [document_id for document_id in chosen if document_id not in loaded]
    if missing:
        loaded.update({record.document_id: record for record in await load_records(session, missing)})
    return ExtractionMatches(totals, [loaded[i] for i in chosen if i in loaded])
