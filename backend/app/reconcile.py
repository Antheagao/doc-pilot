"""Reconciliation: do a document's amounts add up?

The model reads the printed numbers; code decides whether they agree with each
other. `reconcile` takes the amounts a document carries (subtotal, tax, total,
every line item's quantity, unit price, and total, and the v2 extras: discount,
shipping, tip, amount paid and due, tax lines, and the two dates) and runs a
fixed set of rules over them, in `Decimal`, with no I/O. The result is a
`Report`: every check with the values it used, an overall status, the
assumptions made, and at most one suggested correction. `report_to_json` is the
only serializer.

Rules (`RULES_VERSION`; a new rule or tolerance is a new version):

- `line_math[i]`: quantity x unit price = line total. Exact when the product is
  exact in the currency's minor unit; one minor unit of slack only when the
  product needs rounding (3 x 0.333 = 0.999 printed as 1.00).
- `lines_sum`: the line totals add up to the subtotal, with half a minor unit of
  slack for each line whose product needed rounding.
- `total_math`: subtotal - discount + shipping + tax + tip = total, exact. An
  adjustment the document does not print adds no term, so a record without them
  reads "subtotal + tax = total". A null tax counts as 0 and says so in the
  assumptions.
- `tax_rate`: tax / subtotal is between 0 and 0.30, judged on the rate as it is
  reported (four places), so the verdict never contradicts the number shown.
- `non_negative:<subtotal|tax|total>`: the header amounts are not negative. Line
  totals and quantities may be negative (discounts, returns). The discount,
  shipping, tip, and amount paid get the same check when the document has them.
- `lines_present`: a printed subtotal needs line items to add up.
- `tax_lines_sum`: the tax line amounts add up to the tax, exact. A tax that is
  not printed is computed from them (a derived entry, like a subtotal) and the
  check is skipped.
- `tax_line_math[i]`: a tax line's amount = rate / 100 x base, where the rate is
  the printed percent (`"8.875"`). One minor unit of slack only when the product
  needs rounding. A base that is not printed is taken as the subtotal less the
  discount, then (when shipping is printed) that plus shipping, whichever first
  makes the line pass; the one used is written to `assumptions`.
- `balance_math`: total - amount paid = amount due, exact. No amount paid beside
  a printed amount due counts as 0 and says so; no amount due is computed.
- `due_date_order`: the due date is not before the document date. A date that is
  not `YYYY-MM-DD` skips the check, with the reason (there is no `format:` check
  for dates).
- `format:<path>`: emitted only on failure: the value is not a number, is not
  finite, is at least MAX_AMOUNT, or has more decimals than allowed (the
  currency's minor unit for money, MAX_QUANTITY_DECIMALS for quantities and
  unit prices). Checks that need that value are skipped, with the reason.
- `quote:<path>`: the number in the value's evidence quote equals the value
  (app/quotes.py), for each numeric leaf that has an `evidence` key: the
  header amounts, a line's quantity, unit price, and total, and a tax line's
  rate, base, and amount. A mismatch means the model wrote a number it did not
  read. A leaf with no `evidence` key (every v1 extraction) gets no check, and
  neither does a null value, an unreadable one (it has its `format:` check), or
  a field a person has reviewed. Evidence with no quote, or a zero whose quote
  prints no number ("FREE"), gives a skipped check. These come right after the format checks, before the
  arithmetic, and take part in no equation.

Compatibility: a v1-shaped record (none of the v2 extras) gets exactly the
`rules_v1` report apart from `rules_version`: the same checks, terms, formulas,
details, assumptions, and suggestion. A check that only exists in v2 is emitted
only when one of its v2 operands is present, and a v2 operand that is absent is
absent, never 0. tests/fixtures/rules_v1_reports.json pins this over the eval
labels, the committed runs, and the staged 016 record.

Three assumptions let a tax-inclusive or discount-net document pass. Each
applies only when the numbers are exactly consistent with the reading, is
written to `assumptions`, and marks the amount it neutralizes with the source
`included` (coefficient 0 in the equation: the tax or discount is already inside
the other amounts). They are tried only when the ordinary rule fails, so they
never change a document that already passes:

- Gross subtotal: a tax is printed (above 0), the subtotal (printed, or computed
  from the line totals) less the discount plus shipping and tip equals the total,
  and tax / (total - tax) is at most 0.30. `total_math` then reads "subtotal =
  total" (with the adjustments, "subtotal - discount + shipping + tip = total")
  and `tax_rate` is judged on the net base, total - tax.
- Gross lines: the line totals, less the discount plus shipping and tip, add up
  to the total (within the `lines_sum` tolerance), the ordinary `total_math`
  passes, and tax / subtotal is plausible. `lines_sum` then compares the line
  totals with the total.
- Discount applied: a discount is printed, the line totals less the discount
  equal the printed subtotal (within the `lines_sum` tolerance), and the total is
  that subtotal plus shipping, tax, and tip. The subtotal is read as already
  after the discount, so `total_math` does not take the discount off again, and
  a tax line without a printed base is checked against the subtotal itself.

A tax line without a printed base is checked against the subtotal less the
discount, or that plus shipping, as `tax_line_math[i]` describes. When the
gross-subtotal reading is in force the subtotal contains the tax, so those bases
start from the net amount, subtotal - tax.

A misread that does not fit any reading still fails. The one blind spot is a
total misread as exactly the subtotal with its adjustments (the subtotal less the
discount plus shipping and tip, which is the subtotal itself when there are none)
on a document whose tax is plausible: the numbers then look like a gross-subtotal
receipt, and only a person can tell.

A missing subtotal is computed from the line totals, and a missing line total
from quantity x unit price (a printed value that cannot be read is never
replaced). Those operands carry the source `computed`, feed the other checks,
and the check that defines them is skipped. A computed line total is rounded
half up, the way printers and tax authorities round amounts, and nothing else is
rounded. Arithmetic runs in a local `Decimal` context with Inexact trapped, so a
sum that is not exact fails loudly instead of drifting.

The tolerances assume each printed line total is within half a minor unit of
quantity x unit price (a printer that rounds). A printer that truncates can be
off by up to one unit per line; `line_math` allows that one unit, but `lines_sum`
allows only half a unit per rounded line and will fail such a receipt.

Every equation is linear: `sum(coef * operand) = rhs`. That is what lets the
suggestion solve for one operand, and what lets every rule join as more terms
without a new mechanism. The two products, quantity x unit price and rate / 100
x base, are written with the factor as the coefficient: the quantity of the unit
price, the rate (as a fraction) of the base.

The suggestion is deliberately strict. Candidates are the unverified header
amounts, line totals, and tax line amounts that sit in a failing equation, plus
an assumed (null) tax or amount paid. Each is solved for in each failing
equation that contains it, which gives proposals (path, value): exact in the
minor unit, and not negative for a header amount or tax line amount (a line
total may be negative: a discount or a return). Every proposal is applied and
every rule re-run; it survives only when every failing check passes and no
passing check fails. A suggestion is reported only when exactly one proposal
survives, so an ambiguous misread (a total that disagrees with subtotal + tax,
where either could be wrong) gets none, and neither does one whose only survivor
is an assumed amount: a null tax means the model could not read it, so a total
that is right must not be overwritten to match. (An equation loose because of
rounding has no exact solution and is judged by the re-run.) Quantities, unit
prices, rates, and bases are never suggested, and neither is a reviewed field.
The re-runs use the ordinary rules unless the document already leans on an
included reading, so a fix cannot succeed merely by making the document look
tax-inclusive or discount-net.

A failing quote check proposes the one number its quote prints (none when the
quote prints several, or an ambiguous one such as `1,234` that the field's
decimal places do not settle), with the sign of the current value, and goes
through the same filter. A number only a quote proposed must also land on a
check that can judge it, an equation that runs with the field in it; a field no
equation constrains (an amount paid with no amount due) is flagged without a
fix, so a reference number in its quote does not become one. That filter is what lets a
quote settle an ambiguous equation: the quote's number survives only when the
arithmetic then passes, and a competing arithmetic fix on another field is
rejected because the quote check still fails. When a proposal is applied, the
field keeps its quote only if the new value agrees with it; otherwise the quote
is dropped, as it would be once a person corrected the field (a reviewed field
has no quote check). So an arithmetic fix is not rejected merely because the
model's wrong value and its quote misread the same digit.
"""

import dataclasses
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from decimal import (
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    Context,
    Decimal,
    DivisionByZero,
    Inexact,
    InvalidOperation,
    localcontext,
)
from typing import TYPE_CHECKING, Any, NamedTuple

from app.extraction import _coerce_leaf
from app.money import (
    MAX_AMOUNT,
    MAX_QUANTITY_DECIMALS,
    CurrencyPrecision,
    MoneyError,
    currency_precision,
    fits_minor_unit,
    printed_amounts,
    shown_value,
    to_api_string,
    to_decimal,
    to_plain_string,
)
from app.quotes import STATUS_PASS as QUOTE_PASS
from app.quotes import is_skipped, quote_agreement
from app.records import field_value, leaf_value

if TYPE_CHECKING:
    from app.models import ExtractedField

RULES_VERSION = "rules_v2"

# Check and report status (strings, like the other closed sets here).
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIPPED = "skipped"

KIND_EQUATION = "equation"
KIND_BOUND = "bound"
KIND_PRESENCE = "presence"
KIND_FORMAT = "format"
KIND_QUOTE = "quote"

# Where an operand's value came from. The first three are a field's own state
# (the model's answer, a reviewer's approval of it, a reviewer's correction);
# the rest are made by this module.
SOURCE_EXTRACTED = "extracted"
SOURCE_APPROVED = "approved"
SOURCE_CORRECTED = "corrected"
SOURCE_COMPUTED = "computed"
SOURCE_ASSUMED = "assumed"
# A tax or discount already contained in the other amounts of the equation it
# sits in.
SOURCE_INCLUDED = "included"

LINES_PRESENT = "lines_present"
LINES_SUM = "lines_sum"
TOTAL_MATH = "total_math"
TAX_RATE = "tax_rate"
TAX_LINES_SUM = "tax_lines_sum"
BALANCE_MATH = "balance_math"
DUE_DATE_ORDER = "due_date_order"

# The highest tax / subtotal rate that is plausible on a receipt or invoice,
# and the places the rate is reported to.
MAX_TAX_RATE = Decimal("0.30")
RATE_PLACES = 4

# Digits of working precision. Generous enough that every product and sum of
# values bounded by the format checks (below 10^12, at most 6 decimals) is
# exact, so Inexact never fires on valid input.
_PRECISION = 400

_ONE = Decimal(1)

_LINE_TOTAL_PATH_RE = re.compile(r"line_items\[\d+\]\.total")
_TAX_LINE_AMOUNT_PATH_RE = re.compile(r"tax_lines\[\d+\]\.amount")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}", re.ASCII)
_TAX_LINE_KEYS = ("label", "rate", "base", "amount")
# The cells of a line item and of a tax line that are numbers (and so can carry
# a quote), in the order their quote checks come.
_LINE_NUMBER_KEYS = ("quantity", "unit_price", "total")
_TAX_LINE_NUMBER_KEYS = ("rate", "base", "amount")

# The header amounts a suggestion may name, and the ones a person only ever
# reads as non-negative. A null tax or amount paid is counted as 0 and can make
# a fix ambiguous.
_HEADER_MONEY_PATHS = (
    "subtotal", "tax", "total", "discount", "shipping", "tip", "amount_paid", "amount_due",
)
_NEVER_NEGATIVE_PATHS = tuple(path for path in _HEADER_MONEY_PATHS if path != "amount_due")
_ASSUMED_NULL_PATHS = ("tax", "amount_paid")
# The v2 amounts that get a non_negative check when the document has them.
_NON_NEGATIVE_EXTRAS = ("discount", "shipping", "tip", "amount_paid")


def line_math_id(index: int) -> str:
    return f"line_math[{index}]"


def tax_line_math_id(index: int) -> str:
    return f"tax_line_math[{index}]"


def quote_id(path: str) -> str:
    """The id of the quote check for `path`, which is also the review reason of
    the field it flags (`REVIEW_REASON_QUOTE_PREFIX` in app/models.py)."""
    return f"quote:{path}"


def line_path(index: int, key: str) -> str:
    """The path of one line item cell, such as `line_items[2].total`."""
    return f"line_items[{index}].{key}"


def tax_line_path(index: int, key: str) -> str:
    """The path of one tax line cell, such as `tax_lines[1].amount`."""
    return f"tax_lines[{index}].{key}"


# ---- Input ----------------------------------------------------------------


@dataclass(frozen=True)
class FieldInput:
    """One field as the engine receives it: the raw value (anything JSON can
    hold, parsed and validated inside `reconcile`), where it came from, and
    whether a person has reviewed it. `has_evidence` is whether the leaf carried
    an `evidence` key (v2; null counts), and `quote` the verbatim printed text
    in it (None when there is none); a leaf with no `evidence` key (v1) is
    checked against nothing."""

    value: Any = None
    source: str = SOURCE_EXTRACTED
    verified: bool = False
    quote: str | None = None
    has_evidence: bool = False


@dataclass(frozen=True)
class LineInput:
    """One line item's raw cells. The description plays no part in the rules.
    `quotes` has a key for each cell whose leaf carried an evidence object, with
    the quote in it (None when the object has none)."""

    quantity: Any = None
    unit_price: Any = None
    total: Any = None
    quotes: Mapping[str, str | None] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class TaxLineInput:
    """One tax line's raw cells. The rate is the printed percent as a plain
    decimal (`"8.875"` for 8.875%); the label plays no part in the rules. The
    tax lines are one field in storage, so every row carries that field's source
    and verified flag. `quotes` is as on `LineInput`, for the rate, base, and
    amount."""

    label: Any = None
    rate: Any = None
    base: Any = None
    amount: Any = None
    source: str = SOURCE_EXTRACTED
    verified: bool = False
    quotes: Mapping[str, str | None] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class ReconcileInput:
    """Everything `reconcile` reads. `line_items` and `tax_lines` are None when
    the document has no usable list. The line items are one field in storage, so
    they share one source and one verified flag. A v2 field the document does
    not carry is an absent `FieldInput` (or None), never 0, and the dates are
    ISO `YYYY-MM-DD` strings."""

    currency: Any = None
    subtotal: FieldInput = FieldInput()
    tax: FieldInput = FieldInput()
    total: FieldInput = FieldInput()
    line_items: tuple[LineInput, ...] | None = None
    lines_source: str = SOURCE_EXTRACTED
    lines_verified: bool = False
    discount: FieldInput = FieldInput()
    shipping: FieldInput = FieldInput()
    tip: FieldInput = FieldInput()
    amount_paid: FieldInput = FieldInput()
    amount_due: FieldInput = FieldInput()
    document_date: FieldInput = FieldInput()
    due_date: FieldInput = FieldInput()
    tax_lines: tuple[TaxLineInput, ...] | None = None

    @classmethod
    def from_tool_input(cls, tool_input: Any) -> "ReconcileInput":
        """From the model's `record_extraction` input, a leaf at a time through
        `_coerce_leaf`: a malformed leaf is an absent value. Everything is
        extracted and unverified. Used by the evals."""
        tool = tool_input if isinstance(tool_input, dict) else {}

        def leaf(name: str) -> Any:
            return _coerce_leaf(tool.get(name))[0]

        def amount(name: str) -> FieldInput:
            has_evidence, quote = _leaf_evidence(tool.get(name))
            return FieldInput(leaf(name), quote=quote, has_evidence=has_evidence)

        rows = leaf("line_items")
        tax_rows = leaf("tax_lines")
        return cls(
            currency=leaf("currency"),
            subtotal=amount("subtotal"),
            tax=amount("tax"),
            total=amount("total"),
            line_items=(
                tuple(_line_from_leaves(row) for row in rows) if isinstance(rows, list) else None
            ),
            discount=amount("discount"),
            shipping=amount("shipping"),
            tip=amount("tip"),
            amount_paid=amount("amount_paid"),
            amount_due=amount("amount_due"),
            document_date=FieldInput(leaf("document_date")),
            due_date=FieldInput(leaf("due_date")),
            tax_lines=(
                tuple(_tax_line_from_leaves(row) for row in tax_rows)
                if isinstance(tax_rows, list)
                else None
            ),
        )

    @classmethod
    def from_fields(cls, fields: Iterable["ExtractedField"]) -> "ReconcileInput":
        """From one extraction's stored fields, with `records.field_value`
        deciding the value (a reviewer's correction wins). A field is verified
        when it has been reviewed; its source says how. The quotes are read from
        the stored leaves, which keep the evidence the model gave."""
        by_name = {field.field_name: field for field in fields}

        def read(name: str) -> FieldInput:
            field = by_name.get(name)
            if field is None:
                return FieldInput()
            has_evidence, quote = _leaf_evidence(field.value)
            return FieldInput(
                field_value(field),
                _stored_source(field),
                field.reviewed_at is not None,
                quote,
                has_evidence,
            )

        currency = by_name.get("currency")
        lines = by_name.get("line_items")
        rows = field_value(lines) if lines is not None else None
        line_quotes = _stored_cell_quotes(lines, _LINE_NUMBER_KEYS)
        tax_lines = by_name.get("tax_lines")
        tax_rows = _stored_tax_line_rows(tax_lines)
        tax_quotes = _stored_cell_quotes(tax_lines, _TAX_LINE_NUMBER_KEYS)
        tax_source = _stored_source(tax_lines) if tax_lines is not None else SOURCE_EXTRACTED
        tax_verified = tax_lines is not None and tax_lines.reviewed_at is not None
        return cls(
            currency=field_value(currency) if currency is not None else None,
            subtotal=read("subtotal"),
            tax=read("tax"),
            total=read("total"),
            line_items=(
                tuple(_line_from_plain(row, _quotes_at(line_quotes, i)) for i, row in enumerate(rows))
                if isinstance(rows, list)
                else None
            ),
            lines_source=_stored_source(lines) if lines is not None else SOURCE_EXTRACTED,
            lines_verified=lines is not None and lines.reviewed_at is not None,
            discount=read("discount"),
            shipping=read("shipping"),
            tip=read("tip"),
            amount_paid=read("amount_paid"),
            amount_due=read("amount_due"),
            document_date=read("document_date"),
            due_date=read("due_date"),
            tax_lines=(
                tuple(
                    _tax_line_from_plain(row, tax_source, tax_verified, _quotes_at(tax_quotes, i))
                    for i, row in enumerate(tax_rows)
                )
                if isinstance(tax_rows, list)
                else None
            ),
        )


def _leaf_evidence(raw: Any) -> tuple[bool, str | None]:
    """Whether a raw leaf carries evidence, and the quote in it. A leaf has
    evidence when it has an `evidence` key, so `evidence: null` (the model had
    nothing to cite) is a leaf without a quote, where a v1 leaf has no key at
    all. A quote that is not text counts as no quote, and the evidence is judged
    apart from the leaf's value, which `_coerce_leaf` reads."""
    if not isinstance(raw, dict) or "evidence" not in raw:
        return False, None
    evidence = raw["evidence"]
    quote = evidence.get("quote") if isinstance(evidence, dict) else None
    return True, quote if isinstance(quote, str) else None


def _cell_quotes(item: Any, keys: Sequence[str]) -> dict[str, str | None]:
    """The quote of each cell of a raw row whose leaf carries an evidence
    object."""
    if not isinstance(item, dict):
        return {}
    found = {key: _leaf_evidence(item.get(key)) for key in keys}
    return {key: quote for key, (has_evidence, quote) in found.items() if has_evidence}


def _stored_cell_quotes(
    field: "ExtractedField | None", keys: Sequence[str]
) -> list[dict[str, str | None]] | None:
    """`_cell_quotes` for each row of a stored array field, in row order. None
    when there is no stored array to read them from: a corrected field's rows
    are a person's, and have no quotes."""
    if field is None or field.review_action == "corrected" or not isinstance(field.value, list):
        return None
    return [_cell_quotes(item, keys) for item in field.value]


def _quotes_at(quotes: list[dict[str, str | None]] | None, index: int) -> dict[str, str | None]:
    return quotes[index] if quotes is not None and index < len(quotes) else {}


def _line_from_leaves(item: Any) -> LineInput:
    if not isinstance(item, dict):
        return LineInput()
    return LineInput(
        quantity=_coerce_leaf(item.get("quantity"))[0],
        unit_price=_coerce_leaf(item.get("unit_price"))[0],
        total=_coerce_leaf(item.get("total"))[0],
        quotes=_cell_quotes(item, _LINE_NUMBER_KEYS),
    )


def _line_from_plain(row: Any, quotes: Mapping[str, str | None]) -> LineInput:
    if not isinstance(row, dict):
        return LineInput()
    return LineInput(row.get("quantity"), row.get("unit_price"), row.get("total"), quotes)


def _tax_line_from_leaves(item: Any) -> TaxLineInput:
    if not isinstance(item, dict):
        return TaxLineInput()
    return TaxLineInput(
        label=_coerce_leaf(item.get("label"))[0],
        rate=_coerce_leaf(item.get("rate"))[0],
        base=_coerce_leaf(item.get("base"))[0],
        amount=_coerce_leaf(item.get("amount"))[0],
        quotes=_cell_quotes(item, _TAX_LINE_NUMBER_KEYS),
    )


def _tax_line_from_plain(
    row: Any, source: str, verified: bool, quotes: Mapping[str, str | None]
) -> TaxLineInput:
    if not isinstance(row, dict):
        return TaxLineInput(source=source, verified=verified)
    return TaxLineInput(
        row.get("label"),
        row.get("rate"),
        row.get("base"),
        row.get("amount"),
        source,
        verified,
        quotes,
    )


def _stored_tax_line_rows(field: "ExtractedField | None") -> Any:
    """The plain tax line rows of the stored `tax_lines` field: a reviewer's
    correction, else the stored array (a leaf per cell, stored the way
    `line_items` is). None when there is no field or no list."""
    if field is None:
        return None
    if field.review_action == "corrected":
        return field.corrected_value
    if not isinstance(field.value, list):
        return None
    return [
        {key: leaf_value(item.get(key)) for key in _TAX_LINE_KEYS} if isinstance(item, dict) else None
        for item in field.value
    ]


def _stored_source(field: "ExtractedField") -> str:
    if field.review_action == "corrected":
        return SOURCE_CORRECTED
    return SOURCE_APPROVED if field.reviewed_at is not None else SOURCE_EXTRACTED


# ---- Output ---------------------------------------------------------------


@dataclass(frozen=True)
class Operand:
    """A number (or, for the two dates, a day) a check used. `value` is None when
    the field is missing or unusable; `verified` is whether a person has reviewed where it came from
    (a computed or assumed operand is verified when its inputs are). A
    computed operand has no row of its own to blame: `inputs` are the paths it
    was computed from, and a check lists those in `fields` instead of `path`."""

    path: str
    value: Decimal | date | None
    source: str
    verified: bool
    inputs: tuple[str, ...] = ()


class Term(NamedTuple):
    """`coef * operand`. The coefficient is None only on a skipped check whose
    coefficient could not be read."""

    coef: Decimal | None
    operand: Operand


@dataclass(frozen=True)
class Check:
    """One rule's verdict.

    An equation reads `sum(coef * operand for terms) = rhs`: `expected` is the
    left side, `actual` the right, `delta = actual - expected`, and it passes
    when `abs(delta) <= tolerance`. The other kinds put the operands they
    examined in `terms` with coefficient 1, leave `rhs`, `expected`, `delta`,
    and `tolerance` empty, and for a bound put the measured number in `actual`.
    `fields` lists every path involved, quantity cells included; an operand
    that was computed (an unprinted subtotal or line total) contributes the paths
    it was computed from, not its own.
    """

    id: str
    kind: str
    formula: str
    terms: tuple[Term, ...]
    rhs: Operand | None
    expected: Decimal | None
    actual: Decimal | None
    delta: Decimal | None
    tolerance: Decimal | None
    status: str
    detail: str | None
    fields: tuple[str, ...]
    all_operands_verified: bool

    @property
    def operands(self) -> tuple[Operand, ...]:
        found = [term.operand for term in self.terms]
        if self.rhs is not None:
            found.append(self.rhs)
        return tuple(found)


@dataclass(frozen=True)
class Derived:
    """A value the document did not print, computed from ones it did."""

    path: str
    value: Decimal
    formula: str


@dataclass(frozen=True)
class Suggestion:
    """The one change that fixes every failing check without breaking a passing
    one. `current` is what the field holds now."""

    path: str
    value: Decimal
    current: Decimal | None
    check_ids: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class Report:
    rules_version: str
    status: str
    currency: str | None
    minor_units: int
    assumptions: tuple[str, ...]
    derived: tuple[Derived, ...]
    checks: tuple[Check, ...]
    suggestion: Suggestion | None

    @property
    def failing_checks(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if check.status == STATUS_FAIL)


# ---- Arithmetic -----------------------------------------------------------

# Rounding and division are the only inexact operations, and they are explicit:
# the module's own context traps Inexact, these do not.
_ROUNDING = Context(prec=_PRECISION, rounding=ROUND_HALF_UP, traps=[InvalidOperation])
_DIVISION = Context(
    prec=_PRECISION, rounding=ROUND_HALF_EVEN, traps=[InvalidOperation, DivisionByZero]
)


@contextmanager
def _exact() -> Iterator[None]:
    with localcontext() as context:
        context.prec = _PRECISION
        context.traps[Inexact] = True
        yield


def _round_to(value: Decimal, places: int) -> Decimal:
    return value.quantize(_ONE.scaleb(-places), context=_ROUNDING)


def _places_phrase(places: int) -> str:
    if places == 0:
        return "no decimal places"
    return f"at most {places} decimal place{'s' if places != 1 else ''}"


# ---- Reading the input ----------------------------------------------------


@dataclass
class _Line:
    quantity: Operand
    unit_price: Operand
    total: Operand
    # The product exactly, when both factors are readable.
    product: Decimal | None = None

    def needs_rounding(self, minor_units: int) -> bool:
        """True when the product has digits beyond the minor unit, so a printed
        total can only be a rounding of it."""
        return self.product is not None and not fits_minor_unit(self.product, minor_units)


@dataclass
class _TaxLine:
    index: int
    rate: Operand
    base: Operand
    amount: Operand


@dataclass(frozen=True)
class _Quoted:
    """A numeric leaf that carried an evidence object: its operand (value None
    when absent or unusable), the quote in the object (None when it has none),
    and the decimal places the leaf allows, which settle how its quote's
    ambiguous numbers are read."""

    operand: Operand
    quote: str | None
    places: int


@dataclass
class _Parsed:
    """The input after parsing: operands with Decimal values (None when absent
    or unusable) plus the format failures found on the way. The two dates are
    operands with `date` values; one that is printed but not a date is in
    `undated` (and has no format check). `quoted` lists, in check order, the
    leaves that carried evidence."""

    minor_units: int
    subtotal: Operand
    tax: Operand
    total: Operand
    discount: Operand
    shipping: Operand
    tip: Operand
    amount_paid: Operand
    amount_due: Operand
    document_date: Operand
    due_date: Operand
    lines: list[_Line]
    lines_listed: bool
    lines_source: str
    lines_verified: bool
    tax_lines: list[_TaxLine]
    quoted: list[_Quoted]
    format_checks: list[Check]
    unreadable: set[str]
    undated: set[str]

    def money(self, value: Decimal) -> str:
        return to_api_string(value, self.minor_units)

    def present(self, operand: Operand) -> bool:
        """True when the document prints this field, usable or not: an absent
        field adds no term and no check, an unreadable one skips what needs it."""
        return operand.value is not None or operand.path in self.unreadable

    def skip_reason(self, operands: Sequence[Operand]) -> str | None:
        """Why a check over these operands cannot run, or None when it can."""
        reasons = [
            f"{op.path} is unreadable (see format:{op.path})"
            if op.path in self.unreadable
            else f"{op.path} is missing"
            for op in operands
            if op.value is None
        ]
        return f"Skipped because {'; '.join(reasons)}." if reasons else None


def _read(
    path: str,
    raw: Any,
    source: str,
    verified: bool,
    places: int,
    format_checks: list[Check],
    unreadable: set[str],
) -> Operand:
    """Parse one number. A usable value becomes an operand; null is an absent
    operand; anything else is an absent operand plus a failing `format:` check."""
    if raw is None:
        return Operand(path, None, source, verified)
    parsed: Decimal | None = None
    try:
        parsed = to_decimal(raw)
    except MoneyError as exc:
        problem = f"{path}: {exc}"
    else:
        if abs(parsed) >= MAX_AMOUNT:
            problem = f"{path} {shown_value(raw)} is too large; amounts must be below {MAX_AMOUNT:,}."
        elif not fits_minor_unit(parsed, places):
            problem = (
                f"{path} {shown_value(raw)} has too many decimal places; "
                f"it allows {_places_phrase(places)}."
            )
        else:
            return Operand(path, parsed, source, verified)
    shown = Operand(path, parsed, source, verified)
    format_checks.append(
        Check(
            id=f"format:{path}",
            kind=KIND_FORMAT,
            formula=f"{path} is a finite decimal number below {MAX_AMOUNT:,} with {_places_phrase(places)}",
            terms=(Term(_ONE, shown),),
            rhs=None,
            expected=None,
            actual=parsed,
            delta=None,
            tolerance=None,
            status=STATUS_FAIL,
            detail=problem,
            fields=(path,),
            all_operands_verified=verified,
        )
    )
    unreadable.add(path)
    return Operand(path, None, source, verified)


def _read_date(path: str, field: FieldInput, undated: set[str]) -> Operand:
    """Parse one `YYYY-MM-DD` date. Null is an absent operand; anything that is
    not a calendar date is an absent operand and noted in `undated`."""
    raw = field.value
    if raw is not None:
        text = raw.strip() if isinstance(raw, str) else ""
        if _ISO_DATE_RE.fullmatch(text):
            try:
                return Operand(path, date.fromisoformat(text), field.source, field.verified)
            except ValueError:
                pass
        undated.add(path)
    return Operand(path, None, field.source, field.verified)


def _parse(inp: ReconcileInput, minor_units: int) -> _Parsed:
    format_checks: list[Check] = []
    unreadable: set[str] = set()
    undated: set[str] = set()
    quoted: list[_Quoted] = []

    def header(name: str) -> Operand:
        field: FieldInput = getattr(inp, name)
        operand = _read(
            name, field.value, field.source, field.verified, minor_units, format_checks, unreadable
        )
        if field.has_evidence:
            quoted.append(_Quoted(operand, field.quote, minor_units))
        return operand

    subtotal, tax, total = header("subtotal"), header("tax"), header("total")
    discount, shipping, tip = header("discount"), header("shipping"), header("tip")
    amount_paid, amount_due = header("amount_paid"), header("amount_due")
    lines = []
    for index, line in enumerate(inp.line_items or ()):
        cells = {}
        for key, raw, places in (
            ("quantity", line.quantity, MAX_QUANTITY_DECIMALS),
            ("unit_price", line.unit_price, MAX_QUANTITY_DECIMALS),
            ("total", line.total, minor_units),
        ):
            cells[key] = _read(
                line_path(index, key),
                raw,
                inp.lines_source,
                inp.lines_verified,
                places,
                format_checks,
                unreadable,
            )
            if key in line.quotes:
                quoted.append(_Quoted(cells[key], line.quotes[key], places))
        lines.append(_Line(cells["quantity"], cells["unit_price"], cells["total"]))
    tax_lines = []
    for index, row in enumerate(inp.tax_lines or ()):
        cells = {}
        for key, raw, places in (
            ("rate", row.rate, MAX_QUANTITY_DECIMALS),
            ("base", row.base, minor_units),
            ("amount", row.amount, minor_units),
        ):
            cells[key] = _read(
                tax_line_path(index, key),
                raw,
                row.source,
                row.verified,
                places,
                format_checks,
                unreadable,
            )
            if key in row.quotes:
                quoted.append(_Quoted(cells[key], row.quotes[key], places))
        tax_lines.append(_TaxLine(index, cells["rate"], cells["base"], cells["amount"]))
    return _Parsed(
        minor_units=minor_units,
        subtotal=subtotal,
        tax=tax,
        total=total,
        discount=discount,
        shipping=shipping,
        tip=tip,
        amount_paid=amount_paid,
        amount_due=amount_due,
        document_date=_read_date("document_date", inp.document_date, undated),
        due_date=_read_date("due_date", inp.due_date, undated),
        lines=lines,
        lines_listed=bool(inp.line_items),
        lines_source=inp.lines_source,
        lines_verified=inp.lines_verified,
        tax_lines=tax_lines,
        quoted=quoted,
        format_checks=format_checks,
        unreadable=unreadable,
        undated=undated,
    )


# ---- Checks ---------------------------------------------------------------


def _provenance(operands: Sequence[Operand]) -> tuple[tuple[str, ...], bool]:
    """The paths a check involves and whether a person verified them all. A
    computed operand (an unprinted subtotal or line total) stands for the
    values it was computed from: the field that is absent never made a
    contribution to the equation, so it is not named."""
    paths = tuple(dict.fromkeys(path for op in operands for path in _stands_for(op)))
    return paths, all(op.verified for op in operands)


def _stands_for(operand: Operand) -> tuple[str, ...]:
    """The paths an operand stands for: its own, or for a computed one the
    paths it was computed from."""
    if operand.source == SOURCE_COMPUTED and operand.inputs:
        return operand.inputs
    return (operand.path,)


def _equation(
    p: _Parsed,
    check_id: str,
    formula: str,
    terms: Sequence[Term],
    rhs: Operand,
    tolerance: Decimal,
    explain: Callable[[Decimal, Decimal, Decimal], str],
    *,
    also: Sequence[Operand] = (),
    skip: str | None = None,
) -> Check:
    """One linear equation. `also` are operands that matter to the check but
    are not terms (the quantity, which is the coefficient); `skip` forces a
    skipped status with its own reason."""
    operands = [*also, *(term.operand for term in terms), rhs]
    fields, verified = _provenance(operands)
    reason = skip or p.skip_reason(operands)
    expected = actual = delta = None
    status, detail = STATUS_SKIPPED, reason
    if reason is None:
        expected = sum((term.coef * term.operand.value for term in terms), Decimal(0))
        actual = rhs.value
        delta = actual - expected
        if abs(delta) <= tolerance:
            status, detail = STATUS_PASS, None
        else:
            status, detail = STATUS_FAIL, explain(expected, actual, delta)
    return Check(
        id=check_id,
        kind=KIND_EQUATION,
        formula=formula,
        terms=tuple(terms),
        rhs=rhs,
        expected=expected,
        actual=actual,
        delta=delta,
        tolerance=tolerance,
        status=status,
        detail=detail,
        fields=fields,
        all_operands_verified=verified,
    )


def _judgement(
    check_id: str,
    kind: str,
    formula: str,
    operands: Sequence[Operand],
    actual: Decimal | None,
    holds: bool | None,
    detail: str | None,
) -> Check:
    """A bound or presence check. `holds` is None when it could not be
    evaluated (then `detail` is the reason)."""
    fields, verified = _provenance(operands)
    if holds is None:
        status = STATUS_SKIPPED
    elif holds:
        status, detail = STATUS_PASS, None
    else:
        status = STATUS_FAIL
    return Check(
        id=check_id,
        kind=kind,
        formula=formula,
        terms=tuple(Term(_ONE, op) for op in operands),
        rhs=None,
        expected=None,
        actual=actual,
        delta=None,
        tolerance=None,
        status=status,
        detail=detail,
        fields=fields,
        all_operands_verified=verified,
    )


def _quote_checks(p: _Parsed) -> list[Check]:
    """quote:<path> for each leaf that carried an evidence object. Nothing is
    emitted for a value that is absent or unreadable (the format check already
    says so) or that a person has reviewed: their word replaces the model's
    quote. A quote that is missing is a skipped check; everything else is
    judged by `quote_agreement`."""
    checks = []
    for entry in p.quoted:
        operand = entry.operand
        if operand.value is None or operand.verified:
            continue
        # Amounts print to the currency's places; a quantity or a percent as is.
        plain = operand.path.endswith((".quantity", ".rate"))
        verdict = quote_agreement(
            operand.value,
            entry.quote,
            places=entry.places,
            minor_units=None if plain else p.minor_units,
        )
        holds = None if is_skipped(verdict) else verdict.status == QUOTE_PASS
        checks.append(
            _judgement(
                quote_id(operand.path),
                KIND_QUOTE,
                f"{operand.path} equals the number in its quote",
                [operand],
                None,
                holds,
                verdict.detail,
            )
        )
    return checks


def _line_checks(p: _Parsed, derived: list[Derived]) -> list[Check]:
    """line_math for each line. Fills in `_Line.product` and, for a line whose
    total is absent but whose factors are readable, a computed total."""
    unit = _ONE.scaleb(-p.minor_units)
    checks = []
    for index, line in enumerate(p.lines):
        q, u, t = line.quantity, line.unit_price, line.total
        if q.value is not None and u.value is not None:
            line.product = q.value * u.value
        skip = None
        if t.value is None and t.path not in p.unreadable and line.product is not None:
            computed = _round_to(line.product, p.minor_units)
            line.total = t = Operand(
                t.path, computed, SOURCE_COMPUTED, p.lines_verified, (q.path, u.path)
            )
            derived.append(Derived(t.path, computed, "quantity x unit_price"))
            skip = f"{t.path} is not printed; it is computed as quantity x unit_price."
        checks.append(
            _equation(
                p,
                line_math_id(index),
                f"line {index + 1}: quantity x unit_price = total",
                [Term(q.value, u)],
                t,
                unit if line.needs_rounding(p.minor_units) else Decimal(0),
                _explain_line(p, index, q, u),
                also=[q],
                skip=skip,
            )
        )
    return checks


def _explain_line(
    p: _Parsed, index: int, q: Operand, u: Operand
) -> Callable[[Decimal, Decimal, Decimal], str]:
    def explain(expected: Decimal, actual: Decimal, delta: Decimal) -> str:
        return (
            f"Line {index + 1}: {to_plain_string(q.value)} x {p.money(u.value)} is "
            f"{p.money(expected)}, but the line total is {p.money(actual)} "
            f"(off by {p.money(abs(delta))})."
        )

    return explain


def _lines_sum_check(p: _Parsed, derived: list[Derived]) -> tuple[Check, Operand]:
    """lines_sum, and the subtotal operand the rest of the rules use: the
    printed one, or the sum of the line totals when none was printed."""
    totals = [line.total for line in p.lines]
    terms = [Term(_ONE, total) for total in totals]
    formula = "sum of line totals = subtotal"
    half_unit = _ONE.scaleb(-p.minor_units) / 2
    tolerance = half_unit * sum(line.needs_rounding(p.minor_units) for line in p.lines)

    def explain(expected: Decimal, actual: Decimal, delta: Decimal) -> str:
        return (
            f"The line totals add up to {p.money(expected)}, but the subtotal is "
            f"{p.money(actual)} (off by {p.money(abs(delta))})."
        )

    subtotal = p.subtotal
    if not totals:
        return (
            _equation(
                p, LINES_SUM, formula, terms, subtotal, tolerance, explain,
                skip="Skipped because there are no line items.",
            ),
            subtotal,
        )
    if subtotal.value is None and subtotal.path not in p.unreadable:
        missing = p.skip_reason(totals)
        if missing is None:
            total = sum((t.value for t in totals), Decimal(0))
            subtotal = Operand(
                "subtotal", total, SOURCE_COMPUTED, p.lines_verified, tuple(t.path for t in totals)
            )
            derived.append(Derived("subtotal", total, "sum of line totals"))
            skip = "The subtotal is not printed; it is computed as the sum of the line totals."
            return (
                _equation(p, LINES_SUM, formula, terms, subtotal, tolerance, explain, skip=skip),
                subtotal,
            )
    return _equation(p, LINES_SUM, formula, terms, subtotal, tolerance, explain), subtotal


def _sum_text(parts: Sequence[tuple[str, Decimal]]) -> str:
    """A signed sum in words, such as "subtotal - discount + tax". A part with
    coefficient 0 (already inside the others) is left out."""
    text = ""
    for name, coef in parts:
        if coef == 0:
            continue
        text = name if not text else f"{text} - {name}" if coef < 0 else f"{text} + {name}"
    return text


def _total_terms(
    p: _Parsed,
    subtotal: Operand,
    tax: Operand,
    *,
    tax_coef: Decimal = _ONE,
    discount_included: bool = False,
) -> list[Term]:
    """The left side of total_math: subtotal - discount + shipping + tax + tip,
    leaving out an adjustment the document does not print. `tax_coef` 0 reads the
    tax as already inside the subtotal; `discount_included` reads the discount
    as already taken off the subtotal."""
    terms = [Term(_ONE, subtotal)]
    if p.present(p.discount):
        if discount_included:
            terms.append(
                Term(Decimal(0), Operand("discount", p.discount.value, SOURCE_INCLUDED, p.discount.verified))
            )
        else:
            terms.append(Term(-_ONE, p.discount))
    if p.present(p.shipping):
        terms.append(Term(_ONE, p.shipping))
    terms.append(Term(tax_coef, tax))
    if p.present(p.tip):
        terms.append(Term(_ONE, p.tip))
    return terms


def _total_formula(terms: Sequence[Term], note: str = "") -> str:
    return f"{_sum_text([(term.operand.path, term.coef) for term in terms])} = total{note}"


def _total_math_check(
    p: _Parsed, subtotal: Operand, *, discount_applied: bool = False
) -> tuple[Check, bool]:
    """total_math, and whether it counted a null tax as 0. `discount_applied`
    is the reading where the printed subtotal is already after the discount."""
    tax = p.tax
    # A null tax is only assumed to be 0 when there is an equation to run.
    adjustments = [op for op in (p.discount, p.shipping, p.tip) if p.present(op)]
    unknown = p.skip_reason([subtotal, p.total, *adjustments])
    assumed = unknown is None and tax.value is None and tax.path not in p.unreadable
    if assumed:
        tax = Operand("tax", Decimal(0), SOURCE_ASSUMED, p.tax.verified)
    terms = _total_terms(p, subtotal, tax, discount_included=discount_applied)

    def explain(expected: Decimal, actual: Decimal, delta: Decimal) -> str:
        counted = " (no tax is printed, so it counts as 0)" if assumed else ""
        sentence = f"The subtotal {p.money(subtotal.value)}"
        for term in terms[1:]:
            verb = "minus" if term.coef < 0 else "plus"
            sentence += f" {verb} {term.operand.path} {p.money(term.operand.value)}"
            sentence += counted if term.operand is tax else ""
        return (
            f"{sentence} is {p.money(expected)}, but the total is {p.money(actual)} "
            f"(off by {p.money(abs(delta))})."
        )

    note = " (the subtotal is after the discount)" if discount_applied else ""
    check = _equation(
        p,
        TOTAL_MATH,
        _total_formula(terms, note),
        terms,
        p.total,
        Decimal(0),
        explain,
        skip=unknown,
    )
    return check, assumed and check.status != STATUS_SKIPPED


def _rate(tax: Decimal, base: Decimal) -> Decimal:
    """tax / base to the reported places. The verdict uses this same number."""
    return _round_to(_DIVISION.divide(tax, base), RATE_PLACES)


def _plausible_rate(tax: Decimal, base: Decimal) -> bool:
    return base > 0 and _rate(tax, base) <= MAX_TAX_RATE


def _gross_subtotal_check(p: _Parsed, subtotal: Operand) -> Check | None:
    """total_math for a document whose subtotal already includes the tax: a
    printed tax, a subtotal that reaches the total without it, and a plausible
    rate on the net amount. None when the numbers do not fit that reading
    exactly."""
    tax, total = p.tax.value, p.total.value
    if tax is None or total is None or subtotal.value is None:
        return None
    if tax <= 0 or not _plausible_rate(tax, total - tax):
        return None
    included = Operand("tax", tax, SOURCE_INCLUDED, p.tax.verified)
    terms = _total_terms(p, subtotal, included, tax_coef=Decimal(0))
    check = _equation(
        p,
        TOTAL_MATH,
        _total_formula(terms, " (the total includes the tax)"),
        terms,
        p.total,
        Decimal(0),
        lambda expected, actual, delta: "",
    )
    return check if check.status == STATUS_PASS else None


def _gross_lines_check(p: _Parsed, lines_sum: Check) -> Check | None:
    """lines_sum for a document whose line prices include the tax: the lines
    (less the discount, plus shipping and tip) add up to the total (within the
    usual tolerance) while subtotal + tax = total and the rate on the subtotal
    is plausible. The caller has checked that the ordinary lines_sum fails and
    total_math passes."""
    tax, subtotal = p.tax.value, p.subtotal.value
    if tax is None or subtotal is None or tax <= 0 or not _plausible_rate(tax, subtotal):
        return None
    included = Operand("tax", tax, SOURCE_INCLUDED, p.tax.verified)
    adjustments = [
        Term(coef, op)
        for coef, op in ((-_ONE, p.discount), (_ONE, p.shipping), (_ONE, p.tip))
        if p.present(op)
    ]
    terms = [*lines_sum.terms, *adjustments, Term(Decimal(0), included)]
    parts = [("sum of line totals", _ONE), *((t.operand.path, t.coef) for t in adjustments)]
    check = _equation(
        p,
        LINES_SUM,
        f"{_sum_text(parts)} = total (the line prices include the tax)",
        terms,
        p.total,
        lines_sum.tolerance,
        lambda expected, actual, delta: "",
    )
    return check if check.status == STATUS_PASS else None


def _discount_applied_checks(
    p: _Parsed, lines_sum: Check, subtotal: Operand
) -> tuple[Check, Check] | None:
    """lines_sum and total_math for a document whose printed subtotal is already
    after the discount: the line totals less the discount equal the subtotal,
    and the total is the subtotal plus shipping, tax, and tip. None unless both
    hold exactly. The caller has checked that the ordinary lines_sum fails."""
    discount = p.discount
    if discount.value is None or discount.value <= 0:
        return None
    parts = [("sum of line totals", _ONE), ("discount", -_ONE)]
    lines = _equation(
        p,
        LINES_SUM,
        f"{_sum_text(parts)} = subtotal (the subtotal is after the discount)",
        [*lines_sum.terms, Term(-_ONE, discount)],
        subtotal,
        lines_sum.tolerance,
        lambda expected, actual, delta: "",
    )
    total, _ = _total_math_check(p, subtotal, discount_applied=True)
    if lines.status != STATUS_PASS or total.status != STATUS_PASS:
        return None
    return lines, total


def _tax_rate_check(p: _Parsed, subtotal: Operand, *, net_base: bool) -> Check:
    """tax_rate on the subtotal, or on total - tax when the tax is included."""
    if net_base:
        formula = f"0 <= tax / (total - tax) <= {MAX_TAX_RATE}"
        operands, base_name, of_what = [p.tax, p.total], "total - tax", "the net amount (total - tax)"
    else:
        formula = f"0 <= tax / subtotal <= {MAX_TAX_RATE}"
        operands, base_name, of_what = [p.tax, subtotal], "subtotal", "the subtotal"
    reason = p.skip_reason(operands)
    base = None
    if reason is None:
        base = p.total.value - p.tax.value if net_base else subtotal.value
        if base == 0:
            reason = f"Skipped because {base_name} is 0."
    if reason is not None:
        return _judgement(TAX_RATE, KIND_BOUND, formula, operands, None, None, reason)
    rate = _rate(p.tax.value, base)
    detail = (
        f"The tax is {to_api_string(rate, RATE_PLACES)} of {of_what}; "
        f"a rate between 0 and {MAX_TAX_RATE} is expected."
    )
    return _judgement(
        TAX_RATE, KIND_BOUND, formula, operands, rate, 0 <= rate <= MAX_TAX_RATE, detail
    )


def _non_negative_checks(p: _Parsed, subtotal: Operand) -> list[Check]:
    checks = []
    extras = [getattr(p, name) for name in _NON_NEGATIVE_EXTRAS if p.present(getattr(p, name))]
    for operand in (subtotal, p.tax, p.total, *extras):
        check_id, formula = f"non_negative:{operand.path}", f"{operand.path} >= 0"
        reason = p.skip_reason([operand])
        if reason is not None:
            checks.append(_judgement(check_id, KIND_BOUND, formula, [operand], None, None, reason))
            continue
        detail = f"The {operand.path} is {p.money(operand.value)}, which is negative."
        checks.append(
            _judgement(check_id, KIND_BOUND, formula, [operand], operand.value, operand.value >= 0, detail)
        )
    return checks


def _tax_lines_sum_check(p: _Parsed, derived: list[Derived]) -> Check | None:
    """tax_lines_sum, or None when the document has no tax lines. A tax that is
    not printed becomes the sum of the tax line amounts (`p.tax` is replaced, so
    every later rule uses it) and the check is skipped, as an unprinted subtotal
    is."""
    if not p.tax_lines:
        return None
    amounts = [line.amount for line in p.tax_lines]
    tax = p.tax
    skip = None
    if tax.value is None and tax.path not in p.unreadable and p.skip_reason(amounts) is None:
        total = sum((amount.value for amount in amounts), Decimal(0))
        p.tax = tax = Operand(
            "tax",
            total,
            SOURCE_COMPUTED,
            all(amount.verified for amount in amounts),
            tuple(amount.path for amount in amounts),
        )
        derived.append(Derived("tax", total, "sum of tax line amounts"))
        skip = "The tax is not printed; it is computed as the sum of the tax line amounts."

    def explain(expected: Decimal, actual: Decimal, delta: Decimal) -> str:
        return (
            f"The tax line amounts add up to {p.money(expected)}, but the tax is "
            f"{p.money(actual)} (off by {p.money(abs(delta))})."
        )

    return _equation(
        p,
        TAX_LINES_SUM,
        "sum of tax line amounts = tax",
        [Term(_ONE, amount) for amount in amounts],
        tax,
        Decimal(0),
        explain,
        skip=skip,
    )


def _computed_base(path: str, parts: Sequence[tuple[Decimal, Operand]]) -> Operand:
    """A tax line base worked out from other amounts: the signed sum of `parts`,
    absent when any part is."""
    values = [op.value for _, op in parts]
    value = None if any(v is None for v in values) else sum(
        (coef * v for (coef, _), v in zip(parts, values, strict=True)), Decimal(0)
    )
    inputs = tuple(dict.fromkeys(path for _, op in parts for path in _stands_for(op)))
    return Operand(path, value, SOURCE_COMPUTED, all(op.verified for _, op in parts), inputs)


def _fallback_bases(
    p: _Parsed, line: _TaxLine, subtotal: Operand, discount_applied: bool, net_base: bool
) -> list[tuple[Operand, str, tuple[Operand, ...]]]:
    """The bases to try for a tax line that prints none, in order: the subtotal
    less the discount, then that plus shipping. Each is (base, how it reads in
    words, the operands it was worked out from). A discount the subtotal is
    already after is not taken off again, a subtotal that already contains the
    tax (`net_base`) gives up the tax first, and an adjustment the document does
    not print is left out (so the first base can be the subtotal itself)."""
    path = tax_line_path(line.index, "base")
    parts = [(_ONE, subtotal)]
    phrase = "the subtotal"
    if net_base:
        parts.append((-_ONE, p.tax))
        phrase = "the net amount (subtotal less the tax)"
    if p.present(p.discount) and not discount_applied:
        parts.append((-_ONE, p.discount))
        phrase += " less the discount"
    candidates = [(parts, phrase)]
    if p.present(p.shipping):
        candidates.append(([*parts, (_ONE, p.shipping)], f"{phrase} plus shipping"))
    return [
        (
            subtotal if len(chosen) == 1 else _computed_base(path, chosen),
            words,
            tuple(op for _, op in chosen),
        )
        for chosen, words in candidates
    ]


def _explain_tax_line(
    p: _Parsed, number: int, rate: Operand, base: Operand, phrase: str
) -> Callable[[Decimal, Decimal, Decimal], str]:
    def explain(expected: Decimal, actual: Decimal, delta: Decimal) -> str:
        return (
            f"Tax line {number}: {to_plain_string(rate.value)}% of {phrase} {p.money(base.value)} "
            f"is {p.money(expected)}, but the amount is {p.money(actual)} "
            f"(off by {p.money(abs(delta))})."
        )

    return explain


def _tax_line_check(
    p: _Parsed, line: _TaxLine, base: Operand, phrase: str, sources: Sequence[Operand]
) -> Check:
    """tax_line_math for one tax line against one candidate base. `sources` are
    the operands a worked-out base comes from, named when it cannot be."""
    rate, amount = line.rate, line.amount
    coef = None if rate.value is None else rate.value.scaleb(-2)
    product = None if coef is None or base.value is None else coef * base.value
    rounded = product is not None and not fits_minor_unit(product, p.minor_units)
    skip = p.skip_reason(sources) if base.value is None and sources else None
    return _equation(
        p,
        tax_line_math_id(line.index),
        f"tax line {line.index + 1}: rate x base = amount",
        [Term(coef, base)],
        amount,
        _ONE.scaleb(-p.minor_units) if rounded else Decimal(0),
        _explain_tax_line(p, line.index + 1, rate, base, phrase),
        also=[rate],
        skip=skip,
    )


def _tax_line_checks(
    p: _Parsed,
    subtotal: Operand,
    *,
    discount_applied: bool,
    net_base: bool,
    first_base_only: bool,
) -> tuple[list[Check], list[str]]:
    """tax_line_math for each tax line, and the assumptions it made. A printed
    base is used as printed. Without one, each fallback base is tried in order
    and the first that passes is used; when none does, the first is the one
    reported. With `first_base_only` only the first is tried."""
    checks, assumptions = [], []
    for line in p.tax_lines:
        if p.present(line.base):
            checks.append(_tax_line_check(p, line, line.base, "the base", ()))
            continue
        chosen: tuple[Check, str] | None = None
        bases = _fallback_bases(p, line, subtotal, discount_applied, net_base)
        for base, phrase, sources in bases[:1] if first_base_only else bases:
            check = _tax_line_check(p, line, base, phrase, sources)
            chosen = chosen or (check, phrase)
            if check.status == STATUS_PASS:
                chosen = (check, phrase)
                break
        check, phrase = chosen
        checks.append(check)
        if check.status != STATUS_SKIPPED:
            assumptions.append(
                f"tax line {line.index + 1} shows no base amount, so its amount is checked "
                f"against {phrase}."
            )
    return checks, assumptions


def _balance_check(p: _Parsed, derived: list[Derived]) -> tuple[Check | None, bool]:
    """balance_math, or None when the document prints neither the amount paid
    nor the amount due. Returns whether it counted a missing amount paid as 0."""
    paid, due = p.amount_paid, p.amount_due
    if not p.present(paid) and not p.present(due):
        return None, False
    # A missing payment is only assumed to be 0 when there is an equation to run
    # (the amount due is printed, or the check would not be here).
    unknown = None if p.present(paid) else p.skip_reason([p.total, due])
    assumed = not p.present(paid) and unknown is None
    if assumed:
        paid = Operand("amount_paid", Decimal(0), SOURCE_ASSUMED, p.amount_paid.verified)
    skip = None
    if due.value is None and due.path not in p.unreadable and p.skip_reason([p.total, paid]) is None:
        balance = p.total.value - paid.value
        due = Operand(
            "amount_due",
            balance,
            SOURCE_COMPUTED,
            p.total.verified and paid.verified,
            (*_stands_for(p.total), *_stands_for(paid)),
        )
        derived.append(Derived("amount_due", balance, "total - amount paid"))
        skip = "The amount due is not printed; it is computed as total - amount paid."

    def explain(expected: Decimal, actual: Decimal, delta: Decimal) -> str:
        counted = " (no payment is printed, so it counts as 0)" if assumed else ""
        return (
            f"The total {p.money(p.total.value)} less the amount paid {p.money(paid.value)}{counted} "
            f"is {p.money(expected)}, but the amount due is {p.money(actual)} "
            f"(off by {p.money(abs(delta))})."
        )

    check = _equation(
        p,
        BALANCE_MATH,
        "total - amount_paid = amount_due",
        [Term(_ONE, p.total), Term(-_ONE, paid)],
        due,
        Decimal(0),
        explain,
        skip=skip or unknown,
    )
    return check, assumed and check.status != STATUS_SKIPPED


def _due_date_checks(p: _Parsed) -> list[Check]:
    """due_date_order, only when the document prints a due date."""
    due, issued = p.due_date, p.document_date
    if due.value is None and due.path not in p.undated:
        return []
    formula = "due_date >= document_date"
    reasons = [
        f"{op.path} is not a date (use YYYY-MM-DD)"
        if op.path in p.undated
        else f"{op.path} is missing"
        for op in (issued, due)
        if op.value is None
    ]
    if reasons:
        return [
            _judgement(
                DUE_DATE_ORDER, KIND_BOUND, formula, [issued, due], None, None,
                f"Skipped because {'; '.join(reasons)}.",
            )
        ]
    detail = f"The due date {due.value.isoformat()} is before the document date {issued.value.isoformat()}."
    return [
        _judgement(
            DUE_DATE_ORDER, KIND_BOUND, formula, [issued, due], None, due.value >= issued.value, detail
        )
    ]


def _lines_present_check(p: _Parsed) -> Check:
    lines = Operand("line_items", None, p.lines_source, p.lines_verified)
    operands = [p.subtotal, lines]
    formula = "line items are present when a subtotal is printed"
    if p.subtotal.value is None:
        reason = p.skip_reason([p.subtotal])
        return _judgement(LINES_PRESENT, KIND_PRESENCE, formula, operands, None, None, reason)
    return _judgement(
        LINES_PRESENT,
        KIND_PRESENCE,
        formula,
        operands,
        None,
        p.lines_listed,
        "A subtotal is printed but there are no line items to add up "
        "(line_items is empty, null, or not a list).",
    )


# ---- reconcile ------------------------------------------------------------


def _run(
    inp: ReconcileInput, *, allow_included: bool = True, allow_fallbacks: bool = True
) -> Report:
    """Every rule, without the suggestion (which re-runs this). With
    `allow_included` False the included readings (tax already in the subtotal or
    the lines, discount already out of the subtotal) are not tried; with
    `allow_fallbacks` False a tax line without a base is only checked against the
    first fallback base, not the one that adds shipping."""
    currency = inp.currency if isinstance(inp.currency, str) else None
    precision: CurrencyPrecision = currency_precision(currency)
    assumptions = [precision.assumption] if precision.assumption else []
    with _exact():
        p = _parse(inp, precision.minor_units)
        derived: list[Derived] = []
        line_checks = _line_checks(p, derived)
        lines_sum, subtotal = _lines_sum_check(p, derived)
        tax_lines_sum = _tax_lines_sum_check(p, derived)
        total_math, tax_assumed = _total_math_check(p, subtotal)
        if tax_assumed:
            assumptions.append("no tax is shown, so subtotal + tax = total counts it as 0.")
        net_base = False
        gross_subtotal = gross_lines = discount_applied = None
        if allow_included and lines_sum.status == STATUS_FAIL and p.present(p.discount):
            discount_applied = _discount_applied_checks(p, lines_sum, subtotal)
        if discount_applied is not None:
            lines_sum, total_math = discount_applied
            assumptions.append(
                "the subtotal is after the discount: the line totals less the discount make up "
                "the subtotal, and the total is built from that subtotal without taking the "
                "discount off again."
            )
        elif allow_included and total_math.status == STATUS_FAIL:
            gross_subtotal = _gross_subtotal_check(p, subtotal)
        elif allow_included and lines_sum.status == STATUS_FAIL and total_math.status == STATUS_PASS:
            gross_lines = _gross_lines_check(p, lines_sum)
        if gross_subtotal is not None:
            total_math, net_base = gross_subtotal, True
            assumptions.append(
                "prices include tax: subtotal + tax = total reads the subtotal as already "
                f"containing the tax {to_api_string(p.tax.value, p.minor_units)}, and the tax "
                "rate is judged on total - tax."
            )
        if gross_lines is not None:
            lines_sum = gross_lines
            assumptions.append(
                "line prices include tax: the sum of line totals is compared with the "
                "total instead of the subtotal."
            )
        tax_line_checks, base_assumptions = _tax_line_checks(
            p,
            subtotal,
            discount_applied=discount_applied is not None,
            net_base=net_base,
            first_base_only=not allow_fallbacks,
        )
        balance, paid_assumed = _balance_check(p, derived)
        if paid_assumed:
            assumptions.append(
                "no payment is shown, so total - amount paid = amount due counts it as 0."
            )
        assumptions.extend(base_assumptions)
        checks = [
            *p.format_checks,
            *_quote_checks(p),
            _lines_present_check(p),
            *line_checks,
            lines_sum,
            *([tax_lines_sum] if tax_lines_sum is not None else []),
            *tax_line_checks,
            total_math,
            _tax_rate_check(p, subtotal, net_base=net_base),
            *([balance] if balance is not None else []),
            *_due_date_checks(p),
            *_non_negative_checks(p, subtotal),
        ]
    statuses = {check.status for check in checks}
    status = (
        STATUS_FAIL
        if STATUS_FAIL in statuses
        else STATUS_PASS
        if STATUS_PASS in statuses
        else STATUS_SKIPPED
    )
    return Report(
        rules_version=RULES_VERSION,
        status=status,
        currency=precision.code,
        minor_units=precision.minor_units,
        assumptions=tuple(assumptions),
        derived=tuple(derived),
        checks=tuple(checks),
        suggestion=None,
    )


def reconcile(inp: ReconcileInput) -> Report:
    """Run every rule over a document's amounts. Pure: the input is not changed
    and nothing is read or written."""
    report = _run(inp)
    with _exact():
        suggestion = _suggest(inp, report)
    return dataclasses.replace(report, suggestion=suggestion)


# ---- Suggestion -----------------------------------------------------------


def _is_candidate(operand: Operand) -> bool:
    """A header amount, line total, or tax line amount the model read and nobody
    has reviewed, or a tax or amount paid assumed to be null (counted for
    ambiguity, never suggested)."""
    if operand.verified:
        return False
    if operand.source == SOURCE_ASSUMED:
        return operand.path in _ASSUMED_NULL_PATHS
    return operand.source == SOURCE_EXTRACTED and (
        operand.path in _HEADER_MONEY_PATHS
        or bool(_LINE_TOTAL_PATH_RE.fullmatch(operand.path))
        or bool(_TAX_LINE_AMOUNT_PATH_RE.fullmatch(operand.path))
    )


def _never_negative(path: str) -> bool:
    """A header amount or tax line amount is not suggested as a negative number
    (a line total may be: a discount or a return; so may an amount due, which is
    negative after an overpayment)."""
    return path in _NEVER_NEGATIVE_PATHS or bool(_TAX_LINE_AMOUNT_PATH_RE.fullmatch(path))


def _solve(check: Check, path: str) -> Decimal | None:
    """The value of the operand at `path` that makes this equation exact, or
    None when it cannot be solved exactly."""
    if check.rhs is None:
        return None
    try:
        if check.rhs.path == path:
            return sum((t.coef * t.operand.value for t in check.terms), Decimal(0))
        coef = next(t.coef for t in check.terms if t.operand.path == path)
        others = sum(
            (t.coef * t.operand.value for t in check.terms if t.operand.path != path), Decimal(0)
        )
        return (check.rhs.value - others) / coef
    except (Inexact, DivisionByZero, InvalidOperation):
        return None


def _quote_of(inp: ReconcileInput, path: str) -> tuple[bool, str | None]:
    """Whether the leaf at `path` carried an evidence object, and its quote."""
    if path in _HEADER_MONEY_PATHS:
        field: FieldInput = getattr(inp, path)
        return field.has_evidence, field.quote
    if path.startswith("tax_lines["):
        index = int(path[len("tax_lines[") : path.index("]")])
        quotes = (inp.tax_lines or ())[index].quotes
    else:
        index = int(path[len("line_items[") : path.index("]")])
        quotes = (inp.line_items or ())[index].quotes
    key = path.rpartition(".")[2]
    return key in quotes, quotes.get(key)


def _requoted(
    quotes: Mapping[str, str | None], key: str, value: Decimal, places: int
) -> dict[str, str | None]:
    """The cell quotes after `key` is set to `value`: the quote stays only when
    the new value agrees with it, else it goes, as it would when a person
    corrected the cell."""
    kept = dict(quotes)
    if key in kept and quote_agreement(value, kept[key], places=places).status != QUOTE_PASS:
        del kept[key]
    return kept


def _with_value(inp: ReconcileInput, path: str, value: Decimal, places: int) -> ReconcileInput:
    """The input with one suggestable amount replaced. `places` are the decimal
    places such an amount allows (the currency's minor unit)."""
    if path in _HEADER_MONEY_PATHS:
        field: FieldInput = getattr(inp, path)
        agrees = (
            field.has_evidence
            and quote_agreement(value, field.quote, places=places).status == QUOTE_PASS
        )
        return dataclasses.replace(
            inp,
            **{
                path: dataclasses.replace(
                    field,
                    value=value,
                    quote=field.quote if agrees else None,
                    has_evidence=agrees,
                )
            },
        )
    if path.startswith("tax_lines["):
        index = int(path[len("tax_lines[") : path.index("]")])
        tax_lines = list(inp.tax_lines or ())
        tax_lines[index] = dataclasses.replace(
            tax_lines[index],
            amount=value,
            quotes=_requoted(tax_lines[index].quotes, "amount", value, places),
        )
        return dataclasses.replace(inp, tax_lines=tuple(tax_lines))
    index = int(path[len("line_items[") : path.index("]")])
    lines = list(inp.line_items or ())
    lines[index] = dataclasses.replace(
        lines[index], total=value, quotes=_requoted(lines[index].quotes, "total", value, places)
    )
    return dataclasses.replace(inp, line_items=tuple(lines))


def _fixes_everything(before: Report, after: Report) -> bool:
    """Every check that failed now passes, and nothing fails."""
    passing = {check.id for check in after.checks if check.status == STATUS_PASS}
    return after.status != STATUS_FAIL and all(
        check.id in passing for check in before.failing_checks
    )


def _suggest(inp: ReconcileInput, report: Report) -> Suggestion | None:
    proposals: dict[tuple[str, Decimal], Operand] = {}
    # The proposals only a quote made: each also has to land on an equation that
    # can judge it (see below).
    quote_only: set[tuple[str, Decimal]] = set()

    def propose(operand: Operand, value: Decimal | None, *, from_quote: bool = False) -> None:
        if value is None or not fits_minor_unit(value, report.minor_units):
            return
        if value == operand.value or (value < 0 and _never_negative(operand.path)):
            return
        key = (operand.path, value)
        if from_quote and key not in proposals:
            quote_only.add(key)
        elif not from_quote:
            quote_only.discard(key)
        proposals.setdefault(key, operand)

    for check in report.failing_checks:
        if check.kind == KIND_QUOTE:
            operand = check.terms[0].operand
            if _is_candidate(operand):
                # The one number the quote prints, signed as the value is: the
                # quote shows magnitudes ("(5.00)" is a discount line of -5.00).
                numbers = printed_amounts(_quote_of(inp, operand.path)[1] or "", report.minor_units)
                if len(numbers) == 1:
                    propose(
                        operand, -numbers[0] if operand.value < 0 else numbers[0], from_quote=True
                    )
            continue
        if check.kind != KIND_EQUATION:
            continue
        for operand in check.operands:
            if _is_candidate(operand):
                # An equation loose because of rounding has no exact solution; the
                # re-run below judges it instead.
                propose(operand, _solve(check, operand.path))
    # A fix has to work under the ordinary rules, unless this document already
    # leans on a tax-included reading: otherwise changing the tax could "fix" a
    # document by making it look tax-inclusive. The same goes for the second
    # fallback base of a tax line (the one that adds shipping): trying it for
    # every candidate would let a misread amount slip through as a reading, so a
    # fix has to pass on the first base, and a document that passes only on the
    # second gets no suggestion.
    keep_included = any(
        operand.source == SOURCE_INCLUDED for check in report.checks for operand in check.operands
    )
    def survives(key: tuple[str, Decimal]) -> bool:
        after = _run(
            _with_value(inp, *key, report.minor_units),
            allow_included=keep_included,
            allow_fallbacks=False,
        )
        if not _fixes_everything(report, after):
            return False
        # A number from a quote alone must also be one the arithmetic can
        # confirm: with no equation to judge the field (an amount paid with no
        # amount due), any figure in its quote, a reference number say, would
        # become the fix.
        return key not in quote_only or any(
            check.kind == KIND_EQUATION and check.status != STATUS_SKIPPED and key[0] in check.fields
            for check in after.checks
        )

    survivors = [(key, operand) for key, operand in proposals.items() if survives(key)]
    if len(survivors) != 1 or survivors[0][1].source == SOURCE_ASSUMED:
        return None
    (path, value), operand = survivors[0]
    current = operand.value
    fixed = tuple(check.id for check in report.failing_checks)
    return Suggestion(
        path=path,
        value=value,
        current=current,
        check_ids=fixed,
        detail=(
            f"Setting {path} to {to_api_string(value, report.minor_units)} fixes "
            f"{', '.join(fixed)} and breaks no passing check."
        ),
    )


# ---- JSON -----------------------------------------------------------------


def _number(path: str, value: Decimal | date | None, minor_units: int) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    if path.endswith((".quantity", ".rate")):
        return to_plain_string(value)
    return to_api_string(value, minor_units)


def _operand_json(operand: Operand, minor_units: int) -> dict[str, Any]:
    return {
        "path": operand.path,
        "value": _number(operand.path, operand.value, minor_units),
        "source": operand.source,
        "verified": operand.verified,
    }


def _check_json(check: Check, minor_units: int) -> dict[str, Any]:
    places = RATE_PLACES if check.id == TAX_RATE else minor_units

    def amount(value: Decimal | None) -> str | None:
        return None if value is None else to_api_string(value, places)

    return {
        "id": check.id,
        "kind": check.kind,
        "formula": check.formula,
        "terms": [
            {
                "coef": None if term.coef is None else to_plain_string(term.coef),
                "operand": _operand_json(term.operand, minor_units),
            }
            for term in check.terms
        ],
        "rhs": None if check.rhs is None else _operand_json(check.rhs, minor_units),
        "expected": amount(check.expected),
        "actual": amount(check.actual),
        "delta": amount(check.delta),
        "tolerance": amount(check.tolerance),
        "status": check.status,
        "detail": check.detail,
        "fields": list(check.fields),
        "all_operands_verified": check.all_operands_verified,
    }


def report_to_json(report: Report) -> dict[str, Any]:
    """The report as stored (JSONB) and served: plain dicts, lists, strings,
    booleans, and nulls, with no Decimal or float in it.

    Shape, stable for `rules_v2` (and unchanged from `rules_v1`):

        {
          "rules_version": "rules_v2",
          "status": "pass" | "fail" | "skipped",
          "currency": "USD" | null,          # normalized code, null when unstated
          "minor_units": 2,
          "assumptions": ["..."],            # sentences
          "derived": [{"path", "value", "formula"}],
          "checks": [{
            "id", "kind", "formula",
            "terms": [{"coef", "operand"}],  # operand: {"path", "value", "source", "verified"}
                                             # value: a decimal string, or an ISO date for
                                             # document_date and due_date
            "rhs": operand | null,
            "expected", "actual", "delta", "tolerance",   # strings or null
            "status", "detail",              # detail is a sentence, null on a pass
            "fields": ["subtotal", "line_items[2].total"],
            "all_operands_verified": true
          }],
          "suggestion": {"path", "value", "current", "check_ids", "detail"} | null
        }

    Money (amounts, unit prices, expected, actual, delta, tolerance) is a
    decimal string padded to the document's minor units, keeping any extra
    digits so a precision failure stays visible. Quantities, rates, and
    coefficients are plain decimal strings (`"2.5"`, `"0.08875"`), and the
    `tax_rate` check's `actual` is the rate to four places (`"0.0700"`). A value that is absent or unusable is
    null. Checks are in rule order (format failures first); `line_math[i]`,
    `line_items[i]`, `tax_line_math[i]`, and `tax_lines[i]` count from 0.
    """
    mu = report.minor_units
    suggestion = report.suggestion
    return {
        "rules_version": report.rules_version,
        "status": report.status,
        "currency": report.currency,
        "minor_units": mu,
        "assumptions": list(report.assumptions),
        "derived": [
            {"path": d.path, "value": _number(d.path, d.value, mu), "formula": d.formula}
            for d in report.derived
        ],
        "checks": [_check_json(check, mu) for check in report.checks],
        "suggestion": None
        if suggestion is None
        else {
            "path": suggestion.path,
            "value": to_api_string(suggestion.value, mu),
            "current": _number(suggestion.path, suggestion.current, mu),
            "check_ids": list(suggestion.check_ids),
            "detail": suggestion.detail,
        },
    }
