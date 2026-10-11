"""Reconciliation: do a document's amounts add up?

The model reads the printed numbers; code decides whether they agree with each
other. `reconcile` takes the amounts a document carries (subtotal, tax, total,
and every line item's quantity, unit price, and total) and runs a fixed set of
rules over them, in `Decimal`, with no I/O. The result is a `Report`: every
check with the values it used, an overall status, the assumptions made, and at
most one suggested correction. `report_to_json` is the only serializer.

Rules (`RULES_VERSION`; a new rule or tolerance is a new version):

- `line_math[i]`: quantity x unit price = line total. Exact when the product is
  exact in the currency's minor unit; one minor unit of slack only when the
  product needs rounding (3 x 0.333 = 0.999 printed as 1.00).
- `lines_sum`: the line totals add up to the subtotal, with half a minor unit of
  slack for each line whose product needed rounding.
- `total_math`: subtotal + tax = total, exact. A null tax counts as 0 and says
  so in the assumptions.
- `tax_rate`: tax / subtotal is between 0 and 0.30, judged on the rate as it is
  reported (four places), so the verdict never contradicts the number shown.
- `non_negative:<subtotal|tax|total>`: the header amounts are not negative. Line
  totals and quantities may be negative (discounts, returns).
- `lines_present`: a printed subtotal needs line items to add up.
- `format:<path>`: emitted only on failure: the value is not a number, is not
  finite, is at least MAX_AMOUNT, or has more decimals than allowed (the
  currency's minor unit for money, MAX_QUANTITY_DECIMALS for quantities and
  unit prices). Checks that need that value are skipped, with the reason.

Two assumptions let a tax-inclusive document (the common EU receipt) pass. Each
applies only when the numbers are exactly consistent with the tax-included
reading, is written to `assumptions`, and marks the tax operand with the source
`included` (coefficient 0 in the equation: the tax is already inside the other
amounts). They are tried only when the ordinary rule fails, so they never change
a document that already passes:

- Gross subtotal: a tax is printed (above 0), the subtotal (printed, or computed
  from the line totals) equals the total, and tax / (total - tax) is at most
  0.30. `total_math` then reads "subtotal = total" and `tax_rate` is judged on
  the net base, total - tax.
- Gross lines: the line totals add up to the total (within the `lines_sum`
  tolerance), subtotal + tax = total, and tax / subtotal is plausible.
  `lines_sum` then compares the line totals with the total.

A misread that does not fit either reading still fails. The one blind spot is a
total misread as exactly the subtotal on a document whose tax is plausible: the
numbers then look like a gross-subtotal receipt, and only a person can tell.

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
suggestion solve for one operand, and what lets later rules (discount,
shipping, tip, tax lines, amount due) join as more terms without a new
mechanism. The one product, quantity x unit price, is written with the quantity
as the coefficient of the unit price.

The suggestion is deliberately strict. Candidates are the unverified header
amounts and line totals that sit in a failing equation, plus an assumed (null)
tax. Each is solved for in each failing equation that contains it, which gives
proposals (path, value): exact in the minor unit, and not negative for a header
amount (a line total may be negative: a discount or a return). Every proposal is
applied and every rule re-run; it survives only when every failing check passes
and no passing check fails. A suggestion is reported only when exactly one
proposal survives, so an ambiguous misread (a total that disagrees with
subtotal + tax, where either could be wrong) gets none, and neither does one
whose only survivor is the assumed tax: a null tax means the model could not
read it, so a total that is right must not be overwritten to match. (An equation
loose because of rounding has no exact solution and is judged by the re-run.)
Quantities and unit prices are never suggested, and neither is a reviewed field.
The re-runs use the ordinary rules unless the document already leans on a
tax-included reading, so a fix cannot succeed merely by making the document look
tax-inclusive.
"""

import dataclasses
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
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
    shown_value,
    to_api_string,
    to_decimal,
    to_plain_string,
)
from app.records import field_value

if TYPE_CHECKING:
    from app.models import ExtractedField

RULES_VERSION = "rules_v1"

# Check and report status (strings, like the other closed sets here).
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIPPED = "skipped"

KIND_EQUATION = "equation"
KIND_BOUND = "bound"
KIND_PRESENCE = "presence"
KIND_FORMAT = "format"

# Where an operand's value came from. The first three are a field's own state
# (the model's answer, a reviewer's approval of it, a reviewer's correction);
# the rest are made by this module.
SOURCE_EXTRACTED = "extracted"
SOURCE_APPROVED = "approved"
SOURCE_CORRECTED = "corrected"
SOURCE_COMPUTED = "computed"
SOURCE_ASSUMED = "assumed"
# A tax already contained in the other amounts of the equation it sits in.
SOURCE_INCLUDED = "included"

LINES_PRESENT = "lines_present"
LINES_SUM = "lines_sum"
TOTAL_MATH = "total_math"
TAX_RATE = "tax_rate"

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
_HEADER_TOTAL_PATHS = ("subtotal", "tax", "total")


def line_math_id(index: int) -> str:
    return f"line_math[{index}]"


def line_path(index: int, key: str) -> str:
    """The path of one line item cell, such as `line_items[2].total`."""
    return f"line_items[{index}].{key}"


# ---- Input ----------------------------------------------------------------


@dataclass(frozen=True)
class FieldInput:
    """One field as the engine receives it: the raw value (anything JSON can
    hold, parsed and validated inside `reconcile`), where it came from, and
    whether a person has reviewed it."""

    value: Any = None
    source: str = SOURCE_EXTRACTED
    verified: bool = False


@dataclass(frozen=True)
class LineInput:
    """One line item's raw cells. The description plays no part in the rules."""

    quantity: Any = None
    unit_price: Any = None
    total: Any = None


@dataclass(frozen=True)
class ReconcileInput:
    """Everything `reconcile` reads. `line_items` is None when the document has
    no usable list. The line items are one field in storage, so they share one
    source and one verified flag."""

    currency: Any = None
    subtotal: FieldInput = FieldInput()
    tax: FieldInput = FieldInput()
    total: FieldInput = FieldInput()
    line_items: tuple[LineInput, ...] | None = None
    lines_source: str = SOURCE_EXTRACTED
    lines_verified: bool = False

    @classmethod
    def from_tool_input(cls, tool_input: Any) -> "ReconcileInput":
        """From the model's `record_extraction` input, a leaf at a time through
        `_coerce_leaf`: a malformed leaf is an absent value. Everything is
        extracted and unverified. Used by the evals."""
        tool = tool_input if isinstance(tool_input, dict) else {}

        def leaf(name: str) -> Any:
            return _coerce_leaf(tool.get(name))[0]

        rows = leaf("line_items")
        return cls(
            currency=leaf("currency"),
            subtotal=FieldInput(leaf("subtotal")),
            tax=FieldInput(leaf("tax")),
            total=FieldInput(leaf("total")),
            line_items=(
                tuple(_line_from_leaves(row) for row in rows) if isinstance(rows, list) else None
            ),
        )

    @classmethod
    def from_fields(cls, fields: Iterable["ExtractedField"]) -> "ReconcileInput":
        """From one extraction's stored fields, with `records.field_value`
        deciding the value (a reviewer's correction wins). A field is verified
        when it has been reviewed; its source says how."""
        by_name = {field.field_name: field for field in fields}

        def read(name: str) -> FieldInput:
            field = by_name.get(name)
            if field is None:
                return FieldInput()
            return FieldInput(field_value(field), _stored_source(field), field.reviewed_at is not None)

        currency = by_name.get("currency")
        lines = by_name.get("line_items")
        rows = field_value(lines) if lines is not None else None
        return cls(
            currency=field_value(currency) if currency is not None else None,
            subtotal=read("subtotal"),
            tax=read("tax"),
            total=read("total"),
            line_items=(
                tuple(_line_from_plain(row) for row in rows) if isinstance(rows, list) else None
            ),
            lines_source=_stored_source(lines) if lines is not None else SOURCE_EXTRACTED,
            lines_verified=lines is not None and lines.reviewed_at is not None,
        )


def _line_from_leaves(item: Any) -> LineInput:
    if not isinstance(item, dict):
        return LineInput()
    return LineInput(
        quantity=_coerce_leaf(item.get("quantity"))[0],
        unit_price=_coerce_leaf(item.get("unit_price"))[0],
        total=_coerce_leaf(item.get("total"))[0],
    )


def _line_from_plain(row: Any) -> LineInput:
    if not isinstance(row, dict):
        return LineInput()
    return LineInput(row.get("quantity"), row.get("unit_price"), row.get("total"))


def _stored_source(field: "ExtractedField") -> str:
    if field.review_action == "corrected":
        return SOURCE_CORRECTED
    return SOURCE_APPROVED if field.reviewed_at is not None else SOURCE_EXTRACTED


# ---- Output ---------------------------------------------------------------


@dataclass(frozen=True)
class Operand:
    """A number a check used. `value` is None when the field is missing or
    unusable; `verified` is whether a person has reviewed where it came from
    (a computed or assumed operand is verified when its inputs are). A
    computed operand has no row of its own to blame: `inputs` are the paths it
    was computed from, and a check lists those in `fields` instead of `path`."""

    path: str
    value: Decimal | None
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
class _Parsed:
    """The input after parsing: operands with Decimal values (None when absent
    or unusable) plus the format failures found on the way."""

    minor_units: int
    subtotal: Operand
    tax: Operand
    total: Operand
    lines: list[_Line]
    lines_listed: bool
    lines_source: str
    lines_verified: bool
    format_checks: list[Check]
    unreadable: set[str]

    def money(self, value: Decimal) -> str:
        return to_api_string(value, self.minor_units)

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


def _parse(inp: ReconcileInput, minor_units: int) -> _Parsed:
    format_checks: list[Check] = []
    unreadable: set[str] = set()

    def header(name: str) -> Operand:
        field: FieldInput = getattr(inp, name)
        return _read(
            name, field.value, field.source, field.verified, minor_units, format_checks, unreadable
        )

    subtotal, tax, total = header("subtotal"), header("tax"), header("total")
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
        lines.append(_Line(cells["quantity"], cells["unit_price"], cells["total"]))
    return _Parsed(
        minor_units=minor_units,
        subtotal=subtotal,
        tax=tax,
        total=total,
        lines=lines,
        lines_listed=bool(inp.line_items),
        lines_source=inp.lines_source,
        lines_verified=inp.lines_verified,
        format_checks=format_checks,
        unreadable=unreadable,
    )


# ---- Checks ---------------------------------------------------------------


def _provenance(operands: Sequence[Operand]) -> tuple[tuple[str, ...], bool]:
    """The paths a check involves and whether a person verified them all. A
    computed operand (an unprinted subtotal or line total) stands for the
    values it was computed from: the field that is absent never made a
    contribution to the equation, so it is not named."""
    paths = tuple(
        dict.fromkeys(
            path
            for op in operands
            for path in (op.inputs if op.source == SOURCE_COMPUTED and op.inputs else (op.path,))
        )
    )
    return paths, all(op.verified for op in operands)


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


def _total_math_check(p: _Parsed, subtotal: Operand) -> tuple[Check, bool]:
    """total_math, and whether it counted a null tax as 0."""
    tax = p.tax
    # A null tax is only assumed to be 0 when there is an equation to run.
    unknown = p.skip_reason([subtotal, p.total])
    assumed = unknown is None and tax.value is None and tax.path not in p.unreadable
    if assumed:
        tax = Operand("tax", Decimal(0), SOURCE_ASSUMED, p.tax.verified)

    def explain(expected: Decimal, actual: Decimal, delta: Decimal) -> str:
        counted = " (no tax is printed, so it counts as 0)" if assumed else ""
        return (
            f"The subtotal {p.money(subtotal.value)} plus tax {p.money(tax.value)}{counted} is "
            f"{p.money(expected)}, but the total is {p.money(actual)} "
            f"(off by {p.money(abs(delta))})."
        )

    check = _equation(
        p,
        TOTAL_MATH,
        "subtotal + tax = total",
        [Term(_ONE, subtotal), Term(_ONE, tax)],
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
    printed tax, a subtotal equal to the total, and a plausible rate on the net
    amount. None when the numbers do not fit that reading exactly."""
    tax, total = p.tax.value, p.total.value
    if tax is None or total is None or subtotal.value is None:
        return None
    if tax <= 0 or subtotal.value != total or not _plausible_rate(tax, total - tax):
        return None
    included = Operand("tax", tax, SOURCE_INCLUDED, p.tax.verified)
    check = _equation(
        p,
        TOTAL_MATH,
        "subtotal = total (the total includes the tax)",
        [Term(_ONE, subtotal), Term(Decimal(0), included)],
        p.total,
        Decimal(0),
        lambda expected, actual, delta: "",
    )
    return check if check.status == STATUS_PASS else None


def _gross_lines_check(p: _Parsed, lines_sum: Check) -> Check | None:
    """lines_sum for a document whose line prices include the tax: the lines
    add up to the total (within the usual tolerance) while subtotal + tax = total
    and the rate on the subtotal is plausible. The caller has checked that the
    ordinary lines_sum fails and total_math passes."""
    tax, subtotal = p.tax.value, p.subtotal.value
    if tax is None or subtotal is None or tax <= 0 or not _plausible_rate(tax, subtotal):
        return None
    included = Operand("tax", tax, SOURCE_INCLUDED, p.tax.verified)
    check = _equation(
        p,
        LINES_SUM,
        "sum of line totals = total (the line prices include the tax)",
        [*lines_sum.terms, Term(Decimal(0), included)],
        p.total,
        lines_sum.tolerance,
        lambda expected, actual, delta: "",
    )
    return check if check.status == STATUS_PASS else None


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
    for operand in (subtotal, p.tax, p.total):
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


def _run(inp: ReconcileInput, *, allow_included: bool = True) -> Report:
    """Every rule, without the suggestion (which re-runs this). With
    `allow_included` False the tax-included assumptions are not tried."""
    currency = inp.currency if isinstance(inp.currency, str) else None
    precision: CurrencyPrecision = currency_precision(currency)
    assumptions = [precision.assumption] if precision.assumption else []
    with _exact():
        p = _parse(inp, precision.minor_units)
        derived: list[Derived] = []
        line_checks = _line_checks(p, derived)
        lines_sum, subtotal = _lines_sum_check(p, derived)
        total_math, tax_assumed = _total_math_check(p, subtotal)
        if tax_assumed:
            assumptions.append("tax is null; total_math counts it as 0.")
        net_base = False
        gross_subtotal = gross_lines = None
        if allow_included and total_math.status == STATUS_FAIL:
            gross_subtotal = _gross_subtotal_check(p, subtotal)
        elif allow_included and lines_sum.status == STATUS_FAIL and total_math.status == STATUS_PASS:
            gross_lines = _gross_lines_check(p, lines_sum)
        if gross_subtotal is not None:
            total_math, net_base = gross_subtotal, True
            assumptions.append(
                "prices include tax; total_math reads the subtotal as already containing "
                f"the tax {to_api_string(p.tax.value, p.minor_units)}, and tax_rate is "
                "judged on total - tax."
            )
        if gross_lines is not None:
            lines_sum = gross_lines
            assumptions.append(
                "line prices include tax; lines_sum compares the line totals with the "
                "total instead of the subtotal."
            )
        checks = [
            *p.format_checks,
            _lines_present_check(p),
            *line_checks,
            lines_sum,
            total_math,
            _tax_rate_check(p, subtotal, net_base=net_base),
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
    """A header amount or line total the model read and nobody has reviewed, or
    a tax assumed to be null (counted for ambiguity, never suggested)."""
    if operand.verified:
        return False
    if operand.source == SOURCE_ASSUMED:
        return operand.path == "tax"
    return operand.source == SOURCE_EXTRACTED and (
        operand.path in _HEADER_TOTAL_PATHS or bool(_LINE_TOTAL_PATH_RE.fullmatch(operand.path))
    )


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


def _with_value(inp: ReconcileInput, path: str, value: Decimal) -> ReconcileInput:
    if path in _HEADER_TOTAL_PATHS:
        return dataclasses.replace(inp, **{path: dataclasses.replace(getattr(inp, path), value=value)})
    index = int(path[len("line_items[") : path.index("]")])
    lines = list(inp.line_items or ())
    lines[index] = dataclasses.replace(lines[index], total=value)
    return dataclasses.replace(inp, line_items=tuple(lines))


def _fixes_everything(before: Report, after: Report) -> bool:
    """Every check that failed now passes, and nothing fails."""
    passing = {check.id for check in after.checks if check.status == STATUS_PASS}
    return after.status != STATUS_FAIL and all(
        check.id in passing for check in before.failing_checks
    )


def _suggest(inp: ReconcileInput, report: Report) -> Suggestion | None:
    proposals: dict[tuple[str, Decimal], Operand] = {}
    for check in report.failing_checks:
        if check.kind != KIND_EQUATION:
            continue
        for operand in check.operands:
            if not _is_candidate(operand):
                continue
            # An equation loose because of rounding has no exact solution; the
            # re-run below judges it instead.
            value = _solve(check, operand.path)
            if value is None or not fits_minor_unit(value, report.minor_units):
                continue
            if value == operand.value or (value < 0 and operand.path in _HEADER_TOTAL_PATHS):
                continue
            proposals.setdefault((operand.path, value), operand)
    # A fix has to work under the ordinary rules, unless this document already
    # leans on a tax-included reading: otherwise changing the tax could "fix" a
    # document by making it look tax-inclusive.
    keep_included = any(
        operand.source == SOURCE_INCLUDED for check in report.checks for operand in check.operands
    )
    survivors = [
        (key, operand)
        for key, operand in proposals.items()
        if _fixes_everything(report, _run(_with_value(inp, *key), allow_included=keep_included))
    ]
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


def _number(path: str, value: Decimal | None, minor_units: int) -> str | None:
    if value is None:
        return None
    if path.endswith(".quantity"):
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

    Shape, stable for `rules_v1`:

        {
          "rules_version": "rules_v1",
          "status": "pass" | "fail" | "skipped",
          "currency": "USD" | null,          # normalized code, null when unstated
          "minor_units": 2,
          "assumptions": ["..."],            # sentences
          "derived": [{"path", "value", "formula"}],
          "checks": [{
            "id", "kind", "formula",
            "terms": [{"coef", "operand"}],  # operand: {"path", "value", "source", "verified"}
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
    digits so a precision failure stays visible. Quantities and coefficients
    are plain decimal strings (`"2.5"`), and the `tax_rate` check's `actual` is
    the rate to four places (`"0.0700"`). A value that is absent or unusable is
    null. Checks are in rule order (format failures first); `line_math[i]` and
    `line_items[i]` count from 0.
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
