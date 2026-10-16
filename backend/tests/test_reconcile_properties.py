"""Properties of the reconciliation engine over generated documents.

A consistent document is built the way a printer would: line totals are
quantity x unit price rounded half up to the minor unit, the subtotal is their
sum, tax is a plausible share of it, and the total adds up. Then the tests
break one number and check what the engine says. The profile in
tests/conftest.py makes every run draw the same examples."""

import copy
import dataclasses
import json
import re
from datetime import date, timedelta
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_HALF_UP, Decimal

from hypothesis import assume, event, given
from hypothesis import strategies as st

from app.money import currency_precision
from app.reconcile import (
    SOURCE_INCLUDED,
    STATUS_FAIL,
    STATUS_PASS,
    FieldInput,
    LineInput,
    ReconcileInput,
    TaxLineInput,
    reconcile,
    report_to_json,
)

CURRENCIES = ["USD", "EUR", "JPY", "BHD", None]


def unit_of(currency) -> Decimal:
    return Decimal(1).scaleb(-currency_precision(currency).minor_units)


@st.composite
def consistent_inputs(draw) -> ReconcileInput:
    currency = draw(st.sampled_from(CURRENCIES))
    minor_units = currency_precision(currency).minor_units
    unit = unit_of(currency)
    lines = []
    for _ in range(draw(st.integers(1, 8))):
        quantity = draw(
            st.one_of(
                st.integers(1, 20).map(Decimal),
                st.integers(500, 20_000).map(lambda thousandths: Decimal(thousandths).scaleb(-3)),
            )
        )
        extra = draw(st.integers(0, 2))  # unit prices may carry digits beyond the minor unit
        unit_price = Decimal(draw(st.integers(10**extra, 10 ** (5 + extra)))).scaleb(
            -(minor_units + extra)
        )
        total = (quantity * unit_price).quantize(unit, rounding=ROUND_HALF_UP)
        lines.append(LineInput(quantity, unit_price, total))
    if draw(st.booleans()):
        # A return of a quarter of one line, at its price: a negative line total
        # that cannot take the subtotal to zero.
        source = lines[draw(st.integers(0, len(lines) - 1))]
        quantity = -(source.quantity / 4)
        total = (quantity * source.unit_price).quantize(unit, rounding=ROUND_HALF_UP)
        lines.append(LineInput(quantity, source.unit_price, total))
    subtotal = sum((line.total for line in lines), Decimal(0))
    tax = (subtotal * Decimal(draw(st.integers(0, 300))) / 1000).quantize(unit, rounding=ROUND_DOWN)
    tax_printed = draw(st.sampled_from([True, True, True, True, False]))
    return ReconcileInput(
        currency=currency,
        subtotal=FieldInput(subtotal),
        tax=FieldInput(tax if tax_printed else None),
        total=FieldInput(subtotal + (tax if tax_printed else Decimal(0))),
        line_items=tuple(lines),
    )


def rounded_lines(inp: ReconcileInput) -> int:
    """How many lines have a product that is not whole minor units."""
    unit = unit_of(inp.currency)
    return sum((line.quantity * line.unit_price) % unit != 0 for line in inp.line_items)


HEADER_AMOUNTS = (
    "subtotal", "tax", "total", "discount", "shipping", "tip", "amount_paid", "amount_due",
)


def split_path(path: str) -> tuple[str, int, str]:
    """`line_items[2].total` -> ("line_items", 2, "total")."""
    name, rest = path.split("[")
    index, key = rest.split("].")
    return name, int(index), key


def set_path(inp: ReconcileInput, path: str, value: Decimal) -> ReconcileInput:
    """The input with one number replaced, addressed the way checks name it."""
    if path in HEADER_AMOUNTS:
        return dataclasses.replace(inp, **{path: FieldInput(value)})
    name, index, key = split_path(path)
    rows = list(getattr(inp, name))
    rows[index] = dataclasses.replace(rows[index], **{key: value})
    return dataclasses.replace(inp, **{name: tuple(rows)})


def get_path(inp: ReconcileInput, path: str) -> Decimal:
    if path in HEADER_AMOUNTS:
        return getattr(inp, path).value
    name, index, key = split_path(path)
    return getattr(getattr(inp, name)[index], key)


def reads_as_tax_included(inp: ReconcileInput) -> bool:
    """True when the numbers also fit a tax-included receipt (the engine's
    documented blind spot): a perturbation that lands there is a different,
    consistent document, not a misread the engine can see."""
    tax, subtotal, total = inp.tax.value, inp.subtotal.value, inp.total.value
    if tax is None or tax <= 0:
        return False
    lines_total = sum((line.total for line in inp.line_items), Decimal(0))
    return subtotal == total or (subtotal + tax == total and lines_total == total)


def amount_paths(inp: ReconcileInput) -> list[str]:
    paths = ["subtotal", "total"]
    if inp.tax.value is not None:
        paths.append("tax")
    paths += [f"line_items[{i}].total" for i in range(len(inp.line_items))]
    return paths


@given(consistent_inputs())
def test_a_consistent_document_never_fails(inp) -> None:
    report = reconcile(inp)

    assert report.status == STATUS_PASS, [c.detail for c in report.failing_checks]
    assert report.suggestion is None
    assert json.loads(json.dumps(report_to_json(report))) == report_to_json(report)


# ---- Quotes printed the way real documents print them -----------------------

# (group separator, decimal separator, prefix, suffix) of each style.
PRINT_STYLES = {
    "us": (",", ".", "$", ""),
    "eu": (".", ",", "", " EUR"),
    "space": ("\u00a0", ",", "", " \u20ac"),
    "narrow": ("\u202f", ",", "", " \u20ac"),
    "swiss": ("'", ".", "CHF ", ""),
    "plain": ("", ".", "Total: ", ""),
}


def printed(value: Decimal, style: str, parens: bool) -> str:
    """The text a document prints for `value`: every digit it carries, grouped
    and decorated in one of PRINT_STYLES, a negative as "(5.00)" or "-$5.00"."""
    group, point, prefix, suffix = PRINT_STYLES[style]
    whole, _, fraction = format(abs(value), "f").partition(".")
    body = f"{int(whole):,}".replace(",", group) + (f"{point}{fraction}" if fraction else "")
    text = f"{prefix}{body}{suffix}"
    if value >= 0:
        return text
    return f"({text})" if parens else f"-{text}"


@st.composite
def quoted_inputs(draw) -> ReconcileInput:
    """A consistent document in which every printed amount and line cell carries
    a faithful quote, in one style drawn per document."""
    inp = draw(consistent_inputs())
    style = draw(st.sampled_from(sorted(PRINT_STYLES)))
    parens = draw(st.booleans())

    def field(f: FieldInput) -> FieldInput:
        if f.value is None:
            return f
        return dataclasses.replace(f, quote=printed(f.value, style, parens), has_evidence=True)

    lines = tuple(
        dataclasses.replace(
            line,
            quotes={
                key: printed(getattr(line, key), style, parens)
                for key in ("quantity", "unit_price", "total")
            },
        )
        for line in inp.line_items
    )
    return dataclasses.replace(
        inp,
        subtotal=field(inp.subtotal),
        tax=field(inp.tax),
        total=field(inp.total),
        line_items=lines,
    )


def with_value_keeping_quote(inp: ReconcileInput, path: str, value: Decimal) -> ReconcileInput:
    """`set_path` that leaves the quote as it was printed."""
    if path in HEADER_AMOUNTS:
        return dataclasses.replace(inp, **{path: dataclasses.replace(getattr(inp, path), value=value)})
    name, index, key = split_path(path)
    rows = list(getattr(inp, name))
    rows[index] = dataclasses.replace(rows[index], **{key: value})
    return dataclasses.replace(inp, **{name: tuple(rows)})


@given(quoted_inputs())
def test_faithful_quotes_in_any_print_style_never_fail_a_check(inp) -> None:
    report = reconcile(inp)

    quote_checks = [c for c in report.checks if c.id.startswith("quote:")]
    assert quote_checks
    assert report.status == STATUS_PASS, [(c.id, c.detail) for c in report.failing_checks]
    assert {c.status for c in quote_checks} == {STATUS_PASS}


@given(quoted_inputs(), st.data())
def test_a_value_that_differs_from_its_faithful_quote_fails_the_quote_check_and_is_fixed_by_it(
    inp, data
) -> None:
    unit = unit_of(inp.currency)
    path = data.draw(st.sampled_from(amount_paths(inp)))
    original = get_path(inp, path)
    shift = data.draw(st.integers(1, 60)) * unit * data.draw(st.sampled_from([1, -1]))
    perturbed = original + shift
    assume((perturbed < 0) == (original < 0))

    report = reconcile(with_value_keeping_quote(inp, path, perturbed))

    assert f"quote:{path}" in [c.id for c in report.failing_checks]
    assert report.suggestion is not None, (path, shift, [c.id for c in report.failing_checks])
    assert (report.suggestion.path, report.suggestion.value) == (path, original)


@given(consistent_inputs(), st.data())
def test_perturbing_one_amount_beyond_its_tolerance_fails_a_check_naming_it(inp, data) -> None:
    unit = unit_of(inp.currency)
    unit_prices = [f"line_items[{i}].unit_price" for i in range(len(inp.line_items))]
    path = data.draw(st.sampled_from(amount_paths(inp) + unit_prices))
    if path.endswith(".unit_price"):
        quantity = inp.line_items[int(path[len("line_items[") : path.index("]")])].quantity
        # Moves the product by at least three minor units, past any rounding slack.
        shift = (3 * unit / quantity).quantize(Decimal("0.000001"), rounding=ROUND_CEILING)
    else:
        # Past the line_math slack (one unit) and the lines_sum slack (half a
        # unit per rounded line).
        steps = data.draw(st.integers(2 + rounded_lines(inp), 60))
        shift = steps * unit * data.draw(st.sampled_from([1, -1]))

    broken = set_path(inp, path, get_path(inp, path) + shift)
    assume(not reads_as_tax_included(broken))

    report = reconcile(broken)

    failing = [c.id for c in report.failing_checks]
    assert any(path in c.fields for c in report.failing_checks), (path, shift, failing)


@given(consistent_inputs(), st.data())
def test_a_misread_subtotal_or_line_total_has_one_suggestion_the_original(inp, data) -> None:
    unit = unit_of(inp.currency)
    path = data.draw(st.sampled_from([p for p in amount_paths(inp) if p not in ("tax", "total")]))
    steps = data.draw(st.integers(2 + rounded_lines(inp), 60))
    shift = steps * unit * data.draw(st.sampled_from([1, -1]))
    broken = set_path(inp, path, get_path(inp, path) + shift)

    report = reconcile(broken)

    assert report.status == STATUS_FAIL
    assert report.suggestion is not None, (path, shift, [c.id for c in report.failing_checks])
    assert report.suggestion.path == path
    assert report.suggestion.value == get_path(inp, path)
    assert report.suggestion.current == get_path(broken, path)


@given(consistent_inputs(), st.data())
def test_applying_any_suggestion_passes_every_check(inp, data) -> None:
    unit = unit_of(inp.currency)
    broken = inp
    for path in data.draw(st.lists(st.sampled_from(amount_paths(inp)), min_size=1, max_size=2, unique=True)):
        shift = data.draw(st.integers(1, 40)) * unit * data.draw(st.sampled_from([1, -1]))
        broken = set_path(broken, path, get_path(broken, path) + shift)

    report = reconcile(broken)
    if report.suggestion is None:
        event("no suggestion")
        return
    event("suggestion")

    fixed = reconcile(set_path(broken, report.suggestion.path, report.suggestion.value))
    assert fixed.status == STATUS_PASS
    assert [c.id for c in fixed.failing_checks] == []


@given(consistent_inputs(), st.data())
def test_reconcile_does_not_change_its_input(inp, data) -> None:
    unit = unit_of(inp.currency)
    broken = set_path(inp, "subtotal", inp.subtotal.value + data.draw(st.integers(-5, 5)) * unit)
    before = copy.deepcopy(broken)

    first = reconcile(broken)
    second = reconcile(broken)

    assert broken == before
    assert first == second
    assert report_to_json(first) == report_to_json(second)


@given(consistent_inputs())
def test_the_tool_input_builder_does_not_change_the_tool_input(inp) -> None:
    def leaf(value):
        return {"value": value, "confidence": 0.9}

    tool_input = {
        "currency": leaf(inp.currency),
        "subtotal": leaf(float(inp.subtotal.value)),
        "tax": leaf(None),
        "total": leaf(float(inp.total.value)),
        "line_items": leaf(
            [
                {key: leaf(float(getattr(line, key))) for key in ("quantity", "unit_price", "total")}
                for line in inp.line_items
            ]
        ),
    }
    before = copy.deepcopy(tool_input)

    report = reconcile(ReconcileInput.from_tool_input(tool_input))

    assert tool_input == before
    assert report.status in (STATUS_PASS, STATUS_FAIL)


@given(consistent_inputs(), st.data())
def test_the_lines_sum_slack_absorbs_a_rounding_sized_difference(inp, data) -> None:
    unit = unit_of(inp.currency)
    rounded = rounded_lines(inp)
    assume(rounded >= 2)
    shift = data.draw(st.integers(1, rounded // 2)) * unit * data.draw(st.sampled_from([1, -1]))
    subtotal = inp.subtotal.value + shift
    # A smaller subtotal raises the tax rate; keep it plausible.
    assume(inp.tax.value is None or inp.tax.value * 10 <= 3 * subtotal)
    shifted = dataclasses.replace(
        inp, subtotal=FieldInput(subtotal), total=FieldInput(inp.total.value + shift)
    )

    report = reconcile(shifted)

    assert report.status == STATUS_PASS, [c.detail for c in report.failing_checks]
    lines_sum = next(c for c in report.checks if c.id == "lines_sum")
    assert lines_sum.delta == shift
    assert abs(lines_sum.delta) <= lines_sum.tolerance


@given(consistent_inputs(), st.data())
def test_a_misread_total_with_no_tax_is_ambiguous_when_the_gap_is_a_plausible_tax(inp, data) -> None:
    unit = unit_of(inp.currency)
    subtotal = inp.subtotal.value
    gap = data.draw(st.integers(1, 60)) * unit
    # Small enough to be a tax the engine would accept for this subtotal.
    assume(gap * 10 <= 3 * subtotal)
    broken = dataclasses.replace(
        inp, tax=FieldInput(None), total=FieldInput(subtotal + gap)
    )

    report = reconcile(broken)

    # Either the total is wrong or the tax was missed: no single answer.
    assert [c.id for c in report.failing_checks] == ["total_math"]
    assert report.suggestion is None


@given(consistent_inputs(), st.data())
def test_a_total_misread_below_the_subtotal_is_the_only_fix_with_no_tax(inp, data) -> None:
    unit = unit_of(inp.currency)
    subtotal = inp.subtotal.value
    # Past the lines_sum slack, or a smaller subtotal would be an answer too.
    gap = data.draw(st.integers(2 + rounded_lines(inp), 60 + rounded_lines(inp))) * unit
    assume(gap <= subtotal)
    broken = dataclasses.replace(
        inp, tax=FieldInput(None), total=FieldInput(subtotal - gap)
    )

    report = reconcile(broken)

    # A tax cannot be negative, so only the total can be the culprit.
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("total", subtotal)


# ---- rules_v2 ---------------------------------------------------------------

RATES = ["5", "6", "7.5", "8.875", "10"]


@st.composite
def consistent_v2_inputs(draw) -> ReconcileInput:
    """A consistent document with the v2 extras: a discount, shipping, a tip,
    tax lines (each on a printed base, or on none, in which case it is taxed on
    the subtotal less the discount), a payment and an amount due, and dates."""
    inp = draw(consistent_inputs())
    unit = unit_of(inp.currency)
    subtotal = inp.subtotal.value

    def share(percent: int, of: Decimal) -> Decimal:
        return (of * percent / 100).quantize(unit, rounding=ROUND_DOWN)

    discount = draw(st.one_of(st.none(), st.integers(1, 25).map(lambda pct: share(pct, subtotal))))
    shipping = draw(st.one_of(st.none(), st.integers(0, 400).map(lambda n: n * unit)))
    tip = draw(st.one_of(st.none(), st.integers(0, 400).map(lambda n: n * unit)))
    taxable = subtotal - (discount or 0)

    tax_lines = []
    for _ in range(draw(st.integers(0, 2))):
        rate = draw(st.sampled_from(RATES))
        printed = draw(st.booleans())
        base = share(draw(st.integers(1, 100)), subtotal) if printed else taxable
        amount = (Decimal(rate) / 100 * base).quantize(unit, rounding=ROUND_HALF_UP)
        tax_lines.append(TaxLineInput("Tax", rate, base if printed else None, amount))

    if tax_lines:
        tax = sum((row.amount for row in tax_lines), Decimal(0))
        tax_printed = draw(st.booleans())
    else:
        tax, tax_printed = inp.tax.value or Decimal(0), inp.tax.value is not None
    total = subtotal - (discount or 0) + (shipping or 0) + tax + (tip or 0)

    paid = due = None
    mode = draw(st.sampled_from(["none", "both", "due only", "paid only"]))
    if mode != "none":
        paid_amount = share(draw(st.integers(0, 100)), total)
        paid = paid_amount if mode in ("both", "paid only") else None
        due = total - paid_amount if mode == "both" else total if mode == "due only" else None

    issued = date(2025, 1, 1) + timedelta(days=draw(st.integers(0, 300)))
    due_date = draw(st.one_of(st.none(), st.integers(0, 90).map(lambda n: issued + timedelta(days=n))))
    return dataclasses.replace(
        inp,
        tax=FieldInput(tax if tax_printed else None),
        total=FieldInput(total),
        discount=FieldInput(discount),
        shipping=FieldInput(shipping),
        tip=FieldInput(tip),
        amount_paid=FieldInput(paid),
        amount_due=FieldInput(due),
        document_date=FieldInput(issued.isoformat()),
        due_date=FieldInput(None if due_date is None else due_date.isoformat()),
        tax_lines=tuple(tax_lines) or None,
    )


def printed_v2_paths(inp: ReconcileInput) -> list[str]:
    """The numbers a broken copy can be told apart from the original by. An
    amount paid beside a derived amount due, and a derived tax, are not printed
    and are left alone."""
    paths = [f"line_items[{i}].total" for i in range(len(inp.line_items))]
    paths += [name for name in HEADER_AMOUNTS if getattr(inp, name).value is not None]
    if inp.amount_due.value is None:
        paths = [path for path in paths if path != "amount_paid"]
    for index, row in enumerate(inp.tax_lines or ()):
        paths.append(f"tax_lines[{index}].amount")
        if row.base is not None:
            paths.append(f"tax_lines[{index}].base")
    return paths


@given(consistent_v2_inputs())
def test_a_consistent_v2_document_never_fails(inp) -> None:
    report = reconcile(inp)

    assert report.status == STATUS_PASS, [c.detail for c in report.failing_checks]
    assert report.suggestion is None
    assert json.loads(json.dumps(report_to_json(report))) == report_to_json(report)


@given(consistent_v2_inputs(), st.data())
def test_breaking_one_v2_number_fails_a_check_naming_it(inp, data) -> None:
    unit = unit_of(inp.currency)
    path = data.draw(st.sampled_from(printed_v2_paths(inp)))
    if path.endswith(".base"):
        rate = Decimal(inp.tax_lines[int(path[len("tax_lines[") : path.index("]")])].rate)
        # Moves rate x base by more than the one minor unit of slack.
        shift = (3 * unit * 100 / rate).quantize(Decimal(1), rounding=ROUND_CEILING)
    else:
        shift = data.draw(st.integers(2 + rounded_lines(inp), 60)) * unit
    broken = set_path(inp, path, get_path(inp, path) + shift * data.draw(st.sampled_from([1, -1])))

    report = reconcile(broken)

    included = any(
        operand.source == SOURCE_INCLUDED for check in report.checks for operand in check.operands
    )
    assume(not included)  # a different, consistent reading (tax in the total, discount applied)
    failing = [c.id for c in report.failing_checks]
    assert any(path in c.fields for c in report.failing_checks), (path, shift, failing)


V1_CHECK_ID = re.compile(
    r"line_math\[\d+\]|lines_present|lines_sum|total_math|tax_rate"
    r"|non_negative:(subtotal|tax|total)"
    r"|format:(subtotal|tax|total|line_items\[\d+\]\.(quantity|unit_price|total))"
)


@given(consistent_inputs(), st.data())
def test_a_v1_shaped_record_never_emits_a_v2_check_or_assumption(inp, data) -> None:
    unit = unit_of(inp.currency)
    path = data.draw(st.sampled_from(amount_paths(inp)))
    shift = data.draw(st.integers(-60, 60)) * unit
    broken = set_path(inp, path, get_path(inp, path) + shift)

    report = reconcile(broken)

    assert [c.id for c in report.checks if not V1_CHECK_ID.fullmatch(c.id)] == []
    assert {d.path for d in report.derived} <= {"subtotal"} | {
        f"line_items[{i}].total" for i in range(len(inp.line_items))
    }
    assert all("tax line" not in a and "payment" not in a and "discount" not in a for a in report.assumptions)
    assert report.rules_version == "rules_v2"


# ---- A tax printed once, held as the tax and as its only tax line -------------


@st.composite
def mirrored_tax_inputs(draw) -> ReconcileInput:
    """A consistent document whose printed tax is also its only tax line (no
    rate, no base), as a model holds a tax that is printed once, with a discount,
    shipping, and a tip beside it. Half the time both cite the same page and
    quote, which print the true tax."""
    inp = draw(consistent_inputs().filter(lambda i: i.tax.value is not None))
    unit = unit_of(inp.currency)
    tax, subtotal = inp.tax.value, inp.subtotal.value
    discount = draw(st.one_of(st.none(), st.integers(1, 25).map(lambda n: (subtotal * n / 100).quantize(unit, rounding=ROUND_DOWN))))
    shipping = draw(st.one_of(st.none(), st.integers(0, 400).map(lambda n: n * unit)))
    tip = draw(st.one_of(st.none(), st.integers(0, 400).map(lambda n: n * unit)))
    total = subtotal - (discount or 0) + (shipping or 0) + tax + (tip or 0)
    cited = draw(st.booleans())
    quote = f"Tax: {tax}"
    return dataclasses.replace(
        inp,
        tax=FieldInput(tax, quote=quote if cited else None, has_evidence=cited, page=1 if cited else None),
        total=FieldInput(total),
        discount=FieldInput(discount),
        shipping=FieldInput(shipping),
        tip=FieldInput(tip),
        tax_lines=(
            TaxLineInput(
                "Tax",
                None,
                None,
                tax,
                quotes={"amount": quote} if cited else {},
                pages={"amount": 1} if cited else {},
            ),
        ),
    )


@given(mirrored_tax_inputs(), st.data())
def test_misreading_a_tax_held_twice_never_changes_a_correctly_read_field(inp, data) -> None:
    unit = unit_of(inp.currency)
    original = inp.tax.value
    shift = data.draw(st.integers(1, 60)) * unit * data.draw(st.sampled_from([1, -1]))
    assume(original + shift >= 0)
    line = inp.tax_lines[0]
    # The quote keeps printing the true tax; the value, in both places, is read wrong.
    broken = dataclasses.replace(
        inp,
        tax=dataclasses.replace(inp.tax, value=original + shift),
        tax_lines=(dataclasses.replace(line, amount=original + shift),),
    )

    suggestion = reconcile(broken).suggestion

    event("suggested" if suggestion is not None else "none")
    assert suggestion is None or (suggestion.path, suggestion.value) == ("tax", original)
