"""Properties of the reconciliation engine over generated documents.

A consistent document is built the way a printer would: line totals are
quantity x unit price rounded half up to the minor unit, the subtotal is their
sum, tax is a plausible share of it, and the total adds up. Then the tests
break one number and check what the engine says. The profile in
tests/conftest.py makes every run draw the same examples."""

import copy
import dataclasses
import json
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_HALF_UP, Decimal

from hypothesis import assume, event, given
from hypothesis import strategies as st

from app.money import currency_precision
from app.reconcile import (
    STATUS_FAIL,
    STATUS_PASS,
    FieldInput,
    LineInput,
    ReconcileInput,
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


def set_path(inp: ReconcileInput, path: str, value: Decimal) -> ReconcileInput:
    """The input with one number replaced, addressed the way checks name it."""
    if path in ("subtotal", "tax", "total"):
        return dataclasses.replace(inp, **{path: FieldInput(value)})
    index, key = path[len("line_items[") :].split("].")
    lines = list(inp.line_items)
    lines[int(index)] = dataclasses.replace(lines[int(index)], **{key: value})
    return dataclasses.replace(inp, line_items=tuple(lines))


def get_path(inp: ReconcileInput, path: str) -> Decimal:
    if path in ("subtotal", "tax", "total"):
        return getattr(inp, path).value
    index, key = path[len("line_items[") :].split("].")
    return getattr(inp.line_items[int(index)], key)


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
