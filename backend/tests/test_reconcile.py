"""Reconciliation rules (app/reconcile.py): pure functions over the committed
eval labels and runs, plus hand-built records for each rule's edges. No
database and no network. tests/test_reconcile_properties.py holds the
generated cases."""

import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.demo_seed import STAGED_REVIEW_CASES
from app.evals.dataset import EVALS_DIR, load_cases
from app.models import ExtractedField
from app.reconcile import (
    RULES_VERSION,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED,
    FieldInput,
    LineInput,
    ReconcileInput,
    reconcile,
    report_to_json,
)

RESULTS_DIR = EVALS_DIR / "results"
SONNET_RUNS = sorted(RESULTS_DIR.glob("*claude-sonnet*_extract_v1.json"))
HAIKU_RUN = next(RESULTS_DIR.glob("*claude-haiku*_extract_v1.json"))


def leaf(value, confidence=1.0):
    return {"value": value, "confidence": confidence}


def tool_input(*, subtotal, tax, total, lines, currency="USD"):
    """A model tool_input; `lines` are (quantity, unit_price, total) triples
    (or None for the whole line_items value)."""
    rows = (
        lines
        if not isinstance(lines, list)
        else [
            {"description": leaf("item"), "quantity": leaf(q), "unit_price": leaf(u), "total": leaf(t)}
            for q, u, t in lines
        ]
    )
    return {
        "vendor": leaf("Acme"),
        "document_date": leaf("2025-01-01"),
        "line_items": leaf(rows),
        "subtotal": leaf(subtotal),
        "tax": leaf(tax),
        "total": leaf(total),
        "currency": leaf(currency),
    }


def run(**kwargs):
    return reconcile(ReconcileInput.from_tool_input(tool_input(**kwargs)))


def check_of(report, check_id):
    return next(check for check in report.checks if check.id == check_id)


def failing_ids(report):
    return [check.id for check in report.failing_checks]


def label_tool_input(fields):
    return {
        name: leaf(
            [{key: leaf(cell) for key, cell in row.items()} for row in value or []]
            if name == "line_items"
            else value
        )
        for name, value in fields.items()
    }


def artifact_tool_input(entry):
    """What the model returned for one run record: each field's `actual` and
    `confidence` (line item cells are already leaves)."""
    return {
        name: leaf(field["actual"], field["confidence"]) for name, field in entry["fields"].items()
    }


def scored(path):
    return [e for e in json.loads(path.read_text(encoding="utf-8"))["per_doc"] if e["status"] == "scored"]


def stored_field(name, value, confidence=0.99, *, action=None, corrected=None, reviewed=False):
    # line_items is stored as the array itself (app/extraction.py), not a leaf.
    return ExtractedField(
        field_name=name,
        value=value if name == "line_items" else leaf(value, confidence),
        confidence=confidence,
        needs_review=False,
        reviewed_at=datetime(2026, 1, 1, tzinfo=UTC) if reviewed else None,
        review_action=action,
        corrected_value=corrected,
    )


def stored_016(**overrides):
    """The staged 016 record as stored fields: lines 36.60 + 8.65, tax 3.05,
    total 48.30, and the subtotal read as 46.25."""
    rows = [
        {"description": leaf("a"), "quantity": leaf(4), "unit_price": leaf(9.15), "total": leaf(36.6)},
        {"description": leaf("b"), "quantity": leaf(1), "unit_price": leaf(8.65), "total": leaf(8.65)},
    ]
    fields = {
        "line_items": stored_field("line_items", rows),
        "subtotal": stored_field("subtotal", 46.25, 0.68),
        "tax": stored_field("tax", 3.05),
        "total": stored_field("total", 48.3),
        "currency": stored_field("currency", "USD"),
    }
    fields.update(overrides)
    return list(fields.values())


# ---- The committed data -----------------------------------------------------


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case.doc_id)
def test_every_label_passes(case) -> None:
    report = reconcile(ReconcileInput.from_tool_input(label_tool_input(case.fields)))

    assert report.status == STATUS_PASS
    assert failing_ids(report) == []
    assert report.suggestion is None
    assert report.rules_version == RULES_VERSION


@pytest.mark.parametrize(
    ("run_path", "entry"),
    [(path, entry) for path in SONNET_RUNS for entry in scored(path)],
    ids=lambda value: value["doc_id"] if isinstance(value, dict) else value.name[:16],
)
def test_every_sonnet_record_passes(run_path, entry) -> None:
    report = reconcile(ReconcileInput.from_tool_input(artifact_tool_input(entry)))

    assert report.status == STATUS_PASS, (run_path.name, entry["doc_id"], failing_ids(report))


def test_the_sonnet_runs_cover_the_whole_corpus() -> None:
    assert SONNET_RUNS
    assert max(len(scored(path)) for path in SONNET_RUNS) == len(load_cases()) == 25


def test_haiku_records_fail_exactly_where_the_numbers_disagree() -> None:
    failures = {}
    for entry in scored(HAIKU_RUN):
        report = reconcile(ReconcileInput.from_tool_input(artifact_tool_input(entry)))
        if report.status == STATUS_FAIL:
            failures[entry["doc_id"]] = failing_ids(report)

    assert failures == {
        # Three line totals disagree with quantity x unit price, and the lines no
        # longer add up to the (correct) subtotal.
        "018-dense-office-outfitters": ["line_math[3]", "line_math[5]", "line_math[7]", "lines_sum"],
        # Each line is internally consistent but the lines add up to 461.40
        # while the subtotal reads 5470.40 (the printed total is that subtotal plus tax).
        "019-dense-hardware-warehouse": ["lines_sum"],
        # Line 1 and line 5 were misread (2 -> 3 and 4 -> 2 units) with their totals.
        "020-dense-electronics-order": ["lines_sum"],
    }


def test_haiku_019_names_the_line_sum_and_the_printed_subtotal() -> None:
    entry = next(e for e in scored(HAIKU_RUN) if e["doc_id"].startswith("019"))
    report = reconcile(ReconcileInput.from_tool_input(artifact_tool_input(entry)))
    lines_sum = check_of(report, "lines_sum")

    assert (lines_sum.expected, lines_sum.actual) == (Decimal("461.40"), Decimal("5470.4"))
    assert "461.40" in lines_sum.detail
    assert "5470.40" in lines_sum.detail
    # Four wrong header amounts could each be the culprit: no single fix.
    assert report.suggestion is None


# ---- The staged 016 case -----------------------------------------------------


def test_staged_016_fails_two_equations_and_suggests_the_subtotal() -> None:
    staged = next(c for c in STAGED_REVIEW_CASES if c.doc_id.startswith("016"))
    assert (staged.field_name, staged.value) == ("subtotal", 46.25)

    report = reconcile(ReconcileInput.from_fields(stored_016()))

    assert report.status == STATUS_FAIL
    assert failing_ids(report) == ["lines_sum", "total_math"]
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("subtotal", Decimal("45.25"))
    assert report.suggestion.current == Decimal("46.25")
    assert report.suggestion.check_ids == ("lines_sum", "total_math")
    lines_sum = check_of(report, "lines_sum")
    assert (lines_sum.expected, lines_sum.actual, lines_sum.delta) == (
        Decimal("45.25"),
        Decimal("46.25"),
        Decimal("1.00"),
    )
    assert lines_sum.fields == ("line_items[0].total", "line_items[1].total", "subtotal")
    assert lines_sum.all_operands_verified is False


def test_applying_the_staged_suggestion_passes_everything() -> None:
    corrected = stored_016(
        subtotal=stored_field(
            "subtotal", 46.25, 0.68, action="corrected", corrected="45.25", reviewed=True
        )
    )

    report = reconcile(ReconcileInput.from_fields(corrected))

    assert report.status == STATUS_PASS
    subtotal = check_of(report, "lines_sum").rhs
    assert (subtotal.value, subtotal.source, subtotal.verified) == (Decimal("45.25"), "corrected", True)
    assert report.suggestion is None


def test_a_verified_field_never_becomes_the_suggestion() -> None:
    approved = stored_016(
        subtotal=stored_field("subtotal", 46.25, 0.68, action="approved", reviewed=True)
    )

    report = reconcile(ReconcileInput.from_fields(approved))

    # The reviewer froze the subtotal; no other single change fixes both checks.
    assert failing_ids(report) == ["lines_sum", "total_math"]
    assert report.suggestion is None
    assert check_of(report, "lines_sum").rhs.source == "approved"


def test_the_suggestion_can_name_a_line_total() -> None:
    report = run(subtotal=45.25, tax=3.05, total=48.3, lines=[(4, 9.15, 36.6), (1, 8.65, 9.65)])

    assert failing_ids(report) == ["line_math[1]", "lines_sum"]
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == (
        "line_items[1].total",
        Decimal("8.65"),
    )


def test_a_total_only_misread_gets_no_suggestion() -> None:
    # Either the tax or the total could be the misread number.
    report = run(subtotal=45.25, tax=3.05, total=49.3, lines=[(4, 9.15, 36.6), (1, 8.65, 8.65)])

    assert failing_ids(report) == ["total_math"]
    assert report.suggestion is None


def test_a_misread_total_with_no_tax_is_ambiguous() -> None:
    # The unread tax could be the missing 4.05 as easily as the total could be
    # wrong, so the engine does not overwrite a total that may be right.
    report = run(subtotal=45.25, tax=None, total=49.3, lines=[(4, 9.15, 36.6), (1, 8.65, 8.65)])

    assert failing_ids(report) == ["total_math"]
    assert report.suggestion is None


def test_a_reviewed_null_tax_leaves_the_total_as_the_only_candidate() -> None:
    fields = stored_016(
        subtotal=stored_field("subtotal", 45.25),
        tax=stored_field("tax", None, action="approved", reviewed=True),
        total=stored_field("total", 49.3),
    )

    report = reconcile(ReconcileInput.from_fields(fields))

    assert failing_ids(report) == ["total_math"]
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("total", Decimal("45.25"))


def test_a_discount_line_total_can_be_suggested_as_negative() -> None:
    report = run(subtotal=90, tax=9, total=99, lines=[(1, 100, 100), (1, -10, 10)])

    assert failing_ids(report) == ["line_math[1]", "lines_sum"]
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == (
        "line_items[1].total",
        Decimal("-10.00"),
    )
    assert report.suggestion.current == Decimal(10)


def test_a_return_line_total_can_be_suggested_as_negative() -> None:
    report = run(subtotal=5, tax=0, total=5, lines=[(2, 5, 10), (-1, 5, 5)])

    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == (
        "line_items[1].total",
        Decimal("-5.00"),
    )


def test_a_negative_header_amount_is_never_suggested() -> None:
    # The lines say the subtotal and total are -10, but a header amount below
    # zero is not offered as a fix.
    report = run(subtotal=-10, tax=0, total=5, lines=[(1, -10, -10)])

    assert failing_ids(report) == ["total_math", "non_negative:subtotal"]
    assert report.suggestion is None


def test_exact_solutions_that_disagree_within_the_tolerance_still_give_a_suggestion() -> None:
    # Two lines of 0.995 printed as 1.00: the line totals add up to 2.00, within
    # one cent of 1.99. The sum says the subtotal is 2.00, total_math says 1.99;
    # only 1.99 passes both.
    report = run(subtotal=1.59, tax=0.10, total=2.09, lines=[(1, 0.995, 1.0), (1, 0.995, 1.0)])

    assert failing_ids(report) == ["lines_sum", "total_math"]
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("subtotal", Decimal("1.99"))


def test_a_misread_line_total_is_suggested_from_the_line_and_the_sum() -> None:
    report = run(subtotal=10, tax=0, total=10, lines=[(2, 5, 10.03)])

    assert failing_ids(report) == ["line_math[0]", "lines_sum"]
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == (
        "line_items[0].total",
        Decimal("10.00"),
    )


# ---- Tax-inclusive documents ----------------------------------------------------

GROSS_LINES = [(1, 5.9, 5.9), (1, 6.8, 6.8), (1, 15.12, 15.12)]  # 27.82, of which tax 1.82


def test_a_gross_subtotal_passes_with_the_tax_included_assumption() -> None:
    report = run(subtotal=27.82, tax=1.82, total=27.82, lines=GROSS_LINES)

    assert report.status == STATUS_PASS
    assert len(report.assumptions) == 1
    assert "prices include tax" in report.assumptions[0]
    total_math = check_of(report, "total_math")
    assert total_math.status == STATUS_PASS
    assert total_math.formula == "subtotal = total (the total includes the tax)"
    included = total_math.terms[1]
    assert (included.coef, included.operand.path, included.operand.source) == (
        Decimal(0),
        "tax",
        "included",
    )
    # 1.82 on a net amount of 26.00.
    tax_rate = check_of(report, "tax_rate")
    assert (tax_rate.status, tax_rate.actual) == (STATUS_PASS, Decimal("0.0700"))
    assert tax_rate.formula == "0 <= tax / (total - tax) <= 0.30"


def test_a_gross_computed_subtotal_passes_too() -> None:
    report = run(subtotal=None, tax=1.82, total=27.82, lines=GROSS_LINES)

    assert report.status == STATUS_PASS
    assert [d.path for d in report.derived] == ["subtotal"]
    assert "prices include tax" in report.assumptions[0]


def test_gross_lines_with_a_net_subtotal_pass_with_the_assumption() -> None:
    report = run(subtotal=26.0, tax=1.82, total=27.82, lines=GROSS_LINES)

    assert report.status == STATUS_PASS
    assert len(report.assumptions) == 1
    assert "line prices include tax" in report.assumptions[0]
    lines_sum = check_of(report, "lines_sum")
    assert lines_sum.status == STATUS_PASS
    assert lines_sum.rhs.path == "total"
    assert lines_sum.terms[-1].operand.source == "included"
    # The rate is judged on the (net) subtotal here.
    assert check_of(report, "tax_rate").actual == Decimal("0.0700")


def test_the_tax_included_rules_do_not_hide_a_misread() -> None:
    gross_lines = [(1, 5.9, 5.9), (1, 6.8, 6.8), (1, 15.13, 15.13)]
    cases = {
        # The total is misread: neither reading holds.
        "gross subtotal, total misread": {"subtotal": 27.82, "tax": 1.82, "total": 27.9},
        "net subtotal, total misread": {"subtotal": 26.0, "tax": 1.82, "total": 27.9},
        # A line total is misread by a cent, so the lines no longer reach the total.
        "gross lines, one line off": {"subtotal": 26.0, "tax": 1.82, "total": 27.82, "lines": gross_lines},
        # The implied rate is 9.00 / 18.82, far beyond 30%.
        "gross subtotal, absurd tax": {"subtotal": 27.82, "tax": 9.0, "total": 27.82},
    }

    for name, kwargs in cases.items():
        report = run(**{"lines": GROSS_LINES, **kwargs})
        assert report.status == STATUS_FAIL, name
        assert report.assumptions == (), name


def test_a_document_that_already_passes_does_not_take_the_included_reading() -> None:
    case = next(c for c in load_cases() if c.doc_id.startswith("001"))
    report = reconcile(ReconcileInput.from_tool_input(label_tool_input(case.fields)))

    assert report.assumptions == ()
    assert all(op.source != "included" for check in report.checks for op in check.operands)


def test_a_tax_included_receipt_with_a_misread_line_still_gets_a_suggestion() -> None:
    # The gross-subtotal reading already holds, so the fix may lean on it: only
    # the line total is wrong.
    report = run(
        subtotal=27.82, tax=1.82, total=27.82, lines=[(1, 5.9, 5.9), (1, 6.8, 6.8), (1, 15.12, 15.21)]
    )

    assert "line_math[2]" in failing_ids(report)
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == (
        "line_items[2].total",
        Decimal("15.12"),
    )


# ---- Unreadable values are not replaced ---------------------------------------------


def test_an_unreadable_line_total_is_not_replaced_by_a_computed_one() -> None:
    report = run(subtotal=10, tax=1, total=11, lines=[(2, 3, "abc"), (1, 4, 4)])

    assert failing_ids(report) == ["format:line_items[0].total"]
    assert report.derived == ()
    assert check_of(report, "line_math[0]").status == STATUS_SKIPPED
    lines_sum = check_of(report, "lines_sum")
    assert lines_sum.status == STATUS_SKIPPED
    assert "line_items[0].total is unreadable" in lines_sum.detail


def test_a_currency_change_makes_line_totals_format_failures_not_computed() -> None:
    # Whole yen: 200.5 is a precision failure, and the product (200) must not
    # quietly stand in for it.
    report = run(subtotal=200, tax=0, total=200, lines=[(2, 100, 200.5)], currency="JPY")

    assert failing_ids(report) == ["format:line_items[0].total"]
    assert report.derived == ()
    assert check_of(report, "lines_sum").status == STATUS_SKIPPED


# ---- Rate and skipped-check details ---------------------------------------------------


def test_the_tax_rate_verdict_uses_the_rate_it_prints() -> None:
    # 3000.400 / 10000 = 0.30004, shown as 0.3000; 3000.600 shows 0.3001.
    shown_ok = run(subtotal=10000, tax=3000.4, total=13000.4, lines=[(1, 10000, 10000)], currency="BHD")
    shown_high = run(subtotal=10000, tax=3000.6, total=13000.6, lines=[(1, 10000, 10000)], currency="BHD")

    assert (shown_ok.status, check_of(shown_ok, "tax_rate").actual) == (STATUS_PASS, Decimal("0.3000"))
    assert failing_ids(shown_high) == ["tax_rate"]
    assert check_of(shown_high, "tax_rate").actual == Decimal("0.3001")


def test_total_math_without_a_total_does_not_assume_a_tax() -> None:
    report = run(subtotal=10, tax=None, total=None, lines=[(1, 10, 10)])

    total_math = check_of(report, "total_math")
    assert total_math.status == STATUS_SKIPPED
    assert total_math.detail == "Skipped because total is missing."
    assert report.assumptions == ()
    assert all(term.operand.source != "assumed" for term in total_math.terms)


# ---- Rules --------------------------------------------------------------------


def test_a_null_tax_counts_as_zero_and_says_so() -> None:
    case = next(c for c in load_cases() if c.doc_id.startswith("021"))
    report = reconcile(ReconcileInput.from_tool_input(label_tool_input(case.fields)))

    assert report.status == STATUS_PASS
    assert report.assumptions == ("no tax is shown, so subtotal + tax = total counts it as 0.",)
    tax = check_of(report, "total_math").terms[1].operand
    assert (tax.path, tax.value, tax.source) == ("tax", Decimal(0), "assumed")
    assert check_of(report, "tax_rate").status == STATUS_SKIPPED
    assert check_of(report, "non_negative:tax").status == STATUS_SKIPPED


def test_a_null_currency_records_the_two_digit_assumption() -> None:
    case = next(c for c in load_cases() if c.doc_id.startswith("023"))
    report = reconcile(ReconcileInput.from_tool_input(label_tool_input(case.fields)))

    assert report.status == STATUS_PASS
    assert report.currency is None
    assert report.minor_units == 2
    assert report.assumptions == ("no currency stated; assuming 2 decimal places",)


def test_an_unlisted_currency_code_is_assumed_two_digits() -> None:
    report = run(subtotal=5, tax=0, total=5, lines=[(1, 5, 5)], currency="ZZZ")

    assert (report.currency, report.minor_units) == ("ZZZ", 2)
    assert "not an ISO 4217 currency code" in report.assumptions[0]


def test_a_jpy_record_reconciles_in_whole_units() -> None:
    report = run(
        subtotal=3500, tax=350, total=3850, lines=[(2, 1000, 2000), (3, 500, 1500)], currency="JPY"
    )

    assert report.status == STATUS_PASS
    assert report.minor_units == 0
    assert check_of(report, "lines_sum").tolerance == Decimal(0)
    total_math = next(c for c in report_to_json(report)["checks"] if c["id"] == "total_math")
    assert total_math["actual"] == "3850"


def test_jpy_with_a_decimal_amount_is_a_format_failure() -> None:
    report = run(
        subtotal=3500, tax=350, total=3850.5, lines=[(2, 1000, 2000), (3, 500, 1500)], currency="JPY"
    )

    assert failing_ids(report) == ["format:total"]
    detail = check_of(report, "format:total").detail
    assert "3850.5" in detail
    assert "no decimal places" in detail
    assert check_of(report, "total_math").status == STATUS_SKIPPED
    assert "format:total" in check_of(report, "total_math").detail


def test_a_three_decimal_currency_keeps_all_three_digits() -> None:
    report = run(
        subtotal=10.505, tax=0.5, total=11.005, lines=[(2, 3.5, 7), (1, 3.505, 3.505)], currency="BHD"
    )

    assert report.status == STATUS_PASS
    assert report.minor_units == 3
    totals = {check.id: check for check in report.checks}
    assert totals["total_math"].expected == Decimal("11.005")


def test_a_rounded_product_gets_one_minor_unit_of_slack() -> None:
    # 3 x 0.333 = 0.999, printed as 1.00 (or 0.99): inside one cent.
    for printed in (1.0, 0.99):
        report = run(subtotal=printed, tax=0, total=printed, lines=[(3, 0.333, printed)])
        line = check_of(report, "line_math[0]")
        assert (line.status, line.tolerance) == (STATUS_PASS, Decimal("0.01"))
        assert check_of(report, "lines_sum").tolerance == Decimal("0.005")
    off = run(subtotal=1.02, tax=0, total=1.02, lines=[(3, 0.333, 1.02)])
    assert failing_ids(off) == ["line_math[0]"]


def test_an_exact_product_gets_no_slack() -> None:
    report = run(subtotal=10.01, tax=0, total=10.01, lines=[(2, 5, 10.01)])

    assert failing_ids(report) == ["line_math[0]"]
    assert check_of(report, "line_math[0]").tolerance == Decimal(0)
    # Fixing the line total would break the subtotal that matches it.
    assert report.suggestion is None


def test_a_negative_discount_line_is_allowed() -> None:
    report = run(
        subtotal=90, tax=9, total=99, lines=[(1, 100, 100), (1, -10, -10)],
    )

    assert report.status == STATUS_PASS
    assert check_of(report, "lines_sum").actual == Decimal(90)


def test_a_negative_quantity_is_a_return() -> None:
    report = run(subtotal=5, tax=0, total=5, lines=[(2, 5, 10), (-1, 5, -5)])

    assert report.status == STATUS_PASS


def test_negative_header_amounts_fail_non_negative() -> None:
    report = run(subtotal=-5, tax=0, total=-5, lines=[(1, -5, -5)])

    assert failing_ids(report) == ["non_negative:subtotal", "non_negative:total"]
    assert "negative" in check_of(report, "non_negative:total").detail


def test_tax_rate_bounds() -> None:
    high = run(subtotal=100, tax=31, total=131, lines=[(1, 100, 100)])
    edge = run(subtotal=100, tax=30, total=130, lines=[(1, 100, 100)])
    zero_subtotal = run(subtotal=0, tax=5, total=5, lines=[(1, 0, 0)])

    assert failing_ids(high) == ["tax_rate"]
    assert check_of(high, "tax_rate").actual == Decimal("0.3100")
    rate = next(c for c in report_to_json(high)["checks"] if c["id"] == "tax_rate")
    assert (rate["actual"], rate["expected"], rate["tolerance"]) == ("0.3100", None, None)
    assert edge.status == STATUS_PASS
    assert check_of(zero_subtotal, "tax_rate").status == STATUS_SKIPPED
    assert "subtotal is 0" in check_of(zero_subtotal, "tax_rate").detail


def test_the_tax_rate_is_reported_to_four_places() -> None:
    report = run(subtotal=3, tax=0.07, total=3.07, lines=[(1, 3, 3)])

    assert check_of(report, "tax_rate").actual == Decimal("0.0233")


# ---- lines_present (the old H3 carry-over) ---------------------------------------


@pytest.mark.parametrize("lines", ["not a list", {"0": 1}, [], None], ids=["string", "object", "empty", "null"])
def test_a_subtotal_without_usable_line_items_fails_lines_present(lines) -> None:
    report = run(subtotal=10, tax=1, total=11, lines=lines)

    assert failing_ids(report) == ["lines_present"]
    assert "no line items" in check_of(report, "lines_present").detail
    assert check_of(report, "lines_sum").status == STATUS_SKIPPED
    assert check_of(report, "total_math").status == STATUS_PASS


def test_lines_present_is_skipped_without_a_subtotal() -> None:
    report = run(subtotal=None, tax=None, total=11, lines=[])

    assert check_of(report, "lines_present").status == STATUS_SKIPPED
    assert report.status == STATUS_PASS  # only non_negative:total could run


# ---- Derived values -------------------------------------------------------------


def test_a_missing_subtotal_is_computed_from_the_line_totals() -> None:
    report = run(subtotal=None, tax=1, total=11, lines=[(2, 3, 6), (1, 4, 4)])

    assert report.status == STATUS_PASS
    assert [(d.path, d.value) for d in report.derived] == [("subtotal", Decimal(10))]
    lines_sum = check_of(report, "lines_sum")
    assert lines_sum.status == STATUS_SKIPPED
    assert lines_sum.rhs.source == "computed"
    assert check_of(report, "total_math").terms[0].operand.source == "computed"
    assert check_of(report, "total_math").status == STATUS_PASS


def test_a_missing_line_total_is_computed_and_used() -> None:
    report = run(subtotal=10, tax=1, total=11, lines=[(2, 3, 6), (1, 4, None)])

    assert report.status == STATUS_PASS
    assert [(d.path, d.value) for d in report.derived] == [("line_items[1].total", Decimal("4.00"))]
    assert check_of(report, "line_math[1]").status == STATUS_SKIPPED
    assert check_of(report, "lines_sum").status == STATUS_PASS


def test_a_line_with_nothing_to_compute_from_skips_the_sum() -> None:
    report = run(subtotal=10, tax=1, total=11, lines=[(2, 3, 6), (None, 4, None)])

    assert check_of(report, "line_math[1]").status == STATUS_SKIPPED
    lines_sum = check_of(report, "lines_sum")
    assert lines_sum.status == STATUS_SKIPPED
    assert "line_items[1].total is missing" in lines_sum.detail
    assert report.status == STATUS_PASS


def test_nothing_to_check_is_skipped() -> None:
    report = run(subtotal=None, tax=None, total=None, lines=None, currency=None)

    assert report.status == STATUS_SKIPPED
    assert failing_ids(report) == []


# ---- Format failures ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("$5.00", "not a plain number"),
        ("abc", "not a plain number"),
        (float("nan"), "not a finite number"),
        (float("inf"), "not a finite number"),
        (True, "is not an amount"),
        ([1], "is not an amount"),
        (10.005, "too many decimal places"),
        (1e12, "too large"),
        (2.5e22, "too large"),
    ],
)
def test_an_unusable_amount_is_a_format_failure(raw, message) -> None:
    report = run(subtotal=raw, tax=1, total=11, lines=[(1, 10, 10)])

    assert "format:subtotal" in failing_ids(report)
    assert message in check_of(report, "format:subtotal").detail
    for dependent in ("lines_sum", "total_math", "tax_rate", "non_negative:subtotal"):
        assert check_of(report, dependent).status == STATUS_SKIPPED
        assert "format:subtotal" in check_of(report, dependent).detail
    assert report.suggestion is None


def test_a_string_amount_the_parser_accepts_is_not_a_failure() -> None:
    report = run(subtotal="10.00", tax="1", total="11.00", lines=[("1", "10.00", "10.00")])

    assert report.status == STATUS_PASS


def test_quantity_and_unit_price_allow_six_decimals() -> None:
    fine = run(subtotal=0.33, tax=0, total=0.33, lines=[(0.5, 0.666667, 0.33)])
    too_fine = run(subtotal=0.33, tax=0, total=0.33, lines=[(0.5, 0.6666667, 0.33)])

    assert fine.status == STATUS_PASS
    assert failing_ids(too_fine) == ["format:line_items[0].unit_price"]
    assert check_of(too_fine, "line_math[0]").status == STATUS_SKIPPED


def test_a_garbage_line_cell_does_not_fail_the_others() -> None:
    report = run(subtotal=10, tax=1, total=11, lines=[(2, 3, 6), ("x", 4, 4)])

    assert failing_ids(report) == ["format:line_items[1].quantity"]
    assert check_of(report, "lines_sum").status == STATUS_PASS
    assert check_of(report, "line_math[0]").status == STATUS_PASS


def test_a_malformed_tool_input_is_nothing_to_check() -> None:
    for broken in (None, [], {"subtotal": 5}, {"subtotal": {"value": "x"}}):
        report = reconcile(ReconcileInput.from_tool_input(broken))

        assert report.status == STATUS_SKIPPED


# ---- Inputs from stored fields -------------------------------------------------------


def test_from_fields_lets_a_correction_win_and_marks_it_verified() -> None:
    fields = stored_016(
        total=stored_field("total", 99.0, action="corrected", corrected="48.30", reviewed=True)
    )

    report = reconcile(ReconcileInput.from_fields(fields))

    total = check_of(report, "total_math").rhs
    assert (total.value, total.source, total.verified) == (Decimal("48.30"), "corrected", True)
    assert check_of(report, "total_math").all_operands_verified is False
    assert check_of(report, "non_negative:total").all_operands_verified is True


def test_from_fields_treats_a_corrected_line_item_list_as_one_verified_field() -> None:
    fields = stored_016(
        line_items=stored_field(
            "line_items",
            [],
            action="corrected",
            reviewed=True,
            corrected=[
                {"description": "a", "quantity": "4", "unit_price": "9.15", "total": "36.60"},
                {"description": "b", "quantity": "1", "unit_price": "8.65", "total": "8.65"},
            ],
        )
    )

    report = reconcile(ReconcileInput.from_fields(fields))

    assert failing_ids(report) == ["lines_sum", "total_math"]
    cell = check_of(report, "lines_sum").terms[0].operand
    assert (cell.source, cell.verified) == ("corrected", True)
    # Only the subtotal is unverified now.
    assert report.suggestion is not None
    assert report.suggestion.path == "subtotal"


def test_from_fields_with_a_corrected_line_items_null_fails_lines_present() -> None:
    fields = stored_016(
        line_items=stored_field("line_items", [], action="corrected", corrected=None, reviewed=True)
    )

    report = reconcile(ReconcileInput.from_fields(fields))

    assert "lines_present" in failing_ids(report)


def test_from_fields_with_no_fields_is_skipped() -> None:
    assert reconcile(ReconcileInput.from_fields([])).status == STATUS_SKIPPED


# ---- JSON ------------------------------------------------------------------------------


def test_report_to_json_is_plain_json() -> None:
    report = reconcile(ReconcileInput.from_fields(stored_016()))

    payload = report_to_json(report)

    assert json.loads(json.dumps(payload)) == payload
    assert set(payload) == {
        "rules_version",
        "status",
        "currency",
        "minor_units",
        "assumptions",
        "derived",
        "checks",
        "suggestion",
    }
    assert payload["suggestion"] == {
        "path": "subtotal",
        "value": "45.25",
        "current": "46.25",
        "check_ids": ["lines_sum", "total_math"],
        "detail": "Setting subtotal to 45.25 fixes lines_sum, total_math and breaks no passing check.",
    }
    check = next(c for c in payload["checks"] if c["id"] == "line_math[0]")
    assert check["terms"] == [
        {
            "coef": "4",
            "operand": {
                "path": "line_items[0].unit_price",
                "value": "9.15",
                "source": "extracted",
                "verified": False,
            },
        }
    ]
    assert (check["expected"], check["actual"], check["delta"], check["tolerance"]) == (
        "36.60",
        "36.60",
        "0.00",
        "0.00",
    )
    assert set(check) == {
        "id",
        "kind",
        "formula",
        "terms",
        "rhs",
        "expected",
        "actual",
        "delta",
        "tolerance",
        "status",
        "detail",
        "fields",
        "all_operands_verified",
    }


def test_json_keeps_digits_beyond_the_minor_unit_visible() -> None:
    report = run(subtotal=1.0, tax=0, total=1.0, lines=[(3, 0.333, 1.0)])

    line = next(c for c in report_to_json(report)["checks"] if c["id"] == "line_math[0]")
    assert (line["expected"], line["delta"]) == ("0.999", "0.001")
    assert line["terms"][0]["coef"] == "3"
    assert line["terms"][0]["operand"]["value"] == "0.333"


def test_json_quantities_are_plain_strings() -> None:
    report = run(subtotal=5, tax=0, total=5, lines=[(2.5, 2, 5)])

    payload = report_to_json(report)

    line = next(c for c in payload["checks"] if c["id"] == "line_math[0]")
    assert line["terms"][0]["coef"] == "2.5"
    assert line["fields"][0] == "line_items[0].quantity"


def test_a_line_input_and_field_input_are_plain_values() -> None:
    inp = ReconcileInput(
        currency="USD",
        subtotal=FieldInput(Decimal(5)),
        total=FieldInput(Decimal(5)),
        line_items=(LineInput(Decimal(1), Decimal(5), Decimal(5)),),
    )

    assert reconcile(inp).status == STATUS_PASS
