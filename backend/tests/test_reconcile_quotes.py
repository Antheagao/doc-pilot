"""The quote:<path> checks and quote-derived suggestions in app/reconcile.py.

A v2 leaf is `{value, confidence, evidence: {page, quote}}`; these tests build
records out of such leaves, as the model returns them (`from_tool_input`) and as
they are stored (`from_fields`). The v1 behavior, with no evidence anywhere, is
pinned by the golden fixture in test_reconcile.py. No database, no network.
"""

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from v2_samples import v2_tool_input

from app.evals.dataset import load_cases
from app.extraction import build_extracted_fields
from app.extraction_fields import SCHEMA_V2
from app.models import ExtractedField
from app.reconcile import (
    KIND_QUOTE,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED,
    ReconcileInput,
    reconcile,
    report_to_json,
)
from app.schemas import CheckOut


def leaf(value, quote=None, *, evidence=True, confidence=0.99):
    """A leaf as the v2 model returns it. `evidence=False` is a v1 leaf; with
    `quote=None` the evidence object has no quote."""
    out = {"value": value, "confidence": confidence}
    if evidence:
        out["evidence"] = {"page": 1, "quote": quote}
    return out


def v2_input(*, subtotal, tax, total, lines=None, extra=None):
    """A model tool_input in USD. Each header amount is (value, quote) and each
    line a dict of cell -> (value, quote); a bare value is a leaf with no
    evidence. Without `lines` there is one line that adds up to the subtotal,
    so the presence and sum checks have nothing to say."""

    def cell(spec):
        return leaf(*spec) if isinstance(spec, tuple) else leaf(spec, evidence=False)

    if lines is None:
        printed = subtotal[0] if isinstance(subtotal, tuple) else subtotal
        lines = [{"quantity": 1, "unit_price": printed, "total": printed}]

    tool = {
        "currency": leaf("USD", evidence=False),
        "subtotal": cell(subtotal),
        "tax": cell(tax),
        "total": cell(total),
    }
    tool["line_items"] = leaf(
        [
            {"description": leaf("item", evidence=False)}
            | {key: cell(spec) for key, spec in row.items()}
            for row in lines
        ],
        evidence=False,
    )
    for name, spec in (extra or {}).items():
        tool[name] = leaf(spec, evidence=False) if name == "tax_lines" else cell(spec)
    return tool


def run(**kwargs):
    return reconcile(ReconcileInput.from_tool_input(v2_input(**kwargs)))


def check_of(report, check_id):
    return next(check for check in report.checks if check.id == check_id)


def check_ids(report):
    return [check.id for check in report.checks]


def failing_ids(report):
    return [check.id for check in report.failing_checks]


CONSISTENT = {"subtotal": 45.25, "tax": 3.05, "total": 48.3}


# ---- The check ---------------------------------------------------------------


def test_a_quote_that_shows_another_number_fails_its_own_check() -> None:
    report = run(subtotal=(45.25, "Subtotal $45.52"), tax=3.05, total=48.3)

    check = check_of(report, "quote:subtotal")
    assert (check.kind, check.status) == (KIND_QUOTE, STATUS_FAIL)
    assert check.fields == ("subtotal",)
    assert check.detail == "The value is 45.25, but its quote shows 45.52."
    assert check.all_operands_verified is False
    assert failing_ids(report) == ["quote:subtotal"]
    assert report.status == STATUS_FAIL


def test_a_quote_that_shows_the_value_passes_with_no_detail() -> None:
    report = run(subtotal=(45.25, "Subtotal $45.25"), tax=(3.05, "Tax 3.05"), total=(48.3, "TOTAL 48.30"))

    assert [check_of(report, f"quote:{name}").status for name in ("subtotal", "tax", "total")] == [
        STATUS_PASS
    ] * 3
    assert check_of(report, "quote:total").detail is None
    assert report.status == STATUS_PASS


def test_a_quote_with_no_number_fails() -> None:
    report = run(subtotal=(45.25, "Subtotal"), tax=3.05, total=48.3)

    check = check_of(report, "quote:subtotal")
    assert check.status == STATUS_FAIL
    assert check.detail == "The value is 45.25, but its quote contains no number."


@pytest.mark.parametrize("quote", [None, "", "  "])
def test_an_evidence_object_with_no_quote_gives_a_skipped_check(quote) -> None:
    report = run(subtotal=(45.25, quote), tax=3.05, total=48.3)

    check = check_of(report, "quote:subtotal")
    assert (check.status, check.detail) == (STATUS_SKIPPED, "The field has no quote to check.")
    assert report.status == STATUS_PASS


def test_a_v1_leaf_without_evidence_gets_no_quote_check() -> None:
    report = run(**CONSISTENT)

    assert not any(check_id.startswith("quote:") for check_id in check_ids(report))


def test_a_quote_that_is_not_text_counts_as_no_quote() -> None:
    tool = v2_input(subtotal=(45.25, None), tax=3.05, total=48.3)
    tool["subtotal"]["evidence"]["quote"] = 45.25

    report = reconcile(ReconcileInput.from_tool_input(tool))

    assert check_of(report, "quote:subtotal").status == STATUS_SKIPPED


def test_a_null_value_gets_no_quote_check() -> None:
    report = run(subtotal=(45.25, "45.25"), tax=3.05, total=48.3, extra={"discount": (None, "Discount")})

    assert "quote:discount" not in check_ids(report)
    assert "quote:subtotal" in check_ids(report)


def test_an_unreadable_value_has_its_format_check_and_no_quote_check() -> None:
    report = run(subtotal=("forty-five", "Subtotal forty-five"), tax=3.05, total=48.3)

    assert "format:subtotal" in failing_ids(report)
    assert "quote:subtotal" not in check_ids(report)


def test_every_v2_amount_that_carries_evidence_is_checked() -> None:
    extra = {
        name: (1, f"{name} 1.00")
        for name in ("discount", "shipping", "tip", "amount_paid", "amount_due")
    }
    report = run(subtotal=(10, "10.00"), tax=(0, "0.00"), total=(10, "10.00"), extra=extra)

    assert [i for i in check_ids(report) if i.startswith("quote:")] == [
        f"quote:{name}"
        for name in ("subtotal", "tax", "total", "discount", "shipping", "tip", "amount_paid", "amount_due")
    ]


def test_quote_checks_sit_after_the_format_checks_and_before_the_arithmetic() -> None:
    report = run(
        subtotal=("x", "x"),
        tax=(1, "1.00"),
        total=(2, "2.00"),
        lines=[{"total": (1, "1.00")}],
    )

    ids = check_ids(report)
    assert ids[:3] == ["format:subtotal", "quote:tax", "quote:total"]
    assert ids.index("quote:line_items[0].total") < ids.index("lines_present")
    assert ids.index("quote:total") < ids.index("total_math")


def test_the_check_is_in_the_stored_report_and_validates_as_served() -> None:
    report = run(subtotal=(45.25, "Subtotal $45.52"), tax=3.05, total=48.3)

    stored = next(c for c in report_to_json(report)["checks"] if c["id"] == "quote:subtotal")

    assert stored["kind"] == "quote"
    assert stored["status"] == "fail"
    assert stored["formula"] == "subtotal equals the number in its quote"
    assert stored["terms"][0]["operand"] == {
        "path": "subtotal",
        "value": "45.25",
        "source": "extracted",
        "verified": False,
    }
    assert (stored["rhs"], stored["expected"], stored["actual"], stored["delta"]) == (None,) * 4
    assert CheckOut.model_validate(stored).id == "quote:subtotal"


@pytest.mark.parametrize(
    ("value", "quote"),
    [(1, "Total $1,000"), (100, "Total 100,000"), (2.5, "Total $2,500")],
)
def test_a_round_thousand_in_the_quote_is_not_the_value_it_would_be_with_a_decimal_point(
    value, quote
) -> None:
    # Two decimal places cannot hold "1,000" read as 1.000, so the quote says
    # a thousand, and the value that is a thousandth of it is a misread.
    report = run(subtotal=(value, quote), tax=0, total=value)

    check = check_of(report, "quote:subtotal")
    assert check.status == STATUS_FAIL
    assert check.fields == ("subtotal",)


def test_the_same_quote_agrees_with_the_thousand_and_with_a_three_decimal_currency() -> None:
    assert check_of(run(subtotal=(1000, "Total $1,000"), tax=0, total=1000), "quote:subtotal").status == (
        STATUS_PASS
    )
    kwd = reconcile(
        ReconcileInput.from_tool_input(
            {**v2_input(subtotal=(1.234, "Total KWD 1,234"), tax=0, total=1.234), "currency": leaf("KWD", evidence=False)}
        )
    )
    assert check_of(kwd, "quote:subtotal").status == STATUS_PASS


def test_a_unit_price_keeps_both_readings_of_a_three_digit_fraction() -> None:
    report = run(
        subtotal=1.459,
        tax=0,
        total=1.459,
        lines=[{"quantity": 1, "unit_price": (1.459, "1,459 \u20ac/l"), "total": 1.459}],
    )

    assert check_of(report, "quote:line_items[0].unit_price").status == STATUS_PASS


@pytest.mark.parametrize(
    "quote", ["USD 1 234,56", "1\u00a0234,56 \u20ac", "1\u202f234,56 \u20ac", "CHF 1'234.56", "CHF 1\u2019234.56"]
)
def test_a_correct_amount_printed_with_space_or_apostrophe_grouping_passes(quote) -> None:
    report = run(subtotal=(1234.56, quote), tax=0, total=1234.56)

    assert check_of(report, "quote:subtotal").status == STATUS_PASS


@pytest.mark.parametrize("quote", ["Shipping: FREE", "Shipping ____", "Shipping -", "No shipping"])
def test_a_zero_said_in_words_is_a_skipped_check_not_a_failure(quote) -> None:
    report = run(
        subtotal=45.25, tax=3.05, total=48.3, extra={"shipping": (0, quote)}
    )
    # An unprinted shipping adds nothing, so a printed zero changes no arithmetic.
    check = check_of(report, "quote:shipping")
    assert (check.status, check.detail) == (STATUS_SKIPPED, "The value is 0 and its quote prints no number.")
    assert report.status == STATUS_PASS


def test_a_nonzero_value_whose_quote_prints_no_number_still_fails() -> None:
    report = run(subtotal=45.25, tax=3.05, total=48.3, extra={"shipping": (0, "FREE")})
    assert check_of(report, "quote:shipping").status == STATUS_SKIPPED

    report = run(subtotal=45.25, tax=3.05, total=53.3, extra={"shipping": (5, "FREE")})
    assert check_of(report, "quote:shipping").status == STATUS_FAIL


def test_an_evidence_key_set_to_null_is_a_leaf_with_no_quote_not_a_v1_leaf() -> None:
    tool = v2_input(subtotal=(45.25, "45.25"), tax=3.05, total=48.3)
    tool["subtotal"]["evidence"] = None

    report = reconcile(ReconcileInput.from_tool_input(tool))

    check = check_of(report, "quote:subtotal")
    assert (check.status, check.detail) == (STATUS_SKIPPED, "The field has no quote to check.")


def test_a_stored_leaf_with_null_evidence_gets_a_skipped_check_too() -> None:
    subtotal = stored("subtotal", 45.25)
    subtotal.value = {**subtotal.value, "evidence": None}

    report = reconcile(ReconcileInput.from_fields(stored_record(subtotal=subtotal)))

    assert check_of(report, "quote:subtotal").status == STATUS_SKIPPED


# ---- Line and tax line cells ------------------------------------------------


def test_each_line_cell_with_evidence_has_its_own_check() -> None:
    report = run(
        subtotal=45.25,
        tax=3.05,
        total=48.3,
        lines=[
            {"quantity": (4, "4 x"), "unit_price": (9.15, "9.15"), "total": (36.6, "36.60")},
            {"quantity": 1, "unit_price": (8.65, "8.56"), "total": (8.65, "8.65")},
        ],
    )

    ids = [i for i in check_ids(report) if i.startswith("quote:")]
    assert ids == [
        "quote:line_items[0].quantity",
        "quote:line_items[0].unit_price",
        "quote:line_items[0].total",
        "quote:line_items[1].unit_price",
        "quote:line_items[1].total",
    ]
    assert failing_ids(report) == ["quote:line_items[1].unit_price"]
    check = check_of(report, "quote:line_items[1].unit_price")
    assert check.fields == ("line_items[1].unit_price",)
    assert check.detail == "The value is 8.65, but its quote shows 8.56."


def test_a_quantity_is_judged_and_worded_as_a_plain_number() -> None:
    report = run(
        subtotal=45.25,
        tax=3.05,
        total=48.3,
        lines=[{"quantity": (2.5, "Qty 2.50 kg"), "unit_price": 18.1, "total": 45.25}],
    )

    assert check_of(report, "quote:line_items[0].quantity").status == STATUS_PASS
    wrong = run(
        subtotal=45.25,
        tax=3.05,
        total=48.3,
        lines=[{"quantity": (2.5, "Qty 3"), "unit_price": 18.1, "total": 45.25}],
    )
    assert check_of(wrong, "quote:line_items[0].quantity").detail == "The value is 2.5, but its quote shows 3."


def tax_lines_input(rows):
    return v2_input(
        subtotal=100,
        tax=(8.88, "Tax 8.88"),
        total=108.88,
        extra={
            "tax_lines": [
                {"label": leaf("Sales tax", evidence=False)}
                | {key: leaf(*spec) for key, spec in row.items()}
                for row in rows
            ]
        },
    )


def test_each_tax_line_cell_with_evidence_has_its_own_check() -> None:
    tool = tax_lines_input(
        [{"rate": ("8.875", "8.875%"), "base": (100, "100.00"), "amount": (8.88, "8.83")}]
    )
    tool["tax_lines"]["value"][0]["label"] = leaf("Sales tax", evidence=False)

    report = reconcile(ReconcileInput.from_tool_input(tool))

    ids = [i for i in check_ids(report) if i.startswith("quote:tax_lines")]
    assert ids == ["quote:tax_lines[0].rate", "quote:tax_lines[0].base", "quote:tax_lines[0].amount"]
    assert failing_ids(report) == ["quote:tax_lines[0].amount"]
    assert check_of(report, "quote:tax_lines[0].rate").status == STATUS_PASS


# ---- Stored fields ----------------------------------------------------------


def stored(name, value, *, quote=None, evidence=True, reviewed=False, action=None, corrected=None):
    if name in ("line_items", "tax_lines"):
        stored_value = value  # arrays are stored as the array itself
    else:
        stored_value = leaf(value, quote, evidence=evidence)
    return ExtractedField(
        field_name=name,
        value=stored_value,
        confidence=0.99,
        needs_review=False,
        reviewed_at=datetime(2026, 1, 1, tzinfo=UTC) if reviewed else None,
        review_action=action,
        corrected_value=corrected,
    )


def one_line(total):
    return [
        {
            "description": leaf("item", evidence=False),
            "quantity": leaf(1, evidence=False),
            "unit_price": leaf(total, evidence=False),
            "total": leaf(total, evidence=False),
        }
    ]


def stored_record(**overrides):
    fields = {
        "currency": stored("currency", "USD", evidence=False),
        "line_items": stored("line_items", one_line(45.25)),
        "subtotal": stored("subtotal", 45.25, quote="Subtotal $45.52"),
        "tax": stored("tax", 3.05, quote="Tax 3.05"),
        "total": stored("total", 48.3, quote="Total 48.30"),
    }
    fields.update(overrides)
    return list(fields.values())


def test_a_stored_leaf_is_checked_against_its_quote() -> None:
    report = reconcile(ReconcileInput.from_fields(stored_record()))

    assert failing_ids(report) == ["quote:subtotal"]
    assert check_of(report, "quote:tax").status == STATUS_PASS


@pytest.mark.parametrize(
    ("action", "corrected"), [("approved", None), ("corrected", "45.52")], ids=["approved", "corrected"]
)
def test_a_field_a_person_reviewed_gets_no_quote_check(action, corrected) -> None:
    subtotal = stored(
        "subtotal", 45.25, quote="Subtotal $45.52", reviewed=True, action=action, corrected=corrected
    )

    report = reconcile(ReconcileInput.from_fields(stored_record(subtotal=subtotal)))

    assert "quote:subtotal" not in check_ids(report)
    assert "quote:tax" in check_ids(report)


def test_stored_line_cells_keep_their_quotes_until_a_person_replaces_the_rows() -> None:
    rows = [
        {
            "description": leaf("a", evidence=False),
            "quantity": leaf(1, "1", evidence=True),
            "unit_price": leaf(45.25, "45.25"),
            "total": leaf(45.25, "45.52"),
        }
    ]
    report = reconcile(ReconcileInput.from_fields(stored_record(line_items=stored("line_items", rows))))

    assert "quote:line_items[0].total" in failing_ids(report)
    assert check_of(report, "quote:line_items[0].quantity").status == STATUS_PASS

    corrected = stored(
        "line_items",
        rows,
        reviewed=True,
        action="corrected",
        corrected=[{"description": "a", "quantity": "1", "unit_price": "45.25", "total": "45.25"}],
    )
    report = reconcile(ReconcileInput.from_fields(stored_record(line_items=corrected)))
    assert not any(i.startswith("quote:line_items") for i in check_ids(report))


def test_stored_tax_line_cells_keep_their_quotes() -> None:
    rows = [
        {
            "label": leaf("Tax", evidence=False),
            "rate": leaf("8", "8%"),
            "base": leaf(100, "100.00"),
            "amount": leaf(8, "9.00"),
        }
    ]

    report = reconcile(
        ReconcileInput.from_fields(
            [
                stored("currency", "USD", evidence=False),
                stored("line_items", one_line(100)),
                stored("subtotal", 100, quote="100.00"),
                stored("tax", 8, quote="8.00"),
                stored("total", 108, quote="108.00"),
                stored("tax_lines", rows),
            ]
        )
    )

    assert failing_ids(report) == ["quote:tax_lines[0].amount"]


# ---- Suggestions ------------------------------------------------------------


def test_a_quote_settles_an_equation_the_arithmetic_alone_cannot() -> None:
    # The total was misread as 49.30. subtotal + tax != total could be the tax
    # (4.05) or the total (48.30), so the arithmetic alone proposes nothing.
    blind = run(subtotal=45.25, tax=3.05, total=49.3)
    assert failing_ids(blind) == ["total_math"]
    assert blind.suggestion is None

    report = run(subtotal=45.25, tax=(3.05, "Tax 3.05"), total=(49.3, "TOTAL $48.30"))

    assert failing_ids(report) == ["quote:total", "total_math"]
    suggestion = report.suggestion
    assert (suggestion.path, suggestion.value, suggestion.current) == (
        "total",
        Decimal("48.30"),
        Decimal("49.3"),
    )
    # One number fixes the quote check and the arithmetic it caused.
    assert suggestion.check_ids == ("quote:total", "total_math")


def test_the_quote_suggestion_fixes_total_math_on_a_receipt_with_lines() -> None:
    report = run(
        subtotal=(46.25, "Subtotal $45.25"),
        tax=3.05,
        total=48.3,
        lines=[
            {"quantity": 4, "unit_price": 9.15, "total": 36.6},
            {"quantity": 1, "unit_price": 8.65, "total": 8.65},
        ],
    )

    assert failing_ids(report) == ["quote:subtotal", "lines_sum", "total_math"]
    assert (report.suggestion.path, report.suggestion.value) == ("subtotal", Decimal("45.25"))
    assert report.suggestion.check_ids == ("quote:subtotal", "lines_sum", "total_math")


def test_a_competing_arithmetic_fix_is_rejected_because_the_quote_still_fails() -> None:
    # Setting the tax to 4.05 would also satisfy subtotal + tax = total, but the
    # total's quote still disagrees with its value, so only the total survives.
    report = run(subtotal=45.25, tax=3.05, total=(49.3, "TOTAL $48.30"))

    assert report.suggestion.path == "total"


def test_a_quote_with_two_numbers_proposes_nothing_of_its_own() -> None:
    # Neither number is the one that makes the arithmetic work, and no other
    # change fixes the quote check, so there is no fix to offer.
    report = run(subtotal=45.25, tax=3.05, total=(49.3, "less 1.00 = 47.30"))

    assert "quote:total" in failing_ids(report)
    assert report.suggestion is None


def test_an_ambiguous_quote_proposes_nothing() -> None:
    report = run(subtotal=45.25, tax=3.05, total=(49.3, "Total 1,234"))

    assert "quote:total" in failing_ids(report)
    assert report.suggestion is None


def test_a_quote_with_no_number_proposes_nothing() -> None:
    report = run(subtotal=45.25, tax=3.05, total=(49.3, "Total"))

    assert "quote:total" in failing_ids(report)
    assert report.suggestion is None


def test_a_quote_never_suggests_a_quantity_or_a_rate() -> None:
    report = run(
        subtotal=45.25,
        tax=3.05,
        total=48.3,
        lines=[{"quantity": (3, "Qty 4"), "unit_price": 11.3125, "total": 45.25}],
    )

    assert "quote:line_items[0].quantity" in failing_ids(report)
    assert report.suggestion is None


def test_a_line_total_suggestion_keeps_the_sign_of_the_line() -> None:
    report = run(
        subtotal=40.25,
        tax=3,
        total=43.25,
        lines=[
            {"quantity": 1, "unit_price": 45.25, "total": 45.25},
            {"quantity": 1, "unit_price": -5, "total": (-5.5, "Discount (5.00)")},
        ],
    )

    assert failing_ids(report)[0] == "quote:line_items[1].total"
    assert (report.suggestion.path, report.suggestion.value) == ("line_items[1].total", Decimal("-5.00"))


def test_an_arithmetic_fix_survives_a_value_and_quote_that_misread_the_same_digit() -> None:
    # The model read 46.25 and quoted 46.25; the lines say 45.25. The quote
    # agrees with the wrong value, so only the arithmetic fails, and its fix
    # still stands: a person who accepted it would replace the quote.
    report = run(
        subtotal=(46.25, "Subtotal $46.25"),
        tax=3.05,
        total=48.3,
        lines=[
            {"quantity": 4, "unit_price": 9.15, "total": 36.6},
            {"quantity": 1, "unit_price": 8.65, "total": 8.65},
        ],
    )

    assert failing_ids(report) == ["lines_sum", "total_math"]
    assert (report.suggestion.path, report.suggestion.value) == ("subtotal", Decimal("45.25"))


def test_an_arithmetic_fix_is_rejected_when_another_quote_would_still_fail() -> None:
    # The tax quote disagrees with its value, and no change to the subtotal can
    # fix that, so the subtotal fix that repairs the arithmetic is not offered.
    report = run(
        subtotal=(46.25, "Subtotal $46.25"),
        tax=(3.05, "Tax 3.50"),
        total=48.3,
        lines=[
            {"quantity": 4, "unit_price": 9.15, "total": 36.6},
            {"quantity": 1, "unit_price": 8.65, "total": 8.65},
        ],
    )

    assert "quote:tax" in failing_ids(report)
    assert report.suggestion is None


def test_a_quote_number_is_not_suggested_for_a_field_no_equation_judges() -> None:
    # The amount paid is right; its quote is a reference number. With no amount
    # due there is no equation on it, so 12345.00 would be a guess.
    report = run(
        subtotal=45.25, tax=3.05, total=48.3, extra={"amount_paid": (48.3, "Paid ref 12345")}
    )

    assert failing_ids(report) == ["quote:amount_paid"]
    assert report.suggestion is None


def test_a_quote_number_is_suggested_once_an_equation_can_judge_the_field() -> None:
    report = run(
        subtotal=45.25,
        tax=3.05,
        total=48.3,
        extra={"amount_paid": (40.0, "Paid 48.30"), "amount_due": 0},
    )

    # total - amount_paid = amount_due fails at 8.30 and 48.30 settles it.
    assert "balance_math" in failing_ids(report)
    assert (report.suggestion.path, report.suggestion.value) == ("amount_paid", Decimal("48.30"))


def test_a_reviewed_field_is_never_suggested_even_by_its_quote() -> None:
    subtotal = stored("subtotal", 40, quote="Subtotal $45.30", reviewed=True, action="approved")

    report = reconcile(ReconcileInput.from_fields(stored_record(subtotal=subtotal)))

    assert report.suggestion is None


# ---- The committed labels, quoted the way a document prints them -------------


def quoted_label_tool_input(fields, quote_of):
    """A label as a v2 model would return it with a faithful quote on every
    amount: `quote_of(value)` is the printed text for one number."""

    def cell(value, evidence=True):
        out = {"value": value, "confidence": 1.0}
        if evidence and value is not None:
            out["evidence"] = {"page": 1, "quote": quote_of(value)}
        return out

    tool = {name: cell(value, evidence=name in ("subtotal", "tax", "total")) for name, value in fields.items()}
    tool["line_items"] = cell(
        [
            {
                "description": cell(row["description"], evidence=False),
                "quantity": cell(row["quantity"]),
                "unit_price": cell(row["unit_price"]),
                "total": cell(row["total"]),
            }
            for row in fields["line_items"]
        ],
        evidence=False,
    )
    return tool


def stripped(tool):
    """The same tool_input with every evidence object removed."""
    if isinstance(tool, dict):
        return {k: stripped(v) for k, v in tool.items() if k != "evidence"}
    if isinstance(tool, list):
        return [stripped(v) for v in tool]
    return tool


@pytest.mark.parametrize(
    "quote_of",
    [lambda v: f"${v:,.2f}", lambda v: f"{v:.2f}".replace(".", ","), lambda v: f"Total: {v}  (USD)"],
    ids=["us", "eu-comma", "plain"],
)
def test_faithful_quotes_never_flag_a_label_and_change_no_other_check(quote_of) -> None:
    for case in load_cases():
        tool = quoted_label_tool_input(case.fields, quote_of)

        with_quotes = reconcile(ReconcileInput.from_tool_input(tool))
        without = reconcile(ReconcileInput.from_tool_input(stripped(tool)))

        quote_checks = [c for c in with_quotes.checks if c.kind == KIND_QUOTE]
        assert quote_checks, case.doc_id
        assert {c.status for c in quote_checks} == {STATUS_PASS}, case.doc_id
        assert [c for c in with_quotes.checks if c.kind != KIND_QUOTE] == list(without.checks), case.doc_id
        assert with_quotes.status == without.status == STATUS_PASS


# ---- A persisted v2 extraction ------------------------------------------------


def persisted(tool_input):
    """The fields the worker stores for a v2 tool_input, as from_fields reads
    them."""
    return build_extracted_fields(uuid.uuid4(), tool_input, 0.8, SCHEMA_V2)


def test_a_persisted_v2_extraction_reconciles_and_runs_the_quote_checks() -> None:
    report = reconcile(ReconcileInput.from_fields(persisted(v2_tool_input())))

    assert report.status == STATUS_PASS
    quote_checks = [check for check in report.checks if check.kind == KIND_QUOTE]
    assert {check.id for check in quote_checks} >= {
        "quote:subtotal",
        "quote:discount",
        "quote:shipping",
        "quote:tax",
        "quote:total",
        "quote:amount_paid",
        "quote:amount_due",
        "quote:tax_lines[0].amount",
        "quote:line_items[0].total",
    }
    assert all(check.status == STATUS_PASS for check in quote_checks)
    # The stored tax line reads back with its cells, so the tax checks ran too.
    assert check_of(report, "tax_lines_sum").status == STATUS_PASS
    assert check_of(report, "tax_line_math[0]").status == STATUS_PASS


def test_a_persisted_and_a_direct_v2_input_reconcile_identically() -> None:
    tool_input = v2_tool_input()

    direct = reconcile(ReconcileInput.from_tool_input(tool_input))
    stored_report = reconcile(ReconcileInput.from_fields(persisted(tool_input)))

    assert report_to_json(stored_report) == report_to_json(direct)


def test_a_persisted_v2_misread_is_caught_by_its_quote_and_by_the_arithmetic() -> None:
    tool_input = v2_tool_input()
    tool_input["tax_lines"]["value"][0]["amount"]["value"] = 7.9  # quote still says 7.20
    tool_input["shipping"]["evidence"] = {"page": 1, "quote": "Shipping $6.00"}

    report = reconcile(ReconcileInput.from_fields(persisted(tool_input)))

    failing = failing_ids(report)
    assert "quote:tax_lines[0].amount" in failing
    assert "quote:shipping" in failing
    assert "tax_lines_sum" in failing


def test_a_persisted_leaf_whose_evidence_was_dropped_has_no_quote_check() -> None:
    tool_input = v2_tool_input()
    tool_input["subtotal"]["evidence"] = {"page": 0, "quote": "Subtotal $999.00"}  # dropped as bad

    report = reconcile(ReconcileInput.from_fields(persisted(tool_input)))

    assert check_of(report, "quote:subtotal").status == STATUS_SKIPPED
    assert report.status == STATUS_PASS
