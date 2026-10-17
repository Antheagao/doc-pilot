"""Reconciliation rules (app/reconcile.py): pure functions over the committed
eval labels and runs, plus hand-built records for each rule's edges. No
database and no network. tests/test_reconcile_properties.py holds the
generated cases."""

import dataclasses
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from app.demo_seed import STAGED_REVIEW_CASES
from app.evals import checks
from app.evals.dataset import EVALS_DIR, load_cases
from app.extraction_fields import ARRAY_CELLS, SCHEMA_V1, SCHEMAS
from app.models import ExtractedField
from app.reconcile import (
    RULES_VERSION,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED,
    FieldInput,
    LineInput,
    ReconcileInput,
    TaxLineInput,
    reconcile,
    report_to_json,
)

RESULTS_DIR = EVALS_DIR / "results"
GOLDEN_V1_REPORTS = Path(__file__).parent / "fixtures" / "rules_v1_reports.json"
GOLDEN_V1_EDGE_INPUTS = Path(__file__).parent / "fixtures" / "rules_v1_edge_inputs.json"
SONNET_RUNS = sorted(RESULTS_DIR.glob("*claude-sonnet*_extract_v1.json"))
HAIKU_RUN = next(RESULTS_DIR.glob("*claude-haiku*_extract_v1.json"))
SONNET_REPLAY = RESULTS_DIR / "20260731T064853Z_claude-sonnet-5_extract_v1.json"


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
            if name in ARRAY_CELLS and isinstance(value, list)
            else value
        )
        for name, value in fields.items()
    }


# Dataset v1 is documents 001-025. The committed runs and the rules_v1 golden
# fixture describe only those, and only their seven v1 fields.
V1_DOCUMENT_COUNT = 25


def v1_cases():
    return [case for case in load_cases() if int(case.doc_id[:3]) <= V1_DOCUMENT_COUNT]


def v1_fields(case):
    """A label's fields as dataset v1 had them: the seven v1 fields only."""
    return {name: case.fields[name] for name in SCHEMA_V1.field_names}


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
    assert max(len(scored(path)) for path in SONNET_RUNS) == len(v1_cases()) == V1_DOCUMENT_COUNT


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



# ---- rules_v1 compatibility ---------------------------------------------------


def current_v1_shaped_reports() -> dict[str, dict]:
    """The reports behind tests/fixtures/rules_v1_reports.json, recomputed:
    every dataset-v1 label (its seven v1 fields), every scored document of the committed runs, the staged 016
    record, and the hand-built edge shapes of rules_v1_edge_inputs.json (gross
    readings, computed and assumed values, each format failure, suggestions),
    all v1-shaped. The fixture was written by the rules_v1 engine (before
    rules_v2 existed) and is never regenerated."""
    reports = {}
    for case in v1_cases():
        report, _ = checks.replay(checks.label_tool_input(v1_fields(case)), 1.0, SCHEMA_V1)
        reports[f"label:{case.doc_id}"] = report_to_json(report)
    # Only the extract_v1 artifacts the fixture was frozen from: a later run
    # (extract_v3, with evidence) has no key in it and is not v1-shaped.
    for path in sorted(RESULTS_DIR.glob("*_extract_v1.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        schema = SCHEMAS[artifact["prompt_version"]]
        threshold = artifact["review_threshold"]
        for entry in scored(path):
            tool = artifact_tool_input(entry)
            reports[f"run:{path.stem}:{entry['doc_id']}"] = report_to_json(
                checks.replay(tool, threshold, schema)[0]
            )
            for staged in STAGED_REVIEW_CASES:
                if staged.doc_id == entry["doc_id"] and path.name == SONNET_REPLAY.name:
                    tool = {**tool, staged.field_name: leaf(staged.value, staged.confidence)}
                    reports[f"staged:{staged.doc_id}:{staged.field_name}"] = report_to_json(
                        checks.replay(tool, threshold, schema)[0]
                    )
    for case in json.loads(GOLDEN_V1_EDGE_INPUTS.read_text(encoding="utf-8")):
        reports[f"edge:{case['id']}"] = report_to_json(reconcile(edge_input(case)))
    return reports


def edge_input(case) -> ReconcileInput:
    """The v1-shaped record of one edge case in rules_v1_edge_inputs.json: a raw
    tool_input, or a list of stored fields (name, value, confidence, review
    action, correction, reviewed)."""
    if "tool_input" in case:
        return ReconcileInput.from_tool_input(case["tool_input"])
    return ReconcileInput.from_fields(
        [
            ExtractedField(
                field_name=spec["name"],
                value=spec["value"]
                if spec["name"] == "line_items"
                else leaf(spec["value"], spec["confidence"]),
                confidence=spec["confidence"],
                needs_review=False,
                reviewed_at=datetime(2026, 1, 1, tzinfo=UTC) if spec["reviewed"] else None,
                review_action=spec["action"],
                corrected_value=spec["corrected"],
            )
            for spec in case["fields"]
        ]
    )


def test_a_v1_shaped_record_keeps_every_rules_v1_report() -> None:
    frozen = json.loads(GOLDEN_V1_REPORTS.read_text(encoding="utf-8"))
    current = current_v1_shaped_reports()

    assert current.keys() == frozen.keys()
    assert {report["rules_version"] for report in frozen.values()} == {"rules_v1"}
    assert RULES_VERSION == "rules_v2"
    for key, report in current.items():
        assert {**report, "rules_version": "rules_v1"} == frozen[key], key


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


# ---- rules_v2: adjustments, tax lines, balance, dates ------------------------------------


def run_v2(*, lines=((1, 100, 100),), currency="USD", tax_lines=None, **header):
    """A record built directly. `header` are FieldInput values by name
    (subtotal, discount, due_date, ...); `tax_lines` are (rate, base, amount)
    triples, `rate` the printed percent as a string."""
    return reconcile(v2_input(lines=lines, currency=currency, tax_lines=tax_lines, **header))


def v2_input(*, lines=((1, 100, 100),), currency="USD", tax_lines=None, **header):
    return ReconcileInput(
        currency=currency,
        line_items=None if lines is None else tuple(LineInput(*line) for line in lines),
        tax_lines=None
        if tax_lines is None
        else tuple(TaxLineInput(f"Tax {n}", rate, base, amount) for n, (rate, base, amount) in enumerate(tax_lines, 1)),
        **{name: FieldInput(value) for name, value in header.items()},
    )


def v2_tool_input(**fields):
    """A tool_input with the v1 fields already filled in for a 100.00 document."""
    base = tool_input(subtotal=100, tax=0, total=100, lines=[(1, 100, 100)])
    return {**base, **{name: leaf(value) for name, value in fields.items()}}


# total_math with discount, shipping, and tip


def test_total_math_takes_the_discount_off_and_adds_shipping_and_tip() -> None:
    report = run_v2(subtotal=100, discount=10, shipping=8, tax=7.2, tip=5, total=110.2)

    total_math = check_of(report, "total_math")
    assert total_math.status == STATUS_PASS
    assert total_math.formula == "subtotal - discount + shipping + tax + tip = total"
    assert [(t.operand.path, t.coef) for t in total_math.terms] == [
        ("subtotal", Decimal(1)),
        ("discount", Decimal(-1)),
        ("shipping", Decimal(1)),
        ("tax", Decimal(1)),
        ("tip", Decimal(1)),
    ]
    assert total_math.fields == ("subtotal", "discount", "shipping", "tax", "tip", "total")
    assert report.status == STATUS_PASS
    assert report.assumptions == ()


def test_total_math_names_the_adjustments_when_it_fails() -> None:
    report = run_v2(subtotal=100, discount=10, shipping=8, tax=7.2, total=106.2)

    total_math = check_of(report, "total_math")
    assert failing_ids(report) == ["total_math"]
    assert total_math.detail == (
        "The subtotal 100.00 minus discount 10.00 plus shipping 8.00 plus tax 7.20 is 105.20, "
        "but the total is 106.20 (off by 1.00)."
    )


def test_an_absent_adjustment_adds_no_term_and_no_assumption() -> None:
    report = run_v2(subtotal=100, tax=7, shipping=3, total=110)

    total_math = check_of(report, "total_math")
    # Only the printed shipping joins; the discount and tip are not 0, they are not there.
    assert [t.operand.path for t in total_math.terms] == ["subtotal", "shipping", "tax"]
    assert total_math.formula == "subtotal + shipping + tax = total"
    assert report.assumptions == ()


def test_a_null_tax_with_a_discount_still_counts_as_zero_and_says_so() -> None:
    report = run_v2(subtotal=100, discount=10, tax=None, total=90)

    assert report.status == STATUS_PASS
    assert report.assumptions == ("no tax is shown, so subtotal + tax = total counts it as 0.",)
    assert check_of(report, "total_math").formula == "subtotal - discount + tax = total"


def test_an_unreadable_discount_skips_total_math_and_fails_its_format() -> None:
    report = run_v2(subtotal=100, discount="ten", tax=0, total=90)

    assert failing_ids(report) == ["format:discount"]
    total_math = check_of(report, "total_math")
    assert total_math.status == STATUS_SKIPPED
    assert "discount is unreadable" in total_math.detail
    assert check_of(report, "non_negative:discount").status == STATUS_SKIPPED


def test_the_gross_subtotal_reading_carries_the_adjustments() -> None:
    # A tax-included receipt with 5.00 shipping: 27.82 of items (1.82 of it tax)
    # plus 5.00 shipping.
    report = run_v2(
        lines=GROSS_LINES, subtotal=27.82, tax=1.82, shipping=5, total=32.82
    )

    assert report.status == STATUS_PASS
    total_math = check_of(report, "total_math")
    assert total_math.formula == "subtotal + shipping = total (the total includes the tax)"
    assert [(t.operand.path, t.coef, t.operand.source) for t in total_math.terms] == [
        ("subtotal", Decimal(1), "extracted"),
        ("shipping", Decimal(1), "extracted"),
        ("tax", Decimal(0), "included"),
    ]


def test_the_gross_lines_reading_carries_the_adjustments() -> None:
    report = run_v2(
        lines=GROSS_LINES, subtotal=26.0, tax=1.82, discount=2, tip=3, total=28.82
    )

    assert report.status == STATUS_PASS
    lines_sum = check_of(report, "lines_sum")
    assert lines_sum.formula == (
        "sum of line totals - discount + tip = total (the line prices include the tax)"
    )
    assert lines_sum.terms[-1].operand.source == "included"


# lines_sum with a discount


def test_a_subtotal_already_after_the_discount_passes_with_an_assumption() -> None:
    # Lines 100.00, discount 10.00, the printed subtotal is 90.00 (after the
    # discount), then 5.00 shipping and 7.20 tax.
    report = run_v2(subtotal=90, discount=10, shipping=5, tax=7.2, total=102.2)

    assert report.status == STATUS_PASS
    assert report.assumptions == (
        (
            "the subtotal is after the discount: the line totals less the discount make up the "
            "subtotal, and the total is built from that subtotal without taking the discount off again."
        ),
    )
    lines_sum = check_of(report, "lines_sum")
    assert lines_sum.formula == (
        "sum of line totals - discount = subtotal (the subtotal is after the discount)"
    )
    assert (lines_sum.expected, lines_sum.actual) == (Decimal(90), Decimal(90))
    total_math = check_of(report, "total_math")
    assert total_math.formula == "subtotal + shipping + tax = total (the subtotal is after the discount)"
    discount = next(t for t in total_math.terms if t.operand.path == "discount")
    assert (discount.coef, discount.operand.source) == (Decimal(0), "included")


def test_a_subtotal_that_fits_neither_discount_reading_still_fails() -> None:
    report = run_v2(subtotal=95, discount=10, shipping=5, tax=7.2, total=102.2)

    assert "lines_sum" in failing_ids(report)
    assert report.assumptions == ()
    assert check_of(report, "lines_sum").formula == "sum of line totals = subtotal"


def test_the_discount_reading_needs_the_total_to_follow_it_too() -> None:
    # Lines less the discount do reach the subtotal, but the total takes the
    # discount off a second time: that is not the reading, so the lines still
    # do not add up to the subtotal.
    report = run_v2(subtotal=90, discount=10, shipping=5, tax=7.2, total=92.2)

    assert failing_ids(report) == ["lines_sum"]
    assert report.assumptions == ()
    assert check_of(report, "lines_sum").formula == "sum of line totals = subtotal"


def test_the_discount_reading_is_not_tried_for_a_document_that_passes() -> None:
    report = run_v2(subtotal=100, discount=10, shipping=5, tax=7.2, total=102.2)

    assert report.status == STATUS_PASS
    assert report.assumptions == ()
    assert all(op.source != "included" for check in report.checks for op in check.operands)


# tax_lines_sum and tax derived from its lines


def test_the_tax_lines_add_up_to_the_tax() -> None:
    report = run_v2(
        subtotal=100, tax=10, total=110, tax_lines=[("6", 100, 6), ("4", 100, 4)]
    )

    assert report.status == STATUS_PASS
    tax_lines_sum = check_of(report, "tax_lines_sum")
    assert tax_lines_sum.status == STATUS_PASS
    assert tax_lines_sum.formula == "sum of tax line amounts = tax"
    assert tax_lines_sum.fields == ("tax_lines[0].amount", "tax_lines[1].amount", "tax")


def test_tax_lines_that_do_not_add_up_to_the_tax_fail() -> None:
    report = run_v2(subtotal=100, tax=10, total=110, tax_lines=[("6", 100, 6), ("4", 100, 3)])

    tax_lines_sum = check_of(report, "tax_lines_sum")
    assert tax_lines_sum.status == STATUS_FAIL
    assert tax_lines_sum.detail == (
        "The tax line amounts add up to 9.00, but the tax is 10.00 (off by 1.00)."
    )
    assert "tax_lines[1].amount" in tax_lines_sum.fields
    # The second amount alone explains the gap in both of its checks.
    assert failing_ids(report) == ["tax_lines_sum", "tax_line_math[1]"]
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("tax_lines[1].amount", Decimal(4))


def test_a_tax_that_is_not_printed_is_computed_from_its_lines() -> None:
    report = run_v2(subtotal=100, tax=None, total=106, tax_lines=[("4", 100, 4), ("2", 100, 2)])

    assert report.status == STATUS_PASS
    assert [(d.path, d.value, d.formula) for d in report.derived] == [
        ("tax", Decimal(6), "sum of tax line amounts")
    ]
    assert check_of(report, "tax_lines_sum").status == STATUS_SKIPPED
    assert "computed as the sum of the tax line amounts" in check_of(report, "tax_lines_sum").detail
    total_math = check_of(report, "total_math")
    tax = total_math.terms[1].operand
    assert (tax.path, tax.value, tax.source) == ("tax", Decimal(6), "computed")
    # The tax is not assumed to be 0: it was worked out.
    assert report.assumptions == ()
    assert total_math.fields == ("subtotal", "tax_lines[0].amount", "tax_lines[1].amount", "total")


def test_a_derived_tax_still_catches_a_wrong_total() -> None:
    report = run_v2(subtotal=100, tax=None, total=107, tax_lines=[("4", 100, 4), ("2", 100, 2)])

    assert failing_ids(report) == ["total_math"]


def test_an_unreadable_tax_line_amount_skips_the_sum_and_the_derivation() -> None:
    report = run_v2(subtotal=100, tax=None, total=100, tax_lines=[("4", 100, "four")])

    assert "format:tax_lines[0].amount" in failing_ids(report)
    assert check_of(report, "tax_lines_sum").status == STATUS_SKIPPED
    assert report.derived == ()


def test_no_tax_lines_means_no_tax_line_checks() -> None:
    for tax_lines in (None, []):
        report = run_v2(subtotal=100, tax=7, total=107, tax_lines=tax_lines)

        assert not [c for c in report.checks if "tax_line" in c.id]


# tax_line_math


def test_a_tax_line_is_its_rate_times_its_base() -> None:
    report = run_v2(subtotal=100, tax=8, total=108, tax_lines=[("8", 100, 8)])

    line = check_of(report, "tax_line_math[0]")
    assert line.status == STATUS_PASS
    assert line.formula == "tax line 1: rate x base = amount"
    assert [(t.coef, t.operand.path) for t in line.terms] == [(Decimal("0.08"), "tax_lines[0].base")]
    assert line.rhs.path == "tax_lines[0].amount"
    assert line.fields == ("tax_lines[0].rate", "tax_lines[0].base", "tax_lines[0].amount")
    assert line.tolerance == Decimal(0)


def test_a_tax_line_that_is_not_its_rate_times_its_base_fails() -> None:
    report = run_v2(subtotal=100, tax=9, total=109, tax_lines=[("8", 100, 9)])

    line = check_of(report, "tax_line_math[0]")
    assert failing_ids(report) == ["tax_line_math[0]"]
    assert line.detail == (
        "Tax line 1: 8% of the base 100.00 is 8.00, but the amount is 9.00 (off by 1.00)."
    )


def test_two_tax_lines_on_different_printed_bases() -> None:
    report = run_v2(
        lines=((1, 200, 200),),
        subtotal=200,
        tax=10.25,
        total=210.25,
        # 6% of 150.00 and 2.5% of 50.00: each line against its own base.
        tax_lines=[("6", 150, 9), ("2.5", 50, 1.25)],
    )

    assert report.status == STATUS_PASS
    assert report.assumptions == ()
    assert [c.id for c in report.checks if c.id.startswith("tax_line_math")] == [
        "tax_line_math[0]",
        "tax_line_math[1]",
    ]
    wrong = run_v2(
        lines=((1, 200, 200),),
        subtotal=200,
        tax=10.25,
        total=210.25,
        tax_lines=[("6", 150, 9), ("2.5", 150, 1.25)],
    )
    assert failing_ids(wrong) == ["tax_line_math[1]"]


def test_a_rounded_tax_line_gets_one_minor_unit_of_slack() -> None:
    # 8.875% of 100.00 is 8.875, printed as 8.88 or 8.87 but not 8.89.
    for printed in (8.88, 8.87):
        report = run_v2(subtotal=100, tax=printed, total=100 + printed, tax_lines=[("8.875", 100, printed)])
        line = check_of(report, "tax_line_math[0]")
        assert (line.status, line.tolerance) == (STATUS_PASS, Decimal("0.01"))
    off = run_v2(subtotal=100, tax=8.89, total=108.89, tax_lines=[("8.875", 100, 8.89)])
    assert failing_ids(off) == ["tax_line_math[0]"]


def test_an_exact_tax_line_product_gets_no_slack() -> None:
    report = run_v2(subtotal=100, tax=8.01, total=108.01, tax_lines=[("8", 100, 8.01)])

    assert failing_ids(report) == ["tax_line_math[0]"]
    assert check_of(report, "tax_line_math[0]").tolerance == Decimal(0)


def test_a_tax_line_without_a_base_is_checked_against_the_subtotal() -> None:
    report = run_v2(subtotal=100, tax=8, total=108, tax_lines=[("8", None, 8)])

    assert report.status == STATUS_PASS
    assert report.assumptions == (
        "tax line 1 shows no base amount, so its amount is checked against the subtotal.",
    )
    line = check_of(report, "tax_line_math[0]")
    assert line.terms[0].operand.path == "subtotal"


def test_a_tax_line_without_a_base_is_checked_against_the_subtotal_less_the_discount() -> None:
    # 8% of 100.00 - 10.00 = 7.20.
    report = run_v2(subtotal=100, discount=10, tax=7.2, total=97.2, tax_lines=[("8", None, 7.2)])

    assert report.status == STATUS_PASS
    assert report.assumptions == (
        (
            "tax line 1 shows no base amount, so its amount is checked against the subtotal "
            "less the discount."
        ),
    )
    base = check_of(report, "tax_line_math[0]").terms[0].operand
    assert (base.path, base.value, base.source) == ("tax_lines[0].base", Decimal(90), "computed")
    assert check_of(report, "tax_line_math[0]").fields == (
        "tax_lines[0].rate",
        "subtotal",
        "discount",
        "tax_lines[0].amount",
    )


def test_a_tax_line_without_a_base_falls_back_to_the_taxed_shipping_base() -> None:
    # Shipping is taxed too: 8% of 100.00 - 10.00 + 10.00 = 8.00, not 8% of 90.00.
    report = run_v2(
        subtotal=100, discount=10, shipping=10, tax=8, total=108, tax_lines=[("8", None, 8)]
    )

    assert report.status == STATUS_PASS
    assert report.assumptions == (
        (
            "tax line 1 shows no base amount, so its amount is checked against the subtotal "
            "less the discount plus shipping."
        ),
    )
    base = check_of(report, "tax_line_math[0]").terms[0].operand
    assert (base.value, base.source) == (Decimal(100), "computed")


def test_a_tax_line_that_fits_no_fallback_base_reports_the_first() -> None:
    report = run_v2(
        subtotal=100, discount=10, shipping=10, tax=9, total=109, tax_lines=[("8", None, 9)]
    )

    assert failing_ids(report) == ["tax_line_math[0]"]
    assert check_of(report, "tax_line_math[0]").detail == (
        "Tax line 1: 8% of the subtotal less the discount 90.00 is 7.20, but the amount is 9.00 "
        "(off by 1.80)."
    )


def test_a_printed_base_is_never_replaced_by_a_fallback() -> None:
    # The printed base 50.00 is wrong for 8.00; the subtotal would fit, but the
    # printed number is the one judged.
    report = run_v2(subtotal=100, tax=8, total=108, tax_lines=[("8", 50, 8)])

    assert failing_ids(report) == ["tax_line_math[0]"]
    assert report.assumptions == ()


def test_a_tax_line_without_a_rate_is_skipped() -> None:
    report = run_v2(subtotal=100, tax=5, total=105, tax_lines=[(None, None, 5)])

    line = check_of(report, "tax_line_math[0]")
    assert line.status == STATUS_SKIPPED
    assert "tax_lines[0].rate is missing" in line.detail
    assert line.terms[0].coef is None
    assert report.status == STATUS_PASS


def test_a_tax_line_without_a_base_and_a_subtotal_is_skipped() -> None:
    report = run_v2(subtotal=None, tax=8, total=108, tax_lines=[("8", None, 8)], lines=None)

    line = check_of(report, "tax_line_math[0]")
    assert line.status == STATUS_SKIPPED
    assert "subtotal is missing" in line.detail
    assert report.assumptions == ()


def test_an_unreadable_rate_is_a_format_failure() -> None:
    report = run_v2(subtotal=100, tax=8, total=108, tax_lines=[("8%", 100, 8)])

    assert "format:tax_lines[0].rate" in failing_ids(report)
    assert check_of(report, "tax_line_math[0]").status == STATUS_SKIPPED


# balance_math


def test_the_balance_is_the_total_less_what_was_paid() -> None:
    report = run_v2(subtotal=100, tax=0, total=100, amount_paid=40, amount_due=60)

    balance = check_of(report, "balance_math")
    assert balance.status == STATUS_PASS
    assert balance.formula == "total - amount_paid = amount_due"
    assert balance.fields == ("total", "amount_paid", "amount_due")
    assert report.assumptions == ()


def test_a_wrong_amount_due_fails_the_balance() -> None:
    report = run_v2(subtotal=100, tax=0, total=100, amount_paid=40, amount_due=70)

    assert failing_ids(report) == ["balance_math"]
    assert check_of(report, "balance_math").detail == (
        "The total 100.00 less the amount paid 40.00 is 60.00, but the amount due is 70.00 "
        "(off by 10.00)."
    )


def test_no_amount_paid_beside_an_amount_due_counts_as_zero_and_says_so() -> None:
    passing = run_v2(subtotal=100, tax=0, total=100, amount_due=100)
    failing = run_v2(subtotal=100, tax=0, total=100, amount_due=60)

    assert passing.status == STATUS_PASS
    assert passing.assumptions == (
        "no payment is shown, so total - amount paid = amount due counts it as 0.",
    )
    paid = check_of(passing, "balance_math").terms[1].operand
    assert (paid.path, paid.value, paid.source) == ("amount_paid", Decimal(0), "assumed")
    assert failing_ids(failing) == ["balance_math"]
    assert "(no payment is printed, so it counts as 0)" in check_of(failing, "balance_math").detail
    # Nothing is made of a payment that was never there.
    assert "non_negative:amount_paid" not in [c.id for c in passing.checks]


def test_a_missing_amount_due_is_computed() -> None:
    report = run_v2(subtotal=100, tax=0, total=100, amount_paid=40)

    assert report.status == STATUS_PASS
    assert [(d.path, d.value, d.formula) for d in report.derived] == [
        ("amount_due", Decimal(60), "total - amount paid")
    ]
    balance = check_of(report, "balance_math")
    assert balance.status == STATUS_SKIPPED
    assert balance.rhs.source == "computed"
    assert balance.fields == ("total", "amount_paid")


def test_the_balance_is_not_checked_without_a_payment_or_an_amount_due() -> None:
    report = run_v2(subtotal=100, tax=0, total=100)

    assert "balance_math" not in [c.id for c in report.checks]
    assert report.derived == ()


def test_a_balance_without_a_total_is_skipped() -> None:
    report = run_v2(subtotal=100, tax=0, total=None, amount_paid=40, amount_due=60)

    balance = check_of(report, "balance_math")
    assert balance.status == STATUS_SKIPPED
    assert "total is missing" in balance.detail


def test_a_balance_that_the_paid_or_due_amount_could_explain_gets_no_suggestion() -> None:
    report = run_v2(subtotal=100, tax=0, total=100, amount_paid=40, amount_due=70)

    # Amount paid 30 and amount due 60 each fix it, and the total is pinned by
    # total_math: two single fixes, so neither is offered.
    assert report.suggestion is None
    fixed = run_v2(subtotal=100, tax=0, total=100, amount_paid=40, amount_due=60)
    assert fixed.status == STATUS_PASS


def test_a_misread_amount_due_is_suggested_when_the_payment_is_reviewed() -> None:
    inp = v2_input(subtotal=100, tax=0, total=100, amount_paid=40, amount_due=70)
    reviewed = dataclasses.replace(inp, amount_paid=FieldInput(40, "approved", True))

    report = reconcile(reviewed)

    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("amount_due", Decimal(60))


# due_date_order


def test_a_due_date_on_or_after_the_document_date_passes() -> None:
    for due in ("2025-03-01", "2025-02-01"):
        report = run_v2(subtotal=100, tax=0, total=100, document_date="2025-02-01", due_date=due)

        order = check_of(report, "due_date_order")
        assert order.status == STATUS_PASS
        assert order.formula == "due_date >= document_date"
        assert order.fields == ("document_date", "due_date")


def test_a_due_date_before_the_document_date_fails() -> None:
    report = run_v2(subtotal=100, tax=0, total=100, document_date="2025-02-01", due_date="2025-01-15")

    assert failing_ids(report) == ["due_date_order"]
    assert check_of(report, "due_date_order").detail == (
        "The due date 2025-01-15 is before the document date 2025-02-01."
    )


def test_the_due_date_order_is_emitted_only_when_a_due_date_is_printed() -> None:
    report = run_v2(subtotal=100, tax=0, total=100, document_date="2025-02-01")

    assert "due_date_order" not in [c.id for c in report.checks]


@pytest.mark.parametrize(
    ("document_date", "due_date", "reason"),
    [
        (None, "2025-03-01", "document_date is missing"),
        ("2025-02-01", "next week", "due_date is not a date (use YYYY-MM-DD)"),
        ("01/02/2025", "2025-03-01", "document_date is not a date (use YYYY-MM-DD)"),
        ("2025-02-30", "2025-03-01", "document_date is not a date (use YYYY-MM-DD)"),
    ],
)
def test_a_date_that_cannot_be_read_skips_the_order_with_the_reason(
    document_date, due_date, reason
) -> None:
    report = run_v2(subtotal=100, tax=0, total=100, document_date=document_date, due_date=due_date)

    order = check_of(report, "due_date_order")
    assert order.status == STATUS_SKIPPED
    assert order.detail == f"Skipped because {reason}."
    assert report.status == STATUS_PASS
    # Dates have no format: check; they would change v1 reports.
    assert not [c for c in report.checks if c.id.startswith("format:")]


# non_negative for the new amounts


@pytest.mark.parametrize("name", ["discount", "shipping", "tip", "amount_paid"])
def test_the_new_amounts_must_not_be_negative_when_present(name) -> None:
    ok = run_v2(subtotal=100, tax=0, total=100, **{name: 0})
    negative = run_v2(subtotal=100, tax=0, total=100, **{name: -5})

    assert check_of(ok, f"non_negative:{name}").status == STATUS_PASS
    assert f"non_negative:{name}" in failing_ids(negative)
    assert "is negative" in check_of(negative, f"non_negative:{name}").detail


def test_amount_due_may_be_negative() -> None:
    report = run_v2(subtotal=100, tax=0, total=100, amount_paid=120, amount_due=-20)

    assert report.status == STATUS_PASS
    assert "non_negative:amount_due" not in [c.id for c in report.checks]


def test_an_absent_new_amount_has_no_non_negative_check() -> None:
    report = run_v2(subtotal=100, tax=0, total=100)

    assert [c.id for c in report.checks if c.id.startswith("non_negative")] == [
        "non_negative:subtotal",
        "non_negative:tax",
        "non_negative:total",
    ]


# the suggestion over the new amounts


def test_a_misread_discount_is_suggested_back() -> None:
    # Truth: 100.00 - 10.00 + 8.00 shipping + 7.20 tax (8% of 90.00) = 105.20.
    report = run_v2(
        subtotal=100, discount=15, shipping=8, tax=7.2, total=105.2, tax_lines=[("8", None, 7.2)]
    )

    assert failing_ids(report) == ["tax_line_math[0]", "total_math"]
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("discount", Decimal(10))
    assert report.suggestion.current == Decimal(15)
    assert report.suggestion.check_ids == ("tax_line_math[0]", "total_math")


def test_a_total_math_failure_that_several_amounts_explain_gets_no_suggestion() -> None:
    report = run_v2(
        subtotal=100, discount=10, shipping=18, tax=7.2, total=107.2, tax_lines=[("8", 90, 7.2)]
    )

    # Shipping 10, discount 18, and total 115.20 each fix total_math alone.
    assert failing_ids(report) == ["total_math"]
    assert report.suggestion is None


def test_a_misread_tax_line_amount_is_suggested_back() -> None:
    report = run_v2(subtotal=100, tax=10, total=110, tax_lines=[("6", 100, 6), ("4", 100, 5)])

    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("tax_lines[1].amount", Decimal(4))


def test_a_reviewed_tax_line_field_is_never_suggested() -> None:
    inp = v2_input(subtotal=100, tax=10, total=110, tax_lines=[("6", 100, 6), ("4", 100, 5)])
    reviewed = dataclasses.replace(
        inp,
        tax_lines=tuple(
            dataclasses.replace(row, source="approved", verified=True) for row in inp.tax_lines
        ),
    )

    report = reconcile(reviewed)

    assert "tax_lines_sum" in failing_ids(report)
    assert report.suggestion is None


# a tax printed once is held twice: as the tax and as its only tax line


def test_a_misread_tax_that_reaches_its_only_tax_line_does_not_blame_the_total() -> None:
    # The printed tax is 8.00; read as 9.00 in both places. The two agree, so only
    # total_math fails, and changing the total would be the one fix that works.
    report = run_v2(subtotal=100, tax=9, total=108, tax_lines=[(None, None, 9)])

    assert failing_ids(report) == ["total_math"]
    assert check_of(report, "tax_lines_sum").status == STATUS_PASS
    assert report.suggestion is None


def test_the_same_misread_without_tax_lines_is_ambiguous_too() -> None:
    # What rules_v1 says about this record: the tax or the total, so no suggestion.
    assert run_v2(subtotal=100, tax=9, total=108).suggestion is None


def test_a_tax_that_only_the_pair_can_explain_is_fixed_through_the_header_field() -> None:
    # 9.00 read as 90.00 in both places: the total (109.00) would have to become
    # 190.00 and the rate would still be absurd, so the pair is the only fix.
    report = run_v2(subtotal=100, tax=90, total=109, tax_lines=[(None, None, 90)])

    assert failing_ids(report) == ["total_math", "tax_rate"]
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("tax", Decimal(9))
    assert report.suggestion.current == Decimal(90)
    assert report.suggestion.detail.endswith(
        "The only tax line's amount is the same printed number and needs the same value."
    )


def test_a_tax_line_with_a_printed_rate_and_base_is_not_the_taxs_twin() -> None:
    # tax_line_math holds the amount to 9% of 100.00 on its own, so the line is
    # its own reading: the misread fails three checks and nothing is suggested.
    report = run_v2(subtotal=100, tax=90, total=109, tax_lines=[("9", 100, 90)])

    assert failing_ids(report) == ["tax_line_math[0]", "total_math", "tax_rate"]
    assert report.suggestion is None


def test_a_rate_only_tax_line_is_the_taxs_twin_when_its_check_is_loose() -> None:
    # Printed 144.78 (7.25% of 1997.00 is 144.7825), read as 144.79 in both
    # places: the line check allows a minor unit, so only total_math fails, and
    # the freight (84.99) must not be blamed.
    report = run_v2(
        lines=((1, 1997, 1997),),
        subtotal=1997,
        shipping=85,
        tax=144.79,
        total=2226.78,
        tax_lines=[("7.25", None, 144.79)],
    )

    assert failing_ids(report) == ["total_math"]
    assert check_of(report, "tax_line_math[0]").status == STATUS_PASS
    assert report.suggestion is None


def test_a_tax_and_a_tax_line_that_disagree_are_not_one_reading() -> None:
    report = run_v2(subtotal=100, tax=8, total=108, tax_lines=[(None, None, 9)])

    assert failing_ids(report) == ["tax_lines_sum"]
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("tax_lines[0].amount", Decimal(8))
    assert report.suggestion.detail == "Setting tax_lines[0].amount to 8.00 fixes tax_lines_sum and breaks no passing check."


def test_a_reviewed_tax_leaves_the_tax_line_as_the_next_suggestion() -> None:
    # A reviewer who applied the pair's suggestion corrected only the tax: the
    # stored tax line is still the misread, and is suggested on its own field.
    misread = v2_input(subtotal=100, tax=8, total=108, tax_lines=[(None, None, 9)])
    corrected = dataclasses.replace(misread, tax=FieldInput(8, "corrected", True))

    report = reconcile(corrected)

    assert failing_ids(report) == ["tax_lines_sum"]
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("tax_lines[0].amount", Decimal(8))


def test_a_reviewed_tax_line_field_is_not_changed_with_the_tax() -> None:
    # A person's word stands: the approved tax line is not the pair's second
    # cell, so the only change left is the total, which is the intended answer.
    misread = v2_input(subtotal=100, tax=9, total=108, tax_lines=[(None, None, 9)])
    reviewed = dataclasses.replace(
        misread,
        tax_lines=tuple(
            dataclasses.replace(row, source="approved", verified=True) for row in misread.tax_lines
        ),
    )

    report = reconcile(reviewed)

    assert report.suggestion is not None
    assert report.suggestion.path == "total"


# a tax or subtotal nobody printed is computed from cells that could be what is off


def test_a_tax_computed_from_one_misread_tax_line_does_not_blame_the_total() -> None:
    # extract_v2 leaves tax null when only tax lines are printed. The printed 8.00
    # read as 9.00 makes the computed tax 9.00; the total (108.00, right) and the
    # line amount are equally able to be the misread, so nothing is suggested.
    report = run_v2(subtotal=100, tax=None, total=108, tax_lines=[(None, None, 9)])

    assert failing_ids(report) == ["total_math"]
    assert [d.path for d in report.derived] == ["tax"]
    assert report.suggestion is None


def test_a_tax_computed_from_two_base_less_tax_lines_with_one_misread_has_no_suggestion() -> None:
    report = run_v2(
        subtotal=100, tax=None, total=108, tax_lines=[(None, None, 5), (None, None, 4)]
    )

    assert failing_ids(report) == ["total_math"]
    assert report.suggestion is None


def test_a_misread_total_beside_a_computed_tax_is_ambiguous_too() -> None:
    report = run_v2(subtotal=100, tax=None, total=109, tax_lines=[(None, None, 8)])

    assert failing_ids(report) == ["total_math"]
    assert report.suggestion is None


def test_a_computed_tax_that_only_one_tax_line_can_explain_is_fixed_on_that_line() -> None:
    # 8.00 read as 80.00: the total would have to become 180.00, and the rate of
    # the computed tax would still be absurd.
    report = run_v2(subtotal=100, tax=None, total=108, tax_lines=[(None, None, 80)])

    assert failing_ids(report) == ["total_math", "tax_rate"]
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("tax_lines[0].amount", Decimal(8))
    assert report.suggestion.current == Decimal(80)


def test_a_subtotal_computed_from_line_totals_with_no_factors_does_not_blame_the_total() -> None:
    # No quantity or unit price to check a line against: the lines (51 + 50) and
    # the total (108 less tax 8) disagree, and either side could be the misread.
    report = run_v2(
        lines=((None, None, 51), (None, None, 50)), subtotal=None, tax=8, total=108
    )

    assert failing_ids(report) == ["total_math"]
    assert report.suggestion is None


def test_a_computed_amount_is_never_proposed_itself() -> None:
    # A tax read from reviewed tax lines has no candidate input to move.
    inp = v2_input(subtotal=100, tax=None, total=108, tax_lines=[(None, None, 9)])
    reviewed = dataclasses.replace(
        inp,
        tax_lines=tuple(
            dataclasses.replace(row, source="approved", verified=True) for row in inp.tax_lines
        ),
    )

    report = reconcile(reviewed)

    assert failing_ids(report) == ["total_math"]
    assert report.suggestion is not None
    assert report.suggestion.path == "total"


# inputs


def test_from_tool_input_reads_the_new_fields() -> None:
    tool = v2_tool_input(
        discount=10,
        shipping=8,
        tip=2,
        amount_paid=20,
        amount_due=80,
        due_date="2025-03-01",
        tax_lines=[
            {
                "label": leaf("State"),
                "rate": leaf("8.875"),
                "base": leaf(100.0),
                "amount": leaf(8.88),
            }
        ],
    )

    inp = ReconcileInput.from_tool_input(tool)

    assert [inp.discount.value, inp.shipping.value, inp.tip.value] == [10, 8, 2]
    assert [inp.amount_paid.value, inp.amount_due.value] == [20, 80]
    assert (inp.document_date.value, inp.due_date.value) == ("2025-01-01", "2025-03-01")
    assert inp.tax_lines == (TaxLineInput("State", "8.875", 100.0, 8.88),)


def test_from_tool_input_leaves_absent_fields_absent_not_zero() -> None:
    inp = ReconcileInput.from_tool_input(tool_input(subtotal=5, tax=0, total=5, lines=[(1, 5, 5)]))

    for name in ("discount", "shipping", "tip", "amount_paid", "amount_due", "due_date"):
        assert getattr(inp, name) == FieldInput(None)
    assert inp.tax_lines is None


def test_from_tool_input_degrades_a_malformed_tax_line_to_an_empty_row() -> None:
    tool = v2_tool_input(tax_lines=["not a row", {"rate": "oops"}])

    inp = ReconcileInput.from_tool_input(tool)

    assert inp.tax_lines == (TaxLineInput(), TaxLineInput())


def stored_tax_lines(rows, *, action=None, corrected=None, reviewed=False):
    """The tax_lines field stored the way line_items is: the array itself, a
    leaf per cell."""
    return ExtractedField(
        field_name="tax_lines",
        value=[{key: leaf(cell) for key, cell in row.items()} for row in rows],
        confidence=0.99,
        needs_review=False,
        reviewed_at=datetime(2026, 1, 1, tzinfo=UTC) if reviewed else None,
        review_action=action,
        corrected_value=corrected,
    )


def stored_100(**extra):
    fields = {
        "line_items": stored_field(
            "line_items",
            [{"description": leaf("a"), "quantity": leaf(1), "unit_price": leaf(100.0), "total": leaf(100.0)}],
        ),
        "subtotal": stored_field("subtotal", 100.0),
        "tax": stored_field("tax", 10.0),
        "total": stored_field("total", 110.0),
        "currency": stored_field("currency", "USD"),
    }
    fields.update(extra)
    return list(fields.values())


def test_from_fields_reads_the_new_fields_and_the_tax_lines() -> None:
    rows = [
        {"label": "State", "rate": "6", "base": 100.0, "amount": 6.0},
        {"label": "County", "rate": "4", "base": 100.0, "amount": 4.0},
    ]
    fields = stored_100(
        discount=stored_field("discount", 0),
        due_date=stored_field("due_date", "2025-03-01"),
        document_date=stored_field("document_date", "2025-02-01"),
        tax_lines=stored_tax_lines(rows),
    )

    inp = ReconcileInput.from_fields(fields)

    assert inp.discount == FieldInput(0, "extracted", False)
    assert (inp.document_date.value, inp.due_date.value) == ("2025-02-01", "2025-03-01")
    assert inp.tax_lines == (
        TaxLineInput("State", "6", 100.0, 6.0),
        TaxLineInput("County", "4", 100.0, 4.0),
    )
    assert reconcile(inp).status == STATUS_PASS


def test_from_fields_with_no_new_fields_leaves_them_absent() -> None:
    inp = ReconcileInput.from_fields(stored_100())

    for name in ("discount", "shipping", "tip", "amount_paid", "amount_due", "due_date"):
        assert getattr(inp, name) == FieldInput()
    assert inp.tax_lines is None


def test_from_fields_lets_a_corrected_tax_lines_field_win_and_marks_it_verified() -> None:
    wrong = [{"label": "State", "rate": "6", "base": 100.0, "amount": 9.0}]
    fixed = [{"label": "State", "rate": "10", "base": "100.00", "amount": "10.00"}]
    fields = stored_100(
        tax_lines=stored_tax_lines(wrong, action="corrected", corrected=fixed, reviewed=True)
    )

    report = reconcile(ReconcileInput.from_fields(fields))

    assert report.status == STATUS_PASS
    amount = check_of(report, "tax_lines_sum").terms[0].operand
    assert (amount.value, amount.source, amount.verified) == (Decimal(10), "corrected", True)


def test_from_fields_treats_a_tax_lines_field_that_is_not_a_list_as_absent() -> None:
    broken = ExtractedField(
        field_name="tax_lines",
        value={"value": None, "confidence": 0.5},
        confidence=0.5,
        needs_review=False,
        reviewed_at=None,
        review_action=None,
        corrected_value=None,
    )

    assert ReconcileInput.from_fields(stored_100(tax_lines=broken)).tax_lines is None


# v1 shape


V2_ONLY_CHECKS = ("tax_lines_sum", "tax_line_math", "balance_math", "due_date_order")


def test_a_v1_shaped_record_emits_no_v2_check_or_derived_value() -> None:
    report = run(subtotal=100, tax=7, total=107, lines=[(1, 100, 100)])

    assert not [c for c in report.checks if c.id.startswith(V2_ONLY_CHECKS)]
    assert not [c for c in report.checks if c.id.split(":")[-1] in _V2_AMOUNTS]
    assert report.derived == ()
    total_math = check_of(report, "total_math")
    assert total_math.formula == "subtotal + tax = total"
    assert [t.operand.path for t in total_math.terms] == ["subtotal", "tax"]


_V2_AMOUNTS = ("discount", "shipping", "tip", "amount_paid", "amount_due")


# JSON


def test_report_json_prints_iso_dates_and_the_rate_as_a_plain_coefficient() -> None:
    report = run_v2(
        subtotal=100,
        tax=8.88,
        total=108.88,
        document_date="2025-02-01",
        due_date="2025-03-01",
        tax_lines=[("8.875", 100, 8.88)],
    )

    payload = report_to_json(report)

    assert json.loads(json.dumps(payload)) == payload
    assert payload["rules_version"] == "rules_v2"
    order = next(c for c in payload["checks"] if c["id"] == "due_date_order")
    assert [t["operand"]["value"] for t in order["terms"]] == ["2025-02-01", "2025-03-01"]
    assert order["actual"] is None
    line = next(c for c in payload["checks"] if c["id"] == "tax_line_math[0]")
    assert line["terms"][0]["coef"] == "0.08875"
    assert (line["expected"], line["actual"], line["tolerance"]) == ("8.875", "8.88", "0.01")
    assert line["fields"][0] == "tax_lines[0].rate"


# ---- rules_v2: readings and fallbacks ---------------------------------------------------


def test_a_rate_only_tax_line_is_checked_on_the_net_amount_when_the_subtotal_includes_the_tax() -> None:
    # 27.82 of gross items, 1.82 of it tax: 7% of the net 26.00.
    report = run_v2(
        lines=GROSS_LINES, subtotal=27.82, tax=1.82, total=27.82, tax_lines=[("7", None, 1.82)]
    )

    assert report.status == STATUS_PASS
    assert report.suggestion is None
    assert report.assumptions[-1] == (
        "tax line 1 shows no base amount, so its amount is checked against the net amount "
        "(subtotal less the tax)."
    )
    base = check_of(report, "tax_line_math[0]").terms[0].operand
    assert (base.path, base.value, base.source) == ("tax_lines[0].base", Decimal(26), "computed")
    assert check_of(report, "tax_line_math[0]").fields == (
        "tax_lines[0].rate",
        "subtotal",
        "tax",
        "tax_lines[0].amount",
    )


def test_a_rate_only_tax_line_that_misses_the_net_amount_fails_under_the_gross_reading() -> None:
    # The true total 108 was read as 100, which the gross reading accepts; 8% of
    # the net 92.00 is 7.36, not 8.00, so the tax line is what gives it away.
    report = run_v2(subtotal=100, tax=8, total=100, tax_lines=[("8", None, 8)])

    assert check_of(report, "total_math").formula == "subtotal = total (the total includes the tax)"
    line = check_of(report, "tax_line_math[0]")
    assert line.status == STATUS_FAIL
    assert line.detail == (
        "Tax line 1: 8% of the net amount (subtotal less the tax) 92.00 is 7.36, but the amount "
        "is 8.00 (off by 0.64)."
    )


def test_the_net_amount_base_also_takes_the_discount_and_shipping() -> None:
    # Items 108.00 including 8.00 tax, less 10.00 discount, plus 10.00 shipping,
    # make 108.00; shipping is taxed too: 8% of 108 - 8 - 10 + 10 = 100.00.
    report = run_v2(
        lines=((1, 108, 108),),
        subtotal=108,
        discount=10,
        shipping=10,
        tax=8,
        total=108,
        tax_lines=[("8", None, 8)],
    )

    assert report.status == STATUS_PASS
    assert report.assumptions[-1] == (
        "tax line 1 shows no base amount, so its amount is checked against the net amount "
        "(subtotal less the tax) less the discount plus shipping."
    )


def test_a_single_misread_does_not_pass_on_the_shipping_base_and_get_a_wrong_fix() -> None:
    # Tax 8.00 is a misread of 7.20 (in the tax and in its only tax line), but the
    # shipping-inclusive base (100.00) happens to fit it. The total 107.20 is
    # right: it is not changed, and the fix is the tax pair, which passes on the
    # first base (8% of 90.00).
    report = run_v2(
        subtotal=100, discount=10, shipping=10, tax=8, total=107.2, tax_lines=[("8", None, 8)]
    )

    assert failing_ids(report) == ["total_math"]
    assert check_of(report, "tax_line_math[0]").status == STATUS_PASS
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("tax", Decimal("7.2"))


def test_a_misread_amount_alone_does_not_pass_on_the_shipping_base_and_get_a_wrong_fix() -> None:
    # The same document with two tax lines, so the amount is not the tax's twin:
    # the passing second base must not make the misread a fix of the total.
    report = run_v2(
        subtotal=100,
        discount=10,
        shipping=10,
        tax=8,
        total=107.2,
        tax_lines=[("4", None, 4), ("4", None, 4)],
    )

    assert failing_ids(report) == ["total_math"]
    assert report.suggestion is None


def test_a_document_that_passes_on_the_first_base_keeps_its_suggestion() -> None:
    report = run_v2(
        subtotal=100, discount=15, shipping=8, tax=7.2, total=105.2, tax_lines=[("8", None, 7.2)]
    )

    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("discount", Decimal(10))


def test_a_total_misread_as_the_subtotal_with_its_adjustments_passes_as_tax_included() -> None:
    # Documented blind spot: the true total is 116.00 (100 + 8 tax + 8 shipping).
    # Read as 108.00 it fits "subtotal + shipping = total, tax included".
    report = run_v2(subtotal=100, tax=8, shipping=8, total=108)

    assert report.status == STATUS_PASS
    assert "prices include tax" in report.assumptions[0]
    assert check_of(report, "total_math").formula == (
        "subtotal + shipping = total (the total includes the tax)"
    )


def test_an_overpaid_documents_misread_amount_due_gets_its_negative_fix() -> None:
    inp = v2_input(subtotal=100, tax=0, total=100, amount_paid=120, amount_due=-30)
    reviewed = dataclasses.replace(inp, amount_paid=FieldInput(120, "approved", True))

    report = reconcile(reviewed)

    assert failing_ids(report) == ["balance_math"]
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("amount_due", Decimal(-20))
    assert report.suggestion.current == Decimal(-30)


def test_a_negative_header_amount_other_than_the_amount_due_is_never_suggested() -> None:
    # Shipping would have to be -5 to balance this document; it is not offered.
    report = run_v2(subtotal=100, tax=0, shipping=15, total=95)
    reviewed = dataclasses.replace(
        v2_input(subtotal=100, tax=0, shipping=15, total=95),
        subtotal=FieldInput(100, "approved", True),
        total=FieldInput(95, "approved", True),
    )

    assert failing_ids(report) == ["total_math"]
    assert reconcile(reviewed).suggestion is None
