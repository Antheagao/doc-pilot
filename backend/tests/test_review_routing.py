"""Review routing (app/review_routing.py): which fields a person has to look at.

The first half is the pure `route` table over hand-built stored fields; the
second half runs `refresh_reconciliation` against the test database. The
staged 016 record is the running example: lines 36.60 + 8.65 = 45.25, tax 3.05,
total 48.30, and a subtotal misread as 46.25.
"""

import dataclasses
import re
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.extraction import build_extracted_fields
from app.extraction_fields import SCHEMA_V1
from app.models import (
    REVIEW_REASON_CONFIDENCE,
    Document,
    ExtractedField,
    Extraction,
    Reconciliation,
    reconciliation_reason,
)
from app.reconcile import ReconcileInput, Suggestion, reconcile, report_to_json
from app.review_routing import (
    NOT_FLAGGED,
    Routing,
    failing_checks_for,
    field_name_of,
    refresh_reconciliation,
    route,
    suggestion_for,
)

THRESHOLD = 0.8
REASON_PATTERN = re.compile(r"^(confidence|reconciliation:.+)$")
FLAGGED_LINES_SUM = Routing(True, "reconciliation:lines_sum")
FLAGGED_TOTAL_MATH = Routing(True, "reconciliation:total_math")


def leaf(value, confidence=0.99):
    return {"value": value, "confidence": confidence}


def stored(
    name,
    value,
    confidence=0.99,
    *,
    flag: Routing = NOT_FLAGGED,
    action=None,
    corrected=None,
):
    """A stored field. `action` marks it reviewed ("approved" or "corrected");
    `flag` is the routing it holds before the call under test."""
    return ExtractedField(
        field_name=name,
        value=value if name == "line_items" else leaf(value, confidence),
        confidence=confidence,
        needs_review=flag.needs_review,
        review_reason=flag.review_reason,
        reviewed_at=datetime(2026, 1, 1, tzinfo=UTC) if action else None,
        review_action=action,
        corrected_value=corrected,
    )


LINES_016 = [
    {"description": leaf("a"), "quantity": leaf(4), "unit_price": leaf(9.15), "total": leaf(36.6)},
    {"description": leaf("b"), "quantity": leaf(1), "unit_price": leaf(8.65), "total": leaf(8.65)},
]


def record_016(**overrides) -> list[ExtractedField]:
    fields = {
        "line_items": stored("line_items", LINES_016),
        "subtotal": stored("subtotal", 46.25, 0.68),
        "tax": stored("tax", 3.05),
        "total": stored("total", 48.3),
        "currency": stored("currency", "USD"),
    }
    fields.update(overrides)
    return list(fields.values())


def routed(fields: list[ExtractedField]) -> dict[str, Routing]:
    report = reconcile(ReconcileInput.from_fields(fields))
    return route(fields, report, THRESHOLD)


def flagged(table: dict[str, Routing]) -> dict[str, str]:
    return {name: r.review_reason for name, r in table.items() if r.needs_review}


# ---- The 016 scenarios ------------------------------------------------------


def test_the_misread_subtotal_alone_is_flagged_by_the_arithmetic() -> None:
    fields = record_016()
    report = reconcile(ReconcileInput.from_fields(fields))
    assert [c.id for c in report.failing_checks] == ["lines_sum", "total_math"]

    table = route(fields, report, THRESHOLD)

    # The suggestion names the subtotal, so the line items, tax, and total are
    # not worth a person's time yet. Low confidence does not outrank the check.
    assert flagged(table) == {"subtotal": "reconciliation:lines_sum"}


def test_a_correct_subtotal_leaves_nothing_flagged() -> None:
    corrected = stored(
        "subtotal", 46.25, 0.68, flag=FLAGGED_LINES_SUM, action="corrected", corrected="45.25"
    )
    fields = record_016(subtotal=corrected)
    report = reconcile(ReconcileInput.from_fields(fields))
    assert report.status == "pass"

    table = route(fields, report, THRESHOLD)

    # Only the reviewed subtotal keeps its flag (frozen; the queue skips it).
    assert flagged(table) == {"subtotal": "reconciliation:lines_sum"}
    assert {n: r for n, r in table.items() if n != "subtotal"} == {
        n: NOT_FLAGGED for n in ("line_items", "tax", "total", "currency")
    }


def test_approving_the_wrong_subtotal_flags_every_other_operand() -> None:
    approved = stored("subtotal", 46.25, 0.68, flag=FLAGGED_LINES_SUM, action="approved")
    fields = record_016(subtotal=approved)
    report = reconcile(ReconcileInput.from_fields(fields))
    assert [c.id for c in report.failing_checks] == ["lines_sum", "total_math"]
    assert report.suggestion is None

    table = route(fields, report, THRESHOLD)

    assert flagged(table) == {
        "subtotal": "reconciliation:lines_sum",  # frozen
        "line_items": "reconciliation:lines_sum",
        "tax": "reconciliation:total_math",
        "total": "reconciliation:total_math",
    }


def test_correcting_the_culprit_clears_the_other_operands() -> None:
    # A total that disagrees with subtotal + tax: either could be the misread,
    # so all three operands are flagged.
    ambiguous = [
        stored("line_items", LINES_016),
        stored("subtotal", 45.25),
        stored("tax", 3.05, 0.5),
        stored("total", 49.3),
        stored("currency", "USD"),
    ]
    assert flagged(routed(ambiguous)) == {
        "subtotal": "reconciliation:total_math",
        "tax": "reconciliation:total_math",
        "total": "reconciliation:total_math",
    }

    # The reviewer corrects the total. The subtotal and tax were flagged only
    # because of the disagreement: the subtotal goes, the tax is still doubted
    # by the model (confidence 0.5) and keeps that reason.
    after = [
        stored("line_items", LINES_016),
        stored("subtotal", 45.25, flag=FLAGGED_TOTAL_MATH),
        stored("tax", 3.05, 0.5, flag=FLAGGED_TOTAL_MATH),
        stored("total", 49.3, flag=FLAGGED_TOTAL_MATH, action="corrected", corrected="48.30"),
        stored("currency", "USD"),
    ]
    assert reconcile(ReconcileInput.from_fields(after)).status == "pass"
    table = routed(after)
    assert table["subtotal"] == NOT_FLAGGED
    assert table["tax"] == Routing(True, REVIEW_REASON_CONFIDENCE)
    assert table["total"] == FLAGGED_TOTAL_MATH  # frozen


def test_a_reviewed_field_is_never_flagged_again() -> None:
    # Everything but the total is reviewed, and the total contradicts them.
    fields = record_016(
        subtotal=stored("subtotal", 45.25, action="approved"),
        line_items=stored("line_items", LINES_016, action="approved"),
        tax=stored("tax", 3.05, action="approved"),
        total=stored("total", 49.3),
    )
    table = routed(fields)
    assert flagged(table) == {"total": "reconciliation:total_math"}

    # Now the total is reviewed too, still wrong: nothing changes, nothing is flagged.
    fields[-2] = stored("total", 49.3, action="approved")
    table = routed(fields)
    assert flagged(table) == {}
    assert table["total"] == NOT_FLAGGED


def test_a_reviewed_field_keeps_the_flag_it_had() -> None:
    reviewed_low = stored(
        "tax", 3.05, 0.2, flag=Routing(True, REVIEW_REASON_CONFIDENCE), action="approved"
    )
    reviewed_quiet = stored("vendor", "Acme", 0.2, action="approved")
    fields = record_016(tax=reviewed_low)
    fields.append(reviewed_quiet)
    fields[1] = stored("subtotal", 45.25)

    table = routed(fields)

    assert table["tax"] == Routing(True, REVIEW_REASON_CONFIDENCE)
    assert table["vendor"] == NOT_FLAGGED  # reviewed and never flagged: still not


def test_low_confidence_still_flags_when_the_arithmetic_passes() -> None:
    fields = record_016(
        subtotal=stored("subtotal", 45.25),
        tax=stored("tax", 3.05, 0.5),
        total=stored("total", 48.3, 0.8),  # equal to the threshold is not below it
    )
    fields.append(stored("vendor", "Acme", 0.79))

    table = routed(fields)

    assert flagged(table) == {"tax": "confidence", "vendor": "confidence"}


def test_a_suspect_gets_the_first_failing_check_that_names_it() -> None:
    # No tax: total_math cannot tell the tax from the total, so no suggestion.
    fields = [
        stored("line_items", LINES_016),
        stored("subtotal", 45.25),
        stored("tax", None),
        stored("total", 49.3),
        stored("currency", "USD"),
    ]
    report = reconcile(ReconcileInput.from_fields(fields))
    assert [c.id for c in report.failing_checks] == ["total_math"]
    assert report.suggestion is None

    table = route(fields, report, THRESHOLD)

    assert flagged(table) == {
        "subtotal": "reconciliation:total_math",
        "tax": "reconciliation:total_math",
        "total": "reconciliation:total_math",
    }


def test_a_line_total_suggestion_flags_the_line_items_field() -> None:
    rows = [dict(row) for row in LINES_016]
    rows[1] = {**rows[1], "total": leaf(9.65)}
    fields = record_016(subtotal=stored("subtotal", 45.25), line_items=stored("line_items", rows))
    report = reconcile(ReconcileInput.from_fields(fields))
    assert report.suggestion is not None and report.suggestion.path == "line_items[1].total"

    table = route(fields, report, THRESHOLD)

    assert flagged(table) == {"line_items": "reconciliation:line_math[1]"}


def test_a_computed_subtotal_routes_to_the_line_items_not_the_absent_field() -> None:
    # No printed subtotal and no quantities or prices: the subtotal is the sum
    # of the line totals, 36.60 + 9.65 = 46.25, and 46.25 + 3.05 is not 48.30.
    rows = [
        {"description": leaf("a"), "quantity": leaf(None), "unit_price": leaf(None), "total": leaf(36.6)},
        {"description": leaf("b"), "quantity": leaf(None), "unit_price": leaf(None), "total": leaf(9.65)},
    ]
    fields = [
        stored("line_items", rows),
        stored("subtotal", None),
        stored("tax", 3.05),
        stored("total", 48.3),
        stored("currency", "USD"),
    ]
    report = reconcile(ReconcileInput.from_fields(fields))
    assert [c.id for c in report.failing_checks] == ["total_math"]
    assert report.suggestion is None  # the line, the tax, or the total could be the misread
    assert "subtotal" not in report.failing_checks[0].fields

    table = route(fields, report, THRESHOLD)

    assert flagged(table) == {
        "line_items": "reconciliation:total_math",
        "tax": "reconciliation:total_math",
        "total": "reconciliation:total_math",
    }
    assert table["subtotal"] == NOT_FLAGGED
    stored_report = report_to_json(report)
    assert [c["id"] for c in failing_checks_for(stored_report, "line_items")] == ["total_math"]
    assert failing_checks_for(stored_report, "subtotal") == []


def test_a_suggestion_no_failing_check_names_falls_back_to_the_first_failing_check() -> None:
    # The report is edited by hand: a suggestion for a field (the vendor) that
    # neither failing check involves is flagged with the first failing check.
    fields = [*record_016(), stored("vendor", "Bakery")]
    report = reconcile(ReconcileInput.from_fields(fields))
    odd = Suggestion("vendor", Decimal(1), None, ("lines_sum", "total_math"), "edited")

    table = route(fields, dataclasses.replace(report, suggestion=odd), THRESHOLD)

    # The subtotal (confidence 0.68) is no suspect here, so it keeps its own doubt.
    assert flagged(table) == {"vendor": "reconciliation:lines_sum", "subtotal": "confidence"}


@pytest.mark.parametrize(
    ("path", "name"),
    [
        ("subtotal", "subtotal"),
        ("line_items[2].total", "line_items"),
        ("line_items[10].quantity", "line_items"),
        ("tax", "tax"),
        ("tax_lines[1].amount", "tax_lines"),
        ("tax_lines[0].rate", "tax_lines"),
        ("amount_due", "amount_due"),
    ],
)
def test_an_operand_path_belongs_to_one_field(path: str, name: str) -> None:
    assert field_name_of(path) == name


def stored_tax_lines(rows, confidence=0.99, **kwargs) -> ExtractedField:
    """The tax_lines field stored the way line_items is: the array itself, a
    leaf per cell."""
    cells = [{key: leaf(cell, confidence) for key, cell in row.items()} for row in rows]
    return ExtractedField(
        field_name="tax_lines",
        value=cells,
        confidence=confidence,
        needs_review=False,
        reviewed_at=None,
        review_action=None,
        corrected_value=None,
        **kwargs,
    )


def record_with_tax_lines(second_amount: float) -> list[ExtractedField]:
    """One 100.00 line, tax 10.00 over two tax lines (6% and 4% of 100.00),
    total 110.00."""
    line = {"description": leaf("a"), "quantity": leaf(1), "unit_price": leaf(100.0), "total": leaf(100.0)}
    return [
        stored("line_items", [line]),
        stored("subtotal", 100.0),
        stored("tax", 10.0),
        stored("total", 110.0),
        stored("currency", "USD"),
        stored_tax_lines(
            [
                {"label": "State", "rate": "6", "base": 100.0, "amount": 6.0},
                {"label": "County", "rate": "4", "base": 100.0, "amount": second_amount},
            ]
        ),
    ]


def test_a_misread_tax_line_amount_flags_the_tax_lines_field() -> None:
    # Line 2 reads 5.00: 4% of 100.00 is 4.00 and the amounts no longer add up to
    # the tax. Only that amount fixes both checks, and its operand path
    # `tax_lines[1].amount` belongs to the tax_lines field.
    fields = record_with_tax_lines(5.0)
    report = reconcile(ReconcileInput.from_fields(fields))
    assert [c.id for c in report.failing_checks] == ["tax_lines_sum", "tax_line_math[1]"]
    assert report.suggestion is not None
    assert (report.suggestion.path, report.suggestion.value) == ("tax_lines[1].amount", Decimal(4))

    table = route(fields, report, THRESHOLD)

    assert flagged(table) == {"tax_lines": "reconciliation:tax_lines_sum"}


def test_a_consistent_tax_lines_field_is_not_flagged() -> None:
    fields = record_with_tax_lines(4.0)
    report = reconcile(ReconcileInput.from_fields(fields))
    assert [c.id for c in report.checks if c.id.startswith("tax_line")] == [
        "tax_lines_sum",
        "tax_line_math[0]",
        "tax_line_math[1]",
    ]

    assert flagged(route(fields, report, THRESHOLD)) == {}


# ---- The invariant ----------------------------------------------------------


@given(
    subtotal=st.sampled_from([45.25, 46.25, None, -1.0]),
    tax=st.sampled_from([3.05, None, 9.0]),
    total=st.sampled_from([48.3, 49.3, None]),
    confidences=st.lists(st.sampled_from([0.1, 0.79, 0.8, 1.0]), min_size=4, max_size=4),
    reviewed=st.lists(st.sampled_from([None, "approved"]), min_size=4, max_size=4),
)
def test_every_routing_is_flagged_exactly_when_it_has_a_reason(
    subtotal, tax, total, confidences, reviewed
) -> None:
    names = ("line_items", "subtotal", "tax", "total")
    values = (LINES_016, subtotal, tax, total)
    fields = [
        stored(name, value, conf, action=action)
        for name, value, conf, action in zip(names, values, confidences, reviewed, strict=True)
    ]
    report = reconcile(ReconcileInput.from_fields(fields))

    table = route(fields, report, THRESHOLD)

    assert set(table) == set(names)
    for field in fields:
        routing = table[field.field_name]
        assert routing.needs_review == (routing.review_reason is not None)
        if routing.review_reason is not None:
            assert REASON_PATTERN.match(routing.review_reason)
        if field.reviewed_at is not None:
            assert routing == Routing(field.needs_review, field.review_reason)


# ---- The stored report's helpers ---------------------------------------------


def test_report_helpers_pick_the_checks_and_suggestion_for_one_field() -> None:
    report = report_to_json(reconcile(ReconcileInput.from_fields(record_016())))

    assert [c["id"] for c in failing_checks_for(report, "subtotal")] == ["lines_sum", "total_math"]
    assert [c["id"] for c in failing_checks_for(report, "line_items")] == ["lines_sum"]
    assert [c["id"] for c in failing_checks_for(report, "tax")] == ["total_math"]
    assert failing_checks_for(report, "vendor") == []
    assert failing_checks_for({}, "subtotal") == []
    assert suggestion_for(report, "subtotal")["value"] == "45.25"
    assert suggestion_for(report, "tax") is None
    assert suggestion_for({}, "subtotal") is None


def test_the_reason_helper_builds_the_stored_text() -> None:
    assert reconciliation_reason("line_math[3]") == "reconciliation:line_math[3]"


# ---- refresh_reconciliation (database) ----------------------------------------

SETTINGS = Settings(review_threshold=THRESHOLD)


def tool_input_016(subtotal=46.25, subtotal_confidence=0.68) -> dict:
    return {
        "vendor": leaf("Bakery"),
        "document_date": leaf("2025-03-01"),
        "line_items": leaf(LINES_016),
        "subtotal": leaf(subtotal, subtotal_confidence),
        "tax": leaf(3.05),
        "total": leaf(48.3),
        "currency": leaf("USD"),
    }


async def seed_extraction(db_session: AsyncSession, tool_input: dict) -> uuid.UUID:
    document = Document(
        filename="016.png", mime_type="image/png", storage_path="/tmp/016.png", status="extracted"
    )
    db_session.add(document)
    await db_session.flush()
    extraction = Extraction(
        document_id=document.id,
        prompt_version="extract_v1",
        model="claude-sonnet-5",
        raw_response=tool_input,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0,
        latency_ms=1,
    )
    db_session.add(extraction)
    await db_session.flush()
    db_session.add_all(build_extracted_fields(extraction.id, tool_input, THRESHOLD, SCHEMA_V1))
    await db_session.flush()
    return extraction.id


async def fields_of(db_session: AsyncSession, extraction_id: uuid.UUID) -> dict[str, ExtractedField]:
    stmt = select(ExtractedField).where(ExtractedField.extraction_id == extraction_id)
    return {f.field_name: f for f in (await db_session.execute(stmt)).scalars()}


async def test_refresh_stores_the_report_and_flags_the_culprit(db_session: AsyncSession) -> None:
    extraction_id = await seed_extraction(db_session, tool_input_016())

    result = await refresh_reconciliation(db_session, extraction_id, SETTINGS)

    row = (
        await db_session.execute(
            select(Reconciliation).where(Reconciliation.extraction_id == extraction_id)
        )
    ).scalar_one()
    assert result.created is True
    assert (row.status, row.rules_version) == ("fail", "rules_v2")
    assert row.report["suggestion"]["value"] == "45.25"
    assert row.report["status"] == "fail"
    fields = await fields_of(db_session, extraction_id)
    assert flagged({n: Routing(f.needs_review, f.review_reason) for n, f in fields.items()}) == {
        "subtotal": "reconciliation:lines_sum"
    }
    # The subtotal was already flagged for confidence; its reason moves to the check.
    assert [(c.field_name, c.before.review_reason) for c in result.changes] == [
        ("subtotal", "confidence")
    ]


async def test_refresh_twice_keeps_one_row_and_changes_nothing(db_session: AsyncSession) -> None:
    extraction_id = await seed_extraction(db_session, tool_input_016())
    await refresh_reconciliation(db_session, extraction_id, SETTINGS)

    again = await refresh_reconciliation(db_session, extraction_id, SETTINGS)

    assert again.created is False
    assert again.changes == ()
    count = (
        await db_session.execute(
            select(func.count())
            .select_from(Reconciliation)
            .where(Reconciliation.extraction_id == extraction_id)
        )
    ).scalar_one()
    assert count == 1


async def test_refresh_reads_a_reviewers_correction_and_clears_the_flag_it_fixed(
    db_session: AsyncSession,
) -> None:
    extraction_id = await seed_extraction(db_session, tool_input_016())
    await refresh_reconciliation(db_session, extraction_id, SETTINGS)
    subtotal = (await fields_of(db_session, extraction_id))["subtotal"]
    subtotal.reviewed_at = datetime.now(UTC)
    subtotal.review_action = "corrected"
    subtotal.corrected_value = "45.25"
    await db_session.flush()

    result = await refresh_reconciliation(db_session, extraction_id, SETTINGS)

    assert result.report.status == "pass"
    assert result.reconciliation.status == "pass"
    assert result.changes == ()
    fields = await fields_of(db_session, extraction_id)
    assert fields["subtotal"].review_reason == "reconciliation:lines_sum"  # frozen
    assert not any(
        f.needs_review for name, f in fields.items() if name != "subtotal"
    )


async def test_the_database_refuses_a_flag_without_a_reason(db_session: AsyncSession) -> None:
    extraction_id = await seed_extraction(db_session, tool_input_016())
    tax = (await fields_of(db_session, extraction_id))["tax"]

    tax.needs_review = True
    with pytest.raises(IntegrityError, match='"ck_extracted_fields_review_reason_matches"'):
        await db_session.flush()
    await db_session.rollback()


async def test_the_database_refuses_an_unknown_reason(db_session: AsyncSession) -> None:
    extraction_id = await seed_extraction(db_session, tool_input_016())
    tax = (await fields_of(db_session, extraction_id))["tax"]

    tax.needs_review = True
    tax.review_reason = "because"
    with pytest.raises(IntegrityError, match='"ck_extracted_fields_review_reason"'):
        await db_session.flush()
    await db_session.rollback()
