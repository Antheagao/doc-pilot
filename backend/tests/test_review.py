"""Tests for the human review queue endpoints (app/routers/review.py).

Rows are seeded directly through db_session (no worker/VLM involvement):
the review endpoints only care about what's in extracted_fields, so
constructing the extraction state by hand keeps these tests fast and
network-free, same as the rest of the suite.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.config import Settings
from app.extraction import build_extracted_fields
from app.models import (
    REVIEW_REASON_CONFIDENCE,
    Document,
    ExtractedField,
    Extraction,
    Reconciliation,
)
from app.review_routing import refresh_reconciliation


async def _seed_field(
    db_session: AsyncSession,
    *,
    field_name: str = "vendor",
    value: dict | list | None = None,
    confidence: float = 0.5,
    needs_review: bool = True,
    filename: str = "receipt.png",
) -> ExtractedField:
    """Create a document -> extraction -> single field chain and return
    the field. Defaults produce a pending review-queue entry."""
    document = Document(
        filename=filename,
        mime_type="image/png",
        storage_path=f"/tmp/{uuid.uuid4()}.png",
        status="extracted",
    )
    db_session.add(document)
    await db_session.flush()

    extraction = Extraction(
        document_id=document.id,
        prompt_version="extract_v1",
        model="claude-sonnet-5",
        raw_response={},
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_ms=1000,
    )
    db_session.add(extraction)
    await db_session.flush()

    field = ExtractedField(
        extraction_id=extraction.id,
        field_name=field_name,
        value=value if value is not None else {"value": "Acme", "confidence": confidence},
        confidence=confidence,
        needs_review=needs_review,
        review_reason=REVIEW_REASON_CONFIDENCE if needs_review else None,
    )
    db_session.add(field)
    await db_session.commit()
    return field


async def test_queue_lists_pending_field_with_document_context(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    field = await _seed_field(db_session, filename="fuzzy-receipt.png")

    response = await client.get("/review/queue")

    assert response.status_code == 200
    items = {item["field_id"]: item for item in response.json()}
    assert str(field.id) in items
    item = items[str(field.id)]
    assert item["field_name"] == "vendor"
    assert item["filename"] == "fuzzy-receipt.png"
    assert item["confidence"] == 0.5
    assert item["value"] == {"value": "Acme", "confidence": 0.5}
    assert item["model"] == "claude-sonnet-5"


async def test_queue_excludes_confident_and_already_reviewed_fields(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    confident = await _seed_field(db_session, confidence=0.99, needs_review=False)
    reviewed = await _seed_field(db_session)
    resolve = await client.post(
        f"/review/fields/{reviewed.id}/resolve", json={"action": "approve"}
    )
    assert resolve.status_code == 200

    response = await client.get("/review/queue")

    assert response.status_code == 200
    listed_ids = {item["field_id"] for item in response.json()}
    assert str(confident.id) not in listed_ids
    assert str(reviewed.id) not in listed_ids


async def test_queue_count_tracks_resolutions(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    before = (await client.get("/review/queue/count")).json()["pending"]
    field = await _seed_field(db_session)

    after_seed = (await client.get("/review/queue/count")).json()["pending"]
    assert after_seed == before + 1

    resolve = await client.post(
        f"/review/fields/{field.id}/resolve", json={"action": "approve"}
    )
    assert resolve.status_code == 200

    after_resolve = (await client.get("/review/queue/count")).json()["pending"]
    assert after_resolve == before


async def test_approve_marks_field_reviewed_without_touching_value(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    field = await _seed_field(db_session)

    response = await client.post(
        f"/review/fields/{field.id}/resolve", json={"action": "approve"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["review_action"] == "approved"
    assert body["reviewed_at"] is not None
    assert body["corrected_value"] is None
    assert body["value"] == {"value": "Acme", "confidence": 0.5}

    await db_session.refresh(field)
    assert field.review_action == "approved"
    assert field.reviewed_at is not None
    assert field.corrected_value is None


async def test_correct_stores_corrected_value_beside_original(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    field = await _seed_field(db_session)

    response = await client.post(
        f"/review/fields/{field.id}/resolve",
        json={"action": "correct", "corrected_value": "Acme Hardware Co."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["review_action"] == "corrected"
    assert body["corrected_value"] == "Acme Hardware Co."
    # The model's original answer must survive the correction.
    assert body["value"] == {"value": "Acme", "confidence": 0.5}

    await db_session.refresh(field)
    assert field.corrected_value == "Acme Hardware Co."
    assert field.value == {"value": "Acme", "confidence": 0.5}


async def test_correct_to_explicit_null_is_allowed(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    field = await _seed_field(db_session, field_name="tax")

    response = await client.post(
        f"/review/fields/{field.id}/resolve",
        json={"action": "correct", "corrected_value": None},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["review_action"] == "corrected"
    assert body["corrected_value"] is None


async def test_correct_without_corrected_value_rejected(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    field = await _seed_field(db_session)

    response = await client.post(
        f"/review/fields/{field.id}/resolve", json={"action": "correct"}
    )

    assert response.status_code == 400

    await db_session.refresh(field)
    assert field.reviewed_at is None


async def test_resolving_twice_conflicts(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    field = await _seed_field(db_session)

    first = await client.post(
        f"/review/fields/{field.id}/resolve", json={"action": "approve"}
    )
    assert first.status_code == 200

    second = await client.post(
        f"/review/fields/{field.id}/resolve",
        json={"action": "correct", "corrected_value": "late edit"},
    )

    assert second.status_code == 409
    await db_session.refresh(field)
    # The first resolution must be untouched by the rejected second one.
    assert field.review_action == "approved"
    assert field.corrected_value is None


async def test_resolve_unknown_field_returns_404(client: AsyncClient) -> None:
    response = await client.post(
        f"/review/fields/{uuid.uuid4()}/resolve", json={"action": "approve"}
    )

    assert response.status_code == 404


async def test_resolve_invalid_action_rejected(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    field = await _seed_field(db_session)

    response = await client.post(
        f"/review/fields/{field.id}/resolve", json={"action": "shrug"}
    )

    assert response.status_code == 422


async def _seed_receipt(
    db_session: AsyncSession,
    *,
    currency: str | None = "USD",
    flagged: tuple[str, ...] = ("subtotal",),
    subtotal: float = 46.25,
    line_items: list | None = None,
) -> dict[str, ExtractedField]:
    """One extraction with currency, subtotal, and line_items fields; the
    names in `flagged` are pending review. Returns the fields by name."""
    document = Document(
        filename="016-handwriting-bakery-note.png",
        mime_type="image/png",
        storage_path=f"/tmp/{uuid.uuid4()}.png",
        status="extracted",
    )
    db_session.add(document)
    await db_session.flush()
    extraction = Extraction(
        document_id=document.id,
        prompt_version="extract_v1",
        model="claude-sonnet-5",
        raw_response={},
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_ms=1000,
    )
    db_session.add(extraction)
    await db_session.flush()
    stored = {
        "subtotal": {"value": subtotal, "confidence": 0.5},
        "line_items": line_items
        if line_items is not None
        else [
            {
                "description": {"value": "Sourdough", "confidence": 0.9},
                "quantity": {"value": 5, "confidence": 0.9},
                "unit_price": {"value": 9.15, "confidence": 0.5},
                "total": {"value": 45.75, "confidence": 0.9},
            }
        ],
    }
    if currency is not None:
        stored["currency"] = {"value": currency, "confidence": 0.99}
    fields = {}
    for name, value in stored.items():
        fields[name] = ExtractedField(
            extraction_id=extraction.id,
            field_name=name,
            value=value,
            confidence=0.5 if name in flagged else 0.99,
            needs_review=name in flagged,
            review_reason=REVIEW_REASON_CONFIDENCE if name in flagged else None,
        )
        db_session.add(fields[name])
    await db_session.commit()
    return fields


async def _correct(client: AsyncClient, field: ExtractedField, corrected_value):
    return await client.post(
        f"/review/fields/{field.id}/resolve",
        json={"action": "correct", "corrected_value": corrected_value},
    )


async def test_queue_serves_decimal_strings_and_the_document_currency(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    fields = await _seed_receipt(db_session, flagged=("subtotal", "line_items"))

    items = {i["field_name"]: i for i in (await client.get("/review/queue")).json()}

    assert items["subtotal"]["currency"] == "USD"
    assert items["subtotal"]["value"] == {"value": "46.25", "confidence": 0.5}
    assert items["subtotal"]["field_id"] == str(fields["subtotal"].id)
    row = items["line_items"]["value"][0]
    assert row["unit_price"]["value"] == "9.15"
    assert row["quantity"]["value"] == "5"


async def test_queue_currency_is_upper_cased_and_null_when_unstated(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await _seed_receipt(db_session, currency="jpy", subtotal=1500)
    await _seed_receipt(db_session, currency=None, subtotal=7)

    items = (await client.get("/review/queue")).json()

    by_currency = {i["currency"]: i["value"]["value"] for i in items}
    assert by_currency == {"JPY": "1500", None: "7.00"}


async def test_correct_money_stores_the_canonical_string(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    fields = await _seed_receipt(db_session)

    response = await _correct(client, fields["subtotal"], "45.25")

    assert response.status_code == 200
    assert response.json()["corrected_value"] == "45.25"
    await db_session.refresh(fields["subtotal"])
    assert fields["subtotal"].corrected_value == "45.25"


async def test_correct_money_accepts_a_number_and_pads_it(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    fields = await _seed_receipt(db_session)

    response = await _correct(client, fields["subtotal"], 45.5)

    assert response.status_code == 200
    await db_session.refresh(fields["subtotal"])
    assert fields["subtotal"].corrected_value == "45.50"


async def test_correct_money_with_too_many_decimals_is_422_with_a_sentence(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    fields = await _seed_receipt(db_session)

    response = await _correct(client, fields["subtotal"], "45.255")

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "'45.255' has too many decimal places; USD allows at most 2 decimal places."
    )
    await db_session.refresh(fields["subtotal"])
    # A refused correction leaves the field pending.
    assert fields["subtotal"].reviewed_at is None
    assert fields["subtotal"].corrected_value is None


async def test_correct_money_follows_the_document_currency(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    jpy = await _seed_receipt(db_session, currency="JPY", subtotal=1500)

    refused = await _correct(client, jpy["subtotal"], "1500.5")
    assert refused.status_code == 422
    assert refused.json()["detail"] == (
        "'1500.5' has too many decimal places; JPY allows no decimal places."
    )

    accepted = await _correct(client, jpy["subtotal"], "1500")
    assert accepted.status_code == 200
    assert accepted.json()["corrected_value"] == "1500"


@pytest.mark.parametrize(
    "bad",
    ["abc", "1,234.50", "45,25", "45 25", "1e3", "", True, [1], "-1000000000000", "1000000000000"],
)
async def test_correct_money_rejects_non_amounts(
    client: AsyncClient, db_session: AsyncSession, bad
) -> None:
    fields = await _seed_receipt(db_session)

    response = await _correct(client, fields["subtotal"], bad)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)


async def test_correct_line_items_stores_canonical_strings(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    fields = await _seed_receipt(db_session, flagged=("line_items",))
    rows = [
        {"description": "Sourdough", "quantity": 5, "unit_price": "9.1", "total": 45.5},
        {"description": None, "quantity": "2.50", "unit_price": None, "total": None},
    ]

    response = await _correct(client, fields["line_items"], rows)

    assert response.status_code == 200
    assert response.json()["corrected_value"] == [
        {"description": "Sourdough", "quantity": "5", "unit_price": "9.10", "total": "45.50"},
        {"description": None, "quantity": "2.5", "unit_price": None, "total": None},
    ]


@pytest.mark.parametrize(
    "rows, fragment",
    [
        ([{"description": "x", "quantity": 1, "unit_price": 1, "price": 1}], "exactly the keys"),
        ([{"description": "x", "quantity": 1, "unit_price": 1}], "exactly the keys"),
        (["not an object"], "must be an object"),
        ({"description": "x"}, "must be a list"),
        (
            [{"description": 3, "quantity": 1, "unit_price": 1, "total": 1}],
            "description must be text",
        ),
        (
            [{"description": "x", "quantity": 1, "unit_price": "9.155", "total": 1}],
            "Line item 1 unit_price: ",
        ),
    ],
)
async def test_correct_line_items_rejects_a_malformed_row(
    client: AsyncClient, db_session: AsyncSession, rows, fragment: str
) -> None:
    fields = await _seed_receipt(db_session, flagged=("line_items",))

    response = await _correct(client, fields["line_items"], rows)

    assert response.status_code == 422
    assert fragment in response.json()["detail"]


async def test_correct_currency_is_upper_cased_and_validated(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    fields = await _seed_receipt(db_session, flagged=("currency",))

    refused = await _correct(client, fields["currency"], "dollars")
    assert refused.status_code == 422
    assert "three-letter currency code" in refused.json()["detail"]

    accepted = await _correct(client, fields["currency"], " usd ")
    assert accepted.status_code == 200
    assert accepted.json()["corrected_value"] == "USD"


async def test_correct_to_null_is_allowed_for_every_typed_field(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    fields = await _seed_receipt(db_session, flagged=("subtotal", "line_items", "currency"))

    for name in ("subtotal", "line_items", "currency"):
        response = await _correct(client, fields[name], None)
        assert response.status_code == 200, name
        assert response.json()["corrected_value"] is None


async def test_an_older_numeric_correction_is_served_as_a_padded_string(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    fields = await _seed_receipt(db_session, flagged=("subtotal", "line_items"))
    # Corrections stored before the string contract are JSON numbers.
    subtotal, line_items = fields["subtotal"], fields["line_items"]
    subtotal.review_action = "corrected"
    subtotal.corrected_value = 45.2
    line_items.review_action = "corrected"
    line_items.corrected_value = [
        {"description": "Sourdough", "quantity": 5, "unit_price": 9.1, "total": 45.5}
    ]
    await db_session.commit()
    extraction = await db_session.get(Extraction, subtotal.extraction_id)
    assert extraction is not None

    body = (await client.get(f"/documents/{extraction.document_id}")).json()

    served = {f["field_name"]: f for f in body["extraction"]["fields"]}
    assert served["subtotal"]["corrected_value"] == "45.20"
    assert served["line_items"]["corrected_value"] == [
        {"description": "Sourdough", "quantity": "5", "unit_price": "9.10", "total": "45.50"}
    ]


async def test_correct_money_without_a_currency_says_the_assumption(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    fields = await _seed_receipt(db_session, currency=None)

    response = await _correct(client, fields["subtotal"], "45.255")

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "'45.255' has too many decimal places; "
        "no currency is stated, so amounts allow at most 2 decimal places."
    )


@pytest.mark.parametrize("raw", ["45,25", "45 25"])
async def test_correct_money_refuses_a_european_decimal_comma(
    client: AsyncClient, db_session: AsyncSession, raw: str
) -> None:
    fields = await _seed_receipt(db_session)

    response = await _correct(client, fields["subtotal"], raw)

    assert response.status_code == 422
    assert "not a plain number" in response.json()["detail"]
    await db_session.refresh(fields["subtotal"])
    assert fields["subtotal"].reviewed_at is None


async def test_the_422_sentence_cuts_a_long_echoed_value(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    fields = await _seed_receipt(db_session)

    response = await _correct(client, fields["subtotal"], "9" * 500 + "x")

    assert response.status_code == 422
    assert len(response.json()["detail"]) < 250
    assert "..." in response.json()["detail"]


@pytest.mark.parametrize(
    "quantity, fragment",
    [
        ("9" * 50, "quantities must be below"),
        ("0.0000001", "quantities allow at most 6"),
        ("1e308", "not a plain number"),
        ("2,5", "not a plain number"),
    ],
)
async def test_correct_line_items_bounds_the_quantity(
    client: AsyncClient, db_session: AsyncSession, quantity: str, fragment: str
) -> None:
    fields = await _seed_receipt(db_session, flagged=("line_items",))
    rows = [{"description": "x", "quantity": quantity, "unit_price": "1.00", "total": "1.00"}]

    response = await _correct(client, fields["line_items"], rows)

    assert response.status_code == 422
    assert fragment in response.json()["detail"]
    assert response.json()["detail"].startswith("Line item 1 quantity: ")


async def test_correct_line_items_allows_a_negative_fractional_quantity(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    fields = await _seed_receipt(db_session, flagged=("line_items",))
    rows = [{"description": "return", "quantity": "-0.125", "unit_price": "8.00", "total": "-1.00"}]

    response = await _correct(client, fields["line_items"], rows)

    assert response.status_code == 200
    assert response.json()["corrected_value"][0]["quantity"] == "-0.125"


# ---- Reconciliation routing --------------------------------------------------

THRESHOLD_SETTINGS = Settings(review_threshold=0.8)


def _leaf(value, confidence=0.99):
    return {"value": value, "confidence": confidence}


async def _seed_016(db_session: AsyncSession) -> tuple[uuid.UUID, uuid.UUID, dict[str, ExtractedField]]:
    """The staged bakery receipt: lines 36.60 + 8.65 = 45.25, tax 3.05, total
    48.30, and a subtotal misread as 46.25 with confidence 0.68. Stored and
    routed the way the worker does it."""
    tool_input = {
        "vendor": _leaf("Bakery"),
        "document_date": _leaf("2025-03-01"),
        "line_items": _leaf(
            [
                {
                    "description": _leaf("Sourdough"),
                    "quantity": _leaf(4),
                    "unit_price": _leaf(9.15),
                    "total": _leaf(36.6),
                },
                {
                    "description": _leaf("Rolls"),
                    "quantity": _leaf(1),
                    "unit_price": _leaf(8.65),
                    "total": _leaf(8.65),
                },
            ]
        ),
        "subtotal": _leaf(46.25, 0.68),
        "tax": _leaf(3.05),
        "total": _leaf(48.3),
        "currency": _leaf("USD"),
    }
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
    fields = build_extracted_fields(extraction.id, tool_input, THRESHOLD_SETTINGS.review_threshold)
    db_session.add_all(fields)
    await db_session.flush()
    await refresh_reconciliation(db_session, extraction.id, THRESHOLD_SETTINGS)
    await db_session.commit()
    return document.id, extraction.id, {f.field_name: f for f in fields}


async def _queue_for(client: AsyncClient, extraction_id: uuid.UUID) -> dict[str, dict]:
    items = (await client.get("/review/queue")).json()
    return {i["field_name"]: i for i in items if i["extraction_id"] == str(extraction_id)}


async def test_queue_item_carries_the_reason_the_failing_checks_and_the_suggestion(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    _, extraction_id, _ = await _seed_016(db_session)

    queue = await _queue_for(client, extraction_id)

    # Only the suggested field is flagged; the operands it would fix are not.
    assert set(queue) == {"subtotal"}
    item = queue["subtotal"]
    assert item["review_reason"] == "reconciliation:lines_sum"
    assert [c["id"] for c in item["checks"]] == ["lines_sum", "total_math"]
    assert item["checks"][0]["status"] == "fail"
    assert item["checks"][0]["delta"] == "1.00"
    assert item["suggestion"]["path"] == "subtotal"
    assert (item["suggestion"]["value"], item["suggestion"]["current"]) == ("45.25", "46.25")


async def test_a_confidence_flag_has_no_checks_and_no_suggestion(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    field = await _seed_field(db_session)

    item = {i["field_id"]: i for i in (await client.get("/review/queue")).json()}[str(field.id)]

    assert item["review_reason"] == "confidence"
    assert item["checks"] == []
    assert item["suggestion"] is None


async def test_correcting_the_culprit_recomputes_and_clears_the_queue(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    document_id, extraction_id, fields = await _seed_016(db_session)

    response = await _correct(client, fields["subtotal"], "45.25")

    assert response.status_code == 200
    assert response.json()["review_reason"] == "reconciliation:lines_sum"  # frozen once reviewed
    assert await _queue_for(client, extraction_id) == {}
    extraction = (await client.get(f"/documents/{document_id}")).json()["extraction"]
    assert extraction["reconciliation"]["status"] == "pass"
    assert [c for c in extraction["reconciliation"]["checks"] if c["status"] == "fail"] == []
    assert extraction["reconciliation"]["suggestion"] is None
    unreviewed = [f for f in extraction["fields"] if f["reviewed_at"] is None]
    assert not any(f["needs_review"] for f in unreviewed)
    assert all(f["review_reason"] is None for f in unreviewed)


async def test_approving_the_wrong_value_flags_the_other_operands(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    document_id, extraction_id, fields = await _seed_016(db_session)

    response = await client.post(
        f"/review/fields/{fields['subtotal'].id}/resolve", json={"action": "approve"}
    )

    assert response.status_code == 200
    queue = await _queue_for(client, extraction_id)
    assert {n: i["review_reason"] for n, i in queue.items()} == {
        "line_items": "reconciliation:lines_sum",
        "tax": "reconciliation:total_math",
        "total": "reconciliation:total_math",
    }
    assert all(i["suggestion"] is None and i["checks"] for i in queue.values())
    extraction = (await client.get(f"/documents/{document_id}")).json()["extraction"]
    assert extraction["reconciliation"]["status"] == "fail"

    # Correcting the real culprit afterwards clears the rest.
    line_items = [
        {"description": "Sourdough", "quantity": "4", "unit_price": "9.15", "total": "36.60"},
        {"description": "Rolls", "quantity": "1", "unit_price": "9.65", "total": "9.65"},
    ]
    assert (await _correct(client, fields["line_items"], line_items)).status_code == 200
    queue = await _queue_for(client, extraction_id)
    # 36.60 + 9.65 = 46.25 now agrees with the approved subtotal; total_math remains.
    assert {n: i["review_reason"] for n, i in queue.items()} == {
        "tax": "reconciliation:total_math",
        "total": "reconciliation:total_math",
    }


async def test_a_document_nobody_reconciled_serves_no_reconciliation(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    field = await _seed_field(db_session)
    extraction = await db_session.get(Extraction, field.extraction_id)

    body = (await client.get(f"/documents/{extraction.document_id}")).json()

    assert body["extraction"]["reconciliation"] is None
    assert body["extraction"]["fields"][0]["review_reason"] == "confidence"


async def _stored_report(db_session: AsyncSession, extraction_id: uuid.UUID) -> Reconciliation:
    return (
        await db_session.execute(
            select(Reconciliation).where(Reconciliation.extraction_id == extraction_id)
        )
    ).scalar_one()


async def test_a_value_from_a_newer_rules_version_is_served_as_stored(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    document_id, extraction_id, _ = await _seed_016(db_session)
    row = await _stored_report(db_session, extraction_id)
    report = row.report
    lines_sum = next(c for c in report["checks"] if c["id"] == "lines_sum")
    lines_sum["kind"] = "ratio"
    lines_sum["terms"][0]["operand"]["source"] = "discount"
    row.report = report
    flag_modified(row, "report")
    await db_session.commit()

    queue = await _queue_for(client, extraction_id)
    detail = (await client.get(f"/documents/{document_id}")).json()

    assert queue["subtotal"]["checks"][0]["kind"] == "ratio"
    assert queue["subtotal"]["suggestion"]["value"] == "45.25"
    served = next(c for c in detail["extraction"]["reconciliation"]["checks"] if c["id"] == "lines_sum")
    assert served["terms"][0]["operand"]["source"] == "discount"


async def test_an_unreadable_stored_report_costs_only_its_own_context(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    document_id, extraction_id, _ = await _seed_016(db_session)
    other = await _seed_field(db_session)
    row = await _stored_report(db_session, extraction_id)
    report = row.report
    del next(c for c in report["checks"] if c["id"] == "lines_sum")["status"]
    row.report = report
    flag_modified(row, "report")
    await db_session.commit()

    queue_response = await client.get("/review/queue")
    detail_response = await client.get(f"/documents/{document_id}")

    assert queue_response.status_code == 200
    subtotal = next(
        i for i in queue_response.json() if i["extraction_id"] == str(extraction_id)
    )
    # The item is still queued with its reason; only the checks and suggestion go.
    assert subtotal["review_reason"] == "reconciliation:lines_sum"
    assert subtotal["checks"] == []
    assert subtotal["suggestion"] is None
    assert str(other.id) in {i["field_id"] for i in queue_response.json()}
    assert detail_response.status_code == 200
    body = detail_response.json()["extraction"]
    assert body["reconciliation"] is None
    assert len(body["fields"]) == 7
