"""Tests for the human review queue endpoints (app/routers/review.py).

Rows are seeded directly through db_session (no worker/VLM involvement):
the review endpoints only care about what's in extracted_fields, so
constructing the extraction state by hand keeps these tests fast and
network-free, same as the rest of the suite.
"""

import uuid

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, ExtractedField, Extraction


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
