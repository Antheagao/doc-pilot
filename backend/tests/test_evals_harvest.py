"""Tests for harvesting review resolutions into eval cases
(app/evals/harvest.py).

Each test builds document/extraction/field rows directly (as
tests/test_review.py does) and harvests into a tmp_path evals dir, so the
repo's real evals/ corpus is never touched.
"""

import base64
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.evals.dataset import FIELD_KEYS
from app.evals.harvest import harvest_corrections
from app.models import Document, ExtractedField, Extraction

# A minimal valid 1x1 transparent PNG (same as tests/test_documents.py).
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _leaf(value, confidence: float) -> dict:
    return {"value": value, "confidence": confidence}


async def _seed_document(
    db_session: AsyncSession,
    tmp_path: Path,
    *,
    filename: str = "monthly receipt.png",
    field_states: dict | None = None,
) -> Document:
    """Create an extracted document whose scalar fields default to
    high-confidence values; field_states overrides individual
    ExtractedField kwargs (confidence, needs_review, review_*)."""
    field_states = field_states or {}

    storage_path = tmp_path / f"{uuid.uuid4()}.png"
    storage_path.write_bytes(TINY_PNG)

    document = Document(
        filename=filename,
        mime_type="image/png",
        storage_path=str(storage_path),
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

    defaults = {
        "vendor": {"value": _leaf("Acme Hardware", 0.99)},
        "document_date": {"value": _leaf("2026-07-01", 0.99)},
        "currency": {"value": _leaf("USD", 0.99)},
        "subtotal": {"value": _leaf(10.0, 0.99)},
        "tax": {"value": _leaf(0.8, 0.99)},
        "total": {"value": _leaf(10.8, 0.99)},
        "line_items": {
            "value": [
                {
                    "description": _leaf("Hammer", 0.99),
                    "quantity": _leaf(1, 0.99),
                    "unit_price": _leaf(10.0, 0.99),
                    "total": _leaf(10.0, 0.99),
                }
            ]
        },
    }
    for name, kwargs in defaults.items():
        merged = {
            "confidence": 0.99,
            "needs_review": False,
            **kwargs,
            **field_states.get(name, {}),
        }
        db_session.add(
            ExtractedField(extraction_id=extraction.id, field_name=name, **merged)
        )
    await db_session.commit()
    return document


async def test_harvests_fully_reviewed_document(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    now = datetime.now(UTC)
    document = await _seed_document(
        db_session,
        tmp_path,
        field_states={
            # One corrected field: the label must carry the human value.
            "vendor": {
                "value": _leaf("Acme Hrdware", 0.5),
                "confidence": 0.5,
                "needs_review": True,
                "reviewed_at": now,
                "review_action": "corrected",
                "corrected_value": "Acme Hardware Co.",
            },
            # One approved field: the label keeps the extracted value.
            "tax": {
                "confidence": 0.7,
                "needs_review": True,
                "reviewed_at": now,
                "review_action": "approved",
            },
        },
    )
    evals_dir = tmp_path / "evals"

    outcomes = await harvest_corrections(db_session, evals_dir=evals_dir)

    ours = [o for o in outcomes if o.document_id == document.id]
    assert len(ours) == 1
    assert ours[0].status == "harvested"
    doc_id = ours[0].doc_id
    assert doc_id is not None and doc_id.endswith("-review-monthly-receipt")

    label = json.loads((evals_dir / "labels" / f"{doc_id}.json").read_text())
    assert label["source"] == "human-review"
    assert label["source_document_id"] == str(document.id)
    assert set(label["fields"].keys()) == FIELD_KEYS
    assert label["fields"]["vendor"] == "Acme Hardware Co."  # corrected wins
    assert label["fields"]["tax"] == 0.8  # approved keeps extraction
    assert label["fields"]["line_items"] == [
        {"description": "Hammer", "quantity": 1, "unit_price": 10.0, "total": 10.0}
    ]
    # The image is copied beside the label under the same stem.
    assert (evals_dir / "docs" / f"{doc_id}.png").read_bytes() == TINY_PNG


async def test_skips_document_with_pending_review(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    document = await _seed_document(
        db_session,
        tmp_path,
        field_states={
            "vendor": {
                "value": _leaf("Blurry Vendor", 0.4),
                "confidence": 0.4,
                "needs_review": True,
            },
        },
    )
    evals_dir = tmp_path / "evals"

    outcomes = await harvest_corrections(db_session, evals_dir=evals_dir)

    ours = [o for o in outcomes if o.document_id == document.id]
    assert len(ours) == 1
    assert ours[0].status == "skipped_pending"
    assert "vendor" in ours[0].detail
    assert list((evals_dir / "labels").glob("*.json")) == []


async def test_rerun_skips_already_harvested_document(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    document = await _seed_document(db_session, tmp_path)
    evals_dir = tmp_path / "evals"

    first = await harvest_corrections(db_session, evals_dir=evals_dir)
    second = await harvest_corrections(db_session, evals_dir=evals_dir)

    assert [o.status for o in first if o.document_id == document.id] == ["harvested"]
    assert [o.status for o in second if o.document_id == document.id] == [
        "skipped_existing"
    ]
    assert len(list((evals_dir / "labels").glob("*.json"))) == 1


async def test_invalid_human_correction_is_rejected_and_cleaned_up(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    document = await _seed_document(
        db_session,
        tmp_path,
        field_states={
            # A correction in a format the eval corpus refuses (dates must
            # be ISO dashed) -- the label must not survive.
            "document_date": {
                "value": _leaf("07/01/2026", 0.4),
                "confidence": 0.4,
                "needs_review": True,
                "reviewed_at": datetime.now(UTC),
                "review_action": "corrected",
                "corrected_value": "07/01/2026",
            },
        },
    )
    evals_dir = tmp_path / "evals"

    outcomes = await harvest_corrections(db_session, evals_dir=evals_dir)

    ours = [o for o in outcomes if o.document_id == document.id]
    assert ours[0].status == "invalid"
    assert "document_date" in ours[0].detail
    assert list((evals_dir / "labels").glob("*.json")) == []
    assert list((evals_dir / "docs").glob("*")) == []


async def test_numbers_continue_from_existing_labels(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    document = await _seed_document(db_session, tmp_path)
    evals_dir = tmp_path / "evals"
    labels_dir = evals_dir / "labels"
    labels_dir.mkdir(parents=True)
    # Simulate an existing corpus that already reaches 025-...
    (labels_dir / "025-adversarial-office-order-injection.json").write_text("{}")

    outcomes = await harvest_corrections(db_session, evals_dir=evals_dir)

    ours = [o for o in outcomes if o.document_id == document.id]
    assert ours[0].status == "harvested"
    assert ours[0].doc_id is not None
    assert ours[0].doc_id.startswith("026-review-")
