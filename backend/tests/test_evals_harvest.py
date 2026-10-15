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
from app.extraction_fields import SCHEMA_V1, SCHEMA_V2
from app.models import Document, ExtractedField, Extraction

# A minimal valid 1x1 transparent PNG (same as tests/test_documents.py).
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _leaf(value, confidence: float) -> dict:
    return {"value": value, "confidence": confidence}


def _labels_for(evals_dir: Path, document_id: uuid.UUID) -> list[Path]:
    """Label files harvested from a specific document. Tests must scope
    their filesystem assertions this way rather than counting all labels:
    the test database can hold other committed extracted documents (see
    tests/conftest.py), so they may legitimately be harvested
    into the same tmp evals dir during a test's harvest call."""
    matches = []
    for path in (evals_dir / "labels").glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("source_document_id") == str(document_id):
            matches.append(path)
    return matches


async def _seed_document(
    db_session: AsyncSession,
    tmp_path: Path,
    *,
    filename: str = "monthly receipt.png",
    field_states: dict | None = None,
    prompt_version: str = SCHEMA_V2.prompt_version,
) -> Document:
    """Create an extracted document whose scalar fields default to
    high-confidence values; field_states overrides individual
    ExtractedField kwargs (confidence, needs_review, review_*). The fields
    are the ones the prompt version's schema has: all twenty for extract_v2, the
    seven v1 fields for extract_v1."""
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
        prompt_version=prompt_version,
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
        # The v2 fields: a plain "Tax" line, and nothing else the receipt shows.
        "tax_lines": {
            "value": [
                {
                    "label": _leaf("Tax", 0.99),
                    "rate": _leaf(None, 0.99),
                    "base": _leaf(None, 0.99),
                    "amount": _leaf(0.8, 0.99),
                }
            ]
        },
        **{
            name: {"value": _leaf(None, 0.99)}
            for name in SCHEMA_V2.field_names
            if name not in SCHEMA_V1.field_names and name != "tax_lines"
        },
    }
    schema = {s.prompt_version: s for s in (SCHEMA_V1, SCHEMA_V2)}[prompt_version]
    for name, kwargs in defaults.items():
        if name not in schema.field_names:
            continue
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
                "review_reason": "confidence",
                "reviewed_at": now,
                "review_action": "corrected",
                "corrected_value": "Acme Hardware Co.",
            },
            # One approved field: the label keeps the extracted value.
            "tax": {
                "confidence": 0.7,
                "needs_review": True,
                "review_reason": "confidence",
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
    assert label["dataset_version"] == "v2"
    assert label["source_document_id"] == str(document.id)
    assert set(label["fields"].keys()) == FIELD_KEYS == set(SCHEMA_V2.field_names)
    assert label["fields"]["tax_lines"] == [
        {"label": "Tax", "rate": None, "base": None, "amount": 0.8}
    ]
    assert label["fields"]["payment_method"] is None
    assert label["fields"]["vendor"] == "Acme Hardware Co."  # corrected wins
    assert label["fields"]["tax"] == 0.8  # approved keeps extraction
    assert label["fields"]["line_items"] == [
        {"description": "Hammer", "quantity": 1, "unit_price": 10.0, "total": 10.0}
    ]
    # The image is copied beside the label under the same stem.
    assert (evals_dir / "docs" / f"{doc_id}.png").read_bytes() == TINY_PNG


async def test_string_money_corrections_are_harvested_as_numbers(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    now = datetime.now(UTC)
    corrected = {
        "confidence": 0.5,
        "needs_review": True,
        "review_reason": "confidence",
        "reviewed_at": now,
        "review_action": "corrected",
    }
    document = await _seed_document(
        db_session,
        tmp_path,
        field_states={
            "subtotal": {**corrected, "corrected_value": "45.25"},
            "tax": {**corrected, "corrected_value": "0.75"},
            # The tax line that holds the same tax is corrected with it.
            "tax_lines": {
                **corrected,
                "corrected_value": [{"label": "Tax", "rate": None, "base": None, "amount": "0.75"}],
            },
            "total": {**corrected, "corrected_value": "46"},
            "line_items": {
                "value": [
                    {
                        "description": _leaf("Hammer", 0.99),
                        "quantity": _leaf("2.50", 0.99),
                        "unit_price": _leaf(18.1, 0.99),
                        "total": _leaf("45.25", 0.99),
                    }
                ]
            },
        },
    )
    evals_dir = tmp_path / "evals"

    outcomes = await harvest_corrections(db_session, evals_dir=evals_dir)

    (ours,) = [o for o in outcomes if o.document_id == document.id]
    assert ours.status == "harvested"
    fields = json.loads((evals_dir / "labels" / f"{ours.doc_id}.json").read_text())["fields"]
    assert fields["subtotal"] == 45.25 and isinstance(fields["subtotal"], float)
    assert fields["tax"] == 0.75 and isinstance(fields["tax"], float)
    assert fields["total"] == 46 and isinstance(fields["total"], int)
    assert fields["line_items"][0]["total"] == 45.25
    assert fields["line_items"][0]["unit_price"] == 18.1
    assert fields["line_items"][0]["quantity"] == 2.5


async def test_a_label_that_does_not_add_up_is_refused(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    now = datetime.now(UTC)
    document = await _seed_document(
        db_session,
        tmp_path,
        field_states={
            # 10.00 + 0.80 is not 11.80.
            "total": {
                "confidence": 0.5,
                "needs_review": True,
                "review_reason": "confidence",
                "reviewed_at": now,
                "review_action": "corrected",
                "corrected_value": "11.80",
            }
        },
    )
    evals_dir = tmp_path / "evals"

    outcomes = await harvest_corrections(db_session, evals_dir=evals_dir)

    (ours,) = [o for o in outcomes if o.document_id == document.id]
    assert ours.status == "invalid"
    assert ours.doc_id is None
    assert "fail reconciliation" in ours.detail
    assert "total_math" in ours.detail
    assert _labels_for(evals_dir, document.id) == []
    # No image is left without a label.
    label_stems = {path.stem for path in (evals_dir / "labels").glob("*.json")}
    assert {path.stem for path in (evals_dir / "docs").iterdir()} <= label_stems


async def test_a_non_numeric_money_correction_is_still_rejected(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    document = await _seed_document(
        db_session,
        tmp_path,
        field_states={
            "total": {
                "confidence": 0.5,
                "needs_review": True,
                "review_reason": "confidence",
                "reviewed_at": datetime.now(UTC),
                "review_action": "corrected",
                "corrected_value": "1,234.50",
            },
        },
    )
    evals_dir = tmp_path / "evals"

    outcomes = await harvest_corrections(db_session, evals_dir=evals_dir)

    (ours,) = [o for o in outcomes if o.document_id == document.id]
    assert ours.status == "invalid"
    assert "fields.total must be numeric" in ours.detail
    assert _labels_for(evals_dir, document.id) == []


async def test_a_non_object_line_item_row_is_invalid_and_the_run_continues(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    bad = await _seed_document(
        db_session,
        tmp_path,
        filename="bad rows.png",
        field_states={
            "line_items": {
                "confidence": 0.5,
                "needs_review": True,
                "review_reason": "confidence",
                "reviewed_at": datetime.now(UTC),
                "review_action": "corrected",
                "corrected_value": [1, "a", None],
            },
        },
    )
    good = await _seed_document(db_session, tmp_path, filename="good rows.png")
    evals_dir = tmp_path / "evals"

    outcomes = await harvest_corrections(db_session, evals_dir=evals_dir)

    (bad_outcome,) = [o for o in outcomes if o.document_id == bad.id]
    (good_outcome,) = [o for o in outcomes if o.document_id == good.id]
    assert bad_outcome.status == "invalid"
    assert "line_items" in bad_outcome.detail
    assert good_outcome.status == "harvested"
    assert _labels_for(evals_dir, bad.id) == []


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
                "review_reason": "confidence",
            },
        },
    )
    evals_dir = tmp_path / "evals"

    outcomes = await harvest_corrections(db_session, evals_dir=evals_dir)

    ours = [o for o in outcomes if o.document_id == document.id]
    assert len(ours) == 1
    assert ours[0].status == "skipped_pending"
    assert "vendor" in ours[0].detail
    assert _labels_for(evals_dir, document.id) == []


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
    assert len(_labels_for(evals_dir, document.id)) == 1


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
                "review_reason": "confidence",
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
    assert _labels_for(evals_dir, document.id) == []
    # Every image left in docs/ must belong to some surviving label (i.e.
    # no orphan was left behind by our rejected document) -- checked by
    # stem pairing since other documents in the test database may have harvested
    # legitimately into this same tmp dir.
    label_stems = {p.stem for p in (evals_dir / "labels").glob("*.json")}
    for image in (evals_dir / "docs").glob("*"):
        assert image.stem in label_stems


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
    # Numbering must continue past the highest existing label (025-...).
    # Other documents in the test database may harvest ahead of ours in the same
    # run, so assert the floor rather than an exact number.
    assert int(ours[0].doc_id.split("-", 1)[0]) >= 26
    assert "-review-" in ours[0].doc_id


async def test_a_v1_extraction_is_skipped_with_its_reason(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    document = await _seed_document(
        db_session, tmp_path, filename="old extraction.png", prompt_version=SCHEMA_V1.prompt_version
    )
    evals_dir = tmp_path / "evals"

    outcomes = await harvest_corrections(db_session, evals_dir=evals_dir)

    (ours,) = [o for o in outcomes if o.document_id == document.id]
    assert ours.status == "skipped_schema"
    assert ours.doc_id is None
    assert "extract_v1" in ours.detail
    assert "tax_lines" in ours.detail
    assert "re-extract with extract_v2" in ours.detail
    assert _labels_for(evals_dir, document.id) == []
    # Nothing is copied for a document that is not harvested.
    assert not any((evals_dir / "docs").glob("*review-old-extraction*"))


async def test_corrected_tax_lines_are_harvested_as_numbers(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    corrected = {
        "confidence": 0.5,
        "needs_review": True,
        "review_reason": "confidence",
        "reviewed_at": datetime.now(UTC),
        "review_action": "corrected",
    }
    document = await _seed_document(
        db_session,
        tmp_path,
        field_states={
            "tax_lines": {
                **corrected,
                "corrected_value": [
                    {"label": "State tax", "rate": "8", "base": "10.00", "amount": "0.80"}
                ],
            },
        },
    )
    evals_dir = tmp_path / "evals"

    outcomes = await harvest_corrections(db_session, evals_dir=evals_dir)

    (ours,) = [o for o in outcomes if o.document_id == document.id]
    assert ours.status == "harvested"
    fields = json.loads((evals_dir / "labels" / f"{ours.doc_id}.json").read_text())["fields"]
    assert fields["tax_lines"] == [{"label": "State tax", "rate": 8, "base": 10.0, "amount": 0.8}]


async def test_tax_lines_that_do_not_add_up_to_the_tax_are_refused(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    document = await _seed_document(
        db_session,
        tmp_path,
        field_states={
            "tax_lines": {
                "confidence": 0.5,
                "needs_review": True,
                "review_reason": "confidence",
                "reviewed_at": datetime.now(UTC),
                "review_action": "corrected",
                "corrected_value": [{"label": "Tax", "rate": None, "base": None, "amount": "0.90"}],
            },
        },
    )
    evals_dir = tmp_path / "evals"

    outcomes = await harvest_corrections(db_session, evals_dir=evals_dir)

    (ours,) = [o for o in outcomes if o.document_id == document.id]
    assert ours.status == "invalid"
    assert "tax_lines_sum" in ours.detail
    assert _labels_for(evals_dir, document.id) == []


async def test_a_card_number_in_a_label_is_refused(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    document = await _seed_document(
        db_session,
        tmp_path,
        field_states={
            "payment_method": {
                "value": _leaf("VISA 4111 1111 1111 1111", 0.99),
            },
        },
    )
    evals_dir = tmp_path / "evals"

    outcomes = await harvest_corrections(db_session, evals_dir=evals_dir)

    (ours,) = [o for o in outcomes if o.document_id == document.id]
    assert ours.status == "invalid"
    assert "card or account number" in ours.detail
    assert _labels_for(evals_dir, document.id) == []


async def test_a_label_carries_the_stored_evidence_of_unreviewed_values_only(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    def cited(value, quote: str) -> dict:
        return {**_leaf(value, 0.99), "evidence": {"page": 1, "quote": quote}}

    now = datetime.now(UTC)
    document = await _seed_document(
        db_session,
        tmp_path,
        field_states={
            "subtotal": {"value": cited(10.0, "Subtotal $10.00")},
            "total": {"value": cited(10.8, "Total $10.80")},
            "line_items": {
                "value": [
                    {
                        "description": _leaf("Hammer", 0.99),
                        "quantity": _leaf(1, 0.99),
                        "unit_price": cited(10.0, "Hammer 1 $10.00 $10.00"),
                        "total": cited(10.0, "Hammer 1 $10.00 $10.00"),
                    }
                ]
            },
            "tax_lines": {
                "value": [
                    {
                        "label": _leaf("Tax", 0.99),
                        "rate": _leaf(None, 0.99),
                        "base": _leaf(None, 0.99),
                        "amount": cited(0.8, "Tax $0.80"),
                    }
                ]
            },
            # Reviewed values keep no quote: the person's word replaced it.
            "tax": {
                "value": cited(0.8, "Tax $0.80"),
                "needs_review": True,
                "review_reason": "confidence",
                "reviewed_at": now,
                "review_action": "approved",
            },
            "vendor": {
                "value": cited("Acme Hrdware", "Acme Hrdware"),
                "needs_review": True,
                "review_reason": "confidence",
                "reviewed_at": now,
                "review_action": "corrected",
                "corrected_value": "Acme Hardware",
            },
        },
    )
    evals_dir = tmp_path / "evals"

    outcomes = await harvest_corrections(db_session, evals_dir=evals_dir)

    (ours,) = [o for o in outcomes if o.document_id == document.id]
    assert ours.status == "harvested"
    label = json.loads((evals_dir / "labels" / f"{ours.doc_id}.json").read_text())
    assert label["evidence"] == {
        "line_items[0].unit_price": {"page": 1, "quote": "Hammer 1 $10.00 $10.00"},
        "line_items[0].total": {"page": 1, "quote": "Hammer 1 $10.00 $10.00"},
        "subtotal": {"page": 1, "quote": "Subtotal $10.00"},
        "tax_lines[0].amount": {"page": 1, "quote": "Tax $0.80"},
        "total": {"page": 1, "quote": "Total $10.80"},
    }
