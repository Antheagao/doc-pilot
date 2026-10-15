import base64
import uuid

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from v2_samples import seed_v2

from app.models import Document, ExtractedField, Extraction, Job
from app.routers.documents import MAX_UPLOAD_SIZE

# A minimal valid 1x1 transparent PNG.
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


async def test_upload_document_happy_path(
    client: AsyncClient, db_session: AsyncSession, tmp_path
) -> None:
    response = await client.post(
        "/documents",
        files={"file": ("receipt.png", TINY_PNG, "image/png")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "receipt.png"
    assert body["status"] == "uploaded"
    document_id = uuid.UUID(body["id"])

    document = await db_session.get(Document, document_id)
    assert document is not None
    assert document.mime_type == "image/png"
    assert document.status == "uploaded"

    saved_files = list(tmp_path.glob(f"{document_id}*"))
    assert len(saved_files) == 1
    assert saved_files[0].read_bytes() == TINY_PNG

    job_result = await db_session.execute(
        select(Job).where(Job.document_id == document_id)
    )
    job = job_result.scalars().first()
    assert job is not None
    assert job.state == "pending"


async def test_upload_document_bad_mime_type(client: AsyncClient) -> None:
    response = await client.post(
        "/documents",
        files={"file": ("notes.txt", b"just some text", "text/plain")},
    )

    assert response.status_code == 415


async def test_upload_rejects_content_mismatching_declared_type(
    client: AsyncClient,
) -> None:
    """The Content-Type header is client-controlled; bytes that don't
    carry the claimed format's magic number must be rejected, or they'd
    later be served back out under that content type."""
    response = await client.post(
        "/documents",
        files={"file": ("payload.png", b"<html>not a png</html>", "image/png")},
    )

    assert response.status_code == 415


async def test_upload_accepts_pdf_magic_bytes(client: AsyncClient) -> None:
    response = await client.post(
        "/documents",
        files={"file": ("doc.pdf", b"%PDF-1.7 minimal", "application/pdf")},
    )

    assert response.status_code == 201


async def test_responses_carry_nosniff_header(client: AsyncClient) -> None:
    response = await client.get("/documents")

    assert response.headers.get("x-content-type-options") == "nosniff"


async def test_get_document_unknown_id_returns_404(client: AsyncClient) -> None:
    response = await client.get(f"/documents/{uuid.uuid4()}")

    assert response.status_code == 404


async def test_list_documents_returns_uploaded_doc(client: AsyncClient) -> None:
    upload_response = await client.post(
        "/documents",
        files={"file": ("receipt.png", TINY_PNG, "image/png")},
    )
    document_id = upload_response.json()["id"]

    list_response = await client.get("/documents")

    assert list_response.status_code == 200
    ids = [doc["id"] for doc in list_response.json()]
    assert document_id in ids


async def test_upload_document_oversized_rejected(client: AsyncClient) -> None:
    oversized = b"0" * (MAX_UPLOAD_SIZE + 1)

    response = await client.post(
        "/documents",
        files={"file": ("big.png", oversized, "image/png")},
    )

    assert response.status_code == 413


async def test_upload_document_extension_from_mime_type(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    response = await client.post(
        "/documents",
        files={"file": ("evil.exe", TINY_PNG, "image/png")},
    )

    assert response.status_code == 201
    document_id = uuid.UUID(response.json()["id"])

    document = await db_session.get(Document, document_id)
    assert document is not None
    assert document.storage_path.endswith(".png")


def _leaf(value, confidence: float = 0.95) -> dict:
    return {"value": value, "confidence": confidence}


async def _seed_extracted_document(
    db_session: AsyncSession, fields: dict[str, dict | list]
) -> Document:
    """A document with one extraction holding the given stored field values."""
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
    for name, value in fields.items():
        db_session.add(
            ExtractedField(
                extraction_id=extraction.id,
                field_name=name,
                value=value,
                confidence=0.95,
                needs_review=False,
            )
        )
    await db_session.commit()
    return document


async def test_get_document_serves_amounts_as_decimal_strings(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    document = await _seed_extracted_document(
        db_session,
        {
            "currency": _leaf("USD"),
            "subtotal": _leaf(46.25),
            "tax": _leaf(None),
            "total": _leaf(46.2, 0.4),
            "line_items": [
                {
                    "description": _leaf("Sourdough loaf"),
                    "quantity": _leaf(5),
                    "unit_price": _leaf(9.15),
                    "total": _leaf(45.75),
                },
                {
                    "description": _leaf("Rye"),
                    "quantity": _leaf(2.5),
                    "unit_price": _leaf("abc"),
                    "total": _leaf(None),
                },
            ],
        },
    )

    response = await client.get(f"/documents/{document.id}")

    assert response.status_code == 200
    fields = {f["field_name"]: f for f in response.json()["extraction"]["fields"]}
    assert fields["subtotal"]["value"] == {"value": "46.25", "confidence": 0.95}
    assert fields["tax"]["value"] == {"value": None, "confidence": 0.95}
    # Padded to the minor unit; the confidence next to it stays a number.
    assert fields["total"]["value"] == {"value": "46.20", "confidence": 0.4}
    first, second = fields["line_items"]["value"]
    assert first["quantity"]["value"] == "5"
    assert first["unit_price"] == {"value": "9.15", "confidence": 0.95}
    assert first["total"]["value"] == "45.75"
    assert first["description"]["value"] == "Sourdough loaf"
    assert second["quantity"]["value"] == "2.5"
    # Garbage the model stored passes through for a later format check.
    assert second["unit_price"]["value"] == "abc"
    assert second["total"]["value"] is None
    # Machine metrics stay numbers.
    assert response.json()["extraction"]["cost_usd"] == 0.001


async def test_get_document_pads_to_the_document_currency(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    jpy = await _seed_extracted_document(
        db_session, {"currency": _leaf("jpy"), "total": _leaf(1500)}
    )
    kwd = await _seed_extracted_document(
        db_session, {"currency": _leaf("KWD"), "total": _leaf(1.5)}
    )
    no_currency = await _seed_extracted_document(db_session, {"total": _leaf(7)})
    extra_digits = await _seed_extracted_document(
        db_session, {"currency": _leaf("USD"), "total": _leaf(46.255)}
    )

    async def total_of(document: Document) -> str:
        body = (await client.get(f"/documents/{document.id}")).json()
        return next(
            f["value"]["value"]
            for f in body["extraction"]["fields"]
            if f["field_name"] == "total"
        )

    assert await total_of(jpy) == "1500"
    assert await total_of(kwd) == "1.500"
    assert await total_of(no_currency) == "7.00"
    # More digits than the currency allows are kept, so a misread shows.
    assert await total_of(extra_digits) == "46.255"


async def test_get_document_serves_a_v2_extraction_as_decimal_strings_with_evidence(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    document_id, _, _ = await seed_v2(db_session)

    body = (await client.get(f"/documents/{document_id}")).json()["extraction"]
    fields = {f["field_name"]: f for f in body["fields"]}

    assert body["prompt_version"] == "extract_v2"
    assert len(fields) == 20
    # Only the value turns into a decimal string; confidence and evidence stay.
    assert fields["discount"]["value"] == {
        "value": "10.00",
        "confidence": 0.95,
        "evidence": {"page": 1, "quote": "Discount -$10.00"},
    }
    assert fields["amount_due"]["value"]["value"] == "52.20"
    assert fields["tip"]["value"] == {"value": None, "confidence": 0.95, "evidence": None}
    assert fields["payment_card_last4"]["value"]["value"] == "4242"
    assert fields["due_date"]["value"]["value"] == "2026-01-31"
    row = fields["tax_lines"]["value"][0]
    assert row["label"]["value"] == "State tax"
    assert row["rate"] == {
        "value": "8",
        "confidence": 0.95,
        "evidence": {"page": 1, "quote": "State tax 8% on 90.00 7.20"},
    }
    assert row["base"]["value"] == "90.00"
    assert row["amount"]["value"] == "7.20"
    line = fields["line_items"]["value"][0]
    assert line["quantity"]["value"] == "2"
    assert line["total"] == {
        "value": "100.00",
        "confidence": 0.95,
        "evidence": {"page": 1, "quote": "Widget 100.00"},
    }
    assert fields["tax_lines"]["corrected_value"] is None
