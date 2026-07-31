import base64
import uuid

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, Job
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
