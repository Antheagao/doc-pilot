import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings, resolve_upload_dir
from app.db import get_session
from app.models import Document, ExtractedField, Extraction, Job
from app.schemas import (
    DocumentCreateResponse,
    DocumentDetail,
    DocumentListItem,
    ExtractedFieldOut,
    ExtractionOut,
)

router = APIRouter()

ALLOWED_MIME_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "application/pdf": ".pdf",
}

MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB
CHUNK_SIZE = 1024 * 1024


def _safe_ext(mime_type: str) -> str:
    """Derive the storage extension from the validated mime type.

    Never trust the client-supplied filename for this: content_type has
    already been checked against ALLOWED_MIME_TYPES, whereas a filename
    can claim anything (e.g. "evil.exe" uploaded with
    Content-Type: image/png would otherwise be stored as {uuid}.exe).
    """
    return ALLOWED_MIME_TYPES[mime_type]


async def _read_limited(file: UploadFile, max_size: int) -> bytes:
    """Read an upload in chunks, rejecting it as soon as it exceeds
    max_size.

    This is a second line of defense, not the primary guard: Starlette's
    multipart parser spools file parts to a temp file as it reads them,
    ahead of any application code (including UploadFile.read()) — so by
    the time this runs, an oversized body may already be fully on disk.
    The actual early rejection happens in MaxBodySizeMiddleware
    (app/main.py), which aborts the request at the ASGI layer before the
    multipart parser gets a chance to spool it. This check remains as a
    cheap backstop in case that cap and this one ever drift apart.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_size:
            raise HTTPException(status_code=413, detail="File too large")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("", response_model=DocumentCreateResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Document:
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415, detail=f"Unsupported media type: {file.content_type}"
        )

    content = await _read_limited(file, MAX_UPLOAD_SIZE)

    document_id = uuid.uuid4()
    ext = _safe_ext(file.content_type)
    upload_dir = resolve_upload_dir(settings)
    storage_path = upload_dir / f"{document_id}{ext}"
    await run_in_threadpool(upload_dir.mkdir, parents=True, exist_ok=True)
    await run_in_threadpool(storage_path.write_bytes, content)

    document = Document(
        id=document_id,
        filename=file.filename or f"{document_id}{ext}",
        mime_type=file.content_type,
        storage_path=str(storage_path),
        status="uploaded",
    )
    job = Job(document_id=document_id, state="pending")

    try:
        session.add_all([document, job])
        await session.commit()
    except Exception:
        # Don't leave an orphaned file on disk if the DB insert failed.
        # Only the failed-commit path may unlink: once the rows are
        # committed, the file must survive (a job now references it).
        storage_path.unlink(missing_ok=True)
        raise
    await session.refresh(document)

    return document


@router.get("", response_model=list[DocumentListItem])
async def list_documents(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[Document]:
    stmt = (
        select(Document)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> DocumentDetail:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    extraction_stmt = (
        select(Extraction)
        .where(Extraction.document_id == document_id)
        .order_by(Extraction.created_at.desc())
        .limit(1)
    )
    extraction = (await session.execute(extraction_stmt)).scalars().first()

    extraction_out = None
    if extraction is not None:
        fields_stmt = select(ExtractedField).where(
            ExtractedField.extraction_id == extraction.id
        )
        fields = (await session.execute(fields_stmt)).scalars().all()
        extraction_out = ExtractionOut(
            id=extraction.id,
            prompt_version=extraction.prompt_version,
            model=extraction.model,
            cost_usd=extraction.cost_usd,
            latency_ms=extraction.latency_ms,
            input_tokens=extraction.input_tokens,
            output_tokens=extraction.output_tokens,
            created_at=extraction.created_at,
            fields=[ExtractedFieldOut.model_validate(f) for f in fields],
        )

    return DocumentDetail(
        id=document.id,
        filename=document.filename,
        mime_type=document.mime_type,
        status=document.status,
        created_at=document.created_at,
        extraction=extraction_out,
    )


@router.get("/{document_id}/file")
async def get_document_file(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    storage_path = Path(document.storage_path)
    if not storage_path.is_file():
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        storage_path, media_type=document.mime_type, filename=document.filename
    )
