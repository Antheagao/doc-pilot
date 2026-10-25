"""One row of every content model, for the tests of tenancy (test_tenancy.py,
test_rls.py): both layers are shown against the same data."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    EMBEDDING_DIM,
    AskRun,
    Document,
    DocumentChunk,
    DocumentPage,
    ExtractedField,
    Extraction,
    Job,
    Reconciliation,
)

UNIT_VECTOR = [1.0] + [0.0] * (EMBEDDING_DIM - 1)


async def populate(session: AsyncSession, tag: str) -> dict[type, uuid.UUID]:
    """One row of each content model in the organization the session is bound
    to, parents first (the composite foreign keys need them). Returns each
    row's id by model."""
    document = Document(
        filename=f"{tag}.png", mime_type="image/png", storage_path=f"/tmp/{tag}", status="extracted"
    )
    session.add(document)
    await session.flush()
    ask_run = AskRun(
        question=f"{tag}?",
        status="answered",
        answer="yes",
        citations=[],
        tool_calls=[],
        evidence="",
        steps=1,
        model="m",
        prompt_version="v",
    )
    extraction = Extraction(
        document_id=document.id,
        prompt_version="extract_v1",
        model="m",
        raw_response={},
        input_tokens=1,
        output_tokens=1,
        cost_usd=1,
        latency_ms=1,
    )
    page = DocumentPage(document_id=document.id, page_number=1, text=tag, source="gold")
    session.add_all([ask_run, extraction, page])
    await session.flush()
    session.add_all(
        [
            ExtractedField(extraction_id=extraction.id, field_name="total", value={}, confidence=0.9),
            Reconciliation(extraction_id=extraction.id, rules_version="v", status="pass", report={}),
            DocumentChunk(
                document_id=document.id,
                page_id=page.id,
                page_number=1,
                chunk_index=0,
                text=tag,
                char_start=0,
                char_end=len(tag),
                embedding=UNIT_VECTOR,
                embedding_model="m",
            ),
            Job(document_id=document.id, state="done"),
        ]
    )
    await session.flush()
    ids = {Document: document.id, AskRun: ask_run.id, Extraction: extraction.id, DocumentPage: page.id}
    for model, column in (
        (ExtractedField, "extraction_id"),
        (Reconciliation, "extraction_id"),
        (DocumentChunk, "document_id"),
        (Job, "document_id"),
    ):
        parent = {"extraction_id": extraction.id, "document_id": document.id}[column]
        ids[model] = (
            await session.execute(select(model.id).where(getattr(model, column) == parent))
        ).scalar_one()
    return ids
