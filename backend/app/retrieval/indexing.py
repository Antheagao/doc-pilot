"""Build the retrieval index for a document: pages -> chunks -> embeddings
-> document_pages/document_chunks rows.

index_document_pages takes page texts from any source, so the production
path (app/retrieval/index_job.py: VLM transcription) and the retrieval
eval (known-correct "gold" page text, no model call) index through exactly
the same chunking and embedding code. Like every handler-side writer in
this codebase it adds and flushes but never commits: the caller owns the
transaction, so a document's index is replaced atomically or not at all.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.langfuse_link import observation_attributes
from app.models import DocumentChunk, DocumentPage
from app.retrieval.chunking import chunk_page, context_header, document_title
from app.retrieval.embeddings import Embedder
from app.retrieval.normalize import search_aliases
from app.telemetry import (
    DOCPILOT_DOCUMENT_ID,
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    tracer,
)


@dataclass
class PageText:
    """One page of text to index, plus what producing it cost (zero for
    gold text; a transcription's own tokens/cost/latency otherwise)."""

    page_number: int
    text: str
    source: str
    model: str | None = None
    prompt_version: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int
    overlap_chars: int
    context_headers: bool

    @classmethod
    def from_settings(cls, settings: Settings) -> "ChunkingConfig":
        return cls(
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
            context_headers=settings.chunk_context_headers,
        )


async def index_document_pages(
    session: AsyncSession,
    document_id: uuid.UUID,
    pages: list[PageText],
    embedder: Embedder,
    config: ChunkingConfig,
) -> int:
    """Replace the document's pages and chunks with ones built from
    `pages`, and return the number of chunks written. Idempotent:
    re-indexing (a new embedding model, new chunking settings, a fresh
    transcription) deletes the previous rows first, in the same
    transaction as the inserts.

    All of a document's chunks are embedded in one batch call, off the
    event loop -- embedding is CPU-bound, and the worker shares its loop
    with everything else in the process.
    """
    await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    await session.execute(delete(DocumentPage).where(DocumentPage.document_id == document_id))

    pages = sorted(pages, key=lambda page: page.page_number)
    if not pages:
        return 0

    page_rows = [
        DocumentPage(
            document_id=document_id,
            page_number=page.page_number,
            text=page.text,
            source=page.source,
            model=page.model,
            prompt_version=page.prompt_version,
            input_tokens=page.input_tokens,
            output_tokens=page.output_tokens,
            cost_usd=page.cost_usd,
            latency_ms=page.latency_ms,
        )
        for page in pages
    ]
    session.add_all(page_rows)
    await session.flush()  # assigns page ids for the chunk FKs

    title = document_title(pages[0].text) if config.context_headers else None
    planned: list[tuple[DocumentPage, object, str | None]] = []
    for page, page_row in zip(pages, page_rows, strict=True):
        header = (
            context_header(title, page.page_number, len(pages))
            if config.context_headers
            else None
        )
        for chunk in chunk_page(
            page.text,
            page.page_number,
            max_chars=config.max_chars,
            overlap_chars=config.overlap_chars,
        ):
            planned.append((page_row, chunk, header))

    if not planned:
        return 0

    texts = [f"{header}\n{chunk.text}" if header else chunk.text for _, chunk, header in planned]
    with tracer().start_as_current_span(
        f"embeddings {embedder.name}",
        attributes={
            GEN_AI_OPERATION_NAME: "embeddings",
            GEN_AI_REQUEST_MODEL: embedder.name,
            DOCPILOT_DOCUMENT_ID: str(document_id),
            "docpilot.chunk_count": len(texts),
            **observation_attributes("embedding"),
        },
    ):
        vectors = await run_in_threadpool(embedder.embed_documents, texts)

    session.add_all(
        DocumentChunk(
            document_id=document_id,
            page_id=page_row.id,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            context=header,
            search_aliases=search_aliases(chunk.text),
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            embedding=vector,
            embedding_model=embedder.name,
        )
        for (page_row, chunk, header), vector in zip(planned, vectors, strict=True)
    )
    await session.flush()
    return len(planned)
