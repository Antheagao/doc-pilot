"""Worker handler for Job.kind == 'index' (dispatched by app/worker.py):
make one document searchable.

Enqueued by app.extraction.process_document_job, in the same transaction
that persists a successful extraction, so every extracted document gets
exactly one index job. Kept a separate job rather than a step at the end
of extraction so the two fail independently: a transcription 429 retries
the transcription alone, and never re-runs (and re-bills) an extraction
that already succeeded.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.extraction import ExtractionError
from app.models import Document, Job
from app.retrieval.embeddings import get_embedder
from app.retrieval.indexing import ChunkingConfig, index_document_pages
from app.transcription import transcribe_document


async def process_index_job(session: AsyncSession, job: Job) -> None:
    """Worker handler for an 'index' job: transcribe every page of the
    job's document, then index the text.

    Same shape as app.extraction.process_document_job: the session is
    rolled back (releasing its connection) before the multi-second VLM
    calls, and the document re-fetched after. A failure after the
    transcription succeeded is re-raised as a retryable ExtractionError
    carrying the already-billed tokens/cost, so that spend is never
    invisible in the job's last_error.
    """
    document = await session.get(Document, job.document_id)
    if document is None:
        raise ExtractionError(f"document {job.document_id} not found")

    document_id = document.id
    storage_path = document.storage_path
    mime_type = document.mime_type

    await session.rollback()

    pages = await transcribe_document(storage_path, mime_type)

    if await session.get(Document, document_id) is None:
        raise ExtractionError(f"document {document_id} vanished during transcription")

    try:
        await index_document_pages(
            session,
            document_id,
            pages,
            get_embedder(),
            ChunkingConfig.from_settings(get_settings()),
        )
    except Exception as exc:
        input_tokens = sum(page.input_tokens for page in pages)
        output_tokens = sum(page.output_tokens for page in pages)
        cost_usd = sum(page.cost_usd for page in pages)
        raise ExtractionError(
            "indexing failed after a successful, billed transcription "
            f"(input_tokens={input_tokens}, output_tokens={output_tokens}, "
            f"cost_usd={cost_usd:.6f}): {exc}",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        ) from exc
