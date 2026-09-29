"""Seed the labeled eval corpus into the database as if the pipeline had
processed it perfectly: one Document per labeled doc, an Extraction whose
fields are exactly the label, and the gold page text indexed for search.

Used by evals that measure a stage *downstream* of extraction and OCR --
the retrieval eval indexes gold text for the same reason -- so a miss is
the stage's own miss, not an upstream error passed along. Callers own the
transaction (the evals roll it back); nothing here commits.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.evals.dataset import EvalCase
from app.evals.retrieval import GoldDoc
from app.extraction import TOP_LEVEL_FIELDS
from app.models import Document, ExtractedField, Extraction
from app.retrieval.embeddings import Embedder
from app.retrieval.indexing import ChunkingConfig, PageText, index_document_pages

GOLD_MODEL = "gold-labels"


def _leaf(value) -> dict:
    return {"value": value, "confidence": 1.0}


async def seed_labeled_corpus(
    session: AsyncSession,
    cases: list[EvalCase],
    corpus: list[GoldDoc],
    embedder: Embedder,
    chunking: ChunkingConfig,
) -> dict[str, uuid.UUID]:
    """Seed every case that has gold text; returns doc_id -> Document.id."""
    gold_text = {doc.doc_id: doc.text for doc in corpus}
    ids: dict[str, uuid.UUID] = {}
    for case in cases:
        if case.doc_id not in gold_text:
            continue
        document = Document(
            filename=case.doc_id,
            mime_type=case.mime_type,
            storage_path=str(case.image_path),
            status="extracted",
        )
        session.add(document)
        await session.flush()
        extraction = Extraction(
            document_id=document.id,
            prompt_version="labels",
            model=GOLD_MODEL,
            raw_response={},
            input_tokens=0,
            output_tokens=0,
            cost_usd=0,
            latency_ms=0,
        )
        session.add(extraction)
        await session.flush()
        for name in TOP_LEVEL_FIELDS:
            value = case.fields[name]
            stored = (
                [{key: _leaf(cell) for key, cell in row.items()} for row in value or []]
                if name == "line_items"
                else _leaf(value)
            )
            session.add(
                ExtractedField(
                    extraction_id=extraction.id,
                    field_name=name,
                    value=stored,
                    confidence=1.0,
                    needs_review=False,
                )
            )
        await index_document_pages(
            session,
            document.id,
            [PageText(page_number=1, text=gold_text[case.doc_id], source="gold")],
            embedder,
            chunking,
        )
        ids[case.doc_id] = document.id
    await session.flush()
    return ids
