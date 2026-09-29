"""Hybrid search over document_chunks: dense (pgvector cosine), lexical
(Postgres full-text), or both fused with Reciprocal Rank Fusion.

Why both: the two fail differently on this kind of corpus. Dense
retrieval matches meaning ("light for my workspace" -> "LED Desk Lamp")
but is weak on exact tokens that carry no semantics -- an amount like
425.58, an invoice number, a SKU. Full-text search is exactly the
reverse. RRF combines the two *rankings* rather than their raw scores,
which live on incomparable scales (a cosine distance vs. a ts_rank), so
there is no weight to tune. The retrieval eval (app/evals/retrieval.py)
measures each mode separately so this claim is checked, not assumed.

Every hit carries its citation: document, page number, and the chunk's
char offsets into that page's stored text.
"""

import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import Text, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.models import Document, DocumentChunk
from app.retrieval.embeddings import Embedder

SearchMode = Literal["dense", "lexical", "hybrid"]
SEARCH_MODES: tuple[SearchMode, ...] = ("dense", "lexical", "hybrid")

# The standard RRF constant (Cormack et al., 2009): damps the advantage of
# the very top ranks so one list's #1 can't single-handedly outvote
# agreement between the two lists further down.
RRF_K = 60

# How many candidates each side contributes before fusion. Bounds the
# work per query; also the floor for hnsw.ef_search, because an HNSW
# index scan returns at most ef_search rows (pgvector's default is 40).
DEFAULT_CANDIDATES = 50


@dataclass
class SearchHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    # Mode-dependent: cosine similarity (dense), ts_rank_cd (lexical), or
    # the RRF sum (hybrid). Comparable within one result list only.
    score: float
    # 1-based rank on each side, None when that side didn't return the
    # chunk (or wasn't run). Surfaced so a result can explain itself.
    dense_rank: int | None
    lexical_rank: int | None


def _or_tsquery(query: str):
    """plainto_tsquery ANDs every term together, so a natural-language
    question ("where did I buy a desk lamp") only matches chunks that
    contain *every* non-stopword. Rewriting its & to | turns it into an
    OR query, and ts_rank_cd then rewards chunks that match more of the
    terms, closer together -- ranked retrieval rather than boolean
    filtering. Parsing still goes through plainto_tsquery, so user input
    is never interpreted as tsquery syntax.
    """
    and_query = cast(func.plainto_tsquery("english", query), Text)
    return func.to_tsquery("english", func.replace(and_query, " & ", " | "))


async def _dense_ranking(
    session: AsyncSession,
    query_vector: list[float],
    limit: int,
    document_ids: list[uuid.UUID] | None,
) -> list[tuple[uuid.UUID, float]]:
    await session.execute(
        select(func.set_config("hnsw.ef_search", str(max(limit, 40)), True))
    )
    distance = DocumentChunk.embedding.cosine_distance(query_vector)
    stmt = select(DocumentChunk.id, distance.label("distance")).order_by(distance).limit(limit)
    if document_ids is not None:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
    rows = (await session.execute(stmt)).all()
    return [(row.id, 1.0 - float(row.distance)) for row in rows]


async def _lexical_ranking(
    session: AsyncSession,
    query: str,
    limit: int,
    document_ids: list[uuid.UUID] | None,
) -> list[tuple[uuid.UUID, float]]:
    tsquery = _or_tsquery(query)
    # Normalization 1 divides by 1 + log(chunk length), so a long chunk
    # doesn't outrank a short one just by containing more words.
    rank = func.ts_rank_cd(DocumentChunk.tsv, tsquery, 1)
    stmt = (
        select(DocumentChunk.id, rank.label("rank"))
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.tsv.op("@@")(tsquery))
        # Equal ranks are common in full-text scoring; break ties on a
        # stable key so the same query always returns the same order.
        .order_by(
            rank.desc(),
            Document.filename,
            DocumentChunk.page_number,
            DocumentChunk.chunk_index,
        )
        .limit(limit)
    )
    if document_ids is not None:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
    rows = (await session.execute(stmt)).all()
    return [(row.id, float(row.rank)) for row in rows]


def reciprocal_rank_fusion(
    rankings: list[list[uuid.UUID]], k: int = RRF_K
) -> list[tuple[uuid.UUID, float]]:
    """Fuse ranked id lists: score(id) = sum over lists of 1 / (k + rank),
    rank 1-based. Ties keep first-seen order (dense before lexical), so
    the output is deterministic."""
    scores: dict[uuid.UUID, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


async def search(
    session: AsyncSession,
    query: str,
    *,
    embedder: Embedder,
    k: int = 5,
    mode: SearchMode = "hybrid",
    candidates: int = DEFAULT_CANDIDATES,
    document_ids: list[uuid.UUID] | None = None,
) -> list[SearchHit]:
    """Top-k chunks for `query`. document_ids restricts the search to
    those documents (the retrieval eval scopes itself to its own corpus
    this way); None searches everything."""
    if mode not in SEARCH_MODES:
        raise ValueError(f"unknown search mode {mode!r}")

    dense: list[tuple[uuid.UUID, float]] = []
    lexical: list[tuple[uuid.UUID, float]] = []
    if mode in ("dense", "hybrid"):
        query_vector = await run_in_threadpool(embedder.embed_query, query)
        dense = await _dense_ranking(session, query_vector, candidates, document_ids)
    if mode in ("lexical", "hybrid"):
        lexical = await _lexical_ranking(session, query, candidates, document_ids)

    if mode == "dense":
        ranked = dense[:k]
    elif mode == "lexical":
        ranked = lexical[:k]
    else:
        ranked = reciprocal_rank_fusion(
            [[chunk_id for chunk_id, _ in dense], [chunk_id for chunk_id, _ in lexical]]
        )[:k]

    if not ranked:
        return []

    dense_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense, start=1)}
    lexical_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(lexical, start=1)}

    ids = [chunk_id for chunk_id, _ in ranked]
    rows = (
        await session.execute(
            select(DocumentChunk, Document.filename)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.id.in_(ids))
        )
    ).all()
    by_id = {chunk.id: (chunk, filename) for chunk, filename in rows}

    hits = []
    for chunk_id, score in ranked:
        chunk, filename = by_id[chunk_id]
        hits.append(
            SearchHit(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                text=chunk.text,
                score=score,
                dense_rank=dense_ranks.get(chunk_id),
                lexical_rank=lexical_ranks.get(chunk_id),
            )
        )
    return hits
