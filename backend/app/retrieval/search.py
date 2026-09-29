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

import math
import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import Text, case, cast, func, literal, select, true
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.models import Document, DocumentChunk
from app.retrieval.embeddings import Embedder
from app.retrieval.normalize import normalize_query
from app.telemetry import GEN_AI_DATA_SOURCE_ID, GEN_AI_OPERATION_NAME, tracer

SearchMode = Literal["dense", "lexical", "hybrid"]
LexicalScoring = Literal["idf", "ts_rank"]
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

    The rewritten text is cast straight to tsquery, not re-parsed with
    to_tsquery('english', ...): its lexemes are already stemmed, and
    stemming a stem again isn't a no-op -- 'ashevill' becomes 'ashevil',
    'basebal' becomes 'baseb', 'purchas' becomes 'purcha' -- so a query
    for "Asheville" could never match a document that says Asheville.
    The cast takes lexemes as they are, exactly as the stored tsvectors
    hold them. (plainto_tsquery's text output quotes every lexeme, so
    the cast can't see operators in user input either.)
    """
    and_query = cast(func.plainto_tsquery("english", query), Text)
    return cast(func.replace(and_query, " & ", " | "), TSQUERY)


# Extra rows the dense index scan fetches so that ties at the cutoff are
# broken deterministically rather than by scan order.
_TIE_MARGIN = 10


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
    # The inner query is the shape the HNSW index serves (ORDER BY distance
    # LIMIT n, filters applied to what it yields). A chunk with cosine
    # similarity <= 0 shares nothing with the query and isn't a match.
    candidates = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.page_number,
            DocumentChunk.chunk_index,
            distance.label("distance"),
        )
        .where(distance < 1.0)
        .order_by(distance)
        .limit(limit + _TIE_MARGIN)
    )
    if document_ids is not None:
        candidates = candidates.where(DocumentChunk.document_id.in_(document_ids))
    ranked = candidates.subquery()
    # Equal distances are real (identical chunk text embeds identically),
    # and Postgres returns ties in whatever order the scan produced them,
    # so the same query could rank differently from one call to the next.
    # Ties are broken here, outside the index scan -- a second sort key on
    # the scan itself would stop Postgres using the HNSW index at all.
    stmt = (
        select(ranked.c.id, ranked.c.distance)
        .join(Document, Document.id == ranked.c.document_id)
        .order_by(ranked.c.distance, Document.filename, ranked.c.page_number, ranked.c.chunk_index)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [(row.id, 1.0 - float(row.distance)) for row in rows]


async def _lexical_ranking(
    session: AsyncSession,
    query: str,
    limit: int,
    document_ids: list[uuid.UUID] | None,
    scoring: LexicalScoring = "idf",
) -> list[tuple[uuid.UUID, float]]:
    """Full-text ranking over chunks matching any query term.

    scoring="ts_rank" is Postgres's own ts_rank_cd, which has no notion
    of how rare a term is: a query term that appears in every receipt
    ("total", "receipt") counts exactly as much as one that appears in a
    single chunk ("425.58"). scoring="idf" (the default, chosen by the
    retrieval eval -- README "Retrieval eval") scores each chunk by the
    summed IDF of the query terms it contains, BM25-style, with ts_rank_cd
    only breaking ties. Chunks are short enough that term frequency and
    length normalization barely vary, which is why IDF alone recovers most
    of what BM25 would.
    """
    # Amounts rewritten canonically to meet the chunks' search_aliases
    # (app/retrieval/normalize.py); the dense side embeds the raw query.
    query = normalize_query(query)
    tsquery = _or_tsquery(query)
    scope = DocumentChunk.document_id.in_(document_ids) if document_ids is not None else true()
    # Normalization 1 divides by 1 + log(chunk length), so a long chunk
    # doesn't outrank a short one just by containing more words.
    ts_rank = func.ts_rank_cd(DocumentChunk.tsv, tsquery, 1)

    if scoring == "idf":
        weights = await _term_idf(session, query, scope)
        if not weights:
            return []
        score = sum(
            (
                case((DocumentChunk.tsv.op("@@")(cast(func.quote_literal(lexeme), TSQUERY)), idf), else_=0.0)
                for lexeme, idf in weights.items()
            ),
            start=literal(0.0),
        )
    else:
        score = ts_rank

    stmt = (
        select(DocumentChunk.id, score.label("score"))
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.tsv.op("@@")(tsquery), scope)
        # Equal scores are common in full-text scoring; break ties on
        # ts_rank_cd, then a stable key, so the same query always returns
        # the same order.
        .order_by(
            score.desc(),
            ts_rank.desc(),
            Document.filename,
            DocumentChunk.page_number,
            DocumentChunk.chunk_index,
        )
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [(row.id, float(row.score)) for row in rows]


async def _term_idf(session: AsyncSession, query: str, scope) -> dict[str, float]:
    """BM25's IDF for each distinct query lexeme, over the chunks in scope:
    ln((N - df + 0.5) / (df + 0.5) + 1). Lexemes come from Postgres's own
    parser, so they match the stored tsvectors exactly (same stemming, same
    stopwords); each document frequency is a GIN-indexed count."""
    lexemes = (
        await session.execute(
            select(func.unnest(func.tsvector_to_array(func.to_tsvector("english", query))))
        )
    ).scalars().all()
    if not lexemes:
        return {}
    total = (
        await session.execute(select(func.count()).select_from(DocumentChunk).where(scope))
    ).scalar_one()
    counts = (
        await session.execute(
            select(
                *(
                    func.count()
                    .filter(DocumentChunk.tsv.op("@@")(cast(func.quote_literal(lexeme), TSQUERY)))
                    .label(f"df{i}")
                    for i, lexeme in enumerate(lexemes)
                )
            )
            .select_from(DocumentChunk)
            .where(DocumentChunk.tsv.op("@@")(_or_tsquery(query)), scope)
        )
    ).one()
    return {
        lexeme: math.log((total - df + 0.5) / (df + 0.5) + 1)
        for lexeme, df in zip(lexemes, counts, strict=True)
    }


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


async def _search(
    session: AsyncSession,
    query: str,
    *,
    embedder: Embedder,
    k: int = 5,
    mode: SearchMode = "hybrid",
    candidates: int = DEFAULT_CANDIDATES,
    document_ids: list[uuid.UUID] | None = None,
    lexical_scoring: LexicalScoring = "idf",
) -> list[SearchHit]:
    if mode not in SEARCH_MODES:
        raise ValueError(f"unknown search mode {mode!r}")

    dense: list[tuple[uuid.UUID, float]] = []
    lexical: list[tuple[uuid.UUID, float]] = []
    if mode in ("dense", "hybrid"):
        query_vector = await run_in_threadpool(embedder.embed_query, query)
        dense = await _dense_ranking(session, query_vector, candidates, document_ids)
    if mode in ("lexical", "hybrid"):
        lexical = await _lexical_ranking(
            session, query, candidates, document_ids, lexical_scoring
        )

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
        if chunk_id not in by_id:
            # Re-indexed (deleted and re-inserted) between the ranking and
            # this fetch; the document's new chunks show up next search.
            continue
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


async def search(
    session: AsyncSession,
    query: str,
    *,
    embedder: Embedder,
    k: int = 5,
    mode: SearchMode = "hybrid",
    candidates: int = DEFAULT_CANDIDATES,
    document_ids: list[uuid.UUID] | None = None,
    lexical_scoring: LexicalScoring = "idf",
) -> list[SearchHit]:
    """Top-k chunks for `query`. document_ids restricts the search to
    those documents (the retrieval eval scopes itself to its own corpus
    this way); None searches everything.

    Traced as a GenAI `retrieval` span. The query text is deliberately
    not recorded (see app/telemetry.py on content capture); the mode,
    k, and how many hits each side contributed are.
    """
    with tracer().start_as_current_span(
        "retrieval document_chunks",
        attributes={
            GEN_AI_OPERATION_NAME: "retrieval",
            GEN_AI_DATA_SOURCE_ID: "document_chunks",
            "docpilot.search.mode": mode,
            "docpilot.search.k": k,
            "docpilot.search.lexical_scoring": lexical_scoring,
        },
    ) as span:
        hits = await _search(
            session,
            query,
            embedder=embedder,
            k=k,
            mode=mode,
            candidates=candidates,
            document_ids=document_ids,
            lexical_scoring=lexical_scoring,
        )
        span.set_attribute("docpilot.search.results", len(hits))
        span.set_attribute(
            "docpilot.search.results_with_dense_rank",
            sum(hit.dense_rank is not None for hit in hits),
        )
        span.set_attribute(
            "docpilot.search.results_with_lexical_rank",
            sum(hit.lexical_rank is not None for hit in hits),
        )
        return hits
