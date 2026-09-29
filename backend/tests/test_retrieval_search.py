"""Indexing + search against the real Postgres (pgvector) test database.

Uses the HashingEmbedder throughout: deterministic, offline, and lexical,
so these tests pin down the plumbing (chunk rows, offsets, fusion,
scoping, the API shape) rather than embedding quality -- that is what the
retrieval eval measures with the real model.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.models import Document, DocumentChunk, DocumentPage
from app.retrieval.embeddings import HashingEmbedder, get_embedder
from app.retrieval.indexing import ChunkingConfig, PageText, index_document_pages
from app.retrieval.search import reciprocal_rank_fusion, search

EMBEDDER = HashingEmbedder()
WHOLE_PAGE = ChunkingConfig(max_chars=0, overlap_chars=0, context_headers=True)
SMALL = ChunkingConfig(max_chars=80, overlap_chars=20, context_headers=True)

DOCS = {
    "hardware.png": [
        (
            "Ironclad Hardware Supply\n78 Foundry Ave, Detroit, MI\n\n"
            "Steel Hex Bolts (box of 50)   2   $7.10   $14.20\n"
            "Extension Cord 25ft           1  $18.65   $18.65\n\nTotal: $35.66\n"
        )
    ],
    "bakery.png": [
        (
            "Cobblestone Bakery\n15 Market Sq, Savannah, GA\n\n"
            "Herbal Tea Sampler            1   $8.05    $8.05\n"
            "Ceramic Mug 12oz              2   $6.00   $12.00\n\nTotal: $21.71\n"
        )
    ],
    "invoice.pdf": [
        "Northgate Office Outfitters\nInvoice 4471\n\nLED Desk Lamp   2  $19.70  $39.40\n",
        "Northgate Office Outfitters\nPage 2\n\nYoga Mat        1  $22.70  $22.70\n\nTotal: $62.10\n",
    ],
}


async def _index_corpus(
    db_session: AsyncSession, config: ChunkingConfig = WHOLE_PAGE
) -> dict[str, uuid.UUID]:
    ids = {}
    for filename, pages in DOCS.items():
        document = Document(
            filename=filename, mime_type="image/png", storage_path=f"/tmp/{filename}",
            status="extracted",
        )
        db_session.add(document)
        await db_session.flush()
        ids[filename] = document.id
        await index_document_pages(
            db_session,
            document.id,
            [PageText(page_number=n, text=text, source="gold") for n, text in enumerate(pages, 1)],
            EMBEDDER,
            config,
        )
    return ids


async def _search(db_session, query, ids, **kwargs):
    return await search(
        db_session, query, embedder=EMBEDDER, document_ids=list(ids.values()), **kwargs
    )


async def test_indexing_writes_pages_and_chunks_with_exact_offsets(
    db_session: AsyncSession,
) -> None:
    ids = await _index_corpus(db_session, SMALL)

    pages = (
        await db_session.execute(
            select(DocumentPage).where(DocumentPage.document_id == ids["invoice.pdf"])
        )
    ).scalars().all()
    assert sorted(page.page_number for page in pages) == [1, 2]

    chunks = (
        await db_session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id.in_(ids.values()))
        )
    ).scalars().all()
    page_text = {(p.document_id, p.page_number): p.text for p in pages}
    for chunk in chunks:
        assert len(chunk.embedding) == EMBEDDER.dim
        assert chunk.embedding_model == "hashing-v1"
        if chunk.document_id == ids["invoice.pdf"]:
            assert page_text[(chunk.document_id, chunk.page_number)][
                chunk.char_start : chunk.char_end
            ] == chunk.text
            assert chunk.context == (
                f"Northgate Office Outfitters (page {chunk.page_number} of 2)"
            )


async def test_reindexing_replaces_rather_than_duplicates(db_session: AsyncSession) -> None:
    ids = await _index_corpus(db_session)
    document_id = ids["bakery.png"]

    await index_document_pages(
        db_session,
        document_id,
        [PageText(page_number=1, text="Cobblestone Bakery\nSourdough Loaf  1  $7.00", source="gold")],
        EMBEDDER,
        WHOLE_PAGE,
    )

    count = (
        await db_session.execute(
            select(func.count()).select_from(DocumentChunk).where(
                DocumentChunk.document_id == document_id
            )
        )
    ).scalar_one()
    assert count == 1
    hits = await _search(db_session, "sourdough", ids, mode="lexical")
    assert [hit.filename for hit in hits] == ["bakery.png"]
    assert await _search(db_session, "herbal tea sampler", ids, mode="lexical") == []


async def test_blank_pages_index_no_chunks(db_session: AsyncSession) -> None:
    document = Document(filename="blank.png", mime_type="image/png", storage_path="/tmp/b", status="extracted")
    db_session.add(document)
    await db_session.flush()

    written = await index_document_pages(
        db_session, document.id, [PageText(page_number=1, text="", source="gold")], EMBEDDER, SMALL
    )

    assert written == 0


async def test_lexical_search_uses_or_semantics_for_natural_language(
    db_session: AsyncSession,
) -> None:
    """plainto_tsquery alone would AND every term, so a question with
    words the receipt doesn't contain would match nothing."""
    ids = await _index_corpus(db_session)

    hits = await _search(db_session, "where did I buy the extension cord for the garage", ids, mode="lexical")

    assert hits[0].filename == "hardware.png"
    assert hits[0].lexical_rank == 1 and hits[0].dense_rank is None


async def test_lexical_search_matches_exact_amount_tokens(db_session: AsyncSession) -> None:
    ids = await _index_corpus(db_session)

    hits = await _search(db_session, "receipt that came to $21.71", ids, mode="lexical")

    assert hits[0].filename == "bakery.png"


async def test_stopword_only_query_returns_nothing_lexically(db_session: AsyncSession) -> None:
    ids = await _index_corpus(db_session)

    assert await _search(db_session, "the and of", ids, mode="lexical") == []


async def test_dense_search_ranks_by_cosine_similarity(db_session: AsyncSession) -> None:
    ids = await _index_corpus(db_session)

    hits = await _search(db_session, "Ceramic Mug", ids, mode="dense", k=3)

    assert hits[0].filename == "bakery.png"
    assert hits[0].dense_rank == 1 and hits[0].lexical_rank is None
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


async def test_hybrid_search_reports_both_ranks(db_session: AsyncSession) -> None:
    ids = await _index_corpus(db_session)

    hits = await _search(db_session, "LED desk lamp", ids, mode="hybrid")

    assert hits[0].filename == "invoice.pdf"
    assert hits[0].page_number == 1
    assert hits[0].dense_rank is not None and hits[0].lexical_rank is not None


async def test_search_cites_the_right_page_of_a_multi_page_document(
    db_session: AsyncSession,
) -> None:
    ids = await _index_corpus(db_session, SMALL)

    hits = await _search(db_session, "yoga mat", ids, mode="lexical", k=1)

    assert hits[0].filename == "invoice.pdf"
    assert hits[0].page_number == 2
    assert "Yoga Mat" in hits[0].text


async def test_document_ids_scopes_the_search(db_session: AsyncSession) -> None:
    ids = await _index_corpus(db_session)

    hits = await search(
        db_session,
        "Ceramic Mug",
        embedder=EMBEDDER,
        mode="hybrid",
        document_ids=[ids["hardware.png"]],
    )

    assert {hit.filename for hit in hits} <= {"hardware.png"}


async def test_unknown_mode_is_rejected(db_session: AsyncSession) -> None:
    with pytest.raises(ValueError):
        await search(db_session, "x", embedder=EMBEDDER, mode="fuzzy")  # type: ignore[arg-type]


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    fused = reciprocal_rank_fusion([[a, b], [c, b]], k=60)

    assert fused[0][0] == b  # ranked 2nd by both beats 1st by only one
    assert fused[0][1] == pytest.approx(2 / 62)
    assert {item for item, _ in fused} == {a, b, c}


async def test_search_endpoint_returns_citations(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await _index_corpus(db_session, SMALL)
    app.dependency_overrides[get_embedder] = lambda: EMBEDDER

    response = await client.get("/search", params={"q": "yoga mat", "k": 3, "mode": "lexical"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "lexical"
    assert body["embedding_model"] == "hashing-v1"
    top = body["results"][0]
    assert top["filename"] == "invoice.pdf"
    assert top["page_number"] == 2
    assert set(top) >= {"document_id", "char_start", "char_end", "text", "score"}


async def test_search_endpoint_validates_parameters(client: AsyncClient) -> None:
    app.dependency_overrides[get_embedder] = lambda: EMBEDDER

    assert (await client.get("/search", params={"q": ""})).status_code == 422
    assert (await client.get("/search", params={"q": "x", "k": 0})).status_code == 422
    assert (await client.get("/search", params={"q": "x", "mode": "fuzzy"})).status_code == 422


async def _index_texts(db_session: AsyncSession, texts: dict[str, str]) -> dict[str, uuid.UUID]:
    ids = {}
    for filename, text in texts.items():
        document = Document(filename=filename, mime_type="image/png", storage_path="/tmp/x", status="extracted")
        db_session.add(document)
        await db_session.flush()
        ids[filename] = document.id
        await index_document_pages(
            db_session, document.id, [PageText(1, text, "gold")], EMBEDDER, WHOLE_PAGE
        )
    return ids


async def test_idf_ranks_a_rare_term_above_a_common_one(db_session: AsyncSession) -> None:
    """ts_rank_cd scores both documents alike -- each matches one query
    term once. IDF knows "receipt" is on every document and 425.58 on
    one."""
    ids = await _index_texts(
        db_session,
        {
            "a-generic.png": "RECEIPT\nItem  1  $5.85\nTotal: $91.00",
            "b-generic.png": "RECEIPT\nItem  2  $9.10\nTotal: $12.00",
            "c-generic.png": "RECEIPT\nItem  3  $4.20\nTotal: $8.00",
            "z-target.png": "Northgate Office Outfitters\nYoga Mat  2  $22.70\nTotal: $425.58",
        },
    )

    idf = await _search(db_session, "receipt 425.58", ids, mode="lexical", lexical_scoring="idf")
    ts_rank = await _search(db_session, "receipt 425.58", ids, mode="lexical", lexical_scoring="ts_rank")

    assert idf[0].filename == "z-target.png"
    assert ts_rank[0].filename != "z-target.png"  # the failure IDF fixes


async def test_amount_aliases_match_typed_amounts(db_session: AsyncSession) -> None:
    ids = await _index_texts(
        db_session,
        {
            "berlin.png": "Lindenplatz Bakery\nTotal: 27,82 EUR",
            "big.png": "Northgate Office Outfitters\nTotal: $1,234.56",
        },
    )

    for query, expected in (
        ("27.82 euros", "berlin.png"),
        ("27,82", "berlin.png"),
        ("1234.56", "big.png"),
        ("$1,234.56", "big.png"),
    ):
        hits = await _search(db_session, query, ids, mode="lexical")
        assert hits and hits[0].filename == expected, query
    chunk = (
        await db_session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == ids["berlin.png"])
        )
    ).scalar_one()
    assert chunk.search_aliases == "27.82 euro"
    assert "27.82" not in chunk.text  # aliases are search-only, never cited


async def test_chunks_reindexed_mid_search_are_skipped_not_a_500(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.retrieval import search as search_module

    ids = await _index_corpus(db_session)
    real = search_module._lexical_ranking

    async def with_a_vanished_chunk(*args, **kwargs):
        return [(uuid.uuid4(), 9.9), *await real(*args, **kwargs)]

    monkeypatch.setattr(search_module, "_lexical_ranking", with_a_vanished_chunk)

    hits = await _search(db_session, "ceramic mug", ids, mode="lexical")

    assert hits and hits[0].filename == "bakery.png"
