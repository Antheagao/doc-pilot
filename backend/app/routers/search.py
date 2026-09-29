"""Search endpoint over the retrieval index (app/retrieval/).

GET /search?q=...&k=5&mode=hybrid -- top-k chunks with page-level
citations. mode is exposed (rather than hardwired to hybrid) so the
dense/lexical difference is inspectable from the API, the same comparison
the retrieval eval measures in aggregate.
"""

from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.retrieval.embeddings import Embedder, get_embedder
from app.retrieval.search import search
from app.schemas import SearchHitOut, SearchResponse

router = APIRouter()


@router.get("", response_model=SearchResponse)
async def search_documents(
    q: str = Query(..., min_length=1, max_length=500),
    k: int = Query(5, ge=1, le=50),
    mode: Literal["dense", "lexical", "hybrid"] = "hybrid",
    session: AsyncSession = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
) -> SearchResponse:
    hits = await search(session, q, embedder=embedder, k=k, mode=mode)
    return SearchResponse(
        query=q,
        mode=mode,
        embedding_model=embedder.name,
        results=[SearchHitOut.model_validate(hit) for hit in hits],
    )
