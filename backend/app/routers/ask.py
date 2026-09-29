"""POST /ask -- answer a question about the documents with the tool-use
agent (app/agent/), with citations back to pages and extracted fields.

This is the one API route that calls the model, so unlike the rest of the
API it needs ANTHROPIC_API_KEY (see docker-compose.yml). Every call is
billed; the per-question step and dollar caps live in Settings.
"""

import anthropic
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.loop import answer_question
from app.agent.tools import ToolContext
from app.config import Settings, get_settings
from app.db import get_session
from app.extraction import ExtractionError, NonRetryableExtractionError, _build_client
from app.retrieval.embeddings import Embedder, get_embedder
from app.schemas import AskRequest, AskResponse

router = APIRouter()

_client: anthropic.AsyncAnthropic | None = None


def get_anthropic_client(
    settings: Settings = Depends(get_settings),
) -> anthropic.AsyncAnthropic | None:
    """One client for the API process: a client owns a connection pool,
    and building one per request would pay the TLS setup every time and
    leave the pools to the garbage collector. None when no API key is
    configured (the route answers 503, after request validation). A FastAPI
    dependency so tests can substitute a fake."""
    global _client
    if not settings.anthropic_api_key:
        return None
    if _client is None:
        _client = _build_client(settings)
    return _client


@router.post("", response_model=AskResponse)
async def ask(
    request: AskRequest,
    session: AsyncSession = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
    settings: Settings = Depends(get_settings),
    client: anthropic.AsyncAnthropic | None = Depends(get_anthropic_client),
) -> AskResponse:
    if client is None:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured")
    try:
        result = await answer_question(
            ToolContext(session=session, embedder=embedder), request.question, settings, client
        )
    except NonRetryableExtractionError as exc:
        raise HTTPException(status_code=502, detail=f"model request rejected: {exc}") from exc
    except ExtractionError as exc:
        # Transient (rate limit, overload, network): the client may retry.
        headers = (
            {"Retry-After": str(int(exc.retry_after_seconds))} if exc.retry_after_seconds else None
        )
        raise HTTPException(status_code=503, detail=f"model unavailable: {exc}", headers=headers) from exc
    return AskResponse.model_validate(result)
