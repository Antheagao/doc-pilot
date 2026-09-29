"""POST /ask -- answer a question about the documents with the tool-use
agent (app/agent/), with citations back to pages and extracted fields.

This is the one API route that calls the model, so unlike the rest of the
API it needs ANTHROPIC_API_KEY (see docker-compose.yml). Every call is
billed; the per-question step and dollar caps live in Settings.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.loop import answer_question
from app.agent.tools import ToolContext
from app.config import Settings, get_settings
from app.db import get_session
from app.extraction import ExtractionError, NonRetryableExtractionError
from app.retrieval.embeddings import Embedder, get_embedder
from app.schemas import AskRequest, AskResponse

router = APIRouter()


@router.post("", response_model=AskResponse)
async def ask(
    request: AskRequest,
    session: AsyncSession = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
    settings: Settings = Depends(get_settings),
) -> AskResponse:
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured")
    try:
        result = await answer_question(
            ToolContext(session=session, embedder=embedder), request.question, settings
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
