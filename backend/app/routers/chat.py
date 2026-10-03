"""POST /documents/{id}/chat -- a conversation about one document.

Each turn is the /ask agent (app/agent/) with its tools scoped to this
document (ToolContext.document_ids) and the conversation's earlier turns
in front of the new question. Answers cite the page passages and the
extracted fields they came from; a field citation names the field, so the
document page can point at the row it came from.

A turn is stored as an AskRun with document_id and conversation_id set,
so it goes through everything /ask does: the per-client rate limit (the
same bucket), the daily spend cap, feedback (POST /ask/runs/{id}/feedback),
sampled groundedness grading, /stats and the traces. History is rebuilt
from those rows on every turn -- the client only holds the
conversation_id -- so a client can't put words in the assistant's mouth.
"""

import uuid

import anthropic
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.budget import enforce_daily_budget
from app.config import Settings, get_settings
from app.db import get_session, get_session_factory
from app.evals.online import earlier_turns
from app.extraction import ExtractionError
from app.models import AskRun, Document
from app.ratelimit import limit_ask
from app.retrieval.embeddings import Embedder, get_embedder
from app.routers.ask import (
    ChatScope,
    ask_response,
    get_anthropic_client,
    model_error,
    run_agent,
    stream_answer,
)
from app.schemas import AskResponse, ChatRequest, ConversationSummary

router = APIRouter()


async def _document(session: AsyncSession, document_id: uuid.UUID) -> Document:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return document


async def _chat_scope(
    session: AsyncSession, document_id: uuid.UUID, request: ChatRequest
) -> ChatScope:
    """The turn's scope: a new conversation, or the history of an existing
    one -- which must belong to this document."""
    document = await _document(session, document_id)
    if document.status != "extracted":
        raise HTTPException(
            status_code=409,
            detail=f"document is not extracted yet (status: {document.status})",
        )
    if request.conversation_id is None:
        return ChatScope(document_id=document_id, conversation_id=uuid.uuid4())
    owner = await session.scalar(
        select(AskRun.document_id).where(AskRun.conversation_id == request.conversation_id).limit(1)
    )
    if owner != document_id:
        raise HTTPException(status_code=404, detail="conversation not found for this document")
    history = await earlier_turns(session, request.conversation_id)
    return ChatScope(document_id, request.conversation_id, history)


@router.post(
    "/{document_id}/chat", response_model=AskResponse, dependencies=[Depends(limit_ask)]
)
async def chat(
    document_id: uuid.UUID,
    request: ChatRequest,
    session: AsyncSession = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
    settings: Settings = Depends(get_settings),
    client: anthropic.AsyncAnthropic | None = Depends(get_anthropic_client),
) -> AskResponse:
    scope = await _chat_scope(session, document_id, request)
    if client is None:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured")
    await enforce_daily_budget(session, settings)
    try:
        run = await run_agent(session, embedder, settings, client, request.question, scope)
    except ExtractionError as exc:
        raise model_error(exc) from exc
    return ask_response(run)


@router.post("/{document_id}/chat/stream", dependencies=[Depends(limit_ask)])
async def chat_stream(
    document_id: uuid.UUID,
    request: ChatRequest,
    session: AsyncSession = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
    settings: Settings = Depends(get_settings),
    client: anthropic.AsyncAnthropic | None = Depends(get_anthropic_client),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> StreamingResponse:
    """The chat turn as server-sent events -- the same events as POST
    /ask/stream. Everything that can refuse the turn (an unknown document
    or conversation, a missing key, the daily budget) answers with its
    normal status code before the stream starts."""
    scope = await _chat_scope(session, document_id, request)
    if client is None:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured")
    await enforce_daily_budget(session, settings)
    return StreamingResponse(
        stream_answer(session_factory, embedder, settings, client, request.question, scope),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{document_id}/chat/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    document_id: uuid.UUID,
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
) -> list[ConversationSummary]:
    """This document's conversations, most recently active first."""
    await _document(session, document_id)
    last_turn_at = func.max(AskRun.created_at)
    stmt = (
        select(
            AskRun.conversation_id,
            func.array_agg(aggregate_order_by(AskRun.question, AskRun.created_at))[1],
            func.count(),
            func.min(AskRun.created_at),
            last_turn_at,
        )
        .where(AskRun.document_id == document_id)
        .group_by(AskRun.conversation_id)
        .order_by(last_turn_at.desc(), AskRun.conversation_id)
        .limit(limit)
    )
    return [
        ConversationSummary(
            conversation_id=conversation_id,
            first_question=first_question,
            turns=turns,
            started_at=started_at,
            last_turn_at=last_at,
        )
        for conversation_id, first_question, turns, started_at, last_at in await session.execute(stmt)
    ]


@router.get(
    "/{document_id}/chat/conversations/{conversation_id}", response_model=list[AskResponse]
)
async def get_conversation(
    document_id: uuid.UUID,
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> list[AskResponse]:
    """Every turn of one conversation, oldest first, as stored -- with any
    feedback and grading since."""
    stmt = (
        select(AskRun)
        .where(AskRun.document_id == document_id, AskRun.conversation_id == conversation_id)
        .order_by(AskRun.created_at, AskRun.id)
    )
    runs = list((await session.execute(stmt)).scalars())
    if not runs:
        raise HTTPException(status_code=404, detail="conversation not found for this document")
    return [ask_response(run) for run in runs]
