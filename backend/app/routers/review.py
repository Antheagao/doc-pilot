"""Human review queue endpoints.

Fields whose extraction confidence fell below Settings.review_threshold
were flagged needs_review=True at persistence time (see
app/extraction.py). This router exposes that set as a work queue:

- GET  /review/queue        -- pending fields (needs_review, not yet
                               reviewed), oldest extraction first, joined
                               with document context
- GET  /review/queue/count  -- pending total, cheap enough to poll for a
                               nav badge
- POST /review/fields/{id}/resolve -- approve the extracted value or
                               supply a correction

Resolution is field-level on purpose (see README "Key decisions"): a
document with one shaky field shouldn't cost a human a whole-document
re-check. The original extracted value is never overwritten -- a
correction lands in corrected_value beside it, keeping the audit trail
of what the model actually said and giving future eval tooling a labeled
(document, field, human answer) triple to harvest.
"""

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Document, ExtractedField, Extraction
from app.schemas import (
    ExtractedFieldOut,
    ReviewQueueCount,
    ReviewQueueItem,
    ReviewResolveRequest,
)

router = APIRouter()

# The queue predicate, shared by the list and count endpoints so they can
# never drift apart on what "pending" means.
_PENDING = (
    ExtractedField.needs_review.is_(True),
    ExtractedField.reviewed_at.is_(None),
)


@router.get("/queue", response_model=list[ReviewQueueItem])
async def list_review_queue(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[ReviewQueueItem]:
    stmt = (
        select(ExtractedField, Extraction, Document)
        .join(Extraction, ExtractedField.extraction_id == Extraction.id)
        .join(Document, Extraction.document_id == Document.id)
        .where(*_PENDING)
        # Oldest extraction first -- the queue is FIFO so nothing starves;
        # field_name breaks ties deterministically within one extraction.
        .order_by(Extraction.created_at.asc(), ExtractedField.field_name.asc())
        .limit(limit)
        .offset(offset)
    )
    rows = (await session.execute(stmt)).all()
    return [
        ReviewQueueItem(
            field_id=field.id,
            field_name=field.field_name,
            value=field.value,
            confidence=field.confidence,
            document_id=document.id,
            filename=document.filename,
            extraction_id=extraction.id,
            model=extraction.model,
            extracted_at=extraction.created_at,
        )
        for field, extraction, document in rows
    ]


@router.get("/queue/count", response_model=ReviewQueueCount)
async def count_review_queue(
    session: AsyncSession = Depends(get_session),
) -> ReviewQueueCount:
    stmt = select(func.count()).select_from(ExtractedField).where(*_PENDING)
    pending = (await session.execute(stmt)).scalar_one()
    return ReviewQueueCount(pending=pending)


@router.post("/fields/{field_id}/resolve", response_model=ExtractedFieldOut)
async def resolve_field(
    field_id: uuid.UUID,
    payload: ReviewResolveRequest,
    session: AsyncSession = Depends(get_session),
) -> ExtractedField:
    field = await session.get(ExtractedField, field_id)
    if field is None:
        raise HTTPException(status_code=404, detail="Field not found")
    if field.reviewed_at is not None:
        # A second resolution would silently overwrite the first
        # reviewer's decision; surface the conflict instead. (Two
        # reviewers racing the same field both read reviewed_at as NULL
        # only if neither has committed yet -- last write wins then,
        # acceptable for the current single-reviewer deployments; a
        # multi-reviewer setup should move this check into the UPDATE's
        # WHERE clause as a compare-and-set.)
        raise HTTPException(status_code=409, detail="Field already reviewed")

    if payload.action == "correct" and "corrected_value" not in payload.model_fields_set:
        # Distinguish "correct this to null" (explicit corrected_value:
        # null in the body -- legal, asserts the field is absent from the
        # document) from forgetting to send a corrected_value at all.
        raise HTTPException(
            status_code=400,
            detail="action 'correct' requires a corrected_value (null is allowed)",
        )

    field.reviewed_at = datetime.now(UTC)
    if payload.action == "approve":
        field.review_action = "approved"
    else:
        field.review_action = "corrected"
        field.corrected_value = payload.corrected_value

    await session.commit()
    await session.refresh(field)
    return field
