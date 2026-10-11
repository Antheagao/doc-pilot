"""Human review queue endpoints.

Fields are flagged needs_review=True with a review_reason: the model's
confidence fell below Settings.review_threshold, or a failing arithmetic
check names them (app/review_routing.py). This router exposes that set as
a work queue:

- GET  /review/queue        -- pending fields (needs_review, not yet
                               reviewed), oldest extraction first, joined
                               with document context
- GET  /review/queue/count  -- pending total, cheap enough to poll for a
                               nav badge
- POST /review/fields/{id}/resolve -- approve the extracted value or
                               supply a correction (validated: a bad amount
                               is a 422 with a sentence), then recompute the
                               extraction's reconciliation and routing

Resolution is field-level on purpose (see README "Key decisions"): a
document with one shaky field shouldn't cost a human a whole-document
re-check. The original extracted value is never overwritten -- a
correction lands in corrected_value beside it, keeping the audit trail
of what the model actually said and giving future eval tooling a labeled
(document, field, human answer) triple to harvest.
"""

import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import get_session
from app.models import Document, ExtractedField, Extraction, Reconciliation
from app.money import MoneyError
from app.records import (
    api_field_value,
    currency_code,
    currency_fields,
    field_to_out,
    minor_units_for,
    normalize_correction,
    precision_for,
)
from app.review_routing import (
    failing_checks_for,
    lock_extraction,
    refresh_reconciliation,
    suggestion_for,
)
from app.schemas import (
    CheckOut,
    ExtractedFieldOut,
    ReviewQueueCount,
    ReviewQueueItem,
    ReviewResolveRequest,
    SuggestionOut,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# The queue predicate, shared by the list and count endpoints so they can
# never drift apart on what "pending" means.
_PENDING = (
    ExtractedField.needs_review.is_(True),
    ExtractedField.reviewed_at.is_(None),
)


def _reconciliation_context(
    field: ExtractedField, report: dict | None
) -> tuple[list[CheckOut], SuggestionOut | None]:
    """The failing checks and suggestion a queue item shows. A stored report
    this code cannot read costs the item its context, not the queue its
    answer."""
    try:
        checks = [
            CheckOut.model_validate(c)
            for c in failing_checks_for(report or {}, field.field_name)
        ]
        suggestion = suggestion_for(report or {}, field.field_name)
        return checks, None if suggestion is None else SuggestionOut.model_validate(
            suggestion
        )
    except (ValidationError, KeyError, TypeError):
        logger.exception(
            "stored reconciliation for field %s does not validate; serving none",
            field.id,
        )
        return [], None


@router.get("/queue", response_model=list[ReviewQueueItem])
async def list_review_queue(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[ReviewQueueItem]:
    stmt = (
        select(ExtractedField, Extraction, Document, Reconciliation.report)
        .join(Extraction, ExtractedField.extraction_id == Extraction.id)
        .join(Document, Extraction.document_id == Document.id)
        .outerjoin(Reconciliation, Reconciliation.extraction_id == Extraction.id)
        .where(*_PENDING)
        # Oldest extraction first -- the queue is FIFO so nothing starves;
        # field_name breaks ties deterministically within one extraction.
        .order_by(Extraction.created_at.asc(), ExtractedField.field_name.asc())
        .limit(limit)
        .offset(offset)
    )
    rows = (await session.execute(stmt)).all()
    currencies = await currency_fields(
        session, [extraction.id for _, extraction, _, _ in rows]
    )
    items = []
    for field, extraction, document, report in rows:
        checks, suggestion = _reconciliation_context(field, report)
        items.append(
            ReviewQueueItem(
                field_id=field.id,
                field_name=field.field_name,
                value=api_field_value(
                    field.field_name,
                    field.value,
                    minor_units_for(currencies.get(extraction.id)),
                ),
                currency=currency_code(currencies.get(extraction.id)),
                confidence=field.confidence,
                review_reason=field.review_reason,
                checks=checks,
                suggestion=suggestion,
                document_id=document.id,
                filename=document.filename,
                extraction_id=extraction.id,
                model=extraction.model,
                extracted_at=extraction.created_at,
            )
        )
    return items


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
    settings: Settings = Depends(get_settings),
) -> ExtractedFieldOut:
    field = await session.get(ExtractedField, field_id)
    if field is None:
        raise HTTPException(status_code=404, detail="Field not found")
    extraction_id = field.extraction_id
    # Lock the extraction before looking at the field again (see
    # review_routing): two reviewers resolving different fields of one
    # document queue up here instead of deadlocking on each other's rows, and
    # the second sees the first's decision.
    await lock_extraction(session, extraction_id)
    await session.refresh(field)
    if field.reviewed_at is not None:
        # A second resolution would silently overwrite the first
        # reviewer's decision; surface the conflict instead. The extraction
        # lock above makes two reviewers of the same field take turns, so
        # the second reads the first's reviewed_at here.
        raise HTTPException(status_code=409, detail="Field already reviewed")

    if (
        payload.action == "correct"
        and "corrected_value" not in payload.model_fields_set
    ):
        # Distinguish "correct this to null" (explicit corrected_value:
        # null in the body -- legal, asserts the field is absent from the
        # document) from forgetting to send a corrected_value at all.
        raise HTTPException(
            status_code=400,
            detail="action 'correct' requires a corrected_value (null is allowed)",
        )

    corrected_value = None
    if payload.action == "correct":
        currency = (await currency_fields(session, [field.extraction_id])).get(
            field.extraction_id
        )
        try:
            corrected_value = normalize_correction(
                field.field_name, payload.corrected_value, precision_for(currency)
            )
        except MoneyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    field.reviewed_at = datetime.now(UTC)
    if payload.action == "approve":
        field.review_action = "approved"
    else:
        field.review_action = "corrected"
        field.corrected_value = corrected_value

    # The new decision changes what the arithmetic says about the siblings: a
    # corrected culprit clears the other operands, an approved misread flags
    # them. Same transaction, so the queue never shows a half-applied review.
    await session.flush()
    await refresh_reconciliation(session, extraction_id, settings)
    await session.commit()
    await session.refresh(field)
    # Serve the field with the currency as it is now: correcting the
    # currency itself changes the digits its siblings are padded to.
    currency = (await currency_fields(session, [field.extraction_id])).get(
        field.extraction_id
    )
    return field_to_out(field, minor_units_for(currency))
