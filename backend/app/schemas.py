import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class DocumentCreateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    status: str
    created_at: datetime


class DocumentListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    mime_type: str
    status: str
    created_at: datetime


class ExtractedFieldOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    field_name: str
    value: Any
    confidence: float
    needs_review: bool
    reviewed_at: datetime | None = None
    review_action: Literal["approved", "corrected"] | None = None
    corrected_value: Any = None


class ExtractionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    prompt_version: str
    model: str
    cost_usd: float
    latency_ms: int
    input_tokens: int
    output_tokens: int
    created_at: datetime
    fields: list[ExtractedFieldOut]


class DocumentDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    mime_type: str
    status: str
    created_at: datetime
    extraction: ExtractionOut | None


class ReviewQueueItem(BaseModel):
    """One pending review item: a low-confidence field plus just enough
    document/extraction context to review it without a second fetch."""

    field_id: uuid.UUID
    field_name: str
    value: Any
    confidence: float
    document_id: uuid.UUID
    filename: str
    extraction_id: uuid.UUID
    model: str
    extracted_at: datetime


class ReviewQueueCount(BaseModel):
    pending: int


class ReviewResolveRequest(BaseModel):
    """Resolve a pending review field.

    action='approve' confirms the extracted value as-is;
    action='correct' replaces it -- corrected_value then holds the
    human-supplied *semantic* value (the scalar for scalar fields, the
    array for line_items), not a {value, confidence} leaf. An explicit
    null corrected_value is legal (it asserts the field is truly absent
    from the document); the router distinguishes "correct to null" from
    "corrected_value omitted" via model_fields_set.
    """

    action: Literal["approve", "correct"]
    corrected_value: Any = None
