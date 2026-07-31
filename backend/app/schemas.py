import uuid
from datetime import datetime
from typing import Any

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

    field_name: str
    value: Any
    confidence: float
    needs_review: bool


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
