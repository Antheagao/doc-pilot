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


class StatsReview(BaseModel):
    pending: int
    approved: int
    corrected: int


class StatsLastEval(BaseModel):
    model: str | None
    prompt_version: str | None
    dataset_version: str | None
    started_at_utc: str
    overall_accuracy: float | None
    caught_by_review: float | None
    mean_cost_per_doc: float
    n_scored: int


class StatsOut(BaseModel):
    documents_total: int
    documents_by_status: dict[str, int]
    documents_processed: int
    extractions_total: int
    total_cost_usd: float
    mean_cost_per_doc: float | None
    total_input_tokens: int
    total_output_tokens: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    review: StatsReview
    last_eval: StatsLastEval | None


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


class SearchHitOut(BaseModel):
    """One retrieved chunk and its citation: which document, which page,
    and where on that page (char offsets into the page's stored text --
    text == page_text[char_start:char_end])."""

    model_config = ConfigDict(from_attributes=True)

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    score: float
    dense_rank: int | None
    lexical_rank: int | None


class SearchResponse(BaseModel):
    query: str
    mode: Literal["dense", "lexical", "hybrid"]
    embedding_model: str
    results: list[SearchHitOut]
