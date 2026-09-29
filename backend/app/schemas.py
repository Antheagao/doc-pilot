import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class StatsSpend(BaseModel):
    """Every model call the system has paid for, by stage. Transcription
    counts the pages currently indexed (a re-index replaces them)."""

    extraction_usd: float
    transcription_usd: float
    agent_usd: float
    judge_usd: float
    total_usd: float


class StatsAsk(BaseModel):
    """/ask in production: volume, cost, latency, and the two online
    quality signals -- people's feedback and the sampled groundedness
    grader (rates are over readable verdicts; null before any)."""

    runs: int
    answered: int
    mean_cost_usd: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    feedback_up: int
    feedback_down: int
    judge_sampled: int
    judged: int
    judge_grounded_rate: float | None
    judge_answers_rate: float | None


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
    # Extraction + transcription per processed document: what making one
    # document reviewable and searchable costs (mean_cost_per_doc above is
    # extraction alone).
    mean_pipeline_cost_per_doc: float | None
    spend: StatsSpend
    ask: StatsAsk


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


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class CitationOut(BaseModel):
    """One numbered source for the answer. `cited_text` is copied by the
    API from the tool result, never written by the model. For a page
    passage, char_start/char_end locate it in the page's stored text; for
    an extraction record, `fields` names the fields cited."""

    model_config = ConfigDict(from_attributes=True)

    number: int
    document_id: uuid.UUID
    filename: str
    page_number: int | None
    kind: Literal["chunk", "page", "record"]
    cited_text: str
    char_start: int | None
    char_end: int | None
    fields: list[str]


class ToolCallOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    input: dict[str, Any]
    is_error: bool
    result_summary: str


AskStatus = Literal["answered", "refused", "truncated", "step_limit", "budget_exceeded"]


class AskJudgmentOut(BaseModel):
    """The background groundedness grader's verdict on a stored answer
    (app/evals/online.py). grounded/answers_question are None when no
    verdict could be read -- `error` says why."""

    grounded: bool | None
    answers_question: bool | None
    unsupported_claims: list[str]
    explanation: str | None
    model: str | None
    prompt_version: str | None
    cost_usd: float
    error: str | None
    judged_at: datetime


class AskResponse(BaseModel):
    """One /ask run, as answered and as stored: POST /ask returns it, and
    GET /ask/runs/{id} returns it later with any feedback and judgment."""

    id: uuid.UUID
    question: str
    created_at: datetime
    status: AskStatus
    answer: str
    citations: list[CitationOut]
    tool_calls: list[ToolCallOut]
    steps: int
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cost_usd: float
    latency_ms: int
    refusal_category: str | None
    trace_id: str | None
    feedback: Literal["up", "down"] | None
    feedback_note: str | None
    # A judge job was enqueued; `judgment` stays null until it finishes.
    judge_sampled: bool
    judgment: AskJudgmentOut | None


class AskRunSummary(BaseModel):
    """One row of GET /ask/runs."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    question: str
    status: AskStatus
    created_at: datetime
    cost_usd: float
    latency_ms: int
    feedback: Literal["up", "down"] | None
    judge_sampled: bool
    judge_grounded: bool | None


class AskFeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    note: str | None = Field(default=None, max_length=1000)
