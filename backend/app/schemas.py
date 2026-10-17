import uuid
from datetime import date, datetime
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
    """One extracted field. Every number read from the document (subtotal,
    tax, total, and each line item's quantity, unit_price, and total) is a
    decimal string such as "46.20", padded to the document currency's minor
    unit with extra digits kept (see app/records.py); confidence stays a
    number. A stored value that is not a number is passed through as-is."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    field_name: str
    value: Any
    confidence: float
    needs_review: bool
    # Why the field is flagged: "confidence", "reconciliation:<check id>", or
    # "quote:<path>"; null exactly when needs_review is false. A reviewed field
    # keeps both.
    review_reason: str | None = None
    reviewed_at: datetime | None = None
    review_action: Literal["approved", "corrected"] | None = None
    corrected_value: Any = None


# The closed sets of rules_v1 and rules_v2 (operand source: extracted, approved, corrected,
# computed, assumed, included; check kind: equation, bound, presence, format;
# status: pass, fail, skipped) are plain strings here, because the stored
# report is read back by whatever rules version wrote it: a later version
# that adds a value must not turn every read of an older or newer row into a
# validation error. The frontend types keep the narrow unions.


class OperandOut(BaseModel):
    """A number a check used (`reconcile.report_to_json`). `value` is a
    decimal string, or null when the field is missing or unusable."""

    path: str
    value: str | None
    source: str
    verified: bool


class TermOut(BaseModel):
    coef: str | None
    operand: OperandOut


class CheckOut(BaseModel):
    """One rule's verdict. An equation reads `sum(coef * operand) = rhs`:
    `expected` is the left side, `actual` the right, `delta = actual -
    expected`, and it passes when `abs(delta) <= tolerance`. Amounts are
    decimal strings padded to the currency's minor unit; the `tax_rate`
    check's `actual` is the rate to four places."""

    id: str
    kind: str
    formula: str
    terms: list[TermOut]
    rhs: OperandOut | None
    expected: str | None
    actual: str | None
    delta: str | None
    tolerance: str | None
    status: str
    detail: str | None
    fields: list[str]
    all_operands_verified: bool


class DerivedOut(BaseModel):
    path: str
    value: str | None
    formula: str


class SuggestionOut(BaseModel):
    """The one change that fixes every failing check without breaking a
    passing one: set `path` (a field name, or `line_items[i].total`) to
    `value`."""

    path: str
    value: str
    current: str | None
    check_ids: list[str]
    detail: str


class ReconciliationOut(BaseModel):
    """Whether a document's amounts add up (app/reconcile.py), as stored in
    `reconciliations.report`. Amounts stay strings."""

    rules_version: str
    status: str
    currency: str | None
    minor_units: int
    assumptions: list[str]
    derived: list[DerivedOut]
    checks: list[CheckOut]
    suggestion: SuggestionOut | None
    computed_at: datetime


class ExtractionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    prompt_version: str
    model: str
    cost_usd: float
    latency_ms: int
    # input_tokens is the uncached remainder of the prompt; the prompt-cache
    # tokens are counted apart, so the prompt's size is the three summed.
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    created_at: datetime
    fields: list[ExtractedFieldOut]
    # Null for an extraction nobody has reconciled yet (scripts/reconcile_all.py
    # fills those in).
    reconciliation: ReconciliationOut | None = None


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
    # The document's currency as stated (a correction wins), trimmed and
    # upper-cased; null when it states none. A code that is not ISO 4217
    # passes through as stated. `value` is already padded to this currency's
    # minor unit (2 when the code is missing or unlisted).
    currency: str | None
    confidence: float
    # "confidence" or "reconciliation:<check id>" (see ExtractedFieldOut).
    review_reason: str
    # The failing checks of the extraction's reconciliation that involve this
    # field, in report order; empty for a confidence flag nothing contradicts.
    checks: list[CheckOut]
    # The reconciliation's suggested fix, when it targets this field.
    suggestion: SuggestionOut | None
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


class StatsBudget(BaseModel):
    """The daily spend cap (app/budget.py): null when it's off."""

    daily_budget_usd: float | None
    spent_today_usd: float
    resets_in_seconds: int


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
    # Uncached input only; the prompt-cache tokens are the next two fields.
    total_input_tokens: int
    total_output_tokens: int
    total_cache_creation_input_tokens: int = 0
    total_cache_read_input_tokens: int = 0
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    review: StatsReview
    last_eval: StatsLastEval | None
    # Extraction + transcription per processed document: what making one
    # document reviewable and searchable costs (mean_cost_per_doc above is
    # extraction alone).
    mean_pipeline_cost_per_doc: float | None
    spend: StatsSpend
    budget: StatsBudget
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

    Amounts in a correction may be a decimal string or a number; the router
    checks them against the document's currency (422 with a sentence when
    they do not fit) and stores the canonical string.
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


class ChatRequest(BaseModel):
    """One turn of a per-document chat. Omit conversation_id to start a
    conversation; the answer carries the id to send with follow-ups."""

    question: str = Field(min_length=1, max_length=1000)
    conversation_id: uuid.UUID | None = None


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
    # Set for a turn of a per-document chat; null for a POST /ask question.
    document_id: uuid.UUID | None
    conversation_id: uuid.UUID | None
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
    document_id: uuid.UUID | None
    cost_usd: float
    latency_ms: int
    feedback: Literal["up", "down"] | None
    judge_sampled: bool
    judge_grounded: bool | None


class ConversationSummary(BaseModel):
    """One row of GET /documents/{id}/chat/conversations."""

    conversation_id: uuid.UUID
    first_question: str
    turns: int
    started_at: datetime
    last_turn_at: datetime


class AskFeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    note: str | None = Field(default=None, max_length=1000)


# ---- Monitoring dashboard (GET /monitoring/*) -------------------------------


class EvalRunOut(BaseModel):
    """One committed eval run, normalized across suites (app/evals/history.py)."""

    model_config = ConfigDict(from_attributes=True)

    suite: Literal["extraction", "agent", "retrieval"]
    started_at: datetime
    series: str
    detail: str
    n: int
    accuracy: float | None
    accuracy_metric: str
    mean_cost_usd: float | None
    total_cost_usd: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    extra: dict[str, float | None]


class EvalHistoryOut(BaseModel):
    extraction: list[EvalRunOut]
    agent: list[EvalRunOut]
    retrieval: list[EvalRunOut]


class DailySpend(BaseModel):
    """Model spend billed that day (UTC), by what it paid for."""

    documents_usd: float  # extraction + page transcription
    answers_usd: float  # the /ask agent and document chat
    grading_usd: float  # the online groundedness grader
    total_usd: float


class DailyAnswers(BaseModel):
    """/ask questions and chat turns asked that day."""

    runs: int
    answered: int
    judged: int
    # Of the answers the grader read (judged), the share it found grounded.
    grounded_rate: float | None
    feedback_up: int
    feedback_down: int
    mean_cost_usd: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None


class DailyExtractions(BaseModel):
    documents: int
    mean_cost_usd: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None


class DailyPoint(BaseModel):
    date: date
    spend: DailySpend
    answers: DailyAnswers
    extractions: DailyExtractions
