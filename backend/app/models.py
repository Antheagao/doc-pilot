import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base

# Width of document_chunks.embedding. Fixed at the column (not per row)
# because an HNSW index needs a fixed dimension; every embedder in
# app.retrieval.embeddings must produce vectors of exactly this length,
# and changing embedding models to one of a different width is a
# migration, not a config flip. 384 is BAAI/bge-small-en-v1.5's width.
EMBEDDING_DIM = 384

# Job.kind values. 'extract' is the original (and default) job: run the
# VLM extraction. 'index' makes a document searchable: transcribe its
# pages, chunk, embed, write document_pages/document_chunks. They're
# separate jobs so a transcription failure retries on its own, never
# re-running (and re-billing) an extraction that already succeeded.
JOB_KIND_EXTRACT = "extract"
JOB_KIND_INDEX = "index"
# 'judge' grades a stored /ask answer (an AskRun, not a document) with the
# reference-free groundedness grader -- online evaluation, off the request
# path. See app/evals/online.py.
JOB_KIND_JUDGE = "judge"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str] = mapped_column(String, nullable=False)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="uploaded")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_state_created_at", "state", "created_at"),
        # Every job is about something: a document (extract, index) or a
        # stored /ask answer (judge).
        CheckConstraint(
            "document_id IS NOT NULL OR ask_run_id IS NOT NULL", name="ck_jobs_has_target"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True
    )
    ask_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ask_runs.id", ondelete="CASCADE"), nullable=True
    )
    state: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    run_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    kind: Mapped[str] = mapped_column(
        String, nullable=False, default=JOB_KIND_EXTRACT, server_default=JOB_KIND_EXTRACT
    )
    # W3C trace context of whatever enqueued this job (the upload request,
    # or the extract job that queued an index job). The worker starts the
    # job's span as a child of it, so a document's whole lifecycle is one
    # trace across the queue. NULL when tracing is off. See app/telemetry.py.
    traceparent: Mapped[str | None] = mapped_column(String, nullable=True)


class Extraction(Base):
    __tablename__ = "extractions"
    # The daily budget (app/budget.py) sums today's spend on every billed
    # request; these time indexes keep that from scanning all history.
    __table_args__ = (Index("ix_extractions_created_at", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False
    )
    prompt_version: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    raw_response: Mapped[dict] = mapped_column(JSONB, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ExtractedField(Base):
    __tablename__ = "extracted_fields"
    __table_args__ = (
        Index("ix_extracted_fields_extraction_id", "extraction_id"),
        # The review-queue scan: fields with needs_review=true and
        # reviewed_at IS NULL are the pending queue (see routers/review.py).
        Index("ix_extracted_fields_review_queue", "needs_review", "reviewed_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    extraction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("extractions.id"), nullable=False
    )
    field_name: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False)
    needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Human review resolution. All three stay NULL until a reviewer acts;
    # review_action is 'approved' (extracted value confirmed) or
    # 'corrected' (corrected_value holds the human-supplied replacement).
    # The original `value` is never overwritten -- corrections live beside
    # it so the audit trail keeps what the model actually said, and so a
    # correction can later become a labeled eval case.
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    review_action: Mapped[str | None] = mapped_column(String, nullable=True)
    corrected_value: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class DocumentPage(Base):
    """One page of a document's text: the source the retrieval index is
    chunked from, and what a search citation's char offsets point into.

    source is 'transcription' (a VLM call, app/transcription.py -- its
    model/prompt_version/tokens/cost/latency are recorded here the same way
    an Extraction row records its own) or 'gold' (known-correct text
    supplied directly, e.g. by the retrieval eval, with no model call and
    zero cost).
    """

    __tablename__ = "document_pages"
    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_document_pages_document_page"),
        Index("ix_document_pages_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False
    )
    # 1-based, matching how a person (and a PDF viewer) numbers pages.
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str | None] = mapped_column(String, nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DocumentChunk(Base):
    """A retrievable span of one page (app/retrieval/chunking.py).

    `text` is always an exact substring of its page's text --
    page.text[char_start:char_end] -- so a citation can point at (and a UI
    can highlight) precisely the span that was retrieved. `context` is the
    optional header prepended for search only (document title + page
    number, see Settings.chunk_context_headers): it's part of what gets
    embedded and full-text indexed, but never part of the cited span.

    Two indexes serve the two halves of hybrid search
    (app/retrieval/search.py): HNSW over `embedding` with cosine ops for
    the dense side, GIN over the generated `tsv` column for the lexical
    side.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("ix_document_chunks_document_id", "document_id"),
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_document_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False
    )
    page_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_pages.id"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    # Position within its page, 0-based.
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Search-only canonical forms the text parser would miss -- amounts in
    # comma-decimal or thousands-separated formats, currency words (see
    # app/retrieval/normalize.py). Full-text indexed, never embedded or
    # cited.
    search_aliases: Mapped[str | None] = mapped_column(Text, nullable=True)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    # Which embedder produced `embedding` -- vectors from different models
    # live in different spaces and must never be compared against each
    # other, so a model change means re-indexing (scripts/reindex.py).
    embedding_model: Mapped[str] = mapped_column(String, nullable=False)
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(context, '') || ' ' || text || ' ' "
            "|| coalesce(search_aliases, ''))",
            persisted=True,
        ),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class AskRun(Base):
    """One POST /ask: the question, the agent's answer with its resolved
    citations and tool trail, and the evidence it saw -- enough to audit
    an answer, grade it later, or turn it into an eval case, without
    re-running (and re-billing) the agent.

    `evidence` is the run's tool results rendered as plain text
    (app.evals.judge.render_evidence_from_messages): what the groundedness
    grader reads. Two independent verdicts can attach to a run afterwards:
    a person's `feedback` ('up' / 'down', POST /ask/runs/{id}/feedback),
    and, for a sampled share of answers (ASK_JUDGE_SAMPLE_RATE), the
    background grader's `judge_*` columns (a 'judge' job).
    """

    __tablename__ = "ask_runs"
    __table_args__ = (
        CheckConstraint("feedback IN ('up', 'down')", name="ck_ask_runs_feedback"),
        Index("ix_ask_runs_created_at", "created_at"),
        Index("ix_ask_runs_judged_at", "judged_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list] = mapped_column(JSONB, nullable=False)
    tool_calls: Mapped[list] = mapped_column(JSONB, nullable=False)
    evidence: Mapped[str] = mapped_column(Text, nullable=False)
    steps: Mapped[int] = mapped_column(Integer, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_read_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    refusal_category: Mapped[str | None] = mapped_column(String, nullable=True)
    unresolved_citations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trace_id: Mapped[str | None] = mapped_column(String, nullable=True)
    traceparent: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    feedback: Mapped[str | None] = mapped_column(String, nullable=True)
    feedback_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # True when a judge job was enqueued for this run; the judge_* columns
    # stay NULL until it finishes (judged_at set, with a verdict or an
    # error -- a refusal or unreadable verdict is recorded, not retried).
    judge_sampled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    judge_grounded: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    judge_answers_question: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    judge_claims: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    judge_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(String, nullable=True)
    judge_prompt_version: Mapped[str | None] = mapped_column(String, nullable=True)
    judge_cost_usd: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    judge_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    judged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
