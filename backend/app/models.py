import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import CITEXT, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


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

# ExtractedField.review_reason values: why a field is flagged. 'confidence' is
# the model's own doubt (confidence under Settings.review_threshold);
# 'reconciliation:<check_id>' is a failed arithmetic check that names the field
# (app/reconcile.py, routed by app/review_routing.py); 'quote:<path>' is the
# number in the field's evidence quote disagreeing with its value (the check id
# is already that text). NULL exactly when the field is not flagged
# (ck_extracted_fields_review_reason_matches).
REVIEW_REASON_CONFIDENCE = "confidence"
REVIEW_REASON_RECONCILIATION_PREFIX = "reconciliation:"
REVIEW_REASON_QUOTE_PREFIX = "quote:"

# The organization every row written before multi-tenancy belongs to. A literal,
# copied (not imported) into the migration that inserts the row;
# tests/test_auth_core.py pins the two equal.
DEFAULT_ORGANIZATION_ID = uuid.UUID("270ad65a-150a-498e-bbb1-584f6d21d86e")

# Membership.role values: an 'owner' also manages the organization's API keys,
# a 'member' only uses it (ck_memberships_role).
MEMBERSHIP_ROLE_OWNER = "owner"
MEMBERSHIP_ROLE_MEMBER = "member"
MEMBERSHIP_ROLES = (MEMBERSHIP_ROLE_OWNER, MEMBERSHIP_ROLE_MEMBER)

# The global identity tables (app/auth/): no organization owns a row, and a
# login or a key lookup reads them before any organization is known. Every
# other table is content, and the demo reset (app/demo_seed.py) deletes only
# those.
IDENTITY_TABLES = frozenset({"organizations", "users", "memberships", "auth_sessions", "api_keys"})


def reconciliation_reason(check_id: str) -> str:
    """The review_reason for a field flagged by the failing check `check_id`."""
    return f"{REVIEW_REASON_RECONCILIATION_PREFIX}{check_id}"


class TenantMixin:
    """The organization a content row belongs to. Code never sets it: the
    database fills it from the transaction's tenant (the column default reads
    the `docpilot.organization_id` setting that app/tenancy.py applies at the
    start of every transaction), and with no tenant the default is NULL, which
    NOT NULL refuses. app/tenancy.py also filters every ORM select, update, and
    delete on this class to the session's organization.

    A model that a worker claims across organizations (the job queue) sets
    `__queue_scoped__`; see app.tenancy.queue_session.
    """

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="RESTRICT"),
        nullable=False,
        server_default=text("docpilot_current_organization_id()"),
    )


class Document(TenantMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        # The target of the composite foreign keys that tie a child row to its
        # document's organization.
        UniqueConstraint("id", "organization_id", name="uq_documents_id_organization_id"),
        Index("ix_documents_organization_id_created_at", "organization_id", "created_at"),
    )

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


class Job(TenantMixin, Base):
    __tablename__ = "jobs"
    # The worker claims the oldest pending job of any organization, so it
    # reads this table before it knows a tenant (app.tenancy.queue_session).
    __queue_scoped__ = True
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "organization_id"],
            ["documents.id", "documents.organization_id"],
            name="jobs_document_id_organization_id_fkey",
        ),
        ForeignKeyConstraint(
            ["ask_run_id", "organization_id"],
            ["ask_runs.id", "ask_runs.organization_id"],
            ondelete="CASCADE",
            name="jobs_ask_run_id_organization_id_fkey",
        ),
        Index("ix_jobs_state_created_at", "state", "created_at"),
        Index("ix_jobs_organization_id", "organization_id"),
        Index("ix_jobs_document_id_organization_id", "document_id", "organization_id"),
        Index("ix_jobs_ask_run_id_organization_id", "ask_run_id", "organization_id"),
        # Every job is about something: a document (extract, index) or a
        # stored /ask answer (judge).
        CheckConstraint(
            "document_id IS NOT NULL OR ask_run_id IS NOT NULL", name="ck_jobs_has_target"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Both keys are composite (see __table_args__) and MATCH SIMPLE: a job
    # that has no document, or no ask run, is not checked against one.
    document_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    ask_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
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


class Extraction(TenantMixin, Base):
    __tablename__ = "extractions"
    # The daily budget (app/budget.py) sums today's spend on every billed
    # request; these time indexes keep that from scanning all history.
    __table_args__ = (
        UniqueConstraint("id", "organization_id", name="uq_extractions_id_organization_id"),
        ForeignKeyConstraint(
            ["document_id", "organization_id"],
            ["documents.id", "documents.organization_id"],
            name="extractions_document_id_organization_id_fkey",
        ),
        Index("ix_extractions_created_at", "created_at"),
        Index("ix_extractions_organization_id_created_at", "organization_id", "created_at"),
        Index("ix_extractions_document_id_organization_id", "document_id", "organization_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    raw_response: Mapped[dict] = mapped_column(JSONB, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    # input_tokens is the uncached remainder of the call; the prompt-cache
    # tokens are counted apart, and cost_usd prices all three.
    cache_creation_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    cache_read_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    cost_usd: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ExtractedField(TenantMixin, Base):
    __tablename__ = "extracted_fields"
    __table_args__ = (
        ForeignKeyConstraint(
            ["extraction_id", "organization_id"],
            ["extractions.id", "extractions.organization_id"],
            name="extracted_fields_extraction_id_organization_id_fkey",
        ),
        Index("ix_extracted_fields_extraction_id", "extraction_id"),
        Index(
            "ix_extracted_fields_organization_id_needs_review_reviewed_at",
            "organization_id",
            "needs_review",
            "reviewed_at",
        ),
        # The review-queue scan: fields with needs_review=true and
        # reviewed_at IS NULL are the pending queue (see routers/review.py).
        Index("ix_extracted_fields_review_queue", "needs_review", "reviewed_at"),
        CheckConstraint(
            "review_reason ~ '^(confidence|reconciliation:.+|quote:.+)$'",
            name="ck_extracted_fields_review_reason",
        ),
        # A reviewed field keeps needs_review (and its reason) as they were
        # when a person resolved it; the queue filters on reviewed_at too.
        CheckConstraint(
            "needs_review = (review_reason IS NOT NULL)",
            name="ck_extracted_fields_review_reason_matches",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    extraction_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    field_name: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False)
    needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Why needs_review is set (REVIEW_REASON_*); NULL when it is not. Kept in
    # step with the reconciliation by app/review_routing.py until a person
    # reviews the field, after which both are frozen.
    review_reason: Mapped[str | None] = mapped_column(String, nullable=True)
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


class Reconciliation(TenantMixin, Base):
    """The arithmetic check of one extraction (app/reconcile.py): do the
    subtotal, tax, total, and line items agree? One row per extraction,
    recomputed on write by app/review_routing.refresh_reconciliation (the
    worker, the seed, every review resolve, scripts/reconcile_all.py).

    `report` is reconcile.report_to_json(...) verbatim, the audit record of
    what each check used and said; `status` and `rules_version` repeat its
    two keys so a query can filter without opening the JSON. `status` is
    'pass', 'fail', or 'skipped' (nothing could be checked). There is no
    index on status: three values over one row per extraction, and a report
    query starts from the extraction or document and joins on the unique
    extraction_id.
    """

    __tablename__ = "reconciliations"
    __table_args__ = (
        UniqueConstraint("extraction_id", name="uq_reconciliations_extraction_id"),
        ForeignKeyConstraint(
            ["extraction_id", "organization_id"],
            ["extractions.id", "extractions.organization_id"],
            ondelete="CASCADE",
            name="reconciliations_extraction_id_organization_id_fkey",
        ),
        Index("ix_reconciliations_organization_id", "organization_id"),
        CheckConstraint("status IN ('pass', 'fail', 'skipped')", name="ck_reconciliations_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    extraction_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    rules_version: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    report: Mapped[dict] = mapped_column(JSONB, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DocumentPage(TenantMixin, Base):
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
        UniqueConstraint("id", "organization_id", name="uq_document_pages_id_organization_id"),
        ForeignKeyConstraint(
            ["document_id", "organization_id"],
            ["documents.id", "documents.organization_id"],
            name="document_pages_document_id_organization_id_fkey",
        ),
        Index("ix_document_pages_created_at", "created_at"),
        Index("ix_document_pages_organization_id_created_at", "organization_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
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


class DocumentChunk(TenantMixin, Base):
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
        ForeignKeyConstraint(
            ["document_id", "organization_id"],
            ["documents.id", "documents.organization_id"],
            name="document_chunks_document_id_organization_id_fkey",
        ),
        ForeignKeyConstraint(
            ["page_id", "organization_id"],
            ["document_pages.id", "document_pages.organization_id"],
            name="document_chunks_page_id_organization_id_fkey",
        ),
        Index("ix_document_chunks_document_id", "document_id"),
        # The tenant's rows by themselves: the path lexical search takes once
        # the GIN index can no longer be used under row-level security.
        Index("ix_document_chunks_organization_id", "organization_id"),
        Index("ix_document_chunks_page_id_organization_id", "page_id", "organization_id"),
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
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    page_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
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


class AskRun(TenantMixin, Base):
    """One POST /ask: the question, the agent's answer with its resolved
    citations and tool trail, and the evidence it saw -- enough to audit
    an answer, grade it later, or turn it into an eval case, without
    re-running (and re-billing) the agent.

    A turn of a per-document chat (POST /documents/{id}/chat) is a run
    too, with `document_id` (the document the tools were scoped to) and
    `conversation_id` (shared by the conversation's turns) set; both are
    NULL for a POST /ask question. Storing turns as runs means a chat is
    billed, budgeted, rate-limited, rated and graded exactly like /ask.

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
        UniqueConstraint("id", "organization_id", name="uq_ask_runs_id_organization_id"),
        ForeignKeyConstraint(
            ["document_id", "organization_id"],
            ["documents.id", "documents.organization_id"],
            name="ask_runs_document_id_organization_id_fkey",
        ),
        Index("ix_ask_runs_created_at", "created_at"),
        Index("ix_ask_runs_organization_id_created_at", "organization_id", "created_at"),
        Index("ix_ask_runs_judged_at", "judged_at"),
        Index("ix_ask_runs_conversation", "conversation_id", "created_at"),
        Index("ix_ask_runs_document_id", "document_id"),
        CheckConstraint(
            "(document_id IS NULL) = (conversation_id IS NULL)", name="ck_ask_runs_chat_scope"
        ),
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
    document_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)

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


class Organization(Base):
    """A tenant: the unit that owns documents, keys, and spend. Created by
    scripts/accounts.py; there is no self-signup."""

    __tablename__ = "organizations"
    __table_args__ = (
        CheckConstraint("char_length(name) BETWEEN 1 AND 100", name="ck_organizations_name"),
        CheckConstraint("slug ~ '^[a-z0-9][a-z0-9-]{1,62}$'", name="ck_organizations_slug"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    # The stable handle the CLI and --organization flags take.
    slug: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class User(Base):
    """A person who can sign in. `email` is citext, so it is unique and
    compared without regard to case. A disabled user (`disabled_at` set) is
    refused at login and keeps their rows."""

    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("char_length(email) BETWEEN 1 AND 254", name="ck_users_email"),
        # Exactly one "@" with something on each side, and no whitespace or
        # control characters (app/auth/accounts.py canonical_email is stricter
        # and also normalizes, so a stored email is always its canonical form).
        CheckConstraint(
            "email::text ~ '^[^@[:space:][:cntrl:]]+@[^@[:space:][:cntrl:]]+$'",
            name="ck_users_email_shape",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(CITEXT, nullable=False, unique=True)
    # The encoded argon2id hash (app/auth/passwords.py), never the password.
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    disabled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Membership(Base):
    """A user's role in an organization. Removing it ends the user's
    sessions in that organization (the composite foreign key on
    auth_sessions cascades)."""

    __tablename__ = "memberships"
    __table_args__ = (
        CheckConstraint("role IN ('owner', 'member')", name="ck_memberships_role"),
        Index("ix_memberships_user_id", "user_id"),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class AuthSession(Base):
    """A browser sign-in, pinned to one of the user's organizations. Only the
    SHA-256 digest of the cookie's token is stored (app/auth/tokens.py), so a
    database read never yields a usable cookie. The expiry is absolute: a read
    never extends it."""

    __tablename__ = "auth_sessions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "user_id"],
            ["memberships.organization_id", "memberships.user_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint("octet_length(token_digest) = 32", name="ck_auth_sessions_token_digest"),
        Index("ix_auth_sessions_organization_id_user_id", "organization_id", "user_id"),
        Index("ix_auth_sessions_user_id", "user_id"),
        # The sweep of expired sessions filters on it.
        Index("ix_auth_sessions_expires_at", "expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    token_digest: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, unique=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ApiKey(Base):
    """An organization credential for programmatic access (MCP, scripts).
    `prefix` is stored in the clear to find the row and to show in a list;
    `token_digest` is the SHA-256 of the whole `dpk_<prefix>_<secret>` token,
    which is shown once at creation and never stored. A revoked key keeps its
    row for audit."""

    __tablename__ = "api_keys"
    __table_args__ = (
        CheckConstraint("char_length(name) BETWEEN 1 AND 100", name="ck_api_keys_name"),
        CheckConstraint("prefix ~ '^[a-z2-7]{12}$'", name="ck_api_keys_prefix"),
        CheckConstraint("octet_length(token_digest) = 32", name="ck_api_keys_token_digest"),
        Index("ix_api_keys_organization_id", "organization_id"),
        Index("ix_api_keys_created_by_user_id", "created_by_user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    # NULL when an operator made the key through scripts/accounts.py without
    # naming a user, or after that user was removed (ON DELETE SET NULL).
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    prefix: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    token_digest: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # NULL until the key is first used.
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # NULL while the key is active.
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
