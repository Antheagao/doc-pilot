import tempfile
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.extraction_fields import schema_for

# The backend project root (parent of the app/ package). Used to anchor a
# relative upload_dir so saved paths are stable regardless of the process's
# current working directory.
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # env_file is anchored to BACKEND_DIR (not CWD) for the same reason
    # upload_dir resolution is: it must find backend/.env regardless of
    # where the process was launched from.
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    # What the application connects as: a non-owner role that row-level
    # security binds (app/db.py verify_tenant_isolation refuses any other).
    database_url: str = "postgresql+asyncpg://docpilot:docpilot@localhost:5434/docpilot"
    # The database owner, for Alembic (alembic/env.py) and the role
    # provisioner (scripts/provision_db_role.py); the application never uses
    # it. Unset means database_url, which is only right while that is the
    # owner (a bare `alembic upgrade head` on a fresh checkout).
    migration_database_url: str | None = None
    # Browser origins allowed to call the API (CORS) -- the frontend's.
    # A JSON list in the environment: CORS_ORIGINS='["https://..."]'. An
    # origin is never a wildcard (_explicit_cors_origins).
    cors_origins: list[str] = ["http://localhost:3000"]
    anthropic_api_key: str | None = None
    upload_dir: str = "uploads"
    worker_poll_interval: float = 1.0
    # The file the worker touches every few seconds while its event loop is
    # alive (app/worker.py _heartbeat_loop); the compose healthcheck reads
    # its mtime. In the container the temp dir is /tmp.
    worker_heartbeat_file: str = Field(
        default_factory=lambda: str(Path(tempfile.gettempdir()) / "doc-pilot-worker-heartbeat")
    )
    # Reclaiming orphaned jobs (app/worker.py). 0, the default, suits one
    # long-lived worker (docker compose): every 'processing' job is
    # reclaimed once, at startup. A platform that overlaps the old and new
    # worker during a deploy (Cloud Run) sets a lease instead: only jobs
    # that have been processing longer than this many seconds are
    # reclaimed, at startup and then every minute -- so the new worker
    # never steals a job the old one is still finishing.
    worker_reclaim_after_seconds: float = Field(default=0, ge=0)
    # Model used for document extraction (app/extraction.py). Defaults to
    # the current Sonnet id -- see PRICING_PER_MTOK in that module for the
    # price table this must stay in sync with if overridden.
    extraction_model: str = "claude-sonnet-5"
    # Which extraction schema the worker asks the model for
    # (app/extraction_fields.py SCHEMAS): the prompt, repair prompt, and field
    # list move together. Checked at load, so a misspelled version fails at
    # startup rather than on the first job. Stored extractions keep the
    # version they were made with, so changing this never rewrites old rows.
    extraction_prompt_version: str = "extract_v1"
    # Extracted fields with confidence below this are flagged needs_review.
    review_threshold: float = 0.8
    # When a first extraction response violates the extraction tool's
    # schema (app/extraction.py's _schema_violations), attempt exactly one
    # repair reprompt before giving up on the cleaner shape. See
    # extraction.extract_document.
    schema_repair: bool = True
    # Passed straight through to anthropic.AsyncAnthropic (app.extraction
    # ._build_client) -- the SDK's own in-process retry/timeout layer,
    # distinct from and beneath the Postgres job queue's own requeue
    # backoff. See _build_client's docstring for the two-layer story.
    anthropic_max_retries: int = 2
    anthropic_timeout_seconds: float = 120.0
    # Oversized-document chunking (H6, see app/extraction.py's
    # extract_document dispatcher and app/pdf.py). At or below this page
    # count, a PDF is sent whole in a single call exactly as before H6
    # shipped. Above it (and at or below pdf_max_pages), the PDF is
    # split into per-page single-page PDFs and extracted sequentially,
    # then merged.
    pdf_max_pages_per_call: int = 5
    # Hard ceiling: a PDF with more pages than this is refused with
    # NonRetryableExtractionError BEFORE any API call is made, so a
    # misconfigured/oversized upload can't spend API budget it was never
    # going to be able to chunk usefully anyway.
    pdf_max_pages: int = 20

    # --- Retrieval (app/retrieval/, app/transcription.py) -----------------
    # After a successful extraction, enqueue an 'index' job that transcribes
    # each page, chunks it, embeds the chunks, and writes them to pgvector
    # (see app.retrieval.indexing.process_index_job). Off means documents
    # are extracted but never become searchable -- and no transcription
    # spend is incurred.
    index_after_extraction: bool = True
    # Model for per-page transcription, the text source the retrieval
    # index is built from. Plain transcription doesn't need the extraction
    # model's judgment, so this defaults to the cheapest priced model --
    # must have an entry in app.extraction.PRICING_PER_MTOK.
    transcription_model: str = "claude-haiku-4-5"
    # "fastembed" (BAAI/bge-small-en-v1.5 via ONNX, local, no API key) or
    # "hashing" (deterministic feature hashing: offline and instant, but
    # lexical rather than semantic -- what the test suite and CI use).
    # Both produce EMBEDDING_DIM-dimensional vectors (app/models.py).
    embedding_backend: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Where downloaded model files are kept. None means a temp directory;
    # docker-compose.yml points it at a shared volume so the model
    # downloads once, not per container start. The default model is pinned
    # to checksummed files at a fixed commit -- the ones the retrieval eval
    # measures -- see app/retrieval/model_fetch.py.
    embedding_cache_dir: str | None = None
    # A local directory holding an already-downloaded ONNX export of
    # embedding_model, bypassing the download entirely (air-gapped hosts).
    embedding_model_path: str | None = None
    # Chunking (app/retrieval/chunking.py). 0 means one chunk per page.
    # 200/80 is what the retrieval eval measured best on this corpus
    # (README "Retrieval eval"): whole-page chunks blur a 16-item receipt
    # into one vector that matches none of its items well, while recall
    # plateaus between 120 and 200 chars and falls off again by 300.
    # Receipt lines run ~60 chars, so a chunk is ~3 lines. A corpus of
    # prose-heavy documents would want larger chunks -- re-run the eval
    # with --chunk-sizes before changing this.
    chunk_max_chars: int = 200
    chunk_overlap_chars: int = 80
    # Prefix each chunk's embedded/searchable text with its document's
    # title line and page number, so a chunk cut from the middle of a page
    # still carries the context of which document it came from.
    chunk_context_headers: bool = True

    # --- /ask agent (app/agent/) -------------------------------------------
    # Answers questions over the documents with tools: hybrid search, the
    # structured (human-verified) extraction records, and whole pages.
    agent_model: str = "claude-opus-5-5"
    # Opus 5.5 thinks adaptively and can't turn it off; effort is the dial,
    # and its API default is "medium" -- set explicitly so a default change
    # upstream can't silently change cost or quality.
    agent_effort: str = "medium"
    # At most 21,000: the SDK refuses a non-streaming request that could run
    # longer than ten minutes, and a larger value raises a plain ValueError
    # that escapes `except anthropic.AnthropicError` in the agent loop.
    agent_max_tokens: int = Field(16000, le=21_000)
    # Hard stops per question: model calls, and dollars across all of them.
    # Hitting either returns what the agent has so far, marked as such.
    agent_max_steps: int = 8
    agent_max_cost_usd: float = 0.25
    # Server-side refusal fallback (beta server-side-fallback-2026-07-01,
    # fallbacks="default"): a classifier false positive on a receipt
    # question is retried on Anthropic's recommended fallback model instead
    # of failing the request. Claude API only -- turn off elsewhere.
    agent_refusal_fallback: bool = True

    # --- LLM-as-judge (app/evals/judge.py) --------------------------------
    # Grades agent answers for correctness (against label-derived reference
    # facts) and groundedness (against the evidence the agent retrieved).
    # A different model family member than the agent, so it isn't grading
    # its own reasoning; validated against human labels before its numbers
    # are trusted (evals/run_judge_calibration.py).
    judge_model: str = "claude-sonnet-5-5"
    judge_effort: str = "medium"
    judge_max_tokens: int = 8000

    # --- Spend guardrail (app/budget.py) ----------------------------------
    # Uploads and /ask questions are refused (429) once the model spend
    # recorded since midnight UTC -- extraction, transcription, the agent,
    # its grader -- reaches this many dollars. A soft cap: work already
    # admitted finishes. 0 turns it off.
    daily_budget_usd: float = Field(default=5.0, ge=0.0)
    # The same cap for one organization's own spend (app/budget.py), so one
    # tenant cannot use up the installation's budget alone; checked before
    # DAILY_BUDGET_USD. 0, the default, turns it off.
    organization_daily_budget_usd: float = Field(default=0.0, ge=0.0)
    # Per-client requests per minute (app/ratelimit.py), so no one client
    # can spend the day's budget in a minute. 0 turns a limit off. /ask is
    # limited by default (no person asks ten questions a minute); uploads
    # aren't, since dropping in a stack of receipts at once is normal.
    ask_rate_limit_per_minute: int = Field(default=10, ge=0)
    upload_rate_limit_per_minute: int = Field(default=0, ge=0)

    # Online evaluation (app/evals/online.py): the share of answered /ask
    # questions (0.0-1.0) that get a background 'judge' job running the
    # reference-free groundedness grader on JUDGE_MODEL. Off by default:
    # each graded answer is one more billed model call.
    ask_judge_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- MCP server (app/mcp_server/) -------------------------------------
    # The most characters of text one MCP tool result carries. The MCP client
    # is the model, so a result goes straight into its context: 24,000 is a
    # little over 6,000 tokens, more than a ten-passage search or a long page
    # fills (the tools already cap passages at 10 and records at 20), and well
    # under what a client will truncate or reject on its own. A longer result
    # drops its trailing lines and says so (app/mcp_server/server.py).
    mcp_max_result_chars: int = Field(default=24_000, ge=2_000, le=200_000)
    # POSTs per minute per client on /mcp (app/mcp_server/http.py, the "mcp"
    # bucket of app/ratelimit.py); 0 turns it off. Every JSON-RPC message is a
    # POST and counts, so a client's handshake (initialize, the initialized
    # notification, tools/list) costs three before its first tool call. 120 is
    # two a second: an agent working through a question in a burst stays under
    # it, a loop hammering the database does not. The tools never call a model,
    # so this guards the database and embedder, not spend.
    mcp_rate_limit_per_minute: int = Field(default=120, ge=0)

    # --- Sign-in (app/auth/) ----------------------------------------------
    # The browser session cookie's name. The demo stack sets its own: cookies
    # ignore ports, so the dev API (8000) and the demo API (8100) on localhost
    # would otherwise overwrite each other's cookie.
    session_cookie_name: str = "docpilot_session"
    # The cookie's Secure flag. Off by default because the stacks are plain
    # HTTP on loopback, where a browser would drop a Secure cookie; turn it on
    # wherever TLS terminates.
    session_cookie_secure: bool = False
    # How long a sign-in lasts, absolute from the login (a read never extends
    # it, so a request never writes a session row). 12 hours is a working day.
    session_ttl_hours: int = Field(default=12, ge=1)
    # Sign-in attempts per minute per client address, and per account (keyed
    # by a digest of the case-folded email), both checked before any password
    # is hashed (app/ratelimit.py); 0 turns a limit off. The account limit is
    # lower because the address limit alone lets one client try a password
    # against many accounts, and many clients try one account.
    login_rate_limit_per_minute: int = Field(default=10, ge=0)
    login_account_rate_limit_per_minute: int = Field(default=5, ge=0)

    @field_validator("cors_origins")
    @classmethod
    def _explicit_cors_origins(cls, value: list[str]) -> list[str]:
        # Credentialed CORS with a wildcard would let any site read the API as
        # the signed-in user; Starlette also treats only a bare "*" as one, so
        # a "*" anywhere in an entry is refused rather than guessed at.
        if any("*" in origin for origin in value):
            raise ValueError("CORS_ORIGINS must list explicit origins, not a wildcard")
        return value

    @field_validator("extraction_prompt_version")
    @classmethod
    def _known_extraction_prompt_version(cls, value: str) -> str:
        schema_for(value)
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_upload_dir(settings: Settings) -> Path:
    """Resolve settings.upload_dir to an absolute path.

    Relative values (the default, "uploads") are anchored to the backend
    project directory (BACKEND_DIR), so uploads land in the same place
    whether uvicorn is started from backend/ or elsewhere. Absolute values
    (e.g. an overridden tmp_path in tests) are used as-is.
    """
    upload_dir = Path(settings.upload_dir)
    return upload_dir if upload_dir.is_absolute() else BACKEND_DIR / upload_dir
