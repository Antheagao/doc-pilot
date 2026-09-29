from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The backend project root (parent of the app/ package). Used to anchor a
# relative upload_dir so saved paths are stable regardless of the process's
# current working directory.
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # env_file is anchored to BACKEND_DIR (not CWD) for the same reason
    # upload_dir resolution is: it must find backend/.env regardless of
    # where the process was launched from.
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://docpilot:docpilot@localhost:5434/docpilot"
    anthropic_api_key: str | None = None
    upload_dir: str = "uploads"
    worker_poll_interval: float = 1.0
    # Model used for document extraction (app/extraction.py). Defaults to
    # the current Sonnet id -- see PRICING_PER_MTOK in that module for the
    # price table this must stay in sync with if overridden.
    extraction_model: str = "claude-sonnet-5"
    # Extracted fields with confidence below this are flagged needs_review.
    review_threshold: float = 0.8
    # When a first extraction response violates RECORD_EXTRACTION_TOOL's
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
    # Where fastembed caches downloaded model files. None uses fastembed's
    # own default (a temp directory); docker-compose.yml points it at a
    # shared volume so the model downloads once, not per container start.
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
