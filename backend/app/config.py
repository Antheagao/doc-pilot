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
