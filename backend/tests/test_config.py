"""Settings that guard something (app/config.py)."""

import pytest
from pydantic import ValidationError

from app.config import Settings


@pytest.mark.parametrize(
    "origins",
    [["*"], ["http://localhost:3000", "*"], ["https://*.example.com"], [" * "]],
)
def test_a_wildcard_cors_origin_is_refused(origins: list[str]) -> None:
    with pytest.raises(ValidationError, match="explicit origins"):
        Settings(cors_origins=origins)


def test_wildcard_cors_origin_from_the_environment_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", '["*"]')

    with pytest.raises(ValidationError, match="explicit origins"):
        Settings()


def test_explicit_cors_origins_are_kept() -> None:
    origins = ["http://localhost:3000", "http://localhost:3100"]

    assert Settings(cors_origins=origins).cors_origins == origins


def test_the_migration_url_is_the_owners_and_unset_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MIGRATION_DATABASE_URL", raising=False)

    assert Settings(_env_file=None).migration_database_url is None

    monkeypatch.setenv("MIGRATION_DATABASE_URL", "postgresql+asyncpg://owner:pw@db:5432/x")
    assert Settings(_env_file=None).migration_database_url == "postgresql+asyncpg://owner:pw@db:5432/x"


def test_the_stdio_api_key_comes_from_docpilot_api_key_and_never_prints(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "dpk_aaaaaaaaaaaa_" + "b" * 43
    monkeypatch.setenv("DOCPILOT_API_KEY", secret)

    settings = Settings()

    assert settings.mcp_stdio_api_key.get_secret_value() == secret
    assert secret not in repr(settings) and secret not in str(settings.mcp_stdio_api_key)
    monkeypatch.delenv("DOCPILOT_API_KEY")
    assert Settings(_env_file=None).mcp_stdio_api_key is None


def test_embedding_threads_is_unset_by_default_and_an_empty_value_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compose passes `${WORKER_EMBEDDING_THREADS:-}`: an unset variable is an
    empty string in the container."""
    monkeypatch.delenv("EMBEDDING_THREADS", raising=False)
    assert Settings(_env_file=None).embedding_threads is None

    monkeypatch.setenv("EMBEDDING_THREADS", "")
    assert Settings(_env_file=None).embedding_threads is None

    monkeypatch.setenv("EMBEDDING_THREADS", "2")
    assert Settings(_env_file=None).embedding_threads == 2


@pytest.mark.parametrize("value", ["0", "-1", "two"])
def test_embedding_threads_below_one_or_not_a_number_is_refused(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("EMBEDDING_THREADS", value)

    with pytest.raises(ValidationError, match="embedding_threads"):
        Settings(_env_file=None)
