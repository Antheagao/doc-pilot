from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://docpilot:docpilot@localhost:5434/docpilot"
    anthropic_api_key: str | None = None
    upload_dir: str = "uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
