"""Runtime settings, read from environment variables prefixed with AURA_ (or .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from aura_ml import config as ml_config

DEFAULT_SECRET = "dev-only-insecure-secret-change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AURA_", env_file=".env", extra="ignore")

    environment: Literal["development", "production", "test"] = "development"
    database_url: str = "sqlite+aiosqlite:///./aura.db"
    auto_create_tables: bool = True  # production uses `alembic upgrade head` instead

    jwt_secret: str = DEFAULT_SECRET
    jwt_expire_minutes: int = 60 * 24 * 7
    cookie_name: str = "aura_session"
    cookie_secure: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    login_attempts_per_5min: int = 10
    registrations_per_hour: int = 10

    # ML
    ml_mode: Literal["lite", "full"] = (
        "lite"  # lite = no torch: rules, lexicon, sklearn models, mock/remote LLM
    )
    llm_provider: Literal["local", "openai", "mock"] = "mock"
    llm_base: str = ml_config.HF_LLM_BASE
    llm_adapter: Path = ml_config.LLM_ADAPTER_DIR
    openai_base_url: str = "http://localhost:11434/v1"
    openai_model: str = "aura"
    openai_api_key: str | None = None
    artifacts_dir: Path = ml_config.ARTIFACTS_DIR
    device: str | None = None

    # Conversation behaviour
    max_message_chars: int = 4000
    max_audio_seconds: int = 60
    history_messages: int = 8
    summarize_every: int = 4  # user turns between memory-summary refreshes
    max_new_tokens: int = 160
    temperature: float = 0.7

    # Static frontend (single-container deployments)
    frontend_dist: Path | None = None

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
