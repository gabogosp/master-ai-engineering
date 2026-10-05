from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    app_env: str = "development"
    log_level: str = "DEBUG"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86400
    backend_url: str = "http://localhost:8000"
    max_conversation_turns: int = 6

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()