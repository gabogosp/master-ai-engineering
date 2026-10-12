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

    # --- Cache semántica (requiere redis/redis-stack, no redis:7-alpine) ---
    embedding_model: str = "text-embedding-3-small"
    semantic_cache_threshold: float = 0.85
    semantic_cache_ttl_seconds: int = 86400
    # Si es True, la cache hace el lookup y loguea el score pero nunca sirve
    # el hit -- para calibrar el threshold contra tráfico real antes de
    # activarla.
    semantic_cache_log_only: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()