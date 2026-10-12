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

    # --- Compresión de historial (anclas + resumen) y tier dinámico ---
    # Detector de anclas: "heuristic" (regex sobre frases clave) o "llm"
    # (clasificador binario vía Instructor). Heuristic es el default por costo.
    anchor_detection_mode: Literal["heuristic", "llm"] = "heuristic"
    # Modelo barato para llamadas auxiliares (summarizer, anclas en modo
    # "llm", Critic) -- siempre vía OpenAI, sin importar el proveedor
    # primario de estimación.
    compression_model: str = "gpt-4o-mini"
    # Máximo de iteraciones que puede recorrer el Boss (cada iteración = 1
    # llamada al actor + 1 al Critic). 3 es el piso práctico: un borrador
    # inicial + dos reintentos dirigidos. Con solo 2, el actor suele no
    # poder atender todos los issues marcados en el único reintento
    # disponible, y el loop no converge.
    boss_max_iterations: int = 3

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()