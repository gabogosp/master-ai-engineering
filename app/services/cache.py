import hashlib
import json

import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)


def build_cache_key(**parts: object) -> str:
    """Clave determinista a partir de todos los parámetros que afectan la respuesta.

    Cualquier campo que influya en el output del LLM debe pasarse aquí. Si
    alguno cambia (por ejemplo, el system prompt porque se agregó un ejemplo
    CAG nuevo), la clave cambia y la entrada vieja queda huérfana hasta que
    expira por TTL.
    """
    raw = json.dumps(parts, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"llm:{digest}"


class ExactMatchCache:
    """Cache exact-match sobre Redis. Si Redis no está disponible, se degrada
    a "sin cache" en vez de romper la petición: el cache es una optimización,
    no una dependencia dura del servicio."""

    def __init__(self, redis_url: str, ttl_seconds: int):
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl_seconds

    async def get(self, key: str) -> dict | None:
        try:
            raw = await self._client.get(key)
        except redis.RedisError as exc:
            logger.warning("cache_unavailable", operation="get", error=str(exc))
            return None
        return json.loads(raw) if raw else None

    async def set(self, key: str, value: dict) -> None:
        try:
            await self._client.set(key, json.dumps(value), ex=self._ttl)
        except redis.RedisError as exc:
            logger.warning("cache_unavailable", operation="set", error=str(exc))
