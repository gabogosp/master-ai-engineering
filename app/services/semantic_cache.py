"""Cache semántica para /api/v1/estimate.

Dos requests se consideran "el mismo pedido" cuando:

1. Su **bucket** coincide exacto: `prompt_version:project_type:detail_level:
   output_format`. Dos requests con distintas opciones de formulario NUNCA
   comparten entrada de cache aunque la descripción sea parecida -- el
   prompt renderizado es distinto, así que la estimación debería serlo
   también.
2. La similitud coseno entre los embeddings de la descripción es >= threshold.

Con log_only=True la cache sigue haciendo el lookup y logueando el score,
pero nunca sirve el hit -- para calibrar el threshold contra tráfico real
antes de activarla en producción.

El store usa redis/redis-stack: el redis:7-alpine vanilla no trae el
módulo RediSearch y SearchIndex.create() falla al arrancar. Si la cache
semántica no está disponible (sin OPENAI_API_KEY, o sin RediSearch), se
degrada a "sin cache" en vez de romper el endpoint -- igual que
ExactMatchCache.
"""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog
from redis.asyncio import Redis
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from redisvl.utils.vectorize import OpenAITextVectorizer

from app.config import settings
from app.schemas import EstimationRequest, EstimationResult

logger = structlog.get_logger(__name__)

EMBEDDING_DIMS = 1536  # text-embedding-3-small

def _index_schema(index_name: str) -> dict[str, Any]:
    return {
        "index": {
            "name": index_name,
            # El prefix incluye el nombre del índice para que un índice de
            # test (ver tests/test_semantic_cache.py) no pise las keys del
            # índice de producción en el mismo Redis.
            "prefix": f"estimation:semantic:{index_name}",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "bucket", "type": "tag"},
            {"name": "payload_json", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": EMBEDDING_DIMS,
                    "distance_metric": "cosine",
                    "algorithm": "flat",
                },
            },
        ],
    }


def _to_bytes(vector: list[float]) -> bytes:
    """RediSearch guarda los vectores como bytes float32; redisvl rechaza listas."""
    return np.array(vector, dtype=np.float32).tobytes()


@dataclass
class SemanticCacheEntry:
    """Lo que devuelve un hit: el resultado más la metadata real de la
    llamada que lo generó originalmente (no placeholders) -- igual que
    ExactMatchCache, que también devuelve el model/provider/tokens de la
    llamada que se cacheó."""

    result: EstimationResult
    model: str
    provider: str
    input_tokens: int
    output_tokens: int


class EstimationSemanticCache:
    """Cache de similitud vectorial sobre redisvl + Redis Stack."""

    def __init__(
        self,
        *,
        redis_client: Redis,
        vectorizer: OpenAITextVectorizer,
        threshold: float,
        ttl: int,
        log_only: bool = False,
        index_name: str = "estimations",
    ) -> None:
        self._vectorizer = vectorizer
        self._threshold = threshold
        self._ttl = ttl
        self._log_only = log_only
        self._index = AsyncSearchIndex.from_dict(
            _index_schema(index_name), redis_client=redis_client
        )
        self._ready = False

    async def _ensure_index(self) -> bool:
        # create() es async y pega contra Redis -- no se puede llamar desde
        # __init__ (no es async), así que se hace de forma perezosa en el
        # primer uso.
        if self._ready:
            return True
        try:
            await self._index.create(overwrite=False)
        except Exception as exc:
            logger.warning("semantic_cache_index_unavailable", error=str(exc)[:200])
            return False
        self._ready = True
        return True

    @staticmethod
    def bucket_for(request: EstimationRequest, prompt_version: str) -> str:
        return (
            f"{prompt_version}:{request.project_type.value}:"
            f"{request.detail_level.value}:{request.output_format.value}"
        )

    async def lookup(
        self, request: EstimationRequest, prompt_version: str
    ) -> SemanticCacheEntry | None:
        if not await self._ensure_index():
            return None

        bucket = self.bucket_for(request, prompt_version)
        try:
            embedding = await self._vectorizer.aembed(request.description)
            query = VectorQuery(
                vector=_to_bytes(embedding),
                vector_field_name="embedding",
                return_fields=["payload_json", "bucket"],
                num_results=1,
                return_score=True,
                filter_expression=Tag("bucket") == bucket,
            )
            results = await self._index.query(query)
        except Exception as exc:
            logger.warning("semantic_cache_lookup_failed", error=str(exc)[:200])
            return None

        if not results:
            logger.info("semantic_cache_miss", bucket=bucket, reason="empty_index")
            return None

        hit = results[0]
        # redisvl devuelve la *distancia* coseno (0 = idéntico, hasta 2 = opuesto).
        distance = float(hit.get("vector_distance", 1.0))
        similarity = 1.0 - distance
        logger.info(
            "semantic_cache_lookup",
            bucket=bucket,
            similarity=round(similarity, 4),
            threshold=self._threshold,
        )

        if similarity < self._threshold:
            logger.info("semantic_cache_miss", bucket=bucket, reason="below_threshold")
            return None

        if self._log_only:
            logger.info(
                "semantic_cache_hit_log_only", bucket=bucket, similarity=round(similarity, 4)
            )
            return None

        logger.info("semantic_cache_hit", bucket=bucket, similarity=round(similarity, 4))
        payload = json.loads(hit["payload_json"])
        return SemanticCacheEntry(
            result=EstimationResult.model_validate(payload["result"]),
            model=payload["model"],
            provider=payload["provider"],
            input_tokens=payload["input_tokens"],
            output_tokens=payload["output_tokens"],
        )

    async def store(
        self,
        request: EstimationRequest,
        prompt_version: str,
        result: EstimationResult,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        if not await self._ensure_index():
            return

        bucket = self.bucket_for(request, prompt_version)
        payload = {
            "result": result.model_dump(mode="json"),
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        try:
            embedding = await self._vectorizer.aembed(request.description)
            await self._index.load(
                [
                    {
                        "bucket": bucket,
                        "payload_json": json.dumps(payload),
                        "embedding": _to_bytes(embedding),
                    }
                ],
                ttl=self._ttl,
            )
            logger.info("semantic_cache_stored", bucket=bucket, ttl=self._ttl)
        except Exception as exc:
            logger.warning(
                "semantic_cache_store_failed", error_type=type(exc).__name__, error=str(exc)[:200]
            )


_semantic_cache: EstimationSemanticCache | None = None
_attempted = False


def get_semantic_cache() -> EstimationSemanticCache | None:
    """Singleton perezoso. Devuelve None (cache deshabilitada) si falta la
    API key de OpenAI -- la creación del índice en sí se reintenta sola en
    el primer lookup/store si Redis Stack todavía no está arriba."""
    global _semantic_cache, _attempted
    if _attempted:
        return _semantic_cache
    _attempted = True

    if not settings.openai_api_key:
        logger.warning("semantic_cache_disabled", reason="no_openai_key")
        return None

    # OpenAITextVectorizer hace una llamada real a embeddings en su
    # constructor para detectar la dimensión del modelo -- con una API key
    # dummy (tests) o un problema transitorio de red, esto lanza. Igual que
    # ExactMatchCache/get_openai_client, una falla ACÁ no debe romper el
    # resto del pipeline: el resultado es "cache semántica deshabilitada",
    # no un 500 en /estimate.
    try:
        vectorizer = OpenAITextVectorizer(
            model=settings.embedding_model,
            api_config={"api_key": settings.openai_api_key},
        )
        redis_client: Redis = Redis.from_url(settings.redis_url, decode_responses=False)
        _semantic_cache = EstimationSemanticCache(
            redis_client=redis_client,
            vectorizer=vectorizer,
            threshold=settings.semantic_cache_threshold,
            ttl=settings.semantic_cache_ttl_seconds,
            log_only=settings.semantic_cache_log_only,
        )
    except Exception as exc:
        logger.warning(
            "semantic_cache_disabled",
            reason="setup_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return None
    return _semantic_cache
