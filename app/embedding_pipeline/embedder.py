"""Embedder sobre la API de OpenAI (sesión 7, Paso 4).

Usa un cliente SÍNCRONO (`openai.OpenAI`), no el `AsyncOpenAI` que ya
expone app.services.openai_client para moderación -- el enunciado pide
literalmente `def embed_one(...)` / `def embed_many(...)` sin
`async def`, y mezclar un cliente async en un método sync solo para
este módulo obligaría a correr un event loop anidado sin necesidad.
"""

import time

import structlog
from openai import OpenAI, RateLimitError

from app.config import settings
from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

logger = structlog.get_logger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

# Precio vigente al escribir este módulo -- CAMBIA con el tiempo, por
# eso vive como constante nombrada y no como un número suelto dentro
# del cálculo de costo (ver Paso 5, donde el router la usa para llenar
# IngestStats.estimated_cost_usd).
PRICE_PER_1M_INPUT_TOKENS_USD = 0.02

# 3 reintentos con esperas 1s, 2s, 4s -- el enunciado los pide exactos.
_RETRY_WAITS_SECONDS = (1, 2, 4)


class OpenAIEmbedder:
    """Envuelve la API de embeddings de OpenAI: batching, reintento
    exponencial simple ante rate limit, y logging estructurado por
    batch (chunks, tokens, latencia)."""

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY no está configurada -- este pipeline no puede "
                "generar embeddings sin ella (a diferencia de la moderación o "
                "la cache semántica, acá no hay un camino de degradación)."
            )
        self._client = OpenAI(api_key=settings.openai_api_key)

    def embed_one(self, text: str) -> list[float]:
        response = self._create_with_retry([text])
        return response.data[0].embedding

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Llama a la API en batches (no una llamada por chunk) -- el
        tamaño recomendado por el enunciado es 100 chunks por llamada."""
        embedded: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            t0 = time.perf_counter()
            response = self._create_with_retry([c.text for c in batch])
            latency_ms = (time.perf_counter() - t0) * 1000

            logger.info(
                "embedding_batch_completed",
                chunks_in_batch=len(batch),
                total_tokens=response.usage.total_tokens,
                latency_ms=round(latency_ms, 1),
            )

            # La API devuelve `data` en el mismo orden que `input` --
            # zip directo, sin tener que reordenar por `item.index`.
            for chunk, item in zip(batch, response.data, strict=True):
                embedded.append(EmbeddedChunk(**chunk.model_dump(), embedding=item.embedding))
        return embedded

    def _create_with_retry(self, texts: list[str]):
        last_exc: RateLimitError | None = None
        for attempt, wait_seconds in enumerate((0, *_RETRY_WAITS_SECONDS)):
            if wait_seconds:
                time.sleep(wait_seconds)
            try:
                return self._client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            except RateLimitError as exc:
                last_exc = exc
                logger.warning(
                    "embedding_rate_limited", attempt=attempt + 1, wait_seconds=wait_seconds
                )
        assert last_exc is not None
        raise last_exc
