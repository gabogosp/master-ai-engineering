"""Endpoint POST /embeddings/ingest (sesión 7, Paso 5).

Handler declarado `def` normal, NO `async def` -- `embed_many` (Paso 4)
hace una llamada SÍNCRONA y bloqueante a la API de OpenAI. Si este
handler fuera async, esa llamada bloquearía el único event loop del
proceso mientras espera, congelando también las peticiones a
/sessions/*. FastAPI corre los handlers `def` normales en un
threadpool automáticamente, así que basta con no poner `async` acá.
"""

import structlog
from fastapi import APIRouter, HTTPException
from openai import OpenAIError

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import PRICE_PER_1M_INPUT_TOKENS_USD, OpenAIEmbedder
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["embeddings"])

# Singleton seguro de instanciar al importar: tiktoken.encoding_for_model
# no depende de ninguna API key.
_chunker = JSONStructuralChunker()


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    chunks = _chunker.chunk(request.budgets)

    try:
        # OpenAIEmbedder() se instancia ACÁ, no a nivel de módulo: su
        # constructor lanza si falta OPENAI_API_KEY, y este servicio
        # puede correr con llm_provider=anthropic sin esa key -- un
        # singleton a nivel de módulo haría que todo app.main fallara al
        # importar solo por este endpoint, aunque nadie lo use.
        embedder = OpenAIEmbedder()
        embedded = embedder.embed_many(chunks)
    except RuntimeError as exc:
        logger.error("embedding_service_misconfigured", error=str(exc))
        raise HTTPException(
            status_code=500, detail="El servicio de embeddings está mal configurado."
        ) from exc
    except OpenAIError as exc:
        logger.error("embedding_ingest_failed", error_type=type(exc).__name__, exc_info=True)
        raise HTTPException(
            status_code=500, detail="No se pudo generar los embeddings. Inténtalo de nuevo."
        ) from exc

    total_tokens = sum(chunk.token_count for chunk in chunks)
    stats = IngestStats(
        total_budgets=len(request.budgets),
        total_chunks=len(embedded),
        total_tokens=total_tokens,
        estimated_cost_usd=total_tokens * PRICE_PER_1M_INPUT_TOKENS_USD / 1_000_000,
    )
    return IngestResponse(chunks=embedded, stats=stats)
