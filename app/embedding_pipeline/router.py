"""Endpoints del pipeline de embeddings: ingesta persistida y búsqueda
semántica sobre lo ya ingestado.
"""

import asyncio
import time

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAIError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.models import Chunk as ChunkRow
from app.embedding_pipeline.models import Document
from app.embedding_pipeline.retrieval import retrieve
from app.embedding_pipeline.schemas import (
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

logger = structlog.get_logger(__name__)

#  Sin prefix a nivel de router: /ingest queda bajo /embeddings (se
# agrega al path de esa ruta puntual) pero /search es top-level, no
# anidado -- un prefix compartido para todo el módulo forzaría los
# dos bajo el mismo namespace.
router = APIRouter(tags=["embeddings"])

# Singleton seguro de instanciar al importar: tiktoken.encoding_for_model
# no depende de ninguna API key.
_chunker = JSONStructuralChunker()


@router.post("/embeddings/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest, session: AsyncSession = Depends(get_session)
) -> IngestResponse | JSONResponse:
    t0 = time.perf_counter()

    existing = await session.scalar(
        select(Document).where(Document.source_path == request.source_path)
    )
    if existing is not None:
        return JSONResponse(
            status_code=409,
            content={"detail": "Document already ingested", "document_id": existing.id},
        )

    budget = request.content
    chunks = _chunker.chunk([budget])

    try:
        # OpenAIEmbedder() se instancia ACÁ, no a nivel de módulo: su
        # constructor lanza si falta OPENAI_API_KEY, y este servicio
        # puede correr con llm_provider=anthropic sin esa key -- un
        # singleton a nivel de módulo haría que todo app.main fallara al
        # importar solo por este endpoint, aunque nadie lo use.
        embedder = OpenAIEmbedder()
        # embed_many es una llamada SÍNCRONA y bloqueante -- correrla
        # directo congelaría el único event loop del proceso (y con él,
        # cualquier otra petición en curso) mientras esperamos a la API
        # de OpenAI. to_thread la delega a un hilo aparte.
        embedded = await asyncio.to_thread(embedder.embed_many, chunks)
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

    # Nada se escribió en la sesión todavía -- si el embedder falla, no
    # hay ningún documento a medio guardar que limpiar. El documento y
    # sus fragmentos se agregan y confirman juntos, en el mismo commit.
    document = Document(
        source_path=request.source_path,
        document_type=request.document_type,
        extra={
            "budget_id": budget.budget_id,
            "sector": budget.client_metadata.sector,
            "main_technology": budget.main_technology,
            "year": budget.year,
        },
    )
    chunk_rows = [
        ChunkRow(
            document=document,
            chunk_type="budget_component",
            content=chunk.text,
            embedding=chunk.embedding,
            extra=chunk.metadata.model_dump(),
        )
        for chunk in embedded
    ]

    session.add(document)
    session.add_all(chunk_rows)
    await session.commit()

    ingestion_time_ms = (time.perf_counter() - t0) * 1000
    return IngestResponse(
        document_id=document.id,
        chunks_created=len(chunk_rows),
        embedding_dimension=len(embedded[0].embedding),
        ingestion_time_ms=round(ingestion_time_ms, 1),
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest, session: AsyncSession = Depends(get_session)
) -> SearchResponse:
    t0 = time.perf_counter()

    try:
        embedder = OpenAIEmbedder()
        query_vector = await asyncio.to_thread(embedder.embed_one, request.query)
    except RuntimeError as exc:
        logger.error("embedding_service_misconfigured", error=str(exc))
        raise HTTPException(
            status_code=500, detail="El servicio de embeddings está mal configurado."
        ) from exc
    except OpenAIError as exc:
        logger.error("embedding_search_failed", error_type=type(exc).__name__, exc_info=True)
        raise HTTPException(
            status_code=500, detail="No se pudo generar el embedding de la consulta."
        ) from exc

    chunks, metrics = await retrieve(
        session,
        query_text=request.query,
        query_vector=query_vector,
        search_mode=request.search_mode,
        rerank=request.rerank,
        k=request.k,
        recall_k=request.recall_k,
    )

    search_time_ms = (time.perf_counter() - t0) * 1000
    return SearchResponse(
        query=request.query,
        k=request.k,
        search_mode=request.search_mode,
        rerank=request.rerank,
        search_time_ms=round(search_time_ms, 1),
        retrieval_time_ms=metrics["retrieval_time_ms"],
        rerank_time_ms=metrics["rerank_time_ms"],
        results=[
            SearchResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                chunk_type=chunk.chunk_type,
                content=chunk.content,
                distance=chunk.distance,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ],
    )
