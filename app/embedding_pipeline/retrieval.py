"""Orquestación de retrieval: vectorial, híbrido (+ RRF) y reranking
opcional, detrás de un único punto de entrada (`retrieve`).

Separado de `router.py` a propósito: el router solo sabe de HTTP (parsear
el request, generar el embedding de la query, devolver la respuesta); la
decisión de qué ramas de búsqueda correr y cómo combinarlas vive acá.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Literal

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.embedding_pipeline.fusion import reciprocal_rank_fusion
from app.embedding_pipeline.models import Chunk as ChunkRow
from app.embedding_pipeline.reranker import CrossEncoderReranker

logger = structlog.get_logger(__name__)

SearchMode = Literal["vector", "hybrid"]

# Distancia que se le asigna a un candidato que entró SOLO por la rama
# léxica: nunca estuvo en el ranking vectorial, así que no tiene un
# coseno real que mostrar. Es un valor de display/sentinela -- ni la
# fusión RRF ni el reranker lo usan para decidir nada, solo ordenan por
# posición o por score del cross-encoder.
_NO_VECTOR_DISTANCE = 1.0

# Singleton seguro de instanciar al importar: el constructor no carga el
# modelo (carga perezosa en el primer `.rerank()`), igual que el chunker
# en router.py.
_reranker = CrossEncoderReranker(settings.reranker_model)


@dataclass
class RetrievedChunk:
    id: int
    document_id: int
    chunk_type: str
    content: str
    metadata: dict
    distance: float


async def _vector_search(
    session: AsyncSession, query_vector: list[float], limit: int
) -> list:
    distance = ChunkRow.embedding.cosine_distance(query_vector).label("distance")
    stmt = (
        select(
            ChunkRow.id,
            ChunkRow.document_id,
            ChunkRow.chunk_type,
            ChunkRow.content,
            ChunkRow.extra,
            distance,
        )
        .order_by(distance)
        .limit(limit)
    )
    return (await session.execute(stmt)).all()


async def _lexical_search(session: AsyncSession, query_text: str, limit: int) -> list:
    # Mismo idioma que la columna generada (migración 0002) -- el corpus
    # está en inglés, no en español.
    tsquery = func.plainto_tsquery("english", query_text)
    rank = func.ts_rank_cd(ChunkRow.search_vector, tsquery).label("rank")
    stmt = (
        select(
            ChunkRow.id,
            ChunkRow.document_id,
            ChunkRow.chunk_type,
            ChunkRow.content,
            ChunkRow.extra,
            rank,
        )
        .where(ChunkRow.search_vector.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(limit)
    )
    return (await session.execute(stmt)).all()


def _row_to_chunk(row, *, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=row.id,
        document_id=row.document_id,
        chunk_type=row.chunk_type,
        content=row.content,
        metadata=row.extra,
        distance=distance,
    )


async def retrieve(
    session: AsyncSession,
    *,
    query_text: str,
    query_vector: list[float],
    search_mode: SearchMode = "vector",
    rerank: bool = False,
    k: int = 5,
    recall_k: int = 50,
) -> tuple[list[RetrievedChunk], dict]:
    """Recupera los `k` chunks más relevantes para `query_text`.

    `search_mode="vector"` es el comportamiento de antes (solo k-NN
    denso). `search_mode="hybrid"` suma una rama léxica sobre
    `search_vector` y fusiona ambas con RRF. `rerank=True` recupera un
    pool amplio (`recall_k`, p.ej. 50) y lo reordena con el cross-encoder
    antes de cortar a `k` -- la recuperación amplia solo tiene que no
    perder el documento relevante; el reranker es quien lo sube.

    Devuelve los chunks (mejor primero) y un diccionario de métricas
    (tiempos por etapa, candidatos evaluados) para instrumentar la
    medición del golden set.
    """
    started = time.perf_counter()

    # Recuperar amplio siempre que algo después vaya a reordenar (fusión
    # o reranking); recuperar exactamente `k` solo en el camino vectorial
    # puro, donde el orden ya es el final.
    wide = rerank or search_mode == "hybrid"
    vector_limit = recall_k if wide else k

    vector_rows = await _vector_search(session, query_vector, vector_limit)
    lexical_rows = []
    if search_mode == "hybrid":
        lexical_rows = await _lexical_search(session, query_text, recall_k)
    retrieval_ms = (time.perf_counter() - started) * 1000

    # Pool de candidatos (id -> chunk); si un id aparece en ambas ramas,
    # se queda con la distancia vectorial real en vez del sentinela.
    candidates: dict[int, RetrievedChunk] = {}
    for row in vector_rows:
        candidates[row.id] = _row_to_chunk(row, distance=float(row.distance))
    for row in lexical_rows:
        candidates.setdefault(row.id, _row_to_chunk(row, distance=_NO_VECTOR_DISTANCE))

    if search_mode == "hybrid":
        fused = reciprocal_rank_fusion(
            [[row.id for row in vector_rows], [row.id for row in lexical_rows]],
            k=settings.rrf_k,
        )
        ordered = [candidates[chunk_id] for chunk_id, _score in fused]
    else:
        ordered = [candidates[row.id] for row in vector_rows]

    rerank_ms = 0.0
    if rerank and ordered:
        rerank_started = time.perf_counter()
        recall_pool = ordered[:recall_k]
        final = await asyncio.to_thread(
            _reranker.rerank,
            query_text,
            recall_pool,
            top_n=k,
            text_of=lambda chunk: chunk.content,
        )
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
    else:
        final = ordered[:k]

    logger.info(
        "hybrid_retrieve_done",
        search_mode=search_mode,
        rerank=rerank,
        vector_hits=len(vector_rows),
        lexical_hits=len(lexical_rows),
        results=len(final),
    )

    metrics = {
        "vector_hits": len(vector_rows),
        "lexical_hits": len(lexical_rows),
        "retrieval_time_ms": round(retrieval_ms, 1),
        "rerank_time_ms": round(rerank_ms, 1),
    }
    return final, metrics
