"""Modelos Pydantic del pipeline de embeddings.

Mismo patrón que app/schemas.py: Pydantic v2, `Field` con validaciones
explícitas donde el dominio lo permite, docstrings explicando el
PORQUÉ. Los nombres de campos van en inglés aunque los comentarios
siguen en español como el resto del repo.
"""

from pydantic import BaseModel, Field
from typing import Literal

# Universo cerrado SOLO para el sample de presupuestos actual (ver
# data/budgets_sample.json) -- `sector` y `complexity` son
# clasificaciones acotadas a propósito (4 y 3 valores respectivamente).
# `country` y `main_technology` se dejan como `str` libre: son campos
# abiertos en el mundo real (código ISO de país, cualquier stack
# tecnológico nuevo) y cerrarlos con un Literal rompería apenas el
# dataset crezca con presupuestos nuevos.
Sector = Literal["ecommerce", "finance", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]


class ClientMetadata(BaseModel):
    name: str = Field(min_length=1)
    sector: Sector
    country: str = Field(min_length=2, max_length=2)


class BudgetComponent(BaseModel):
    """Un componente de un presupuesto -- la unidad que el chunker
    convierte 1:1 en un Chunk."""

    component_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tech_stack: list[str] = Field(min_length=1)
    estimated_hours: int = Field(gt=0)
    complexity: Complexity
    dependencies: list[str] = Field(default_factory=list)


class Budget(BaseModel):
    budget_id: str = Field(min_length=1)
    client_metadata: ClientMetadata
    project_summary: str = Field(min_length=1)
    main_technology: str = Field(min_length=1)
    year: int = Field(ge=2000, le=2100)
    total_estimated_hours: int = Field(gt=0)
    components: list[BudgetComponent] = Field(min_length=1)


class ChunkMetadata(BaseModel):
    """Campos filtrables que viajan con el chunk pero NO se embeben --
    sirven para cuando haya una base vectorial (pgvector) y se quiera
    filtrar por sector/tecnología/año antes de buscar por similitud.
    Tipados en vez de un dict suelto para no arriesgar un typo en una
    clave que nadie valida."""

    budget_id: str
    component_id: str
    client_sector: Sector
    main_technology: str
    year: int
    complexity: Complexity
    estimated_hours: int


class Chunk(BaseModel):
    """Un fragmento listo para embeber. `text` es lo que efectivamente
    se manda a la API de embeddings; `metadata` viaja aparte, nunca se
    concatena al texto."""

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: ChunkMetadata
    token_count: int = Field(ge=0)


class EmbeddedChunk(Chunk):
    """`Chunk` + el vector. 1536 floats porque text-embedding-3-small
    se usa con la dimensión por defecto -- no se fuerza una dimensión
    menor (Matryoshka) acá."""

    embedding: list[float] = Field(min_length=1536, max_length=1536)


class IngestRequest(BaseModel):
    """Un documento a la vez -- cada presupuesto histórico se persiste
    como su propia fila en `documents`, con sus fragmentos en `chunks`.
    `content` es el JSON del presupuesto, con la misma forma que
    `Budget` (lo valida el chunker antes de usarlo)."""

    source_path: str = Field(min_length=1)
    document_type: str = Field(min_length=1)
    content: Budget


class IngestResponse(BaseModel):
    document_id: int
    chunks_created: int = Field(ge=0)
    embedding_dimension: int
    ingestion_time_ms: float = Field(ge=0)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(gt=0)
    # "vector" preserva el comportamiento previo (solo k-NN denso).
    # "hybrid" suma la rama léxica (tsvector) fusionada por RRF.
    search_mode: Literal["vector", "hybrid"] = "vector"
    # Recall-then-rerank: trae `recall_k` candidatos y los reordena con
    # el cross-encoder antes de cortar a `k`. Apagado por default porque
    # tiene un costo de latencia real (ver golden set de medición).
    rerank: bool = False
    recall_k: int = Field(default=50, gt=0)


class SearchResult(BaseModel):
    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    k: int
    search_mode: Literal["vector", "hybrid"]
    rerank: bool
    search_time_ms: float = Field(ge=0)
    # Desglose para el golden set: cuánto fue generar el embedding +
    # correr las ramas de recall vs. cuánto fue el reranking en sí.
    retrieval_time_ms: float = Field(ge=0)
    rerank_time_ms: float = Field(ge=0)
    results: list[SearchResult]
