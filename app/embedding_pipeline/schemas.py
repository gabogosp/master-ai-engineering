"""Modelos Pydantic del pipeline de embeddings (sesión 7, Paso 2).

Mismo patrón que app/schemas.py: Pydantic v2, `Field` con validaciones
explícitas donde el dominio lo permite, docstrings explicando el
PORQUÉ. Los nombres de campos van en inglés (lo pide el enunciado
explícitamente) aunque los comentarios siguen en español como el resto
del repo.
"""

from pydantic import BaseModel, Field
from typing import Literal

# Universo cerrado SOLO para el sample de 15 presupuestos de esta sesión
# (ver data/budgets_sample.json) -- `sector` y `complexity` son
# clasificaciones acotadas a propósito (4 y 3 valores respectivamente).
# `country` y `main_technology` se dejan como `str` libre: son campos
# abiertos en el mundo real (código ISO de país, cualquier stack
# tecnológico nuevo) y cerrarlos con un Literal rompería apenas el
# dataset creciera en la sesión en vivo.
Sector = Literal["ecommerce", "finance", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]


class ClientMetadata(BaseModel):
    name: str = Field(min_length=1)
    sector: Sector
    country: str = Field(min_length=2, max_length=2)


class BudgetComponent(BaseModel):
    """Un componente de un presupuesto -- la unidad que, en el Paso 3,
    se convierte 1:1 en un Chunk."""

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
    sirven para cuando en sesión 8 (pgvector) se quiera filtrar por
    sector/tecnología/año antes de buscar por similitud. Son
    literalmente los que pide el Paso 3, tipados en vez de un dict
    suelto para no arriesgar un typo en una clave que nadie valida."""

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
    se usa con la dimensión por defecto (el enunciado pide explícitamente
    NO forzar una dimensión menor en este ejercicio)."""

    embedding: list[float] = Field(min_length=1536, max_length=1536)


class IngestRequest(BaseModel):
    budgets: list[Budget] = Field(min_length=1)


class IngestStats(BaseModel):
    total_budgets: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)


class IngestResponse(BaseModel):
    chunks: list[EmbeddedChunk]
    stats: IngestStats
