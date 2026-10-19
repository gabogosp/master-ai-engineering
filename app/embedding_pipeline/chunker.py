"""Chunker estructural para presupuestos JSON (sesión 7, Paso 3).

Granularidad: un componente de un presupuesto = un chunk. No hay
overlap ni fixed-size splitting de las descripciones largas -- el
enunciado lo pide explícitamente así: la estructura del JSON ya da la
unidad de chunk, partirla a mano sería reinventar lo que recursive/
semantic chunking resuelven de forma más general, y eso está fuera de
scope de este ejercicio (se compara en la sesión en vivo).
"""

import tiktoken

from app.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk, ChunkMetadata

_EMBEDDING_MODEL = "text-embedding-3-small"


class JSONStructuralChunker:
    """Convierte una lista de `Budget` en una lista de `Chunk`, uno por
    `BudgetComponent`."""

    def __init__(self) -> None:
        self._encoding = tiktoken.encoding_for_model(_EMBEDDING_MODEL)

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for budget in budgets:
            for component in budget.components:
                chunks.append(self._chunk_one(budget, component))
        return chunks

    def _chunk_one(self, budget: Budget, component: BudgetComponent) -> Chunk:
        text = self._render_text(budget, component)
        metadata = ChunkMetadata(
            budget_id=budget.budget_id,
            component_id=component.component_id,
            client_sector=budget.client_metadata.sector,
            main_technology=budget.main_technology,
            year=budget.year,
            complexity=component.complexity,
            estimated_hours=component.estimated_hours,
        )
        return Chunk(
            chunk_id=f"{budget.budget_id}::{component.component_id}",
            text=text,
            metadata=metadata,
            token_count=len(self._encoding.encode(text)),
        )

    @staticmethod
    def _render_text(budget: Budget, component: BudgetComponent) -> str:
        """El contexto del presupuesto padre va ANTES del detalle del
        componente a propósito (contextual chunk header, ver artículo 3
        del material async): sin él, un chunk como "Authentication
        backend" pierde la pista de a qué cliente/sector/stack
        pertenece en cuanto se lo separa de sus hermanos."""
        tech_stack = ", ".join(component.tech_stack)
        return (
            f"[Project: {budget.project_summary}]\n"
            f"[Client sector: {budget.client_metadata.sector} | Year: {budget.year} | "
            f"Main tech: {budget.main_technology}]\n"
            "\n"
            f"Component: {component.name}\n"
            f"Description: {component.description}\n"
            f"Tech stack: {tech_stack}\n"
            f"Complexity: {component.complexity}\n"
            f"Estimated hours: {component.estimated_hours}"
        )
