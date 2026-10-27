"""initial_schema

Revision ID: 0001
Revises: 
Create Date: 2026-10-27 08:16:20.982480

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("source_path", sa.Text, nullable=False),
        sa.Column("document_type", sa.String(50), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # Atributos estables del documento van en columnas propias
        # (document_type, ingested_at); lo que el pipeline de ingesta
        # pueda ir enriqueciendo sin forma fija queda acá, sin forzar
        # una migración por cada atributo nuevo.
        sa.Column("metadata", postgresql.JSONB, server_default="{}", nullable=False),
    )
    op.create_index("ix_documents_source_path", "documents", ["source_path"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "document_id",
            sa.BigInteger,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_type", sa.String(50), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        # Dimensión fija a la del modelo de embeddings en uso
        # (text-embedding-3-small). Nullable: separa la persistencia
        # del texto del cálculo del vector, para no bloquear el insert
        # del fragmento si el embedder falla.
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("metadata", postgresql.JSONB, server_default="{}", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_chunk_type", "chunks", ["chunk_type"])
    op.create_index(
        "ix_chunks_metadata_gin", "chunks", ["metadata"], postgresql_using="gin"
    )

    # Deliberadamente sin índice sobre `embedding` todavía: con el
    # volumen actual, un sequential scan es la referencia contra la que
    # se mide el impacto de cualquier índice (HNSW, IVFFlat) antes de
    # adoptarlo en producción.


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("chunks")
    op.drop_table("documents")
    # La extensión se deja instalada: es un objeto de todo el cluster de
    # Postgres, no de esta migración puntual -- dropearla podría romper
    # cualquier otra tabla/base que también la use, y recrearla en el
    # próximo upgrade es gratis (CREATE EXTENSION IF NOT EXISTS).
