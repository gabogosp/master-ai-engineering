"""chunk_search_vector

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Columna generada (no una tabla aparte a mantener en sync a mano) --
    # Postgres la recalcula sola en cada INSERT/UPDATE de `content`.
    #
    # Config 'english', no 'spanish': el corpus real (data/budgets_sample.json)
    # está en inglés -- verificado leyendo su contenido, no asumido --, así
    # que 'spanish' perdería el stemming y descartaría stopwords del idioma
    # equivocado.
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN search_vector tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """
    )
    op.create_index(
        "ix_chunks_search_vector_gin",
        "chunks",
        ["search_vector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_chunks_search_vector_gin", table_name="chunks")
    op.execute("ALTER TABLE chunks DROP COLUMN search_vector")
