from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.schemas import Budget


def _budget(budget_id: str, n_components: int = 2) -> Budget:
    return Budget.model_validate(
        {
            "budget_id": budget_id,
            "client_metadata": {"name": "TestCorp", "sector": "finance", "country": "ES"},
            "project_summary": "Mobile banking API with OAuth 2.0",
            "main_technology": "ruby_on_rails",
            "year": 2024,
            "total_estimated_hours": 240,
            "components": [
                {
                    "component_id": f"COMP-{i:03d}",
                    "name": f"Component {i}",
                    "description": f"Descripción del componente {i}.",
                    "tech_stack": ["ruby_on_rails", "postgresql"],
                    "estimated_hours": 40,
                    "complexity": "medium",
                    "dependencies": [],
                }
                for i in range(1, n_components + 1)
            ],
        }
    )


def test_chunk_produces_one_chunk_per_component():
    chunker = JSONStructuralChunker()
    chunks = chunker.chunk([_budget("BUD-A", n_components=3)])
    assert len(chunks) == 3


def test_chunk_id_format_is_budget_id_double_colon_component_id():
    chunker = JSONStructuralChunker()
    chunks = chunker.chunk([_budget("BUD-A", n_components=1)])
    assert chunks[0].chunk_id == "BUD-A::COMP-001"


def test_chunk_text_includes_parent_context_and_component_detail():
    """El contexto del padre (project_summary, sector, tech) va ANTES
    del detalle del componente -- sin eso, "Component 1" perdería la
    pista de a qué cliente/sector pertenece."""
    chunker = JSONStructuralChunker()
    chunk = chunker.chunk([_budget("BUD-A", n_components=1)])[0]
    assert "Mobile banking API with OAuth 2.0" in chunk.text
    assert "finance" in chunk.text
    assert "Component: Component 1" in chunk.text
    assert "ruby_on_rails, postgresql" in chunk.text


def test_chunk_metadata_mirrors_budget_and_component_fields():
    chunker = JSONStructuralChunker()
    chunk = chunker.chunk([_budget("BUD-A", n_components=1)])[0]
    assert chunk.metadata.budget_id == "BUD-A"
    assert chunk.metadata.component_id == "COMP-001"
    assert chunk.metadata.client_sector == "finance"
    assert chunk.metadata.main_technology == "ruby_on_rails"
    assert chunk.metadata.estimated_hours == 40


def test_chunk_token_count_is_positive_and_deterministic():
    chunker = JSONStructuralChunker()
    chunk_a = chunker.chunk([_budget("BUD-A", n_components=1)])[0]
    chunk_b = chunker.chunk([_budget("BUD-A", n_components=1)])[0]
    assert chunk_a.token_count > 0
    assert chunk_a.token_count == chunk_b.token_count


def test_chunk_flattens_multiple_budgets_in_order():
    chunker = JSONStructuralChunker()
    chunks = chunker.chunk([_budget("BUD-A", n_components=2), _budget("BUD-B", n_components=1)])
    assert [c.chunk_id for c in chunks] == ["BUD-A::COMP-001", "BUD-A::COMP-002", "BUD-B::COMP-001"]
