import pytest
from pydantic import ValidationError

from app.embedding_pipeline.schemas import (
    Budget,
    ChunkMetadata,
    EmbeddedChunk,
    IngestRequest,
)


def _component(**overrides) -> dict:
    defaults = dict(
        component_id="AUTH-001",
        name="OAuth backend",
        description="Implementa OAuth 2.0.",
        tech_stack=["ruby_on_rails"],
        estimated_hours=120,
        complexity="high",
        dependencies=[],
    )
    defaults.update(overrides)
    return defaults


def _budget(**overrides) -> dict:
    defaults = dict(
        budget_id="BUD-TEST-001",
        client_metadata={"name": "TestCorp", "sector": "finance", "country": "ES"},
        project_summary="Mobile banking API",
        main_technology="ruby_on_rails",
        year=2024,
        total_estimated_hours=120,
        components=[_component()],
    )
    defaults.update(overrides)
    return defaults


def test_budget_parses_with_all_fields():
    budget = Budget.model_validate(_budget())
    assert budget.client_metadata.sector == "finance"
    assert budget.components[0].component_id == "AUTH-001"


def test_sector_rejects_value_outside_the_known_universe():
    with pytest.raises(ValidationError):
        Budget.model_validate(_budget(client_metadata={"name": "X", "sector": "aerospace", "country": "ES"}))


def test_complexity_rejects_value_outside_low_medium_high():
    with pytest.raises(ValidationError):
        Budget.model_validate(_budget(components=[_component(complexity="extreme")]))


def test_ingest_request_rejects_an_empty_budget_list():
    with pytest.raises(ValidationError):
        IngestRequest.model_validate({"budgets": []})


def test_embedded_chunk_rejects_an_embedding_with_the_wrong_dimension():
    metadata = ChunkMetadata(
        budget_id="BUD-TEST-001",
        component_id="AUTH-001",
        client_sector="finance",
        main_technology="ruby_on_rails",
        year=2024,
        complexity="high",
        estimated_hours=120,
    )
    with pytest.raises(ValidationError):
        EmbeddedChunk(
            chunk_id="BUD-TEST-001::AUTH-001",
            text="algo",
            metadata=metadata,
            token_count=5,
            embedding=[0.1, 0.2],  # 2 floats, no 1536
        )


def test_embedded_chunk_accepts_exactly_1536_floats():
    metadata = ChunkMetadata(
        budget_id="BUD-TEST-001",
        component_id="AUTH-001",
        client_sector="finance",
        main_technology="ruby_on_rails",
        year=2024,
        complexity="high",
        estimated_hours=120,
    )
    chunk = EmbeddedChunk(
        chunk_id="BUD-TEST-001::AUTH-001",
        text="algo",
        metadata=metadata,
        token_count=5,
        embedding=[0.0] * 1536,
    )
    assert len(chunk.embedding) == 1536
