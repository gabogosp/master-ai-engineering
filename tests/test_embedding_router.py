from unittest.mock import MagicMock, patch

import httpx
from openai import APIError

import app.embedding_pipeline.router as router_module


def _budget_payload(budget_id: str = "BUD-A") -> dict:
    return {
        "budget_id": budget_id,
        "client_metadata": {"name": "TestCorp", "sector": "finance", "country": "ES"},
        "project_summary": "Mobile banking API",
        "main_technology": "ruby_on_rails",
        "year": 2024,
        "total_estimated_hours": 120,
        "components": [
            {
                "component_id": "AUTH-001",
                "name": "OAuth backend",
                "description": "Implementa OAuth 2.0.",
                "tech_stack": ["ruby_on_rails"],
                "estimated_hours": 120,
                "complexity": "high",
                "dependencies": [],
            }
        ],
    }


def _fake_embed_many(chunks):
    """Reemplaza OpenAIEmbedder.embed_many sin pegarle a la API real --
    el endpoint solo necesita que cada chunk termine con un `embedding`
    de 1536 floats para pasar la validación de EmbeddedChunk."""
    from app.embedding_pipeline.schemas import EmbeddedChunk

    return [EmbeddedChunk(**chunk.model_dump(), embedding=[0.0] * 1536) for chunk in chunks]


def test_ingest_returns_200_with_chunks_and_stats(client):
    fake_embedder = MagicMock()
    fake_embedder.embed_many.side_effect = _fake_embed_many

    with patch.object(router_module, "OpenAIEmbedder", return_value=fake_embedder):
        response = client.post("/embeddings/ingest", json={"budgets": [_budget_payload()]})

    assert response.status_code == 200
    body = response.json()
    assert body["stats"]["total_budgets"] == 1
    assert body["stats"]["total_chunks"] == 1
    assert body["stats"]["total_tokens"] > 0
    assert body["stats"]["estimated_cost_usd"] >= 0
    assert len(body["chunks"][0]["embedding"]) == 1536


def test_ingest_returns_422_on_malformed_payload(client):
    response = client.post("/embeddings/ingest", json={"budgets": [{"budget_id": "sin-el-resto"}]})
    assert response.status_code == 422


def test_ingest_returns_500_when_the_embeddings_api_fails(client):
    fake_embedder = MagicMock()
    boom = APIError(message="boom", request=httpx.Request("POST", "https://api.openai.com"), body=None)
    fake_embedder.embed_many.side_effect = boom

    with patch.object(router_module, "OpenAIEmbedder", return_value=fake_embedder):
        response = client.post("/embeddings/ingest", json={"budgets": [_budget_payload()]})

    assert response.status_code == 500


def test_ingest_returns_500_when_the_embedder_is_misconfigured(client):
    with patch.object(router_module, "OpenAIEmbedder", side_effect=RuntimeError("falta la API key")):
        response = client.post("/embeddings/ingest", json={"budgets": [_budget_payload()]})

    assert response.status_code == 500
