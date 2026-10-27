from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIError

import app.embedding_pipeline.router as router_module
from app.db import get_session
from app.main import app


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


def _ingest_payload(source_path: str = "data/budgets/test.json", budget_id: str = "BUD-A") -> dict:
    return {
        "source_path": source_path,
        "document_type": "historical_budget",
        "content": _budget_payload(budget_id),
    }


def _fake_embed_many(chunks):
    """Reemplaza OpenAIEmbedder.embed_many sin pegarle a la API real --
    el endpoint solo necesita que cada chunk termine con un `embedding`
    de 1536 floats para pasar la validación de EmbeddedChunk."""
    from app.embedding_pipeline.schemas import EmbeddedChunk

    return [EmbeddedChunk(**chunk.model_dump(), embedding=[0.0] * 1536) for chunk in chunks]


class _FakeSession:
    """Doble de prueba de AsyncSession: cubre exactamente los métodos
    que el router usa (scalar/add/add_all/commit/execute), sin hablar
    con Postgres de verdad -- igual que el embedder, la persistencia
    real se ejercita en la verificación manual contra el contenedor,
    no en estos tests unitarios."""

    def __init__(self, existing_document=None, execute_rows=None):
        self.existing_document = existing_document
        self.execute_rows = execute_rows or []
        self.added: list = []
        self.committed = False
        self._next_id = 100

    async def scalar(self, _stmt):
        return self.existing_document

    def add(self, obj):
        self.added.append(obj)

    def add_all(self, objs):
        self.added.extend(objs)

    async def commit(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = self._next_id
                self._next_id += 1
        self.committed = True

    async def execute(self, _stmt):
        return SimpleNamespace(all=lambda: self.execute_rows)


@pytest.fixture
def fake_session():
    return _FakeSession()


@pytest.fixture(autouse=True)
def _override_db_session(fake_session):
    async def override():
        yield fake_session

    app.dependency_overrides[get_session] = override
    yield
    app.dependency_overrides.pop(get_session, None)


# --- POST /embeddings/ingest ---


def test_ingest_persists_document_and_chunks(client, fake_session):
    fake_embedder = MagicMock()
    fake_embedder.embed_many.side_effect = _fake_embed_many

    with patch.object(router_module, "OpenAIEmbedder", return_value=fake_embedder):
        response = client.post("/embeddings/ingest", json=_ingest_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["chunks_created"] == 1
    assert body["embedding_dimension"] == 1536
    assert body["ingestion_time_ms"] >= 0
    assert isinstance(body["document_id"], int)

    # Un Document + un Chunk, confirmados en la misma transacción.
    assert fake_session.committed is True
    assert len(fake_session.added) == 2


def test_ingest_returns_409_when_the_source_path_already_exists(client, fake_session):
    fake_session.existing_document = SimpleNamespace(id=42)

    response = client.post("/embeddings/ingest", json=_ingest_payload())

    assert response.status_code == 409
    assert response.json() == {"detail": "Document already ingested", "document_id": 42}
    # Nada se escribe si el documento ya existía.
    assert fake_session.committed is False


def test_ingest_returns_422_on_malformed_payload(client):
    response = client.post(
        "/embeddings/ingest",
        json={"source_path": "x", "document_type": "y", "content": {"budget_id": "sin-el-resto"}},
    )
    assert response.status_code == 422


def test_ingest_returns_500_when_the_embeddings_api_fails(client):
    fake_embedder = MagicMock()
    boom = APIError(message="boom", request=httpx.Request("POST", "https://api.openai.com"), body=None)
    fake_embedder.embed_many.side_effect = boom

    with patch.object(router_module, "OpenAIEmbedder", return_value=fake_embedder):
        response = client.post("/embeddings/ingest", json=_ingest_payload())

    assert response.status_code == 500


def test_ingest_returns_500_when_the_embedder_is_misconfigured(client):
    with patch.object(router_module, "OpenAIEmbedder", side_effect=RuntimeError("falta la API key")):
        response = client.post("/embeddings/ingest", json=_ingest_payload())

    assert response.status_code == 500


# --- POST /search ---


def _chunk_row(**overrides) -> SimpleNamespace:
    defaults = dict(
        id=1,
        document_id=1,
        chunk_type="budget_component",
        content="OAuth 2.0 authentication backend for fintech",
        extra={"sector": "finance"},
        distance=0.231,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_search_returns_ranked_results(client, fake_session):
    fake_session.execute_rows = [
        _chunk_row(id=1, distance=0.1),
        _chunk_row(id=2, distance=0.3),
    ]
    fake_embedder = MagicMock()
    fake_embedder.embed_one.return_value = [0.0] * 1536

    with patch.object(router_module, "OpenAIEmbedder", return_value=fake_embedder):
        response = client.post("/search", json={"query": "OAuth backend", "k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "OAuth backend"
    assert body["k"] == 2
    assert [r["chunk_id"] for r in body["results"]] == [1, 2]
    assert body["results"][0]["distance"] == 0.1


def test_search_returns_422_when_k_is_not_positive(client):
    response = client.post("/search", json={"query": "OAuth backend", "k": 0})
    assert response.status_code == 422


def test_search_returns_500_when_the_embeddings_api_fails(client):
    fake_embedder = MagicMock()
    boom = APIError(message="boom", request=httpx.Request("POST", "https://api.openai.com"), body=None)
    fake_embedder.embed_one.side_effect = boom

    with patch.object(router_module, "OpenAIEmbedder", return_value=fake_embedder):
        response = client.post("/search", json={"query": "OAuth backend", "k": 5})

    assert response.status_code == 500
