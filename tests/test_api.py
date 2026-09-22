from unittest.mock import AsyncMock
import httpx
import openai
from fastapi.testclient import TestClient

from app.services.llm_service import (
    IncompleteEstimationError,
    LLMConfigurationError,
    LLMResult,
)


def test_health_endpoint(client: TestClient):
    """Verifica que /health responda 200 con el estado del servicio."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data
    assert "provider" in data
    assert "model" in data


def test_estimate_validation_too_short(client: TestClient):
    """La API debe rechazar con 422 transcripciones de menos de 20 caracteres."""
    response = client.post(
        "/api/v1/estimate",
        json={"transcription": "Demasiado corta"},
    )
    assert response.status_code == 422


def test_estimate_validation_missing_field(client: TestClient):
    """La API debe rechazar con 422 peticiones sin el campo transcription."""
    response = client.post("/api/v1/estimate", json={})
    assert response.status_code == 422


def test_estimate_success(client: TestClient, monkeypatch):
    """Verifica la respuesta 200 ante una estimación exitosa con LLM simulado."""
    mock_result = LLMResult(
        estimation="## Estimación Mock\nTotal: 100 horas",
        model="gpt-4o-mini-mock",
        provider="openai",
        input_tokens=500,
        output_tokens=120,
    )

    mock_generate = AsyncMock(return_value=mock_result)
    monkeypatch.setattr("app.routers.estimations.generate_estimation", mock_generate)

    payload = {
        "transcription": "El cliente necesita una plataforma para reservar turnos online con recordatorios por email."
    }
    response = client.post("/api/v1/estimate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["estimation"] == mock_result.estimation
    assert data["model"] == mock_result.model
    assert data["provider"] == mock_result.provider
    assert data["input_tokens"] == 500
    assert data["output_tokens"] == 120
    assert "created_at" in data


def test_estimate_incomplete_error(client: TestClient, monkeypatch):
    """Si el modelo corta la respuesta, debe responder 502."""
    mock_generate = AsyncMock(side_effect=IncompleteEstimationError("Respuesta cortada"))
    monkeypatch.setattr("app.routers.estimations.generate_estimation", mock_generate)

    response = client.post(
        "/api/v1/estimate",
        json={"transcription": "Transcripción válida con más de veinte caracteres para la prueba."},
    )
    assert response.status_code == 502
    assert "incompleta" in response.json()["detail"].lower()


def test_estimate_configuration_error(client: TestClient, monkeypatch):
    """Si hay un fallo de configuración de API key o modelo, debe responder 500."""
    mock_generate = AsyncMock(side_effect=LLMConfigurationError("API key inválida"))
    monkeypatch.setattr("app.routers.estimations.generate_estimation", mock_generate)

    response = client.post(
        "/api/v1/estimate",
        json={"transcription": "Transcripción válida con más de veinte caracteres para la prueba."},
    )
    assert response.status_code == 500
    assert "mal configurado" in response.json()["detail"].lower()


def test_estimate_rate_limit_error(client: TestClient, monkeypatch):
    """Si el proveedor da Rate Limit o falta saldo, debe responder 503."""
    fake_request = httpx.Request("POST", "https://api.openai.com")
    fake_response = httpx.Response(429, request=fake_request)
    mock_generate = AsyncMock(
        side_effect=openai.RateLimitError(
            message="Rate limit reached", response=fake_response, body=None
        )
    )
    monkeypatch.setattr("app.routers.estimations.generate_estimation", mock_generate)

    response = client.post(
        "/api/v1/estimate",
        json={"transcription": "Transcripción válida con más de veinte caracteres para la prueba."},
    )
    assert response.status_code == 503
    assert "no puede atender la petición" in response.json()["detail"].lower()


def test_estimate_provider_api_error(client: TestClient, monkeypatch):
    """Si el proveedor tiene un error 5xx u otro fallo de conexión, debe responder 502."""
    fake_request = httpx.Request("POST", "https://api.openai.com")
    mock_generate = AsyncMock(
        side_effect=openai.APIError(
            message="Internal server error", request=fake_request, body=None
        )
    )
    monkeypatch.setattr("app.routers.estimations.generate_estimation", mock_generate)

    response = client.post(
        "/api/v1/estimate",
        json={"transcription": "Transcripción válida con más de veinte caracteres para la prueba."},
    )
    assert response.status_code == 502
    assert "no se pudo obtener respuesta" in response.json()["detail"].lower()
