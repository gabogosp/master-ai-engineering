import json
from unittest.mock import AsyncMock, patch

import instructor
import litellm
import pytest

import app.services.llm_service as svc
from app.schemas import EstimationResult


@pytest.fixture(autouse=True)
def _bypass_exact_match_cache():
    with patch.object(svc._cache, "get", AsyncMock(return_value=None)):
        yield


@pytest.fixture(autouse=True)
def _instructor_json_mode():
    """Mismo motivo que en test_sessions_integration.py: mock_response solo
    simula texto plano, no tool_calls -- forzamos Mode.JSON para que
    Instructor lo pueda parsear."""
    svc._instructor_client = None

    def _json_mode_client() -> instructor.AsyncInstructor:
        return instructor.from_litellm(svc._get_router().acompletion, mode=instructor.Mode.JSON)

    with patch.object(svc, "_get_instructor_client", side_effect=_json_mode_client):
        yield
    svc._instructor_client = None


def _estimation_result_json() -> str:
    return json.dumps(
        {
            "summary": "Resumen de prueba con longitud suficiente para validar el schema.",
            "confidence_pct": 80,
            "phases": [
                {
                    "name": "Implementación",
                    "duration_weeks": 3,
                    "cost_eur": 9000,
                    "summary": "Desarrollo del sistema de reservas y notificaciones por email.",
                }
            ],
            "total_duration_weeks": 3,
            "total_cost_eur": 9000,
        }
    )


async def test_generate_structured_returns_latency_and_cost_on_a_fresh_call():
    async def fake_acompletion(self, *args, **kwargs):
        return litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            mock_response=_estimation_result_json(),
        )

    with patch.object(svc.Router, "acompletion", new=fake_acompletion):
        result = await svc.generate_structured("system prompt", "user prompt", EstimationResult)

    assert result.cached is False
    assert result.latency_ms > 0
    assert result.cost_usd is not None and result.cost_usd >= 0


async def test_generate_structured_cache_hit_reports_zero_latency_and_cost():
    cached_entry = {
        "result": {
            "summary": "Resumen cacheado con longitud suficiente para validar.",
            "confidence_pct": 70,
            "phases": [
                {
                    "name": "Implementación",
                    "duration_weeks": 2,
                    "cost_eur": 5000,
                    "summary": "Desarrollo cacheado del sistema de reservas.",
                }
            ],
            "total_duration_weeks": 2,
            "total_cost_eur": 5000,
        },
        "model": "gpt-4o-mini-2024-07-18",
        "provider": "openai",
        "input_tokens": 100,
        "output_tokens": 50,
    }
    with patch.object(svc._cache, "get", AsyncMock(return_value=cached_entry)):
        result = await svc.generate_structured("system prompt", "user prompt", EstimationResult)

    assert result.cached is True
    assert result.latency_ms == 0.0
    assert result.cost_usd == 0.0
