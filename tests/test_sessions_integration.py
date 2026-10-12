import json
from collections.abc import Callable
from unittest.mock import AsyncMock, patch

import instructor
import litellm
import pytest
from httpx import ASGITransport, AsyncClient

import app.services.llm_service as svc
from app.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def _bypass_exact_match_cache():
    """Estos tests verifican qué llega al LLM (mensajes, contenido de
    adjuntos), no el comportamiento del cache -- eso ya lo cubre
    test_cache.py. Sin esto, una corrida anterior con el mismo transcript
    puede dejar la respuesta en el Redis real y la siguiente corrida nunca
    llegaría a invocar al Router, que es justo lo que estos tests quieren
    observar."""
    with patch.object(svc._cache, "get", AsyncMock(return_value=None)):
        yield


@pytest.fixture(autouse=True)
def _instructor_json_mode_for_mocked_router():
    """El turno de sesión genera la estimación via Instructor (salida
    estructurada). El modo default de producción (Mode.TOOLS) espera
    tool_calls reales en la respuesta, pero litellm's `mock_response` solo
    puede simular texto plano -- así que para estos tests, que mockean
    Router.acompletion para inspeccionar los `messages` recibidos, forzamos
    Mode.JSON (parsea directo el texto como JSON). Ya se verificó aparte,
    contra la API real, que el Mode.TOOLS de producción funciona
    correctamente; este override es sólo para que el mock basado en texto
    sea parseable por Instructor."""
    svc._instructor_client = None

    def _json_mode_client() -> instructor.AsyncInstructor:
        return instructor.from_litellm(svc._get_router().acompletion, mode=instructor.Mode.JSON)

    with patch.object(svc, "_get_instructor_client", side_effect=_json_mode_client):
        yield
    svc._instructor_client = None


def _estimation_result_json(summary: str = "Resumen de la estimación de prueba.") -> str:
    return json.dumps(
        {
            "summary": summary,
            "confidence_pct": 80,
            "phases": [
                {
                    "name": "Implementación",
                    "duration_weeks": 4,
                    "cost_eur": 10_000,
                    "summary": "Desarrollo del backend y frontend principal del proyecto.",
                }
            ],
            "total_duration_weeks": 4,
            "total_cost_eur": 10_000,
        }
    )

PDF_WITH_BUDGET = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/MediaBox[0 0 300 300]/Contents 5 0 R>>endobj
4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
5 0 obj<</Length 46>>
stream
BT /F1 12 Tf 10 250 Td (Presupuesto previo: 200 horas) Tj ET
endstream
endobj
xref
0 6
0000000000 65535 f
trailer<</Size 6/Root 1 0 R>>
startxref
0
%%EOF"""

# Marcador literal del system.j2 de metadata_extraction -- nos permite, en
# el mock, distinguir una llamada de estimación de una de extracción sin
# acoplarnos a detalles internos del Router.
METADATA_SYSTEM_MARKER = "extract structured project metadata"


def _scripted_acompletion(script: Callable[[list[dict]], str]):
    """Fábrica de un fake Router.acompletion: `script(messages)` decide qué
    texto devuelve el LLM simulado según los mensajes que recibió."""

    async def _fake(self, *args, **kwargs):
        mock_response = script(kwargs["messages"])
        return litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            mock_response=mock_response,
            stream=kwargs.get("stream", False),
            stream_options=kwargs.get("stream_options"),
        )

    return _fake


async def test_project_metadata_updates_across_two_turns():
    """Dos peticiones en la misma sesión: el project_metadata extraído en
    el turno 1 debe seguir presente en la respuesta del turno 2, fusionado
    con lo nuevo que aporta ese segundo turno (no reemplazado ni perdido)."""

    def script(messages: list[dict]) -> str:
        system_content = messages[0]["content"]
        if METADATA_SYSTEM_MARKER in system_content:
            last_user = messages[-1]["content"]
            if "veterinaria" in last_user.lower():
                return (
                    '{"project_name": "Turnos VetCare", "assumed_team_size": null, '
                    '"mentioned_technologies": [], "agreed_scope": null}'
                )
            return (
                '{"project_name": null, "assumed_team_size": null, '
                '"mentioned_technologies": ["FastAPI"], "agreed_scope": null}'
            )
        return _estimation_result_json()

    fake = _scripted_acompletion(script)
    with patch.object(svc.Router, "acompletion", new=fake):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            session_resp = await client.post("/api/v1/sessions")
            assert session_resp.status_code == 200
            session_id = session_resp.json()["session_id"]

            turn1 = await client.post(
                f"/api/v1/sessions/{session_id}/estimate",
                data={"transcript": "El cliente es una clínica veterinaria."},
            )
            assert turn1.status_code == 200
            assert turn1.json()["project_metadata"]["project_name"] == "Turnos VetCare"

            turn2 = await client.post(
                f"/api/v1/sessions/{session_id}/estimate",
                data={"transcript": "Van a usar FastAPI para el backend."},
            )
            assert turn2.status_code == 200
            metadata2 = turn2.json()["project_metadata"]
            assert metadata2["project_name"] == "Turnos VetCare"  # se preservó
            assert metadata2["mentioned_technologies"] == ["FastAPI"]  # se agregó
            assert turn2.json()["history_turns"] == 2


async def test_attachment_content_reaches_the_llm():
    """Test cualitativo: el texto extraído del PDF adjunto debe llegar al
    LLM como parte del mensaje de usuario. Lo verificamos haciendo que el
    LLM simulado "eco'ee" un dato que solo existe en el PDF (no en el
    transcript) si lo ve en el mensaje que recibió -- así confirmamos que
    el pipeline extracción -> concatenación -> prompt realmente lo pasó."""

    def script(messages: list[dict]) -> str:
        system_content = messages[0]["content"]
        if METADATA_SYSTEM_MARKER in system_content:
            return (
                '{"project_name": null, "assumed_team_size": null, '
                '"mentioned_technologies": [], "agreed_scope": null}'
            )
        user_content = messages[-1]["content"]
        if "Presupuesto previo: 200 horas" in user_content:
            return _estimation_result_json(
                summary="Se detectó un presupuesto previo de 200 horas en el adjunto, "
                "que se usa como referencia para esta estimación."
            )
        return _estimation_result_json(summary="Sin información adicional en el adjunto.")

    fake = _scripted_acompletion(script)
    with patch.object(svc.Router, "acompletion", new=fake):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            session_resp = await client.post("/api/v1/sessions")
            session_id = session_resp.json()["session_id"]

            response = await client.post(
                f"/api/v1/sessions/{session_id}/estimate",
                data={"transcript": "Necesitamos un sistema de turnos."},
                files={"attachments": ("propuesta.pdf", PDF_WITH_BUDGET, "application/pdf")},
            )

            assert response.status_code == 200
            assert "200 horas" in response.json()["result"]["summary"]


async def test_sliding_window_caps_messages_sent_to_llm():
    """Manda 8 turnos a la misma sesión y confirma que el array `messages`
    que efectivamente llega al LLM nunca supera la ventana configurada
    (system + los últimos MAX_TURNS turnos + el turno nuevo), aunque la
    sesión ya tenga más historial acumulado que eso."""
    captured_message_counts: list[int] = []

    async def fake_acompletion(self, *args, **kwargs):
        messages = kwargs["messages"]
        if METADATA_SYSTEM_MARKER in messages[0]["content"]:
            mock_response = (
                '{"project_name": null, "assumed_team_size": null, '
                '"mentioned_technologies": [], "agreed_scope": null}'
            )
        else:
            captured_message_counts.append(len(messages))
            mock_response = _estimation_result_json()
        return litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            mock_response=mock_response,
        )

    with patch.object(svc.Router, "acompletion", new=fake_acompletion):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            session_resp = await client.post("/api/v1/sessions")
            session_id = session_resp.json()["session_id"]

            for i in range(8):
                response = await client.post(
                    f"/api/v1/sessions/{session_id}/estimate",
                    data={"transcript": f"Turno número {i}, agregando más contexto."},
                )
                assert response.status_code == 200

    # system (1) + MAX_TURNS turnos previos (2 c/u) + el turno nuevo (1)
    max_expected = 1 + settings.max_conversation_turns * 2 + 1
    assert max(captured_message_counts) <= max_expected
    # Y con 8 turnos (> MAX_TURNS=6 por defecto) el techo sí se alcanzó --
    # confirma que la ventana realmente recorta, no que nunca creció tanto.
    assert captured_message_counts[-1] == max_expected
