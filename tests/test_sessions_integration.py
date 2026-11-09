import json
from collections.abc import Callable
from unittest.mock import AsyncMock, patch

import instructor
import litellm
import pytest
from httpx import ASGITransport, AsyncClient
from structlog.testing import capture_logs

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
# Marcador literal del system.j2 del summarizer acumulativo -- permite, en
# el mock, distinguir una llamada de compresión de historial de una
# llamada de estimación real, igual que METADATA_SYSTEM_MARKER.
SUMMARIZER_SYSTEM_MARKER = "summarizer de conversaciones"
# Idem para el Critic del flujo Actor-Critic-Boss.
CRITIC_SYSTEM_MARKER = "independent auditor of software estimations"


def _critic_feedback_json(verdict: str = "accept") -> str:
    import json as _json

    if verdict == "accept":
        return _json.dumps({"verdict": "accept", "issues": [], "confidence_in_review": 90})
    return _json.dumps(
        {
            "verdict": verdict,
            "issues": [
                {
                    "category": "math_error",
                    "severity": "critical",
                    "field_path": "total_cost_eur",
                    "description": "La suma de fases no coincide con el total declarado.",
                }
            ],
            "confidence_in_review": 85,
        }
    )


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
    (system + [resumen, una vez que la compresión lo generó] + los últimos
    MAX_TURNS turnos + el turno nuevo), aunque la sesión ya tenga más
    historial acumulado que eso -- los turnos viejos se pliegan en el
    resumen acumulativo en vez de mandarse crudos."""
    captured_message_counts: list[int] = []

    async def fake_acompletion(self, *args, **kwargs):
        messages = kwargs["messages"]
        system_content = messages[0]["content"]
        if METADATA_SYSTEM_MARKER in system_content:
            mock_response = (
                '{"project_name": null, "assumed_team_size": null, '
                '"mentioned_technologies": [], "agreed_scope": null}'
            )
        elif SUMMARIZER_SYSTEM_MARKER in system_content:
            # No cuenta para la aserción de la ventana -- es la llamada de
            # compresión, no una estimación real.
            mock_response = _estimation_result_json()
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

    # system (1) + MAX_TURNS turnos previos (2 c/u) + el turno nuevo (1).
    # El resumen (una vez que la compresión lo genera) no agrega un mensaje
    # aparte: `to_messages_list` lo funde en el primer mensaje `user` para
    # no romper la alternancia estricta user/assistant -- así que el techo
    # no cambia aunque haya resumen acumulado.
    max_expected = 1 + settings.max_conversation_turns * 2 + 1
    assert max(captured_message_counts) <= max_expected
    # Y con 8 turnos (> MAX_TURNS=6 por defecto) el techo sí se alcanzó --
    # confirma que la ventana realmente recorta, no que nunca creció tanto.
    assert captured_message_counts[-1] == max_expected


async def test_debug_endpoint_reflects_compression_and_tier():
    """GET /sessions/{id} debe exponer lo que produjo la ventana
    comprimida (mensajes activos, anclas, caracteres de resumen) y el
    último tier resuelto, sin tener que re-correr nada."""

    def script(messages: list[dict]) -> str:
        system_content = messages[0]["content"]
        if METADATA_SYSTEM_MARKER in system_content:
            return (
                '{"project_name": null, "assumed_team_size": null, '
                '"mentioned_technologies": [], "agreed_scope": null}'
            )
        if SUMMARIZER_SYSTEM_MARKER in system_content:
            return _estimation_result_json()
        return _estimation_result_json()

    fake = _scripted_acompletion(script)
    with patch.object(svc.Router, "acompletion", new=fake):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            session_resp = await client.post("/api/v1/sessions")
            session_id = session_resp.json()["session_id"]

            info = await client.get(f"/api/v1/sessions/{session_id}")
            assert info.status_code == 200
            data = info.json()
            assert data["message_count"] == 0
            assert data["max_turns"] == settings.max_conversation_turns
            assert data["anchors_count"] == 0
            assert data["summary_chars"] == 0
            assert data["last_resolved_tier"] is None

            await client.post(
                f"/api/v1/sessions/{session_id}/estimate",
                data={"transcript": "Firmamos un NDA antes de compartir el alcance."},
            )

            info = await client.get(f"/api/v1/sessions/{session_id}")
            data = info.json()
            assert data["message_count"] == 1
            assert data["last_resolved_tier"] == "executive"
            assert data["last_tier_rule"] == "nda_detected"


async def test_debug_endpoint_404_for_unknown_session():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/sessions/does-not-exist")
    assert response.status_code == 404


async def test_acb_endpoint_accepts_on_first_pass():
    """Critic acepta en la primera pasada: el Boss no debe pedir una
    segunda iteración, y la traza debe reflejar exactamente 1 ronda."""

    def script(messages: list[dict]) -> str:
        system_content = messages[0]["content"]
        if METADATA_SYSTEM_MARKER in system_content:
            return (
                '{"project_name": null, "assumed_team_size": null, '
                '"mentioned_technologies": [], "agreed_scope": null}'
            )
        if CRITIC_SYSTEM_MARKER in system_content:
            return _critic_feedback_json("accept")
        return _estimation_result_json()

    fake = _scripted_acompletion(script)
    with patch.object(svc.Router, "acompletion", new=fake):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            session_resp = await client.post("/api/v1/sessions")
            session_id = session_resp.json()["session_id"]

            response = await client.post(
                f"/api/v1/sessions/{session_id}/estimate-acb",
                data={"transcript": "El cliente es una clínica veterinaria."},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["acb"]["final_decision"] == "accept"
    assert data["acb"]["iterations_run"] == 1
    assert data["result"]["summary"] == "Resumen de la estimación de prueba."
    assert data["history_turns"] == 1


async def test_acb_endpoint_iterates_then_accepts():
    """Critic pide una iteración, el segundo borrador (distinto texto) sí
    es aceptado -- confirma que el feedback viaja de verdad al segundo
    llamado del actor (vía el bloque <critic_feedback>)."""
    actor_call_count = 0

    def script(messages: list[dict]) -> str:
        nonlocal actor_call_count
        system_content = messages[0]["content"]
        if METADATA_SYSTEM_MARKER in system_content:
            return (
                '{"project_name": null, "assumed_team_size": null, '
                '"mentioned_technologies": [], "agreed_scope": null}'
            )
        if CRITIC_SYSTEM_MARKER in system_content:
            # Primera revisión: pide iteración. Segunda: acepta.
            user_content = messages[-1]["content"]
            if "corregido" in user_content:
                return _critic_feedback_json("accept")
            return _critic_feedback_json("needs_iteration")
        actor_call_count += 1
        if actor_call_count == 1:
            return _estimation_result_json(summary="Primer borrador con un error de aritmética.")
        # El segundo llamado del actor debe recibir el feedback del critic
        # tejido en el user message (ver <critic_feedback> en session_user.j2).
        assert any("math_error" in m["content"] for m in messages)
        return _estimation_result_json(summary="Segundo borrador, ya corregido.")

    fake = _scripted_acompletion(script)
    with patch.object(svc.Router, "acompletion", new=fake):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            session_resp = await client.post("/api/v1/sessions")
            session_id = session_resp.json()["session_id"]

            response = await client.post(
                f"/api/v1/sessions/{session_id}/estimate-acb",
                data={"transcript": "El cliente es una clínica veterinaria."},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["acb"]["final_decision"] == "accept"
    assert data["acb"]["iterations_run"] == 2
    assert data["result"]["summary"] == "Segundo borrador, ya corregido."
    # El historial guarda el turno "limpio" -- sin el ruido del
    # critic_feedback tejido en el prompt intermedio.
    assert data["history_turns"] == 1


async def test_turn_observed_event_carries_all_13_fields_and_turn_index_increments():
    """Un único evento turn_observed con los 13 campos, emitido justo
    antes del return de /estimate -- no logs sueltos que haya que
    reconciliar por timestamp."""

    def script(messages: list[dict]) -> str:
        system_content = messages[0]["content"]
        if METADATA_SYSTEM_MARKER in system_content:
            return (
                '{"project_name": null, "assumed_team_size": null, '
                '"mentioned_technologies": [], "agreed_scope": null}'
            )
        return _estimation_result_json()

    fake = _scripted_acompletion(script)
    with patch.object(svc.Router, "acompletion", new=fake):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            session_resp = await client.post("/api/v1/sessions")
            session_id = session_resp.json()["session_id"]

            with capture_logs() as logs:
                r1 = await client.post(
                    f"/api/v1/sessions/{session_id}/estimate",
                    data={"transcript": "El cliente es una clínica veterinaria."},
                )
            assert r1.status_code == 200

            with capture_logs() as logs2:
                r2 = await client.post(
                    f"/api/v1/sessions/{session_id}/estimate",
                    data={"transcript": "También quieren un panel para el personal."},
                )
            assert r2.status_code == 200

    def _find_turn_observed(captured):
        matches = [e for e in captured if e.get("event") == "turn_observed"]
        assert len(matches) == 1, f"se esperaba exactamente 1 turn_observed, hubo {len(matches)}"
        return matches[0]

    event1 = _find_turn_observed(logs)
    event2 = _find_turn_observed(logs2)

    expected_fields = {
        "turn_index",
        "session_id",
        "enriched_transcript_chars",
        "attachments_total_chars",
        "messages_in_window",
        "anchors_count",
        "summary_chars",
        "tokens_in",
        "tokens_out",
        "cost_usd",
        "latency_ms",
        "cache_hit_kind",
        "last_resolved_tier",
    }
    assert expected_fields.issubset(event1.keys())

    # Monótono: nunca baja, aunque la ventana comprima en sesiones largas.
    assert event1["turn_index"] == 1
    assert event2["turn_index"] == 2
    assert event2["messages_in_window"] == 4  # 2 pares tras el segundo turno
    assert event1["cache_hit_kind"] == "none"  # nunca "semantic" en el flujo conversacional


async def test_turn_observed_fires_for_acb_too_and_shares_the_turn_counter():
    """EstimationService._log_turn_observed es compartido entre
    estimate_conversational y estimate_with_acb -- antes del refactor a
    EstimationService, el endpoint ACB no emitía turn_observed porque la
    lógica vivía duplicada en cada router y solo se había agregado para
    el camino actor. Este test fija que ahora sí, y que turn_index es un
    contador ÚNICO de la sesión (no uno por endpoint)."""

    def script(messages: list[dict]) -> str:
        system_content = messages[0]["content"]
        if METADATA_SYSTEM_MARKER in system_content:
            return (
                '{"project_name": null, "assumed_team_size": null, '
                '"mentioned_technologies": [], "agreed_scope": null}'
            )
        if CRITIC_SYSTEM_MARKER in system_content:
            return _critic_feedback_json("accept")
        return _estimation_result_json()

    fake = _scripted_acompletion(script)
    with patch.object(svc.Router, "acompletion", new=fake):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            session_resp = await client.post("/api/v1/sessions")
            session_id = session_resp.json()["session_id"]

            with capture_logs() as logs_actor:
                r1 = await client.post(
                    f"/api/v1/sessions/{session_id}/estimate",
                    data={"transcript": "El cliente es una clínica veterinaria."},
                )
            assert r1.status_code == 200

            with capture_logs() as logs_acb:
                r2 = await client.post(
                    f"/api/v1/sessions/{session_id}/estimate-acb",
                    data={"transcript": "También quieren un panel para el personal."},
                )
            assert r2.status_code == 200

    actor_events = [e for e in logs_actor if e.get("event") == "turn_observed"]
    acb_events = [e for e in logs_acb if e.get("event") == "turn_observed"]
    assert len(actor_events) == 1
    assert len(acb_events) == 1
    assert actor_events[0]["turn_index"] == 1
    assert acb_events[0]["turn_index"] == 2  # mismo contador, no reinicia por endpoint
