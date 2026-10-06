from pathlib import Path
from unittest.mock import patch

import httpx
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "streamlit_app.py")

FAKE_RESPONSE_BODY = {
    "text": "## Estimación de prueba\nTotal: 42 horas",
    "prompt_version": "v1",
    "model": "gpt-4o-mini-mock",
    "provider": "openai",
    "input_tokens": 111,
    "output_tokens": 22,
    "created_at": "2026-09-27T00:00:00Z",
}

FAKE_SESSION_ID = "11111111-1111-1111-1111-111111111111"

FAKE_SESSION_SSE_BODY = (
    b'data: "## Estimaci\\u00f3n de prueba\\n"\n\n'
    b'data: "Total: 42 horas"\n\n'
    b"event: meta\n"
    b'data: {"prompt_version": "v1", "model": "gpt-4o-mini-mock", "provider": "openai", '
    b'"input_tokens": 111, "output_tokens": 22, '
    b'"project_metadata": {"project_name": "Turnos VetCare", "assumed_team_size": 2, '
    b'"mentioned_technologies": ["FastAPI"], "agreed_scope": null}, "history_turns": 1}\n\n'
)


def _fake_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/v1/sessions":
        return httpx.Response(200, json={"session_id": FAKE_SESSION_ID})
    if path == "/api/v1/estimate":
        return httpx.Response(200, json=FAKE_RESPONSE_BODY)
    if path == f"/api/v1/sessions/{FAKE_SESSION_ID}/estimate/stream":
        return httpx.Response(
            200, content=FAKE_SESSION_SSE_BODY, headers={"content-type": "text/event-stream"}
        )
    raise AssertionError(f"Llamada inesperada a {path}")


_RealClient = httpx.Client


def _mock_httpx_client(*args, **kwargs):
    """Reemplaza httpx.Client por uno con un MockTransport: intercepta la
    llamada real por red y devuelve respuestas fijas para /sessions,
    /estimate y /sessions/{id}/estimate, sin necesitar un servidor
    corriendo.

    Usa `_RealClient` (guardada antes del patch) para no recursar contra el
    propio mock.
    """
    return _RealClient(transport=httpx.MockTransport(_fake_handler))


def _find_button(at, label: str):
    return next(b for b in at.button if b.label == label)


def test_streamlit_initial_render():
    """Verifica que el formulario (ya no el chat) renderice sus elementos
    iniciales, incluida la pestaña de conversación (que crea una sesión al
    cargar la página)."""
    with patch("httpx.Client", side_effect=_mock_httpx_client):
        at = AppTest.from_file(APP_PATH).run()

        assert not at.exception
        assert at.title[0].value == "📋 Estimador de Proyectos"
        assert len(at.text_area) == 1
        assert len(at.selectbox) == 4  # tipo, detalle, formato, versión de prompt
        assert len(at.sidebar) > 0
        assert at.session_state["session_id"] == FAKE_SESSION_ID


def test_streamlit_form_submission_shows_full_response():
    """Simula completar el formulario y enviarlo: la respuesta llega de una
    sola vez (bloqueante, texto libre) y las métricas quedan en session_state."""
    with patch("httpx.Client", side_effect=_mock_httpx_client):
        at = AppTest.from_file(APP_PATH).run()

        at.text_area[0].set_value(
            "El cliente necesita un sistema de tickets de soporte con estados y comentarios internos."
        )
        _find_button(at, "Generar estimación").click().run()

        assert not at.exception

        full_text = "\n".join(md.value for md in at.markdown)
        assert "42 horas" in full_text

        metrics = at.session_state["last_metrics"]
        assert metrics["model"] == "gpt-4o-mini-mock"
        assert metrics["provider"] == "openai"
        assert metrics["prompt_version"] == "v1"
        assert metrics["input_tokens"] == 111
        assert metrics["output_tokens"] == 22
        assert "elapsed_time" in metrics


def test_streamlit_form_rejects_short_description():
    """El formulario debe validar la longitud mínima antes de llamar al
    backend (no se llega a pegarle a /api/v1/estimate)."""
    with patch("httpx.Client", side_effect=_mock_httpx_client):
        at = AppTest.from_file(APP_PATH).run()

        at.text_area[0].set_value("Muy corto")
        _find_button(at, "Generar estimación").click().run()

        assert not at.exception
        assert any("al menos 20 caracteres" in e.value for e in at.error)
        assert "last_metrics" not in at.session_state


def test_streamlit_conversation_turn_updates_metadata():
    """Simula un turno en la pestaña de conversación: la sesión ya se creó
    al cargar la página, el turno llega como chat_input, y el
    project_metadata + history_turns devueltos quedan en session_state."""
    with patch("httpx.Client", side_effect=_mock_httpx_client):
        at = AppTest.from_file(APP_PATH).run()

        at.chat_input[0].set_value(
            "El cliente es una clínica veterinaria y quiere un sistema de turnos."
        ).run()

        assert not at.exception

        messages = at.session_state["conversation_messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "42 horas" in messages[1]["content"]

        metadata = at.session_state["session_metadata"]
        assert metadata["project_name"] == "Turnos VetCare"
        assert metadata["mentioned_technologies"] == ["FastAPI"]
        assert at.session_state["session_history_turns"] == 1
