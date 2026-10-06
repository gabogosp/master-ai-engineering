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


def _fake_handler(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/api/v1/estimate"
    return httpx.Response(200, json=FAKE_RESPONSE_BODY)


_RealClient = httpx.Client


def _mock_httpx_client(*args, **kwargs):
    """Reemplaza httpx.Client por uno con un MockTransport: intercepta la
    llamada real por red y devuelve la misma respuesta JSON bloqueante que
    produce nuestro endpoint /estimate, sin necesitar un servidor corriendo.

    Usa `_RealClient` (guardada antes del patch) para no recursar contra el
    propio mock.
    """
    return _RealClient(transport=httpx.MockTransport(_fake_handler))


def test_streamlit_initial_render():
    """Verifica que el formulario (ya no el chat) renderice sus elementos iniciales."""
    at = AppTest.from_file(APP_PATH).run()

    assert not at.exception
    assert at.title[0].value == "📋 Estimador de Proyectos"
    assert len(at.text_area) == 1
    assert len(at.selectbox) == 4  # tipo, detalle, formato, versión de prompt
    assert len(at.sidebar) > 0


def test_streamlit_form_submission_shows_full_response():
    """Simula completar el formulario y enviarlo: la respuesta llega de una
    sola vez (bloqueante, texto libre) y las métricas quedan en session_state."""
    with patch("httpx.Client", side_effect=_mock_httpx_client):
        at = AppTest.from_file(APP_PATH).run()

        at.text_area[0].set_value(
            "El cliente necesita un sistema de tickets de soporte con estados y comentarios internos."
        )
        at.button[0].click().run()

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
    """El formulario debe validar la longitud mínima antes de llamar al backend."""
    with patch("httpx.Client", side_effect=_mock_httpx_client) as mock_client:
        at = AppTest.from_file(APP_PATH).run()

        at.text_area[0].set_value("Muy corto")
        at.button[0].click().run()

        assert not at.exception
        assert any("al menos 20 caracteres" in e.value for e in at.error)
        mock_client.assert_not_called()
