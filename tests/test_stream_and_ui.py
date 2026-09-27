from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "streamlit_app.py")


async def fake_stream_estimation(transcription: str, on_complete=None):
    """Reemplaza app.services.llm_service.stream_estimation: yieldea texto en
    trozos y al final invoca on_complete, igual que haría el wrapper real."""
    from app.services.llm_service import LLMResult

    for piece in ["## Estimación: App Móvil\n", "Total: 120 horas"]:
        yield piece

    if on_complete:
        on_complete(
            LLMResult(
                estimation="## Estimación: App Móvil\nTotal: 120 horas",
                model="gpt-4o-mini-mock",
                provider="openai",
                input_tokens=320,
                output_tokens=95,
            )
        )


def test_streamlit_initial_render():
    """Verifica que la app de Streamlit renderice correctamente sus elementos iniciales."""
    at = AppTest.from_file(APP_PATH).run()

    assert not at.exception
    assert at.title[0].value == "📋 Estimador de Proyectos (CAG)"
    assert len(at.chat_input) == 1
    assert at.chat_input[0].placeholder == "Pega aquí la transcripción de la reunión..."
    assert len(at.sidebar) > 0


def test_streamlit_chat_streaming_interaction():
    """Simula una interacción de chat completa con streaming a través del wrapper
    (app.services.llm_service.stream_estimation) y verifica sesión y métricas."""
    with patch("app.services.llm_service.stream_estimation", fake_stream_estimation):
        at = AppTest.from_file(APP_PATH).run()

        # Simulamos que el usuario envía una transcripción
        at.chat_input[0].set_value(
            "El cliente necesita una app móvil para iOS y Android de gestión de pedidos."
        ).run()

        # Verificamos que no haya excepciones
        assert not at.exception

        # Verificamos que se hayan renderizado 2 mensajes (usuario y asistente)
        assert len(at.chat_message) == 2

        # Verificamos que la sesión mantenga los mensajes
        messages = at.session_state["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "120 horas" in messages[1]["content"]

        # Verificamos que las métricas se hayan capturado en el session_state,
        # incluido el proveedor (nuevo desde que el wrapper resuelve fallback)
        metrics = at.session_state["last_metrics"]
        assert metrics is not None
        assert metrics["model"] == "gpt-4o-mini-mock"
        assert metrics["provider"] == "openai"
        assert metrics["input_tokens"] == 320
        assert metrics["output_tokens"] == 95
        assert "elapsed_time" in metrics
