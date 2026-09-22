from pathlib import Path
from unittest.mock import MagicMock, patch
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "streamlit_app.py")


def test_streamlit_initial_render():
    """Verifica que la app de Streamlit renderice correctamente sus elementos iniciales."""
    at = AppTest.from_file(APP_PATH).run()

    assert not at.exception
    assert at.title[0].value == "📋 Estimador de Proyectos (CAG)"
    assert len(at.chat_input) == 1
    assert at.chat_input[0].placeholder == "Pega aquí la transcripción de la reunión..."
    assert len(at.sidebar) > 0


def test_streamlit_chat_streaming_interaction():
    """Simula una interacción de chat completa con streaming de OpenAI y verifica sesión y métricas."""
    # Mock de los eventos que emite OpenAI durante el streaming
    mock_event_1 = MagicMock()
    mock_event_1.type = "response.output_text.delta"
    mock_event_1.delta = "## Estimación: App Móvil\nTotal: 120 horas"

    mock_event_2 = MagicMock()
    mock_event_2.type = "response.completed"
    mock_event_2.response.model = "gpt-4o-mini-2024-07-18"
    mock_event_2.response.usage.input_tokens = 320
    mock_event_2.response.usage.output_tokens = 95

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.responses.create.return_value = [mock_event_1, mock_event_2]

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

        # Verificamos que las métricas se hayan capturado en el session_state
        metrics = at.session_state["last_metrics"]
        assert metrics is not None
        assert metrics["model"] == "gpt-4o-mini-2024-07-18"
        assert metrics["input_tokens"] == 320
        assert metrics["output_tokens"] == 95
        assert "elapsed_time" in metrics
