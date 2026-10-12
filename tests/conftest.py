import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Establecemos claves dummy para el entorno de test
os.environ["OPENAI_API_KEY"] = "sk-mock-openai-test-key"
os.environ["ANTHROPIC_API_KEY"] = "sk-mock-anthropic-test-key"

from app.main import app


@pytest.fixture(autouse=True)
def _disable_semantic_cache_by_default():
    """La cache semántica construye su vectorizer haciendo una llamada real
    a embeddings de OpenAI (para detectar la dimensión del modelo) -- con
    la API key dummy de test eso es una llamada de red que siempre falla
    (se degrada bien, pero cuesta ~decenas de segundos de retries/backoff).
    La deshabilitamos por default; los tests que sí quieren ejercitarla
    (tests/test_semantic_cache.py) la mockean explícitamente por su cuenta."""
    with patch("app.routers.estimations.get_semantic_cache", return_value=None):
        yield


@pytest.fixture
def client():
    """Cliente de pruebas para interactuar con la API FastAPI en memoria.

    Se usa como context manager (no un TestClient suelto) para que todas las
    llamadas dentro de un mismo test compartan un único event loop. Si no,
    cualquier código async con estado persistente entre llamadas (como nuestro
    cliente de Redis) puede terminar atado a un loop ya cerrado.
    """
    with TestClient(app) as test_client:
        yield test_client
