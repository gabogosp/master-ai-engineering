import os
import pytest
from fastapi.testclient import TestClient

# Establecemos claves dummy para el entorno de test
os.environ["OPENAI_API_KEY"] = "sk-mock-openai-test-key"
os.environ["ANTHROPIC_API_KEY"] = "sk-mock-anthropic-test-key"

from app.main import app


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
