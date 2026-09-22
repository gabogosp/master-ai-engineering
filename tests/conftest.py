import os
import pytest
from fastapi.testclient import TestClient

# Establecemos claves dummy para el entorno de test
os.environ["OPENAI_API_KEY"] = "sk-mock-openai-test-key"
os.environ["ANTHROPIC_API_KEY"] = "sk-mock-anthropic-test-key"

from app.main import app
from app.config import settings


@pytest.fixture
def client() -> TestClient:
    """Cliente de pruebas para interactuar con la API FastAPI en memoria."""
    return TestClient(app)
