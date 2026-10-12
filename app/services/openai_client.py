"""Cliente de OpenAI compartido para funciones que son específicas de esa
API más allá del wrapper de estimación (moderación, embeddings) -- no
participa en las llamadas al LLM para generar estimaciones, esas van por
el Router de LiteLLM.

Devuelve None si no hay OPENAI_API_KEY configurada (ej. proveedor primario
es Anthropic sin key de OpenAI): tanto el guardrail de moderación como la
cache semántica están preparados para degradarse solos en ese caso.
"""

from openai import AsyncOpenAI

from app.config import settings

_client: AsyncOpenAI | None = None
_attempted = False


def get_openai_client() -> AsyncOpenAI | None:
    global _client, _attempted
    if not _attempted:
        _attempted = True
        if settings.openai_api_key:
            _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client
