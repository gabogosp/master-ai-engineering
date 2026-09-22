import logging

from fastapi import FastAPI

from app.config import settings
from app.routers import estimations

# Solo nuestro código usa LOG_LEVEL. Las librerías se quedan en WARNING porque,
# en DEBUG, pueden imprimir mucho, incluido el contenido de las peticiones.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("app").setLevel(settings.log_level.upper())

app = FastAPI(
    title="Estimador CAG",
    description=(
        "API que genera estimaciones de software a partir de la transcripción de una "
        "reunión con un cliente. Usa arquitectura CAG: ejemplos de estimaciones previas "
        "se inyectan directamente en el prompt del LLM."
    ),
    version="0.1.0",
)

app.include_router(estimations.router, prefix="/api/v1")


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {
        "status": "ok",
        "environment": settings.app_env,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
    }
