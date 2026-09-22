import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.llm_service import (
    CONFIGURATION_ERRORS,
    PROVIDER_ERRORS,
    RATE_LIMIT_ERRORS,
    IncompleteEstimationError,
    generate_estimation,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["estimations"])


class EstimationRequest(BaseModel):
    transcription: str = Field(
        min_length=20,
        max_length=100_000,
        description="Texto de la transcripción de la reunión con el cliente",
    )


class EstimationResponse(BaseModel):
    estimation: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    created_at: datetime


@router.post("/estimate", response_model=EstimationResponse)
async def estimate(request: EstimationRequest) -> EstimationResponse:
    # El detalle de cada fallo va al log. Al cliente solo le damos un mensaje genérico.
    try:
        result = await generate_estimation(request.transcription)
    except IncompleteEstimationError as exc:
        logger.warning("Estimación incompleta: %s", exc)
        raise HTTPException(
            status_code=502, detail="La estimación quedó incompleta. Inténtalo de nuevo."
        )
    except CONFIGURATION_ERRORS:
        logger.exception("Configuración inválida del servicio de LLM (clave o modelo)")
        raise HTTPException(status_code=500, detail="El servicio de LLM está mal configurado.")
    except RATE_LIMIT_ERRORS:
        logger.warning("Límite de uso o falta de saldo en el proveedor de LLM")
        raise HTTPException(
            status_code=503,
            detail="El proveedor de LLM no puede atender la petición ahora. Inténtalo más tarde.",
        )
    except PROVIDER_ERRORS:
        logger.exception("Falló la llamada al proveedor de LLM")
        raise HTTPException(
            status_code=502, detail="No se pudo obtener respuesta del proveedor de LLM."
        )

    return EstimationResponse(
        estimation=result.estimation,
        model=result.model,
        provider=result.provider,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        created_at=datetime.now(timezone.utc),
    )
