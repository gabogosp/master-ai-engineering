from collections.abc import AsyncIterable
from datetime import datetime, timezone
from typing import Annotated

import structlog
from fastapi import APIRouter, HTTPException, Query
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.prompts.loader import render_estimation_prompt
from app.schemas import EstimationRequest, EstimationResponse
from app.services.llm_service import (
    CONFIGURATION_ERRORS,
    PROVIDER_ERRORS,
    RATE_LIMIT_ERRORS,
    IncompleteEstimationError,
    LLMResult,
    generate_estimation,
    stream_estimation,
)

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["estimations"])

DEFAULT_PROMPT_VERSION = "v1"
PromptVersionParam = Annotated[
    str, Query(description="Versión de prompt a usar, ej. 'v1' o 'v2'.")
]


@router.post("/estimate", response_model=EstimationResponse)
async def estimate(
    request: EstimationRequest,
    prompt_version: PromptVersionParam = DEFAULT_PROMPT_VERSION,
) -> EstimationResponse:
    try:
        system, user = render_estimation_prompt(request, version=prompt_version)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # El detalle de cada fallo va al log. Al cliente solo le damos un mensaje genérico.
    try:
        result = await generate_estimation(system, user)
    except IncompleteEstimationError as exc:
        logger.warning("estimation_incomplete", error=str(exc))
        raise HTTPException(
            status_code=502, detail="La estimación quedó incompleta. Inténtalo de nuevo."
        )
    except CONFIGURATION_ERRORS as exc:
        logger.error(
            "llm_configuration_error",
            error_type=type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="El servicio de LLM está mal configurado.")
    except RATE_LIMIT_ERRORS as exc:
        logger.warning("llm_rate_limited", error_type=type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="El proveedor de LLM no puede atender la petición ahora. Inténtalo más tarde.",
        )
    except PROVIDER_ERRORS as exc:
        logger.error(
            "llm_provider_error",
            error_type=type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=502, detail="No se pudo obtener respuesta del proveedor de LLM."
        )

    return EstimationResponse(
        text=result.estimation,
        prompt_version=prompt_version,
        model=result.model,
        provider=result.provider,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        created_at=datetime.now(timezone.utc),
    )


@router.post("/estimate/stream", response_class=EventSourceResponse)
async def estimate_stream(
    request: EstimationRequest,
    prompt_version: PromptVersionParam = DEFAULT_PROMPT_VERSION,
) -> AsyncIterable[ServerSentEvent]:
    """Igual que /estimate pero via Server-Sent Events, para clientes que
    consuman streaming (incluido el formulario de Streamlit, que le pega a
    este endpoint por HTTP en vez de importar el wrapper directo).

    Una vez que el primer chunk sale, la respuesta ya está comprometida a
    200 + text/event-stream: un fallo a mitad de camino no puede convertirse
    en un 500/502/503 HTTP. Por eso los errores (incluida una `prompt_version`
    inexistente) se comunican como un evento `error` dentro del stream, con
    el mismo mensaje genérico que usa /estimate; el detalle completo sigue
    yendo solo al log.
    """
    try:
        system, user = render_estimation_prompt(request, version=prompt_version)
    except ValueError as exc:
        logger.warning("invalid_prompt_version", prompt_version=prompt_version, error=str(exc))
        yield ServerSentEvent(event="error", data=f"Versión de prompt inválida: {prompt_version}")
        return

    metadata: dict = {"prompt_version": prompt_version}

    def _on_complete(result: LLMResult) -> None:
        metadata.update(
            model=result.model,
            provider=result.provider,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    try:
        async for chunk in stream_estimation(system, user, on_complete=_on_complete):
            yield ServerSentEvent(data=chunk)
    except IncompleteEstimationError as exc:
        logger.warning("estimation_incomplete", error=str(exc), streaming=True)
        yield ServerSentEvent(
            event="error", data="La estimación quedó incompleta. Inténtalo de nuevo."
        )
        return
    except CONFIGURATION_ERRORS as exc:
        logger.error(
            "llm_configuration_error",
            error_type=type(exc).__name__,
            streaming=True,
            exc_info=True,
        )
        yield ServerSentEvent(event="error", data="El servicio de LLM está mal configurado.")
        return
    except RATE_LIMIT_ERRORS as exc:
        logger.warning("llm_rate_limited", error_type=type(exc).__name__, streaming=True)
        yield ServerSentEvent(
            event="error",
            data="El proveedor de LLM no puede atender la petición ahora. Inténtalo más tarde.",
        )
        return
    except PROVIDER_ERRORS as exc:
        logger.error(
            "llm_provider_error",
            error_type=type(exc).__name__,
            streaming=True,
            exc_info=True,
        )
        yield ServerSentEvent(
            event="error", data="No se pudo obtener respuesta del proveedor de LLM."
        )
        return

    yield ServerSentEvent(event="meta", data=metadata)
