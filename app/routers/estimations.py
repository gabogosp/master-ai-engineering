from datetime import datetime, timezone
from typing import Annotated

import structlog
from fastapi import APIRouter, HTTPException, Query

from app.guardrails.input import InputGuardrailViolation, check_input
from app.guardrails.output import enforce_scope_response
from app.prompts.loader import render_estimation_prompt
from app.schemas import EstimationRequest, EstimationResponse, EstimationResult
from app.services.llm_service import (
    CONFIGURATION_ERRORS,
    PROVIDER_ERRORS,
    RATE_LIMIT_ERRORS,
    EstimationValidationError,
    generate_structured,
)
from app.services.openai_client import get_openai_client
from app.services.semantic_cache import get_semantic_cache

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
        await check_input(request.description, openai_client=get_openai_client())
    except InputGuardrailViolation as exc:
        logger.info("estimation_blocked_by_input_guardrail", reason=exc.reason)
        raise HTTPException(
            status_code=400, detail={"reason": exc.reason, "message": exc.message}
        )

    # Cache semántica: un pedido con una descripción distinta pero
    # equivalente (mismo bucket prompt_version/project_type/detail_level/
    # output_format + similitud de embeddings >= threshold) se sirve sin
    # llamar al LLM. Se chequea ANTES de renderizar el prompt porque solo
    # necesita los campos crudos del request, no el texto ya renderizado.
    semantic_cache = get_semantic_cache()
    if semantic_cache is not None:
        semantic_hit = await semantic_cache.lookup(request, prompt_version)
        if semantic_hit is not None:
            return EstimationResponse(
                result=semantic_hit.result,
                prompt_version=prompt_version,
                model=semantic_hit.model,
                provider=semantic_hit.provider,
                input_tokens=semantic_hit.input_tokens,
                output_tokens=semantic_hit.output_tokens,
                created_at=datetime.now(timezone.utc),
                cached=True,
            )

    try:
        system, user = render_estimation_prompt(request, version=prompt_version)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # El detalle de cada fallo va al log. Al cliente solo le damos un mensaje genérico.
    try:
        structured = await generate_structured(system, user, EstimationResult)
    except EstimationValidationError as exc:
        logger.warning("estimation_validation_exhausted", error=str(exc)[:300])
        raise HTTPException(
            status_code=502,
            detail="El modelo no logró producir una estimación válida. Inténtalo de nuevo.",
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

    result = enforce_scope_response(structured.result)

    if semantic_cache is not None:
        await semantic_cache.store(
            request,
            prompt_version,
            result,
            model=structured.model,
            provider=structured.provider,
            input_tokens=structured.input_tokens,
            output_tokens=structured.output_tokens,
        )

    return EstimationResponse(
        result=result,
        prompt_version=prompt_version,
        model=structured.model,
        provider=structured.provider,
        input_tokens=structured.input_tokens,
        output_tokens=structured.output_tokens,
        created_at=datetime.now(timezone.utc),
        cached=structured.cached,
    )
