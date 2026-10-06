"""Guardrail de output: refuerzo sobre los model_validator de Pydantic.

Los validators a nivel de schema (en app/schemas.py) son la primera línea:
cuando fallan, Instructor reintenta contra el LLM hasta max_retries veces.
`enforce_scope_response` es un *filtro* (no una excepción): reescribe el
`summary` cuando el LLM produjo una respuesta de confianza baja sin el
prefijo "Out of scope:". En la práctica el validator ya debería haber
disparado antes de llegar acá -- este filtro cubre casos de borde
(confidence_pct == 30 justo, o si el umbral se afloja en el futuro).
"""

import structlog

from app.schemas import (
    LOW_CONFIDENCE_THRESHOLD,
    OUT_OF_SCOPE_PREFIX,
    EstimationResult,
    Phase,
)

logger = structlog.get_logger(__name__)

_NOT_ESTIMATED_PHASE = Phase(
    name="Sin estimar",
    duration_weeks=1,
    cost_eur=0,
    summary="No se puede dimensionar sin más información sobre alcance, integraciones y equipo.",
)


def enforce_scope_response(result: EstimationResult) -> EstimationResult:
    """Reescribe el resultado si la confianza es baja y el summary no lo
    declara. Política: filtro -- nunca lanza, siempre devuelve un
    EstimationResult bien formado. El usuario recibe un mensaje claro en
    vez de un error."""
    is_low_confidence = result.confidence_pct < LOW_CONFIDENCE_THRESHOLD
    already_marked = result.summary.startswith(OUT_OF_SCOPE_PREFIX)

    if not is_low_confidence or already_marked:
        return result

    logger.info(
        "enforce_scope_response_filtering",
        confidence_pct=result.confidence_pct,
        original_summary_chars=len(result.summary),
    )
    new_summary = (
        f"{OUT_OF_SCOPE_PREFIX} no hay información suficiente para estimar con confianza. "
        f"Razonamiento original del modelo: {result.summary[:400]}"
    )
    return EstimationResult(
        summary=new_summary[:1200],
        confidence_pct=result.confidence_pct,
        phases=[_NOT_ESTIMATED_PHASE],
        total_duration_weeks=1,
        total_cost_eur=0,
    )
