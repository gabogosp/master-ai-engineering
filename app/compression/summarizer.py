"""Summarizer acumulativo para los turnos evictados de la conversación.

Lo invoca `CompressionPolicy` cuando turnos no-ancla caen fuera de la
ventana deslizante. Toma el resumen corriente (puede estar vacío) más los
mensajes recién evictados, y los funde en un resumen nuevo. El resultado
REEMPLAZA al anterior -- hay un único resumen rolling por sesión, no una
cadena de resúmenes.

Es una llamada a LLM aparte, con un modelo barato por defecto. Si falla, se
conserva el resumen anterior intacto -- mejor perder un pase de compresión
que borrar estado que el modelo ya se había comprometido a recordar.
"""

import structlog
from pydantic import BaseModel, Field

from app.prompts.loader import render_conversation_summary_prompt
from app.services.llm_service import auxiliary_model_group, generate_structured

logger = structlog.get_logger(__name__)


class _SummaryEnvelope(BaseModel):
    """Envoltorio estructurado mínimo para que el LLM se comprometa con un
    solo campo string -- usamos Instructor por consistencia con el resto
    del stack en vez de texto libre."""

    summary: str = Field(min_length=1, max_length=4000)


class CumulativeSummarizer:
    async def summarize(self, *, previous_summary: str | None, evicted: list[dict]) -> str:
        """Devuelve el resumen acumulado actualizado.

        Ante cualquier error de LLM, loguea y devuelve `previous_summary or
        ""` para que la sesión siga funcionando -- la compresión es
        best-effort, la ventana deslizante ya garantiza que los turnos
        recientes están intactos."""
        if not evicted:
            return previous_summary or ""

        system_prompt, user_message = render_conversation_summary_prompt(
            previous_summary=previous_summary, evicted=evicted
        )

        try:
            structured = await generate_structured(
                system_prompt,
                user_message,
                _SummaryEnvelope,
                max_retries=1,
                model_group=auxiliary_model_group(),
            )
        except Exception as exc:
            logger.warning(
                "summarizer_failed", error_type=type(exc).__name__, error=str(exc)[:200]
            )
            return previous_summary or ""

        logger.info(
            "summarizer_completed",
            evicted_count=len(evicted),
            previous_chars=len(previous_summary or ""),
            new_chars=len(structured.result.summary),
            model=structured.model,
        )
        return structured.result.summary
