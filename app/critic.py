"""Critic -- auditoría independiente de una EstimationResult.

El Critic es una función stateless: (transcript, metadata, tier,
estimación) -> CriticFeedback. NO muta la sesión y NO decide qué hacer con
su propio output -- eso es trabajo del Boss. Mantener las responsabilidades
separadas es lo que hace auditable al patrón Actor-Critic-Boss.

Una falla dentro del Critic NO bloquea el pipeline. El Boss recibe un
veredicto sintético "accept con confianza cero" en ese caso, para que la
estimación del actor siga su curso sin modificar -- degradación elegante en
vez de un error duro frente al usuario.
"""

import structlog

from app.prompts.loader import render_critic_prompt
from app.schemas import CriticFeedback, EstimationResult
from app.services.llm_service import auxiliary_model_group, generate_structured
from app.sessions import ProjectMetadata
from app.tier_resolver import Tier

logger = structlog.get_logger(__name__)


class Critic:
    async def review(
        self,
        *,
        transcript: str,
        metadata: ProjectMetadata,
        tier: Tier,
        result: EstimationResult,
    ) -> CriticFeedback:
        system_prompt, user_message = render_critic_prompt(
            transcript=transcript, metadata=metadata, tier=tier, result=result
        )
        try:
            structured = await generate_structured(
                system_prompt,
                user_message,
                CriticFeedback,
                max_retries=3,
                model_group=auxiliary_model_group(),
            )
        except Exception as exc:
            logger.warning(
                "critic_failed_fallback_accept", error_type=type(exc).__name__, error=str(exc)[:200]
            )
            # Degradado: fingimos que el critic aceptó con confianza cero,
            # para que el Boss devuelva la salida del actor sin modificar.
            return CriticFeedback(verdict="accept", issues=[], confidence_in_review=0)

        feedback = structured.result
        logger.info(
            "critic_completed",
            verdict=feedback.verdict,
            issue_count=len(feedback.issues),
            confidence=feedback.confidence_in_review,
            model=structured.model,
        )
        return feedback
