"""EstimationService -- orquesta el turno conversacional (actor simple y
Actor-Critic-Boss), separado de las preocupaciones HTTP del router.

Nace como un refactor DELIBERADAMENTE incremental: por ahora cubre solo el
camino conversacional (POST /sessions/{id}/estimate y .../estimate-acb),
que es el que venimos tocando en cada bloque de la consigna de sesión 6.
El formulario de un solo turno (POST /api/v1/estimate,
app/routers/estimations.py) se queda como función suelta en
app.services.llm_service por ahora -- se migra cuando un bloque futuro lo
necesite, no antes, para no arriesgar código ya estable.

Los métodos de esta clase NO conocen FastAPI ni HTTPException: dejan
propagar las excepciones tal como las lanzan sus colaboradores
(InputGuardrailViolation, EstimationValidationError, los errores de
`openai`/`litellm`...). El router (app/routers/sessions.py) es el único
que sabe traducir eso a códigos HTTP -- ver `_map_llm_errors` ahí.

La clase no recibe dependencias por constructor (no hay todavía un
mecanismo de DI en el repo, tipo `Depends()` + `dependencies.py`): cada
colaborador (generate_structured_conversation_turn, Critic, Boss,
check_input...) ya resuelve sus propios singletons a nivel de módulo. Eso
significa que, para testear esta clase con fakes en vez de mockear esos
singletons, haría falta un segundo incremento de refactor -- no se hizo
acá a propósito, para no agrandar el cambio más de lo que pide esta tarea.
"""

from datetime import datetime, timezone

import structlog

from app.boss import Boss
from app.compression.policy import apply_compression
from app.config import settings
from app.critic import Critic
from app.guardrails.input import check_input
from app.guardrails.output import enforce_scope_response
from app.prompts.loader import render_session_prompt
from app.schemas import (
    ACBResponse,
    CriticFeedback,
    DetailLevel,
    EstimationResponse,
    EstimationResult,
    OutputFormat,
    ProjectType,
)
from app.services.llm_service import StructuredLLMResult, generate_structured_conversation_turn
from app.services.metadata_extraction import extract_and_merge_metadata
from app.services.openai_client import get_openai_client
from app.sessions import Session
from app.tier_resolver import Tier, resolve_tier

logger = structlog.get_logger(__name__)

PROMPT_VERSION = "v1"


class EstimationService:
    """Orquestador stateless del turno conversacional."""

    async def estimate_conversational(
        self,
        *,
        session: Session,
        full_transcript: str,
        attachments_total_chars: int,
        project_type: ProjectType | None = None,
        detail_level: DetailLevel | None = None,
        output_format: OutputFormat | None = None,
        tier_override: Tier | None = None,
    ) -> EstimationResponse:
        resolved_tier = await self._prepare(session, full_transcript, tier_override)

        system, user = render_session_prompt(
            full_transcript,
            session.metadata,
            version=PROMPT_VERSION,
            tier=resolved_tier,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
        )
        messages = session.history.to_messages_list(system)
        messages.append({"role": "user", "content": user})

        structured = await generate_structured_conversation_turn(messages, EstimationResult)
        result = enforce_scope_response(structured.result)

        assistant_turn = result.model_dump_json()
        await self._persist_turn(session, full_transcript, user, assistant_turn)
        self._log_turn_observed(session, full_transcript, attachments_total_chars, structured)

        return EstimationResponse(
            result=result,
            prompt_version=PROMPT_VERSION,
            model=structured.model,
            provider=structured.provider,
            input_tokens=structured.input_tokens,
            output_tokens=structured.output_tokens,
            created_at=datetime.now(timezone.utc),
            cached=structured.cached,
            project_metadata=session.metadata.model_dump(),
            history_turns=len(session.history),
        )

    async def estimate_with_acb(
        self,
        *,
        session: Session,
        full_transcript: str,
        attachments_total_chars: int,
        project_type: ProjectType | None = None,
        detail_level: DetailLevel | None = None,
        output_format: OutputFormat | None = None,
        tier_override: Tier | None = None,
    ) -> ACBResponse:
        resolved_tier = await self._prepare(session, full_transcript, tier_override)

        # Metadata (model/provider/tokens/latency/costo) de la ÚLTIMA
        # llamada real al actor -- el Boss solo devuelve el EstimationResult
        # final, así que la capturamos por closure para reportarla después.
        last_structured: StructuredLLMResult[EstimationResult] | None = None

        async def _actor(critic_feedback: CriticFeedback | None) -> EstimationResult:
            nonlocal last_structured
            system, user = render_session_prompt(
                full_transcript,
                session.metadata,
                version=PROMPT_VERSION,
                tier=resolved_tier,
                critic_feedback=critic_feedback,
                project_type=project_type,
                detail_level=detail_level,
                output_format=output_format,
            )
            messages = session.history.to_messages_list(system)
            messages.append({"role": "user", "content": user})
            structured = await generate_structured_conversation_turn(messages, EstimationResult)
            last_structured = structured
            return enforce_scope_response(structured.result)

        critic = Critic()

        async def _critic(draft: EstimationResult) -> CriticFeedback:
            return await critic.review(
                transcript=full_transcript, metadata=session.metadata, tier=resolved_tier, result=draft
            )

        boss = Boss(max_iterations=settings.boss_max_iterations)
        final_result, trace = await boss.run(actor=_actor, critic=_critic)
        assert last_structured is not None  # boss.run siempre llama al actor al menos una vez

        # El historial guarda un turno "limpio" (sin el feedback del Critic
        # tejido adentro) -- el ruido de la orquestación ACB no debe viajar
        # a turnos futuros, solo el resultado final.
        _, canonical_user = render_session_prompt(
            full_transcript, session.metadata, version=PROMPT_VERSION, tier=resolved_tier
        )
        assistant_turn = final_result.model_dump_json()
        await self._persist_turn(session, full_transcript, canonical_user, assistant_turn)
        self._log_turn_observed(session, full_transcript, attachments_total_chars, last_structured)

        return ACBResponse(
            result=final_result,
            prompt_version=PROMPT_VERSION,
            model=last_structured.model,
            provider=last_structured.provider,
            input_tokens=last_structured.input_tokens,
            output_tokens=last_structured.output_tokens,
            created_at=datetime.now(timezone.utc),
            cached=last_structured.cached,
            project_metadata=session.metadata.model_dump(),
            history_turns=len(session.history),
            acb=trace,
        )

    # -- pasos compartidos entre los dos métodos públicos ------------------

    @staticmethod
    async def _prepare(session: Session, full_transcript: str, tier_override: Tier | None) -> Tier:
        """Guardrail de input + resolución de tier -- comunes a actor y ACB.

        Deja propagar `InputGuardrailViolation` tal cual; el router la
        traduce a 400."""
        await check_input(full_transcript, openai_client=get_openai_client())

        resolved_tier, tier_rule = resolve_tier(
            transcript=full_transcript, metadata=session.metadata, override=tier_override
        )
        session.last_resolved_tier = resolved_tier.value
        session.last_tier_rule = tier_rule
        return resolved_tier

    @staticmethod
    async def _persist_turn(session: Session, full_transcript: str, user: str, assistant_turn: str) -> None:
        """El historial y el project_metadata se actualizan recién si la
        llamada al LLM salió bien -- un turno fallido no debe ensuciar la
        sesión (por eso esto se llama DESPUÉS de tener ya el resultado
        final, nunca antes)."""
        session.history.add_turn(
            {"role": "user", "content": user}, {"role": "assistant", "content": assistant_turn}
        )
        # La compresión (promoción a ancla + resumen acumulativo) es un
        # paso explícito y posterior a add_turn -- ver app.compression.policy.
        await apply_compression(session.history, anchor_detection_mode=settings.anchor_detection_mode)
        session.metadata = await extract_and_merge_metadata(
            full_transcript, assistant_turn, session.metadata
        )

    @staticmethod
    def _log_turn_observed(
        session: Session,
        full_transcript: str,
        attachments_total_chars: int,
        structured: StructuredLLMResult,
    ) -> None:
        """Un único evento agregado por turno (Bloque 1 / Paso 1 de la
        consigna de sesión 6), en vez de los logs sueltos que ya emiten
        llm_service/compression/metadata_extraction a lo largo del camino:
        para extraer evals/stress/results.csv de una corrida de
        stress-testing, un evento = una fila es mucho más simple que
        reconciliar logs separados por timestamp y session_id.

        Compartido entre estimate_conversational y estimate_with_acb --
        antes de este refactor, ACB no emitía turn_observed porque la
        consigna original solo mencionaba el camino actor; al vivir acá
        los dos lo heredan gratis, sin tener que acordarnos de agregarlo
        a mano en cada lugar nuevo.

        cache_hit_kind nunca vale "semantic" -- esa cache solo está
        wireada en el endpoint de formulario de un solo turno, no en el
        flujo conversacional (ver sección "Cache semántica" del README).
        """
        session.turn_index += 1
        logger.info(
            "turn_observed",
            turn_index=session.turn_index,
            session_id=session.session_id,
            enriched_transcript_chars=len(full_transcript),
            attachments_total_chars=attachments_total_chars,
            messages_in_window=len(session.history.messages),
            anchors_count=len(session.history.anchors) // 2,
            summary_chars=len(session.history.summary or ""),
            tokens_in=structured.input_tokens,
            tokens_out=structured.output_tokens,
            cost_usd=structured.cost_usd,
            latency_ms=structured.latency_ms,
            cache_hit_kind="exact" if structured.cached else "none",
            last_resolved_tier=session.last_resolved_tier,
        )


estimation_service = EstimationService()
