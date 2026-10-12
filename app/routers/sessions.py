from datetime import datetime, timezone
from typing import Annotated

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.attachments import (
    UnsupportedAttachmentError,
    build_transcript_with_attachments,
    extract_text,
)
from app.boss import Boss
from app.compression.policy import apply_compression
from app.config import settings
from app.critic import Critic
from app.guardrails.input import InputGuardrailViolation, check_input
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
    SessionCreatedResponse,
    SessionInfoResponse,
)
from app.services.llm_service import (
    CONFIGURATION_ERRORS,
    PROVIDER_ERRORS,
    RATE_LIMIT_ERRORS,
    EstimationValidationError,
    generate_structured_conversation_turn,
)
from app.services.metadata_extraction import extract_and_merge_metadata
from app.services.openai_client import get_openai_client
from app.sessions import Session, session_store
from app.tier_resolver import Tier, resolve_tier

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["sessions"])

PROMPT_VERSION = "v1"


async def _prepare_turn(
    session_id: str,
    transcript: str,
    attachments: list[UploadFile] | None,
    tier_override: Tier | None = None,
) -> tuple[Session, str, Tier, str]:
    """Preludio compartido entre /estimate y /estimate-acb: busca la
    sesión, extrae los adjuntos, corre el guardrail de input sobre el
    transcript ya enriquecido, y resuelve el tier de audiencia.

    `tier_override`, si viene, gana siempre sobre la cadena de reglas (ver
    app.tier_resolver.resolve_tier) -- lo usa, por ejemplo, el runner de
    evals para fijar el tier esperado de un caso dorado en vez de confiar
    en que la heurística lo derive sola.

    Devuelve (session, full_transcript, resolved_tier, tier_rule). Lanza
    HTTPException para los casos de sesión/adjunto -- el llamador maneja
    los errores de LLM por separado, que difieren entre los dos endpoints.
    """
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="La sesión no existe o expiró.")

    attachment_texts: list[tuple[str, str]] = []
    for upload in attachments or []:
        try:
            text = await extract_text(upload)
        except UnsupportedAttachmentError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        attachment_texts.append((upload.filename or "adjunto", text))

    full_transcript = build_transcript_with_attachments(transcript, attachment_texts)

    # Guardrail de input sobre el transcript YA CON los adjuntos concatenados:
    # un PDF/Word también puede traer PII o un intento de prompt-injection.
    try:
        await check_input(full_transcript, openai_client=get_openai_client())
    except InputGuardrailViolation as exc:
        logger.info(
            "estimation_blocked_by_input_guardrail", reason=exc.reason, session_id=session_id
        )
        raise HTTPException(
            status_code=400, detail={"reason": exc.reason, "message": exc.message}
        )

    # Tier de audiencia: override explícito si vino, si no se deriva del
    # transcript + la metadata acumulada. Se cachea en la sesión para que
    # GET /sessions/{id} lo pueda mostrar sin recalcularlo.
    resolved_tier, tier_rule = resolve_tier(
        transcript=full_transcript, metadata=session.metadata, override=tier_override
    )
    session.last_resolved_tier = resolved_tier.value
    session.last_tier_rule = tier_rule

    return session, full_transcript, resolved_tier, tier_rule


@router.post("/sessions", response_model=SessionCreatedResponse)
async def create_session() -> SessionCreatedResponse:
    session = session_store.create()
    return SessionCreatedResponse(session_id=session.session_id)


@router.get("/sessions/{session_id}", response_model=SessionInfoResponse)
async def get_session(session_id: str) -> SessionInfoResponse:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="La sesión no existe o expiró.")
    return SessionInfoResponse(
        session_id=session.session_id,
        message_count=len(session.history),
        max_turns=session.history.max_turns,
        metadata=session.metadata.model_dump(),
        anchors_count=len(session.history.anchors) // 2,
        summary_chars=len(session.history.summary or ""),
        last_resolved_tier=session.last_resolved_tier,
        last_tier_rule=session.last_tier_rule,
    )


@router.post("/sessions/{session_id}/estimate", response_model=EstimationResponse)
async def estimate_turn(
    session_id: str,
    transcript: Annotated[str, Form(min_length=1)],
    attachments: Annotated[list[UploadFile] | None, File()] = None,
    project_type: Annotated[ProjectType | None, Form()] = None,
    detail_level: Annotated[DetailLevel | None, Form()] = None,
    output_format: Annotated[OutputFormat | None, Form()] = None,
    tier: Annotated[Tier | None, Form()] = None,
) -> EstimationResponse:
    session, full_transcript, resolved_tier, _tier_rule = await _prepare_turn(
        session_id, transcript, attachments, tier_override=tier
    )

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

    # El detalle de cada fallo va al log. Al cliente solo le damos un mensaje genérico
    # (mismo mapeo de errores que /api/v1/estimate).
    try:
        structured = await generate_structured_conversation_turn(messages, EstimationResult)
    except EstimationValidationError as exc:
        logger.warning(
            "estimation_validation_exhausted", error=str(exc)[:300], session_id=session_id
        )
        raise HTTPException(
            status_code=502,
            detail="El modelo no logró producir una estimación válida. Inténtalo de nuevo.",
        )
    except CONFIGURATION_ERRORS as exc:
        logger.error(
            "llm_configuration_error",
            error_type=type(exc).__name__,
            session_id=session_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="El servicio de LLM está mal configurado.")
    except RATE_LIMIT_ERRORS as exc:
        logger.warning(
            "llm_rate_limited", error_type=type(exc).__name__, session_id=session_id
        )
        raise HTTPException(
            status_code=503,
            detail="El proveedor de LLM no puede atender la petición ahora. Inténtalo más tarde.",
        )
    except PROVIDER_ERRORS as exc:
        logger.error(
            "llm_provider_error",
            error_type=type(exc).__name__,
            session_id=session_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=502, detail="No se pudo obtener respuesta del proveedor de LLM."
        )

    result = enforce_scope_response(structured.result)

    # El historial y el project_metadata se actualizan recién si la llamada
    # al LLM salió bien -- un turno fallido no debe ensuciar la sesión.
    # Guardamos el JSON del resultado estructurado como el turno del
    # asistente (no un resumen en texto): así, en turnos futuros, el modelo
    # ve exactamente qué fases y totales propuso y puede revisarlos con
    # precisión en vez de tener que re-derivarlos de una prosa.
    assistant_turn = result.model_dump_json()
    session.history.add_turn(
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant_turn},
    )
    # La compresión (promoción a ancla + resumen acumulativo) es un paso
    # explícito y posterior a `add_turn` -- ver app.compression.policy.
    await apply_compression(
        session.history,
        anchor_detection_mode=settings.anchor_detection_mode,
    )
    session.metadata = await extract_and_merge_metadata(
        full_transcript, assistant_turn, session.metadata
    )

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


@router.post("/sessions/{session_id}/estimate-acb", response_model=ACBResponse)
async def estimate_turn_acb(
    session_id: str,
    transcript: Annotated[str, Form(min_length=1)],
    attachments: Annotated[list[UploadFile] | None, File()] = None,
    project_type: Annotated[ProjectType | None, Form()] = None,
    detail_level: Annotated[DetailLevel | None, Form()] = None,
    output_format: Annotated[OutputFormat | None, Form()] = None,
    tier: Annotated[Tier | None, Form()] = None,
) -> ACBResponse:
    """Variante Actor-Critic-Boss del turno conversacional: un auditor
    independiente (Critic) revisa cada borrador del actor, y el Boss
    decide si aceptarlo, pedir una nueva iteración con el feedback tejido
    en el prompt, o sintetizar un fallback anotado si el loop no converge.

    La sesión se actualiza SOLO con el resultado final (aprobado o
    sintetizado) -- los borradores intermedios del actor son descartables;
    desde el punto de vista del usuario, el turno produjo un único mensaje
    del asistente.
    """
    session, full_transcript, resolved_tier, _tier_rule = await _prepare_turn(
        session_id, transcript, attachments, tier_override=tier
    )

    # Metadata (model/provider/tokens) de la ÚLTIMA llamada real al actor --
    # el Boss solo devuelve el EstimationResult final, así que la capturamos
    # por closure para poder reportarla en la respuesta.
    last_structured = None

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
            transcript=full_transcript,
            metadata=session.metadata,
            tier=resolved_tier,
            result=draft,
        )

    # Mismo mapeo de errores que /estimate -- cualquier excepción de LLM
    # dentro del actor sube sin que el Boss la intercepte.
    try:
        boss = Boss(max_iterations=settings.boss_max_iterations)
        final_result, trace = await boss.run(actor=_actor, critic=_critic)
    except EstimationValidationError as exc:
        logger.warning(
            "estimation_validation_exhausted", error=str(exc)[:300], session_id=session_id
        )
        raise HTTPException(
            status_code=502,
            detail="El modelo no logró producir una estimación válida. Inténtalo de nuevo.",
        )
    except CONFIGURATION_ERRORS as exc:
        logger.error(
            "llm_configuration_error",
            error_type=type(exc).__name__,
            session_id=session_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="El servicio de LLM está mal configurado.")
    except RATE_LIMIT_ERRORS as exc:
        logger.warning(
            "llm_rate_limited", error_type=type(exc).__name__, session_id=session_id
        )
        raise HTTPException(
            status_code=503,
            detail="El proveedor de LLM no puede atender la petición ahora. Inténtalo más tarde.",
        )
    except PROVIDER_ERRORS as exc:
        logger.error(
            "llm_provider_error",
            error_type=type(exc).__name__,
            session_id=session_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=502, detail="No se pudo obtener respuesta del proveedor de LLM."
        )

    assert last_structured is not None  # boss.run siempre llama al actor al menos una vez

    # El historial guarda un turno "limpio" (sin el feedback del Critic
    # tejido adentro) -- el ruido de la orquestación ACB no debe viajar a
    # turnos futuros, solo el resultado final.
    _, canonical_user = render_session_prompt(
        full_transcript, session.metadata, version=PROMPT_VERSION, tier=resolved_tier
    )
    assistant_turn = final_result.model_dump_json()
    session.history.add_turn(
        {"role": "user", "content": canonical_user},
        {"role": "assistant", "content": assistant_turn},
    )
    await apply_compression(
        session.history, anchor_detection_mode=settings.anchor_detection_mode
    )
    session.metadata = await extract_and_merge_metadata(
        full_transcript, assistant_turn, session.metadata
    )

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
