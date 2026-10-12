from typing import Annotated

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.attachments import (
    UnsupportedAttachmentError,
    build_transcript_with_attachments,
    extract_text,
)
from app.guardrails.input import InputGuardrailViolation
from app.schemas import (
    ACBResponse,
    DetailLevel,
    EstimationResponse,
    OutputFormat,
    ProjectType,
    SessionCreatedResponse,
    SessionInfoResponse,
)
from app.services.estimation import estimation_service
from app.services.llm_service import (
    CONFIGURATION_ERRORS,
    PROVIDER_ERRORS,
    RATE_LIMIT_ERRORS,
    EstimationValidationError,
)
from app.sessions import Session, session_store
from app.tier_resolver import Tier

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["sessions"])


async def _resolve_session_and_enrich(
    session_id: str, transcript: str, attachments: list[UploadFile] | None
) -> tuple[Session, str, int]:
    """Preludio HTTP compartido entre /estimate y /estimate-acb: busca la
    sesión y extrae el texto de los adjuntos. A propósito NO hace nada de
    negocio (ni guardrail de input, ni resolución de tier) -- eso vive en
    EstimationService, que no sabe nada de UploadFile/HTTPException.

    Devuelve (session, full_transcript, attachments_total_chars). Este
    último separado del total de full_transcript porque el evento
    turn_observed (ver EstimationService) quiere poder distinguir "cuánto
    aportó el transcript" de "cuánto aportaron los adjuntos" al armar las
    curvas de carga por tamaño de adjunto.
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

    attachments_total_chars = sum(len(text) for _, text in attachment_texts)
    full_transcript = build_transcript_with_attachments(transcript, attachment_texts)
    return session, full_transcript, attachments_total_chars


def _map_llm_errors(exc: Exception, *, session_id: str) -> HTTPException:
    """Traduce las excepciones que deja propagar EstimationService a
    códigos HTTP. Un único lugar para los dos endpoints (/estimate y
    /estimate-acb), que antes de este refactor duplicaban exactamente
    este mismo mapeo de 5 ramas."""
    if isinstance(exc, InputGuardrailViolation):
        logger.info(
            "estimation_blocked_by_input_guardrail", reason=exc.reason, session_id=session_id
        )
        return HTTPException(
            status_code=400, detail={"reason": exc.reason, "message": exc.message}
        )
    if isinstance(exc, EstimationValidationError):
        logger.warning(
            "estimation_validation_exhausted", error=str(exc)[:300], session_id=session_id
        )
        return HTTPException(
            status_code=502,
            detail="El modelo no logró producir una estimación válida. Inténtalo de nuevo.",
        )
    if isinstance(exc, CONFIGURATION_ERRORS):
        logger.error(
            "llm_configuration_error",
            error_type=type(exc).__name__,
            session_id=session_id,
            exc_info=True,
        )
        return HTTPException(status_code=500, detail="El servicio de LLM está mal configurado.")
    if isinstance(exc, RATE_LIMIT_ERRORS):
        logger.warning("llm_rate_limited", error_type=type(exc).__name__, session_id=session_id)
        return HTTPException(
            status_code=503,
            detail="El proveedor de LLM no puede atender la petición ahora. Inténtalo más tarde.",
        )
    if isinstance(exc, PROVIDER_ERRORS):
        logger.error(
            "llm_provider_error",
            error_type=type(exc).__name__,
            session_id=session_id,
            exc_info=True,
        )
        return HTTPException(
            status_code=502, detail="No se pudo obtener respuesta del proveedor de LLM."
        )
    # No reconocida: la dejamos pasar tal cual -- FastAPI la convierte en
    # un 500 genérico, y el traceback completo queda en el log del server.
    raise exc


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
        summary=session.history.summary,
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
    session, full_transcript, attachments_total_chars = await _resolve_session_and_enrich(
        session_id, transcript, attachments
    )
    try:
        return await estimation_service.estimate_conversational(
            session=session,
            full_transcript=full_transcript,
            attachments_total_chars=attachments_total_chars,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            tier_override=tier,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _map_llm_errors(exc, session_id=session_id) from exc


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
    session, full_transcript, attachments_total_chars = await _resolve_session_and_enrich(
        session_id, transcript, attachments
    )
    try:
        return await estimation_service.estimate_with_acb(
            session=session,
            full_transcript=full_transcript,
            attachments_total_chars=attachments_total_chars,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            tier_override=tier,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _map_llm_errors(exc, session_id=session_id) from exc
