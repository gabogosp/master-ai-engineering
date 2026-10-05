from datetime import datetime, timezone
from typing import Annotated

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.attachments import (
    UnsupportedAttachmentError,
    build_transcript_with_attachments,
    extract_text,
)
from app.prompts.loader import render_session_prompt
from app.schemas import EstimationResponse, SessionCreatedResponse
from app.services.llm_service import (
    CONFIGURATION_ERRORS,
    PROVIDER_ERRORS,
    RATE_LIMIT_ERRORS,
    IncompleteEstimationError,
    generate_conversation_turn,
)
from app.services.metadata_extraction import extract_and_merge_metadata
from app.sessions import session_store

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["sessions"])

PROMPT_VERSION = "v1"


@router.post("/sessions", response_model=SessionCreatedResponse)
async def create_session() -> SessionCreatedResponse:
    session = session_store.create()
    return SessionCreatedResponse(session_id=session.session_id)


@router.post("/sessions/{session_id}/estimate", response_model=EstimationResponse)
async def estimate_turn(
    session_id: str,
    transcript: Annotated[str, Form(min_length=1)],
    attachments: Annotated[list[UploadFile] | None, File()] = None,
) -> EstimationResponse:
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

    system, user = render_session_prompt(
        full_transcript, session.metadata, version=PROMPT_VERSION
    )
    messages = session.history.to_messages_list(system)
    messages.append({"role": "user", "content": user})

    # El detalle de cada fallo va al log. Al cliente solo le damos un mensaje genérico
    # (mismo mapeo de errores que /api/v1/estimate).
    try:
        result = await generate_conversation_turn(messages)
    except IncompleteEstimationError as exc:
        logger.warning("estimation_incomplete", error=str(exc), session_id=session_id)
        raise HTTPException(
            status_code=502, detail="La estimación quedó incompleta. Inténtalo de nuevo."
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

    # El historial y el project_metadata se actualizan recién si la llamada
    # al LLM salió bien -- un turno fallido no debe ensuciar la sesión.
    session.history.add_turn(
        {"role": "user", "content": user},
        {"role": "assistant", "content": result.estimation},
    )
    session.metadata = await extract_and_merge_metadata(
        full_transcript, result.estimation, session.metadata
    )

    return EstimationResponse(
        text=result.estimation,
        prompt_version=PROMPT_VERSION,
        model=result.model,
        provider=result.provider,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        created_at=datetime.now(timezone.utc),
        project_metadata=session.metadata.model_dump(),
        history_turns=len(session.history),
    )
