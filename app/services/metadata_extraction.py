import json

import structlog

from app.prompts.loader import render_metadata_extraction_prompt
from app.services.llm_service import generate_estimation
from app.sessions import ProjectMetadata

logger = structlog.get_logger(__name__)


def _parse_json_response(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json") :]
    return json.loads(text.strip())


async def extract_and_merge_metadata(
    transcript: str,
    assistant_response: str,
    previous_metadata: ProjectMetadata,
) -> ProjectMetadata:
    """Segunda llamada al LLM (prompt propio, reutilizando el wrapper con
    cache/fallback de llm_service) que extrae qué se aprendió en este
    turno, y lo fusiona sobre lo que ya se sabía.

    Si el LLM devuelve algo que no parsea como el JSON esperado, se
    degrada a NO actualizar nada -- la extracción de metadata es una
    mejora de la experiencia conversacional, no algo que deba poder
    tumbar una estimación que sí se generó bien. El detalle del fallo
    queda en el log, nunca se propaga como error al cliente.
    """
    system, user = render_metadata_extraction_prompt(
        transcript, assistant_response, previous_metadata
    )

    try:
        # generate_estimation ya es genérica (solo system/user -> LLMResult)
        # desde el refactor de sesión 04: no hace falta una función nueva
        # en llm_service.py para esto, aunque el nombre quedó pensado para
        # estimaciones.
        result = await generate_estimation(system, user)
        data = _parse_json_response(result.estimation)
        update = ProjectMetadata(
            project_name=data.get("project_name"),
            assumed_team_size=data.get("assumed_team_size"),
            mentioned_technologies=data.get("mentioned_technologies") or [],
            agreed_scope=data.get("agreed_scope"),
        )
    except Exception as exc:
        logger.warning(
            "metadata_extraction_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return previous_metadata

    merged = previous_metadata.merge(update)
    logger.info(
        "metadata_extracted",
        project_name=merged.project_name,
        technologies_count=len(merged.mentioned_technologies),
    )
    return merged
