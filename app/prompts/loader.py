import hashlib
from functools import lru_cache
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas import EstimationRequest
from app.sessions import ProjectMetadata

PROMPTS_DIR = Path(__file__).parent / "estimation"
METADATA_PROMPTS_DIR = Path(__file__).parent / "metadata_extraction"

logger = structlog.get_logger(__name__)


def _jinja_environment(base_dir: Path, version: str) -> Environment:
    version_dir = base_dir / version
    if not version_dir.is_dir():
        raise ValueError(f"No existe la versión de prompt '{version}' en {base_dir}")

    return Environment(
        loader=FileSystemLoader(str(version_dir)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


@lru_cache(maxsize=None)
def _environment(version: str) -> Environment:
    """Un Environment de Jinja2 por versión, con el loader apuntando
    directo a esa carpeta. Así `{% include "examples.j2" %}` dentro de un
    template no necesita saber en qué versión está — y agregar v2/ no
    requiere tocar ningún .j2 de v1/.
    """
    return _jinja_environment(PROMPTS_DIR, version)


@lru_cache(maxsize=None)
def _metadata_environment(version: str) -> Environment:
    """Mismo patrón que _environment, pero para el set de templates de
    extracción de project_metadata -- son prompts independientes del
    prompt de estimación, con su propio ciclo de versionado."""
    return _jinja_environment(METADATA_PROMPTS_DIR, version)


def render_estimation_prompt(request: EstimationRequest, version: str = "v1") -> tuple[str, str]:
    """Renderiza (system, user) para el endpoint de estimación.

    Cambiar `version` (ej. a "v2") usa otra carpeta de templates sin tocar
    el resto del código — ese es el punto de versionar los prompts como
    archivos en vez de strings en Python.
    """
    env = _environment(version)

    system = env.get_template("system.j2").render(
        project_type=request.project_type,
        detail_level=request.detail_level,
        output_format=request.output_format,
        # El flujo de formulario (un solo turno) no tiene project_metadata
        # de sesión; el bloque <project_metadata> queda vacío.
        project_metadata=None,
    )
    user = env.get_template("user.j2").render(
        project_type=request.project_type,
        description=request.description,
        # StrictUndefined exige que la variable exista aunque sea None: el
        # {% if reference_projects %} del template necesita poder evaluarla.
        reference_projects=request.reference_projects,
    )

    content_hash = hashlib.sha256(f"{system}\n{user}".encode()).hexdigest()[:16]
    logger.info(
        "prompt_rendered",
        prompt_version=version,
        content_hash=content_hash,
        project_type=request.project_type.value,
        detail_level=request.detail_level.value,
        output_format=request.output_format.value,
    )

    return system, user


def render_session_prompt(
    transcript: str,
    project_metadata: ProjectMetadata,
    version: str = "v1",
) -> tuple[str, str]:
    """Renderiza (system, user) para un turno del endpoint conversacional
    (POST /sessions/{session_id}/estimate).

    A diferencia de render_estimation_prompt, acá no hay project_type ni
    output_format/detail_level tipados -- el turno es texto libre
    (transcript + adjuntos). El system.j2 sigue siendo el mismo archivo
    que el flujo de formulario: sus bloques condicionales por formato
    simplemente no matchean nada cuando esos valores son None, así que no
    se le fuerza ningún formato de salida a la conversación.
    """
    env = _environment(version)

    system = env.get_template("system.j2").render(
        project_type=None,
        detail_level=None,
        output_format=None,
        project_metadata=project_metadata,
    )
    user = env.get_template("session_user.j2").render(transcript=transcript)

    content_hash = hashlib.sha256(f"{system}\n{user}".encode()).hexdigest()[:16]
    logger.info(
        "session_prompt_rendered",
        prompt_version=version,
        content_hash=content_hash,
        project_metadata_known=not project_metadata.is_empty(),
    )

    return system, user


def render_metadata_extraction_prompt(
    transcript: str,
    assistant_response: str,
    previous_metadata: ProjectMetadata,
    version: str = "v1",
) -> tuple[str, str]:
    """Renderiza (system, user) para la segunda llamada al LLM que
    extrae project_metadata de un turno (ver
    app.services.metadata_extraction.extract_and_merge_metadata)."""
    env = _metadata_environment(version)

    system = env.get_template("system.j2").render()
    user = env.get_template("user.j2").render(
        previous_metadata_json=previous_metadata.model_dump_json(),
        transcript=transcript,
        assistant_response=assistant_response,
    )
    return system, user
