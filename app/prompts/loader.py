import hashlib
from functools import lru_cache
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas import EstimationRequest
from app.sessions import ProjectMetadata

PROMPTS_DIR = Path(__file__).parent / "estimation"
METADATA_PROMPTS_DIR = Path(__file__).parent / "metadata_extraction"
CONVERSATION_SUMMARY_PROMPTS_DIR = Path(__file__).parent / "conversation_summary"
CRITIC_PROMPTS_DIR = Path(__file__).parent / "critic"

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


@lru_cache(maxsize=None)
def _conversation_summary_environment(version: str) -> Environment:
    """Mismo patrón, para los templates del summarizer acumulativo
    (app.compression.summarizer)."""
    return _jinja_environment(CONVERSATION_SUMMARY_PROMPTS_DIR, version)


@lru_cache(maxsize=None)
def _critic_environment(version: str) -> Environment:
    """Mismo patrón, para los templates del Critic (app.critic)."""
    return _jinja_environment(CRITIC_PROMPTS_DIR, version)


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
        # El tier de audiencia es un concepto conversacional (se deriva del
        # transcript + metadata acumulada); el formulario de un solo turno
        # no lo resuelve, así que el bloque <audience> queda vacío.
        tier=None,
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
    tier=None,
    critic_feedback=None,
    project_type=None,
    detail_level=None,
    output_format=None,
) -> tuple[str, str]:
    """Renderiza (system, user) para un turno del endpoint conversacional
    (POST /sessions/{session_id}/estimate o .../estimate-acb).

    A diferencia de render_estimation_prompt, acá el turno es texto libre
    (transcript + adjuntos) -- pero el caller puede opcionalmente pasar
    `project_type`/`detail_level`/`output_format` para pedir el mismo
    control de formato que tiene el formulario de un solo turno (lo
    necesita, por ejemplo, el runner de evals -- ver evals/run.py). El
    system.j2 es el mismo archivo que el flujo de formulario: sus bloques
    condicionales simplemente no matchean nada cuando estos valores quedan
    en None, así que no se le fuerza ningún formato a una conversación que
    no lo pide.

    `tier` (app.tier_resolver.Tier | None) sí es propio de este flujo: lo
    resuelve el router antes de renderizar, a partir del transcript + la
    metadata acumulada (ver app.tier_resolver.resolve_tier).

    `critic_feedback` (app.schemas.CriticFeedback | None) solo lo usa el
    flujo Actor-Critic-Boss: en la segunda iteración en adelante, el Boss
    vuelve a renderizar el prompt con el feedback de la ronda anterior para
    que el actor sepa qué corregir.
    """
    env = _environment(version)

    system = env.get_template("system.j2").render(
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
        project_metadata=project_metadata,
        tier=tier,
    )
    user = env.get_template("session_user.j2").render(
        transcript=transcript, critic_feedback=critic_feedback
    )

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


def render_conversation_summary_prompt(
    *,
    previous_summary: str | None,
    evicted: list[dict],
    version: str = "v1",
) -> tuple[str, str]:
    """Renderiza (system, user) para el summarizer acumulativo (ver
    app.compression.summarizer.CumulativeSummarizer).

    `evicted` es una lista de dicts {"role": ..., "content": ...} -- los
    mensajes que la ventana deslizante está por descartar."""
    env = _conversation_summary_environment(version)
    context = {"previous_summary": previous_summary or "", "evicted": evicted}
    system = env.get_template("system.j2").render(**context)
    user = env.get_template("user.j2").render(**context)
    return system, user


def render_critic_prompt(
    *,
    transcript: str,
    metadata: ProjectMetadata,
    tier,
    result,
    version: str = "v1",
) -> tuple[str, str]:
    """Renderiza (system, user) para el Critic (ver app.critic.Critic).

    `tier` acepta tanto el enum Tier como su valor string."""
    env = _critic_environment(version)
    context = {
        "transcript": transcript,
        "metadata": metadata,
        "tier": tier.value if hasattr(tier, "value") else str(tier),
        "result": result,
    }
    system = env.get_template("system.j2").render(**context)
    user = env.get_template("user.j2").render(**context)
    return system, user
