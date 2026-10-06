import hashlib
from functools import lru_cache
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas import EstimationRequest

PROMPTS_DIR = Path(__file__).parent / "estimation"

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=None)
def _environment(version: str) -> Environment:
    """Un Environment de Jinja2 por versión, con el loader apuntando
    directo a esa carpeta. Así `{% include "examples.j2" %}` dentro de un
    template no necesita saber en qué versión está — y agregar v2/ no
    requiere tocar ningún .j2 de v1/.
    """
    version_dir = PROMPTS_DIR / version
    if not version_dir.is_dir():
        raise ValueError(f"No existe la versión de prompt '{version}' en {PROMPTS_DIR}")

    return Environment(
        loader=FileSystemLoader(str(version_dir)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


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
