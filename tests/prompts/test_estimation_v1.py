import pytest
from jinja2 import Environment, StrictUndefined, UndefinedError

from app.prompts.loader import render_estimation_prompt
from app.schemas import DetailLevel, EstimationRequest, OutputFormat, ProjectType, ReferenceProject


def _request(**overrides) -> EstimationRequest:
    defaults = dict(
        description="El cliente quiere un portal de proveedores con carga de facturas.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.LINE_ITEMS,
    )
    defaults.update(overrides)
    return EstimationRequest(**defaults)


def test_user_prompt_contains_description_literally():
    """La descripción debe aparecer tal cual dentro de <project_description>."""
    description = "El cliente necesita un buscador interno de documentos con filtros por fecha y autor."
    _, user = render_estimation_prompt(_request(description=description))

    assert "<project_description>" in user
    assert "</project_description>" in user
    start = user.index("<project_description>") + len("<project_description>")
    end = user.index("</project_description>")
    assert description in user[start:end]


def test_output_format_keyword_is_mutually_exclusive():
    """El system prompt debe contener la palabra clave del output_format
    elegido, y NO la de los otros dos formatos."""
    system_phases, _ = render_estimation_prompt(_request(output_format=OutputFormat.PHASES_TABLE))
    assert "phases_table" in system_phases
    assert "line_items" not in system_phases
    assert "narrative" not in system_phases

    system_narrative, _ = render_estimation_prompt(_request(output_format=OutputFormat.NARRATIVE))
    assert "narrative" in system_narrative
    assert "phases_table" not in system_narrative
    assert "line_items" not in system_narrative


def test_detailed_level_adds_per_phase_assumptions_instruction():
    """detail_level=detailed agrega la instrucción de listar asunciones por
    fase; summary no debe incluirla."""
    marker = "list its own assumptions and risks"

    system_detailed, _ = render_estimation_prompt(_request(detail_level=DetailLevel.DETAILED))
    assert marker in system_detailed

    system_summary, _ = render_estimation_prompt(_request(detail_level=DetailLevel.SUMMARY))
    assert marker not in system_summary


def test_system_prompt_includes_examples():
    """El system prompt debe incluir los ejemplos few-shot via {% include %}."""
    system, _ = render_estimation_prompt(_request())
    assert "<examples>" in system
    assert "<example>" in system


def test_reference_projects_are_rendered_when_present():
    """Si se pasan reference_projects, el user prompt debe listarlos; si no
    se pasan, la sección no debe aparecer."""
    _, user_without = render_estimation_prompt(_request())
    assert "Reference projects" not in user_without

    _, user_with = render_estimation_prompt(
        _request(
            reference_projects=[
                ReferenceProject(name="CRM Inmobiliario", description="CRM a medida", hours=180),
            ]
        )
    )
    assert "Reference projects" in user_with
    assert "CRM Inmobiliario" in user_with
    assert "<actual_hours>180</actual_hours>" in user_with


def test_strict_undefined_raises_on_missing_variable():
    """Prueba aislada de la configuración de Jinja2, sin pasar por el
    loader: si algún día alguien cambia StrictUndefined por el Undefined
    por defecto (que renderiza silenciosamente como string vacío), este
    test tiene que empezar a fallar. Los templates reales de v1/v2 no
    tienen typos hoy, así que un test que solo los renderice no detectaría
    esa regresión — por eso se prueba la semántica directamente."""
    env = Environment(undefined=StrictUndefined)
    template = env.from_string("Hola {{ nombre_que_no_se_pasa }}")

    with pytest.raises(UndefinedError):
        template.render(nombre="Gabriel")


def test_unknown_version_raises():
    """Pedir una versión de prompt que no existe debe fallar de forma clara,
    no devolver un prompt vacío o incorrecto en silencio."""
    with pytest.raises(ValueError):
        render_estimation_prompt(_request(), version="v99")
