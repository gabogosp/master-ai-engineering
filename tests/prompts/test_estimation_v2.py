from app.prompts.loader import render_estimation_prompt
from app.schemas import DetailLevel, EstimationRequest, OutputFormat, ProjectType


def _request(**overrides) -> EstimationRequest:
    defaults = dict(
        description="El cliente quiere un catálogo de productos con búsqueda y filtros.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.LINE_ITEMS,
    )
    defaults.update(overrides)
    return EstimationRequest(**defaults)


def test_v2_renders_successfully_and_differs_from_v1():
    """v2 debe renderizar sin errores y tener un tono distinto a v1 (no ser
    el mismo texto), aunque comparta la misma estructura de reglas."""
    system_v1, _ = render_estimation_prompt(_request(), version="v1")
    system_v2, _ = render_estimation_prompt(_request(), version="v2")

    assert system_v1 != system_v2
    assert "no-fluff estimate" in system_v2
    assert "no-fluff estimate" not in system_v1


def test_v2_keeps_output_format_and_detail_level_conditionals():
    """La variación de tono no debe romper los bloques condicionales que
    testeamos para v1: siguen siendo mutuamente excluyentes en v2."""
    system_phases, _ = render_estimation_prompt(
        _request(output_format=OutputFormat.PHASES_TABLE), version="v2"
    )
    assert "phases_table" in system_phases
    assert "narrative" not in system_phases

    system_detailed, _ = render_estimation_prompt(
        _request(detail_level=DetailLevel.DETAILED), version="v2"
    )
    assert "lists its own assumptions and risks" in system_detailed
