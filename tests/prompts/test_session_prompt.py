from app.prompts.loader import render_estimation_prompt, render_session_prompt
from app.schemas import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from app.sessions import ProjectMetadata

SCOPE_CHANGE_MARKER = "MUST adjust the numeric"  # v1; v2 usa otra redacción, ver test aparte


def test_session_prompt_renders_empty_metadata_block_on_first_turn():
    """En el primer turno (sin datos conocidos todavía) el bloque
    <project_metadata> debe estar presente pero vacío."""
    system, _ = render_session_prompt("texto del primer turno", ProjectMetadata())
    assert "<project_metadata>" in system
    assert "Project name:" not in system


def test_session_prompt_fills_metadata_block_when_known():
    metadata = ProjectMetadata(project_name="CRM Ventas", assumed_team_size=2)
    system, _ = render_session_prompt("texto del turno", metadata)
    assert "Project name: CRM Ventas" in system
    assert "Assumed team size: 2" in system


def test_session_prompt_user_wraps_transcript():
    _, user = render_session_prompt("El cliente quiere sumar un sistema de citas.", ProjectMetadata())
    assert "<turn_transcript>" in user
    assert "El cliente quiere sumar un sistema de citas." in user


def test_scope_change_instruction_only_in_conversational_flow():
    """La instrucción de ajustar los números cuando cambia el alcance solo
    tiene sentido en una conversación multi-turno (hay un 'antes' con el que
    comparar) -- no debe aparecer en el flujo de formulario de un solo turno."""
    request = EstimationRequest(
        description="El cliente quiere un portal de proveedores con carga de facturas.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.LINE_ITEMS,
    )
    system_form, _ = render_estimation_prompt(request)
    system_session, _ = render_session_prompt("texto del turno", ProjectMetadata())

    assert SCOPE_CHANGE_MARKER not in system_form
    assert SCOPE_CHANGE_MARKER in system_session
