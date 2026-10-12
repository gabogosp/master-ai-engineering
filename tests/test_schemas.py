from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas import (
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
    OutputFormat,
    Phase,
    ProjectType,
    ReferenceProject,
)

VALID_REQUEST_KWARGS = dict(
    description="El cliente quiere un portal de proveedores con carga de facturas.",
    project_type=ProjectType.WEB_SAAS,
    detail_level=DetailLevel.MEDIUM,
    output_format=OutputFormat.LINE_ITEMS,
)


def test_valid_request_is_accepted():
    request = EstimationRequest(**VALID_REQUEST_KWARGS)
    assert request.reference_projects is None


def test_description_too_short_is_rejected():
    with pytest.raises(ValidationError):
        EstimationRequest(**{**VALID_REQUEST_KWARGS, "description": "Muy corta"})


def test_description_too_long_is_rejected():
    with pytest.raises(ValidationError):
        EstimationRequest(**{**VALID_REQUEST_KWARGS, "description": "x" * 2001})


def test_description_at_boundaries_is_accepted():
    EstimationRequest(**{**VALID_REQUEST_KWARGS, "description": "x" * 20})
    EstimationRequest(**{**VALID_REQUEST_KWARGS, "description": "x" * 2000})


def test_missing_required_field_is_rejected():
    kwargs = dict(VALID_REQUEST_KWARGS)
    kwargs.pop("project_type")
    with pytest.raises(ValidationError):
        EstimationRequest(**kwargs)


def test_invalid_enum_value_is_rejected():
    with pytest.raises(ValidationError):
        EstimationRequest(**{**VALID_REQUEST_KWARGS, "project_type": "quantum_computer"})


def test_reference_projects_accepts_a_valid_list():
    request = EstimationRequest(
        **VALID_REQUEST_KWARGS,
        reference_projects=[
            ReferenceProject(name="CRM Inmobiliario", description="CRM a medida", hours=180),
        ],
    )
    assert request.reference_projects[0].hours == 180


def test_reference_project_requires_positive_hours():
    with pytest.raises(ValidationError):
        ReferenceProject(name="X", description="Y", hours=0)

    with pytest.raises(ValidationError):
        ReferenceProject(name="X", description="Y", hours=-5)


def test_reference_project_rejects_empty_name_or_description():
    with pytest.raises(ValidationError):
        ReferenceProject(name="", description="Y", hours=10)

    with pytest.raises(ValidationError):
        ReferenceProject(name="X", description="", hours=10)


def test_estimation_response_round_trip():
    result = EstimationResult(
        summary="Resumen con suficiente longitud para pasar la validación del campo.",
        confidence_pct=80,
        phases=[
            Phase(
                name="Implementación",
                duration_weeks=4,
                cost_eur=10_000,
                summary="Desarrollo del backend y frontend principal del proyecto.",
            )
        ],
        total_duration_weeks=4,
        total_cost_eur=10_000,
    )
    response = EstimationResponse(
        result=result,
        prompt_version="v1",
        model="gpt-4o-mini",
        provider="openai",
        input_tokens=100,
        output_tokens=50,
        created_at=datetime.now(timezone.utc),
    )
    dumped = response.model_dump()
    assert dumped["prompt_version"] == "v1"
    assert dumped["input_tokens"] == 100
    assert dumped["result"]["total_cost_eur"] == 10_000


def test_critic_feedback_accept_allows_empty_issues():
    from app.schemas import CriticFeedback

    feedback = CriticFeedback(verdict="accept", issues=[], confidence_in_review=90)
    assert feedback.issues == []


def test_critic_feedback_needs_iteration_requires_blocking_issue():
    from app.schemas import CriticFeedback, CriticIssue

    with pytest.raises(ValidationError):
        CriticFeedback(
            verdict="needs_iteration",
            issues=[
                CriticIssue(
                    category="tier_mismatch",
                    severity="minor",
                    field_path="summary",
                    description="Tono levemente informal para el tier ejecutivo.",
                )
            ],
            confidence_in_review=80,
        )


def test_critic_feedback_needs_iteration_accepts_with_major_issue():
    from app.schemas import CriticFeedback, CriticIssue

    feedback = CriticFeedback(
        verdict="needs_iteration",
        issues=[
            CriticIssue(
                category="math_error",
                severity="major",
                field_path="total_cost_eur",
                description="La suma de fases no coincide con el total.",
            )
        ],
        confidence_in_review=80,
    )
    assert feedback.verdict == "needs_iteration"


def test_critic_feedback_reject_requires_at_least_one_issue():
    from app.schemas import CriticFeedback

    with pytest.raises(ValidationError):
        CriticFeedback(verdict="reject", issues=[], confidence_in_review=80)
