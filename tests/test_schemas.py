from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas import (
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    OutputFormat,
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
    response = EstimationResponse(
        text="## Estimación\nTotal: 10 horas",
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
