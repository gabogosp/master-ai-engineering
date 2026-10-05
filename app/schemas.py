from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class ReferenceProject(BaseModel):
    """Un proyecto pasado similar, aportado como contexto de calibración
    (no se estima; ayuda al modelo a anclar horas/alcance)."""

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1000)
    hours: int = Field(gt=0)


class EstimationRequest(BaseModel):
    description: str = Field(min_length=20, max_length=2000)
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat
    reference_projects: list[ReferenceProject] | None = None


class SessionCreatedResponse(BaseModel):
    session_id: str


class EstimationResponse(BaseModel):
    text: str
    prompt_version: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    created_at: datetime
    # Solo poblados por el endpoint de sesión (POST /sessions/{id}/estimate);
    # el endpoint de formulario de un solo turno los deja en None.
    project_metadata: dict | None = None
    history_turns: int | None = None
