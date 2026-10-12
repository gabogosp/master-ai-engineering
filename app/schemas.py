from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


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


# --- Salida estructurada (Instructor + Pydantic) ----------------------------
#
# Dos reglas de negocio que el LLM no puede romper: Instructor reintenta
# contra el modelo (devolviéndole el ValueError) hasta `max_retries` veces
# cuando un model_validator falla. Esto reemplaza el parche de prompting que
# habíamos hecho antes para la aritmética ("el total debe ser la suma del
# desglose") por una validación real, por código.

OUT_OF_SCOPE_PREFIX = "Out of scope:"
LOW_CONFIDENCE_THRESHOLD = 30


class Phase(BaseModel):
    """Una fase del desglose de la estimación."""

    name: str = Field(min_length=1, max_length=64)
    duration_weeks: int = Field(ge=1, le=52)
    cost_eur: int = Field(ge=0, le=1_000_000)
    summary: str = Field(min_length=10, max_length=600)


class EstimationResult(BaseModel):
    """Estimación estructurada. Los dos validators de abajo son las reglas de
    negocio que el LLM no puede romper -- Instructor reintenta contra el
    modelo cuando alguno de los dos lanza.

    El orden de los campos es deliberado: `phases` va ANTES que los totales,
    así el modelo (generación autorregresiva) se compromete primero con los
    números por fase y recién después tiene que sumarlos para los totales.
    Poner los totales primero hace que el modelo elija un número redondo y
    ajuste las fases para que cuadren, cosa que hace mal aritméticamente.
    """

    summary: str = Field(min_length=10, max_length=1200)
    confidence_pct: int = Field(ge=0, le=100)
    phases: list[Phase] = Field(min_length=1, max_length=8)
    total_duration_weeks: int = Field(ge=1, le=104)
    total_cost_eur: int = Field(ge=0, le=2_000_000)

    @model_validator(mode="after")
    def phases_sum_matches_total(self) -> "EstimationResult":
        phase_sum = sum(p.cost_eur for p in self.phases)
        if phase_sum != self.total_cost_eur:
            raise ValueError(
                f"phases sum ({phase_sum} EUR) does not match total_cost_eur "
                f"({self.total_cost_eur} EUR); adjust either the phases or the total"
            )
        return self

    @model_validator(mode="after")
    def low_confidence_requires_out_of_scope_prefix(self) -> "EstimationResult":
        if self.confidence_pct < LOW_CONFIDENCE_THRESHOLD and not self.summary.startswith(
            OUT_OF_SCOPE_PREFIX
        ):
            raise ValueError(
                f"confidence_pct < {LOW_CONFIDENCE_THRESHOLD} requires summary to "
                f"start with {OUT_OF_SCOPE_PREFIX!r}; refuse the estimation if the "
                f"description is too vague to size"
            )
        return self


class EstimationResponse(BaseModel):
    result: EstimationResult
    prompt_version: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    created_at: datetime
    cached: bool = False
    # Solo poblados por el endpoint de sesión (POST /sessions/{id}/estimate);
    # el endpoint de formulario de un solo turno los deja en None.
    project_metadata: dict | None = None
    history_turns: int | None = None
