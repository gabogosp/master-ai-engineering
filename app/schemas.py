from datetime import datetime
from enum import Enum
from typing import Literal

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


class SessionInfoResponse(BaseModel):
    """Vista de debug de una sesión (GET /sessions/{id}): además del
    historial y la metadata, expone lo que la ventana deslizante comprimida
    produjo -- cuántos mensajes quedan en la ventana activa, cuántos se
    promovieron a ancla, cuántos caracteres tiene el resumen acumulado, y
    el último tier de audiencia resuelto (con la regla que lo disparó)."""

    session_id: str
    message_count: int
    max_turns: int
    metadata: dict
    anchors_count: int = 0
    summary_chars: int = 0
    # Texto del resumen acumulado, no solo su longitud -- lo necesita
    # MemoryDriftMetric (evals/stress/metrics.py, fact_field="summary") para
    # buscar un hecho dentro del resumen comprimido, no solo contarlo.
    summary: str | None = None
    # Texto de los mensajes promovidos a ancla, no solo `anchors_count` --
    # un hecho declarado en sesión 6 (Bloque 2) puede sobrevivir SOLO
    # dentro de un ancla (ni en el resumen ni en metadata), y
    # MemoryDriftMetric con fact_field="any" solo puede verlo si el texto
    # de la ancla efectivamente viaja en este snapshot.
    anchors: list[str] = Field(default_factory=list)
    last_resolved_tier: str | None = None
    last_tier_rule: str | None = None


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


class TurnObservation(BaseModel):
    """Telemetría unificada de un turno conversacional (sesión 6, Bloque
    1/4 de evals/stress). Se construye una sola vez en EstimationService,
    se loguea (`logger.info("turn_observed", **observation.model_dump())`)
    Y se adjunta a la respuesta -- el runner de stress la lee directo del
    JSON, sin tener que parsear logs con `capture_logs`.

    `cache_hit_kind` sí puede valer "exact" acá (a diferencia de un diseño
    que lo deje fijo en "none"): nuestro flujo conversacional SÍ pasa por
    la cache exact-match (la semántica es la única que está excluida, ver
    sección "Cache semántica" del README) -- hardcodear "none" sería
    inexacto para este sistema en particular.
    """

    turn_index: int = Field(ge=1)
    session_id: str
    enriched_transcript_chars: int = Field(ge=0)
    attachments_total_chars: int = Field(ge=0)
    messages_in_window: int = Field(ge=0)
    anchors_count: int = Field(ge=0)
    summary_chars: int = Field(ge=0)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    cache_hit_kind: Literal["none", "exact", "semantic"] = "none"
    last_resolved_tier: str | None = None


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
    observation: TurnObservation | None = None


# --- Actor-Critic-Boss (POST /sessions/{id}/estimate-acb) -------------------
#
# El Critic audita una EstimationResult ya generada por el actor (el mismo
# generate_structured_conversation_turn de siempre) y devuelve feedback
# ESTRUCTURADO -- nunca prosa libre, para que el Boss pueda decidir en código
# qué hacer con cada issue en vez de tener que parsear texto.

CriticIssueCategory = Literal[
    "math_error",
    "hallucination",
    "scope_mismatch",
    "phase_imbalance",
    "missing_assumption",
    "unrealistic_estimate",
    "tier_mismatch",
]

CriticIssueSeverity = Literal["critical", "major", "minor"]

CriticVerdict = Literal["accept", "needs_iteration", "reject"]


class CriticIssue(BaseModel):
    """Un defecto puntual señalado por el Critic.

    `field_path` usa notación con puntos/corchetes sobre la estimación bajo
    revisión (`phases[2].cost_eur`, `summary`, `total_cost_eur`). El Boss
    puede reenviarlo tal cual al prompt del actor en la siguiente iteración."""

    category: CriticIssueCategory
    severity: CriticIssueSeverity
    field_path: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=5, max_length=500)
    suggested_fix: str | None = Field(default=None, max_length=300)


class CriticFeedback(BaseModel):
    """Salida completa del Critic. El Boss la trata como autoritativa.

    Validator: `needs_iteration` exige al menos un issue critical/major --
    si el Critic solo marca issues minor, debería aceptar directamente en
    vez de gastar otra llamada al actor."""

    verdict: CriticVerdict
    issues: list[CriticIssue] = Field(default_factory=list, max_length=12)
    confidence_in_review: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def iteration_requires_blocking_issue(self) -> "CriticFeedback":
        if self.verdict == "needs_iteration":
            blocking = [i for i in self.issues if i.severity in {"critical", "major"}]
            if not blocking:
                raise ValueError(
                    "verdict 'needs_iteration' requires at least one issue with "
                    "severity in {critical, major}; minor-only issues should accept"
                )
        if self.verdict == "reject" and not self.issues:
            raise ValueError("verdict 'reject' requires at least one issue describing why")
        return self


BossDecision = Literal["accept", "iterate", "synthesize"]


class ACBIteration(BaseModel):
    """Registro de auditoría de una ronda actor+critic -- se persiste en la
    respuesta para poder mostrar la traza completa (panel de debug)."""

    iteration: int = Field(ge=0)
    decision_after: BossDecision
    critic_verdict: str
    critic_confidence: int = Field(ge=0, le=100)
    issue_summary: list[str] = Field(default_factory=list)


class BossTrace(BaseModel):
    iterations: list[ACBIteration] = Field(default_factory=list)
    final_decision: BossDecision
    iterations_run: int = Field(ge=0)


class ACBResponse(EstimationResponse):
    """Misma forma que EstimationResponse más la traza de Actor-Critic-Boss."""

    acb: BossTrace
