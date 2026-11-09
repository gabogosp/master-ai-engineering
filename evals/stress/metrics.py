"""Tres métricas de stress-test del CAG.

Viven en un módulo separado de `evals/metrics.py`, no reexportado desde
`evals/__init__.py`, por una razón de FORMA, no de tema: las métricas de
`evals/metrics.py` tienen la firma `evaluate(case: GoldenCase, result:
EstimationResult) -> MetricResult` -- comparan una salida contra lo que un
caso dorado espera. Las de acá no reciben ni un GoldenCase ni un
EstimationResult:

- `LatencyBudgetMetric` / `CostBudgetMetric` evalúan un `TurnObservation`
  (el objeto que `EstimationService` ya loguea y adjunta a la respuesta
  HTTP, ver app/schemas.py) contra un presupuesto fijo.
- `MemoryDriftMetric` evalúa el snapshot de sesión que devuelve
  `GET /sessions/{id}` (`SessionInfoResponse`, como dict) buscando si un
  hecho sigue siendo recuperable después de que la ventana deslizante lo
  comprimió.

Forzarlas a la firma `(case, result)` de `evals/metrics.py` significaría
inventar un `GoldenCase`/`EstimationResult` falso solo para satisfacer un
tipo que no describe lo que de verdad se está midiendo -- de ahí el módulo
propio. Lo que SÍ se reusa es `MetricResult`: el shape (name, score,
passed, details) es el mismo, así que cualquier reporter que ya sepa
imprimir una lista de `MetricResult` funciona para las dos familias sin
cambios.
"""

import json
from typing import Literal

from evals.metrics import MetricResult

FactField = Literal["project_name", "technologies", "scope", "summary", "any"]


class LatencyBudgetMetric:
    """¿El turno respondió dentro del presupuesto de latencia?"""

    name = "latency_budget"

    def __init__(self, budget_ms: float) -> None:
        self.budget_ms = budget_ms

    def evaluate(self, observation) -> MetricResult:
        latency_ms = observation.latency_ms
        passed = latency_ms <= self.budget_ms
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"latency_ms={latency_ms} budget_ms={self.budget_ms}",
        )


class CostBudgetMetric:
    """¿El turno costó dentro del presupuesto en USD?"""

    name = "cost_budget"

    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = budget_usd

    def evaluate(self, observation) -> MetricResult:
        cost_usd = observation.cost_usd
        passed = cost_usd <= self.budget_usd
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"cost_usd={cost_usd} budget_usd={self.budget_usd}",
        )


class MemoryDriftMetric:
    """¿Un hecho de un turno anterior sigue siendo recuperable en el
    snapshot actual de la sesión (`GET /sessions/{id}`), después de que la
    ventana deslizante/compresión lo haya podido descartar?

    `fact_field` acota dónde buscar -- igual que `ScenarioTurn.fact_field`
    en evals/stress/scenarios.py (de hecho, el valor a pasar es
    literalmente ese campo para el turno que declaró el hecho):

    - "project_name"  -- snapshot["metadata"]["project_name"]
    - "technologies"  -- snapshot["metadata"]["mentioned_technologies"]
                          (join, porque es una lista)
    - "scope"         -- snapshot["metadata"]["agreed_scope"]
    - "summary"       -- snapshot["summary"] (el resumen acumulado; texto,
                          no `summary_chars`)
    - "any"           -- todo el snapshot serializado a JSON, incluido
                          `snapshot["anchors"]` (el texto de los mensajes
                          promovidos a ancla, no solo `anchors_count`) --
                          para hechos que podrían sobrevivir en más de un
                          lugar y no vale la pena acotar.

    Búsqueda case-insensitive por substring -- no es NLP, es un detector
    de regresión burdo pero barato: si el hecho ni siquiera aparece como
    texto, seguro que se perdió.
    """

    name = "memory_drift"

    def __init__(self, fact: str, fact_field: FactField = "any") -> None:
        self.fact = fact
        self.fact_field = fact_field

    def _haystack(self, snapshot: dict) -> str:
        metadata = snapshot.get("metadata") or {}
        if self.fact_field == "project_name":
            return str(metadata.get("project_name") or "")
        if self.fact_field == "technologies":
            return " ".join(metadata.get("mentioned_technologies") or [])
        if self.fact_field == "scope":
            return str(metadata.get("agreed_scope") or "")
        if self.fact_field == "summary":
            return str(snapshot.get("summary") or "")
        # ensure_ascii=False: si no, "clínicas" queda como "clínicas"
        # en el JSON y la búsqueda de substring en texto real nunca matchea.
        return json.dumps(snapshot, default=str, ensure_ascii=False)

    def evaluate(self, snapshot: dict) -> MetricResult:
        haystack = self._haystack(snapshot).lower()
        passed = self.fact.lower() in haystack
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=(
                f"fact={self.fact!r} fact_field={self.fact_field!r} "
                f"{'encontrado' if passed else 'NO encontrado'} en el snapshot"
            ),
        )
