"""Métricas determinísticas sobre un par (GoldenCase, EstimationResult).

Viven dentro del repo (en vez de delegar a un juez LLM tipo DeepEval) para
que la suite corra sin gastar un LLM y diga exactamente *por qué* falló un
caso. Son determinísticas a propósito: su trabajo es atrapar regresiones
obvias, no reemplazar el juicio humano sobre la calidad de la prosa.
"""

from dataclasses import dataclass

from app.schemas import OUT_OF_SCOPE_PREFIX, EstimationResult
from evals.dataset import GoldenCase


@dataclass
class MetricResult:
    name: str
    score: float
    passed: bool
    details: str


class SchemaAdherenceMetric:
    """Instructor + los model_validator de EstimationResult ya fuerzan la
    aritmética y la regla de confianza baja -- esta métrica existe para
    marcar si alguna de esas garantías *regresó* (ej. un cambio futuro que
    rompe un validator). También chequea la cantidad de fases contra el
    rango esperado, cuando el caso lo define."""

    name = "schema_adherence"

    def evaluate(self, case: GoldenCase, result: EstimationResult) -> MetricResult:
        problems: list[str] = []

        phase_sum = sum(p.cost_eur for p in result.phases)
        if phase_sum != result.total_cost_eur:
            problems.append(f"suma de fases ({phase_sum}) != total_cost_eur ({result.total_cost_eur})")

        if result.confidence_pct < 30 and not result.summary.startswith(OUT_OF_SCOPE_PREFIX):
            problems.append("confianza baja sin el prefijo 'Out of scope:'")

        if case.expected_phase_count_range:
            lo, hi = case.expected_phase_count_range
            if not (lo <= len(result.phases) <= hi):
                problems.append(f"cantidad de fases {len(result.phases)} fuera de [{lo}, {hi}]")

        score = 1.0 if not problems else 0.0
        return MetricResult(
            name=self.name, score=score, passed=not problems, details="; ".join(problems) or "schema ok"
        )


class CostBoundsMetric:
    """Cota de sanidad para costo y duración absolutos. Los casos
    out-of-scope esperan el sobre placeholder (cost=0, duration=1)."""

    name = "cost_bounds"

    def evaluate(self, case: GoldenCase, result: EstimationResult) -> MetricResult:
        problems: list[str] = []

        if case.expected_out_of_scope:
            if result.total_cost_eur != 0 or result.total_duration_weeks != 1:
                problems.append(
                    "se esperaba el sobre out-of-scope (cost=0, duration=1), llegó "
                    f"cost={result.total_cost_eur}, weeks={result.total_duration_weeks}"
                )
        else:
            if case.expected_cost_range_eur:
                lo, hi = case.expected_cost_range_eur
                if not (lo <= result.total_cost_eur <= hi):
                    problems.append(f"costo {result.total_cost_eur} EUR fuera de [{lo}, {hi}]")
            if case.expected_duration_weeks_range:
                lo, hi = case.expected_duration_weeks_range
                if not (lo <= result.total_duration_weeks <= hi):
                    problems.append(f"duración {result.total_duration_weeks}sem fuera de [{lo}, {hi}]")

        score = 1.0 if not problems else 0.0
        return MetricResult(
            name=self.name,
            score=score,
            passed=not problems,
            details="; ".join(problems) or "costo y duración dentro de rango",
        )


class ContentRecallMetric:
    """Recall liviano -- ¿el summary/las fases mencionan lo que el usuario
    pidió de verdad? Los casos out-of-scope se saltan este chequeo."""

    name = "content_recall"

    def evaluate(self, case: GoldenCase, result: EstimationResult) -> MetricResult:
        if case.expected_out_of_scope:
            return MetricResult(
                name=self.name, score=1.0, passed=True, details="saltado -- caso out-of-scope por diseño"
            )

        haystack = " ".join(
            [result.summary] + [phase.summary + " " + phase.name for phase in result.phases]
        ).lower()

        missing_summary = [term for term in case.expected_in_summary if term.lower() not in haystack]

        tech_hit = True
        if case.expected_technologies_any_of:
            tech_hit = any(t.lower() in haystack for t in case.expected_technologies_any_of)

        problems: list[str] = []
        if missing_summary:
            problems.append(f"faltan en el summary: {missing_summary}")
        if not tech_hit:
            problems.append(f"ninguna de {case.expected_technologies_any_of} se menciona")

        score = 1.0 if not problems else 0.0
        return MetricResult(
            name=self.name,
            score=score,
            passed=not problems,
            details="; ".join(problems) or "contenido esperado presente",
        )


_DEFAULT_METRICS = (SchemaAdherenceMetric(), CostBoundsMetric(), ContentRecallMetric())


def run_all_metrics(case: GoldenCase, result: EstimationResult) -> list[MetricResult]:
    return [m.evaluate(case, result) for m in _DEFAULT_METRICS]
