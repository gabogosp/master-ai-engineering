from evals.dataset import GoldenCase, load_dataset
from evals.metrics import ContentRecallMetric, CostBoundsMetric, SchemaAdherenceMetric
from app.schemas import EstimationResult, Phase


def _case(**overrides) -> GoldenCase:
    defaults = dict(
        id="case-test",
        transcript="x" * 30,
        project_type="web_saas",
    )
    defaults.update(overrides)
    return GoldenCase(**defaults)


def _result(
    summary: str = "Resumen de prueba con longitud suficiente para validar.",
    confidence: int = 80,
    phases: list[Phase] | None = None,
    total_cost_eur: int = 9000,
    total_duration_weeks: int = 3,
) -> EstimationResult:
    if phases is None:
        phases = [
            Phase(
                name="Implementación",
                duration_weeks=3,
                cost_eur=9000,
                summary="Desarrollo del CRM con React y Postgres para el equipo de ventas.",
            )
        ]
    return EstimationResult(
        summary=summary,
        confidence_pct=confidence,
        phases=phases,
        total_duration_weeks=total_duration_weeks,
        total_cost_eur=total_cost_eur,
    )


def test_golden_dataset_file_has_16_cases_and_all_parse():
    cases = load_dataset()
    assert len(cases) == 16
    assert len({c.id for c in cases}) == 16  # ids únicos


def test_golden_dataset_covers_out_of_scope_and_tier_cases():
    cases = {c.id: c for c in load_dataset()}
    assert cases["case-005-out-of-scope-vague"].expected_out_of_scope
    assert cases["case-007-nda-executive"].expected_tier == "executive"
    assert cases["case-010-small-team-pm"].expected_tier == "pm"
    assert cases["case-011-developer-microservices"].expected_tier == "developer"


# --- SchemaAdherenceMetric ---------------------------------------------------


def test_schema_adherence_passes_when_sums_match():
    metric = SchemaAdherenceMetric()
    result = metric.evaluate(_case(), _result())
    assert result.passed


def test_schema_adherence_fails_when_phase_sum_mismatches_total():
    metric = SchemaAdherenceMetric()
    base = _result()
    bad = EstimationResult.model_construct(**{**base.__dict__, "total_cost_eur": 1})
    outcome = metric.evaluate(_case(), bad)
    assert not outcome.passed
    assert "suma de fases" in outcome.details


def test_schema_adherence_checks_phase_count_range():
    metric = SchemaAdherenceMetric()
    case = _case(expected_phase_count_range=(3, 6))
    outcome = metric.evaluate(case, _result())  # 1 fase, fuera de [3,6]
    assert not outcome.passed
    assert "cantidad de fases" in outcome.details


# --- CostBoundsMetric ---------------------------------------------------------


def test_cost_bounds_passes_within_range():
    metric = CostBoundsMetric()
    case = _case(expected_cost_range_eur=(5000, 20000), expected_duration_weeks_range=(1, 10))
    outcome = metric.evaluate(case, _result())
    assert outcome.passed


def test_cost_bounds_fails_outside_range():
    metric = CostBoundsMetric()
    case = _case(expected_cost_range_eur=(50000, 100000))
    outcome = metric.evaluate(case, _result(total_cost_eur=9000))
    assert not outcome.passed


def test_cost_bounds_out_of_scope_expects_placeholder_envelope():
    metric = CostBoundsMetric()
    case = _case(expected_out_of_scope=True)
    placeholder = EstimationResult(
        summary="Out of scope: no hay información suficiente.",
        confidence_pct=20,
        phases=[Phase(name="Sin estimar", duration_weeks=1, cost_eur=0, summary="No se puede dimensionar.")],
        total_duration_weeks=1,
        total_cost_eur=0,
    )
    outcome = metric.evaluate(case, placeholder)
    assert outcome.passed


def test_cost_bounds_out_of_scope_fails_if_envelope_has_real_numbers():
    metric = CostBoundsMetric()
    case = _case(expected_out_of_scope=True)
    outcome = metric.evaluate(case, _result())  # cost=9000, no es el placeholder
    assert not outcome.passed


# --- ContentRecallMetric ------------------------------------------------------


def test_content_recall_passes_when_terms_present():
    metric = ContentRecallMetric()
    case = _case(expected_in_summary=["CRM"], expected_technologies_any_of=["React", "Postgres"])
    outcome = metric.evaluate(case, _result())
    assert outcome.passed


def test_content_recall_fails_when_expected_term_missing():
    metric = ContentRecallMetric()
    case = _case(expected_in_summary=["blockchain"])
    outcome = metric.evaluate(case, _result())
    assert not outcome.passed
    assert "blockchain" in outcome.details


def test_content_recall_fails_when_no_technology_mentioned():
    metric = ContentRecallMetric()
    case = _case(expected_technologies_any_of=["Kafka", "Spark"])
    outcome = metric.evaluate(case, _result())
    assert not outcome.passed


def test_content_recall_skips_out_of_scope_cases():
    metric = ContentRecallMetric()
    case = _case(expected_out_of_scope=True, expected_in_summary=["nunca va a matchear esto"])
    outcome = metric.evaluate(case, _result())
    assert outcome.passed
    assert "saltado" in outcome.details
