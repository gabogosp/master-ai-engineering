from app.schemas import TurnObservation
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric


def _observation(**overrides) -> TurnObservation:
    defaults = dict(
        turn_index=1,
        session_id="sess-1",
        enriched_transcript_chars=100,
        attachments_total_chars=0,
        messages_in_window=2,
        anchors_count=0,
        summary_chars=0,
        tokens_in=50,
        tokens_out=50,
        cost_usd=0.01,
        latency_ms=500.0,
    )
    defaults.update(overrides)
    return TurnObservation(**defaults)


# --- LatencyBudgetMetric -----------------------------------------------------


def test_latency_budget_metric_passes_when_under_budget():
    metric = LatencyBudgetMetric(budget_ms=1000)
    result = metric.evaluate(_observation(latency_ms=500.0))
    assert result.passed
    assert result.score == 1.0


def test_latency_budget_metric_fails_when_over_budget():
    metric = LatencyBudgetMetric(budget_ms=1000)
    result = metric.evaluate(_observation(latency_ms=1500.0))
    assert not result.passed
    assert result.score == 0.0


def test_latency_budget_metric_boundary_exact_budget_passes():
    """El límite es inclusivo (<=) -- justo en el presupuesto, pasa."""
    metric = LatencyBudgetMetric(budget_ms=1000)
    result = metric.evaluate(_observation(latency_ms=1000.0))
    assert result.passed


# --- CostBudgetMetric ---------------------------------------------------------


def test_cost_budget_metric_passes_when_under_budget():
    metric = CostBudgetMetric(budget_usd=0.05)
    result = metric.evaluate(_observation(cost_usd=0.01))
    assert result.passed
    assert result.score == 1.0


def test_cost_budget_metric_fails_when_over_budget():
    metric = CostBudgetMetric(budget_usd=0.05)
    result = metric.evaluate(_observation(cost_usd=0.10))
    assert not result.passed
    assert result.score == 0.0


def test_cost_budget_metric_boundary_exact_budget_passes():
    metric = CostBudgetMetric(budget_usd=0.05)
    result = metric.evaluate(_observation(cost_usd=0.05))
    assert result.passed


# --- MemoryDriftMetric ---------------------------------------------------------


def _snapshot(**overrides) -> dict:
    defaults = dict(
        session_id="sess-1",
        message_count=4,
        max_turns=6,
        metadata={
            "project_name": "MediFlow",
            "mentioned_technologies": ["React Native", "Flutter"],
            "agreed_scope": "MVP con agenda y recordatorios",
        },
        anchors_count=1,
        summary_chars=120,
        summary="El proyecto MediFlow es una app de agenda para clínicas.",
        last_resolved_tier=None,
        last_tier_rule=None,
    )
    defaults.update(overrides)
    return defaults


def test_memory_drift_metric_passes_when_fact_survives_in_its_field():
    metric = MemoryDriftMetric(fact="mediflow", fact_field="project_name")
    result = metric.evaluate(_snapshot())
    assert result.passed
    assert result.score == 1.0


def test_memory_drift_metric_fails_when_fact_is_gone():
    metric = MemoryDriftMetric(fact="kairos", fact_field="project_name")
    result = metric.evaluate(_snapshot())
    assert not result.passed
    assert result.score == 0.0


def test_memory_drift_metric_boundary_any_field_finds_fact_outside_its_declared_field():
    """fact_field='any' busca en todo el snapshot serializado -- un hecho
    que vive en 'summary' debe encontrarse aunque no se busque ahí
    explícitamente, caso límite del fallback más amplio."""
    metric = MemoryDriftMetric(fact="clínicas", fact_field="any")
    result = metric.evaluate(_snapshot())
    assert result.passed


def test_memory_drift_metric_technologies_field_joins_list_before_searching():
    metric = MemoryDriftMetric(fact="flutter", fact_field="technologies")
    result = metric.evaluate(_snapshot())
    assert result.passed
