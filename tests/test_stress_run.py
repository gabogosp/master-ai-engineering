import pytest

from app.schemas import TurnObservation
from evals.stress.run import FIXTURES_DIR, _attachment_path_for, _build_row, _percentile


def _observation(**overrides) -> TurnObservation:
    defaults = dict(
        turn_index=3,
        session_id="sess-1",
        enriched_transcript_chars=100,
        attachments_total_chars=0,
        messages_in_window=4,
        anchors_count=0,
        summary_chars=0,
        tokens_in=50,
        tokens_out=50,
        cost_usd=0.005,
        latency_ms=1000.0,
    )
    defaults.update(overrides)
    return TurnObservation(**defaults)


# --- _attachment_path_for ------------------------------------------------


def test_attachment_path_for_zero_kb_is_no_attachment():
    assert _attachment_path_for(0) is None


def test_attachment_path_for_known_size_points_at_fixture():
    path = _attachment_path_for(20)
    assert path == FIXTURES_DIR / "attach_20kb.pdf"


def test_attachment_path_for_missing_fixture_generates_it_on_demand():
    """Un tamaño sin PDF en disco no debe fallar: debe generarlo ahí
    mismo (son determinísticos, ver build_pdfs.py), no obligar al
    usuario a correr otro comando a mano."""
    path = FIXTURES_DIR / "attach_1kb.pdf"
    path.unlink(missing_ok=True)
    try:
        result = _attachment_path_for(1)
        assert result == path
        assert path.exists()
    finally:
        path.unlink(missing_ok=True)


# --- _percentile ----------------------------------------------------------


def test_percentile_p50_of_odd_list_is_the_middle_value():
    assert _percentile([1.0, 2.0, 3.0], 50) == 2.0


def test_percentile_p95_of_uniform_list_is_near_the_top():
    values = [float(x) for x in range(1, 101)]  # 1..100
    assert _percentile(values, 95) == pytest.approx(95.05, abs=0.5)


def test_percentile_empty_list_returns_zero():
    assert _percentile([], 50) == 0.0


# --- _build_row -------------------------------------------------------------


def test_build_row_passes_both_budgets_and_full_recall_when_everything_is_fine():
    snapshot = {"metadata": {"project_name": "MediFlow"}}
    row = _build_row(
        scenario_name="growing",
        attachment_kb=0,
        repeat=0,
        observation=_observation(latency_ms=500.0, cost_usd=0.001),
        cost_usd_cumulative=0.001,
        latency_budget_ms=8000.0,
        cost_budget_usd=0.01,
        tracked_facts=[("MediFlow", "project_name")],
        snapshot=snapshot,
    )
    assert row["latency_budget_pass"] is True
    assert row["cost_budget_pass"] is True
    assert row["memory_drift_recall"] == 1.0
    assert row["memory_drift_pass"] is True
    assert row["turn_index"] == 3


def test_build_row_fails_latency_budget_when_over():
    row = _build_row(
        scenario_name="growing",
        attachment_kb=0,
        repeat=0,
        observation=_observation(latency_ms=15000.0),
        cost_usd_cumulative=0.005,
        latency_budget_ms=8000.0,
        cost_budget_usd=0.01,
        tracked_facts=[],
        snapshot={"metadata": {}},
    )
    assert row["latency_budget_pass"] is False
    # Sin hechos trackeados todavía: recall vacío cuenta como 1.0 (nada
    # que perder) -- no debe confundirse con un drift real.
    assert row["memory_drift_recall"] == 1.0


def test_build_row_boundary_partial_recall_when_one_of_two_facts_is_gone():
    snapshot = {"metadata": {"project_name": "FitPulse", "mentioned_technologies": ["Flutter"]}}
    row = _build_row(
        scenario_name="pivot",
        attachment_kb=5,
        repeat=1,
        observation=_observation(),
        cost_usd_cumulative=0.02,
        latency_budget_ms=8000.0,
        cost_budget_usd=0.01,
        tracked_facts=[("FitPulse", "project_name"), ("React Native", "technologies")],
        snapshot=snapshot,
    )
    assert row["memory_drift_recall"] == 0.5
    assert row["memory_drift_pass"] is False
