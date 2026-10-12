import pytest

from evals.stress.report import _completion_stats, _failure_hotspot, _misses, build_report


def _row(**overrides) -> dict:
    defaults = dict(
        scenario="growing",
        attachment_kb=0,
        repeat=0,
        turn_index=1,
        session_id="sess-1",
        enriched_transcript_chars=100,
        attachments_total_chars=0,
        messages_in_window=2,
        anchors_count=0,
        summary_chars=0,
        tokens_in=1000,
        tokens_out=200,
        cost_usd=0.001,
        cost_usd_cumulative=0.001,
        latency_ms=2000.0,
        cache_hit_kind="none",
        last_resolved_tier="default",
        latency_budget_pass=True,
        cost_budget_pass=True,
        memory_drift_recall=1.0,
        memory_drift_pass=True,
    )
    defaults.update(overrides)
    return defaults


def test_misses_excludes_exact_cache_hits():
    rows = [_row(cache_hit_kind="exact"), _row(cache_hit_kind="none")]
    assert len(_misses(rows)) == 1


def test_completion_stats_counts_sessions_reaching_turn_20():
    rows = [
        _row(session_id="a", turn_index=20),
        _row(session_id="b", turn_index=5),  # se cortó antes
    ]
    total, complete, rate = _completion_stats(rows)
    assert total == 2
    assert complete == 1
    assert rate == 0.5


def test_completion_stats_boundary_empty_rows_do_not_divide_by_zero():
    total, complete, rate = _completion_stats([])
    assert (total, complete, rate) == (0, 0, 0.0)


def test_completion_stats_total_override_covers_conversations_with_zero_rows():
    """Una conversación que falla en su primer turno no deja ninguna fila
    en el CSV -- el total derivado de las filas la subcuenta. El
    override existe exactamente para ese caso."""
    rows = [_row(session_id="a", turn_index=20)]
    total, complete, rate = _completion_stats(rows, total_conversations=3)
    assert total == 3
    assert complete == 1
    assert rate == pytest.approx(1 / 3)


def test_build_report_does_not_crash_on_a_single_row():
    report = build_report([_row()])
    assert "# Stress-test del CAG" in report
    assert "growing" in report


def test_failure_hotspot_finds_the_turn_where_most_broken_conversations_stop():
    rows = [
        _row(session_id="a", scenario="contradiction", turn_index=7),
        _row(session_id="b", scenario="contradiction", turn_index=7),
        _row(session_id="c", scenario="contradiction", turn_index=7),
        _row(session_id="d", scenario="growing", turn_index=3),
    ]
    scenario, turn, count = _failure_hotspot(rows)
    assert (scenario, turn, count) == ("contradiction", 8, 3)


def test_failure_hotspot_is_none_when_everyone_completes():
    rows = [_row(session_id="a", turn_index=20), _row(session_id="b", turn_index=20)]
    assert _failure_hotspot(rows) is None


def test_report_flags_when_anchors_never_fired():
    report = build_report([_row(anchors_count=0)])
    assert "anchors_count` fue 0" in report


def test_report_omits_anchor_caveat_when_anchors_did_fire():
    report = build_report([_row(anchors_count=1)])
    assert "anchors_count` fue 0" not in report
