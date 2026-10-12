from evals.stress.report import _completion_stats, _misses, build_report


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


def test_build_report_does_not_crash_on_a_single_row():
    report = build_report([_row()])
    assert "# Stress-test del CAG" in report
    assert "growing" in report
