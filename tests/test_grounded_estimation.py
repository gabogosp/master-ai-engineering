import re
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import structlog
from pydantic import ValidationError

from app.embedding_pipeline.retrieval import RetrievedChunk
from app.grounded_estimation import citations as citations_module
from app.grounded_estimation import service as service_module
from app.grounded_estimation.citations import verify_citations
from app.grounded_estimation.generator import (
    EstimateGenerationError,
    generate_estimate,
    to_prompt_chunks,
)
from app.grounded_estimation.schemas import (
    CitationReport,
    Estimate,
    EstimateLineItem,
    LineStatus,
    LineVerdict,
    SourceReference,
)


def _source(chunk_id: str = "1", document_id: str = "BUD-A") -> SourceReference:
    return SourceReference(chunk_id=chunk_id, document_id=document_id, evidence="Estimated hours: 100")


def _grounded_line(component: str = "Auth", hours: float = 100, chunk_ids=("1",)) -> EstimateLineItem:
    return EstimateLineItem(
        component=component,
        hours=hours,
        rationale="Derived from a past budget.",
        grounded=True,
        sources=[_source(cid) for cid in chunk_ids],
    )


def _ungrounded_line(component: str = "Reporting") -> EstimateLineItem:
    return EstimateLineItem(
        component=component,
        hours=0,
        rationale="Insufficient data: no chunk covers reporting.",
        grounded=False,
        sources=[],
    )


def _chunk(chunk_id: int, budget_id: str = "BUD-A") -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id=1,
        chunk_type="budget_component",
        content=f"Component {chunk_id}\nEstimated hours: 100",
        metadata={"budget_id": budget_id},
        distance=0.1,
    )


# --- schema integrity rules -------------------------------------------------


def test_grounded_line_without_sources_is_rejected():
    with pytest.raises(ValidationError, match="cites no source"):
        EstimateLineItem(component="Auth", hours=10, rationale="x", grounded=True, sources=[])


@pytest.mark.parametrize("hours", [0, -5])
def test_grounded_line_needs_positive_hours(hours):
    with pytest.raises(ValidationError, match="hours <= 0"):
        EstimateLineItem(
            component="Auth", hours=hours, rationale="x", grounded=True, sources=[_source()]
        )


def test_ungrounded_line_cannot_invent_hours():
    with pytest.raises(ValidationError, match="hours must be 0"):
        EstimateLineItem(
            component="Auth",
            hours=40,
            rationale="Insufficient data: guess",
            grounded=False,
            sources=[],
        )


def test_ungrounded_line_cannot_list_sources():
    with pytest.raises(ValidationError, match="sources=\\[\\]"):
        EstimateLineItem(
            component="Auth",
            hours=0,
            rationale="Insufficient data: x",
            grounded=False,
            sources=[_source()],
        )


def test_ungrounded_line_must_be_marked_as_insufficient_data():
    with pytest.raises(ValidationError, match="Insufficient data:"):
        EstimateLineItem(component="Auth", hours=0, rationale="no idea", grounded=False, sources=[])


def test_valid_ungrounded_line_is_accepted():
    line = _ungrounded_line()

    assert not line.grounded and line.hours == 0 and line.sources == []


def test_fractional_hours_are_allowed_on_grounded_lines():
    assert _grounded_line(hours=12.5).hours == 12.5


def test_total_ignores_ungrounded_lines():
    estimate = Estimate(
        summary="s", lines=[_grounded_line(hours=100), _grounded_line("B", 50), _ungrounded_line()]
    )
    assert estimate.grounded_hours == 150
    assert estimate.has_sufficient_context


def test_estimate_without_lines_has_no_hours_and_no_context():
    estimate = Estimate(summary="s", lines=[])

    assert estimate.grounded_hours == 0
    assert not estimate.has_sufficient_context


def test_estimate_with_only_ungrounded_lines_has_insufficient_context():
    estimate = Estimate(summary="s", lines=[_ungrounded_line("A"), _ungrounded_line("B")])

    assert estimate.grounded_hours == 0
    assert not estimate.has_sufficient_context


def test_schema_stays_inside_the_strict_mode_keyword_subset():
    """Length/range keywords would be sent to the Responses API strict mode,
    which only supports a subset of JSON Schema. Integrity rules must stay in
    validators."""
    forbidden = {"minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems", "pattern"}

    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    assert not forbidden & set(keys(Estimate.model_json_schema()))


def test_citation_report_counts_each_status():
    report = CitationReport(
        verdicts=[
            LineVerdict(component="a", status=LineStatus.GROUNDED, dangling_chunk_ids=[]),
            LineVerdict(component="b", status=LineStatus.GROUNDED, dangling_chunk_ids=[]),
            LineVerdict(component="c", status=LineStatus.DANGLING_CITATION, dangling_chunk_ids=["9"]),
            LineVerdict(component="d", status=LineStatus.INSUFFICIENT_DATA, dangling_chunk_ids=[]),
        ]
    )

    assert (report.grounded_count, report.dangling_count, report.insufficient_count) == (2, 1, 1)
    assert report.has_dangling_citations


# --- verify_citations -------------------------------------------------------


def test_verify_classifies_grounded_dangling_and_insufficient_lines():
    estimate = Estimate(
        summary="s",
        lines=[
            _grounded_line("Auth", chunk_ids=("1",)),
            _grounded_line("Ledger", chunk_ids=("999",)),
            _ungrounded_line("Reporting"),
        ],
    )

    report = verify_citations(estimate, {"1", "2"})

    assert [v.status for v in report.verdicts] == [
        LineStatus.GROUNDED,
        LineStatus.DANGLING_CITATION,
        LineStatus.INSUFFICIENT_DATA,
    ]
    assert report.verdicts[1].dangling_chunk_ids == ["999"]
    assert (report.grounded_count, report.dangling_count, report.insufficient_count) == (1, 1, 1)
    assert report.has_dangling_citations


def test_one_invented_source_makes_the_whole_line_dangling():
    estimate = Estimate(summary="s", lines=[_grounded_line(chunk_ids=("1", "404"))])

    report = verify_citations(estimate, {"1"})

    assert report.verdicts[0].status is LineStatus.DANGLING_CITATION
    assert report.verdicts[0].dangling_chunk_ids == ["404"]


def test_every_invented_source_of_a_line_is_listed_in_order():
    estimate = Estimate(summary="s", lines=[_grounded_line(chunk_ids=("7", "1", "8"))])

    report = verify_citations(estimate, {"1"})

    assert report.verdicts[0].dangling_chunk_ids == ["7", "8"]


def test_verdicts_follow_the_order_and_names_of_the_lines():
    estimate = Estimate(
        summary="s",
        lines=[_ungrounded_line("Z"), _grounded_line("A", chunk_ids=("1",)), _grounded_line("M", chunk_ids=("2",))],
    )

    report = verify_citations(estimate, {"1", "2"})

    assert [v.component for v in report.verdicts] == ["Z", "A", "M"]


def test_estimate_without_lines_gives_an_empty_report():
    report = verify_citations(Estimate(summary="s", lines=[]), set())

    assert report.verdicts == []
    assert not report.has_dangling_citations


def test_a_grounded_line_is_dangling_when_nothing_was_retrieved():
    estimate = Estimate(summary="s", lines=[_grounded_line(chunk_ids=("1",))])

    report = verify_citations(estimate, set())

    assert report.verdicts[0].status is LineStatus.DANGLING_CITATION


def test_ungrounded_lines_never_dangle_even_with_an_empty_context():
    estimate = Estimate(summary="s", lines=[_ungrounded_line()])

    report = verify_citations(estimate, set())

    assert report.verdicts[0].status is LineStatus.INSUFFICIENT_DATA


def test_near_miss_ids_are_not_accepted():
    estimate = Estimate(summary="s", lines=[_grounded_line(chunk_ids=("01",))])

    report = verify_citations(estimate, {"1"})

    assert report.verdicts[0].status is LineStatus.DANGLING_CITATION


def test_same_valid_chunk_cited_by_two_lines_is_fine():
    estimate = Estimate(
        summary="s",
        lines=[_grounded_line("A", chunk_ids=("1",)), _grounded_line("B", chunk_ids=("1",))],
    )

    assert verify_citations(estimate, {"1"}).grounded_count == 2


def _capturing_logger():
    cap = structlog.testing.LogCapture()
    logger = structlog.wrap_logger(
        None,
        processors=[structlog.contextvars.merge_contextvars, cap],
        wrapper_class=structlog.make_filtering_bound_logger(0),
    )
    return cap, logger


def test_verify_logs_are_correlated_with_request_id():
    cap, test_logger = _capturing_logger()
    estimate = Estimate(summary="s", lines=[_grounded_line(chunk_ids=("999",))])

    with patch.object(citations_module, "logger", test_logger):
        with structlog.contextvars.bound_contextvars(request_id="req-123"):
            verify_citations(estimate, {"1"})

    events = {entry["event"]: entry for entry in cap.entries}
    assert events["citation_report"]["request_id"] == "req-123"
    assert events["dangling_citation"]["request_id"] == "req-123"
    assert events["dangling_citation"]["log_level"] == "error"


def test_no_error_is_logged_when_every_citation_is_valid():
    cap, test_logger = _capturing_logger()
    estimate = Estimate(summary="s", lines=[_grounded_line(chunk_ids=("1",)), _ungrounded_line()])

    with patch.object(citations_module, "logger", test_logger):
        verify_citations(estimate, {"1"})

    assert [entry["event"] for entry in cap.entries] == ["citation_report"]
    assert cap.entries[0]["grounded"] == 1 and cap.entries[0]["insufficient"] == 1
    assert cap.entries[0]["dangling"] == 0


def test_one_error_log_per_dangling_line():
    cap, test_logger = _capturing_logger()
    estimate = Estimate(
        summary="s",
        lines=[_grounded_line("A", chunk_ids=("8",)), _grounded_line("B", chunk_ids=("9",))],
    )

    with patch.object(citations_module, "logger", test_logger):
        verify_citations(estimate, {"1"})

    errors = [e for e in cap.entries if e["event"] == "dangling_citation"]
    assert [e["component"] for e in errors] == ["A", "B"]


# --- generator --------------------------------------------------------------


class _FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(output_parsed=outcome)


def _fake_client(*outcomes):
    responses = _FakeResponses(outcomes)
    return SimpleNamespace(responses=responses), responses


def _validation_error() -> ValidationError:
    try:
        EstimateLineItem(component="A", hours=1, rationale="x", grounded=True, sources=[])
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


def test_prompt_chunks_expose_string_chunk_id_and_budget_document_id():
    prompt_chunks = to_prompt_chunks([_chunk(7, "BUD-Z")])

    assert prompt_chunks == [
        {"chunk_id": "7", "document_id": "BUD-Z", "content": _chunk(7, "BUD-Z").content}
    ]


def test_prompt_chunks_keep_the_retrieval_order():
    prompt_chunks = to_prompt_chunks([_chunk(3), _chunk(1), _chunk(2)])

    assert [c["chunk_id"] for c in prompt_chunks] == ["3", "1", "2"]


async def test_generator_puts_chunk_ids_in_the_prompt():
    good = Estimate(summary="s", lines=[_grounded_line()])
    client, responses = _fake_client(good)

    result = await generate_estimate(client, query="q", chunks=[_chunk(7)], model="m")

    assert result is good
    user_message = responses.calls[0]["input"][1]["content"]
    assert 'chunk_id="7"' in user_message
    assert responses.calls[0]["text_format"] is Estimate


async def test_generator_sends_the_model_and_a_deterministic_temperature():
    client, responses = _fake_client(Estimate(summary="s", lines=[]))

    await generate_estimate(client, query="q", chunks=[_chunk(1)], model="gpt-test")

    assert responses.calls[0]["model"] == "gpt-test"
    assert responses.calls[0]["temperature"] == 0
    assert [m["role"] for m in responses.calls[0]["input"]] == ["system", "user"]


async def test_generator_appends_the_correction_before_the_first_attempt():
    client, responses = _fake_client(Estimate(summary="s", lines=[]))

    await generate_estimate(
        client, query="q", chunks=[_chunk(1)], model="m", correction="use only valid ids"
    )

    last_message = responses.calls[0]["input"][-1]
    assert last_message == {"role": "user", "content": "use only valid ids"}


async def test_generator_retries_with_feedback_after_a_schema_violation():
    good = Estimate(summary="s", lines=[_grounded_line()])
    client, responses = _fake_client(_validation_error(), good)

    result = await generate_estimate(client, query="q", chunks=[_chunk(1)], model="m")

    assert result is good
    assert len(responses.calls) == 2
    assert "rejected" in responses.calls[1]["input"][-1]["content"]
    assert "cites no source" in responses.calls[1]["input"][-1]["content"]


async def test_generator_retries_when_the_model_returns_nothing_parsed():
    good = Estimate(summary="s", lines=[_grounded_line()])
    client, responses = _fake_client(None, good)

    result = await generate_estimate(client, query="q", chunks=[_chunk(1)], model="m")

    assert result is good
    assert len(responses.calls) == 2


async def test_generator_gives_up_after_max_attempts():
    client, responses = _fake_client(_validation_error(), _validation_error())

    with pytest.raises(EstimateGenerationError):
        await generate_estimate(client, query="q", chunks=[_chunk(1)], model="m", max_attempts=2)

    assert len(responses.calls) == 2


async def test_generator_gives_up_when_the_model_keeps_returning_nothing():
    client, responses = _fake_client(None, None, None)

    with pytest.raises(EstimateGenerationError, match="3 attempts"):
        await generate_estimate(client, query="q", chunks=[_chunk(1)], model="m")

    assert len(responses.calls) == 3


async def test_generator_does_not_swallow_api_errors():
    client, _ = _fake_client(RuntimeError("network down"))

    with pytest.raises(RuntimeError, match="network down"):
        await generate_estimate(client, query="q", chunks=[_chunk(1)], model="m")


# --- service ----------------------------------------------------------------


class _FakeEmbedder:
    def embed_one(self, text):
        return [0.0] * 1536


@contextmanager
def _patched_service(chunks, generations, client=object()):
    """Replace the service boundaries (embedder, retrieval, LLM) and record
    what the service passed to each."""
    seen = SimpleNamespace(retrieve_kwargs=[], generate_kwargs=[])

    async def fake_retrieve(session, **kwargs):
        seen.retrieve_kwargs.append(kwargs)
        return chunks, {}

    async def fake_generate(client_arg, **kwargs):
        seen.generate_kwargs.append(kwargs)
        return generations.pop(0)

    with (
        patch.object(service_module, "OpenAIEmbedder", _FakeEmbedder),
        patch.object(service_module, "retrieve", fake_retrieve),
        patch.object(service_module, "generate_estimate", fake_generate),
        patch.object(service_module, "get_openai_client", lambda: client),
    ):
        yield seen


async def test_service_retries_once_when_a_citation_dangles_and_reports_the_fix():
    bad = Estimate(summary="s", lines=[_grounded_line("Auth", 100, ("999",))])
    fixed = Estimate(summary="s", lines=[_grounded_line("Auth", 100, ("1",))])

    with _patched_service([_chunk(1)], [bad, fixed]) as seen:
        result = await service_module.build_grounded_estimate(None, query="q", request_id="r1")

    assert result.corrected_after_dangling
    assert result.citation_report.dangling_count == 0
    assert result.verified_hours == 100
    assert "999" in seen.generate_kwargs[1]["correction"]


async def test_service_does_not_retry_when_every_citation_is_valid():
    good = Estimate(summary="s", lines=[_grounded_line("Auth", 100, ("1",))])

    with _patched_service([_chunk(1)], [good]) as seen:
        result = await service_module.build_grounded_estimate(None, query="q")

    assert len(seen.generate_kwargs) == 1
    assert "correction" not in seen.generate_kwargs[0]
    assert not result.corrected_after_dangling


async def test_service_retries_at_most_once_even_if_the_citation_still_dangles():
    bad = Estimate(summary="s", lines=[_grounded_line("Auth", 100, ("999",))])

    with _patched_service([_chunk(1)], [bad, bad, bad]) as seen:
        result = await service_module.build_grounded_estimate(None, query="q")

    assert len(seen.generate_kwargs) == 2
    assert result.citation_report.has_dangling_citations
    assert result.corrected_after_dangling


async def test_correction_lists_the_invalid_and_the_valid_ids():
    bad = Estimate(summary="s", lines=[_grounded_line("Auth", 100, ("999",))])
    fixed = Estimate(summary="s", lines=[_grounded_line("Auth", 100, ("2",))])

    with _patched_service([_chunk(2), _chunk(1)], [bad, fixed]) as seen:
        await service_module.build_grounded_estimate(None, query="q")

    correction = seen.generate_kwargs[1]["correction"]
    assert "['999']" in correction
    assert "['1', '2']" in correction


async def test_verified_hours_exclude_lines_whose_citation_still_dangles():
    bad = Estimate(
        summary="s",
        lines=[_grounded_line("Auth", 100, ("1",)), _grounded_line("Ledger", 60, ("999",))],
    )

    with _patched_service([_chunk(1)], [bad, bad]):
        result = await service_module.build_grounded_estimate(None, query="q")

    assert result.estimate.grounded_hours == 160
    assert result.verified_hours == 100
    assert result.citation_report.has_dangling_citations


async def test_verified_hours_equal_claimed_hours_when_nothing_dangles():
    good = Estimate(
        summary="s",
        lines=[_grounded_line("A", 100, ("1",)), _grounded_line("B", 40.5, ("2",)), _ungrounded_line()],
    )

    with _patched_service([_chunk(1), _chunk(2)], [good]):
        result = await service_module.build_grounded_estimate(None, query="q")

    assert result.verified_hours == result.estimate.grounded_hours == 140.5


async def test_service_skips_the_llm_when_nothing_was_retrieved():
    with _patched_service([], []) as seen:
        result = await service_module.build_grounded_estimate(None, query="q")

    assert seen.generate_kwargs == []
    assert result.estimate.lines == []
    assert result.estimate.summary.startswith("Insufficient context")
    assert not result.estimate.has_sufficient_context
    assert result.verified_hours == 0
    assert result.chunks == []


async def test_service_forwards_the_retrieval_parameters():
    good = Estimate(summary="s", lines=[_grounded_line("A", 10, ("1",))])

    with _patched_service([_chunk(1)], [good]) as seen:
        await service_module.build_grounded_estimate(
            None, query="my query", k=3, search_mode="vector", rerank=True
        )

    kwargs = seen.retrieve_kwargs[0]
    assert kwargs["query_text"] == "my query"
    assert (kwargs["k"], kwargs["search_mode"], kwargs["rerank"]) == (3, "vector", True)
    assert len(kwargs["query_vector"]) == 1536


async def test_service_uses_hybrid_without_rerank_by_default():
    good = Estimate(summary="s", lines=[_grounded_line("A", 10, ("1",))])

    with _patched_service([_chunk(1)], [good]) as seen:
        await service_module.build_grounded_estimate(None, query="q")

    kwargs = seen.retrieve_kwargs[0]
    assert (kwargs["k"], kwargs["search_mode"], kwargs["rerank"]) == (5, "hybrid", False)


async def test_service_passes_the_retrieved_chunks_and_configured_model_to_the_generator():
    good = Estimate(summary="s", lines=[_grounded_line("A", 10, ("1",))])
    chunks = [_chunk(1), _chunk(2)]

    with _patched_service(chunks, [good]) as seen:
        with patch.object(service_module.settings, "grounded_generation_model", "model-x"):
            await service_module.build_grounded_estimate(None, query="q")

    assert seen.generate_kwargs[0]["chunks"] == chunks
    assert seen.generate_kwargs[0]["model"] == "model-x"


async def test_service_keeps_the_given_request_id_and_generates_one_otherwise():
    good = Estimate(summary="s", lines=[_grounded_line("A", 10, ("1",))])

    with _patched_service([_chunk(1)], [good, good]):
        given = await service_module.build_grounded_estimate(None, query="q", request_id="abc")
        generated = await service_module.build_grounded_estimate(None, query="q")

    assert given.request_id == "abc"
    assert re.fullmatch(r"[0-9a-f]{32}", generated.request_id)


async def test_request_id_does_not_leak_into_later_logs():
    good = Estimate(summary="s", lines=[_grounded_line("A", 10, ("1",))])

    with _patched_service([_chunk(1)], [good]):
        await service_module.build_grounded_estimate(None, query="q", request_id="abc")

    assert "request_id" not in structlog.contextvars.get_contextvars()


async def test_service_fails_loudly_without_an_openai_client():
    with _patched_service([_chunk(1)], [], client=None):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            await service_module.build_grounded_estimate(None, query="q")
