from types import SimpleNamespace
from unittest.mock import patch

import pytest
import structlog
from pydantic import ValidationError

from app.embedding_pipeline.retrieval import RetrievedChunk
from app.grounded_estimation import citations as citations_module
from app.grounded_estimation.citations import verify_citations
from app.grounded_estimation.generator import (
    EstimateGenerationError,
    generate_estimate,
    to_prompt_chunks,
)
from app.grounded_estimation.schemas import (
    Estimate,
    EstimateLineItem,
    LineStatus,
    SourceReference,
)
from app.grounded_estimation import service as service_module


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
        EstimateLineItem(
            component="Auth", hours=10, rationale="x", grounded=True, sources=[]
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
        EstimateLineItem(
            component="Auth", hours=0, rationale="no idea", grounded=False, sources=[]
        )


def test_total_ignores_ungrounded_lines():
    estimate = Estimate(
        summary="s", lines=[_grounded_line(hours=100), _grounded_line("B", 50), _ungrounded_line()]
    )
    assert estimate.grounded_hours == 150
    assert estimate.has_sufficient_context


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


def test_verify_logs_are_correlated_with_request_id():
    cap = structlog.testing.LogCapture()
    test_logger = structlog.wrap_logger(
        None,
        processors=[structlog.contextvars.merge_contextvars, cap],
        wrapper_class=structlog.make_filtering_bound_logger(0),
    )
    estimate = Estimate(summary="s", lines=[_grounded_line(chunk_ids=("999",))])

    with patch.object(citations_module, "logger", test_logger):
        with structlog.contextvars.bound_contextvars(request_id="req-123"):
            verify_citations(estimate, {"1"})

    events = {entry["event"]: entry for entry in cap.entries}
    assert events["citation_report"]["request_id"] == "req-123"
    assert events["dangling_citation"]["request_id"] == "req-123"
    assert events["dangling_citation"]["log_level"] == "error"


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

    assert prompt_chunks[0]["chunk_id"] == "7"
    assert prompt_chunks[0]["document_id"] == "BUD-Z"


async def test_generator_puts_chunk_ids_in_the_prompt():
    good = Estimate(summary="s", lines=[_grounded_line()])
    client, responses = _fake_client(good)

    result = await generate_estimate(client, query="q", chunks=[_chunk(7)], model="m")

    assert result is good
    user_message = responses.calls[0]["input"][1]["content"]
    assert 'chunk_id="7"' in user_message
    assert responses.calls[0]["text_format"] is Estimate


async def test_generator_retries_with_feedback_after_a_schema_violation():
    good = Estimate(summary="s", lines=[_grounded_line()])
    client, responses = _fake_client(_validation_error(), good)

    result = await generate_estimate(client, query="q", chunks=[_chunk(1)], model="m")

    assert result is good
    assert len(responses.calls) == 2
    assert "rejected" in responses.calls[1]["input"][-1]["content"]


async def test_generator_gives_up_after_max_attempts():
    client, _ = _fake_client(_validation_error(), _validation_error())

    with pytest.raises(EstimateGenerationError):
        await generate_estimate(
            client, query="q", chunks=[_chunk(1)], model="m", max_attempts=2
        )


# --- service ----------------------------------------------------------------


class _FakeEmbedder:
    def embed_one(self, text):
        return [0.0] * 1536


def _patch_service(chunks, client, generations):
    async def fake_retrieve(session, **kwargs):
        return chunks, {}

    calls = []

    async def fake_generate(client_arg, **kwargs):
        calls.append(kwargs)
        return generations.pop(0)

    return calls, (
        patch.object(service_module, "OpenAIEmbedder", _FakeEmbedder),
        patch.object(service_module, "retrieve", fake_retrieve),
        patch.object(service_module, "generate_estimate", fake_generate),
        patch.object(service_module, "get_openai_client", lambda: client),
    )


async def test_service_retries_once_when_a_citation_dangles_and_reports_the_fix():
    bad = Estimate(summary="s", lines=[_grounded_line("Auth", 100, ("999",))])
    fixed = Estimate(summary="s", lines=[_grounded_line("Auth", 100, ("1",))])
    calls, patches = _patch_service([_chunk(1)], object(), [bad, fixed])

    with patches[0], patches[1], patches[2], patches[3]:
        result = await service_module.build_grounded_estimate(None, query="q", request_id="r1")

    assert result.corrected_after_dangling
    assert result.citation_report.dangling_count == 0
    assert result.verified_hours == 100
    assert "999" in calls[1]["correction"]


async def test_verified_hours_exclude_lines_whose_citation_still_dangles():
    bad = Estimate(
        summary="s",
        lines=[_grounded_line("Auth", 100, ("1",)), _grounded_line("Ledger", 60, ("999",))],
    )
    calls, patches = _patch_service([_chunk(1)], object(), [bad, bad])

    with patches[0], patches[1], patches[2], patches[3]:
        result = await service_module.build_grounded_estimate(None, query="q")

    assert result.estimate.grounded_hours == 160
    assert result.verified_hours == 100
    assert result.citation_report.has_dangling_citations


async def test_service_skips_the_llm_when_nothing_was_retrieved():
    calls, patches = _patch_service([], object(), [])

    with patches[0], patches[1], patches[2], patches[3]:
        result = await service_module.build_grounded_estimate(None, query="q")

    assert calls == []
    assert not result.estimate.has_sufficient_context
    assert result.verified_hours == 0
