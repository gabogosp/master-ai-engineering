from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIError

import app.grounded_estimation.router as router_module
import app.grounded_estimation.service as service_module
from app.db import get_session
from app.embedding_pipeline.retrieval import RetrievedChunk
from app.grounded_estimation.generator import EstimateGenerationError
from app.grounded_estimation.schemas import Estimate, EstimateLineItem, SourceReference
from app.grounded_estimation.service import GroundedEstimateResult
from app.grounded_estimation.citations import verify_citations
from app.main import app

URL = "/estimate/grounded"
VALID_BODY = {"query": "Real-time payment gateway with fraud detection"}


def _chunk(chunk_id: int, budget_id: str = "BUD-A") -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id=1,
        chunk_type="budget_component",
        content=f"Component {chunk_id}\nEstimated hours: 100",
        metadata={"budget_id": budget_id},
        distance=0.1,
    )


def _grounded(component: str, hours: float, chunk_id: str, budget_id: str = "BUD-A"):
    return EstimateLineItem(
        component=component,
        hours=hours,
        rationale="Derived from a past budget.",
        grounded=True,
        sources=[
            SourceReference(chunk_id=chunk_id, document_id=budget_id, evidence="Estimated hours: 100")
        ],
    )


def _ungrounded(component: str):
    return EstimateLineItem(
        component=component,
        hours=0,
        rationale="Insufficient data: nothing covers it.",
        grounded=False,
        sources=[],
    )


def _result(estimate: Estimate, chunks, **overrides) -> GroundedEstimateResult:
    ids = {str(chunk.id) for chunk in chunks}
    report = verify_citations(estimate, ids)
    verified = sum(
        line.hours
        for line, verdict in zip(estimate.lines, report.verdicts, strict=True)
        if verdict.status.value == "grounded"
    )
    values = dict(
        request_id="req-1",
        estimate=estimate,
        citation_report=report,
        chunks=list(chunks),
        verified_hours=verified,
        corrected_after_dangling=False,
    )
    values.update(overrides)
    return GroundedEstimateResult(**values)


@pytest.fixture(autouse=True)
def _override_db_session():
    async def override():
        yield object()

    app.dependency_overrides[get_session] = override
    yield
    app.dependency_overrides.pop(get_session, None)


def _post(client, body=None, mocked: AsyncMock | None = None):
    mocked = mocked or AsyncMock(return_value=_result(Estimate(summary="s", lines=[]), []))
    with patch.object(router_module, "build_grounded_estimate", mocked):
        return client.post(URL, json=VALID_BODY if body is None else body), mocked


# --- 200: shape -------------------------------------------------------------


def test_returns_the_estimate_with_sources_report_and_retrieved_chunks(client):
    estimate = Estimate(
        summary="Good coverage.",
        lines=[_grounded("Gateway", 220, "9", "BUD-3"), _ungrounded("Reporting")],
    )
    mocked = AsyncMock(return_value=_result(estimate, [_chunk(9, "BUD-3"), _chunk(10, "BUD-3")]))

    response, _ = _post(client, mocked=mocked)

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-1"
    assert body["claimed_hours"] == 220 and body["verified_hours"] == 220
    assert body["corrected_after_dangling"] is False

    gateway, reporting = body["estimate"]["lines"]
    assert gateway["grounded"] is True
    assert gateway["sources"] == [
        {"chunk_id": "9", "document_id": "BUD-3", "evidence": "Estimated hours: 100"}
    ]
    assert reporting["grounded"] is False and reporting["hours"] == 0 and reporting["sources"] == []

    assert [v["status"] for v in body["citation_report"]["verdicts"]] == [
        "grounded",
        "insufficient_data",
    ]
    assert [(c["chunk_id"], c["document_id"]) for c in body["retrieved_chunks"]] == [
        ("9", "BUD-3"),
        ("10", "BUD-3"),
    ]
    assert "Estimated hours: 100" in body["retrieved_chunks"][0]["content"]


def test_every_cited_chunk_id_is_among_the_returned_chunks(client):
    estimate = Estimate(summary="s", lines=[_grounded("A", 10, "9"), _grounded("B", 20, "10")])
    mocked = AsyncMock(return_value=_result(estimate, [_chunk(9), _chunk(10)]))

    response, _ = _post(client, mocked=mocked)

    body = response.json()
    returned = {c["chunk_id"] for c in body["retrieved_chunks"]}
    cited = {s["chunk_id"] for line in body["estimate"]["lines"] for s in line["sources"]}
    assert cited <= returned


def test_a_dangling_citation_is_visible_in_the_response(client):
    estimate = Estimate(
        summary="s", lines=[_grounded("Auth", 100, "1"), _grounded("Ledger", 60, "999")]
    )
    mocked = AsyncMock(return_value=_result(estimate, [_chunk(1)], corrected_after_dangling=True))

    response, _ = _post(client, mocked=mocked)

    body = response.json()
    assert body["claimed_hours"] == 160
    assert body["verified_hours"] == 100
    assert body["corrected_after_dangling"] is True
    ledger = body["citation_report"]["verdicts"][1]
    assert ledger["status"] == "dangling_citation"
    assert ledger["dangling_chunk_ids"] == ["999"]


def test_empty_retrieval_returns_an_empty_estimate(client):
    empty = Estimate(summary="Insufficient context: nothing was retrieved.", lines=[])
    mocked = AsyncMock(return_value=_result(empty, []))

    response, _ = _post(client, mocked=mocked)

    body = response.json()
    assert response.status_code == 200
    assert body["estimate"]["lines"] == [] and body["retrieved_chunks"] == []
    assert body["claimed_hours"] == 0 and body["verified_hours"] == 0


# --- request forwarding -----------------------------------------------------


def test_defaults_are_k5_hybrid_without_rerank(client):
    _, mocked = _post(client)

    kwargs = mocked.await_args.kwargs
    assert kwargs["query"] == VALID_BODY["query"]
    assert (kwargs["k"], kwargs["search_mode"], kwargs["rerank"]) == (5, "hybrid", False)


def test_custom_parameters_are_forwarded(client):
    _, mocked = _post(
        client, {"query": "q", "k": 3, "search_mode": "vector", "rerank": True}
    )

    kwargs = mocked.await_args.kwargs
    assert (kwargs["k"], kwargs["search_mode"], kwargs["rerank"]) == (3, "vector", True)


# --- 422: input validation --------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": ""},
        {"query": "q", "k": 0},
        {"query": "q", "k": -1},
        {"query": "q", "k": "many"},
        {"query": "q", "search_mode": "lexical"},
        {"query": "q", "rerank": "maybe"},
        {"k": 5},
    ],
)
def test_invalid_input_is_rejected_before_any_work_happens(client, body):
    response, mocked = _post(client, body)

    assert response.status_code == 422
    mocked.assert_not_awaited()


# --- 500 / 502: failures ----------------------------------------------------


def test_misconfiguration_returns_500_without_leaking_the_cause(client):
    mocked = AsyncMock(side_effect=RuntimeError("OPENAI_API_KEY is required: sk-secret"))

    response, _ = _post(client, mocked=mocked)

    assert response.status_code == 500
    assert "sk-secret" not in response.text
    assert response.json()["detail"] == "Grounded estimation is misconfigured."


def test_model_that_never_complies_returns_502(client):
    mocked = AsyncMock(side_effect=EstimateGenerationError("no valid estimate after 3 attempts"))

    response, _ = _post(client, mocked=mocked)

    assert response.status_code == 502
    assert "attempts" not in response.text


def test_openai_failure_returns_502(client):
    boom = APIError(message="boom", request=httpx.Request("POST", "https://api.openai.com"), body=None)

    response, _ = _post(client, mocked=AsyncMock(side_effect=boom))

    assert response.status_code == 502
    assert response.json()["detail"] == "Could not generate a valid estimate."


# --- wiring: real router + real service, only the outside world faked ------


class _FakeEmbedder:
    def embed_one(self, text):
        return [0.0] * 1536


def _wired(generations, chunks):
    async def fake_retrieve(session, **kwargs):
        return chunks, {}

    async def fake_generate(client, **kwargs):
        return generations.pop(0)

    return (
        patch.object(service_module, "OpenAIEmbedder", _FakeEmbedder),
        patch.object(service_module, "retrieve", fake_retrieve),
        patch.object(service_module, "generate_estimate", fake_generate),
        patch.object(service_module, "get_openai_client", lambda: object()),
    )


def test_full_request_through_the_real_service_with_a_dangling_retry(client):
    bad = Estimate(summary="s", lines=[_grounded("Auth", 100, "999", "BUD-A")])
    fixed = Estimate(summary="s", lines=[_grounded("Auth", 100, "1", "BUD-A")])
    patches = _wired([bad, fixed], [_chunk(1, "BUD-A")])

    with patches[0], patches[1], patches[2], patches[3]:
        response = client.post(URL, json=VALID_BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["corrected_after_dangling"] is True
    assert body["citation_report"]["verdicts"][0]["status"] == "grounded"
    assert body["verified_hours"] == body["claimed_hours"] == 100
    assert body["retrieved_chunks"][0]["document_id"] == "BUD-A"
    assert len(body["request_id"]) == 32


def test_full_request_keeps_the_dangling_line_visible_when_the_retry_does_not_fix_it(client):
    bad = Estimate(summary="s", lines=[_grounded("Auth", 100, "999")])
    patches = _wired([bad, bad], [_chunk(1)])

    with patches[0], patches[1], patches[2], patches[3]:
        response = client.post(URL, json=VALID_BODY)

    body = response.json()
    assert response.status_code == 200
    assert body["citation_report"]["verdicts"][0]["status"] == "dangling_citation"
    assert body["verified_hours"] == 0 and body["claimed_hours"] == 100


# --- routing ----------------------------------------------------------------


def test_the_new_route_is_registered_without_displacing_the_existing_ones(client):
    paths = client.get("/openapi.json").json()["paths"]

    assert "post" in paths[URL]
    assert "/search" in paths and "/embeddings/ingest" in paths
    assert "/api/v1/estimate" in paths


def test_only_post_is_allowed(client):
    assert client.get(URL).status_code == 405
