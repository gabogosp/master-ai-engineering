import json
from unittest.mock import AsyncMock

import pytest

from app.agent.schemas import (
    CalculateEstimateArgs,
    ComponentReferences,
    ComponentToValidate,
    SearchBudgetsArgs,
    ValidateEstimateArgs,
)
from app.agent.tools import (
    AGENT_TOOLS,
    CONTINGENCY_FACTOR,
    AgentToolbox,
    calculate_estimate,
    search_budgets,
    validate_estimate,
)
from app.embedding_pipeline.retrieval import RetrievedChunk

# --- schemas ----------------------------------------------------------------


@pytest.mark.parametrize("tool", AGENT_TOOLS, ids=lambda t: t["name"])
def test_tool_schema_is_flat_and_strict(tool):
    assert tool["type"] == "function" and tool["strict"] is True
    assert "function" not in tool  # Responses shape, not Chat Completions
    assert tool["description"].strip()


def _objects(schema):
    if isinstance(schema, dict):
        if "properties" in schema:
            yield schema
        for value in schema.values():
            yield from _objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _objects(value)


@pytest.mark.parametrize("tool", AGENT_TOOLS, ids=lambda t: t["name"])
def test_every_object_in_the_schema_satisfies_strict_mode(tool):
    for obj in _objects(tool["parameters"]):
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])


def test_the_three_tools_are_exposed():
    assert [t["name"] for t in AGENT_TOOLS] == ["search_budgets", "calculate_estimate", "validate_estimate"]


# --- calculate_estimate -----------------------------------------------------


def _calc(*components):
    return calculate_estimate(
        CalculateEstimateArgs(components=[ComponentReferences(name=n, reference_amounts=r) for n, r in components])
    )


def test_calculate_uses_the_median_plus_contingency():
    result = _calc(("Auth", [100, 200, 600]))
    assert result["components"][0]["estimated_hours"] == round(200 * (1 + CONTINGENCY_FACTOR), 1)
    assert result["components"][0]["reference_count"] == 3


def test_calculate_totals_the_components():
    result = _calc(("A", [100]), ("B", [200]))
    assert result["total_hours"] == 115.0 + 230.0


def test_component_without_references_costs_zero_and_is_flagged():
    result = _calc(("A", [100]), ("SAP", []))
    sap = result["components"][1]
    assert sap["estimated_hours"] == 0.0 and sap["unbudgeted"] is True
    assert result["total_hours"] == 115.0
    assert "SAP" in result["summary"]


# --- validate_estimate ------------------------------------------------------


def _validate(components, total):
    return validate_estimate(
        ValidateEstimateArgs(
            components=[
                ComponentToValidate(name=n, estimated_hours=h, reference_amounts=r) for n, h, r in components
            ],
            total_hours=total,
        )
    )


def _codes(result):
    return [i["code"] for i in result["issues"]]


def test_a_coherent_estimate_is_valid():
    result = _validate([("A", 115.0, [100]), ("B", 230.0, [200])], 345.0)
    assert result["valid"] is True and result["issues"] == []


def test_flags_a_component_without_reference():
    assert _codes(_validate([("A", 0.0, [])], 0.0)) == ["no_reference"]


def test_flags_hours_that_no_reference_backs():
    assert _codes(_validate([("A", 50.0, [])], 50.0)) == ["hours_without_reference"]


def test_flags_non_positive_hours_despite_references():
    assert "non_positive_hours" in _codes(_validate([("A", 0.0, [100])], 0.0))


def test_flags_hours_outside_the_reference_range():
    assert "outside_reference_range" in _codes(_validate([("A", 900.0, [100, 120])], 900.0))
    assert "outside_reference_range" in _codes(_validate([("A", 10.0, [100, 120])], 10.0))


def test_flags_an_implausibly_large_component_and_total():
    codes = _codes(_validate([("A", 5000.0, [4800])], 13000.0))
    assert "implausible_component" in codes and "implausible_total" in codes


def test_flags_a_total_that_does_not_match_the_components():
    result = _validate([("A", 115.0, [100])], 300.0)
    assert result["valid"] is False and _codes(result) == ["total_mismatch"]


def test_tolerates_rounding_in_the_total():
    assert _validate([("A", 115.0, [100])], 115.3)["valid"] is True


# --- search_budgets ---------------------------------------------------------


def _chunk(chunk_id, budget_id, sector, year, hours, name="Auth backend", tech="python_fastapi"):
    return RetrievedChunk(
        id=chunk_id,
        document_id=1,
        chunk_type="budget_component",
        content=f"[Project: x]\n\nComponent: {name}\nDescription: Does a thing.\nEstimated hours: {hours}",
        metadata={
            "budget_id": budget_id,
            "client_sector": sector,
            "year": year,
            "main_technology": tech,
            "complexity": "high",
            "estimated_hours": hours,
        },
        distance=0.1,
    )


CHUNKS = [
    _chunk(1, "BUD-A", "finance", 2024, 120),
    _chunk(2, "BUD-B", "healthcare", 2023, 90, tech="node"),
    _chunk(3, "BUD-C", "healthcare", 2024, 60),
]


def _retrieve(chunks):
    return AsyncMock(return_value=(list(chunks), {}))


async def test_search_maps_chunks_to_items_with_hours_and_metadata():
    retrieve = _retrieve(CHUNKS[:1])
    result = await search_budgets(
        object(), SearchBudgetsArgs(query="auth backend"), embed_one=lambda _: [0.1], retrieve_fn=retrieve
    )
    assert result["count"] == 1
    item = result["items"][0]
    assert item["budget_id"] == "BUD-A" and item["component"] == "Auth backend"
    assert item["sector"] == "finance" and item["estimated_hours"] == 120
    assert item["content_preview"] == "Does a thing."


async def test_search_wraps_hybrid_retrieval_with_reranking():
    retrieve = _retrieve(CHUNKS)
    await search_budgets(
        object(), SearchBudgetsArgs(query="auth backend"), embed_one=lambda _: [0.1], retrieve_fn=retrieve
    )
    kwargs = retrieve.call_args.kwargs
    assert kwargs["search_mode"] == "hybrid" and kwargs["rerank"] is True
    assert kwargs["query_text"] == "auth backend" and kwargs["query_vector"] == [0.1]


@pytest.mark.parametrize(
    ("filters", "expected_ids"),
    [
        ({"sector": "healthcare"}, ["BUD-B", "BUD-C"]),
        ({"sector": "Healthcare", "year_from": 2024}, ["BUD-C"]),
        ({"year_to": 2023}, ["BUD-B"]),
        ({"main_technology": "node"}, ["BUD-B"]),
        ({"sector": None}, ["BUD-A", "BUD-B", "BUD-C"]),
    ],
)
async def test_search_filters_on_metadata(filters, expected_ids):
    result = await search_budgets(
        object(),
        SearchBudgetsArgs(query="auth backend", filters=filters),
        embed_one=lambda _: [0.1],
        retrieve_fn=_retrieve(CHUNKS),
    )
    assert [i["budget_id"] for i in result["items"]] == expected_ids


# --- toolbox dispatcher -----------------------------------------------------


def _toolbox(retrieve=None):
    return AgentToolbox(object(), embed_one=lambda _: [0.1], retrieve_fn=retrieve or _retrieve(CHUNKS))


async def test_toolbox_runs_calculate_and_summarises_it():
    outcome = await _toolbox().execute(
        "calculate_estimate", json.dumps({"components": [{"name": "A", "reference_amounts": [100]}]})
    )
    assert not outcome.is_error and outcome.result["total_hours"] == 115.0
    assert outcome.observation == "total=115.0h across 1 components"


async def test_toolbox_search_observation_lists_budgets_and_hours():
    outcome = await _toolbox().execute("search_budgets", json.dumps({"query": "auth backend", "filters": None}))
    assert "BUD-A Auth backend (finance, 2024) = 120h" in outcome.observation


@pytest.mark.parametrize(
    ("name", "arguments", "fragment"),
    [
        ("search_budgets", "{not json", "not valid JSON"),
        ("search_budgets", json.dumps({"filters": None}), "query"),
        ("calculate_estimate", json.dumps({"components": []}), "components"),
        ("calculate_estimate", json.dumps({"components": [{"name": "A"}]}), "reference_amounts"),
        ("no_such_tool", "{}", "Unknown tool"),
    ],
)
async def test_bad_calls_come_back_as_errors_instead_of_raising(name, arguments, fragment):
    outcome = await _toolbox().execute(name, arguments)
    assert outcome.is_error and fragment in outcome.result["error"]


async def test_a_failing_tool_does_not_raise():
    boom = AsyncMock(side_effect=ConnectionError("db down"))
    outcome = await _toolbox(boom).execute("search_budgets", json.dumps({"query": "auth backend", "filters": None}))
    assert outcome.is_error and "ConnectionError" in outcome.result["error"]
