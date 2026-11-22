import json

import pytest

from app.agent.loop import AgentError, run_estimation_agent
from app.agent.tools import AgentToolbox
from tests.agent.fakes import FakeClient, call, estimate, message, reasoning, response
from tests.agent.test_agent_tools import CHUNKS, _retrieve

SEARCH_ARGS = json.dumps({"query": "auth backend", "filters": None})
CALC_ARGS = json.dumps({"components": [{"name": "Auth", "reference_amounts": [100]}]})


def _toolbox():
    return AgentToolbox(object(), embed_one=lambda _: [0.1], retrieve_fn=_retrieve(CHUNKS))


async def _run(client, max_iterations=10):
    return await run_estimation_agent(
        client, _toolbox(), transcript="meeting", model="m", effort="medium", max_iterations=max_iterations
    )


def _happy_client(final=(("Auth", 115.0),)):
    return FakeClient(
        [
            response(
                "r1",
                reasoning("Two components, search both."),
                call("c1", "search_budgets", SEARCH_ARGS),
                call("c2", "search_budgets", json.dumps({"query": "mobile app", "filters": None})),
            ),
            response("r2", reasoning("Cost it."), call("c3", "calculate_estimate", CALC_ARGS)),
            response("r3", message("Analysis complete.")),
        ],
        parsed=estimate(*final),
    )


async def test_loop_runs_until_the_model_stops_calling_tools():
    result = await _run(_happy_client())
    assert result.stopped_reason == "completed" and result.iterations == 3
    assert [s.action.split("(")[0] for s in result.trace.steps] == [
        "search_budgets",
        "search_budgets",
        "calculate_estimate",
        "finish",
    ]
    assert result.search_calls == 2


async def test_each_output_references_its_call_id_and_calls_are_chained():
    client = _happy_client()
    await _run(client)
    first, second = client.responses.create_calls[1:]
    assert first["previous_response_id"] == "r1"
    assert [(o["type"], o["call_id"]) for o in first["input"]] == [
        ("function_call_output", "c1"),
        ("function_call_output", "c2"),
    ]
    assert second["previous_response_id"] == "r2" and second["input"][0]["call_id"] == "c3"
    assert json.loads(second["input"][0]["output"])["total_hours"] == 115.0


async def test_first_call_carries_prompt_transcript_tools_and_reasoning_settings():
    client = _happy_client()
    await _run(client)
    first = client.responses.create_calls[0]
    assert first["input"] == "meeting" and first["model"] == "m"
    assert first["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert [t["name"] for t in first["tools"]] == ["search_budgets", "calculate_estimate", "validate_estimate"]
    assert "10 turns" in first["instructions"]


async def test_trace_pairs_reasoning_action_and_observation_per_step():
    result = await _run(_happy_client())
    step1, step2, step3, finish = result.trace.steps
    assert step1.reasoning == "Two components, search both."
    assert step1.action == 'search_budgets({"query": "auth backend", "filters": null})'
    assert "BUD-A Auth backend" in step1.observation
    assert "parallel call" in step2.reasoning and "STEP 1" in step2.reasoning
    assert step3.observation.startswith("total=115.0h")
    assert finish.observation == "Analysis complete."
    rendered = result.trace.render()
    assert rendered.startswith("STEP 1\n  reasoning:") and "  action:      search_budgets(" in rendered
    assert "  observation: " in rendered


async def test_missing_reasoning_summary_is_stated_not_invented():
    client = FakeClient([response("r1", message("done"))], parsed=estimate(("A", 1.0)))
    result = await _run(client)
    assert "no reasoning summary" in result.trace.steps[0].reasoning


async def test_total_is_computed_from_components_and_checked_against_the_calculation():
    ok = await _run(_happy_client())
    assert ok.total_hours == 115.0 and ok.consistent_with_calculation is True

    drifted = await _run(_happy_client(final=(("Auth", 400.0),)))
    assert drifted.total_hours == 400.0 and drifted.consistent_with_calculation is False


async def test_consistency_is_unknown_when_nothing_was_calculated():
    client = FakeClient([response("r1", message("done"))], parsed=estimate(("A", 10.0)))
    assert (await _run(client)).consistent_with_calculation is None


async def test_tokens_are_summed_across_all_calls():
    result = await _run(_happy_client())
    assert result.input_tokens == 3 * 10 + 7 and result.output_tokens == 3 * 5 + 3


async def test_iteration_cap_stops_the_loop_and_answers_pending_calls():
    looping = response("r1", call("c1", "search_budgets", SEARCH_ARGS))
    client = FakeClient([looping], parsed=estimate(("A", 0.0)))
    result = await _run(client, max_iterations=3)
    assert result.stopped_reason == "max_iterations" and result.iterations == 3
    assert len(client.responses.create_calls) == 3
    final_input = client.responses.parse_calls[0]["input"]
    assert final_input[0] == {
        "type": "function_call_output",
        "call_id": "c1",
        "output": final_input[0]["output"],
    }
    assert final_input[-1]["role"] == "user"


async def test_invalid_arguments_are_returned_to_the_model_and_the_loop_continues():
    client = FakeClient(
        [
            response("r1", call("c1", "calculate_estimate", json.dumps({"components": []}))),
            response("r2", message("done")),
        ],
        parsed=estimate(("A", 1.0)),
    )
    result = await _run(client)
    assert result.stopped_reason == "completed"
    assert result.trace.steps[0].observation.startswith("ERROR:")
    sent_back = json.loads(client.responses.create_calls[1]["input"][0]["output"])
    assert "error" in sent_back


async def test_final_step_uses_structured_output_chained_to_the_last_response():
    client = _happy_client()
    await _run(client)
    parse = client.responses.parse_calls[0]
    assert parse["previous_response_id"] == "r3"
    assert parse["text_format"].__name__ == "AgentEstimate"


async def test_unparsable_final_output_raises_agent_error():
    client = FakeClient([response("r1", message("done"))], parsed=None)
    with pytest.raises(AgentError):
        await _run(client)
