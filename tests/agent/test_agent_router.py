from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIError

import app.agent.router as router_module
from app.agent.loop import AgentError
from app.agent.schemas import AgentRunResult, AgentStep, AgentTrace
from app.db import get_session
from app.main import app
from tests.agent.fakes import estimate

URL = "/estimate/agent"
BODY = {"transcript": "A discovery meeting transcript that is comfortably longer than fifty characters."}


def _result() -> AgentRunResult:
    return AgentRunResult(
        estimate=estimate(("Auth", 115.0)),
        total_hours=115.0,
        consistent_with_calculation=True,
        trace=AgentTrace(steps=[AgentStep(index=1, reasoning="r", action="finish()", observation="done")]),
        iterations=2,
        stopped_reason="completed",
        search_calls=1,
        model="gpt-5",
        input_tokens=10,
        output_tokens=5,
    )


@pytest.fixture(autouse=True)
def _wiring():
    async def override():
        yield object()

    app.dependency_overrides[get_session] = override
    with (
        patch.object(router_module, "OpenAIEmbedder"),
        patch.object(router_module, "get_openai_client", return_value=object()),
    ):
        yield
    app.dependency_overrides.pop(get_session, None)


def _post(client, run: AsyncMock, body=None):
    with patch.object(router_module, "run_estimation_agent", run):
        return client.post(URL, json=BODY if body is None else body)


def test_returns_the_estimate_and_the_rendered_trace(client):
    response = _post(client, AsyncMock(return_value=_result()))
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["total_hours"] == 115.0
    assert body["result"]["stopped_reason"] == "completed"
    assert body["rendered_trace"].startswith("STEP 1\n  reasoning:   r")


def test_model_and_effort_default_to_settings_and_can_be_overridden(client):
    run = AsyncMock(return_value=_result())
    _post(client, run)
    assert run.call_args.kwargs["model"] == "gpt-5" and run.call_args.kwargs["effort"] == "medium"

    run = AsyncMock(return_value=_result())
    _post(client, run, {**BODY, "model": "gpt-5-mini", "effort": "low"})
    assert run.call_args.kwargs["model"] == "gpt-5-mini" and run.call_args.kwargs["effort"] == "low"


@pytest.mark.parametrize(
    "body",
    [{}, {"transcript": "too short"}, {**BODY, "effort": "extreme"}],
)
def test_invalid_requests_are_rejected_with_422(client, body):
    run = AsyncMock(return_value=_result())
    assert _post(client, run, body).status_code == 422
    run.assert_not_called()


def test_agent_failure_is_a_502(client):
    assert _post(client, AsyncMock(side_effect=AgentError("no parse"))).status_code == 502


def test_openai_failure_is_a_502(client):
    error = APIError("boom", httpx.Request("POST", "https://api.openai.com"), body=None)
    assert _post(client, AsyncMock(side_effect=error)).status_code == 502


def test_missing_configuration_is_a_500(client):
    assert _post(client, AsyncMock(side_effect=RuntimeError("no key"))).status_code == 500
    with patch.object(router_module, "get_openai_client", return_value=None):
        assert _post(client, AsyncMock(return_value=_result())).status_code == 500
