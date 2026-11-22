"""Scripted stand-in for the OpenAI client: replays prepared responses and
records every call, so the agent loop can be tested without the network."""

from types import SimpleNamespace as NS
from typing import Any

from app.agent.schemas import AgentComponent, AgentEstimate


def reasoning(text: str = "") -> NS:
    return NS(type="reasoning", summary=[NS(text=text)] if text else [])


def call(call_id: str, name: str, arguments: str) -> NS:
    return NS(type="function_call", call_id=call_id, name=name, arguments=arguments)


def message(text: str) -> NS:
    return NS(type="message", content=[NS(type="output_text", text=text)])


def response(response_id: str, *output: Any) -> NS:
    return NS(id=response_id, output=list(output), usage=NS(input_tokens=10, output_tokens=5))


def estimate(*components: tuple[str, float]) -> AgentEstimate:
    return AgentEstimate(
        components=[
            AgentComponent(
                name=name, hours=hours, reference_budget_ids=["BUD-1"], rationale="r", has_references=hours > 0
            )
            for name, hours in components
        ],
        assumptions=[],
        summary="s",
    )


class FakeResponses:
    def __init__(self, scripted: list[NS], parsed: AgentEstimate | None) -> None:
        self._scripted = list(scripted)
        self._parsed = parsed
        self.create_calls: list[dict[str, Any]] = []
        self.parse_calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> NS:
        self.create_calls.append(kwargs)
        # Past the script, keep repeating the last turn (used for the cap test).
        return self._scripted.pop(0) if len(self._scripted) > 1 else self._scripted[0]

    async def parse(self, **kwargs: Any) -> NS:
        self.parse_calls.append(kwargs)
        return NS(output_parsed=self._parsed, usage=NS(input_tokens=7, output_tokens=3))


class FakeClient:
    def __init__(self, scripted: list[NS], parsed: AgentEstimate | None = None) -> None:
        self.responses = FakeResponses(scripted, parsed)
