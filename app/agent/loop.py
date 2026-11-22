"""Manual agent loop over the OpenAI Responses API.

The loop is driven here, not by the API: each `responses.create` call returns
`reasoning` and `function_call` items and stops. We run the requested tools,
send the results back as `function_call_output` items carrying the same
`call_id`, and chain the next call with `previous_response_id`. The loop ends
when the model answers without calling a tool, or after `max_iterations` model
turns as a safeguard.

Every tool call becomes one trace step (reasoning, action, observation).
The structured estimate is requested in a last `responses.parse` call chained
to the final response; its total is computed here, not by the model.
"""

import json
from typing import Any

import structlog

from app.agent.schemas import (
    AgentEstimate,
    AgentRunResult,
    AgentStep,
    AgentTrace,
    StopReason,
)
from app.agent.tools import AGENT_TOOLS, TOTAL_TOLERANCE_HOURS, AgentToolbox
from app.prompts.loader import render_agent_system_prompt

logger = structlog.get_logger(__name__)

NO_SUMMARY = "(the model returned no reasoning summary for this turn)"
FINAL_REQUEST = (
    "The analysis is complete. Return the final estimate as structured output.\n"
    "- One entry per component identified in the analysis.\n"
    "- hours: exactly the value calculate_estimate returned for that component, with no changes.\n"
    "- has_references: false, and hours 0, for a component with no comparable reference.\n"
    "- reference_budget_ids: the budget ids of the references you actually used; empty if none.\n"
    "- rationale: one or two sentences saying why those references fit, or why none did.\n"
    "- assumptions: every issue validate_estimate reported, plus any weak match or missing evidence. "
    "Do not add assumptions unrelated to the evidence.\n"
    "- summary: two sentences at most, covering how well the archive covers the project."
)


class AgentError(Exception):
    """The agent could not produce a usable estimate."""


def _reasoning_summary(output: list[Any]) -> str:
    texts = []
    for item in output:
        if getattr(item, "type", None) == "reasoning":
            texts.extend(part.text for part in (getattr(item, "summary", None) or []) if part.text)
    return "\n".join(texts).strip()


def _function_calls(output: list[Any]) -> list[Any]:
    return [item for item in output if getattr(item, "type", None) == "function_call"]


def _final_message(output: list[Any]) -> str:
    texts = []
    for item in output:
        if getattr(item, "type", None) == "message":
            for part in item.content:
                if getattr(part, "text", None):
                    texts.append(part.text)
    return " ".join(texts).strip()


def _action_text(name: str, raw_arguments: str) -> str:
    try:
        arguments = json.dumps(json.loads(raw_arguments), ensure_ascii=False)
    except json.JSONDecodeError:
        arguments = raw_arguments
    return f"{name}({arguments})"


def _tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    return (getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0)


async def run_estimation_agent(
    client: Any,
    toolbox: AgentToolbox,
    *,
    transcript: str,
    model: str,
    effort: str,
    max_iterations: int,
) -> AgentRunResult:
    instructions = render_agent_system_prompt(max_iterations=max_iterations)
    reasoning = {"effort": effort, "summary": "auto"}
    trace = AgentTrace()
    search_calls = 0
    last_calculated_total: float | None = None
    input_tokens = output_tokens = 0

    response = await client.responses.create(
        model=model,
        instructions=instructions,
        input=transcript,
        tools=AGENT_TOOLS,
        reasoning=reasoning,
    )
    iterations = 1
    stopped_reason: StopReason = "completed"
    pending_outputs: list[dict[str, Any]] = []

    while True:
        tokens_in, tokens_out = _tokens(response)
        input_tokens += tokens_in
        output_tokens += tokens_out
        calls = _function_calls(response.output)
        summary = _reasoning_summary(response.output) or NO_SUMMARY

        if not calls:
            trace.steps.append(
                AgentStep(
                    index=len(trace.steps) + 1,
                    reasoning=summary,
                    action="finish()",
                    observation=_final_message(response.output) or "(no closing message)",
                )
            )
            break

        outputs: list[dict[str, Any]] = []
        first_step_of_turn = len(trace.steps) + 1
        for position, call in enumerate(calls):
            outcome = await toolbox.execute(call.name, call.arguments)
            if call.name == "search_budgets":
                search_calls += 1
            if call.name == "calculate_estimate" and not outcome.is_error:
                last_calculated_total = outcome.result["total_hours"]
            step_reasoning = (
                summary if position == 0 else f"(parallel call from the same turn as STEP {first_step_of_turn})"
            )
            trace.steps.append(
                AgentStep(
                    index=len(trace.steps) + 1,
                    reasoning=step_reasoning,
                    action=_action_text(call.name, call.arguments),
                    observation=outcome.observation,
                )
            )
            logger.info("agent_step", step=len(trace.steps), tool=call.name, error=outcome.is_error)
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(outcome.result, ensure_ascii=False),
                }
            )

        if iterations >= max_iterations:
            stopped_reason = "max_iterations"
            pending_outputs = outputs
            logger.warning("agent_max_iterations_reached", iterations=iterations)
            break

        response = await client.responses.create(
            model=model,
            instructions=instructions,
            previous_response_id=response.id,
            input=outputs,
            tools=AGENT_TOOLS,
            reasoning=reasoning,
        )
        iterations += 1

    final = await client.responses.parse(
        model=model,
        instructions=instructions,
        previous_response_id=response.id,
        input=[*pending_outputs, {"role": "user", "content": FINAL_REQUEST}],
        text_format=AgentEstimate,
        reasoning={"effort": effort},
    )
    tokens_in, tokens_out = _tokens(final)
    input_tokens += tokens_in
    output_tokens += tokens_out
    estimate: AgentEstimate | None = final.output_parsed
    if estimate is None:
        raise AgentError("The model did not return a parsable structured estimate.")

    total = round(sum(component.hours for component in estimate.components), 1)
    consistent = (
        None if last_calculated_total is None else abs(total - last_calculated_total) <= TOTAL_TOLERANCE_HOURS
    )
    logger.info(
        "agent_run_done",
        iterations=iterations,
        stopped_reason=stopped_reason,
        search_calls=search_calls,
        total_hours=total,
        consistent=consistent,
    )
    return AgentRunResult(
        estimate=estimate,
        total_hours=total,
        consistent_with_calculation=consistent,
        trace=trace,
        iterations=iterations,
        stopped_reason=stopped_reason,
        search_calls=search_calls,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
