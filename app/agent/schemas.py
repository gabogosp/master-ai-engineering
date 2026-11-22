"""Models of the estimation agent: tool arguments, trace and final result.

The tool-argument models validate what the LLM sends BEFORE a tool runs, so a
malformed or invented argument becomes an error message the model can read and
correct, never an exception that kills the loop.

Models used as a structured output (`AgentEstimate`) carry no `Field`
constraints: OpenAI strict mode rejects them, so invariants are checked in code.
"""

from typing import Literal

from pydantic import BaseModel, Field


# --- Tool arguments --------------------------------------------------------
class SearchBudgetsFilters(BaseModel):
    sector: str | None = None
    main_technology: str | None = None
    year_from: int | None = None
    year_to: int | None = None


class SearchBudgetsArgs(BaseModel):
    query: str = Field(min_length=3)
    filters: SearchBudgetsFilters | None = None


class ComponentReferences(BaseModel):
    name: str = Field(min_length=1)
    reference_amounts: list[float]


class CalculateEstimateArgs(BaseModel):
    components: list[ComponentReferences] = Field(min_length=1)


class ComponentToValidate(BaseModel):
    name: str = Field(min_length=1)
    estimated_hours: float
    reference_amounts: list[float]


class ValidateEstimateArgs(BaseModel):
    components: list[ComponentToValidate] = Field(min_length=1)
    total_hours: float


# --- Trace -----------------------------------------------------------------
class AgentStep(BaseModel):
    index: int
    reasoning: str
    action: str
    observation: str


StopReason = Literal["completed", "max_iterations"]


class AgentTrace(BaseModel):
    steps: list[AgentStep] = Field(default_factory=list)

    def render(self) -> str:
        blocks = [
            f"STEP {step.index}\n"
            f"  reasoning:   {step.reasoning}\n"
            f"  action:      {step.action}\n"
            f"  observation: {step.observation}"
            for step in self.steps
        ]
        return "\n\n".join(blocks)


# --- Final result ----------------------------------------------------------
class AgentComponent(BaseModel):
    name: str
    hours: float
    reference_budget_ids: list[str]
    rationale: str
    has_references: bool


class AgentEstimate(BaseModel):
    """Structured output requested from the model in the final step. The total
    is not part of it: it is computed from the components."""

    components: list[AgentComponent]
    assumptions: list[str]
    summary: str


class AgentRunResult(BaseModel):
    estimate: AgentEstimate
    total_hours: float
    # True when the final components add up to the last calculate_estimate
    # total, i.e. the answer was not reworded after the calculation.
    consistent_with_calculation: bool | None
    trace: AgentTrace
    iterations: int
    stopped_reason: StopReason
    search_calls: int
    model: str
    input_tokens: int
    output_tokens: int


# --- HTTP ------------------------------------------------------------------
class AgentEstimateRequest(BaseModel):
    transcript: str = Field(min_length=50, max_length=60_000)
    model: str | None = None
    effort: Literal["minimal", "low", "medium", "high"] | None = None


class AgentEstimateResponse(BaseModel):
    result: AgentRunResult
    rendered_trace: str
