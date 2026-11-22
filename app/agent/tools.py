"""Tools of the estimation agent: JSON schemas for the Responses API and the
Python implementations behind them.

`search_budgets` wraps the existing retrieval (hybrid search + cross-encoder
reranking, `app.embedding_pipeline.retrieval.retrieve`); it does not
reimplement it. `calculate_estimate` and `validate_estimate` are
deterministic: no LLM is involved in any number the agent reports.

The schemas follow the Responses API shape (flat: `type`/`name` at the same
level, no `function` wrapper) with `strict: true`. Strict mode requires every
property to be listed in `required` and `additionalProperties: false`, so
optional values are expressed as nullable.
"""

import asyncio
import json
import re
import statistics
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import (
    CalculateEstimateArgs,
    SearchBudgetsArgs,
    SearchBudgetsFilters,
    ValidateEstimateArgs,
)
from app.embedding_pipeline.retrieval import RetrievedChunk, retrieve

logger = structlog.get_logger(__name__)

# Flat contingency added to each component's central estimate. Kept as a named
# constant so there are no hidden multipliers.
CONTINGENCY_FACTOR = 0.15
RESULTS_PER_SEARCH = 5
# When filters are given, retrieve a wider pool first and filter afterwards:
# the retrieval layer has no metadata filtering of its own.
FILTERED_POOL_SIZE = 25
PREVIEW_CHARS = 220

# Plausibility bounds used by validate_estimate.
MAX_PLAUSIBLE_COMPONENT_HOURS = 3000.0
MAX_PLAUSIBLE_TOTAL_HOURS = 12000.0
RANGE_LOWER_FACTOR = 0.5
RANGE_UPPER_FACTOR = 1.5
TOTAL_TOLERANCE_HOURS = 0.5


# --- Schemas for the Responses API ------------------------------------------
SEARCH_BUDGETS_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "search_budgets",
    "description": (
        "Search the archive of historical software project budgets for components similar to ONE "
        "specific requirement, and return them with the engineer-hours each one took. Call it once "
        "per component you need to estimate, with a query focused on that component alone; do not "
        "mix several components in one query. The archive is written in English, so write the "
        "query in English and describe what the component does and its technology. Returns up to "
        "5 items, best match first, each with budget_id, component, sector, year, main_technology, "
        "complexity, estimated_hours and a content_preview. An empty list means nothing similar "
        "exists. Check that the items are really comparable before using their hours. Does not "
        "calculate anything."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language description of one component, in English. Example: "
                    "'offline-capable mobile app for delivery drivers with signature capture'."
                ),
            },
            "filters": {
                "type": ["object", "null"],
                "description": (
                    "Optional narrowing of the archive by metadata. Pass null to search without "
                    "constraints, which is the right first attempt. Use it to refine only when "
                    "results are dominated by an irrelevant sector or era."
                ),
                "properties": {
                    "sector": {
                        "type": ["string", "null"],
                        "description": "Client sector: finance, ecommerce, healthcare or industrial.",
                    },
                    "main_technology": {
                        "type": ["string", "null"],
                        "description": "Main technology of the budget, e.g. python_fastapi, node, go.",
                    },
                    "year_from": {
                        "type": ["integer", "null"],
                        "description": "Earliest budget year to include.",
                    },
                    "year_to": {
                        "type": ["integer", "null"],
                        "description": "Latest budget year to include.",
                    },
                },
                "required": ["sector", "main_technology", "year_from", "year_to"],
                "additionalProperties": False,
            },
        },
        "required": ["query", "filters"],
        "additionalProperties": False,
    },
}

CALCULATE_ESTIMATE_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "calculate_estimate",
    "description": (
        "Compute the estimated engineer-hours of one or more components, and their total, from the "
        "historical reference hours found with search_budgets. It takes the median of each "
        "component's references and adds a 15% contingency. Use it instead of doing arithmetic "
        "yourself; call it again with the full list when you want the consolidated total. Pass a "
        "component with an empty reference_amounts list if you found nothing comparable: it is "
        "costed at 0 and flagged as unbudgeted, never guessed. Does not search anything."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "description": "Components to cost, each with its reference hours.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Short component name."},
                        "reference_amounts": {
                            "type": "array",
                            "description": (
                                "estimated_hours values copied from comparable search_budgets "
                                "items. Empty if none."
                            ),
                            "items": {"type": "number"},
                        },
                    },
                    "required": ["name", "reference_amounts"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["components"],
        "additionalProperties": False,
    },
}

VALIDATE_ESTIMATE_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "validate_estimate",
    "description": (
        "Run sanity checks on a finished estimate: components with no reference budget, hours "
        "outside the range of their references, non-positive or implausibly large figures, and a "
        "total that does not match the sum of the components. Call it once, as the last step, with "
        "the figures returned by calculate_estimate. Returns valid=true or a list of issues to fix "
        "or report. It never changes the estimate."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "description": "The estimated components.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Component name."},
                        "estimated_hours": {
                            "type": "number",
                            "description": "Hours calculate_estimate returned for it.",
                        },
                        "reference_amounts": {
                            "type": "array",
                            "description": "Reference hours used for it; empty if none.",
                            "items": {"type": "number"},
                        },
                    },
                    "required": ["name", "estimated_hours", "reference_amounts"],
                    "additionalProperties": False,
                },
            },
            "total_hours": {
                "type": "number",
                "description": "The total_hours that calculate_estimate returned.",
            },
        },
        "required": ["components", "total_hours"],
        "additionalProperties": False,
    },
}

AGENT_TOOLS: list[dict[str, Any]] = [
    SEARCH_BUDGETS_TOOL,
    CALCULATE_ESTIMATE_TOOL,
    VALIDATE_ESTIMATE_TOOL,
]


# --- Deterministic tools ------------------------------------------------------
def calculate_estimate(args: CalculateEstimateArgs) -> dict[str, Any]:
    breakdown: list[dict[str, Any]] = []
    total = 0.0
    for component in args.components:
        refs = component.reference_amounts
        if refs:
            hours = round(statistics.median(refs) * (1 + CONTINGENCY_FACTOR), 1)
            unbudgeted = False
        else:
            hours = 0.0
            unbudgeted = True
        total += hours
        breakdown.append(
            {
                "name": component.name,
                "reference_count": len(refs),
                "estimated_hours": hours,
                "unbudgeted": unbudgeted,
            }
        )
    total = round(total, 1)
    unbudgeted_names = [c["name"] for c in breakdown if c["unbudgeted"]]
    summary = f"total={total}h across {len(breakdown)} components"
    if unbudgeted_names:
        summary += f"; no references for: {', '.join(unbudgeted_names)}"
    return {"components": breakdown, "total_hours": total, "summary": summary}


def validate_estimate(args: ValidateEstimateArgs) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []

    def add(component: str | None, code: str, message: str) -> None:
        issues.append({"component": component, "code": code, "message": message})

    for component in args.components:
        refs, hours, name = component.reference_amounts, component.estimated_hours, component.name
        if not refs:
            if hours > 0:
                add(name, "hours_without_reference", f"{hours}h estimated but no reference budget backs it.")
            else:
                add(name, "no_reference", "No comparable budget was found; the component is costed at 0.")
            continue
        if hours <= 0:
            add(name, "non_positive_hours", f"{hours}h estimated despite having references.")
            continue
        if hours < min(refs) * RANGE_LOWER_FACTOR or hours > max(refs) * RANGE_UPPER_FACTOR:
            add(
                name,
                "outside_reference_range",
                f"{hours}h is far from its references ({min(refs)}-{max(refs)}h).",
            )
        if hours > MAX_PLAUSIBLE_COMPONENT_HOURS:
            add(name, "implausible_component", f"{hours}h exceeds {MAX_PLAUSIBLE_COMPONENT_HOURS}h for one component.")

    summed = round(sum(c.estimated_hours for c in args.components), 1)
    if abs(summed - args.total_hours) > TOTAL_TOLERANCE_HOURS:
        add(None, "total_mismatch", f"total_hours={args.total_hours} but the components add up to {summed}.")
    if args.total_hours > MAX_PLAUSIBLE_TOTAL_HOURS:
        add(None, "implausible_total", f"total of {args.total_hours}h exceeds {MAX_PLAUSIBLE_TOTAL_HOURS}h.")

    valid = not issues
    summary = "valid: no issues found" if valid else f"{len(issues)} issue(s): " + "; ".join(
        f"{i['code']}" + (f"[{i['component']}]" if i["component"] else "") for i in issues
    )
    return {"valid": valid, "issues": issues, "summary": summary}


# --- Retrieval-backed tool ----------------------------------------------------
_COMPONENT_NAME = re.compile(r"^Component:\s*(.+)$", re.MULTILINE)
_DESCRIPTION = re.compile(r"^Description:\s*(.+)$", re.MULTILINE)


def _chunk_to_item(chunk: RetrievedChunk) -> dict[str, Any]:
    name = _COMPONENT_NAME.search(chunk.content)
    description = _DESCRIPTION.search(chunk.content)
    meta = chunk.metadata
    return {
        "chunk_id": str(chunk.id),
        "budget_id": meta.get("budget_id"),
        "component": name.group(1).strip() if name else None,
        "sector": meta.get("client_sector"),
        "year": meta.get("year"),
        "main_technology": meta.get("main_technology"),
        "complexity": meta.get("complexity"),
        "estimated_hours": meta.get("estimated_hours"),
        "content_preview": (description.group(1).strip() if description else chunk.content)[:PREVIEW_CHARS],
    }


def _matches(item: dict[str, Any], filters: SearchBudgetsFilters) -> bool:
    if filters.sector and (item["sector"] or "").lower() != filters.sector.lower():
        return False
    if filters.main_technology and (item["main_technology"] or "").lower() != filters.main_technology.lower():
        return False
    year = item["year"]
    if filters.year_from is not None and (year is None or year < filters.year_from):
        return False
    if filters.year_to is not None and (year is None or year > filters.year_to):
        return False
    return True


RetrieveFn = Callable[..., Awaitable[tuple[list[RetrievedChunk], dict]]]


async def search_budgets(
    session: AsyncSession,
    args: SearchBudgetsArgs,
    *,
    embed_one: Callable[[str], list[float]],
    retrieve_fn: RetrieveFn = retrieve,
) -> dict[str, Any]:
    filters = args.filters
    has_filters = filters is not None and any(v is not None for v in filters.model_dump().values())
    # The embedder is a blocking sync client: keep it off the event loop.
    query_vector = await asyncio.to_thread(embed_one, args.query)
    chunks, _metrics = await retrieve_fn(
        session,
        query_text=args.query,
        query_vector=query_vector,
        search_mode="hybrid",
        rerank=True,
        k=FILTERED_POOL_SIZE if has_filters else RESULTS_PER_SEARCH,
    )
    items = [_chunk_to_item(chunk) for chunk in chunks]
    if has_filters:
        items = [item for item in items if _matches(item, filters)][:RESULTS_PER_SEARCH]
    return {"query": args.query, "count": len(items), "items": items}


# --- Dispatcher ---------------------------------------------------------------
@dataclass
class ToolOutcome:
    result: dict[str, Any]
    observation: str
    is_error: bool = False


def _observe_search(result: dict[str, Any]) -> str:
    if not result["items"]:
        return f"0 items for {result['query']!r}"
    listing = "; ".join(
        f"{i['budget_id']} {i['component']} ({i['sector']}, {i['year']}) = {i['estimated_hours']}h"
        for i in result["items"]
    )
    return f"{result['count']} items for {result['query']!r}: {listing}"


class AgentToolbox:
    """Executes the tool calls the model asks for. Never raises on bad input:
    errors come back as a result the model reads on its next turn."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        embed_one: Callable[[str], list[float]],
        retrieve_fn: RetrieveFn = retrieve,
    ) -> None:
        self._session = session
        self._embed_one = embed_one
        self._retrieve_fn = retrieve_fn

    async def execute(self, name: str, raw_arguments: str) -> ToolOutcome:
        try:
            payload = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            return self._error(f"Arguments are not valid JSON: {exc}")

        try:
            if name == "search_budgets":
                result = await search_budgets(
                    self._session,
                    SearchBudgetsArgs.model_validate(payload),
                    embed_one=self._embed_one,
                    retrieve_fn=self._retrieve_fn,
                )
                return ToolOutcome(result, _observe_search(result))
            if name == "calculate_estimate":
                result = calculate_estimate(CalculateEstimateArgs.model_validate(payload))
                return ToolOutcome(result, result["summary"])
            if name == "validate_estimate":
                result = validate_estimate(ValidateEstimateArgs.model_validate(payload))
                return ToolOutcome(result, result["summary"])
        except ValidationError as exc:
            return self._error(f"Invalid arguments for {name}: {_short_validation(exc)}")
        except Exception as exc:  # a failing tool must not end the loop
            logger.error("agent_tool_failed", tool=name, error_type=type(exc).__name__, exc_info=True)
            return self._error(f"{name} failed: {type(exc).__name__}")
        return self._error(f"Unknown tool {name!r}. Available: search_budgets, calculate_estimate, validate_estimate.")

    @staticmethod
    def _error(message: str) -> ToolOutcome:
        return ToolOutcome({"error": message}, f"ERROR: {message}", is_error=True)


def _short_validation(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:3])


__all__ = ["AGENT_TOOLS", "AgentToolbox", "ToolOutcome"]
