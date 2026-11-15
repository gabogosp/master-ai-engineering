from collections.abc import Sequence

import structlog
from openai import AsyncOpenAI
from pydantic import ValidationError

from app.embedding_pipeline.retrieval import RetrievedChunk
from app.grounded_estimation.schemas import Estimate
from app.prompts.loader import render_grounded_estimation_prompt

logger = structlog.get_logger(__name__)

DEFAULT_MAX_ATTEMPTS = 3


class EstimateGenerationError(RuntimeError):
    """The model never produced an estimate that satisfies the schema rules."""


def to_prompt_chunks(chunks: Sequence[RetrievedChunk]) -> list[dict]:
    """Ids are passed as strings: they are what the model must copy back, and
    verification compares them as strings."""
    return [
        {
            "chunk_id": str(chunk.id),
            "document_id": chunk.metadata["budget_id"],
            "content": chunk.content,
        }
        for chunk in chunks
    ]


async def generate_estimate(
    client: AsyncOpenAI,
    *,
    query: str,
    chunks: Sequence[RetrievedChunk],
    model: str,
    correction: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Estimate:
    """Generate a line-level estimate from the retrieved chunks.

    `correction` lets the caller append feedback (e.g. citations that pointed
    to ids the model was never given) before the first attempt. Schema
    violations raised by the Estimate validators are fed back to the model
    and retried, up to `max_attempts`.
    """
    system_prompt, user_prompt = render_grounded_estimation_prompt(
        query=query, chunks=to_prompt_chunks(chunks)
    )
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if correction:
        messages.append({"role": "user", "content": correction})

    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.responses.parse(
                model=model,
                input=messages,
                text_format=Estimate,
                temperature=0,
            )
        except ValidationError as exc:
            problems = "; ".join(error["msg"] for error in exc.errors())
            logger.warning("grounded_estimate_invalid", attempt=attempt, problems=problems)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous answer broke the rules and was rejected: "
                        f"{problems}. Answer again and fix it."
                    ),
                }
            )
            continue

        if response.output_parsed is None:
            logger.warning("grounded_estimate_empty", attempt=attempt)
            continue

        logger.info("grounded_estimate_generated", attempt=attempt, model=model)
        return response.output_parsed

    raise EstimateGenerationError(
        f"no valid estimate after {max_attempts} attempts (model={model})"
    )
