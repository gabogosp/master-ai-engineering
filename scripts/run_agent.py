"""Runs the estimation agent on a transcript file and prints its trace.

    PYTHONPATH=. uv run python scripts/run_agent.py evals/agent/transcripts/complex.txt \
        --out evals/agent/trace_complex.txt

Uses the real retrieval (the database must be up and the corpus ingested) and
the OpenAI API (OPENAI_API_KEY required).
"""

import argparse
import asyncio
from pathlib import Path

from app.agent.loop import run_estimation_agent
from app.agent.tools import AgentToolbox
from app.config import settings
from app.db import AsyncSessionLocal
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.services.openai_client import get_openai_client


def render_report(result, transcript_name: str) -> str:
    estimate = result.estimate
    lines = [
        "=" * 78,
        f"AGENT RUN  transcript={transcript_name}  model={result.model}",
        "=" * 78,
        result.trace.render(),
        "",
        "=" * 78,
        "FINAL ESTIMATE",
        "=" * 78,
    ]
    for component in estimate.components:
        refs = ", ".join(component.reference_budget_ids) or "none"
        flag = "" if component.has_references else "  [NO REFERENCES]"
        lines.append(f"- {component.name}: {component.hours} h  (references: {refs}){flag}")
        lines.append(f"    {component.rationale}")
    lines += [
        f"TOTAL: {result.total_hours} h",
        f"Consistent with last calculate_estimate: {result.consistent_with_calculation}",
        "Assumptions:",
        *[f"  - {a}" for a in estimate.assumptions],
        f"Summary: {estimate.summary}",
        "",
        f"iterations={result.iterations}  search_calls={result.search_calls}  "
        f"stopped_reason={result.stopped_reason}  tokens_in={result.input_tokens}  "
        f"tokens_out={result.output_tokens}",
    ]
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--model", default=settings.agent_model)
    parser.add_argument("--effort", default=settings.agent_reasoning_effort)
    parser.add_argument("--out", type=Path, default=None, help="also write the report to this file")
    args = parser.parse_args()

    client = get_openai_client()
    if client is None:
        raise SystemExit("OPENAI_API_KEY is not configured.")

    async with AsyncSessionLocal() as session:
        toolbox = AgentToolbox(session, embed_one=OpenAIEmbedder().embed_one)
        result = await run_estimation_agent(
            client,
            toolbox,
            transcript=args.transcript.read_text(encoding="utf-8"),
            model=args.model,
            effort=args.effort,
            max_iterations=settings.agent_max_iterations,
        )

    report = render_report(result, args.transcript.name)
    print(report)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\nTrace written to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
