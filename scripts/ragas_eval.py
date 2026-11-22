"""Score the grounded estimation pipeline with the four RAGAS metrics.

    uv run --group eval python scripts/ragas_eval.py

Hits the live /estimate/grounded endpoint (BACKEND_URL, default
localhost:8000) for each query of evals/golden_set.json, then asks an LLM
judge to score faithfulness, answer relevancy, context precision and context
recall. Everything needed to re-score later is saved next to the golden set.
"""

import argparse
import asyncio
import json
import math
import os
from datetime import date
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.config import settings

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
ROOT = Path(__file__).resolve().parent.parent
GOLDEN_SET_PATH = ROOT / "evals" / "golden_set.json"
RESULTS_PATH = ROOT / "evals" / "ragas_baseline.json"

EMBEDDING_MODEL = "text-embedding-3-small"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
RETRIEVAL = {"k": 5, "search_mode": "hybrid", "rerank": False}


def render_answer(estimate: dict) -> str:
    """The estimate as plain text: the form in which RAGAS reads an answer."""
    lines = [estimate["summary"]]
    for line in estimate["lines"]:
        lines.append(f"- {line['component']}: {line['hours']:g} hours. {line['rationale']}")
    return "\n".join(lines)


def fetch_estimate(client: httpx.Client, query: str) -> dict:
    response = client.post(
        f"{BACKEND_URL}/estimate/grounded",
        json={"query": query, **RETRIEVAL},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


async def score_query(metrics: dict, question: str, answer: str, contexts: list[str], reference: str):
    async def safe(name: str, coro):
        try:
            return name, (await coro).value
        except Exception as exc:  # one failing metric must not hide the other three
            print(f"  ! {name} failed: {type(exc).__name__}: {exc}")
            return name, None

    results = await asyncio.gather(
        safe("faithfulness", metrics["faithfulness"].ascore(
            user_input=question, response=answer, retrieved_contexts=contexts)),
        safe("answer_relevancy", metrics["answer_relevancy"].ascore(
            user_input=question, response=answer)),
        safe("context_precision", metrics["context_precision"].ascore(
            user_input=question, reference=reference, retrieved_contexts=contexts)),
        safe("context_recall", metrics["context_recall"].ascore(
            user_input=question, retrieved_contexts=contexts, reference=reference)),
    )
    return dict(results)


def _mean(values: list) -> float | None:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return sum(clean) / len(clean) if clean else None


def _spread(values: list) -> float | None:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return max(clean) - min(clean) if len(clean) > 1 else None


def _count(report: dict, status: str) -> int:
    return sum(1 for verdict in report["verdicts"] if verdict["status"] == status)


def _fmt(value) -> str:
    return "  n/a" if value is None else f"{value:5.2f}"


async def main(runs: int) -> None:
    golden = json.loads(GOLDEN_SET_PATH.read_text())["queries"]

    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    # temperature=0: with the default the same input scored 0.0 or 0.5 across runs.
    # max_tokens: the default truncates the judge's output on long answers.
    judge = llm_factory(
        settings.ragas_judge_model, client=openai_client, temperature=0, max_tokens=4096
    )
    embeddings = OpenAIEmbeddings(client=openai_client, model=EMBEDDING_MODEL)
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }

    rows = []
    with httpx.Client() as http:
        for case in golden:
            runs_data = []
            for run_number in range(1, runs + 1):
                print(f"{case['id']} run {run_number}/{runs}: generating and scoring...")
                body = fetch_estimate(http, case["query"])
                answer = render_answer(body["estimate"])
                contexts = [chunk["content"] for chunk in body["retrieved_chunks"]]
                scores = await score_query(
                    metrics, case["query"], answer, contexts, case["ground_truth"]
                )
                runs_data.append(
                    {
                        "request_id": body["request_id"],
                        "answer": answer,
                        "contexts": contexts,
                        "citation_counts": {
                            status: _count(body["citation_report"], status)
                            for status in ("grounded", "dangling_citation", "insufficient_data")
                        },
                        "scores": scores,
                    }
                )
            rows.append(
                {
                    "id": case["id"],
                    "question": case["query"],
                    "ground_truth": case["ground_truth"],
                    "runs": runs_data,
                    "scores": {
                        name: _mean([run["scores"][name] for run in runs_data])
                        for name in METRIC_NAMES
                    },
                    "spread": {
                        name: _spread([run["scores"][name] for run in runs_data])
                        for name in METRIC_NAMES
                    },
                }
            )

    averages = {name: _mean([row["scores"][name] for row in rows]) for name in METRIC_NAMES}

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "date": date.today().isoformat(),
                "runs_per_query": runs,
                "generator_model": settings.grounded_generation_model,
                "judge_model": settings.ragas_judge_model,
                "embedding_model": EMBEDDING_MODEL,
                "retrieval": RETRIEVAL,
                "rows": rows,
                "average": averages,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"\nMean of {runs} run(s) per query")
    print("| Query | Faithfulness | Answer relevancy | Context precision | Context recall |")
    print("|---|---|---|---|---|")
    for row in rows:
        cells = " | ".join(_fmt(row["scores"][name]) for name in METRIC_NAMES)
        print(f"| {row['id']} | {cells} |")
    print("| **Average** | " + " | ".join(_fmt(averages[name]) for name in METRIC_NAMES) + " |")

    print("\nLargest run-to-run spread (max - min) per metric:")
    for name in METRIC_NAMES:
        spreads = [row["spread"][name] for row in rows if row["spread"][name] is not None]
        print(f"  {name}: {max(spreads):.2f}" if spreads else f"  {name}: n/a")
    print(f"\nSaved to {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="estimates generated and scored per query")
    asyncio.run(main(parser.parse_args().runs))
