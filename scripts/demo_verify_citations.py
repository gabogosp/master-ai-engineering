"""Show citation verification on a real estimate and on a tampered copy.

    PYTHONPATH=. uv run python scripts/demo_verify_citations.py

1. Requests a real estimate from the live /estimate/grounded endpoint and
   re-verifies it locally against the chunk ids the endpoint says it retrieved.
2. Plants an invented chunk id in one grounded line and verifies again, to show
   that a dangling citation is caught.
"""

import os

import httpx
import structlog

from app.grounded_estimation.citations import verify_citations
from app.grounded_estimation.schemas import CitationReport, Estimate

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
QUERY = "Real-time payment gateway with fraud detection and settlement reconciliation with the acquirer"
INVENTED_CHUNK_ID = "999"


def print_report(title: str, estimate: Estimate, report: CitationReport) -> None:
    print(f"\n=== {title} ===")
    print(
        f"lines: {len(report.verdicts)}  grounded: {report.grounded_count}  "
        f"dangling: {report.dangling_count}  insufficient: {report.insufficient_count}"
    )
    print(f"{'component':<34}{'status':<20}{'cited':<12}dangling")
    for line, verdict in zip(estimate.lines, report.verdicts, strict=True):
        cited = [source.chunk_id for source in line.sources]
        print(
            f"{line.component[:32]:<34}{verdict.status.value:<20}"
            f"{str(cited):<12}{verdict.dangling_chunk_ids or '-'}"
        )


def main() -> None:
    response = httpx.post(
        f"{BACKEND_URL}/estimate/grounded",
        json={"query": QUERY, "k": 5, "search_mode": "hybrid"},
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()

    estimate = Estimate.model_validate(body["estimate"])
    retrieved_ids = {chunk["chunk_id"] for chunk in body["retrieved_chunks"]}
    print(f"request_id: {body['request_id']}")
    print(f"retrieved chunk ids: {sorted(retrieved_ids, key=int)}")

    with structlog.contextvars.bound_contextvars(request_id=body["request_id"]):
        print_report("Real estimate", estimate, verify_citations(estimate, retrieved_ids))

        tampered = estimate.model_copy(deep=True)
        victim = next(line for line in tampered.lines if line.grounded)
        victim.sources[0].chunk_id = INVENTED_CHUNK_ID
        print(f"\n(planting chunk_id={INVENTED_CHUNK_ID!r} in line {victim.component!r})")
        print_report(
            "Same estimate with an invented citation",
            tampered,
            verify_citations(tampered, retrieved_ids),
        )


if __name__ == "__main__":
    main()
