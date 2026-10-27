"""Ejercita /search con cinco queries representativas del corpus.

Uso::

    uv run python scripts/query_examples.py
    uv run python scripts/query_examples.py --base-url http://localhost:8000

Reemplaza a `compare.py` (medía similitud entre dos textos sueltos,
sin tocar la base). Estas cinco queries están elegidas para mirar el
corpus desde ángulos distintos: un match casi perfecto, una
reformulación con vocabulario distinto, un dominio ajeno al corpus,
una consulta ambigua y una muy específica -- cada una ejercita una
forma diferente de fallar o acertar en la búsqueda semántica.
"""

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

QUERIES = [
    "REST API development with JWT authentication for financial sector",
    "secure backend service with token-based access control for banking applications",
    "mobile application for restaurant reservations",
    "integration with external system",
    "migration from monolith to microservices architecture using Kubernetes",
]

CONTENT_PREVIEW_CHARS = 120


def run_query(client: httpx.Client, query: str, k: int = 5) -> None:
    response = client.post("/search", json={"query": query, "k": k})
    response.raise_for_status()
    body = response.json()

    print(f"\nQuery: {query}")
    print(f"search_time_ms: {body['search_time_ms']}")
    print("-" * 100)
    for result in body["results"]:
        preview = result["content"].replace("\n", " ")[:CONTENT_PREVIEW_CHARS]
        print(
            f"  chunk_id={result['chunk_id']:<6} distance={result['distance']:.4f}  "
            f"chunk_type={result['chunk_type']:<18} {preview}..."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=settings.backend_url)
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        for query in QUERIES:
            run_query(client, query)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
