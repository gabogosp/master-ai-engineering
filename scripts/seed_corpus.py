"""Siembra el corpus de presupuestos históricos en la base.

Uso::

    uv run python scripts/seed_corpus.py
    uv run python scripts/seed_corpus.py --base-url http://localhost:8000

Un documento = un presupuesto, así que el corpus completo (hoy,
`data/budgets_sample.json`) se sube con una llamada por presupuesto a
`POST /embeddings/ingest`. Un documento ya ingestado (409) se reporta
y se sigue con el resto -- correr este script dos veces no debería
fallar ni duplicar nada.
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

# Relativo al repo, no resuelto a absoluto: `source_path` se persiste
# como identificador estable de dedupe -- si fuera la ruta absoluta del
# checkout, el mismo corpus clonado en otra máquina (u otro path)
# generaría documentos "nuevos" para presupuestos ya ingestados.
DEFAULT_CORPUS_PATH = Path("data/budgets_sample.json")


def seed(base_url: str, corpus_path: Path) -> int:
    budgets = json.loads(corpus_path.read_text())

    ingested = 0
    already_present = 0
    failed = 0

    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        for budget in budgets:
            payload = {
                "source_path": f"{corpus_path.as_posix()}#{budget['budget_id']}",
                "document_type": "historical_budget",
                "content": budget,
            }
            response = client.post("/embeddings/ingest", json=payload)

            if response.status_code == 200:
                body = response.json()
                print(
                    f"[ingestado] {budget['budget_id']} -> document_id={body['document_id']} "
                    f"chunks={body['chunks_created']}"
                )
                ingested += 1
            elif response.status_code == 409:
                print(f"[ya existía] {budget['budget_id']}")
                already_present += 1
            else:
                print(f"[FALLÓ] {budget['budget_id']}: {response.status_code} {response.text}")
                failed += 1

    print(
        f"\nTotal: {len(budgets)} presupuestos -- {ingested} ingestados, "
        f"{already_present} ya existían, {failed} fallaron."
    )
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=settings.backend_url)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    args = parser.parse_args()

    return seed(args.base_url, args.corpus)


if __name__ == "__main__":
    raise SystemExit(main())
