"""Chequeo previo: ¿el cross-encoder descarga, carga y rankea bien?

Correr ANTES de tocar el pipeline de retrieval -- si esto falla, no tiene
sentido seguir con la integración.

    PYTHONPATH=. uv run python scripts/verify_reranker.py

Usa un par de sanity obvio (un documento responde la consulta, el otro
no) y falla si el modelo no los distingue en el orden esperado.
"""

import sys

from app.config import settings
from app.embedding_pipeline.reranker import CrossEncoderReranker

_QUERY = "e-commerce checkout and shopping cart platform"
_DOCUMENTS = [
    "Headless e-commerce storefront with cart, checkout and order management.",
    "Hospital patient appointment scheduling and telemedicine consultations.",
]


def main() -> int:
    reranker = CrossEncoderReranker(settings.reranker_model)
    print(f"Cargando modelo de reranking: {reranker.model_name} ...")

    try:
        scores = reranker.score(_QUERY, _DOCUMENTS)
    except Exception as exc:  # noqa: BLE001 -- cualquier falla de carga/scoring debe verse.
        print(f"FALLÓ la carga o el scoring: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"query={_QUERY!r}")
    print(f"doc_relevante:   score={scores[0]:.4f}")
    print(f"doc_irrelevante: score={scores[1]:.4f}")

    if scores[0] <= scores[1]:
        print(
            "ADVERTENCIA: el documento relevante no superó al irrelevante. "
            "El modelo cargó pero su ranking se ve mal -- revisar el nombre del modelo.",
            file=sys.stderr,
        )
        return 2

    print("OK: el reranker cargó y priorizó correctamente el documento relevante.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
