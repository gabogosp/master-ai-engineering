"""CLI de similitud coseno entre dos textos (sesión 7, Paso 6).

Uso::

    uv run python scripts/compare.py --text-a "..." --text-b "..."

Reusa `OpenAIEmbedder.embed_one` (Paso 4). La similitud coseno se
calcula a mano -- producto escalar sobre producto de normas -- con la
biblioteca estándar; el enunciado pide explícitamente no importar
numpy ni scikit-learn solo para esto.
"""

import argparse
import math
import sys
from pathlib import Path

# El enunciado pide invocar este script como archivo plano
# ("python scripts/compare.py", no "python -m scripts.compare"), lo
# que deja `scripts/` -- no la raíz del repo -- como primera entrada de
# sys.path. Sin este ajuste, `from app...` falla con ModuleNotFoundError
# en cuanto se invoca tal cual lo pide el enunciado. Mismo truco que usa
# uvicorn internamente al cargar "app.main:app" por string.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-a", required=True)
    parser.add_argument("--text-b", required=True)
    args = parser.parse_args()

    embedder = OpenAIEmbedder()
    vector_a = embedder.embed_one(args.text_a)
    vector_b = embedder.embed_one(args.text_b)
    similarity = cosine_similarity(vector_a, vector_b)

    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {similarity:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
