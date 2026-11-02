"""Imprime la huella del embedding de un texto arbitrario (dimensión,
norma, primer y último componente) contra el módulo de embeddings del
pipeline -- no contra /embeddings/ingest, que solo acepta un `Budget`
estructurado, no texto libre como una transcripción.
"""

import sys
from pathlib import Path

from app.embedding_pipeline.embedder import OpenAIEmbedder


def main(path: str) -> None:
    text = Path(path).read_text()
    vector = OpenAIEmbedder().embed_one(text)
    norm = sum(component * component for component in vector) ** 0.5

    print(f"dimension={len(vector)}")
    print(f"norm={norm:.6f}")
    print(f"first_component={vector[0]:.6f}")
    print(f"last_component={vector[-1]:.6f}")


if __name__ == "__main__":
    main(sys.argv[1])
