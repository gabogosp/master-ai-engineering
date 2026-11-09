"""Mide precisión@5 y latencia de las cuatro configuraciones de retrieval
(A: vectorial, B: híbrida, C: vectorial+rerank, D: híbrida+rerank) contra
un golden set de 5 consultas anotadas a mano.

    PYTHONPATH=. uv run python scripts/golden_set_eval.py

Le pega al endpoint /search real (BACKEND_URL, default localhost:8000),
nunca mockea nada -- igual que trace_embed_vector.py, el objetivo es
evidencia reproducible, no una simulación.

El golden set está escrito en inglés a propósito: el corpus real
(data/budgets_sample.json) está en inglés, y una query en español no
tendría overlap léxico con él -- la rama híbrida no tendría nada que
demostrar.
"""

import os
import time

import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Cada entrada: la consulta y el conjunto de budget_id realmente
# relevantes para ella, anotados a mano contra los 15 presupuestos del
# corpus (ver data/budgets_sample.json).
#
# Las dos primeras son deliberadamente ambiguas en el vocabulario de
# "pagos" (el caso que motiva el ejercicio: un presupuesto puede estar
# cerca en el espacio vectorial y no ser el proyecto correcto). Las
# otras tres son casos limpios de un solo sector, para contrastar.
GOLDEN_SET = [
    {
        # Describe una tienda online con checkout -- el proyecto
        # correcto es el storefront de e-commerce, NO el gateway de
        # pagos de finance, aunque ambos mencionen "payment"/"checkout".
        "query": "Online payment processing for checkout and order management",
        "relevant_budget_ids": {"BUD-2024-005"},
    },
    {
        # Simétrico al anterior: acá sí es, de verdad, un proyecto de
        # pagos de finance -- mismo vocabulario de superficie, dominio
        # correcto distinto.
        "query": "Real-time payment gateway with fraud detection and settlement reconciliation with the acquirer",
        "relevant_budget_ids": {"BUD-2024-003"},
    },
    {
        "query": "Remote patient monitoring platform ingesting wearable device data with alerts",
        "relevant_budget_ids": {"BUD-2024-010"},
    },
    {
        "query": "Factory IoT telemetry platform for predictive maintenance",
        "relevant_budget_ids": {"BUD-2024-013"},
    },
    {
        "query": "Multi-vendor marketplace with vendor payouts and buyer-seller dispute handling",
        "relevant_budget_ids": {"BUD-2024-006"},
    },
]

CONFIGURATIONS = [
    ("A", "vector", False),
    ("B", "hybrid", False),
    ("C", "vector", True),
    ("D", "hybrid", True),
]

K = 5


def _warm_up_reranker(client: httpx.Client) -> None:
    """La primera llamada con rerank=True carga el modelo (~20-30s en
    CPU). Si no se precalienta, ese costo de carga -- que es un costo
    de arranque del proceso, no de la consulta -- contaminaría la
    latencia medida de la primera configuración con rerank."""
    print("Precalentando el reranker (carga del modelo, no se mide)...")
    client.post(
        f"{BACKEND_URL}/search",
        json={"query": "warm up", "k": 1, "search_mode": "vector", "rerank": True},
        timeout=120,
    )
    print("Reranker listo.\n")


def _precision_at_k(results: list[dict], relevant_budget_ids: set[str]) -> float:
    if not results:
        return 0.0
    hits = sum(1 for r in results if r["metadata"].get("budget_id") in relevant_budget_ids)
    return hits / len(results)


def _reciprocal_rank(results: list[dict], relevant_budget_ids: set[str]) -> float:
    """1 / posición (1-based) del primer resultado relevante; 0 si no
    hay ninguno en el top-k. Precision@5 cuenta CUÁNTOS resultados son
    relevantes pero es ciega al ORDEN -- si reordenar sube el único
    resultado relevante de la posición 2 a la 1, precision@5 no se
    mueve pero el reciprocal rank sí (0.5 -> 1.0). Complementa la
    métrica pedida en vez de reemplazarla."""
    for position, result in enumerate(results, start=1):
        if result["metadata"].get("budget_id") in relevant_budget_ids:
            return 1.0 / position
    return 0.0


def main() -> None:
    with httpx.Client() as client:
        _warm_up_reranker(client)

        summary: dict[str, list[float]] = {}
        print(f"{'Config':<8}{'Query':<70}{'Precision@5':<14}{'MRR@5':<10}{'Latency (ms)':<14}")
        print("-" * 116)

        for label, search_mode, rerank in CONFIGURATIONS:
            precisions = []
            reciprocal_ranks = []
            latencies = []
            for case in GOLDEN_SET:
                started = time.perf_counter()
                response = client.post(
                    f"{BACKEND_URL}/search",
                    json={
                        "query": case["query"],
                        "k": K,
                        "search_mode": search_mode,
                        "rerank": rerank,
                    },
                    timeout=60,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                body = response.json()
                precision = _precision_at_k(body["results"], case["relevant_budget_ids"])
                reciprocal_rank = _reciprocal_rank(body["results"], case["relevant_budget_ids"])

                precisions.append(precision)
                reciprocal_ranks.append(reciprocal_rank)
                latencies.append(elapsed_ms)
                print(
                    f"{label:<8}{case['query'][:68]:<70}{precision:<14.2f}"
                    f"{reciprocal_rank:<10.2f}{elapsed_ms:<14.1f}"
                )

            summary[label] = [
                sum(precisions) / len(precisions),
                sum(reciprocal_ranks) / len(reciprocal_ranks),
                sum(latencies) / len(latencies),
            ]

        print("\n=== Promedio por configuración (5 queries) ===")
        print(
            f"{'Config':<8}{'Search':<10}{'Rerank':<10}{'Precision@5':<14}"
            f"{'MRR@5':<10}{'Latency (ms)':<14}"
        )
        print("-" * 66)
        for label, search_mode, rerank in CONFIGURATIONS:
            avg_precision, avg_mrr, avg_latency = summary[label]
            print(
                f"{label:<8}{search_mode:<10}{str(rerank):<10}{avg_precision:<14.2f}"
                f"{avg_mrr:<10.2f}{avg_latency:<14.1f}"
            )


if __name__ == "__main__":
    main()
