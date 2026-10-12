"""Runner de stress-test (sesión 6, Bloque 5 / Paso 5): corre los tres
escenarios de `evals/stress/scenarios.py` contra sesiones reales, cruzando
tamaño de adjunto y repeticiones, y vuelca una fila por turno a un CSV.

Uso::

    uv run python -m evals.stress.run \\
        --scenarios growing,pivot,contradiction \\
        --attachment-sizes 0,5,20,50,100 \\
        --repeats 3 \\
        --output evals/stress/results.csv

    # Contra un server real levantado aparte (más representativo de
    # latencia real que in-process):
    uv run python -m evals.stress.run --http http://localhost:8000 ...

El enunciado deja varias decisiones de diseño a criterio propio -- quedan
documentadas acá porque cada una tiene un motivo concreto, no es relleno:

1. Transporte: igual que `evals/run.py`, por defecto corre IN-PROCESS
   contra la app FastAPI con `httpx.AsyncClient` + `ASGITransport` (el
   mismo patrón que ya usa `tests/test_sessions_integration.py`). A
   diferencia de un `TestClient` síncrono, esto permite lanzar varias
   conversaciones EN PARALELO (`--concurrency`) sobre un único event
   loop, que es la única forma realista de correr el grid completo
   (3 escenarios × 5 tamaños de adjunto × repeats, 20 turnos cada una)
   sin que la corrida tome horas.

2. Cómo se lee `turn_observed`: NO se parsean logs (`docker compose
   logs | grep turn_observed`, que es la opción que sugiere la
   consigna). Desde el Bloque 4, `POST .../estimate` devuelve el campo
   `observation` -- un `TurnObservation` completo -- directo en el body
   de la respuesta. Cada turno ya trae su propia telemetría sin una
   llamada ni un parseo de log aparte. Sólo se complementa con un
   `GET /sessions/{id}` extra por turno para la metadata/resumen que
   necesita `MemoryDriftMetric` (no vive en `TurnObservation`, que es
   telemetría de UN turno, no un snapshot acumulado).

3. Dónde se adjunta el PDF: SOLO en el primer turno de cada
   conversación, nunca en los 20. El adjunto representa "el cliente
   comparte un documento de referencia al arrancar la conversación";
   adjuntarlo en cada turno multiplicaría el costo por 20 sin aportar
   una señal nueva, porque el texto extraído queda tejido en el prompt
   que se persiste al historial desde el primer turno (ver
   `EstimationService._persist_turn`) -- los turnos siguientes ya cargan
   con ese peso aunque no se reenvíe el archivo.

4. La "N" de la curva de recall: la consigna no pide truncar cada
   escenario a N ∈ {1,3,6,10,20} como hizo el Bloque 2 -- cada
   conversación corre sus 20 turnos completos UNA sola vez, y el
   `turn_index` de cada fila ES la N. Después de cada turno se evalúa
   `MemoryDriftMetric` contra CADA hecho introducido en los turnos
   1..turn_index (vía el `GET /sessions/{id}` de ese momento); el
   promedio de esos resultados es `memory_drift_recall` para esa fila.
   Eso da la curva "recall vs N" sin correr el escenario 5 veces por
   separado.

5. Cache semántica: el flujo conversacional nunca pasa por ahí (solo
   está wireada en el endpoint de formulario de un solo turno) -- la
   columna de esa tasa de hit en el reporte va a salir en 0% siempre, a
   propósito. No es un bug del runner ni de la cache.

6. Presupuestos de `LatencyBudgetMetric`/`CostBudgetMetric`:
   configurables por CLI, con defaults pensados como cota de sanidad
   POR TURNO (no acumulada): 8000 ms y 0.01 USD.
"""

import argparse
import asyncio
import csv
import statistics
from pathlib import Path
from typing import Any

import httpx
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.schemas import TurnObservation
from evals.stress.metrics import CostBudgetMetric, FactField, LatencyBudgetMetric, MemoryDriftMetric
from evals.stress.scenarios import Scenario, ScenarioTurn, get_scenario

API_PREFIX = "/api/v1"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

FIELDNAMES = [
    "scenario",
    "attachment_kb",
    "repeat",
    "turn_index",
    "session_id",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "cost_usd_cumulative",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
    "latency_budget_pass",
    "cost_budget_pass",
    "memory_drift_recall",
    "memory_drift_pass",
]


def _attachment_path_for(attachment_kb: int) -> Path | None:
    """0 KB = no adjuntar nada (igual que `build_pdfs.py`: no genera un
    archivo de 0 KB, directamente no hay archivo que adjuntar)."""
    if attachment_kb == 0:
        return None
    path = FIXTURES_DIR / f"attach_{attachment_kb}kb.pdf"
    if not path.exists():
        raise FileNotFoundError(
            f"falta {path.name} -- generalo con "
            "`uv run python -m evals.stress.fixtures.build_pdfs`"
        )
    return path


def _percentile(values: list[float], pct: float) -> float:
    """Percentil simple por interpolación lineal (sin pandas/numpy, para
    no sumar una dependencia solo por esto). `pct` en [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _build_row(
    *,
    scenario_name: str,
    attachment_kb: int,
    repeat: int,
    observation: TurnObservation,
    cost_usd_cumulative: float,
    latency_budget_ms: float,
    cost_budget_usd: float,
    tracked_facts: list[tuple[str, FactField]],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Arma la fila de CSV para un turno: los 13 campos de
    `TurnObservation` tal cual, más el acumulado de costo y las 3
    métricas (2 presupuestos por turno + recall de memoria)."""
    latency_result = LatencyBudgetMetric(latency_budget_ms).evaluate(observation)
    cost_result = CostBudgetMetric(cost_budget_usd).evaluate(observation)

    drift_scores = [
        MemoryDriftMetric(fact, field).evaluate(snapshot).score for fact, field in tracked_facts
    ]
    recall = sum(drift_scores) / len(drift_scores) if drift_scores else 1.0

    return {
        "scenario": scenario_name,
        "attachment_kb": attachment_kb,
        "repeat": repeat,
        "turn_index": observation.turn_index,
        "session_id": observation.session_id,
        "enriched_transcript_chars": observation.enriched_transcript_chars,
        "attachments_total_chars": observation.attachments_total_chars,
        "messages_in_window": observation.messages_in_window,
        "anchors_count": observation.anchors_count,
        "summary_chars": observation.summary_chars,
        "tokens_in": observation.tokens_in,
        "tokens_out": observation.tokens_out,
        "cost_usd": observation.cost_usd,
        "cost_usd_cumulative": cost_usd_cumulative,
        "latency_ms": observation.latency_ms,
        "cache_hit_kind": observation.cache_hit_kind,
        "last_resolved_tier": observation.last_resolved_tier,
        "latency_budget_pass": latency_result.passed,
        "cost_budget_pass": cost_result.passed,
        "memory_drift_recall": round(recall, 4),
        "memory_drift_pass": recall >= 1.0,
    }


def _form_body(turn: ScenarioTurn, scenario: Scenario) -> dict[str, str]:
    return {
        "transcript": turn.transcript,
        "project_type": scenario.project_type.value,
        "detail_level": scenario.detail_level.value,
        "output_format": scenario.output_format.value,
    }


async def _run_conversation(
    client: AsyncClient,
    semaphore: asyncio.Semaphore,
    scenario: Scenario,
    attachment_kb: int,
    repeat: int,
    latency_budget_ms: float,
    cost_budget_usd: float,
) -> tuple[list[dict[str, Any]], bool]:
    """Corre UNA conversación completa (los `len(scenario.turns)` turnos,
    en orden, sobre una única sesión) y devuelve (filas, completó_todo).
    Las conversaciones entre sí son independientes (sesiones distintas)
    -- por eso `semaphore` acota cuántas corren a la vez, pero los turnos
    DENTRO de una conversación son estrictamente secuenciales (el turno N
    depende del estado que dejó el N-1).

    `completó_todo=False` no es necesariamente un bug del runner: una
    conversación larga puede chocar con un fallo real del LLM (ej.
    Instructor agota los reintentos de validación aritmética) -- eso ES
    una de las formas en que el CAG se rompe bajo estrés, así que se
    cuenta y se reporta, no se esconde."""
    attachment_path = _attachment_path_for(attachment_kb)
    rows: list[dict[str, Any]] = []
    completed = True

    async with semaphore:
        session_resp = await client.post(f"{API_PREFIX}/sessions")
        session_resp.raise_for_status()
        session_id = session_resp.json()["session_id"]

        cost_usd_cumulative = 0.0
        tracked_facts: list[tuple[str, FactField]] = []

        for turn in scenario.turns:
            kwargs: dict[str, Any] = {"data": _form_body(turn, scenario)}
            if turn.turn_index == 1 and attachment_path is not None:
                kwargs["files"] = {
                    "attachments": (attachment_path.name, attachment_path.read_bytes(), "application/pdf")
                }
            try:
                response = await client.post(
                    f"{API_PREFIX}/sessions/{session_id}/estimate", **kwargs
                )
                response.raise_for_status()
                payload = response.json()
                observation = TurnObservation.model_validate(payload["observation"])

                snapshot_resp = await client.get(f"{API_PREFIX}/sessions/{session_id}")
                snapshot_resp.raise_for_status()
                snapshot = snapshot_resp.json()
            except (httpx.HTTPStatusError, httpx.HTTPError) as exc:
                print(
                    f"ERROR scenario={scenario.name} attachment_kb={attachment_kb} "
                    f"repeat={repeat} turn={turn.turn_index}: {exc}"
                )
                completed = False
                break  # sesión en estado inconsistente -- no seguir sumando turnos

            cost_usd_cumulative += observation.cost_usd
            if turn.fact_to_remember:
                tracked_facts.append((turn.fact_to_remember, turn.fact_field))

            rows.append(
                _build_row(
                    scenario_name=scenario.name,
                    attachment_kb=attachment_kb,
                    repeat=repeat,
                    observation=observation,
                    cost_usd_cumulative=cost_usd_cumulative,
                    latency_budget_ms=latency_budget_ms,
                    cost_budget_usd=cost_budget_usd,
                    tracked_facts=tracked_facts,
                    snapshot=snapshot,
                )
            )

    return rows, completed


def _make_client(http_base: str | None) -> AsyncClient:
    if http_base:
        return AsyncClient(base_url=http_base, timeout=180.0)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=180.0)


async def _amain(args: argparse.Namespace) -> int:
    scenarios = [get_scenario(name) for name in args.scenarios.split(",")]
    attachment_sizes = [int(x) for x in args.attachment_sizes.split(",")]
    semaphore = asyncio.Semaphore(args.concurrency)

    jobs_desc = (
        f"{len(scenarios)} escenarios × {len(attachment_sizes)} tamaños de adjunto × "
        f"{args.repeats} repeats = {len(scenarios) * len(attachment_sizes) * args.repeats} "
        "conversaciones"
    )
    print(f"Corriendo {jobs_desc} (concurrency={args.concurrency})")

    async with _make_client(args.http) as client:
        tasks = [
            _run_conversation(
                client, semaphore, scenario, attachment_kb, repeat,
                args.latency_budget_ms, args.cost_budget_usd,
            )
            for scenario in scenarios
            for attachment_kb in attachment_sizes
            for repeat in range(args.repeats)
        ]
        results = await asyncio.gather(*tasks)

    rows = [row for conversation_rows, _ in results for row in conversation_rows]
    completed_conversations = sum(1 for _, completed in results if completed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    total_turns = sum(len(s.turns) for s in scenarios) * len(attachment_sizes) * args.repeats
    print(f"\n{len(rows)}/{total_turns} turnos completados. CSV en {args.output}")
    print(f"  {completed_conversations}/{len(results)} conversaciones llegaron hasta el último turno")
    if rows:
        latencies = [r["latency_ms"] for r in rows]
        print(f"  latency_ms P50={_percentile(latencies, 50):.0f} P95={_percentile(latencies, 95):.0f}")
        print(f"  costo total USD={sum(r['cost_usd'] for r in rows):.4f}")
        print(f"  recall medio={statistics.mean(r['memory_drift_recall'] for r in rows):.3f}")

    return 0 if len(rows) == total_turns else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", default="growing,pivot,contradiction")
    parser.add_argument("--attachment-sizes", default="0,5,20,50,100")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--latency-budget-ms", type=float, default=8000.0)
    parser.add_argument("--cost-budget-usd", type=float, default=0.01)
    parser.add_argument("--http", default=None, help="Base URL para modo HTTP (ej. http://localhost:8000)")
    parser.add_argument("--output", type=Path, default=Path("evals/stress/results.csv"))
    args = parser.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
