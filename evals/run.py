"""CLI runner: corre el dataset dorado contra la app FastAPI en vivo.

El transporte por defecto es el `TestClient` in-process -- no hace falta
levantar un server aparte, así que la suite se puede correr sola. `--http
BASE` cambia a un target HTTP real (ej. para medir contra el server de
`uvicorn` corriendo con `--reload`, más representativo de latencia real).

Uso::

    uv run python -m evals.run --mode actor
    uv run python -m evals.run --mode acb --limit 5 --output /tmp/acb.json
    uv run python -m evals.run --mode actor --http http://localhost:8000

El runner imprime una fila compacta por caso + métrica, y un resumen al
final. `--output` persiste el reporte completo (scores por caso + meta)
como JSON para poder diffear entre corridas.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import EstimationResult
from evals.dataset import GoldenCase, load_dataset
from evals.metrics import MetricResult, run_all_metrics

Mode = Literal["actor", "acb"]

API_PREFIX = "/api/v1"

# Las violaciones de guardrail de input que, para un caso marcado
# expected_out_of_scope, CUENTAN como el resultado esperado -- un 400 por
# moderación/PII/prompt-injection es, en los hechos, el mismo "no se puede
# estimar esto" que un confidence_pct < 30 del LLM.
_GUARDRAIL_OUT_OF_SCOPE_REASONS = {"prompt_injection", "moderation", "pii"}


def _form_body(case: GoldenCase) -> dict[str, str]:
    body = {
        "transcript": case.transcript,
        "project_type": case.project_type,
        "detail_level": case.detail_level,
        "output_format": case.output_format,
    }
    if case.tier and case.tier != "auto":
        body["tier"] = case.tier
    return body


def _guardrail_payload(reason: str, message: str) -> dict[str, Any]:
    """Sintetiza un payload con forma de EstimationResponse para un rechazo
    de guardrail en capa 1.

    Cuando el guardrail de input bloquea un transcript adversarial/inseguro
    antes de llamar al LLM, eso *es* el resultado out-of-scope. Lo mapeamos
    al mismo sobre que hubiera producido el LLM (cost=0, duration=1,
    summary con el prefijo "Out of scope:") para que las métricas existentes
    lo puntúen como pass en los casos que esperan out_of_scope=true.
    """
    summary = f"Out of scope: rechazado por el guardrail de input ({reason}): {message}"[:1200]
    return {
        "result": {
            "summary": summary,
            "confidence_pct": 0,
            "phases": [
                {
                    "name": "n/a",
                    "duration_weeks": 1,
                    "cost_eur": 0,
                    "summary": "Bloqueado por el guardrail de input; no se generó estimación.",
                }
            ],
            "total_duration_weeks": 1,
            "total_cost_eur": 0,
        },
        "prompt_version": f"guardrail/{reason}",
        "cached": False,
    }


def _maybe_guardrail_payload(case: GoldenCase, response: Any) -> dict[str, Any] | None:
    if response.status_code != 400 or not case.expected_out_of_scope:
        return None
    try:
        detail = response.json().get("detail") or {}
    except Exception:
        return None
    reason = detail.get("reason") if isinstance(detail, dict) else None
    if reason not in _GUARDRAIL_OUT_OF_SCOPE_REASONS:
        return None
    return _guardrail_payload(reason, detail.get("message", ""))


def _endpoint(session_id: str, mode: Mode) -> str:
    suffix = "estimate-acb" if mode == "acb" else "estimate"
    return f"{API_PREFIX}/sessions/{session_id}/{suffix}"


def _run_case_in_process(client: TestClient, case: GoldenCase, mode: Mode) -> tuple[dict[str, Any], int]:
    sid = client.post(f"{API_PREFIX}/sessions").json()["session_id"]
    t0 = time.perf_counter()
    response = client.post(_endpoint(sid, mode), data=_form_body(case))
    latency_ms = int((time.perf_counter() - t0) * 1000)
    synthetic = _maybe_guardrail_payload(case, response)
    if synthetic is not None:
        return synthetic, latency_ms
    response.raise_for_status()
    return response.json(), latency_ms


def _run_case_over_http(case: GoldenCase, mode: Mode, base_url: str) -> tuple[dict[str, Any], int]:
    with httpx.Client(base_url=base_url, timeout=180.0) as client:
        sid = client.post(f"{API_PREFIX}/sessions").json()["session_id"]
        t0 = time.perf_counter()
        response = client.post(_endpoint(sid, mode), data=_form_body(case))
        latency_ms = int((time.perf_counter() - t0) * 1000)
        synthetic = _maybe_guardrail_payload(case, response)
        if synthetic is not None:
            return synthetic, latency_ms
        response.raise_for_status()
        return response.json(), latency_ms


def _evaluate(case: GoldenCase, payload: dict[str, Any]) -> list[MetricResult]:
    result = EstimationResult.model_validate(payload["result"])
    return run_all_metrics(case, result)


def _format_row(case_id: str, latency_ms: int, metrics: list[MetricResult]) -> str:
    parts = [f"{case_id:<28}", f"{latency_ms:>6} ms"]
    for m in metrics:
        mark = "PASS" if m.passed else "FAIL"
        parts.append(f"{m.name}={mark}({m.score:.2f})")
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("actor", "acb"), default="actor")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--http", default=None, help="Base URL para modo HTTP (ej. http://localhost:8000)"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cases = load_dataset()
    if args.limit is not None:
        cases = cases[: args.limit]

    rows: list[dict[str, Any]] = []
    pass_count = 0
    metric_totals: dict[str, list[bool]] = {}

    print(f"Corriendo {len(cases)} casos con mode={args.mode}")

    def _run_all(in_process_client: TestClient | None) -> None:
        nonlocal pass_count
        for case in cases:
            try:
                if args.http:
                    payload, latency_ms = _run_case_over_http(case, args.mode, args.http)
                else:
                    assert in_process_client is not None
                    payload, latency_ms = _run_case_in_process(in_process_client, case, args.mode)
            except Exception as exc:
                print(f"{case.id}: ERROR {type(exc).__name__}: {str(exc)[:200]}")
                rows.append({"id": case.id, "error": str(exc)[:400]})
                continue

            metrics = _evaluate(case, payload)
            print(_format_row(case.id, latency_ms, metrics))

            row_passed = all(m.passed for m in metrics)
            if row_passed:
                pass_count += 1
            rows.append(
                {
                    "id": case.id,
                    "latency_ms": latency_ms,
                    "passed": row_passed,
                    "metrics": [m.__dict__ for m in metrics],
                    "result_summary": payload["result"]["summary"][:200],
                    "prompt_version": payload.get("prompt_version"),
                    "acb_final_decision": payload.get("acb", {}).get("final_decision"),
                }
            )
            for m in metrics:
                metric_totals.setdefault(m.name, []).append(m.passed)

    if args.http:
        _run_all(None)
    else:
        # `with` es necesario (no un TestClient suelto): todas las
        # llamadas del batch tienen que compartir un único event loop, o
        # los singletons async con estado (el cliente de Redis) terminan
        # atados a un loop que ya se cerró -- mismo problema que ya
        # documentamos en tests/conftest.py.
        with TestClient(app) as in_process_client:
            _run_all(in_process_client)

    print("\nResumen")
    print(f"  {pass_count}/{len(cases)} casos pasando todas las métricas")
    for name, results in metric_totals.items():
        print(f"  {name}: {sum(results)}/{len(results)}")

    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "cases_total": len(cases),
                    "cases_passing": pass_count,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nReporte escrito en {args.output}")

    return 0 if pass_count == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
