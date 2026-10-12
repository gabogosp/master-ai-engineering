"""Genera evals/stress/REPORT.md a partir de evals/stress/results.csv
(sesión 6, Bloque 5 / Paso 6).

Uso::

    uv run python -m evals.stress.report \\
        --input evals/stress/results.csv --output evals/stress/REPORT.md

Todo el reporte (tablas Y los dos párrafos de lectura) se calcula a partir
de los datos reales del CSV -- ningún número está hardcodeado. Los
párrafos de "dónde se rompe el CAG" se construyen con umbrales simples
(primer N con recall medio < 0.8, qué dimensión creció más entre el
primer y el último turno) aplicados a lo que la corrida real produjo, no
a una hipótesis previa.

Un detalle no obvio: el turno 1 de cada (escenario, tamaño de adjunto) es
idéntico carácter a carácter entre repeats (misma transcripción inicial,
sin historial todavía), así que la cache exact-match de Redis (ver
`app/services/cache.py::_cache_key_for_messages`, que hashea los mensajes
completos) le da HIT a 2 de cada 3 repeats en ese turno -- cache_hit_kind
="exact" con cost_usd=latency_ms=0. Es comportamiento real de producción,
no un bug del runner, pero si se mete sin filtrar en los percentiles de
latencia los deja artificialmente bajos. Por eso los cálculos de
latencia EXCLUYEN filas con cache hit (`_misses`); el costo total y el
hit rate SÍ se calculan sobre todas las filas, porque ahí el cache hit
es justamente el dato que se quiere medir.
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.stress.run import _percentile


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["turn_index"] = int(row["turn_index"])
        row["attachment_kb"] = int(row["attachment_kb"])
        row["repeat"] = int(row["repeat"])
        row["anchors_count"] = int(row["anchors_count"])
        row["latency_ms"] = float(row["latency_ms"])
        row["cost_usd"] = float(row["cost_usd"])
        row["cost_usd_cumulative"] = float(row["cost_usd_cumulative"])
        row["tokens_in"] = int(row["tokens_in"])
        row["memory_drift_recall"] = float(row["memory_drift_recall"])
        row["memory_drift_pass"] = row["memory_drift_pass"] == "True"
        row["latency_budget_pass"] = row["latency_budget_pass"] == "True"
        row["cost_budget_pass"] = row["cost_budget_pass"] == "True"
    return rows


def _misses(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["cache_hit_kind"] != "exact"]


def _summary_table(rows: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["attachment_kb"])].append(row)

    lines = [
        "| Escenario | Adjunto (KB) | P50 latency (ms) | P95 latency (ms) | Costo total (USD) "
        "| Exact cache hit | Semantic cache hit | Recall medio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (scenario, kb), group in sorted(groups.items()):
        latencies = [r["latency_ms"] for r in _misses(group)]
        exact_rate = sum(1 for r in group if r["cache_hit_kind"] == "exact") / len(group)
        semantic_rate = sum(1 for r in group if r["cache_hit_kind"] == "semantic") / len(group)
        total_cost = sum(r["cost_usd"] for r in group)
        mean_recall = statistics.mean(r["memory_drift_recall"] for r in group)
        lines.append(
            f"| {scenario} | {kb} | {_percentile(latencies, 50):.0f} | "
            f"{_percentile(latencies, 95):.0f} | {total_cost:.4f} | {exact_rate:.0%} | "
            f"{semantic_rate:.0%} | {mean_recall:.2f} |"
        )
    return "\n".join(lines)


def _latency_vs_tokens_curve(rows: list[dict[str, Any]]) -> str:
    """Bins de tokens_in en deciles -- no hay suficientes valores únicos
    como para que un bin por valor tenga sentido, y un decil alcanza para
    ver la tendencia sin un gráfico."""
    misses = _misses(rows)
    if not misses:
        return "_(sin datos -- todas las filas fueron cache hit)_"
    ordered = sorted(misses, key=lambda r: r["tokens_in"])
    n_bins = min(10, len(ordered))
    bin_size = max(1, len(ordered) // n_bins)

    lines = ["| tokens_in (rango) | latency_ms (media) | n |", "|---|---|---|"]
    for i in range(0, len(ordered), bin_size):
        chunk = ordered[i : i + bin_size]
        lo, hi = chunk[0]["tokens_in"], chunk[-1]["tokens_in"]
        mean_latency = statistics.mean(r["latency_ms"] for r in chunk)
        lines.append(f"| {lo}-{hi} | {mean_latency:.0f} | {len(chunk)} |")
    return "\n".join(lines)


def _cost_cumulative_vs_turn_curve(rows: list[dict[str, Any]]) -> str:
    """`n` total (suma de las 3 columnas de escenario) al final de cada
    fila: como el runner corta la conversación apenas un turno falla
    (ver `_completion_stats`), son cada vez menos sesiones las que
    llegan a un N alto -- el número de filas que sostienen el promedio
    de los últimos turnos NO es el mismo que en el turno 1, y hay que
    poder verlo para no leer la cola de la curva como si tuviera la
    misma confianza que el principio."""
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["turn_index"])].append(row["cost_usd_cumulative"])

    scenarios = sorted({s for s, _ in groups})
    turn_indices = sorted({t for _, t in groups})
    header = "| turn_index | " + " | ".join(scenarios) + " | n |"
    sep = "|---|" + "---|" * len(scenarios) + "---|"
    lines = [header, sep]
    for t in turn_indices:
        cells = [f"{statistics.mean(groups[(s, t)]):.4f}" if (s, t) in groups else "-" for s in scenarios]
        n = sum(len(groups.get((s, t), [])) for s in scenarios)
        lines.append(f"| {t} | " + " | ".join(cells) + f" | {n} |")
    return "\n".join(lines)


def _recall_vs_turn_curve(rows: list[dict[str, Any]]) -> str:
    groups: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        groups[row["turn_index"]].append(row["memory_drift_recall"])

    lines = ["| turn_index (N) | MemoryDriftMetric (recall medio) | n |", "|---|---|---|"]
    for t in sorted(groups):
        lines.append(f"| {t} | {statistics.mean(groups[t]):.2f} | {len(groups[t])} |")
    return "\n".join(lines)


def _completion_stats(
    rows: list[dict[str, Any]], total_conversations: int | None = None
) -> tuple[int, int, float]:
    """(conversaciones_totales, conversaciones_completas, tasa). Una
    conversación "completa" es la que llegó a turn_index == 20 -- el
    runner corta una conversación apenas un turno falla, así que cualquier
    sesión con turn_index máximo < 20 es una conversación que se rompió a
    mitad de camino.

    El total NO se puede reconstruir siempre solo del CSV: una
    conversación que falla en su PRIMER turno no deja ninguna fila (no
    hay `session_id` que contar), así que queda invisible para
    `len(max_turn_by_session)`. `evals.stress.run` sí conoce el total real
    (lanzó exactamente `len(scenarios) * len(attachment_sizes) *
    repeats` conversaciones) y lo imprime en su propio resumen -- pasalo
    acá con `--total-conversations` para que coincida con ese número en
    vez de subcontar silenciosamente."""
    max_turn_by_session: dict[str, int] = defaultdict(int)
    for row in rows:
        max_turn_by_session[row["session_id"]] = max(max_turn_by_session[row["session_id"]], row["turn_index"])
    total = total_conversations if total_conversations is not None else len(max_turn_by_session)
    complete = sum(1 for n in max_turn_by_session.values() if n >= 20)
    rate = complete / total if total else 0.0
    return total, complete, rate


def _failure_hotspot(rows: list[dict[str, Any]]) -> tuple[str, int, int] | None:
    """(escenario, turno_en_que_falló, cantidad) de la combinación más
    repetida entre las conversaciones que NO llegaron al turno 20 -- o
    `None` si todas completaron. No es ruido disperso: si varias
    conversaciones del mismo escenario se rompen en el MISMO turno, es
    una pista concreta de qué contenido de ese turno le cuesta al modelo
    (ej. pedirle que recalcule costos tras una corrección), no una falla
    aleatoria distribuida parejo en el tiempo."""
    max_turn_by_session: dict[str, tuple[str, int]] = {}
    for row in rows:
        sid = row["session_id"]
        prev_turn = max_turn_by_session.get(sid, (row["scenario"], 0))[1]
        max_turn_by_session[sid] = (row["scenario"], max(prev_turn, row["turn_index"]))

    counts: dict[tuple[str, int], int] = defaultdict(int)
    for scenario, max_turn in max_turn_by_session.values():
        if max_turn < 20:
            counts[(scenario, max_turn + 1)] += 1
    if not counts:
        return None
    (scenario, failing_turn), count = max(counts.items(), key=lambda kv: kv[1])
    return scenario, failing_turn, count


def _reading_paragraphs(rows: list[dict[str, Any]], total_conversations: int | None = None) -> str:
    recall_by_turn: dict[int, list[float]] = defaultdict(list)
    cost_by_turn: dict[int, list[float]] = defaultdict(list)
    latency_misses_by_turn: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        recall_by_turn[row["turn_index"]].append(row["memory_drift_recall"])
        # Costo POR turno, no el acumulado -- el acumulado es una suma
        # corrida que por definición siempre crece con N; compararlo
        # turno 1 vs. turno 20 no mide degradación, mide aritmética.
        # Lo que sí es una señal real de degradación es si CADA turno
        # individual se vuelve más caro a medida que el historial crece.
        cost_by_turn[row["turn_index"]].append(row["cost_usd"])
        if row["cache_hit_kind"] != "exact":
            latency_misses_by_turn[row["turn_index"]].append(row["latency_ms"])

    max_n = max(recall_by_turn)
    recall_1 = statistics.mean(recall_by_turn.get(1, [1.0]))
    recall_n = statistics.mean(recall_by_turn.get(max_n, [1.0]))

    breaking_turn = next(
        (n for n in sorted(recall_by_turn) if statistics.mean(recall_by_turn[n]) < 0.8),
        None,
    )

    cost_1 = statistics.mean(cost_by_turn.get(1, [0.0])) or 1e-9
    cost_n = statistics.mean(cost_by_turn.get(max_n, [0.0]))
    cost_growth = cost_n / cost_1

    lat_1 = statistics.mean(latency_misses_by_turn.get(1, [1.0])) or 1e-9
    lat_values_n = latency_misses_by_turn.get(max_n) or [lat_1]
    lat_n = statistics.mean(lat_values_n)
    latency_growth = lat_n / lat_1

    recall_drop_pct = (recall_1 - recall_n) * 100

    total_convos, complete_convos, complete_rate = _completion_stats(rows, total_conversations)

    anchor_caveat = ""
    if all(row["anchors_count"] == 0 for row in rows):
        anchor_caveat = (
            f" Importante para leer este número: `anchors_count` fue 0 en las {len(rows)} filas "
            "de esta corrida -- el detector heurístico de anclas (`app/compression/anchors.py`) nunca "
            "disparó, porque sus patrones buscan frases específicas (NDA, contrato firmado, "
            "presupuesto cerrado, deadline duro...) que el contenido guionado de los tres "
            "escenarios no usa textualmente. En los hechos, esta corrida mide la retención del "
            "SUMMARIZER solo, no del mecanismo de anclas -- un corpus de escenarios que dispare "
            "el heurístico daría una curva de recall distinta, probablemente mejor."
        )

    if breaking_turn is not None:
        drift_sentence = (
            f"El recall medio del fact-tracker cae por debajo de 0.8 a partir del turno "
            f"{breaking_turn} (arranca en {recall_1:.2f} en el turno 1 y termina en "
            f"{recall_n:.2f} en el turno {max_n}, una caída de {recall_drop_pct:.0f} puntos). "
            "Ese es, en los datos de esta corrida, el punto en el que la ventana deslizante "
            "y el resumen acumulativo empiezan a perder hechos más rápido de lo que los "
            f"retienen.{anchor_caveat}"
        )
    else:
        drift_sentence = (
            f"El recall medio del fact-tracker se mantiene por encima de 0.8 en los {max_n} "
            f"turnos (de {recall_1:.2f} en el turno 1 a {recall_n:.2f} en el turno {max_n}): en "
            "esta corrida, el mecanismo de anclas + resumen acumulativo alcanza para no perder "
            f"los hechos trackeados dentro de la ventana de 20 turnos probada.{anchor_caveat}"
        )

    if cost_growth >= latency_growth and cost_growth >= 1.5:
        dominant_sentence = (
            f"La dimensión que más degrada con la longitud de la conversación es el COSTO POR "
            f"TURNO (no el acumulado, que siempre crece por ser una suma corrida): crece "
            f"{cost_growth:.1f}x entre el turno 1 y el turno {max_n} -- cada turno nuevo manda "
            "más historial comprimido al modelo, y eso se traduce en más tokens de entrada por "
            f"turno -- mientras que la latencia por turno crece {latency_growth:.1f}x en el "
            "mismo tramo."
        )
    elif latency_growth > cost_growth and latency_growth >= 1.5:
        dominant_sentence = (
            f"La dimensión que más degrada con la longitud de la conversación es la LATENCIA "
            f"POR TURNO: crece {latency_growth:.1f}x entre el turno 1 y el turno {max_n}, por "
            f"encima del crecimiento de costo por turno ({cost_growth:.1f}x) en el mismo tramo."
        )
    else:
        dominant_sentence = (
            f"Ni el costo por turno ({cost_growth:.1f}x) ni la latencia por turno "
            f"({latency_growth:.1f}x) crecen de "
            "forma dramática entre el turno 1 y el último -- a 20 turnos, la degradación "
            "principal de este CAG es la pérdida de memoria, no el tiempo o el precio de cada "
            "llamada."
        )

    error_sentence = (
        f"{complete_convos}/{total_convos} conversaciones ({complete_rate:.0%}) llegaron "
        "completas hasta el turno 20; el resto se cortó antes por un error real del modelo "
        "(típicamente Instructor agotando sus reintentos cuando la aritmética de las fases no "
        "cierra con el total) -- también es una forma legítima en la que este CAG se rompe bajo "
        "estrés, no solo la degradación gradual de costo/latencia/memoria."
    )
    hotspot = _failure_hotspot(rows)
    if hotspot is not None:
        hotspot_scenario, hotspot_turn, hotspot_count = hotspot
        if hotspot_count >= 3:
            error_sentence += (
                f" No es ruido disperso: {hotspot_count} de esas conversaciones rotas son del "
                f"escenario '{hotspot_scenario}' y se cortan exactamente en el turno "
                f"{hotspot_turn} -- ese turno concentra el punto de quiebre más repetido de toda "
                "la corrida, lo que sugiere un contenido específico (no solo la longitud "
                "acumulada) que le cuesta al actor sin Critic/Boss de por medio."
            )

    rag_sentence = (
        "El tamaño del adjunto solo pesa en el turno en que se sube (acá, el primero): el texto "
        "extraído queda tejido en el prompt que se persiste al historial desde ese turno, así "
        "que un documento grande no se vuelve a pagar en los turnos siguientes, pero tampoco se "
        "vuelve más barato de cargar -- tatuado en el historial, crece la ventana para siempre. "
        "Un caso límite que justificaría saltar a RAG es, precisamente, ese: un documento de "
        "referencia grande que hay que consultar en MUCHOS turnos de la conversación (no solo "
        "mencionarlo una vez), donde CAG paga el costo de arrastrarlo entero en cada turno "
        "mientras RAG solo recuperaría el fragmento relevante cuando efectivamente se lo necesita."
    )

    return (
        f"{drift_sentence} {dominant_sentence}\n\n{error_sentence} {rag_sentence}"
    )


def build_report(rows: list[dict[str, Any]], total_conversations: int | None = None) -> str:
    total_convos, complete_convos, complete_rate = _completion_stats(rows, total_conversations)
    return f"""# Stress-test del CAG -- Reporte (sesión 6, Bloque 5/6)

{len(rows)} filas (una por turno completado), {total_convos} conversaciones intentadas,
{complete_convos} completas ({complete_rate:.0%}).

## Resumen por escenario × tamaño de adjunto

{_summary_table(rows)}

_Cache semántica: 0% en todas las filas a propósito -- el flujo conversacional
(`POST /sessions/{{id}}/estimate`) nunca pasa por ella, solo el endpoint de
formulario de un solo turno la usa. La latencia de este resumen excluye los
turnos con exact cache hit (ver docstring de este módulo)._

## Curvas

### latency_ms vs tokens_in

{_latency_vs_tokens_curve(rows)}

### cost_usd acumulado vs turn_index (por escenario)

{_cost_cumulative_vs_turn_curve(rows)}

### MemoryDriftMetric (recall) vs N (turnos)

{_recall_vs_turn_curve(rows)}

## Lectura

{_reading_paragraphs(rows, total_conversations)}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("evals/stress/results.csv"))
    parser.add_argument("--output", type=Path, default=Path("evals/stress/REPORT.md"))
    parser.add_argument(
        "--total-conversations",
        type=int,
        default=None,
        help=(
            "Total real de conversaciones intentadas (ver docstring de "
            "_completion_stats) -- pasalo si alguna falló en su primer "
            "turno, porque esas no dejan fila en el CSV y el total "
            "derivado del archivo las subcuenta."
        ),
    )
    args = parser.parse_args()

    rows = _read_rows(args.input)
    if not rows:
        print(f"{args.input} está vacío -- corré evals.stress.run primero.")
        return 1

    args.output.write_text(build_report(rows, args.total_conversations), encoding="utf-8")
    print(f"REPORT.md escrito en {args.output} ({len(rows)} filas procesadas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
