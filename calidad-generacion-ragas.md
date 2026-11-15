# Citación verificable por línea y línea base de calidad (RAGAS)

La estimación deja de citar fuentes "en bloque": cada línea (cada componente del proyecto) apunta al fragmento concreto de un presupuesto histórico del que se derivó, y un verificador comprueba por código que esa fuente existe realmente entre los fragmentos que se le entregaron al modelo. Sobre eso se mide la calidad de la generación con las cuatro métricas de RAGAS, como línea base.

Todo el código nuevo está en inglés (nombres, comentarios, logs, prompts); estas notas y el golden set (`ground_truth`) van en español.

## 1. Qué se construyó

Módulo nuevo `app/grounded_estimation/`, sin tocar el endpoint de estimaciones existente (su contrato HTTP no cambia):

| Pieza | Dónde | Qué hace |
|---|---|---|
| Schema | `schemas.py` | `SourceReference{chunk_id, document_id, evidence}` y `EstimateLineItem{component, hours, rationale, grounded, sources}`. Un validador impone la regla de integridad: `grounded=True` exige al menos una fuente y horas > 0; `grounded=False` exige sin fuentes, **0 horas** y un rationale que empiece con `Insufficient data:`. |
| Prompt | `app/prompts/grounded_estimation/v1/` | Cada chunk llega con su `chunk_id` y `document_id`. Se exige citar ids exactos, copiar `evidence` **textual** (por ejemplo `Estimated hours: 220`) y marcar `grounded=false` cuando no hay soporte, en lugar de estimar sin respaldo. |
| Generador | `generator.py` | `client.responses.parse` con `text_format` estricto. Si el modelo incumple la regla de integridad, se le muestra el error y reintenta (hasta 3 veces). |
| Verificador | `citations.py` | `verify_citations(estimate, retrieved_chunk_ids) -> CitationReport`. Clasifica cada línea en *fundamentada*, *cita colgante* (id que nunca estuvo en el contexto) o *sin datos suficientes*. Loguea con structlog; el `request_id` viaja por contextvars, así que todos los logs de la petición salen correlacionados. Una cita colgante se loguea como `error`. |
| Orquestación | `service.py`, `router.py` | `POST /estimate/grounded`: busca (híbrida, k=5), genera, verifica y, si hay citas colgantes, hace **un** reintento corrigiendo al modelo con los ids válidos. El informe de la primera pasada queda logueado aunque el reintento lo arregle. La respuesta separa `claimed_hours` (lo que afirma el modelo) de `verified_hours` (solo líneas con todas sus citas verificadas). |

El total de horas nunca se le pide al modelo: se calcula en código sumando solo las líneas fundamentadas.

Tests (90 casos nuevos; la suite completa da 274 passed, también dentro de Docker):

- `tests/test_grounded_estimation.py`: reglas de integridad del schema (incluido que el schema se mantenga dentro del subconjunto que acepta el modo estricto), clasificación del verificador con sus bordes (cita colgante introducida deliberadamente, fuente válida más inventada, ids casi iguales, contexto vacío), correlación por `request_id` en los logs, reintentos del generador y del servicio (un único reintento como máximo), parámetros que el servicio le pasa a la búsqueda y al modelo, y que no se llame al LLM si no se recuperó nada.
- `tests/test_grounded_router.py`: el endpoint por HTTP. Forma de la respuesta, entrada inválida (422), fallos del modelo y de OpenAI (502) y mala configuración (500) sin filtrar detalles internos, y un recorrido completo por el servicio real con solo el exterior simulado.
- `tests/prompts/test_grounded_estimation_prompt.py`: render del prompt de sistema y de usuario.
- `tests/test_stream_and_ui.py`: la pestaña "Estimación con fuentes" de la interfaz (resultado, cita inválida visible, parámetros enviados, errores del backend y de conexión).

## 2. Informe de verificación de citaciones

Generado con `scripts/demo_verify_citations.py` contra el servicio real. Primero una estimación real; después la misma estimación con una cita inventada introducida deliberadamente (`chunk_id=999`) para comprobar que se detecta.

```
request_id: 47689f2e9fb64544a36bcf744c7e46d0
retrieved chunk ids: ['9', '10', '11', '12', '23']

=== Real estimate ===
lines: 5  grounded: 4  dangling: 0  insufficient: 1
component                         status              cited       dangling
Settlement reconciliation         grounded            ['11']      -
Payment gateway core              grounded            ['9']       -
Real-time fraud detection         grounded            ['10']      -
Merchant dashboard                grounded            ['12']      -
Fraud detection and settlement r  insufficient_data   []          -

(planting chunk_id='999' in line 'Settlement reconciliation')

=== Same estimate with an invented citation ===
lines: 5  grounded: 3  dangling: 1  insufficient: 1
Settlement reconciliation         dangling_citation   ['999']     ['999']
Payment gateway core              grounded            ['9']       -
Real-time fraud detection         grounded            ['10']      -
Merchant dashboard                grounded            ['12']      -
Fraud detection and settlement r  insufficient_data   []          -
```

Qué demuestra este informe: las líneas fundamentadas citan fragmentos reales del contexto; la cita colgante introducida se detecta y se loguea como error con el mismo `request_id`; la línea sin soporte queda con 0 horas y marcada como sin datos suficientes, sin cifras inventadas.

Sobre las 15 generaciones de la evaluación (5 consultas × 3 corridas): **52 líneas fundamentadas, 14 sin datos suficientes y 0 citas colgantes**. El modelo no inventó ningún id, así que la protección está comprobada con la cita introducida deliberadamente, pero en uso real todavía no llegó a activarse.

## 3. Golden set extendido

`evals/golden_set.json` conserva las 5 consultas de la evaluación de recuperación (mismas consultas, mismos presupuestos relevantes anotados a mano) y suma una estimación de referencia (`ground_truth`) por consulta. Cada referencia se deriva de las horas registradas en `data/budgets_sample.json` para los componentes que la consulta **realmente pide**; los componentes del mismo presupuesto que la consulta no menciona (por ejemplo el dashboard de comercios en la pasarela de pagos) quedan fuera deliberadamente. `scripts/golden_set_eval.py` lee ahora de este mismo archivo, así que hay una sola fuente de verdad.

Las referencias nombran el **proyecto histórico**, no su id de presupuesto: el id no aparece en el texto de los chunks y el juez no puede verificar información que no recibe.

## 4. Evaluación RAGAS

`scripts/ragas_eval.py` (`uv run --group eval python scripts/ragas_eval.py --runs 3`). Para cada consulta registra `question` (la consulta), `answer` (la estimación generada, en texto), `contexts` (los chunks recuperados que recibió el generador) y `ground_truth`.

Configuración: generador `gpt-4o-mini` (temperature 0); juez `gpt-4o` con temperatura 0; embeddings `text-embedding-3-small`; recuperación híbrida, k=5, sin reranking. Cada consulta se genera y se puntúa 3 veces y se promedia.

| Consulta | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| Q1 Pago online / checkout | 0.55 | 0.62 | 0.33 | 0.92 |
| Q2 Pasarela de pagos | 0.65 | 0.58 | 0.78 | 0.67 |
| Q3 Monitoreo remoto de pacientes | 0.78 | 0.56 | 0.00 | 0.67 |
| Q4 Telemetría industrial | 0.68 | 0.66 | 0.00 | 0.56 |
| Q5 Marketplace multi-vendedor | 0.55 | 0.54 | 0.00 | 0.56 |
| **Promedio** | **0.64** | **0.59** | **0.22** | **0.67** |

Variación entre corridas de la misma consulta (máximo de cada métrica): faithfulness 0.37, answer relevancy 0.21, context precision 0.67, context recall 0.67. Los resultados completos, con cada corrida, están en `evals/ragas_baseline.json`.

### Notas de método

- El juez con temperatura por defecto puntuaba la **misma entrada** como 0.0 o 0.5 en distintas llamadas; con temperatura 0 se estabiliza. Está fijado en el script.
- El límite de tokens por defecto del juez cortaba su respuesta en `faithfulness` sobre respuestas largas (fallaba esa métrica). Se subió a 4096.
- Una primera versión de las referencias citaba ids de presupuesto que no existen en el texto de los chunks; se reescribieron con el nombre del proyecto.
- Con una sola corrida los valores de faithfulness cambiaban mucho entre ejecuciones (cada una regenera la estimación), por eso la línea base promedia 3.

## 5. Observaciones sobre los resultados

- **Context precision (0.22) y context recall (0.67).** Reflejan más el diseño de la medición que la calidad de la recuperación. Cada fragmento contiene un solo componente, pero la referencia describe varios módulos y un total, y el juez califica con 0 cualquier fragmento que por sí solo no baste para justificar la respuesta completa. En Q3, Q4 y Q5 los primeros fragmentos recuperados pertenecían al presupuesto correcto y, aun así, la precisión fue 0.00.
- **Faithfulness (0.64, con una variación de hasta 0.37 entre corridas).** No baja por citas inventadas, ya que no hubo ninguna. Baja por el texto libre del `rationale`, que el contexto no respalda; por componentes que la consulta no pedía (por ejemplo "Merchant dashboard" en Q1 y Q2); y por líneas marcadas "Insufficient data" cuyo componente sí tenía su fragmento en el contexto (5 de las 14, casi siempre un componente que la consulta no pedía).
- **Alcance de la verificación.** `verify_citations` solo comprueba que el id citado exista. No detecta un fragmento real citado que no respalda la línea, ni un "sin datos" injustificado. El paso siguiente natural es verificar la `evidence` contra el texto del fragmento y medir cuánto mejora la fidelidad.
