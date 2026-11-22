# Agente de estimación con bucle manual

Una transcripción real rara vez encaja en un único camino de recuperación: un proyecto que mezcla un backend de negocio, una integración con un ERP y una app móvil necesita presupuestos de referencia distintos para cada pieza, y no se sabe de antemano cuántas búsquedas harán falta. Este módulo añade un agente que decide cada paso por sí mismo, escrito a mano sobre la Responses API de OpenAI y sin ningún framework, para que el bucle sea visible por completo.

Alcance: el agente, sus tres herramientas, la traza, un endpoint (`POST /estimate/agent`) y un script de consola. No se modificó la recuperación existente ni el pipeline de estimación con fuentes.

## 1. Cómo funciona

Un agente es un bucle: un modelo decide, el código ejecuta la herramienta que el modelo pidió y le devuelve el resultado, y el ciclo se repite hasta que el modelo responde sin pedir nada más.

```
transcripción ──► responses.create(tools, instructions)
                       │
        ┌──────────────┴──────────────┐
        │ ¿hay items function_call?   │
        └──────┬───────────────┬──────┘
             sí│               │no
               ▼               ▼
   ejecutar cada herramienta   responses.parse(AgentEstimate)
   function_call_output        (estimación estructurada)
   (mismo call_id)
   responses.create(previous_response_id=…)
               └──► vuelta al inicio (máximo N iteraciones)
```

El encadenamiento lo conduce el código (`app/agent/loop.py`): se recorren los items `function_call` de `response.output`, se ejecuta cada herramienta con sus argumentos ya validados, se devuelve un `function_call_output` con el mismo `call_id` y se llama de nuevo con `previous_response_id`. Cada llamada a herramienta produce un paso de la traza.

Dos salvaguardas evitan que el bucle quede atrapado: el modelo puede terminar solo, y existe un tope de iteraciones (`agent_max_iterations`, 10 por defecto). Si se alcanza el tope, las llamadas pendientes se contestan igualmente y la estimación se pide con lo reunido hasta ese punto; el resultado lo indica en `stopped_reason`.

## 2. Las herramientas

Los esquemas siguen la forma plana de la Responses API (`type` y `name` al mismo nivel, sin envoltorio `function`) con `strict: true`. El modo estricto exige que todas las propiedades figuren en `required` y que `additionalProperties` sea `false`, de modo que los valores opcionales se expresan como `null`. Un test recorre los tres esquemas y comprueba esas reglas en cada objeto anidado.

| Herramienta | Qué hace | Implementación |
|---|---|---|
| `search_budgets(query, filters)` | Devuelve hasta 5 componentes de presupuestos históricos con sus horas y metadatos | Envuelve `retrieve()`: búsqueda híbrida más reranking con cross-encoder. No reimplementa la recuperación |
| `calculate_estimate(components)` | Mediana de las referencias de cada componente más 15 % de contingencia, y total | Determinista, sin modelo |
| `validate_estimate(components, total_hours)` | Comprobaciones de coherencia sobre el resultado | Determinista, sin modelo |

Las descripciones están escritas para un lector que solo ve el esquema: dicen cuándo usar cada herramienta, cuándo no, qué devuelve y qué significa un resultado vacío. Por ejemplo, la de `search_budgets` indica que se llama una vez por componente, que la consulta va en inglés y que hay que comprobar que los resultados sean realmente comparables antes de usar sus horas.

**Filtros.** El retrieval existente no filtra por metadatos, así que `search_budgets` recupera un conjunto más amplio (25 candidatos) cuando se pasan filtros y filtra después por `sector`, `main_technology`, `year_from` y `year_to`. El corpus no tiene un campo de "tipo de componente", por lo que no se ofrece un filtro con ese nombre.

**`validate_estimate`** detecta: componentes sin referencia, horas sin referencia que las respalde, horas no positivas pese a tener referencias, horas fuera del rango de sus referencias (menos de la mitad del mínimo o más de una vez y media el máximo), componentes o totales inverosímiles, y un total que no coincide con la suma. Nunca modifica la estimación: solo informa.

**Argumentos inválidos.** Cada llamada se valida con Pydantic antes de ejecutarse. Un JSON malformado, un campo ausente, una herramienta inexistente o un fallo interno vuelven al modelo como un resultado `{"error": ...}` que puede leer y corregir en el turno siguiente, sin interrumpir el bucle.

## 3. La traza

Cada paso registra razonamiento, acción y observación. El razonamiento es el resumen que emite la propia API (`reasoning={"summary": "auto"}`); cuando un turno no devuelve resumen, la traza lo dice en lugar de inventarlo. Si el modelo pide varias herramientas en un mismo turno, solo el primer paso lleva el razonamiento y los demás lo indican.

```
STEP 11
  reasoning:   <resumen de razonamiento del modelo>
  action:      calculate_estimate({"components": [...]})
  observation: total=425.5h across 6 components; no references for: Order management API (B2B pharmacies), SAP integration via IDocs (...), Offline-capable driver mobile app (Android & iOS)
```

La traza completa de la transcripción compleja está en `evals/agent/trace_complex.txt`.

## 4. Resultado sobre la transcripción compleja

Transcripción: `evals/agent/transcripts/complex.txt`, un distribuidor farmacéutico con cuatro piezas (núcleo de pedidos, stock y rutas; integración con SAP; app móvil de conductores con modo sin conexión; panel de análisis). Modelo `gpt-5`, esfuerzo `medium`, retrieval real sobre el corpus de 15 presupuestos.

| Medida | Valor |
|---|---|
| Iteraciones del bucle | 8 de 10 posibles |
| Llamadas a `search_budgets` | 10 |
| Componentes identificados | 6 (el modelo separó pedidos, stock y rutas) |
| Llamadas a `calculate_estimate` / `validate_estimate` | 1 / 1 |
| Terminación | Por sí solo (`stopped_reason: completed`) |
| Total | 425,5 h, coincide con el cálculo (`consistent_with_calculation: true`) |
| Tokens | 74 787 de entrada, 7 164 de salida |

| Componente | Horas | Referencia |
|---|---|---|
| Gestión de inventario con reposición automática | 161,0 | BUD-2024-011 |
| Planificación diaria de rutas | 207,0 | BUD-2024-007 |
| Panel de análisis con filtros y alertas | 57,5 | BUD-2024-014 (cobertura parcial) |
| API de pedidos B2B | 0 | sin referencia comparable |
| Integración con SAP vía IDocs | 0 | sin referencia comparable |
| App móvil de conductores con modo sin conexión | 0 | sin referencia comparable |

El resultado es útil precisamente porque no está completo. El corpus no contiene ningún proyecto de SAP, ninguna app móvil de reparto y ningún sistema de pedidos mayoristas. El agente lo detectó tras buscar con dos formulaciones distintas, dejó esos tres componentes en cero, los marcó como sin presupuesto, y `validate_estimate` lo confirmó con tres hallazgos `no_reference` que quedaron recogidos como supuestos. El total de 425,5 h es por tanto un mínimo, no una estimación del proyecto entero.

**Comparación con el pipeline fijo.** Con la misma transcripción, `POST /estimate/grounded` (una sola recuperación para todo el texto) devolvió los cuatro componentes sin respaldo y 0 horas en total. Una única consulta que mezcla piezas sin relación trae fragmentos mezclados y el modelo, correctamente, no los usa. La búsqueda por componente es lo que permite al agente aprovechar el corpus. La contrapartida es que el agente tiene más margen de criterio: aceptó como referencia del panel un componente de reporte de operaciones de almacén, de 50 horas, y lo declaró como cobertura parcial. El pipeline con fuentes verificables es más estricto en ese punto.

## 5. Decisiones y problemas encontrados

- **La primera versión del prompt no terminaba.** Pedía refinar las búsquedas hasta tener referencias. Con el corpus sin SAP ni móvil, el modelo encadenó 14 búsquedas, agotó las 10 iteraciones y nunca llamó a `calculate_estimate`, así que la estimación final quedó en cero. El tope funcionó como salvaguarda y el cierre estructurado no inventó horas, pero el comportamiento era inaceptable. Se corrigió en el prompt: se limita el refinamiento a una búsqueda extra por componente, se declara que la ausencia de referencias es un hallazgo válido, y se informa al modelo de su presupuesto de turnos, con la obligación de calcular antes de agotarlo. Es una decisión de diseño de agentes: un modelo que desconoce su límite tiende a seguir buscando.
- **La estimación estructurada se pide en una llamada aparte**, encadenada a la última respuesta, con `responses.parse` y un esquema sin restricciones de campo (el modo estricto no las admite). El total no lo escribe el modelo: se suma en código a partir de los componentes y se compara con el último `calculate_estimate` para exponer cualquier discrepancia.
- **Las consultas van en inglés.** El corpus está en inglés y las transcripciones no; la rama léxica de la búsqueda híbrida no aporta nada con consultas en español (ya medido antes), mientras que la rama vectorial sí se entiende. El prompt le indica al modelo que formule las consultas en inglés.
- **Herramientas secuenciales.** Aunque el modelo pida varias búsquedas en un turno, se ejecutan una tras otra: una `AsyncSession` no admite uso concurrente.
- **Rutas de código.** El agente vive en `app/agent/` (`schemas.py`, `tools.py`, `loop.py`, `router.py`), su prompt en `app/prompts/agent/v1/` y el script en `scripts/run_agent.py`.

## 6. Pruebas

55 tests nuevos (329 en total), todos sin red gracias a un cliente falso que reproduce respuestas guionadas y registra cada llamada:

- **Esquemas**: forma plana, `strict`, y cada objeto anidado con `required` igual a sus propiedades y `additionalProperties: false`.
- **Herramientas**: mediana y contingencia, componente sin referencias, cada código de `validate_estimate`, tolerancia de redondeo, mapeo de chunks a ítems, uso de búsqueda híbrida con reranking, filtros por metadatos y el despachador con entradas inválidas.
- **Bucle**: terminación natural, cada `function_call_output` con su `call_id`, encadenamiento con `previous_response_id`, ajustes de razonamiento y herramientas en la primera llamada, forma de la traza, total calculado en código y coherencia con el cálculo, suma de tokens, tope de iteraciones (con las llamadas pendientes contestadas), argumentos inválidos devueltos al modelo y salida estructurada no interpretable.
- **HTTP**: respuesta con traza renderizada, valores por defecto y sobrescritura de modelo y esfuerzo, `422` ante peticiones inválidas, `502` ante fallos del agente o de OpenAI, `500` ante configuración ausente.
- **Prompt**: nombra las tres herramientas, el presupuesto de turnos y las reglas de consultas en inglés y de no inventar horas.

Además se verificó el endpoint contra el servicio real en Docker (`gpt-5-mini`, transcripción simple: 6 iteraciones, 3 búsquedas, total coherente) y que una transcripción demasiado corta devuelve `422`.

## 7. Cómo probarlo

```bash
# Traza por consola (necesita la base levantada con el corpus ingerido)
PYTHONPATH=. uv run python scripts/run_agent.py evals/agent/transcripts/simple.txt --model gpt-5-mini --effort low

# Ejecución completa sobre la transcripción compleja
PYTHONPATH=. uv run python scripts/run_agent.py evals/agent/transcripts/complex.txt --out evals/agent/trace_complex.txt

# Tests
uv run pytest tests/agent tests/prompts/test_agent_prompt.py -v
```

## 8. Limitaciones

- La calidad de las referencias está acotada por el corpus (15 presupuestos, 52 componentes). Con un archivo mayor, la proporción de componentes sin referencia debería bajar.
- Las horas de cada componente son la mediana de las referencias más un 15 % fijo. No se ajustan por tecnología, complejidad ni año.
- El agente decide qué referencias son comparables; a diferencia del pipeline con fuentes, no hay un verificador que compruebe cita por cita. Una mejora natural es pasar la estimación final del agente por `verify_citations`.
- Los resúmenes de razonamiento los decide la API: en varios turnos de la ejecución real no se devolvió ninguno.
- No hay interfaz en Streamlit para este endpoint.
