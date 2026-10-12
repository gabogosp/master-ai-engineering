# Stress-test del CAG -- Reporte (sesión 6, Bloque 5/6)

351 filas (una por turno completado), 45 conversaciones intentadas,
5 completas (11%).

## Resumen por escenario × tamaño de adjunto

| Escenario | Adjunto (KB) | P50 latency (ms) | P95 latency (ms) | Costo total (USD) | Exact cache hit | Semantic cache hit | Recall medio |
|---|---|---|---|---|---|---|---|
| contradiction | 0 | 4926 | 11796 | 0.0244 | 10% | 0% | 0.25 |
| contradiction | 5 | 5875 | 17530 | 0.0171 | 12% | 0% | 0.48 |
| contradiction | 20 | 6854 | 12561 | 0.0132 | 14% | 0% | 0.50 |
| contradiction | 50 | 3957 | 8066 | 0.0356 | 10% | 0% | 0.51 |
| contradiction | 100 | 8650 | 19309 | 0.0621 | 14% | 0% | 0.49 |
| growing | 0 | 9984 | 17616 | 0.0019 | 50% | 0% | 0.72 |
| growing | 5 | 5936 | 15193 | 0.0052 | 30% | 0% | 0.51 |
| growing | 20 | 5885 | 6748 | 0.0059 | 33% | 0% | 0.61 |
| growing | 50 | 8342 | 22191 | 0.0221 | 0% | 0% | 0.53 |
| growing | 100 | 11853 | 22302 | 0.0344 | 13% | 0% | 0.41 |
| pivot | 0 | 4280 | 6113 | 0.0253 | 27% | 0% | 0.89 |
| pivot | 5 | 5709 | 12353 | 0.0279 | 25% | 0% | 0.79 |
| pivot | 20 | 6137 | 8188 | 0.0086 | 0% | 0% | 0.78 |
| pivot | 50 | 3147 | 11200 | 0.0390 | 28% | 0% | 0.74 |
| pivot | 100 | 3248 | 6960 | 0.0099 | 29% | 0% | 0.92 |

_Cache semántica: 0% en todas las filas a propósito -- el flujo conversacional
(`POST /sessions/{id}/estimate`) nunca pasa por ella, solo el endpoint de
formulario de un solo turno la usa. La latencia de este resumen excluye los
turnos con exact cache hit (ver docstring de este módulo)._

## Curvas

### latency_ms vs tokens_in

| tokens_in (rango) | latency_ms (media) | n |
|---|---|---|
| 2697-5189 | 3020 | 28 |
| 5217-5449 | 3198 | 28 |
| 5450-6009 | 3909 | 28 |
| 6090-7446 | 5508 | 28 |
| 7463-10947 | 6103 | 28 |
| 11007-13953 | 7003 | 28 |
| 14041-15431 | 6850 | 28 |
| 15436-25318 | 9065 | 28 |
| 25905-33794 | 10170 | 28 |
| 33800-83359 | 13895 | 28 |
| 83930-135268 | 22098 | 5 |

### cost_usd acumulado vs turn_index (por escenario)

| turn_index | contradiction | growing | pivot | n |
|---|---|---|---|---|
| 1 | 0.0003 | 0.0005 | 0.0005 | 40 |
| 2 | 0.0012 | 0.0015 | 0.0009 | 37 |
| 3 | 0.0023 | 0.0025 | 0.0015 | 35 |
| 4 | 0.0034 | 0.0036 | 0.0022 | 30 |
| 5 | 0.0048 | 0.0055 | 0.0024 | 28 |
| 6 | 0.0063 | 0.0079 | 0.0032 | 25 |
| 7 | 0.0078 | 0.0088 | 0.0039 | 23 |
| 8 | 0.0078 | 0.0098 | 0.0046 | 17 |
| 9 | 0.0088 | 0.0111 | 0.0050 | 16 |
| 10 | 0.0097 | 0.0101 | 0.0063 | 14 |
| 11 | 0.0101 | 0.0110 | 0.0073 | 12 |
| 12 | 0.0112 | 0.0119 | 0.0080 | 11 |
| 13 | 0.0123 | 0.0128 | 0.0069 | 10 |
| 14 | 0.0134 | 0.0142 | 0.0079 | 10 |
| 15 | 0.0154 | 0.0152 | 0.0088 | 9 |
| 16 | 0.0167 | 0.0162 | 0.0092 | 8 |
| 17 | 0.0178 | 0.0172 | 0.0103 | 8 |
| 18 | 0.0193 | 0.0188 | 0.0111 | 7 |
| 19 | 0.0207 | 0.0200 | 0.0129 | 6 |
| 20 | 0.0219 | - | 0.0141 | 5 |

### MemoryDriftMetric (recall) vs N (turnos)

| turn_index (N) | MemoryDriftMetric (recall medio) | n |
|---|---|---|
| 1 | 0.93 | 40 |
| 2 | 0.62 | 37 |
| 3 | 0.44 | 35 |
| 4 | 0.59 | 30 |
| 5 | 0.46 | 28 |
| 6 | 0.45 | 25 |
| 7 | 0.50 | 23 |
| 8 | 0.58 | 17 |
| 9 | 0.51 | 16 |
| 10 | 0.64 | 14 |
| 11 | 0.62 | 12 |
| 12 | 0.58 | 11 |
| 13 | 0.64 | 10 |
| 14 | 0.65 | 10 |
| 15 | 0.70 | 9 |
| 16 | 0.70 | 8 |
| 17 | 0.67 | 8 |
| 18 | 0.63 | 7 |
| 19 | 0.62 | 6 |
| 20 | 0.63 | 5 |

## Lectura

El recall medio del fact-tracker cae por debajo de 0.8 a partir del turno 2 (arranca en 0.93 en el turno 1 y termina en 0.63 en el turno 20, una caída de 30 puntos). Ese es, en los datos de esta corrida, el punto en el que la ventana deslizante y el resumen acumulativo empiezan a perder hechos más rápido de lo que los retienen. La dimensión que más degrada con la longitud de la conversación es el COSTO POR TURNO (no el acumulado, que siempre crece por ser una suma corrida): crece 2.7x entre el turno 1 y el turno 20 -- cada turno nuevo manda más historial comprimido al modelo, y eso se traduce en más tokens de entrada por turno -- mientras que la latencia por turno crece 1.4x en el mismo tramo.

5/45 conversaciones (11%) llegaron completas hasta el turno 20; el resto se cortó antes por un error real del modelo (típicamente Instructor agotando sus reintentos cuando la aritmética de las fases no cierra con el total) -- también es una forma legítima en la que este CAG se rompe bajo estrés, no solo la degradación gradual de costo/latencia/memoria. No es ruido disperso: 6 de esas conversaciones rotas son del escenario 'contradiction' y se cortan exactamente en el turno 8 -- ese turno concentra el punto de quiebre más repetido de toda la corrida, lo que sugiere un contenido específico (no solo la longitud acumulada) que le cuesta al actor sin Critic/Boss de por medio. El tamaño del adjunto solo pesa en el turno en que se sube (acá, el primero): el texto extraído queda tejido en el prompt que se persiste al historial desde ese turno, así que un documento grande no se vuelve a pagar en los turnos siguientes, pero tampoco se vuelve más barato de cargar -- tatuado en el historial, crece la ventana para siempre. Un caso límite que justificaría saltar a RAG es, precisamente, ese: un documento de referencia grande que hay que consultar en MUCHOS turnos de la conversación (no solo mencionarlo una vez), donde CAG paga el costo de arrastrarlo entero en cada turno mientras RAG solo recuperaría el fragmento relevante cuando efectivamente se lo necesita.
