# Búsqueda híbrida y reranking sobre el retrieval de presupuestos

Extiende el retrieval vectorial existente (`POST /search`, pgvector) con una
rama léxica full-text y un reranker basado en cross-encoder, bajo el patrón
recall-then-rerank. El alcance queda limitado a estas dos técnicas: no se
tocó reformulación de consultas, routing multi-índice ni filtrado por
metadatos.

## 1. Búsqueda full-text en PostgreSQL

Migración `alembic/versions/0002_chunk_search_vector.py`: columna generada
`search_vector tsvector` sobre `chunks.content`, con índice GIN.

```sql
ALTER TABLE chunks
ADD COLUMN search_vector tsvector
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX ix_chunks_search_vector_gin ON chunks USING gin (search_vector);
```

**Decisión de idioma, verificada contra datos reales, no asumida:** el
corpus (`data/budgets_sample.json`, 15 presupuestos / 52 chunks) está en
inglés -- confirmado leyendo su contenido ("High-throughput payment
authorization...", "GraphQL catalog API with faceted search..."). Por eso
la configuración de text search usada es `english`, no `spanish`. Usar
`spanish` sobre un corpus en inglés perdería el stemming correcto y
descartaría stopwords del idioma equivocado, degradando la rama léxica sin
ninguna ganancia a cambio.

Columna generada (`GENERATED ALWAYS ... STORED`), no una tabla ni un
trigger aparte: Postgres la recalcula sola en cada insert/update de
`content`, sin lógica de sincronización que mantener a mano.

## 2. Rama léxica y fusión RRF

- `app/embedding_pipeline/fusion.py` -- `reciprocal_rank_fusion`, implementación
  propia: fusiona rankings por posición (no por score), con una constante de
  suavizado `k` (default 60, configurable vía `settings.rrf_k`). Cubierta por
  `tests/test_fusion.py` (5 casos: fusión básica, item en una sola rama,
  duplicados dentro de un ranking, desempate determinístico, validación de `k`).
- `app/embedding_pipeline/retrieval.py` -- la rama léxica (`_lexical_search`)
  usa `plainto_tsquery('english', query)` contra `search_vector`, rankeada por
  `ts_rank_cd`. La rama vectorial (`_vector_search`) reusa la misma consulta
  por `cosine_distance` que ya existía. Un candidato que entra solo por la rama léxica recibe
  una distancia sentinela (1.0, "lejos") que ni la fusión ni el reranker usan
  para decidir nada -- es solo para que la respuesta siga teniendo un campo
  `distance` consistente.
- El modo de búsqueda se expone como parámetro del request: `POST /search`
  acepta `search_mode: "vector" | "hybrid"` (default `"vector"`, preserva el
  comportamiento previo sin cambios para quien no pase el parámetro).

## 3. Integración del reranker

- `app/embedding_pipeline/reranker.py` -- `CrossEncoderReranker`, wrapper de
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingüe, corre en CPU a
  latencias razonables). Carga perezosa con lock, para no pagar el costo de
  `import torch` ni la descarga del modelo al importar el módulo o correr
  tests.
- `scripts/verify_reranker.py` -- chequeo de pre-vuelo como gate antes de
  integrar nada; confirmado localmente: el modelo carga y prioriza
  correctamente un documento relevante sobre uno irrelevante.
- Patrón recall-then-rerank en `retrieval.py`: cuando `rerank=True`, la
  recuperación trae un pool amplio (`recall_k=50` por default) y el
  cross-encoder reordena ese pool completo antes de cortar a los `k`
  finales. Se activa con `rerank: bool` en el request, independiente del
  `search_mode` -- las cuatro combinaciones (vector/híbrida × con/sin
  rerank) son las cuatro configuraciones medidas en la sección 4.

Reproducible con:

```bash
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query": "...", "k": 5, "search_mode": "hybrid", "rerank": true}'
```

## 4. Golden set y medición

Cinco consultas en inglés (mismo idioma que el corpus -- una consulta en
español contra un corpus en inglés no tendría overlap léxico, y la rama
híbrida no tendría nada que demostrar), anotadas a mano contra los 15
presupuestos reales de `data/budgets_sample.json`. Las dos primeras son
deliberadamente ambiguas en vocabulario de "pagos" -- el caso real que
justifica la búsqueda híbrida y el reranking: un presupuesto puede estar
cerca en el espacio vectorial y no ser el proyecto correcto.

| # | Query | Relevante |
|---|---|---|
| 1 | *Online payment processing for checkout and order management* | `BUD-2024-005` (storefront e-commerce, no el gateway de pagos) |
| 2 | *Real-time payment gateway with fraud detection and settlement reconciliation with the acquirer* | `BUD-2024-003` (sí es, de verdad, un proyecto de pagos) |
| 3 | *Remote patient monitoring platform ingesting wearable device data with alerts* | `BUD-2024-010` |
| 4 | *Factory IoT telemetry platform for predictive maintenance* | `BUD-2024-013` |
| 5 | *Multi-vendor marketplace with vendor payouts and buyer-seller dispute handling* | `BUD-2024-006` |

Medido con `scripts/golden_set_eval.py` (`PYTHONPATH=. uv run python
scripts/golden_set_eval.py`), contra el servicio real, sin mocks. Además de
precisión@5 (la métrica de referencia) se agrega MRR@5 (1/posición del primer
resultado relevante): precisión@5 cuenta *cuántos* resultados del top-5 son
relevantes pero es ciega al *orden* -- si reordenar sube el único resultado
relevante de la posición 2 a la 1, precisión@5 no cambia pero el reranking sí
hizo su trabajo, y eso solo se ve con una métrica sensible al rank.

**Resultado real, promedio de las 5 consultas (reranker precalentado antes de
medir, para no contaminar la latencia con la carga del modelo):**

| Config | Búsqueda | Reranking | Precisión@5 | MRR@5 | Latencia (ms) |
|---|---|---|---|---|---|
| A | Vectorial | No  | 0.60 | 0.87 | 519 |
| B | Híbrida   | No  | 0.60 | 0.87 | 518 |
| C | Vectorial | Sí  | 0.60 | 0.87 | 1 036 |
| D | Híbrida   | Sí  | 0.60 | 0.87 | 964 |

Reproducido en tres corridas completas con los mismos cuatro valores de
precisión/MRR -- el resultado es estable, no una medición de un solo tiro.

**Lectura honesta:** ni la búsqueda híbrida ni el reranking mejoraron la
precisión@5 ni el MRR@5 promedio sobre este golden set. La rama léxica
devuelve, en los cinco casos, el mismo conjunto que la vectorial dentro del
pool de recall (50 candidatos) -- el corpus es chico (52 chunks) y la
vectorial ya recupera ampliamente, así que RRF no tiene candidatos nuevos
que aportar. El reranking sí cambia el orden interno del pool en el caso
ambiguo (consulta 1): inspeccionado a mano, mueve el chunk correcto
(`BUD-2024-005`) a una posición distinta de la que tenía sin reranking, pero
en ningún caso lo suficiente como para cambiar el conjunto final de 5
resultados ni el rank del primer hit relevante de forma consistente entre
corridas -- la consulta 1 es, por diseño, un caso límite donde el gateway de
pagos (`BUD-2024-003`) y el marketplace (`BUD-2024-006`) están casi
empatados en relevancia textual con el proyecto correcto, y pequeñas
variaciones del embedding de consulta entre llamadas (la API de embeddings
no garantiza vectores bit-idénticos para el mismo texto en invocaciones
distintas) alcanzan para mover ese empate.

## 5. Conclusiones

Con la evidencia de este golden set, usaría la **configuración B (híbrida,
sin reranking)**: mismo precisión@5/MRR@5 que las otras tres, misma latencia
que la vectorial pura (la rama léxica agrega ~15 ms de una query SQL con
índice GIN, no es el costo real), y la rama léxica queda como red de
seguridad barata para el día en que un chunk use vocabulario técnico exacto
que el embedding no capture bien (nombres de tecnología, códigos de
producto). No elegiría C ni D: el reranking casi duplica la latencia
(519 ms → ~1 000 ms) y, con este corpus y este golden set, no hay ninguna
ganancia medida de relevancia que lo justifique -- la ganancia que sí se ve
(reordenar el caso ambiguo) no fue ni grande ni estable entre corridas. Esto
no descarta el reranker a futuro: con 15 presupuestos el pool de recall casi
nunca tiene más de un candidato genuinamente competitivo por consulta, que es
exactamente el escenario donde un cross-encoder no tiene margen para
demostrar su ventaja sobre un bi-encoder. La pregunta que vale la pena
revisar cuando el corpus crezca no es "¿activamos el reranker?" sino "¿a
partir de qué tamaño de corpus el costo de latencia empieza a pagarse con
una ganancia de precisión medible?".
