# Sanity check de embeddings

Resultados reales de `scripts/compare.py` sobre `text-embedding-3-small`, contra tres parejas de referencia (textos cercanos, no relacionados, y genéricos/ambiguos).

| Pareja | Texto A | Texto B | Similitud coseno | Expectativa orientativa |
|---|---|---|---|---|
| A — cercanos | "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" | "Authorization service using JSON Web Tokens for a banking application" | **0.5957** | > 0.6 |
| B — no relacionados | "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" | "Database migration from MySQL to PostgreSQL with zero downtime" | **0.1920** | < 0.4 |
| C — genéricos/ambiguos | "Backend services" | "API development" | **0.5407** | sin expectativa fija |

## Comentario

Lo que más llama la atención: la Pareja A quedó **por debajo** del umbral orientativo de 0.6 (0.5957), aunque por muy poco. Semánticamente los dos textos hablan de lo mismo (auth/JWT en un contexto bancario), pero `text-embedding-3-small` parece ponderar bastante la forma superficial del texto (vocabulario distinto: "OAuth 2.0"/"mobile app" vs "Authorization service"/"banking application") y no solo el significado — el margen con el umbral es chico, así que no cambia la lectura cualitativa (sigue siendo la pareja más similar de las tres), pero sí muestra que un umbral fijo como 0.6 es sensible a la redacción exacta, no solo al tema.

La Pareja B confirma la expectativa con margen amplio (0.19, bien por debajo de 0.4): dos dominios técnicos sin relación semántica quedan claramente separados.

La Pareja C es la más interesante para seguir investigando: dos frases genéricas y casi tautológicas ("Backend services" / "API development") dan una similitud moderada-alta (0.54), más cerca de la Pareja A que de la B. Esto sugiere que el modelo agrupa textos cortos y poco específicos en una región genérica del espacio de embeddings en vez de discriminarlos -- un presupuesto con descripciones de componente muy telegráficas (de una sola línea, sin detalle técnico) probablemente genere chunks difíciles de distinguir entre sí por similitud semántica pura.
