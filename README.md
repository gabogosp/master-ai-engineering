# Estimador CAG

Servicio de IA (FastAPI) que genera estimaciones de software con un LLM, más un cliente web (Streamlit). Soporta dos modos: un formulario tipado de un solo turno, y una conversación multi-turno con memoria de sesión y adjuntos (PDF/Word).

Arquitectura:

```
Streamlit (formulario tipado)
        │  HTTP (POST /api/v1/estimate, bloqueante)
        ▼
FastAPI ── Jinja2 (prompts versionados) ── LiteLLM Router (fallback openai↔anthropic)
        │
        ├── Redis (cache exact-match)
        └── structlog (logging estructurado)
```

Usa arquitectura **CAG (Cache-Augmented Generation)**: un conjunto de estimaciones previas se inyecta como contexto estático (few-shot) directamente en el prompt de cada petición, vía templates Jinja2 versionados (`app/prompts/estimation/`). No hay base de datos vectorial ni búsqueda semántica.

## Requisitos

- Python 3.13 o superior
- [uv](https://docs.astral.sh/uv/)
- Docker (para levantar Redis)
- Una API key de OpenAI y/o de Anthropic

## Instalación

```bash
uv sync
cp .env.example .env
```

Después edita `.env` y rellena los valores (ver la siguiente sección).

## Configuración

Las variables se leen del archivo `.env`. Cualquier variable que falte usa su valor por defecto.

| Variable | Descripción | Por defecto |
|---|---|---|
| `OPENAI_API_KEY` | API key de OpenAI | sin valor |
| `ANTHROPIC_API_KEY` | API key de Anthropic | sin valor |
| `LLM_PROVIDER` | Proveedor primario: `openai` o `anthropic` | `openai` |
| `LLM_MODEL` | Modelo del proveedor primario | `gpt-4o-mini` |
| `APP_ENV` | Entorno de ejecución (`development` → logs en consola, `production` → logs en JSON) | `development` |
| `LOG_LEVEL` | Nivel de logging | `DEBUG` |
| `REDIS_URL` | Conexión al cache exact-match | `redis://localhost:6379/0` |
| `CACHE_TTL_SECONDS` | TTL de las entradas en cache | `86400` (24h) |
| `BACKEND_URL` | URL del servicio IA que consume el cliente Streamlit | `http://localhost:8000` |
| `MAX_CONVERSATION_TURNS` | Ventana deslizante del historial conversacional (ver sección de sesiones) | `6` |

Con **una sola** API key configurada, el servicio funciona sin fallback. Con las dos, si el proveedor primario falla (rate limit, error 5xx, timeout), el [Router de LiteLLM](https://docs.litellm.ai/) reintenta automáticamente con el otro proveedor.

El archivo `.env` contiene secretos y está en el `.gitignore`. No lo subas al repositorio.

## Ejecución

Se necesitan **tres procesos** corriendo en paralelo (en desarrollo, en terminales separadas):

**1. Redis** (cache exact-match):

```bash
docker compose up -d redis
```

**2. Servicio IA (FastAPI)** — arráncalo desde la raíz del proyecto, porque el `.env` se lee desde la carpeta actual:

```bash
uv run uvicorn app.main:app --reload
```

La documentación interactiva (Swagger) queda en http://localhost:8000/docs.

**3. Cliente web (Streamlit)** — es un cliente HTTP real del servicio IA (necesita el paso anterior corriendo):

```bash
uv run streamlit run streamlit_app.py
```

La aplicación se abre en http://localhost:8501.

## Endpoints

### `GET /health`

Devuelve el estado del servicio, el entorno, el proveedor y el modelo activos.

### `POST /api/v1/estimate`

Recibe una descripción estructurada del proyecto y devuelve la estimación completa (bloqueante). Acepta `?prompt_version=v1|v2` como query param opcional (default `v1`).

```bash
curl -X POST "http://localhost:8000/api/v1/estimate?prompt_version=v1" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "El cliente necesita una landing page con formulario de contacto. Plazo de 2 semanas. El diseño ya existe en Figma.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table"
  }'
```

Campos de `EstimationRequest`:

| Campo | Tipo | Descripción |
|---|---|---|
| `description` | string (20–2000 caracteres) | Descripción del proyecto |
| `project_type` | enum | `mobile_app` \| `web_saas` \| `internal_tool` \| `data_pipeline` |
| `detail_level` | enum | `summary` \| `medium` \| `detailed` |
| `output_format` | enum | `phases_table` \| `line_items` \| `narrative` |
| `reference_projects` | lista opcional | Proyectos pasados similares (`name`, `description`, `hours`), como calibración |

Respuesta (`EstimationResponse`):

```json
{
  "text": "## Estimación: Landing Page ...",
  "prompt_version": "v1",
  "model": "gpt-4o-mini-2024-07-18",
  "provider": "openai",
  "input_tokens": 856,
  "output_tokens": 204,
  "created_at": "2026-09-22T00:27:46.008932Z"
}
```

Códigos de error:

| Código | Causa |
|---|---|
| `422` | El body no cumple la validación, o `prompt_version` no existe. |
| `500` | El servicio está mal configurado: falta la API key del proveedor elegido. |
| `502` | El proveedor de LLM falló o devolvió una estimación incompleta. |
| `503` | El proveedor limita las peticiones o la cuenta no tiene saldo. |

El detalle de cada error queda en el log del servidor (structlog) y no se envía al cliente.

### `POST /api/v1/estimate/stream`

Igual que `/estimate` pero via Server-Sent Events: un evento `data` por chunk de texto generado, y un evento final `event: meta` con los metadatos (`model`, `provider`, `input_tokens`, `output_tokens`, `prompt_version`). Si algo falla, se emite un evento `event: error` con un mensaje genérico (la respuesta ya está comprometida a `200 text/event-stream`, no puede convertirse en un código HTTP distinto).

**El cliente Streamlit no usa este endpoint.** El contrato de esta entrega es deliberadamente bloqueante y de texto libre (`EstimationResponse.text`) — es el punto de partida sobre el que se trabaja streaming + salida estructurada en el directo de sesión 04. Este endpoint queda disponible para otros clientes que sí quieran consumir streaming.

## Prompts versionados

Los prompts viven como archivos Jinja2, no como strings en el código:

```
app/prompts/
├── loader.py                    # render_estimation_prompt(request, version="v1")
└── estimation/
    ├── v1/
    │   ├── system.j2             # rol, reglas, bloques condicionales por output_format/detail_level
    │   ├── user.j2                # envuelve la descripción + reference_projects opcionales
    │   └── examples.j2            # few-shot examples, incluido con {% include %}
    └── v2/                        # variación deliberada de tono, misma estructura
        ├── system.j2
        ├── user.j2
        └── examples.j2
```

Agregar una versión nueva (`v3/`) no requiere tocar el resto del código: el loader arma un `Environment` de Jinja2 por versión (cacheado), y el endpoint la selecciona con `?prompt_version=v3`. Cada render queda registrado en el log (`prompt_rendered`) con la versión usada y un hash del contenido, para poder auditar en producción qué prompt exacto generó cada estimación.

![Formulario del Estimador con una estimación generada, incluida la observabilidad en la sidebar](docs/streamlit-form.png)

## Interfaz Web (Streamlit)

`streamlit_app.py` es un **cliente HTTP** del servicio IA (no importa su código Python). Tiene dos pestañas:

**⚡ Estimación rápida** (sesión 04):
- Formulario tipado: descripción + `project_type`/`detail_level`/`output_format`/`prompt_version`, y una sección opcional para proyectos de referencia.
- Llamada bloqueante: hace `POST /api/v1/estimate` y muestra un spinner hasta que llega la respuesta completa — sin streaming, a propósito (ver nota en `/estimate/stream` arriba).
- Observabilidad en la sidebar: modelo, proveedor y tokens de la última llamada, y el system prompt exacto que se usó (renderizado localmente con el mismo loader, solo para inspección — la llamada real la resuelve el servidor).

**💬 Conversación** (sesión 05):
- Crea una sesión (`POST /sessions`) al cargar la página.
- Chat (`st.chat_input`) + subida de adjuntos (`st.file_uploader`, PDF/Word).
- **Streaming**: consume `POST /sessions/{id}/estimate/stream` (SSE) — el texto aparece token a token en vez del patrón spinner-y-todo-de-golpe. A diferencia de "Estimación rápida" (donde el bloqueo es a propósito, confirmado explícitamente para ese ejercicio), acá no había ninguna restricción equivalente, así que se corrigió el anti-patrón.
- Panel expandible con el `project_metadata` actual y cuántos turnos hay en la ventana deslizante — útil para ver en vivo la separación entre memoria y historial. Se actualiza cuando termina el stream (evento `meta`).
- Botón "Nueva conversación" que crea una sesión nueva y resetea el estado local.

## Conversación multi-turno con memoria (sesión 05)

Hasta acá, el estimador era transaccional: una petición, una respuesta, sin estado. Esta entrega agrega un flujo **conversacional** con memoria de sesión — el contexto del proyecto se preserva entre turnos sin reenviar todo el historial crudo cada vez.

### `POST /api/v1/sessions`

Crea una sesión vacía y devuelve `{"session_id": "<uuid4>"}`. Las sesiones viven en un diccionario en memoria del proceso — sin base de datos ni Redis. Se pierden al reiniciar el servicio; es una volatilidad aceptada a propósito en esta fase (ver docstring de `SessionStore` en `app/sessions.py`).

### `POST /api/v1/sessions/{session_id}/estimate`

`multipart/form-data` con dos campos:

| Campo | Tipo | Descripción |
|---|---|---|
| `transcript` | string | Lo nuevo que aporta este turno (no todo el historial — eso lo gestiona el servidor) |
| `attachments` | archivos (opcional) | PDFs o `.docx` con especificaciones adicionales |

```bash
SID=$(curl -s -X POST http://localhost:8000/api/v1/sessions | python3 -c "import sys,json;print(json.load(sys.stdin)['session_id'])")

curl -X POST "http://localhost:8000/api/v1/sessions/$SID/estimate" \
  -F "transcript=El cliente es una clínica veterinaria y quiere un sistema de turnos."
```

La respuesta es el mismo `EstimationResponse` del endpoint de formulario, con dos campos extra que solo este endpoint completa:

```json
{
  "text": "## Estimación ...",
  "prompt_version": "v1",
  "model": "gpt-4o-mini-2024-07-18",
  "provider": "openai",
  "input_tokens": 1420,
  "output_tokens": 240,
  "created_at": "2026-10-05T23:00:00Z",
  "project_metadata": {
    "project_name": "Sistema de Turnos VetCare",
    "assumed_team_size": null,
    "mentioned_technologies": [],
    "agreed_scope": "Sistema de turnos online con recordatorios por email"
  },
  "history_turns": 1
}
```

### `POST /api/v1/sessions/{session_id}/estimate/stream`

Igual que el anterior pero via Server-Sent Events — es el que usa el cliente Streamlit. Un evento `data` por chunk de texto, y un evento final `event: meta` con los mismos campos extra (`project_metadata`, `history_turns`) más `model`/`provider`/tokens/`prompt_version`. Mismo matiz que `/api/v1/estimate/stream`: una vez que sale el primer chunk la respuesta está comprometida a `200 text/event-stream`, así que los errores (sesión inexistente incluida) se comunican como `event: error`, no como un código HTTP distinto.

A diferencia de `/api/v1/estimate/stream` (que el formulario de la pestaña "Estimación rápida" **no** usa, por la confirmación explícita del profesor sobre ese ejercicio), este sí es el que consume la pestaña "Conversación" — ahí no había una restricción equivalente contra el streaming.

### Decisión: adjuntos — Camino B (extracción local)

Se eligió **extracción local** (`pypdf` para PDF, `python-docx` para Word) en vez de subir el archivo directo a la Files API de un proveedor (Camino A). Razón: el wrapper de LLM (`llm_service.py`) es agnóstico de proveedor desde el Bloque A — el Router de LiteLLM hace fallback automático entre OpenAI y Anthropic. Atar un adjunto a la Files API de un proveedor específico rompería esa propiedad justo para las peticiones con adjuntos. Extraer el texto localmente (`app/attachments.py`) mantiene el fallback intacto en todos los casos, y de paso deja el terreno preparado para chunking de RAG (módulo 3). El texto de cada adjunto se concatena al transcript con un separador `--- attachment: <filename> ---`.

### Decisión: extracción de `project_metadata` — LLM extractor

Después de cada turno, una **segunda llamada al LLM** (prompt propio en `app/prompts/metadata_extraction/v1/`) extrae en JSON qué se aprendió del turno, y se fusiona sobre lo que ya se sabía (`ProjectMetadata.merge`, en `app/sessions.py`). Se eligió esto en vez de una heurística con regex porque el texto es libre y en español, con redacción variable turno a turno — una regex sería frágil. El costo extra (una llamada más por turno, con un modelo barato) se consideró aceptable. Si el LLM devuelve algo que no parsea como JSON, se degrada a **no actualizar nada** en vez de romper la petición — la extracción de metadata es una mejora de la experiencia conversacional, no algo que deba poder tumbar una estimación que sí se generó bien.

### `project_metadata` separado del historial

`project_metadata` (hechos conocidos: nombre del proyecto, equipo asumido, tecnologías, alcance) vive **aparte** del historial de mensajes (`ConversationHistory`). El historial es "qué se dijo"; el metadata es "qué sabemos". Se inyecta en el `<project_metadata>` del `system.j2` **regenerado en cada turno** — nunca queda desactualizado, y no ocupa espacio en la ventana deslizante del historial.

![Conversación de tres turnos con el panel de project_metadata visible, mostrando cómo la estimación evoluciona (100 → 128 horas) a medida que se acumula información](docs/streamlit-conversacion.png)

### Ventana deslizante del historial

`ConversationHistory` guarda pares user+assistant y, al armar `to_messages_list()`, conserva solo los últimos `MAX_CONVERSATION_TURNS` (6 por defecto, configurable). El system prompt no vive en el historial — se regenera fresco cada vez a partir del `project_metadata` actual, así que siempre viaja aunque los turnos más viejos se descarten.

## Estructura

```
master-ai-engineering/
├── streamlit_app.py             # Cliente web (2 pestañas: formulario y conversación)
├── docker-compose.yml           # Redis para el cache exact-match
├── tests/
│   ├── conftest.py
│   ├── test_api.py              # Endpoints, códigos de error, prompt_version
│   ├── test_cache.py            # Clave de cache exact-match
│   ├── test_config.py           # Settings
│   ├── test_schemas.py          # Validación Pydantic de los schemas
│   ├── test_sessions_integration.py  # Sesiones end-to-end (httpx.AsyncClient)
│   ├── test_stream_and_ui.py    # Streamlit vía AppTest (mock de httpx)
│   └── prompts/
│       ├── test_estimation_v1.py
│       └── test_estimation_v2.py
├── app/
│   ├── main.py                  # App FastAPI + logging + endpoint /health
│   ├── config.py                # Configuración desde .env
│   ├── schemas.py                # EstimationRequest/Response, enums, ReferenceProject
│   ├── sessions.py                # ProjectMetadata, ConversationHistory, SessionStore
│   ├── attachments.py             # Extracción local de texto (pypdf / python-docx)
│   ├── logging_config.py         # structlog (consola en dev, JSON en prod)
│   ├── prompts/
│   │   ├── loader.py              # render_estimation_prompt, render_session_prompt
│   │   ├── estimation/{v1,v2}/    # Templates Jinja2 de estimación
│   │   └── metadata_extraction/v1/  # Templates Jinja2 de extracción de metadata
│   ├── routers/
│   │   ├── estimations.py         # POST /estimate y /estimate/stream
│   │   └── sessions.py            # POST /sessions, /estimate y /estimate/stream (SSE)
│   └── services/
│       ├── llm_service.py         # Wrapper LiteLLM: fallback, cache, streaming, multi-turno
│       ├── metadata_extraction.py # Segunda llamada al LLM que extrae project_metadata
│       └── cache.py                # Cache exact-match sobre Redis
```

## Tests

```bash
uv run pytest -v
```

Todos corren con mocks (sin API keys reales; Redis sí debe estar corriendo porque el wrapper lo usa en cada llamada, aunque las entradas se descartan entre corridas):

- Endpoints y mapeo de errores HTTP (`422`, `500`, `502`, `503`), incluida la versión de prompt.
- Validación pura de Pydantic (`EstimationRequest`, `ReferenceProject`): límites de longitud, enum inválido, campo requerido faltante.
- Clave de cache exact-match (determinismo, sensibilidad a cada parámetro).
- Templates de prompt (`v1` y `v2`): contenido literal de la descripción, bloques condicionales mutuamente excluyentes, `reference_projects`, versión inexistente, semántica de `StrictUndefined`.
- Streamlit (`AppTest`) con `httpx.MockTransport`: formulario, respuesta bloqueante, validación, turno conversacional.
- **Integración de sesiones** (`test_sessions_integration.py`, con `httpx.AsyncClient` sobre la app vía `ASGITransport`): `project_metadata` se actualiza y se fusiona a través de dos turnos; el contenido de un PDF adjunto efectivamente llega al LLM; la ventana deslizante nunca manda más de `MAX_CONVERSATION_TURNS` turnos al modelo aunque la sesión tenga más historial acumulado; el endpoint SSE de sesión produce chunks de texto y un evento `meta` final con el mismo resultado que el bloqueante.

## Cómo mejorar las estimaciones

El "conocimiento" del sistema son los ejemplos few-shot de `app/prompts/estimation/v1/examples.j2` (y `v2/examples.j2`). Para mejorar la calidad, agregá ejemplos ahí con el mismo formato XML-ish que los existentes — no hace falta tocar código Python. Cada ejemplo nuevo aumenta los tokens de entrada de cada petición, y por tanto su coste.
