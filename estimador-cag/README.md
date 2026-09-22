# Estimador CAG

API REST hecha con FastAPI que recibe la transcripción de una reunión con un cliente y devuelve una estimación de software generada por un LLM.

Usa arquitectura **CAG (Cache-Augmented Generation)**: un conjunto de estimaciones previas se inyecta como contexto estático directamente en el prompt de cada petición. No hay base de datos vectorial ni búsqueda.

## Requisitos

- Python 3.13 o superior
- [uv](https://docs.astral.sh/uv/)
- Una API key de OpenAI o de Anthropic

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
| `OPENAI_API_KEY` | API key de OpenAI | sin valor (requerida si se usa OpenAI) |
| `ANTHROPIC_API_KEY` | API key de Anthropic | sin valor (requerida si se usa Anthropic) |
| `LLM_PROVIDER` | Proveedor a utilizar: `openai` o `anthropic` | `openai` |
| `LLM_MODEL` | Modelo a utilizar | `gpt-4o-mini` |
| `APP_ENV` | Entorno de ejecución | `development` |
| `LOG_LEVEL` | Nivel de logging | `DEBUG` |

Para usar Anthropic, cambia estas dos variables juntas y reinicia el servidor:

```
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5
```

El archivo `.env` contiene secretos y está en el `.gitignore`. No lo subas al repositorio.

## Ejecución

### Opción 1: Interfaz Web Conversacional (Streamlit)

Arranca la interfaz gráfica de chat con soporte de streaming y observabilidad CAG:

```bash
uv run streamlit run streamlit_app.py
```

La aplicación se abrirá automáticamente en http://localhost:8501.

### Opción 2: Backend API (FastAPI)

Arranca el servidor desde la raíz del proyecto, porque el `.env` se lee desde la carpeta actual:

```bash
uv run uvicorn app.main:app --reload
```

La documentación interactiva (Swagger) queda en http://localhost:8000/docs.

## Endpoints

### `GET /health`

Devuelve el estado del servicio, el entorno, el proveedor y el modelo activos.

### `POST /api/v1/estimate`

Recibe la transcripción de una reunión y devuelve la estimación.

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{"transcription": "El cliente necesita una landing page con formulario de contacto. Plazo de 2 semanas. El diseño ya existe en Figma."}'
```

Respuesta:

```json
{
  "estimation": "## Estimación: Landing Page ...",
  "model": "gpt-4o-mini-2024-07-18",
  "provider": "openai",
  "input_tokens": 856,
  "output_tokens": 204,
  "created_at": "2026-09-22T00:27:46.008932Z"
}
```

`transcription` debe tener entre 20 y 100.000 caracteres.

Códigos de error:

| Código | Causa |
|---|---|
| `422` | La transcripción no cumple la validación (vacía, demasiado corta o demasiado larga). |
| `500` | El servicio está mal configurado: falta la API key o el modelo o la clave no son válidos. |
| `502` | El proveedor de LLM falló o devolvió una estimación incompleta. |
| `503` | El proveedor limita las peticiones o la cuenta no tiene saldo. |

El detalle de cada error queda en el log del servidor y no se envía al cliente.

## Interfaz Web (Streamlit - Fase 2)

La interfaz conversacional (`streamlit_app.py`) proporciona:
- **Chat interactivo**: Entrada directa de transcripciones con retención del historial de conversación durante la sesión (`st.session_state`).
- **Streaming token a token**: Visualización en tiempo real conforme el modelo genera la estimación (`st.write_stream`).
- **Observabilidad CAG en panel lateral (`st.sidebar`)**:
  - Consulta en modo lectura del System Prompt activo y reglas de estimación.
  - Catálogo de estimaciones de ejemplo inyectadas estáticamente.
  - Telemetría de la última llamada: modelo utilizado, tokens de entrada, tokens de salida y tiempo de respuesta en segundos.

## Estructura

```
estimador-cag/
├── streamlit_app.py        # Interfaz web de chat con streaming (Fase 2)
├── app/
│   ├── main.py             # Aplicación FastAPI y endpoint /health
│   ├── config.py           # Configuración desde .env (Pydantic Settings)
│   ├── routers/
│   │   └── estimations.py  # Endpoint POST /api/v1/estimate y sus schemas
│   ├── services/
│   │   └── llm_service.py  # Construcción del prompt y llamadas a OpenAI/Anthropic
│   └── context/
│       └── examples.py     # Estimaciones previas inyectadas en el prompt
```

## Cómo mejorar las estimaciones

El "conocimiento" del sistema son los ejemplos de `app/context/examples.py`. Para mejorar la calidad, añade más ejemplos a la lista `ESTIMATION_EXAMPLES`, con el mismo formato que los existentes. El prompt los incluye automáticamente. Cada ejemplo nuevo aumenta los tokens de entrada de cada petición, y por tanto su coste.

