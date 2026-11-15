import time

import httpx
import streamlit as st

from app.config import settings
from app.prompts.loader import render_estimation_prompt
from app.schemas import DetailLevel, EstimationRequest, OutputFormat, ProjectType

PROMPT_VERSIONS = ["v1", "v2"]
MAX_REFERENCE_PROJECTS = 2

st.set_page_config(page_title="Estimador CAG", page_icon="📋")
st.title("📋 Estimador de Proyectos")


def call_estimate(payload: dict, prompt_version: str) -> dict:
    """POST bloqueante a /api/v1/estimate. La respuesta ya no es texto libre
    sino un EstimationResult estructurado (ver app.schemas) -- Instructor
    valida la forma contra el modelo antes de devolverla, lo que a su vez
    hace que no se pueda transmitir en streaming token por token."""
    url = f"{settings.backend_url}/api/v1/estimate"
    with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
        response = client.post(url, json=payload, params={"prompt_version": prompt_version})

    if response.status_code != 200:
        detail = response.text
        if "application/json" in response.headers.get("content-type", ""):
            detail = response.json().get("detail", detail)
        raise RuntimeError(f"Error {response.status_code}: {detail}")

    return response.json()


def call_session_estimate(session_id: str, transcript: str, uploaded_files: list) -> dict:
    """POST bloqueante multipart/form-data a /api/v1/sessions/{id}/estimate."""
    url = f"{settings.backend_url}/api/v1/sessions/{session_id}/estimate"
    files = [
        ("attachments", (f.name, f.getvalue(), f.type or "application/octet-stream"))
        for f in uploaded_files
    ]
    with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
        response = client.post(url, data={"transcript": transcript}, files=files or None)

    if response.status_code != 200:
        detail = response.text
        if "application/json" in response.headers.get("content-type", ""):
            detail = response.json().get("detail", detail)
        raise RuntimeError(f"Error {response.status_code}: {detail}")

    return response.json()


def format_estimation_markdown(result: dict) -> str:
    """Convierte un EstimationResult (dict) en un bloque markdown único:
    summary, una fase por bullet (con su propio summary) y los totales. Se
    guarda como texto plano en el historial de conversación, así el replay
    de mensajes (st.markdown sobre el string guardado) no necesita conocer
    la forma estructurada."""
    lines = [result["summary"], ""]
    for phase in result["phases"]:
        lines.append(
            f"- **{phase['name']}** — {phase['duration_weeks']} semana(s), "
            f"{phase['cost_eur']:,} EUR: {phase['summary']}"
        )
    lines.append("")
    lines.append(
        f"**Total:** {result['total_duration_weeks']} semana(s) · "
        f"{result['total_cost_eur']:,} EUR · Confianza: {result['confidence_pct']}%"
    )
    return "\n".join(lines)


def render_metrics(placeholder, metrics: dict | None):
    if not metrics:
        placeholder.info("Aún no se ha generado ninguna estimación.")
        return
    with placeholder.container():
        if metrics.get("error"):
            st.error(metrics["error"])
            return
        st.markdown(
            f"**Modelo:** `{metrics.get('model', '?')}` · "
            f"**Proveedor:** `{metrics.get('provider', '?')}` · "
            f"**Prompt:** `{metrics.get('prompt_version', '?')}`"
        )
        col1, col2 = st.columns(2)
        col1.metric("Tokens Entrada", metrics.get("input_tokens", 0))
        col2.metric("Tokens Salida", metrics.get("output_tokens", 0))
        st.metric("Tiempo de respuesta", f"{metrics.get('elapsed_time', 0)} s")


def render_prompt_preview(placeholder, system_prompt: str | None, prompt_version: str | None):
    if not system_prompt:
        placeholder.info("Generá una estimación para ver el system prompt utilizado.")
        return
    with placeholder.container():
        st.caption(f"System prompt de la última estimación (versión `{prompt_version}`).")
        with st.expander("Ver system prompt"):
            st.text(system_prompt)


def create_conversation_session() -> str:
    url = f"{settings.backend_url}/api/v1/sessions"
    with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
        response = client.post(url)
    response.raise_for_status()
    return response.json()["session_id"]


def render_project_metadata(placeholder, metadata: dict | None, history_turns: int | None):
    with placeholder.container():
        if not metadata or not any(metadata.values()):
            st.info("Todavía no se conoce ningún dato del proyecto.")
        else:
            st.markdown(f"**Nombre del proyecto:** {metadata.get('project_name') or '—'}")
            st.markdown(f"**Equipo asumido:** {metadata.get('assumed_team_size') or '—'}")
            techs = metadata.get("mentioned_technologies") or []
            st.markdown(f"**Tecnologías mencionadas:** {', '.join(techs) if techs else '—'}")
            st.markdown(f"**Alcance acordado:** {metadata.get('agreed_scope') or '—'}")
        st.caption(
            f"Turnos en la ventana deslizante: {history_turns if history_turns is not None else 0} "
            f"(máx. {MAX_CONVERSATION_TURNS_HINT})"
        )


MAX_CONVERSATION_TURNS_HINT = 6  # debe coincidir con settings.max_conversation_turns

CITATION_STATUS_LABELS = {
    "grounded": "✅ Fundamentada",
    "dangling_citation": "⚠️ Cita inválida",
    "insufficient_data": "➖ Sin datos suficientes",
}
SEARCH_MODE_LABELS = {"hybrid": "Híbrida (recomendada)", "vector": "Solo vectorial"}
MIN_GROUNDED_QUERY_CHARS = 20


def call_grounded_estimate(payload: dict) -> dict:
    """POST bloqueante a /estimate/grounded. Es una ruta de primer nivel: no
    cuelga de /api/v1 como las estimaciones originales."""
    url = f"{settings.backend_url}/estimate/grounded"
    try:
        with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
            response = client.post(url, json=payload)
    except httpx.RequestError as exc:
        raise RuntimeError(f"No se pudo conectar con el servicio: {exc}") from exc

    if response.status_code != 200:
        detail = response.text
        if "application/json" in response.headers.get("content-type", ""):
            detail = response.json().get("detail", detail)
        raise RuntimeError(f"Error {response.status_code}: {detail}")

    return response.json()


def render_grounded_result(body: dict) -> None:
    """Muestra la estimación con fuentes: totales, estado de cada línea y,
    por componente, la evidencia citada junto al texto real del fragmento
    histórico, para que quien la lea pueda comprobar la cita por sí mismo."""
    estimate = body["estimate"]
    verdicts = body["citation_report"]["verdicts"]
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in body["retrieved_chunks"]}
    counts = {
        status: sum(1 for verdict in verdicts if verdict["status"] == status)
        for status in CITATION_STATUS_LABELS
    }

    st.markdown(estimate["summary"])

    cols = st.columns(4)
    cols[0].metric("Horas verificadas", f"{body['verified_hours']:g}")
    for col, status in zip(cols[1:], CITATION_STATUS_LABELS, strict=True):
        col.metric(CITATION_STATUS_LABELS[status], counts[status])

    if not body["retrieved_chunks"]:
        st.warning("No se recuperó ningún fragmento histórico: no hay base para estimar.")
    if counts["dangling_citation"]:
        st.error(
            f"{counts['dangling_citation']} línea(s) citan fuentes que no existen en el "
            "contexto recuperado. Sus horas no cuentan como verificadas."
        )
    if body["claimed_hours"] != body["verified_hours"]:
        st.warning(
            f"El modelo declaró {body['claimed_hours']:g} h, pero solo "
            f"{body['verified_hours']:g} h tienen citas verificadas."
        )
    if body["corrected_after_dangling"]:
        st.info("El modelo citó una fuente inexistente y se le pidió corregirla con un reintento.")

    st.dataframe(
        [
            {
                "Componente": line["component"],
                "Horas": line["hours"],
                "Estado": CITATION_STATUS_LABELS[verdict["status"]],
                "Fuentes": ", ".join(
                    f"chunk {source['chunk_id']} ({source['document_id']})"
                    for source in line["sources"]
                )
                or "—",
            }
            for line, verdict in zip(estimate["lines"], verdicts, strict=True)
        ],
        hide_index=True,
        width="stretch",
    )

    if estimate["lines"]:
        st.markdown("**Detalle por componente** (abrí cada uno para ver su evidencia)")
    for line, verdict in zip(estimate["lines"], verdicts, strict=True):
        label = (
            f"{CITATION_STATUS_LABELS[verdict['status']]} · "
            f"{line['component']} · {line['hours']:g} h"
        )
        with st.expander(label):
            st.markdown(line["rationale"])
            for source in line["sources"]:
                st.markdown(
                    f"**Fuente:** fragmento `{source['chunk_id']}` del presupuesto "
                    f"`{source['document_id']}`"
                )
                st.markdown(f"**Evidencia citada:** “{source['evidence']}”")
                chunk = chunks_by_id.get(source["chunk_id"])
                if chunk is None:
                    st.error("Ese fragmento no existe en el contexto recuperado: la cita es inválida.")
                else:
                    st.caption("Texto real del fragmento citado:")
                    st.code(chunk["content"], language=None, wrap_lines=True)

    with st.expander(f"📚 Contexto recuperado ({len(body['retrieved_chunks'])} fragmentos)"):
        for chunk in body["retrieved_chunks"]:
            st.markdown(f"**Fragmento `{chunk['chunk_id']}`** · presupuesto `{chunk['document_id']}`")
            st.code(chunk["content"], language=None, wrap_lines=True)

    st.caption(f"Identificador de la petición (para buscarla en los logs): `{body['request_id']}`")


# --- Barra lateral: observabilidad de la última llamada (estimación rápida) ---
with st.sidebar:
    st.header("🧠 Observabilidad")
    prompt_placeholder = st.empty()
    st.divider()
    st.subheader("Última llamada")
    metrics_placeholder = st.empty()

render_prompt_preview(
    prompt_placeholder,
    st.session_state.get("last_system_prompt"),
    st.session_state.get("last_metrics", {}).get("prompt_version"),
)
render_metrics(metrics_placeholder, st.session_state.get("last_metrics"))


tab_quick, tab_conversation, tab_grounded = st.tabs(
    ["⚡ Estimación rápida", "💬 Conversación", "🔎 Estimación con fuentes"]
)

# --- Pestaña 1: formulario tipado de un solo turno ---
with tab_quick:
    with st.form("estimation_form"):
        description = st.text_area(
            "Descripción del proyecto",
            placeholder="Describí el proyecto: qué necesita el cliente, alcance, plazos...",
            height=150,
        )
        col1, col2, col3 = st.columns(3)
        project_type = col1.selectbox("Tipo de proyecto", [t.value for t in ProjectType])
        detail_level = col2.selectbox(
            "Nivel de detalle", [d.value for d in DetailLevel], index=1
        )
        output_format = col3.selectbox("Formato de salida", [f.value for f in OutputFormat])
        prompt_version = st.selectbox(
            "Versión de prompt",
            PROMPT_VERSIONS,
            help="v1 y v2 comparten las mismas reglas de negocio; v2 tiene un tono más directo.",
        )

        with st.expander("➕ Proyectos de referencia (opcional)"):
            st.caption(
                "Proyectos pasados similares, para calibrar horas y alcance. "
                "Dejá el nombre vacío para omitir una fila."
            )
            reference_inputs = []
            for i in range(MAX_REFERENCE_PROJECTS):
                rcol1, rcol2, rcol3 = st.columns([2, 3, 1])
                name = rcol1.text_input("Nombre", key=f"ref_name_{i}")
                ref_description = rcol2.text_input("Descripción breve", key=f"ref_desc_{i}")
                hours = rcol3.number_input(
                    "Horas reales", min_value=0, step=1, key=f"ref_hours_{i}"
                )
                reference_inputs.append((name, ref_description, hours))

        submitted = st.form_submit_button("Generar estimación")

    if submitted:
        if len(description) < 20:
            st.error("La descripción debe tener al menos 20 caracteres.")
        else:
            reference_projects = [
                {"name": name, "description": ref_description, "hours": hours}
                for name, ref_description, hours in reference_inputs
                if name and ref_description and hours > 0
            ]

            payload = {
                "description": description,
                "project_type": project_type,
                "detail_level": detail_level,
                "output_format": output_format,
                "reference_projects": reference_projects or None,
            }

            # Render local del prompt, solo para mostrarlo en la sidebar. No
            # participa en la llamada real: esa la hace el servicio IA por HTTP,
            # renderizando el mismo template de nuevo del lado del servidor.
            system_preview, _ = render_estimation_prompt(
                EstimationRequest(**payload), version=prompt_version
            )

            st.subheader("Resultado")
            start_time = time.perf_counter()
            try:
                with st.spinner("Generando estimación..."):
                    result = call_estimate(payload, prompt_version)
            except RuntimeError as exc:
                st.error(str(exc))
                metrics_holder = {"error": str(exc)}
            else:
                st.markdown(format_estimation_markdown(result["result"]))
                metrics_holder = {
                    "model": result["model"],
                    "provider": result["provider"],
                    "prompt_version": result["prompt_version"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "elapsed_time": round(time.perf_counter() - start_time, 2),
                }

            st.session_state.last_metrics = metrics_holder
            st.session_state.last_system_prompt = system_preview

            # Actualizamos los placeholders de la sidebar in-place (sin rerun):
            # st.empty() permite reescribir su contenido más adelante en la
            # misma corrida del script.
            render_prompt_preview(
                prompt_placeholder, system_preview, metrics_holder.get("prompt_version")
            )
            render_metrics(metrics_placeholder, metrics_holder)


# --- Pestaña 2: conversación multi-turno con memoria ---
with tab_conversation:
    if "session_id" not in st.session_state:
        st.session_state.session_id = create_conversation_session()
        st.session_state.conversation_messages = []
        st.session_state.session_metadata = None
        st.session_state.session_history_turns = 0

    col_header, col_button = st.columns([4, 1])
    col_header.caption(f"Sesión: `{st.session_state.session_id}`")
    if col_button.button("🔄 Nueva conversación"):
        st.session_state.session_id = create_conversation_session()
        st.session_state.conversation_messages = []
        st.session_state.session_metadata = None
        st.session_state.session_history_turns = 0
        st.rerun()

    with st.expander("🧠 Memoria del proyecto (project_metadata)", expanded=True):
        metadata_placeholder = st.empty()
        render_project_metadata(
            metadata_placeholder,
            st.session_state.session_metadata,
            st.session_state.session_history_turns,
        )

    for message in st.session_state.conversation_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    uploaded_files = st.file_uploader(
        "Adjuntos (PDF o Word, opcional)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.session_id}",
    )

    if transcript := st.chat_input("Contá más sobre el proyecto..."):
        with st.chat_message("user"):
            st.markdown(transcript)
        st.session_state.conversation_messages.append({"role": "user", "content": transcript})

        start_time = time.perf_counter()
        with st.chat_message("assistant"):
            try:
                with st.spinner("Generando estimación..."):
                    response = call_session_estimate(
                        st.session_state.session_id, transcript, uploaded_files or []
                    )
            except RuntimeError as exc:
                st.error(str(exc))
                st.session_state.last_metrics = {"error": str(exc)}
            else:
                formatted = format_estimation_markdown(response["result"])
                st.markdown(formatted)
                st.session_state.conversation_messages.append(
                    {"role": "assistant", "content": formatted}
                )
                st.session_state.session_metadata = response.get("project_metadata")
                st.session_state.session_history_turns = response.get("history_turns")
                render_project_metadata(
                    metadata_placeholder,
                    st.session_state.session_metadata,
                    st.session_state.session_history_turns,
                )

                # La sidebar de "Observabilidad" es compartida entre las dos
                # pestañas: un turno de conversación también la actualiza, igual
                # que lo hace una estimación rápida.
                st.session_state.last_metrics = {
                    "model": response.get("model"),
                    "provider": response.get("provider"),
                    "prompt_version": response.get("prompt_version"),
                    "input_tokens": response.get("input_tokens"),
                    "output_tokens": response.get("output_tokens"),
                    "elapsed_time": round(time.perf_counter() - start_time, 2),
                }
        render_metrics(metrics_placeholder, st.session_state.last_metrics)


# --- Pestaña 3: estimación con fuentes verificables ---
with tab_grounded:
    st.caption(
        "Cada línea de la estimación cita el fragmento de presupuesto histórico del que "
        "se derivó, y el sistema verifica que esa fuente exista de verdad entre los "
        "fragmentos recuperados. Los presupuestos históricos están en inglés: las "
        "consultas en inglés dan mejores resultados."
    )
    with st.form("grounded_form"):
        grounded_query = st.text_area(
            "Descripción del proyecto",
            placeholder="Ej.: Real-time payment gateway with fraud detection and settlement reconciliation",
            height=120,
            key="grounded_query",
        )
        gcol1, gcol2, gcol3 = st.columns(3)
        grounded_search_mode = gcol1.selectbox(
            "Tipo de búsqueda",
            list(SEARCH_MODE_LABELS),
            format_func=SEARCH_MODE_LABELS.get,
            key="grounded_search_mode",
        )
        grounded_k = gcol2.number_input(
            "Fragmentos a recuperar (k)", min_value=1, max_value=10, value=5, step=1,
            key="grounded_k",
        )
        grounded_rerank = gcol3.checkbox(
            "Reordenar con reranker (más lento)", key="grounded_rerank"
        )
        grounded_submitted = st.form_submit_button("Generar estimación con fuentes")

    if grounded_submitted:
        if len(grounded_query) < MIN_GROUNDED_QUERY_CHARS:
            st.session_state.grounded_outcome = {
                "error": f"La descripción debe tener al menos {MIN_GROUNDED_QUERY_CHARS} caracteres."
            }
        else:
            try:
                with st.spinner("Buscando fuentes y generando la estimación..."):
                    st.session_state.grounded_outcome = {
                        "body": call_grounded_estimate(
                            {
                                "query": grounded_query,
                                "k": int(grounded_k),
                                "search_mode": grounded_search_mode,
                                "rerank": grounded_rerank,
                            }
                        )
                    }
            except RuntimeError as exc:
                st.session_state.grounded_outcome = {"error": str(exc)}

    # Se guarda en session_state para que el resultado siga visible aunque otra
    # pestaña dispare un rerun.
    outcome = st.session_state.get("grounded_outcome")
    if outcome and "error" in outcome:
        st.error(outcome["error"])
    elif outcome:
        st.subheader("Resultado")
        render_grounded_result(outcome["body"])
