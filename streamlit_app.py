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
    """POST bloqueante a /api/v1/estimate: la respuesta llega completa de una
    sola vez (texto libre, sin streaming). Es el contrato que pide esta
    entrega — /api/v1/estimate/stream (SSE) sigue existiendo en el backend
    para otros clientes, pero este formulario ya no lo usa."""
    url = f"{settings.backend_url}/api/v1/estimate"
    with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
        response = client.post(url, json=payload, params={"prompt_version": prompt_version})

    if response.status_code != 200:
        detail = response.text
        if "application/json" in response.headers.get("content-type", ""):
            detail = response.json().get("detail", detail)
        raise RuntimeError(f"Error {response.status_code}: {detail}")

    return response.json()


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


# --- Barra lateral: observabilidad de la última llamada ---
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


# --- Formulario tipado (reemplaza al chat de texto libre) ---
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
            st.markdown(result["text"])
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
        render_prompt_preview(prompt_placeholder, system_preview, metrics_holder.get("prompt_version"))
        render_metrics(metrics_placeholder, metrics_holder)
