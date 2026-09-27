import time

import streamlit as st

from app.services.llm_service import ESTIMATION_EXAMPLES, SYSTEM_INSTRUCTIONS, stream_estimation


st.set_page_config(page_title="Estimador CAG", page_icon="📋")
st.title("📋 Estimador de Proyectos (CAG)")

def stream_for_ui(transcription: str):
    """Delega el streaming al wrapper (proveedor + fallback + cache ya
    resueltos ahí). Esta función solo conecta el resultado final con el
    estado de sesión de Streamlit."""
    start_time = time.perf_counter()

    def _on_complete(result):
        st.session_state.last_metrics = {
            "model": result.model,
            "provider": result.provider,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "elapsed_time": round(time.perf_counter() - start_time, 2),
        }

    return stream_estimation(transcription, on_complete=_on_complete)

def render_metrics(placeholder, metrics):
    if not metrics:
        placeholder.info("Aún no se ha generado ninguna estimación.")
        return
    with placeholder.container():
        st.markdown(f"**Modelo:** `{metrics['model']}` · **Proveedor:** `{metrics['provider']}`")
        col1, col2 = st.columns(2)
        col1.metric("Tokens Entrada", metrics["input_tokens"])
        col2.metric("Tokens Salida", metrics["output_tokens"])
        st.metric("Tiempo de respuesta", f"{metrics['elapsed_time']} s")

# --- Barra lateral: Inspección del contexto CAG ---
with st.sidebar:
    st.header("🧠 Contexto CAG")
    st.caption("Información estática inyectada en cada consulta al modelo.")

    # 1. System Prompt en modo solo lectura
    with st.expander("📜 System Prompt activo"):
        st.text(SYSTEM_INSTRUCTIONS)

    # 2. Ejemplos de estimaciones inyectados
    with st.expander(f"📚 Estimaciones de ejemplo ({len(ESTIMATION_EXAMPLES)})"):
        for i, example in enumerate(ESTIMATION_EXAMPLES, start=1):
            st.markdown(f"**Ejemplo {i}:**")
            st.info(example["meeting_summary"])
            st.markdown(example["estimation"])
            if i < len(ESTIMATION_EXAMPLES):
                st.divider()
    st.divider()
    st.subheader("Última llamada")
    metrics_placeholder = st.empty()
    # Dibujamos las métricas existentes (si las hay de una llamada previa)
    render_metrics(metrics_placeholder, st.session_state.get("last_metrics"))

# 1. Inicializar la memoria de la conversación si no existe
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Re-dibujar el historial existente en cada re-ejecución
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Capturar nueva entrada del usuario
if prompt := st.chat_input("Pega aquí la transcripción de la reunión..."):
    # Mostrar el mensaje del usuario inmediatamente
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Respuesta provisional (simulada) para probar el flujo
    with st.chat_message("assistant"):
        # st.write_stream consume el generador async token a token y al finalizar devuelve el string completo
        full_response = st.write_stream(stream_for_ui(prompt))
        # Actualizamos la barra lateral inmediatamente con los datos recién calculados
    render_metrics(metrics_placeholder, st.session_state.get("last_metrics"))


    # Guardamos la estimación completa en el historial para que persista
    st.session_state.messages.append({"role": "assistant", "content": full_response})
