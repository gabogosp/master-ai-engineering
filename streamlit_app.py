import streamlit as st
import openai
import time
from app.config import settings
from app.services.llm_service import build_system_prompt, TEMPERATURE, MAX_OUTPUT_TOKENS, SYSTEM_INSTRUCTIONS, ESTIMATION_EXAMPLES


st.set_page_config(page_title="Estimador CAG", page_icon="📋")
st.title("📋 Estimador de Proyectos (CAG)")

def stream_estimation(transcription: str):
    client = openai.OpenAI(api_key=settings.openai_api_key)
    start_time = time.perf_counter()  # Empezamos a cronometrar
    # Llamamos a OpenAI con stream=True
    response_stream = client.responses.create(
        model=settings.llm_model,
        instructions=build_system_prompt(),  # Reutiliza el prompt con los ejemplos CAG
        input=transcription,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        stream=True,
    )

    # Cada vez que llega un fragmento de texto, hacemos yield
    for event in response_stream:
        if event.type == "response.output_text.delta":
            yield event.delta
        elif event.type == "response.completed":  # Evento final con estadísticas
            elapsed = time.perf_counter() - start_time
            st.session_state.last_metrics = {
                "model": event.response.model,
                "input_tokens": event.response.usage.input_tokens if event.response.usage else 0,
                "output_tokens": event.response.usage.output_tokens if event.response.usage else 0,
                "elapsed_time": round(elapsed, 2),
            }

def render_metrics(placeholder, metrics):
    if not metrics:
        placeholder.info("Aún no se ha generado ninguna estimación.")
        return
    with placeholder.container():
        st.markdown(f"**Modelo:** `{metrics['model']}`")
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
        # st.write_stream consume el generador token a token y al finalizar devuelve el string completo
        full_response = st.write_stream(stream_estimation(prompt))
        # Actualizamos la barra lateral inmediatamente con los datos recién calculados
    render_metrics(metrics_placeholder, st.session_state.get("last_metrics"))


    # Guardamos la estimación completa en el historial para que persista
    st.session_state.messages.append({"role": "assistant", "content": full_response})
