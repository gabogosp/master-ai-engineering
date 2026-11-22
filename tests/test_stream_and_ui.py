from pathlib import Path
from unittest.mock import patch

import httpx
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "streamlit_app.py")

FAKE_RESULT = {
    "summary": "Resumen de prueba para el sistema de tickets.",
    "confidence_pct": 80,
    "phases": [
        {
            "name": "Implementación",
            "duration_weeks": 4,
            "cost_eur": 10_000,
            "summary": "Desarrollo del backend y frontend principal del proyecto.",
        }
    ],
    "total_duration_weeks": 4,
    "total_cost_eur": 10_000,
}

FAKE_RESPONSE_BODY = {
    "result": FAKE_RESULT,
    "prompt_version": "v1",
    "model": "gpt-4o-mini-mock",
    "provider": "openai",
    "input_tokens": 111,
    "output_tokens": 22,
    "created_at": "2026-09-27T00:00:00Z",
}

FAKE_SESSION_ID = "11111111-1111-1111-1111-111111111111"

FAKE_SESSION_RESPONSE_BODY = {
    **FAKE_RESPONSE_BODY,
    "project_metadata": {
        "project_name": "Turnos VetCare",
        "assumed_team_size": 2,
        "mentioned_technologies": ["FastAPI"],
        "agreed_scope": None,
    },
    "history_turns": 1,
}


def _fake_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/v1/sessions":
        return httpx.Response(200, json={"session_id": FAKE_SESSION_ID})
    if path == "/api/v1/estimate":
        return httpx.Response(200, json=FAKE_RESPONSE_BODY)
    if path == f"/api/v1/sessions/{FAKE_SESSION_ID}/estimate":
        return httpx.Response(200, json=FAKE_SESSION_RESPONSE_BODY)
    raise AssertionError(f"Llamada inesperada a {path}")


_RealClient = httpx.Client


def _mock_httpx_client(*args, **kwargs):
    """Reemplaza httpx.Client por uno con un MockTransport: intercepta la
    llamada real por red y devuelve respuestas fijas para /sessions,
    /estimate y /sessions/{id}/estimate, sin necesitar un servidor
    corriendo.

    Usa `_RealClient` (guardada antes del patch) para no recursar contra el
    propio mock.
    """
    return _RealClient(transport=httpx.MockTransport(_fake_handler))


def _find_button(at, label: str):
    return next(b for b in at.button if b.label == label)


def test_streamlit_initial_render():
    """Verifica que el formulario (ya no el chat) renderice sus elementos
    iniciales, incluida la pestaña de conversación (que crea una sesión al
    cargar la página)."""
    with patch("httpx.Client", side_effect=_mock_httpx_client):
        at = AppTest.from_file(APP_PATH).run()

        assert not at.exception
        assert at.title[0].value == "📋 Estimador de Proyectos"
        # quick estimate + grounded estimate + agent
        assert len(at.text_area) == 3
        # type, detail, format, prompt version + search mode (grounded)
        # + sample, model and effort (agent)
        assert len(at.selectbox) == 8
        assert len(at.sidebar) > 0
        assert at.session_state["session_id"] == FAKE_SESSION_ID


def test_streamlit_form_submission_shows_full_response():
    """Simula completar el formulario y enviarlo: la respuesta llega de una
    sola vez (bloqueante, estructurada) y las métricas quedan en
    session_state."""
    with patch("httpx.Client", side_effect=_mock_httpx_client):
        at = AppTest.from_file(APP_PATH).run()

        at.text_area[0].set_value(
            "El cliente necesita un sistema de tickets de soporte con estados y comentarios internos."
        )
        _find_button(at, "Generar estimación").click().run()

        assert not at.exception

        full_text = "\n".join(md.value for md in at.markdown)
        assert FAKE_RESULT["summary"] in full_text
        assert "Implementación" in full_text

        metrics = at.session_state["last_metrics"]
        assert metrics["model"] == "gpt-4o-mini-mock"
        assert metrics["provider"] == "openai"
        assert metrics["prompt_version"] == "v1"
        assert metrics["input_tokens"] == 111
        assert metrics["output_tokens"] == 22
        assert "elapsed_time" in metrics


def test_streamlit_form_rejects_short_description():
    """El formulario debe validar la longitud mínima antes de llamar al
    backend (no se llega a pegarle a /api/v1/estimate)."""
    with patch("httpx.Client", side_effect=_mock_httpx_client):
        at = AppTest.from_file(APP_PATH).run()

        at.text_area[0].set_value("Muy corto")
        _find_button(at, "Generar estimación").click().run()

        assert not at.exception
        assert any("al menos 20 caracteres" in e.value for e in at.error)
        assert "last_metrics" not in at.session_state


def test_streamlit_conversation_turn_updates_metadata():
    """Simula un turno en la pestaña de conversación: la sesión ya se creó
    al cargar la página, el turno llega como chat_input, y el
    project_metadata + history_turns devueltos quedan en session_state."""
    with patch("httpx.Client", side_effect=_mock_httpx_client):
        at = AppTest.from_file(APP_PATH).run()

        at.chat_input[0].set_value(
            "El cliente es una clínica veterinaria y quiere un sistema de turnos."
        ).run()

        assert not at.exception

        messages = at.session_state["conversation_messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert FAKE_RESULT["summary"] in messages[1]["content"]

        metadata = at.session_state["session_metadata"]
        assert metadata["project_name"] == "Turnos VetCare"
        assert metadata["mentioned_technologies"] == ["FastAPI"]
        assert at.session_state["session_history_turns"] == 1

        # La sidebar de Observabilidad es compartida entre pestañas: un
        # turno de conversación también debe actualizar last_metrics.
        last_metrics = at.session_state["last_metrics"]
        assert last_metrics["model"] == "gpt-4o-mini-mock"
        assert last_metrics["provider"] == "openai"
        assert last_metrics["input_tokens"] == 111
        assert last_metrics["output_tokens"] == 22
        assert "elapsed_time" in last_metrics


# --- Pestaña "Estimación con fuentes" -------------------------------------

GROUNDED_BUTTON = "Generar estimación con fuentes"
GROUNDED_QUERY = "Real-time payment gateway with fraud detection and settlement reconciliation"

_CHUNK_9 = (
    "[Project: Real-time payments gateway]\n\nComponent: Payment gateway core\n"
    "Description: High-throughput authorization service.\nEstimated hours: 220"
)
_CHUNK_10 = "Component: Real-time fraud detection\nEstimated hours: 200"


def _grounded_body(*, dangling=False, corrected=False, chunks=True) -> dict:
    cited_chunk = "999" if dangling else "9"
    gateway_status = "dangling_citation" if dangling else "grounded"
    return {
        "request_id": "req-ui-1",
        "estimate": {
            "summary": "La cobertura del contexto es buena para una pasarela de pagos.",
            "lines": [
                {
                    "component": "Payment gateway core",
                    "hours": 220.0,
                    "rationale": "Derivado de un presupuesto histórico similar.",
                    "grounded": True,
                    "sources": [
                        {
                            "chunk_id": cited_chunk,
                            "document_id": "BUD-2024-003",
                            "evidence": "Estimated hours: 220",
                        }
                    ],
                },
                {
                    "component": "Reporting",
                    "hours": 0.0,
                    "rationale": "Insufficient data: ningún fragmento cubre reportes.",
                    "grounded": False,
                    "sources": [],
                },
            ],
        },
        "citation_report": {
            "verdicts": [
                {
                    "component": "Payment gateway core",
                    "status": gateway_status,
                    "dangling_chunk_ids": [cited_chunk] if dangling else [],
                },
                {"component": "Reporting", "status": "insufficient_data", "dangling_chunk_ids": []},
            ]
        },
        "claimed_hours": 220.0,
        "verified_hours": 0.0 if dangling else 220.0,
        "corrected_after_dangling": corrected,
        "retrieved_chunks": (
            [
                {"chunk_id": "9", "document_id": "BUD-2024-003", "content": _CHUNK_9},
                {"chunk_id": "10", "document_id": "BUD-2024-003", "content": _CHUNK_10},
            ]
            if chunks
            else []
        ),
    }


class _GroundedBackend:
    """Backend simulado: delega todo lo demás al handler de siempre y
    registra cada llamada a /estimate/grounded."""

    def __init__(self, response: httpx.Response | Exception):
        self.response = response
        self.requests: list[dict] = []

    def _handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/estimate/grounded":
            return _fake_handler(request)
        import json

        self.requests.append(json.loads(request.content))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def factory(self, *args, **kwargs):
        return _RealClient(transport=httpx.MockTransport(self._handler))


def _run_grounded(backend: _GroundedBackend, *, query=GROUNDED_QUERY, configure=None):
    with patch("httpx.Client", side_effect=backend.factory):
        at = AppTest.from_file(APP_PATH).run()
        at.text_area(key="grounded_query").set_value(query)
        if configure:
            configure(at)
        _find_button(at, GROUNDED_BUTTON).click().run()
        return at


def _ok(**kwargs) -> _GroundedBackend:
    return _GroundedBackend(httpx.Response(200, json=_grounded_body(**kwargs)))


def test_grounded_tab_is_present_alongside_the_other_tabs():
    with patch("httpx.Client", side_effect=_mock_httpx_client):
        at = AppTest.from_file(APP_PATH).run()

    assert [tab.label for tab in at.tabs] == [
        "⚡ Estimación rápida",
        "💬 Conversación",
        "🔎 Estimación con fuentes",
        "🤖 Agente de estimación",
    ]


def test_grounded_tab_shows_summary_totals_and_one_row_per_component():
    at = _run_grounded(_ok())

    assert not at.exception and not at.error
    assert "La cobertura del contexto es buena" in "\n".join(m.value for m in at.markdown)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Horas verificadas"] == "220"
    assert metrics["✅ Fundamentada"] == "1"
    assert metrics["⚠️ Cita inválida"] == "0"
    assert metrics["➖ Sin datos suficientes"] == "1"
    rows = at.dataframe[0].value
    assert list(rows["Componente"]) == ["Payment gateway core", "Reporting"]
    assert list(rows["Horas"]) == [220.0, 0.0]
    assert "chunk 9 (BUD-2024-003)" in rows["Fuentes"][0]
    assert rows["Fuentes"][1] == "—"


def test_each_component_expander_shows_the_cited_evidence_next_to_the_real_chunk():
    at = _run_grounded(_ok())

    labels = [e.label for e in at.expander]
    assert "✅ Fundamentada · Payment gateway core · 220 h" in labels
    assert "➖ Sin datos suficientes · Reporting · 0 h" in labels
    detail = next(e for e in at.expander if e.label.startswith("✅"))
    text = "\n".join(m.value for m in detail.markdown)
    assert "Evidencia citada" in text and "Estimated hours: 220" in text
    assert _CHUNK_9 in [c.value for c in detail.code]


def test_retrieved_context_is_listed_with_every_chunk():
    at = _run_grounded(_ok())

    context = next(e for e in at.expander if e.label.startswith("📚 Contexto recuperado (2"))
    assert [c.value for c in context.code] == [_CHUNK_9, _CHUNK_10]


def test_request_id_is_shown_so_it_can_be_found_in_the_logs():
    at = _run_grounded(_ok())

    assert any("req-ui-1" in c.value for c in at.caption)


def test_a_dangling_citation_is_shown_as_an_error_and_marked_on_its_row():
    at = _run_grounded(_ok(dangling=True, corrected=True))

    assert any("no existen en el contexto recuperado" in e.value for e in at.error)
    assert any("no existe en el contexto recuperado" in e.value for e in at.error)
    assert any("solo 0 h tienen citas verificadas" in w.value for w in at.warning)
    assert any("se le pidió corregirla" in i.value for i in at.info)
    assert at.dataframe[0].value["Estado"][0] == "⚠️ Cita inválida"
    assert {m.label: m.value for m in at.metric}["⚠️ Cita inválida"] == "1"


def test_no_retrieved_chunks_shows_a_warning_instead_of_a_blank_result():
    body = _grounded_body(chunks=False)
    body["estimate"]["lines"] = []
    body["citation_report"]["verdicts"] = []
    body["claimed_hours"] = body["verified_hours"] = 0.0
    at = _run_grounded(_GroundedBackend(httpx.Response(200, json=body)))

    assert not at.exception
    assert any("No se recuperó ningún fragmento" in w.value for w in at.warning)


def test_default_request_is_hybrid_k5_without_rerank():
    backend = _ok()

    _run_grounded(backend)

    assert backend.requests == [
        {"query": GROUNDED_QUERY, "k": 5, "search_mode": "hybrid", "rerank": False}
    ]


def test_the_chosen_search_options_are_sent_to_the_backend():
    backend = _ok()

    def configure(at):
        at.selectbox(key="grounded_search_mode").select("vector")
        at.number_input(key="grounded_k").set_value(3)
        at.checkbox(key="grounded_rerank").check()

    _run_grounded(backend, configure=configure)

    assert backend.requests[0]["search_mode"] == "vector"
    assert backend.requests[0]["k"] == 3
    assert backend.requests[0]["rerank"] is True


def test_a_short_description_is_rejected_without_calling_the_backend():
    backend = _ok()

    at = _run_grounded(backend, query="Muy corto")

    assert backend.requests == []
    assert any("al menos 20 caracteres" in e.value for e in at.error)


def test_backend_errors_are_shown_with_their_detail():
    backend = _GroundedBackend(
        httpx.Response(502, json={"detail": "Could not generate a valid estimate."})
    )

    at = _run_grounded(backend)

    assert not at.exception
    assert any(
        "Error 502" in e.value and "Could not generate a valid estimate." in e.value
        for e in at.error
    )


def test_connection_failures_are_shown_as_a_readable_error():
    backend = _GroundedBackend(httpx.ConnectError("connection refused"))

    at = _run_grounded(backend)

    assert not at.exception
    assert any("No se pudo conectar con el servicio" in e.value for e in at.error)


def test_the_result_stays_visible_after_the_page_reruns():
    backend = _ok()
    with patch("httpx.Client", side_effect=backend.factory):
        at = AppTest.from_file(APP_PATH).run()
        at.text_area(key="grounded_query").set_value(GROUNDED_QUERY)
        _find_button(at, GROUNDED_BUTTON).click().run()
        at.run()

    assert len(backend.requests) == 1
    assert {m.label: m.value for m in at.metric}["Horas verificadas"] == "220"


# --- Estimation agent tab ---------------------------------------------------

AGENT_BUTTON = "Ejecutar el agente"
AGENT_TRANSCRIPT = "Reunión de descubrimiento. " + "Necesitamos una plataforma con varias piezas. " * 4


def _agent_body(*, stopped="completed", consistent=True) -> dict:
    components = [
        {
            "name": "Order management API",
            "hours": 195.5,
            "reference_budget_ids": ["BUD-2024-006"],
            "rationale": "Comparable order lifecycle scope.",
            "has_references": True,
        },
        {
            "name": "SAP integration",
            "hours": 0.0,
            "reference_budget_ids": [],
            "rationale": "No comparable reference in the archive.",
            "has_references": False,
        },
    ]
    steps = [
        {
            "index": 1,
            "reasoning": "Two components, search both.",
            "action": 'search_budgets({"query": "order API", "filters": null})',
            "observation": "5 items for 'order API'",
        },
        {
            "index": 2,
            "reasoning": "Cost them.",
            "action": "calculate_estimate({})",
            "observation": "total=195.5h across 2 components",
        },
    ]
    return {
        "result": {
            "estimate": {
                "components": components,
                "assumptions": ["No comparable budget found for SAP integration."],
                "summary": "Partial coverage of the archive.",
            },
            "total_hours": 195.5,
            "consistent_with_calculation": consistent,
            "trace": {"steps": steps},
            "iterations": 5,
            "stopped_reason": stopped,
            "search_calls": 9,
            "model": "gpt-5",
            "input_tokens": 46_000,
            "output_tokens": 6_800,
        },
        "rendered_trace": "STEP 1\n  reasoning:   Two components, search both.",
    }


class _AgentBackend(_GroundedBackend):
    def _handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/estimate/agent":
            return _fake_handler(request)
        import json

        self.requests.append(json.loads(request.content))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _agent_ok(**kwargs) -> _AgentBackend:
    return _AgentBackend(httpx.Response(200, json=_agent_body(**kwargs)))


def _run_agent(backend: _AgentBackend, *, transcript=AGENT_TRANSCRIPT, configure=None):
    with patch("httpx.Client", side_effect=backend.factory):
        at = AppTest.from_file(APP_PATH).run()
        at.text_area(key="agent_transcript").set_value(transcript)
        if configure:
            configure(at)
        _find_button(at, AGENT_BUTTON).click().run()
        return at


def test_agent_tab_shows_totals_components_and_summary():
    at = _run_agent(_agent_ok())

    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Total estimado (h)"] == "195.5"
    assert metrics["Iteraciones"] == "5" and metrics["Búsquedas"] == "9"
    assert metrics["Componentes sin referencia"] == "1"
    assert any("Partial coverage of the archive." in m.value for m in at.markdown)
    rows = at.dataframe[-1].value
    assert list(rows["Componente"]) == ["Order management API", "SAP integration"]
    assert list(rows["Estado"]) == ["✅ Con referencia", "➖ Sin referencia"]


def test_agent_tab_flags_components_without_reference_and_lists_assumptions():
    at = _run_agent(_agent_ok())

    assert any("SAP integration" in i.value and "es un mínimo" in i.value for i in at.info)
    assert any("No comparable budget found for SAP integration." in m.value for m in at.markdown)


def test_agent_tab_shows_each_trace_step_with_reasoning_action_and_observation():
    at = _run_agent(_agent_ok())

    labels = [e.label for e in at.expander]
    assert "Paso 1 · search_budgets" in labels and "Paso 2 · calculate_estimate" in labels
    step_one = next(e for e in at.expander if e.label == "Paso 1 · search_budgets")
    assert any("Two components, search both." in m.value for m in step_one.markdown)
    assert any("search_budgets(" in c.value for c in step_one.code)
    assert any("5 items for 'order API'" in c.value for c in step_one.code)
    assert any(label == "📄 Traza completa en texto" for label in labels)


def test_agent_tab_warns_when_the_iteration_cap_was_reached():
    at = _run_agent(_agent_ok(stopped="max_iterations"))

    assert any("tope de iteraciones" in w.value for w in at.warning)


def test_agent_tab_warns_when_final_hours_differ_from_the_last_calculation():
    at = _run_agent(_agent_ok(consistent=False))

    assert any("no coinciden con el último cálculo" in w.value for w in at.warning)


def test_agent_default_request_uses_the_configured_model_and_effort():
    backend = _agent_ok()
    _run_agent(backend)

    assert backend.requests == [
        {"transcript": AGENT_TRANSCRIPT, "model": "gpt-5", "effort": "medium"}
    ]


def test_agent_chosen_model_and_effort_are_sent_to_the_backend():
    backend = _agent_ok()

    def configure(at):
        at.selectbox(key="agent_model").select("gpt-5-mini")
        at.selectbox(key="agent_effort").select("low")

    _run_agent(backend, configure=configure)

    assert backend.requests[0]["model"] == "gpt-5-mini" and backend.requests[0]["effort"] == "low"


def test_agent_short_transcript_is_rejected_without_calling_the_backend():
    backend = _agent_ok()
    at = _run_agent(backend, transcript="Muy corta")

    assert backend.requests == []
    assert any("al menos 50 caracteres" in e.value for e in at.error)


def test_agent_backend_errors_are_shown_with_their_detail():
    backend = _AgentBackend(
        httpx.Response(502, json={"detail": "The agent could not produce an estimate."})
    )
    at = _run_agent(backend)

    assert not at.exception
    assert any("Error 502" in e.value and "could not produce" in e.value for e in at.error)


def test_agent_connection_failures_are_shown_as_a_readable_error():
    at = _run_agent(_AgentBackend(httpx.ConnectError("connection refused")))

    assert not at.exception
    assert any("No se pudo conectar con el servicio" in e.value for e in at.error)


def test_agent_sample_selector_loads_the_example_transcript_into_the_text_box():
    with patch("httpx.Client", side_effect=_mock_httpx_client):
        at = AppTest.from_file(APP_PATH).run()
        at.selectbox(key="agent_sample").select("Ejemplo simple (1 componente)").run()

    assert not at.exception
    assert "Clínica Mirador" in at.text_area(key="agent_transcript").value


def test_agent_result_stays_visible_after_the_page_reruns():
    backend = _agent_ok()
    with patch("httpx.Client", side_effect=backend.factory):
        at = AppTest.from_file(APP_PATH).run()
        at.text_area(key="agent_transcript").set_value(AGENT_TRANSCRIPT)
        _find_button(at, AGENT_BUTTON).click().run()
        at.run()

    assert len(backend.requests) == 1
    assert {m.label: m.value for m in at.metric}["Total estimado (h)"] == "195.5"
