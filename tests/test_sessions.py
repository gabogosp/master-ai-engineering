from app.sessions import ConversationHistory, ProjectMetadata


def _roles(messages: list[dict]) -> list[str]:
    return [m["role"] for m in messages]


def test_to_messages_list_without_summary_or_anchors():
    history = ConversationHistory(max_turns=6)
    history.add_turn({"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"})
    messages = history.to_messages_list("SYSTEM")
    assert _roles(messages) == ["system", "user", "assistant"]
    assert messages[0]["content"] == "SYSTEM"


def test_to_messages_list_merges_summary_into_first_user_message():
    """El resumen no debe viajar como un mensaje `user` aparte: eso
    generaría dos `user` consecutivos (el resumen + el primer turno de la
    ventana), rompiendo la alternancia estricta que exige, por ejemplo, la
    API de mensajes de Anthropic."""
    history = ConversationHistory(max_turns=6)
    history.add_turn({"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"})
    history.summary = "El cliente es una clínica veterinaria."

    messages = history.to_messages_list("SYSTEM")

    assert _roles(messages) == ["system", "user", "assistant"]
    # El resumen se fusiona dentro del primer mensaje user, antes de su
    # contenido original -- no lo reemplaza.
    assert "El cliente es una clínica veterinaria." in messages[1]["content"]
    assert messages[1]["content"].endswith("u1")
    # Nunca dos roles iguales consecutivos.
    roles = _roles(messages)
    assert all(roles[i] != roles[i - 1] for i in range(1, len(roles)))


def test_to_messages_list_summary_with_empty_body_stands_alone():
    """Si no hay ni anclas ni ventana (sesión recién creada con un resumen
    ya seteado a mano, caso de borde), el resumen es el único mensaje --
    no hay nada con qué fusionarlo."""
    history = ConversationHistory(max_turns=6)
    history.summary = "Resumen sin turnos activos."

    messages = history.to_messages_list("SYSTEM")

    assert _roles(messages) == ["system", "user"]
    assert messages[1]["content"].endswith("Resumen sin turnos activos.")


def test_to_messages_list_merges_summary_into_first_anchor_when_anchors_present():
    history = ConversationHistory(max_turns=6)
    history.anchors = [
        {"role": "user", "content": "NDA firmado."},
        {"role": "assistant", "content": "Entendido."},
    ]
    history.add_turn({"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"})
    history.summary = "Resumen previo."

    messages = history.to_messages_list("SYSTEM")

    assert _roles(messages) == ["system", "user", "assistant", "user", "assistant"]
    assert "Resumen previo." in messages[1]["content"]
    assert messages[1]["content"].endswith("NDA firmado.")


def test_len_counts_only_active_window_not_anchors_or_summary():
    history = ConversationHistory(max_turns=6)
    history.add_turn({"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"})
    history.anchors = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    history.summary = "algo"
    assert len(history) == 1


def test_project_metadata_merge_keeps_known_facts_when_update_is_empty():
    base = ProjectMetadata(project_name="CRM Ventas", assumed_team_size=3)
    update = ProjectMetadata()
    merged = base.merge(update)
    assert merged.project_name == "CRM Ventas"
    assert merged.assumed_team_size == 3
