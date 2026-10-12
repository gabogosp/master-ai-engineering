from app.compression.anchors import AnchorDetector
from app.compression.policy import CompressionPolicy
from app.sessions import ConversationHistory


async def test_heuristic_detects_nda_as_anchor():
    detector = AnchorDetector(mode="heuristic")
    match = await detector.detect({"role": "user", "content": "Firmamos un NDA con el cliente."})
    assert match.is_anchor
    assert "nda" in match.matched_rules


async def test_heuristic_detects_frozen_scope_as_anchor():
    detector = AnchorDetector(mode="heuristic")
    match = await detector.detect(
        {"role": "user", "content": "El alcance congelado para esta fase ya está confirmado."}
    )
    assert match.is_anchor
    assert "scope_frozen" in match.matched_rules


async def test_heuristic_does_not_flag_ordinary_turn():
    detector = AnchorDetector(mode="heuristic")
    match = await detector.detect(
        {"role": "user", "content": "¿Cuánto tardaría la fase de QA?"}
    )
    assert not match.is_anchor
    assert match.matched_rules == []


class _FakeAnchorDetector:
    """Detector controlable a mano: decide por índice de turno en vez de
    por contenido, para poder probar la política de compresión sin
    depender de que el texto matchee una heurística real."""

    def __init__(self, anchor_indices: set[int]) -> None:
        self._anchor_indices = anchor_indices
        self._calls = 0

    async def detect(self, message: dict):
        from app.compression.anchors import AnchorMatch

        idx = self._calls
        self._calls += 1
        if idx in self._anchor_indices:
            return AnchorMatch(is_anchor=True, matched_rules=["fake_rule"])
        return AnchorMatch(is_anchor=False)


class _FakeSummarizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, list[dict]]] = []

    async def summarize(self, *, previous_summary, evicted):
        self.calls.append((previous_summary, evicted))
        return f"resumen con {len(evicted)} mensajes evictados"


def _history_with_turns(n: int, max_turns: int = 2) -> ConversationHistory:
    history = ConversationHistory(max_turns=max_turns)
    for i in range(n):
        history.add_turn(
            {"role": "user", "content": f"turno {i}"},
            {"role": "assistant", "content": f"respuesta {i}"},
        )
    return history


async def test_policy_is_noop_under_the_cap():
    history = _history_with_turns(2, max_turns=2)  # 4 mensajes, cap=4 -> no excede
    policy = CompressionPolicy(
        anchor_detector=_FakeAnchorDetector(set()), summarizer=_FakeSummarizer()
    )
    await policy.apply(history)
    assert len(history.messages) == 4
    assert history.summary is None
    assert history.anchors == []


async def test_policy_evicts_oldest_pair_into_summary_when_not_anchor():
    history = _history_with_turns(3, max_turns=2)  # 6 mensajes, cap=4 -> evict 1 par
    summarizer = _FakeSummarizer()
    policy = CompressionPolicy(anchor_detector=_FakeAnchorDetector(set()), summarizer=summarizer)
    await policy.apply(history)

    assert len(history.messages) == 4  # cap respetado
    assert history.anchors == []
    assert history.summary == "resumen con 2 mensajes evictados"
    # El summarizer recibió el turno 0 (el más viejo), no otro.
    assert summarizer.calls[0][1][0]["content"] == "turno 0"


async def test_policy_promotes_anchor_pair_instead_of_summarizing():
    history = _history_with_turns(3, max_turns=2)
    # El primer turno evaluado (índice 0, el más viejo) es ancla.
    policy = CompressionPolicy(
        anchor_detector=_FakeAnchorDetector({0}), summarizer=_FakeSummarizer()
    )
    await policy.apply(history)

    assert len(history.messages) == 4
    assert len(history.anchors) == 2
    assert history.anchors[0]["content"] == "turno 0"
    assert history.summary is None  # nada no-ancla fue evictado


async def test_policy_is_idempotent_on_second_call_with_no_change():
    history = _history_with_turns(3, max_turns=2)
    policy = CompressionPolicy(anchor_detector=_FakeAnchorDetector(set()), summarizer=_FakeSummarizer())
    await policy.apply(history)
    state_after_first = (len(history.messages), history.summary, len(history.anchors))

    await policy.apply(history)  # ya está bajo el cap -- no debe volver a comprimir
    assert (len(history.messages), history.summary, len(history.anchors)) == state_after_first
