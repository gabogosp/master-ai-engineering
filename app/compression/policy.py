"""Cuándo comprimir, qué comprimir, y qué conservar palabra por palabra.

La política es la única pieza que muta `ConversationHistory` más allá de
`add_turn`. Corre después de cada turno, con este loop, en orden:

1. Mientras la ventana deslizante supere el tope (`max_turns * 2`
   mensajes), saca el par user/assistant más viejo del frente.
2. Pasa el lado de usuario del par al `AnchorDetector`. Si es un ancla,
   mueve AMBOS mensajes del par a `history.anchors` para que el compromiso
   sobreviva palabra por palabra. Si no, lo encola para el summarizer.
3. Terminado el loop, si se sacaron pares no-ancla, se los pasa al
   `CumulativeSummarizer` junto con el resumen previo, y se reemplaza
   `history.summary` con el texto nuevo.

La política es idempotente a propósito: una segunda llamada sin cambios en
`messages` es un no-op.
"""

import structlog

from app.compression.anchors import AnchorDetector
from app.compression.summarizer import CumulativeSummarizer
from app.sessions import ConversationHistory

logger = structlog.get_logger(__name__)


class CompressionPolicy:
    def __init__(self, *, anchor_detector: AnchorDetector, summarizer: CumulativeSummarizer) -> None:
        self.anchor_detector = anchor_detector
        self.summarizer = summarizer

    @staticmethod
    def should_compress(history: ConversationHistory) -> bool:
        return len(history.messages) > history.max_turns * 2

    async def apply(self, history: ConversationHistory) -> None:
        """Muta `history` in place: promueve anclas y absorbe el resto en
        el resumen corriente. No-op si la ventana está bajo el tope."""
        if not self.should_compress(history):
            return

        evicted_for_summary: list[dict] = []
        promoted_anchor_rules: list[list[str]] = []

        while len(history.messages) > history.max_turns * 2:
            if len(history.messages) < 2:
                break
            user_msg = history.messages[0]
            assistant_msg = history.messages[1]

            match = await self.anchor_detector.detect(user_msg)
            if match.is_anchor:
                history.anchors.append(user_msg)
                history.anchors.append(assistant_msg)
                promoted_anchor_rules.append(match.matched_rules)
            else:
                evicted_for_summary.append(user_msg)
                evicted_for_summary.append(assistant_msg)

            del history.messages[:2]

        if evicted_for_summary:
            history.summary = await self.summarizer.summarize(
                previous_summary=history.summary, evicted=evicted_for_summary
            )

        logger.info(
            "history_compressed",
            promoted_anchors=len(promoted_anchor_rules),
            anchor_rules=[r for rules in promoted_anchor_rules for r in rules],
            evicted_to_summary=len(evicted_for_summary),
            summary_chars=len(history.summary or ""),
            anchors_count=len(history.anchors),
            recent_messages=len(history.messages),
        )


async def apply_compression(
    history: ConversationHistory,
    *,
    anchor_detection_mode: str = "heuristic",
) -> None:
    """Wrapper de conveniencia: arma el detector + summarizer con el wiring
    por defecto y corre la política una vez. Angosto a propósito -- el
    router no necesita conocer la estructura interna del módulo."""
    detector = AnchorDetector(mode=anchor_detection_mode)
    summarizer = CumulativeSummarizer()
    policy = CompressionPolicy(anchor_detector=detector, summarizer=summarizer)
    await policy.apply(history)
