"""Boss -- orquesta Actor y Critic.

El Boss es una máquina de estados chica:

    actor() -> critic_review() -> decide
                                   +-- accept -> devolver
                                   +-- needs_iteration AND quedan iteraciones
                                   |              -> re-llamar al actor con el feedback
                                   |              -> loop
                                   +-- (se acabaron las iteraciones o reject)
                                                  -> sintetizar fallback

El Boss NO hace llamadas a LLM por su cuenta -- solo decide qué hacer
después. Eso mantiene la lógica de decisión demostrable: cada elección
queda trazada en BossTrace.iterations y es reproducible a partir de los
inputs.

El actor se pasa como un callable en vez de un objeto. La firma es::

    actor(critic_feedback: CriticFeedback | None) -> Awaitable[EstimationResult]

así el Boss puede re-invocar al actor en la iteración N+1 con el feedback
de la iteración N. El wiring vive en el router (POST /estimate-acb).
"""

from collections.abc import Awaitable, Callable

import structlog

from app.schemas import ACBIteration, BossDecision, BossTrace, CriticFeedback, EstimationResult

logger = structlog.get_logger(__name__)

ActorCallable = Callable[[CriticFeedback | None], Awaitable[EstimationResult]]
CriticCallable = Callable[[EstimationResult], Awaitable[CriticFeedback]]


class Boss:
    """Orquestador stateless. Un `run` por estimación."""

    def __init__(self, *, max_iterations: int = 3) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations debe ser >= 1")
        self.max_iterations = max_iterations

    async def run(
        self, *, actor: ActorCallable, critic: CriticCallable
    ) -> tuple[EstimationResult, BossTrace]:
        trace = BossTrace(final_decision="accept", iterations_run=0)
        feedback: CriticFeedback | None = None
        current: EstimationResult | None = None

        for iteration in range(self.max_iterations):
            current = await actor(feedback)
            review = await critic(current)

            decision = self._decide(review, iterations_left=(self.max_iterations - iteration - 1))
            trace.iterations.append(
                ACBIteration(
                    iteration=iteration,
                    decision_after=decision,
                    critic_verdict=review.verdict,
                    critic_confidence=review.confidence_in_review,
                    issue_summary=[
                        f"[{i.severity}] {i.category} @ {i.field_path}" for i in review.issues[:5]
                    ],
                )
            )
            trace.iterations_run = iteration + 1
            logger.info(
                "boss_iteration",
                iteration=iteration,
                decision=decision,
                verdict=review.verdict,
                issues=len(review.issues),
            )

            if decision == "accept":
                trace.final_decision = "accept"
                return current, trace
            if decision == "synthesize":
                trace.final_decision = "synthesize"
                return self._synthesize_fallback(current, review), trace

            # decision == "iterate" -> loop con feedback
            feedback = review

        # Se agotó el loop: queda el último draft del actor con el critic
        # todavía marcando issues.
        trace.final_decision = "synthesize"
        logger.info("boss_iteration_budget_exhausted", iterations=self.max_iterations)
        assert current is not None  # garantizado porque max_iterations >= 1
        return self._synthesize_fallback(current, feedback), trace

    @staticmethod
    def _decide(review: CriticFeedback, *, iterations_left: int) -> BossDecision:
        if review.verdict == "accept":
            return "accept"
        if review.verdict == "reject":
            return "synthesize"
        # needs_iteration
        if iterations_left > 0:
            return "iterate"
        return "synthesize"

    @staticmethod
    def _synthesize_fallback(
        last_result: EstimationResult, last_feedback: CriticFeedback | None
    ) -> EstimationResult:
        """Cuando el Boss no puede aceptar, devuelve el último draft del
        actor anotado con los issues abiertos del Critic y confianza
        reducida -- nunca un sobre vacío.

        Razón: descartar el draft esconde la mejor respuesta disponible
        detrás de un placeholder ("Out of scope"). La señal honesta de que
        el loop no convergió ya viaja en `BossTrace.final_decision ==
        "synthesize"`; al usuario lo sirve mejor una estimación con
        salvedades que números en cero.
        """
        if last_feedback is None or not last_feedback.issues:
            note = "⚠ Salvedades de una revisión independiente: (sin detalle disponible)\n\n"
            new_summary = (note + last_result.summary)[:1200]
            return last_result.model_copy(update={"summary": new_summary})

        issue_lines = [
            f"- [{issue.severity}] {issue.category} ({issue.field_path}): {issue.description}"
            for issue in last_feedback.issues
        ]
        caveats_block = (
            "⚠ Salvedades de una revisión independiente (el loop no convergió del todo):\n"
            + "\n".join(issue_lines)
            + "\n\n"
        )
        new_summary = (caveats_block + last_result.summary)[:1200]
        # Piso en 30 para quedar por encima de LOW_CONFIDENCE_THRESHOLD --
        # bajar de eso dispararía el validator que exige el prefijo
        # "Out of scope:", lo que contradice el propósito de devolver un
        # draft anotado y usable.
        reduced_confidence = max(30, last_result.confidence_pct // 2)
        return last_result.model_copy(
            update={"summary": new_summary, "confidence_pct": reduced_confidence}
        )
