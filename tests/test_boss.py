import pytest

from app.boss import Boss
from app.schemas import CriticFeedback, CriticIssue, EstimationResult, Phase


def _result(summary: str = "Resumen de prueba con longitud suficiente.", confidence: int = 80) -> EstimationResult:
    return EstimationResult(
        summary=summary,
        confidence_pct=confidence,
        phases=[
            Phase(
                name="Implementación",
                duration_weeks=3,
                cost_eur=9000,
                summary="Desarrollo del sistema de reservas y notificaciones.",
            )
        ],
        total_duration_weeks=3,
        total_cost_eur=9000,
    )


def _accept_feedback() -> CriticFeedback:
    return CriticFeedback(verdict="accept", issues=[], confidence_in_review=90)


def _needs_iteration_feedback() -> CriticFeedback:
    return CriticFeedback(
        verdict="needs_iteration",
        issues=[
            CriticIssue(
                category="math_error",
                severity="critical",
                field_path="total_cost_eur",
                description="La suma de fases no coincide con el total declarado.",
            )
        ],
        confidence_in_review=85,
    )


def _reject_feedback() -> CriticFeedback:
    return CriticFeedback(
        verdict="reject",
        issues=[
            CriticIssue(
                category="scope_mismatch",
                severity="critical",
                field_path="summary",
                description="La estimación no tiene nada que ver con lo pedido.",
            )
        ],
        confidence_in_review=95,
    )


async def test_boss_accepts_on_first_pass_when_critic_accepts():
    async def actor(feedback):
        assert feedback is None  # primera iteración, sin feedback previo
        return _result()

    async def critic(draft):
        return _accept_feedback()

    boss = Boss(max_iterations=3)
    result, trace = await boss.run(actor=actor, critic=critic)

    assert result.summary == _result().summary
    assert trace.final_decision == "accept"
    assert trace.iterations_run == 1
    assert trace.iterations[0].decision_after == "accept"


async def test_boss_iterates_then_accepts():
    calls = []

    async def actor(feedback):
        calls.append(feedback)
        if feedback is None:
            return _result(summary="Primer borrador, con un error de aritmética evidente.")
        return _result(summary="Segundo borrador, ya corregido según el feedback del critic.")

    verdicts = iter([_needs_iteration_feedback(), _accept_feedback()])

    async def critic(draft):
        return next(verdicts)

    boss = Boss(max_iterations=3)
    result, trace = await boss.run(actor=actor, critic=critic)

    assert len(calls) == 2
    assert calls[0] is None
    assert calls[1] is not None and calls[1].verdict == "needs_iteration"
    assert "corregido" in result.summary
    assert trace.final_decision == "accept"
    assert trace.iterations_run == 2
    assert [i.decision_after for i in trace.iterations] == ["iterate", "accept"]


async def test_boss_synthesizes_on_reject():
    async def actor(feedback):
        return _result(confidence=80)

    async def critic(draft):
        return _reject_feedback()

    boss = Boss(max_iterations=3)
    result, trace = await boss.run(actor=actor, critic=critic)

    assert trace.final_decision == "synthesize"
    assert trace.iterations_run == 1
    # Confianza reducida a la mitad, nunca por debajo del piso de 30.
    assert result.confidence_pct == 40
    assert "Salvedades de una revisión independiente" in result.summary
    assert "scope_mismatch" in result.summary


async def test_boss_synthesizes_when_iteration_budget_exhausted():
    async def actor(feedback):
        return _result(confidence=50)

    async def critic(draft):
        return _needs_iteration_feedback()  # nunca acepta

    boss = Boss(max_iterations=2)
    result, trace = await boss.run(actor=actor, critic=critic)

    assert trace.final_decision == "synthesize"
    assert trace.iterations_run == 2
    assert [i.decision_after for i in trace.iterations] == ["iterate", "synthesize"]
    assert result.confidence_pct == max(30, 50 // 2)


async def test_synthesized_confidence_never_drops_below_out_of_scope_threshold():
    """El piso de 30 es deliberado: bajar de eso dispararía el validator
    de EstimationResult que exige el prefijo 'Out of scope:', contradiciendo
    el propósito de devolver un draft anotado y usable."""

    async def actor(feedback):
        # model_construct bypassea el validator de EstimationResult (que
        # exige "Out of scope:" por debajo de 30) -- acá solo nos interesa
        # probar la aritmética del piso del Boss, no esa regla aparte.
        return EstimationResult.model_construct(**{**_result().model_dump(), "confidence_pct": 10})

    async def critic(draft):
        return _needs_iteration_feedback()

    boss = Boss(max_iterations=1)
    result, _trace = await boss.run(actor=actor, critic=critic)

    assert result.confidence_pct == 30


def test_boss_rejects_max_iterations_below_one():
    with pytest.raises(ValueError):
        Boss(max_iterations=0)
