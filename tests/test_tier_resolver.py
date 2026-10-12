from app.sessions import ProjectMetadata
from app.tier_resolver import Tier, resolve_tier


def _metadata(**overrides) -> ProjectMetadata:
    return ProjectMetadata(**overrides)


def test_default_tier_when_nothing_matches():
    tier, rule = resolve_tier(transcript="Queremos un sitio web simple.", metadata=_metadata())
    assert tier == Tier.DEFAULT
    assert rule == "default"


def test_nda_in_transcript_resolves_executive():
    tier, rule = resolve_tier(
        transcript="Firmamos un NDA antes de compartir el alcance.", metadata=_metadata()
    )
    assert tier == Tier.EXECUTIVE
    assert rule == "nda_detected"


def test_nda_in_agreed_scope_also_resolves_executive():
    tier, rule = resolve_tier(
        transcript="Seguimos con el proyecto.",
        metadata=_metadata(agreed_scope="Cubierto por acuerdo de confidencialidad"),
    )
    assert tier == Tier.EXECUTIVE
    assert rule == "nda_detected"


def test_regulatory_keyword_resolves_executive():
    tier, rule = resolve_tier(
        transcript="El sistema tiene que cumplir con HIPAA.", metadata=_metadata()
    )
    assert tier == Tier.EXECUTIVE
    assert rule == "regulatory_context"


def test_nda_takes_precedence_over_regulatory():
    """nda_detected es la primera regla de la cadena -- si ambas matchean,
    gana la que tiene precedencia, no la más específica."""
    tier, rule = resolve_tier(
        transcript="NDA firmado. El sistema debe cumplir GDPR.", metadata=_metadata()
    )
    assert tier == Tier.EXECUTIVE
    assert rule == "nda_detected"


def test_two_distinct_dev_keywords_resolve_developer():
    tier, rule = resolve_tier(
        transcript="Vamos a correrlo en Kubernetes con un microservicio en gRPC.",
        metadata=_metadata(),
    )
    assert tier == Tier.DEVELOPER
    assert rule == "technical_audience"


def test_single_dev_keyword_does_not_resolve_developer():
    """Una sola mención técnica es ruido, no señal -- hacen falta >= 2
    keywords distintas."""
    tier, rule = resolve_tier(
        transcript="Lo corremos en Docker.", metadata=_metadata()
    )
    assert tier == Tier.DEFAULT
    assert rule == "default"


def test_small_team_resolves_pm():
    tier, rule = resolve_tier(
        transcript="Somos un equipo chico.", metadata=_metadata(assumed_team_size=2)
    )
    assert tier == Tier.PM
    assert rule == "low_budget_pm"


def test_team_size_above_threshold_does_not_resolve_pm():
    tier, rule = resolve_tier(
        transcript="Tenemos un equipo mediano.", metadata=_metadata(assumed_team_size=5)
    )
    assert tier == Tier.DEFAULT
    assert rule == "default"


def test_explicit_override_wins_over_every_rule():
    tier, rule = resolve_tier(
        transcript="Firmamos un NDA antes de empezar.",
        metadata=_metadata(),
        override=Tier.DEVELOPER,
    )
    assert tier == Tier.DEVELOPER
    assert rule == "explicit_override"
