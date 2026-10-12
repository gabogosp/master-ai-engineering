"""Deriva el "tier" de audiencia en runtime a partir del contexto de la
conversación.

El tier moldea el encuadre del prompt (bullets ejecutivos vs. hitos de PM vs.
detalle técnico vs. el tono por defecto). Dos formas de fijarlo:

- **Override explícito** por quien llama (ej. un parámetro `tier=executive`
  en la API). Siempre gana.
- **Derivación implícita** a partir del transcript + el project_metadata
  acumulado.

La derivación es una cadena de reglas puras, en orden de precedencia. Cada
regla es un predicado que devuelve True/False; la primera que matchea decide
el tier. Si ninguna matchea, el tier es DEFAULT. El resolver devuelve tanto
el tier como el *nombre de la regla* que disparó -- esa explicabilidad es lo
que permite mostrar "executive (nda_detected)" en vez de solo "executive".
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import structlog

from app.sessions import ProjectMetadata

logger = structlog.get_logger(__name__)


class Tier(str, Enum):
    EXECUTIVE = "executive"
    PM = "pm"
    DEVELOPER = "developer"
    DEFAULT = "default"


@dataclass
class ResolutionContext:
    transcript: str
    metadata: ProjectMetadata


@dataclass
class TierRule:
    name: str
    tier: Tier
    predicate: Callable[[ResolutionContext], bool]


_NDA_PATTERN = re.compile(
    r"\b(nda|acuerdo de confidencialidad|non[- ]?disclosure|confidencial|under embargo|legal hold)\b",
    re.IGNORECASE,
)
_REGULATORY_PATTERN = re.compile(
    r"\b(hipaa|gdpr|rgpd|sox|pci[- ]?dss|fda|iso[- ]?27001|ccpa)\b", re.IGNORECASE
)
_DEV_KEYWORDS_PATTERN = re.compile(
    r"\b(docker|kubernetes|k8s|microservicio|microservicios|microservice|microservices|"
    r"terraform|iac|helm|grpc|graphql|kafka|airflow|spark|rabbitmq)\b",
    re.IGNORECASE,
)


def _has_nda(ctx: ResolutionContext) -> bool:
    if _NDA_PATTERN.search(ctx.transcript):
        return True
    if ctx.metadata.agreed_scope and _NDA_PATTERN.search(ctx.metadata.agreed_scope):
        return True
    return False


def _has_regulatory_context(ctx: ResolutionContext) -> bool:
    if _REGULATORY_PATTERN.search(ctx.transcript):
        return True
    if ctx.metadata.agreed_scope and _REGULATORY_PATTERN.search(ctx.metadata.agreed_scope):
        return True
    for tech in ctx.metadata.mentioned_technologies:
        if _REGULATORY_PATTERN.search(tech):
            return True
    return False


def _is_small_team(ctx: ResolutionContext) -> bool:
    return ctx.metadata.assumed_team_size is not None and ctx.metadata.assumed_team_size <= 2


def _technical_audience(ctx: ResolutionContext) -> bool:
    hits = _DEV_KEYWORDS_PATTERN.findall(ctx.transcript)
    # Exigimos al menos dos keywords técnicas DISTINTAS para subir el tier --
    # una sola mención ("usamos docker") es demasiado ruidosa.
    return len({h.lower() for h in hits}) >= 2


_RULES: tuple[TierRule, ...] = (
    TierRule("nda_detected", Tier.EXECUTIVE, _has_nda),
    TierRule("regulatory_context", Tier.EXECUTIVE, _has_regulatory_context),
    TierRule("technical_audience", Tier.DEVELOPER, _technical_audience),
    TierRule("low_budget_pm", Tier.PM, _is_small_team),
)


def resolve_tier(
    *,
    transcript: str,
    metadata: ProjectMetadata,
    override: Tier | None = None,
) -> tuple[Tier, str]:
    """Devuelve (tier, rule_name).

    Precedencia: 1) override explícito gana siempre, 2) si no, se evalúa la
    cadena de reglas en orden -- la primera que matchea gana, 3) si ninguna
    matchea, DEFAULT."""
    if override is not None:
        logger.info("tier_resolved", tier=override.value, rule="explicit_override")
        return override, "explicit_override"

    ctx = ResolutionContext(transcript=transcript, metadata=metadata)
    for rule in _RULES:
        try:
            if rule.predicate(ctx):
                logger.info("tier_resolved", tier=rule.tier.value, rule=rule.name)
                return rule.tier, rule.name
        except Exception as exc:
            logger.warning(
                "tier_rule_predicate_failed",
                rule=rule.name,
                error_type=type(exc).__name__,
                error=str(exc)[:120],
            )

    logger.info("tier_resolved", tier=Tier.DEFAULT.value, rule="default")
    return Tier.DEFAULT, "default"
