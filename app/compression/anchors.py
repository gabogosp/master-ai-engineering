"""Detección de anclas -- qué turnos deben sobrevivir a la ventana deslizante.

Un turno se vuelve ancla cuando carga un compromiso durable que la
conversación no puede permitirse olvidar: acuerdos firmados, alcance
congelado, contexto legal o regulatorio, presupuesto cerrado. El detector es
a propósito conservador -- un falso negativo solo degrada el turno a la
ventana normal (donde después lo absorbe el summarizer), mientras que un
falso positivo infla el prompt para siempre, porque las anclas nunca se
evictan.

Dos modos:

- "heuristic" (default): regex barata sobre una lista curada de frases. Sin
  llamada a LLM, determinístico.
- "llm": clasificador binario vía Instructor. Más robusto a paráfrasis, pero
  agrega una llamada barata por turno -- opt-in (ANCHOR_DETECTION_MODE=llm).
"""

import re
from dataclasses import dataclass, field
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from app.services.llm_service import auxiliary_model_group, generate_structured

logger = structlog.get_logger(__name__)


# Biblioteca curada de frases. Cada entrada matchea una frase verbal
# distintiva, un término legal/compliance, o un compromiso explícito. La
# lista es deliberadamente corta: preferimos perder un ancla real a
# sobre-anclar turnos irrelevantes.
_HEURISTIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "nda",
        re.compile(
            r"\b(nda|acuerdo de confidencialidad|non[- ]?disclosure|under embargo|legal hold)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "signed_contract",
        re.compile(
            r"\b(firm[oó]|firmamos)\s+(el\s+)?(contrato|acuerdo|sow|msa)\b"
            r"|\b(signed|countersigned)\s+(the\s+)?(contract|sow|msa|agreement)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "scope_frozen",
        re.compile(
            r"\balcance\s+(congelado|cerrado|fijo)\b|\bscope\s+(is\s+)?(frozen|locked|final|fixed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "budget_locked",
        re.compile(
            r"\bpresupuesto\s+(cerrado|aprobado|tope)\b"
            r"|\b(budget|cap|ceiling)\s+(is\s+)?(locked|fixed|approved|capped)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "compliance",
        re.compile(r"\b(hipaa|gdpr|rgpd|sox|pci[- ]?dss|fda|iso[- ]?27001)\b", re.IGNORECASE),
    ),
    (
        "deadline_hard",
        re.compile(
            r"\bdeadline\s+(duro|innegociable)\b|\bhard\s+deadline\b|\bmust\s+go\s+live\s+by\b",
            re.IGNORECASE,
        ),
    ),
    (
        "contractual",
        re.compile(
            r"\bcontractualmente\s+(obligad[oa]|comprometid[oa])\b"
            r"|\bcontractually\s+(bound|required|obliged)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "explicit_commitment",
        re.compile(
            r"\b(el cliente|nosotros)\s+(acordamos?|nos comprometimos?)\s+a\b"
            r"|\b(we|the (client|customer))\s+(agreed|committed)\s+to\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass
class AnchorMatch:
    """Resultado de un chequeo de ancla. Lleva los nombres de las reglas que
    dispararon, para que el log estructurado pueda atribuir la decisión."""

    is_anchor: bool
    matched_rules: list[str] = field(default_factory=list)


class _AnchorClassification(BaseModel):
    """Schema Pydantic del detector basado en LLM."""

    is_anchor: bool = Field(
        description=(
            "True si este turno de usuario introduce un compromiso durable, una "
            "restricción legal/compliance, un alcance congelado o un presupuesto "
            "cerrado que la conversación no debe olvidar entre turnos."
        )
    )
    reason: str = Field(default="", max_length=200)


class AnchorDetector:
    """Decide si un turno merece sobrevivir a la eviction.

    El detector inspecciona el turno del *usuario* -- la respuesta del
    asistente es función de ese turno, así que alcanza con marcar una vez
    por par."""

    def __init__(self, *, mode: Literal["heuristic", "llm"] = "heuristic") -> None:
        self.mode = mode

    async def detect(self, message: dict) -> AnchorMatch:
        if self.mode == "llm":
            return await self._detect_llm(message)
        return self._detect_heuristic(message)

    @staticmethod
    def _detect_heuristic(message: dict) -> AnchorMatch:
        content = message.get("content") or ""
        matched = [name for name, pattern in _HEURISTIC_PATTERNS if pattern.search(content)]
        if matched:
            logger.info(
                "anchor_detected", strategy="heuristic", rules=matched, content_chars=len(content)
            )
            return AnchorMatch(is_anchor=True, matched_rules=matched)
        return AnchorMatch(is_anchor=False)

    async def _detect_llm(self, message: dict) -> AnchorMatch:
        content = message.get("content") or ""
        system_prompt = (
            "Sos un clasificador. Decidí si el turno de usuario introduce un "
            "compromiso durable sobre el proyecto -- acuerdos firmados, alcance "
            "congelado, presupuesto cerrado, o contexto legal/compliance (NDA, "
            "HIPAA, GDPR, etc). Devolvé is_anchor=true solo si olvidar el turno "
            "perjudicaría la conversación. reason es una oración corta."
        )
        user_message = f"Turno de usuario:\n{content}"
        try:
            structured = await generate_structured(
                system_prompt,
                user_message,
                _AnchorClassification,
                max_retries=1,
                model_group=auxiliary_model_group(),
            )
        except Exception as exc:
            logger.warning(
                "anchor_llm_failed_fallback_heuristic",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            return self._detect_heuristic(message)

        classification = structured.result
        if classification.is_anchor:
            logger.info("anchor_detected", strategy="llm", reason=classification.reason[:120])
            return AnchorMatch(is_anchor=True, matched_rules=[f"llm:{classification.reason[:60]}"])
        return AnchorMatch(is_anchor=False)
