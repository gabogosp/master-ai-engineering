"""Guardrails de input: tres capas que corren antes de tocar cache o LLM.

1. Moderación (OpenAI Moderation API): bloquea odio, violencia, sexual, etc.
   Necesita OPENAI_API_KEY -- si no está configurada (ej. proveedor primario
   es Anthropic sin key de OpenAI), esta capa se salta sola, no rompe nada.
2. Heurística de prompt-injection (regex sobre patrones conocidos): segunda
   línea de defensa barata contra "ignora las instrucciones anteriores".
3. Heurística de PII (regex sobre emails, teléfonos, IBAN): no es exhaustiva
   a propósito, es una demostración del *patrón*, no un redactor
   grado-compliance.

Política: las tres lanzan InputGuardrailViolation (excepción, nunca
corrigen-y-reintentan). El motivo vive en `reason` para que la capa HTTP
elija un código de estado y el cliente pueda mostrar un mensaje claro.
"""

import re
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)

Reason = Literal["moderation", "prompt_injection", "pii"]


class InputGuardrailViolation(Exception):
    """Lanzada por check_input cuando alguna de las capas rechaza el texto."""

    def __init__(self, message: str, *, reason: Reason) -> None:
        super().__init__(message)
        self.message = message
        self.reason = reason


_PROMPT_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"ignor[ae]\s+(las\s+)?(instrucciones|reglas|prompts?)\s+(anteriores|previas)",
        re.IGNORECASE,
    ),
    re.compile(
        r"ignore\s+(previous|prior|all|the)\s+(instructions?|prompts?|rules?)",
        re.IGNORECASE,
    ),
    re.compile(r"</?\s*(system|instructions?|prompt)\s*>", re.IGNORECASE),
    re.compile(r"(nuevas?\s+instrucciones?|new\s+instructions?)\s*[:.\-]", re.IGNORECASE),
    re.compile(r"(olvida|forget)\s+(todo|everything|all|previous|lo\s+anterior)", re.IGNORECASE),
    re.compile(r"\b(you\s+are\s+now|ahora\s+sos|ahora\s+eres)\b", re.IGNORECASE),
    re.compile(
        r"\b(disregard|ignora)\b.{0,40}\b(instructions?|prompts?|rules?|context|"
        r"instrucciones|reglas|contexto|anterior|previous|prior)",
        re.IGNORECASE | re.DOTALL,
    ),
]

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
# Conservador a propósito, para no marcar fechas o números de versión como
# si fueran teléfonos.
_PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\d[\s.-]?){9,12}\d")


async def check_input(text: str, *, openai_client=None) -> None:
    """Corre las tres capas en orden y lanza en la primera violación.

    `openai_client` es opcional: en tests se puede pasar None y solo se
    ejercitan las capas de regex. En producción, si hay OPENAI_API_KEY,
    se pasa un cliente real.
    """
    if openai_client is not None:
        await _check_moderation(text, openai_client)
    _check_prompt_injection(text)
    _check_pii(text)


async def _check_moderation(text: str, openai_client) -> None:
    try:
        response = await openai_client.moderations.create(input=text)
    except Exception as exc:  # fallos de red/auth de moderación no deben tumbar la petición
        logger.warning("moderation_call_failed", error_type=type(exc).__name__, error=str(exc))
        return

    result = response.results[0]
    if getattr(result, "flagged", False):
        categories = _extract_flagged_categories(result)
        logger.info("moderation_flagged", categories=categories)
        raise InputGuardrailViolation(
            f"Contenido marcado por moderación: {', '.join(categories) or 'sin especificar'}",
            reason="moderation",
        )


def _extract_flagged_categories(result) -> list[str]:
    categories = getattr(result, "categories", None)
    if categories is None:
        return []
    data = categories.model_dump() if hasattr(categories, "model_dump") else categories.__dict__
    return [name for name, flagged in data.items() if flagged]


def _check_prompt_injection(text: str) -> None:
    for pattern in _PROMPT_INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            logger.info(
                "prompt_injection_detected", pattern=pattern.pattern, match=match.group(0)[:80]
            )
            raise InputGuardrailViolation(
                f"Se detectó texto con forma de instrucción sospechosa: {match.group(0)[:80]!r}",
                reason="prompt_injection",
            )


def _check_pii(text: str) -> None:
    if _EMAIL_RE.search(text):
        raise InputGuardrailViolation(
            "Se detectó un email en el texto — eliminá los datos personales.",
            reason="pii",
        )
    if _IBAN_RE.search(text):
        raise InputGuardrailViolation(
            "Se detectó un IBAN en el texto — eliminá los datos personales.",
            reason="pii",
        )
    if _PHONE_RE.search(text):
        raise InputGuardrailViolation(
            "Se detectó un teléfono en el texto — eliminá los datos personales.",
            reason="pii",
        )
