import time
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass

import openai
import structlog
from litellm import Router

from app.config import settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.services.cache import ExactMatchCache, build_cache_key

MAX_OUTPUT_TOKENS = 4096
TEMPERATURE = 0.3

SYSTEM_INSTRUCTIONS = """You are an expert software estimator. Your job is to produce project \
estimates from the transcript of a meeting with a client.

Rules:
- Use the previous estimation examples as a reference for format, level of detail and hourly rate \
(50 EUR/h).
- Follow exactly the same structure as the examples: title, task breakdown with hours, total, \
recommended team, duration, cost, and assumptions and risks.
- Base the estimate only on what the transcript says. If important information is missing, state \
it in the assumptions and risks section instead of inventing it.
- Compare the upper end of your estimated duration with the deadline the client mentioned. If it is \
longer than the deadline, add a bullet to the assumptions and risks section that starts with \
"Riesgo de plazo:" and states the client's deadline, your estimated duration, and what scope could \
be cut or which extra people would be needed to meet it. If the estimate fits within the deadline, \
do not add that bullet. If the client gave no deadline, do not invent one.
- Always write the estimation in Spanish, using the same section headings as the examples. \
Return only the estimation, with no additional text."""

# Nombres lógicos de los dos grupos del Router. El código de negocio solo
# conoce PRIMARY_GROUP; el Router decide qué modelo físico atiende cada uno.
PRIMARY_GROUP = "estimator-primary"
FALLBACK_GROUP = "estimator-fallback"

# Modelo de respaldo por proveedor. Si LLM_PROVIDER=openai fallamos hacia
# Anthropic y viceversa. Solo se activa si existe la API key de ese proveedor.
FALLBACK_MODEL = {
    "openai": "claude-haiku-4-5-20251001",
    "anthropic": "gpt-4o-mini",
}

logger = structlog.get_logger(__name__)


class IncompleteEstimationError(Exception):
    """El proveedor cortó la respuesta o no la completó."""


class LLMConfigurationError(Exception):
    """Falta configuración del servicio, por ejemplo la API key del proveedor."""


# LiteLLM traduce los errores de cualquier proveedor (OpenAI, Anthropic, o
# cualquiera de los más de 100 que soporta) a subclases de las excepciones de
# `openai`. Por eso alcanza con capturar `openai.*`: cubre tanto un error real
# de OpenAI como uno de Anthropic que pasó por LiteLLM. Ya no hace falta
# conocer las excepciones nativas de `anthropic`.
CONFIGURATION_ERRORS = (
    LLMConfigurationError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.BadRequestError,
)
RATE_LIMIT_ERRORS = (openai.RateLimitError,)
# APIError es la base de toda la jerarquía de errores de openai (incluida
# APIConnectionError e InternalServerError), así que actúa como catch-all.
PROVIDER_ERRORS = (openai.APIError,)


@dataclass
class LLMResult:
    estimation: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int


def build_system_prompt() -> str:
    examples = "\n\n".join(
        f'<example number="{i}">\n'
        f"<meeting_summary>\n{ex['meeting_summary']}\n</meeting_summary>\n"
        f"<estimation>\n{ex['estimation']}\n</estimation>\n"
        f"</example>"
        for i, ex in enumerate(ESTIMATION_EXAMPLES, start=1)
    )
    return f"{SYSTEM_INSTRUCTIONS}\n\n## Previous estimation examples\n\n{examples}"


def _api_key_for(provider: str) -> str | None:
    return settings.openai_api_key if provider == "openai" else settings.anthropic_api_key


def _other_provider(provider: str) -> str:
    return "anthropic" if provider == "openai" else "openai"


def _build_router() -> Router:
    primary_provider = settings.llm_provider
    primary_key = _api_key_for(primary_provider)
    if not primary_key:
        raise LLMConfigurationError(
            f"LLM_PROVIDER={primary_provider} pero falta su API key en el .env"
        )

    model_list = [
        {
            "model_name": PRIMARY_GROUP,
            "litellm_params": {"model": settings.llm_model, "api_key": primary_key},
        }
    ]

    fallback_provider = _other_provider(primary_provider)
    fallback_key = _api_key_for(fallback_provider)
    fallbacks = []
    if fallback_key:
        model_list.append(
            {
                "model_name": FALLBACK_GROUP,
                "litellm_params": {
                    "model": FALLBACK_MODEL[primary_provider],
                    "api_key": fallback_key,
                },
            }
        )
        fallbacks = [{PRIMARY_GROUP: [FALLBACK_GROUP]}]

    return Router(model_list=model_list, fallbacks=fallbacks, num_retries=1)


# Se construye una sola vez y se reutiliza. Si falta la API key, la excepción
# se lanza recién en el primer request (no al importar el módulo), igual que
# se comportaba el código anterior.
_router: Router | None = None
_cache = ExactMatchCache(settings.redis_url, settings.cache_ttl_seconds)


def _get_router() -> Router:
    global _router
    if _router is None:
        _router = _build_router()
    return _router


def _cache_key_for(system_prompt: str, transcription: str) -> str:
    return build_cache_key(
        system=system_prompt,
        user=transcription,
        model=settings.llm_model,
        temperature=TEMPERATURE,
    )


async def generate_estimation(transcription: str) -> LLMResult:
    system_prompt = build_system_prompt()
    cache_key = _cache_key_for(system_prompt, transcription)

    call_logger = logger.bind(
        requested_provider=settings.llm_provider, requested_model=settings.llm_model
    )

    cached = await _cache.get(cache_key)
    if cached is not None:
        call_logger.info("llm_cache_hit")
        return LLMResult(**cached)

    router = _get_router()
    call_logger.info("llm_call_started", cache_hit=False)
    start = time.perf_counter()

    try:
        response = await router.acompletion(
            model=PRIMARY_GROUP,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcription},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    except Exception as exc:
        call_logger.error(
            "llm_call_failed",
            error_type=type(exc).__name__,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        raise

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    choice = response.choices[0]
    actual_provider = response._hidden_params.get("custom_llm_provider", settings.llm_provider)

    if choice.finish_reason != "stop":
        call_logger.warning(
            "llm_call_incomplete",
            finish_reason=choice.finish_reason,
            actual_provider=actual_provider,
            latency_ms=latency_ms,
        )
        raise IncompleteEstimationError(
            f"El modelo terminó con finish_reason='{choice.finish_reason}'"
        )

    result = LLMResult(
        estimation=choice.message.content,
        model=response.model,
        provider=actual_provider,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )

    call_logger.info(
        "llm_call_completed",
        actual_provider=actual_provider,
        fallback_used=actual_provider != settings.llm_provider,
        tokens_in=result.input_tokens,
        tokens_out=result.output_tokens,
        latency_ms=latency_ms,
        cost_usd=response._hidden_params.get("response_cost"),
    )
    await _cache.set(cache_key, asdict(result))
    return result


async def stream_estimation(
    transcription: str,
    on_complete: Callable[[LLMResult], None] | None = None,
) -> AsyncIterator[str]:
    """Yieldea la estimación token a token a través del wrapper (mismo cache
    y fallback que `generate_estimation`).

    Esta función no sabe nada de Streamlit ni de SSE: solo produce texto. Si
    el llamador necesita los metadatos finales (modelo, proveedor, tokens),
    pasa `on_complete`, que se invoca una única vez al terminar el stream con
    el `LLMResult` completo. Así Streamlit puede llenar su `session_state` y
    un futuro endpoint SSE puede armar su evento `meta`, sin que este módulo
    conozca a ninguno de los dos.
    """
    system_prompt = build_system_prompt()
    cache_key = _cache_key_for(system_prompt, transcription)
    call_logger = logger.bind(
        requested_provider=settings.llm_provider, requested_model=settings.llm_model
    )

    cached = await _cache.get(cache_key)
    if cached is not None:
        call_logger.info("llm_cache_hit", streaming=True)
        result = LLMResult(**cached)
        yield result.estimation
        if on_complete:
            on_complete(result)
        return

    router = _get_router()
    call_logger.info("llm_call_started", cache_hit=False, streaming=True)
    start = time.perf_counter()

    accumulated: list[str] = []
    finish_reason: str | None = None
    usage = None
    model_name = settings.llm_model
    actual_provider = settings.llm_provider

    try:
        stream = await router.acompletion(
            model=PRIMARY_GROUP,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcription},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            actual_provider = chunk._hidden_params.get("custom_llm_provider", actual_provider)
            model_name = chunk.model or model_name
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    accumulated.append(delta)
                    yield delta
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
            if getattr(chunk, "usage", None):
                usage = chunk.usage
    except Exception as exc:
        call_logger.error(
            "llm_call_failed",
            error_type=type(exc).__name__,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        raise

    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    if finish_reason != "stop":
        call_logger.warning(
            "llm_call_incomplete",
            finish_reason=finish_reason,
            actual_provider=actual_provider,
            latency_ms=latency_ms,
        )
        raise IncompleteEstimationError(f"El modelo terminó con finish_reason='{finish_reason}'")

    result = LLMResult(
        estimation="".join(accumulated),
        model=model_name,
        provider=actual_provider,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )

    call_logger.info(
        "llm_call_completed",
        actual_provider=actual_provider,
        fallback_used=actual_provider != settings.llm_provider,
        tokens_in=result.input_tokens,
        tokens_out=result.output_tokens,
        latency_ms=latency_ms,
        streaming=True,
    )
    await _cache.set(cache_key, asdict(result))
    if on_complete:
        on_complete(result)
