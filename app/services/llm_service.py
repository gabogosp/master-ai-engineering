import time
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass

import openai
import structlog
from litellm import Router

from app.config import settings
from app.services.cache import ExactMatchCache, build_cache_key

MAX_OUTPUT_TOKENS = 4096
TEMPERATURE = 0.3

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


def _cache_key_for(system: str, user: str) -> str:
    # Todo parámetro que pueda afectar la respuesta va acá, aunque hoy sea
    # una constante de módulo (MAX_OUTPUT_TOKENS, TEMPERATURE): si alguna vez
    # se vuelven configurables por request, dos respuestas con distinto
    # max_tokens no deben colisionar en la misma entrada de caché.
    return build_cache_key(
        system=system,
        user=user,
        model=settings.llm_model,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
    )


def _cache_key_for_messages(messages: list[dict]) -> str:
    return build_cache_key(
        messages=messages,
        model=settings.llm_model,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
    )


async def _call_llm(messages: list[dict], cache_key: str, call_logger) -> LLMResult:
    """Lógica compartida (cache, Router, logging) entre una llamada de un
    solo turno (`generate_estimation`) y una multi-turno
    (`generate_conversation_turn`): a esta altura ambas son lo mismo, un
    array `messages` y una clave de cache."""
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
            messages=messages,
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


async def generate_estimation(system: str, user: str) -> LLMResult:
    """Llama al modelo con `system` y `user` ya renderizados (ver
    app.prompts.loader.render_estimation_prompt). Este módulo no sabe nada
    de proyectos, CAG ni templates: solo dos strings y una llamada al LLM.
    """
    cache_key = _cache_key_for(system, user)
    call_logger = logger.bind(
        requested_provider=settings.llm_provider, requested_model=settings.llm_model
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return await _call_llm(messages, cache_key, call_logger)


async def generate_conversation_turn(messages: list[dict]) -> LLMResult:
    """Para el flujo multi-turno (POST /sessions/{id}/estimate): `messages`
    ya viene armado por ConversationHistory.to_messages_list() -- system +
    ventana deslizante de turnos previos + el turno nuevo.

    La clave de cache se arma sobre el array completo: dos conversaciones
    con exactamente el mismo historial cachean igual; cualquier turno
    nuevo (historial distinto al crecer) es, por definición, un cache
    miss -- coherente con que cada turno agrega información real.
    """
    cache_key = _cache_key_for_messages(messages)
    call_logger = logger.bind(
        requested_provider=settings.llm_provider,
        requested_model=settings.llm_model,
        conversation_turns=(len(messages) - 1) // 2,
    )
    return await _call_llm(messages, cache_key, call_logger)


async def _stream_llm(
    messages: list[dict],
    cache_key: str,
    call_logger,
    on_complete: Callable[[LLMResult], None] | None,
) -> AsyncIterator[str]:
    """Lógica compartida de streaming (cache, Router, logging) entre un
    stream de un solo turno (`stream_estimation`) y uno multi-turno
    (`stream_conversation_turn`) -- mismo patrón que `_call_llm` para las
    versiones bloqueantes."""
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
            messages=messages,
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


async def stream_estimation(
    system: str,
    user: str,
    on_complete: Callable[[LLMResult], None] | None = None,
) -> AsyncIterator[str]:
    """Yieldea la estimación token a token a través del wrapper (mismo cache
    y fallback que `generate_estimation`), recibiendo `system`/`user` ya
    renderizados.

    Esta función no sabe nada de Streamlit ni de SSE: solo produce texto. Si
    el llamador necesita los metadatos finales (modelo, proveedor, tokens),
    pasa `on_complete`, que se invoca una única vez al terminar el stream con
    el `LLMResult` completo. Así cada consumidor (Streamlit, un endpoint SSE)
    arma su propia UI sin que este módulo conozca a ninguno de los dos.
    """
    cache_key = _cache_key_for(system, user)
    call_logger = logger.bind(
        requested_provider=settings.llm_provider, requested_model=settings.llm_model
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    async for chunk in _stream_llm(messages, cache_key, call_logger, on_complete):
        yield chunk


async def stream_conversation_turn(
    messages: list[dict],
    on_complete: Callable[[LLMResult], None] | None = None,
) -> AsyncIterator[str]:
    """Versión streaming de `generate_conversation_turn`: `messages` ya
    viene armado por ConversationHistory.to_messages_list() + el turno
    nuevo. Mismo cache/fallback/logging que su contraparte bloqueante,
    vía `_stream_llm`."""
    cache_key = _cache_key_for_messages(messages)
    call_logger = logger.bind(
        requested_provider=settings.llm_provider,
        requested_model=settings.llm_model,
        conversation_turns=(len(messages) - 1) // 2,
    )
    async for chunk in _stream_llm(messages, cache_key, call_logger, on_complete):
        yield chunk
