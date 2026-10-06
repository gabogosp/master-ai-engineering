import time
from dataclasses import asdict, dataclass
from typing import Generic, TypeVar

import instructor
import openai
import structlog
from litellm import Router
from pydantic import BaseModel

from app.config import settings
from app.services.cache import ExactMatchCache, build_cache_key

T = TypeVar("T", bound=BaseModel)

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


class EstimationValidationError(Exception):
    """Instructor agotó los reintentos (max_retries) sin que el modelo
    produjera una respuesta que pasara los model_validator de
    EstimationResult -- el modelo no logró corregir la aritmética o la
    regla de confianza baja a tiempo."""


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


@dataclass
class StructuredLLMResult(Generic[T]):
    result: T
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cached: bool = False


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


# El cliente de Instructor envuelve `router.acompletion` (no `litellm.completion`
# directo): así las llamadas con salida estructurada también se benefician del
# fallback openai<->anthropic del Router, en vez de perderlo -- verificado a
# mano que Instructor efectivamente invoca al Router y que el mensaje de error
# de un model_validator fallido vuelve a viajar en el reintento.
_instructor_client: instructor.AsyncInstructor | None = None


def _get_instructor_client() -> instructor.AsyncInstructor:
    global _instructor_client
    if _instructor_client is None:
        router = _get_router()
        _instructor_client = instructor.from_litellm(router.acompletion)
    return _instructor_client


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


async def _call_structured(
    messages: list[dict],
    cache_key: str,
    call_logger,
    response_model: type[T],
    max_retries: int,
) -> StructuredLLMResult[T]:
    """Lógica compartida de llamadas estructuradas (cache, Instructor sobre
    el Router, logging) entre un turno único (`generate_structured`) y uno
    multi-turno (`generate_structured_conversation_turn`)."""
    cached = await _cache.get(cache_key)
    if cached is not None:
        call_logger.info("llm_cache_hit", structured=True)
        return StructuredLLMResult(
            result=response_model.model_validate(cached["result"]),
            model=cached["model"],
            provider=cached["provider"],
            input_tokens=cached["input_tokens"],
            output_tokens=cached["output_tokens"],
            cached=True,
        )

    client = _get_instructor_client()
    call_logger.info("llm_call_started", cache_hit=False, structured=True)
    start = time.perf_counter()

    try:
        parsed, completion = await client.chat.completions.create_with_completion(
            model=PRIMARY_GROUP,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            response_model=response_model,
            max_retries=max_retries,
        )
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        # Instructor envuelve TODO en InstructorRetryException, incluidos los
        # errores reales de proveedor (auth, rate limit, conexión) -- pero
        # preserva el original en __cause__. Si la causa es una excepción de
        # proveedor conocida, la re-lanzamos tal cual para que el mapeo de
        # errores existente (CONFIGURATION_ERRORS/RATE_LIMIT_ERRORS/
        # PROVIDER_ERRORS) la siga reconociendo sin cambios. Si no, es que el
        # modelo agotó los reintentos sin pasar nuestros model_validator.
        cause = exc.__cause__
        if isinstance(cause, CONFIGURATION_ERRORS + RATE_LIMIT_ERRORS + PROVIDER_ERRORS):
            call_logger.error(
                "llm_call_failed",
                error_type=type(cause).__name__,
                latency_ms=latency_ms,
            )
            raise cause from exc

        call_logger.warning(
            "llm_structured_validation_exhausted",
            error_type=type(cause).__name__ if cause else type(exc).__name__,
            latency_ms=latency_ms,
        )
        raise EstimationValidationError(
            f"El modelo no logró una respuesta válida tras los reintentos: {exc}"
        ) from exc

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    actual_provider = completion._hidden_params.get("custom_llm_provider", settings.llm_provider)

    result = StructuredLLMResult(
        result=parsed,
        model=completion.model,
        provider=actual_provider,
        input_tokens=completion.usage.prompt_tokens,
        output_tokens=completion.usage.completion_tokens,
    )

    call_logger.info(
        "llm_call_completed",
        actual_provider=actual_provider,
        fallback_used=actual_provider != settings.llm_provider,
        tokens_in=result.input_tokens,
        tokens_out=result.output_tokens,
        latency_ms=latency_ms,
        cost_usd=completion._hidden_params.get("response_cost"),
        structured=True,
    )
    await _cache.set(
        cache_key,
        {
            "result": parsed.model_dump(mode="json"),
            "model": result.model,
            "provider": result.provider,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        },
    )
    return result


async def generate_structured(
    system: str,
    user: str,
    response_model: type[T],
    max_retries: int = 6,
) -> StructuredLLMResult[T]:
    """Turno único: `system`/`user` ya renderizados (ver
    app.prompts.loader.render_estimation_prompt)."""
    cache_key = build_cache_key(
        system=system,
        user=user,
        model=settings.llm_model,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
        response_model=response_model.__name__,
    )
    call_logger = logger.bind(
        requested_provider=settings.llm_provider,
        requested_model=settings.llm_model,
        response_model=response_model.__name__,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return await _call_structured(messages, cache_key, call_logger, response_model, max_retries)


async def generate_structured_conversation_turn(
    messages: list[dict],
    response_model: type[T],
    max_retries: int = 6,
) -> StructuredLLMResult[T]:
    """Multi-turno: `messages` ya armado por
    ConversationHistory.to_messages_list() + el turno nuevo."""
    cache_key = _cache_key_for_messages(messages) + f":{response_model.__name__}"
    call_logger = logger.bind(
        requested_provider=settings.llm_provider,
        requested_model=settings.llm_model,
        response_model=response_model.__name__,
        conversation_turns=(len(messages) - 1) // 2,
    )
    return await _call_structured(messages, cache_key, call_logger, response_model, max_retries)
