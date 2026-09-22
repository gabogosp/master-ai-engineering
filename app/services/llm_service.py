from dataclasses import dataclass

import anthropic
import openai

from app.config import settings
from app.context.examples import ESTIMATION_EXAMPLES

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


class IncompleteEstimationError(Exception):
    """El proveedor cortó la respuesta o no la completó."""


class LLMConfigurationError(Exception):
    """Falta configuración del servicio, por ejemplo la API key del proveedor."""


# Grupos de errores para que el router los traduzca a respuestas HTTP sin conocer los SDK.
# Problema de configuración del servidor: clave o modelo inválidos.
CONFIGURATION_ERRORS = (
    LLMConfigurationError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.BadRequestError,
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.BadRequestError,
)
# El proveedor limita las peticiones o la cuenta no tiene saldo.
RATE_LIMIT_ERRORS = (openai.RateLimitError, anthropic.RateLimitError)
# Cualquier otro fallo del proveedor: conexión, timeout, error 5xx...
PROVIDER_ERRORS = (openai.APIError, anthropic.APIError)


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


async def _call_openai(transcription: str) -> LLMResult:
    if not settings.openai_api_key:
        raise LLMConfigurationError("OPENAI_API_KEY no está definida en el .env")

    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.responses.create(
        model=settings.llm_model,
        instructions=build_system_prompt(),
        input=transcription,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        store=False,
    )
    if response.status != "completed":
        raise IncompleteEstimationError(
            f"OpenAI devolvió status '{response.status}': {response.incomplete_details}"
        )
    return LLMResult(
        estimation=response.output_text,
        model=response.model,
        provider="openai",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


async def _call_anthropic(transcription: str) -> LLMResult:
    if not settings.anthropic_api_key:
        raise LLMConfigurationError("ANTHROPIC_API_KEY no está definida en el .env")

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    response = await client.messages.create(
        model=settings.llm_model,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": transcription}],
        # El SDK 1.x de Anthropic ya no acepta `temperature` como argumento. Haiku 4.5 la
        # admite en el cuerpo de la petición, pero los modelos más nuevos (Opus 4.7 en
        # adelante) la rechazan: si cambias LLM_MODEL a uno de ellos, quita esta línea.
        extra_body={"temperature": TEMPERATURE},
    )
    if response.stop_reason != "end_turn":
        raise IncompleteEstimationError(
            f"Anthropic terminó con stop_reason '{response.stop_reason}'"
        )
    text = "".join(block.text for block in response.content if block.type == "text")
    return LLMResult(
        estimation=text,
        model=response.model,
        provider="anthropic",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


async def generate_estimation(transcription: str) -> LLMResult:
    if settings.llm_provider == "anthropic":
        return await _call_anthropic(transcription)
    return await _call_openai(transcription)
