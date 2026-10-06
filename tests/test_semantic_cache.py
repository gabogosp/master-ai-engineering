import uuid

import pytest
from redis.asyncio import Redis

from app.config import settings
from app.schemas import (
    DetailLevel,
    EstimationRequest,
    EstimationResult,
    OutputFormat,
    Phase,
    ProjectType,
)
from app.services.semantic_cache import EstimationSemanticCache


def _request(**overrides) -> EstimationRequest:
    defaults = dict(
        description="texto de prueba para la cache semántica",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )
    defaults.update(overrides)
    return EstimationRequest(**defaults)


def _result(summary: str = "Resumen de prueba con longitud suficiente.") -> EstimationResult:
    return EstimationResult(
        summary=summary,
        confidence_pct=70,
        phases=[
            Phase(
                name="Implementación",
                duration_weeks=3,
                cost_eur=9_000,
                summary="Desarrollo del sistema de reservas y notificaciones.",
            )
        ],
        total_duration_weeks=3,
        total_cost_eur=9_000,
    )


def test_bucket_for_is_deterministic_and_sensitive_to_every_field():
    """Dos requests idénticas dan el mismo bucket; cambiar prompt_version,
    project_type, detail_level u output_format cambia el bucket -- son
    justamente los cuatro componentes que hacen que dos descripciones
    parecidas NO deban compartir cache si el resto del pedido difiere."""
    base = _request()
    assert (
        EstimationSemanticCache.bucket_for(base, "v1") == "v1:web_saas:medium:phases_table"
    )
    assert EstimationSemanticCache.bucket_for(base, "v1") == EstimationSemanticCache.bucket_for(
        base, "v1"
    )

    base_bucket = EstimationSemanticCache.bucket_for(base, "v1")
    variants = [
        _request(project_type=ProjectType.MOBILE_APP),
        _request(detail_level=DetailLevel.DETAILED),
        _request(output_format=OutputFormat.NARRATIVE),
    ]
    for variant in variants:
        assert EstimationSemanticCache.bucket_for(variant, "v1") != base_bucket
    assert EstimationSemanticCache.bucket_for(base, "v2") != base_bucket


class _FakeVectorizer:
    """Evita llamar a la API real de embeddings en el test: cada texto
    tiene un vector fijo asignado a mano, para controlar la similitud
    coseno resultante sin depender de OpenAI."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    async def aembed(self, text: str) -> list[float]:
        return self._vectors[text]


def _unit_vector(dims: int, hot_index: int) -> list[float]:
    vec = [0.0] * dims
    vec[hot_index] = 1.0
    return vec


def _nearly_same_vector(base: list[float]) -> list[float]:
    # Una perturbación mínima -- coseno con el original > 0.999.
    perturbed = list(base)
    perturbed[1] = 0.01
    return perturbed


@pytest.fixture
async def semantic_cache():
    """Cache semántica real (Redis Stack corriendo localmente, igual que
    el resto de la suite asume para ExactMatchCache) pero con un
    vectorizer fake y un índice con nombre único por test -- así no pisa
    el índice "estimations" de producción ni colisiona entre tests."""
    redis_client: Redis = Redis.from_url(settings.redis_url, decode_responses=False)
    index_name = f"test_estimations_{uuid.uuid4().hex[:8]}"
    vectorizer = _FakeVectorizer({})
    cache = EstimationSemanticCache(
        redis_client=redis_client,
        vectorizer=vectorizer,
        threshold=0.9,
        ttl=60,
        log_only=False,
        index_name=index_name,
    )
    yield cache, vectorizer
    # Limpieza directa sobre el mismo cliente ya en uso: `cache._index`
    # abre su propia conexión perezosa en el primer uso real (search()/
    # create()), que puede quedar atada a un loop distinto al de este
    # fixture async al momento del teardown -- el mismo problema de
    # siempre con clientes async creados en un loop y usados en otro.
    try:
        await redis_client.execute_command("FT.DROPINDEX", index_name, "DD")
    except Exception:
        pass
    await redis_client.aclose()


async def test_lookup_hits_on_similar_text_in_same_bucket(semantic_cache):
    cache, vectorizer = semantic_cache
    original_text = "texto original para estimar"
    similar_text = "texto casi igual para estimar"

    base_vector = _unit_vector(1536, hot_index=0)
    vectorizer._vectors[original_text] = base_vector
    vectorizer._vectors[similar_text] = _nearly_same_vector(base_vector)

    request = _request(description=original_text)
    result = _result()
    await cache.store(
        request, "v1", result, model="gpt-4o-mini", provider="openai", input_tokens=10,
        output_tokens=5,
    )

    hit = await cache.lookup(_request(description=similar_text), "v1")
    assert hit is not None
    assert hit.result.summary == result.summary
    assert hit.model == "gpt-4o-mini"
    assert hit.provider == "openai"


async def test_lookup_misses_on_unrelated_text_same_bucket(semantic_cache):
    cache, vectorizer = semantic_cache
    stored_text = "texto original para estimar"
    unrelated_text = "texto completamente distinto y no relacionado"

    vectorizer._vectors[stored_text] = _unit_vector(1536, hot_index=0)
    vectorizer._vectors[unrelated_text] = _unit_vector(1536, hot_index=1500)

    await cache.store(
        _request(description=stored_text), "v1", _result(), model="gpt-4o-mini",
        provider="openai", input_tokens=10, output_tokens=5,
    )

    miss = await cache.lookup(_request(description=unrelated_text), "v1")
    assert miss is None


async def test_lookup_misses_across_different_buckets(semantic_cache):
    """Mismo texto (mismo embedding), pero el pedido pide otro
    output_format -- el bucket cambia y la cache NO debe devolver el hit
    aunque la similitud textual sea perfecta."""
    cache, vectorizer = semantic_cache
    text = "texto idéntico en ambos pedidos"
    vectorizer._vectors[text] = _unit_vector(1536, hot_index=0)

    await cache.store(
        _request(description=text, output_format=OutputFormat.PHASES_TABLE),
        "v1",
        _result(),
        model="gpt-4o-mini",
        provider="openai",
        input_tokens=10,
        output_tokens=5,
    )

    miss = await cache.lookup(
        _request(description=text, output_format=OutputFormat.NARRATIVE), "v1"
    )
    assert miss is None


async def test_log_only_mode_never_serves_a_hit(semantic_cache):
    """Con log_only=True, incluso un match perfecto no debe servirse --
    solo loguea el score, para calibrar el threshold antes de activarla."""
    cache, vectorizer = semantic_cache
    cache._log_only = True

    text = "texto idéntico para log only"
    vectorizer._vectors[text] = _unit_vector(1536, hot_index=0)

    await cache.store(
        _request(description=text), "v1", _result(), model="gpt-4o-mini", provider="openai",
        input_tokens=10, output_tokens=5,
    )

    hit = await cache.lookup(_request(description=text), "v1")
    assert hit is None
