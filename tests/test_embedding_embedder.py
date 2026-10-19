from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIError, RateLimitError

import app.embedding_pipeline.embedder as embedder_module
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import Chunk, ChunkMetadata


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(status_code=429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


def _vec(value: float) -> list[float]:
    """1536 floats -- EmbeddedChunk.embedding exige esa longitud exacta
    (ver test_embedding_schemas.py), así que los fakes tienen que
    cumplirla aunque el contenido no importe para estos tests."""
    return [value] * 1536


def _fake_response(vectors: list[list[float]], total_tokens: int):
    data = [MagicMock(embedding=v) for v in vectors]
    usage = MagicMock(total_tokens=total_tokens)
    return MagicMock(data=data, usage=usage)


def _chunk(chunk_id: str, text: str = "algo") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            budget_id="BUD-A",
            component_id="C1",
            client_sector="finance",
            main_technology="ruby_on_rails",
            year=2024,
            complexity="high",
            estimated_hours=10,
        ),
        token_count=3,
    )


def test_constructor_raises_without_an_api_key():
    with patch.object(embedder_module.settings, "openai_api_key", None):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAIEmbedder()


def test_embed_one_returns_the_first_vector():
    embedder = OpenAIEmbedder()
    embedder._client = MagicMock()
    embedder._client.embeddings.create.return_value = _fake_response([[0.1, 0.2]], total_tokens=5)

    vector = embedder.embed_one("hola")
    assert vector == [0.1, 0.2]


def test_embed_many_preserves_chunk_order_and_attaches_embeddings():
    embedder = OpenAIEmbedder()
    embedder._client = MagicMock()
    embedder._client.embeddings.create.return_value = _fake_response(
        [_vec(0.1), _vec(0.2)], total_tokens=10
    )

    chunks = [_chunk("A::1"), _chunk("B::1")]
    embedded = embedder.embed_many(chunks)

    assert [c.chunk_id for c in embedded] == ["A::1", "B::1"]
    assert embedded[0].embedding == _vec(0.1)
    assert embedded[1].embedding == _vec(0.2)


def test_embed_many_splits_into_batches_of_batch_size():
    embedder = OpenAIEmbedder()
    embedder._client = MagicMock()
    embedder._client.embeddings.create.side_effect = [
        _fake_response([_vec(0.1), _vec(0.2)], total_tokens=4),
        _fake_response([_vec(0.3)], total_tokens=2),
    ]

    with patch.object(embedder_module, "BATCH_SIZE", 2):
        embedded = embedder.embed_many([_chunk("A"), _chunk("B"), _chunk("C")])

    assert embedder._client.embeddings.create.call_count == 2
    assert [c.embedding for c in embedded] == [_vec(0.1), _vec(0.2), _vec(0.3)]


def test_retries_on_rate_limit_and_eventually_succeeds():
    """1er intento sin espera, falla; antes del 2do espera 1s, falla;
    antes del 3ro espera 2s, tiene éxito -- confirma que el loop no
    sigue esperando una vez que un intento funciona."""
    embedder = OpenAIEmbedder()
    embedder._client = MagicMock()
    embedder._client.embeddings.create.side_effect = [
        _rate_limit_error(),
        _rate_limit_error(),
        _fake_response([[0.1]], total_tokens=1),
    ]

    with patch.object(embedder_module.time, "sleep") as fake_sleep:
        vector = embedder.embed_one("hola")

    assert vector == [0.1]
    assert fake_sleep.call_args_list == [((1,),), ((2,),)]


def test_propagates_rate_limit_error_after_exhausting_all_retries():
    embedder = OpenAIEmbedder()
    embedder._client = MagicMock()
    embedder._client.embeddings.create.side_effect = [
        _rate_limit_error(),
        _rate_limit_error(),
        _rate_limit_error(),
        _rate_limit_error(),
    ]

    with patch.object(embedder_module.time, "sleep") as fake_sleep:
        with pytest.raises(RateLimitError):
            embedder.embed_one("hola")

    assert fake_sleep.call_args_list == [((1,),), ((2,),), ((4,),)]


def test_non_rate_limit_errors_propagate_without_retrying():
    embedder = OpenAIEmbedder()
    embedder._client = MagicMock()
    boom = APIError(message="boom", request=httpx.Request("POST", "https://api.openai.com"), body=None)
    embedder._client.embeddings.create.side_effect = boom

    with patch.object(embedder_module.time, "sleep") as fake_sleep:
        with pytest.raises(APIError):
            embedder.embed_one("hola")

    fake_sleep.assert_not_called()
