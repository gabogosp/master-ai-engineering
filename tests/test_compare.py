import pytest

from scripts.compare import cosine_similarity


def test_cosine_similarity_of_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_of_opposite_vectors_is_minus_one():
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_cosine_similarity_boundary_scaled_vectors_stay_at_one():
    """Escalar un vector no debería cambiar su similitud con el
    original -- el coseno es invariante a la magnitud, solo le importa
    la dirección."""
    assert cosine_similarity([2.0, 0.0], [10.0, 0.0]) == pytest.approx(1.0)
