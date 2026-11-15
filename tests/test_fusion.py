import pytest

from app.embedding_pipeline.fusion import reciprocal_rank_fusion


def test_fuses_two_rankings_by_position_not_score():
    vector_ranking = [10, 20, 30]
    lexical_ranking = [20, 10, 40]

    fused = reciprocal_rank_fusion([vector_ranking, lexical_ranking], k=60)
    fused_ids = [item_id for item_id, _score in fused]

    # 10 y 20 aparecen en ambas ramas -- deben rankear por encima de 30 y
    # 40, que solo aparecen en una.
    assert set(fused_ids[:2]) == {10, 20}
    assert set(fused_ids[2:]) == {30, 40}


def test_item_only_in_one_ranking_still_scores():
    fused = dict(reciprocal_rank_fusion([[1, 2], []], k=60))
    assert fused[1] == pytest.approx(1 / 60)
    assert fused[2] == pytest.approx(1 / 61)


def test_duplicate_within_a_ranking_counts_only_its_first_occurrence():
    fused = dict(reciprocal_rank_fusion([[1, 1, 2]], k=60))
    assert fused[1] == pytest.approx(1 / 60)
    # La posición de "2" sigue siendo su índice real (2) en la lista de
    # entrada -- el duplicado se ignora pero no corre las posiciones.
    assert fused[2] == pytest.approx(1 / 62)


def test_ties_break_by_ascending_id():
    fused = reciprocal_rank_fusion([[5], [9]], k=60)
    assert [item_id for item_id, _score in fused] == [5, 9]


def test_rejects_non_positive_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1, 2]], k=0)
