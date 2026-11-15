"""Reciprocal Rank Fusion -- combina rankings independientes en uno solo.

La búsqueda vectorial devuelve distancia coseno (menor = mejor); la
búsqueda léxica devuelve un score de `ts_rank_cd` (mayor = mejor, y en
una escala totalmente distinta). Normalizar ambos a una escala común es
delicado y depende del corpus. RRF evita ese problema: ignora el score
de cada rama y solo usa la POSICIÓN de cada documento dentro de su
ranking. Un documento que aparece en varias ramas acumula
`1 / (k + posición)` por cada una -- aparecer bien ubicado en ambas ramas
pesa más que aparecer primero en una sola.
"""

from __future__ import annotations

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[int, float]]:
    """Fusiona varios rankings de ids en uno solo, mejor primero.

    `rankings` es una lista de rankings (uno por rama de búsqueda), cada
    uno ya ordenado de mejor a peor. Una rama que no devolvió nada aporta
    una lista vacía. `k` es la constante de suavizado: valores altos
    aplanan la curva (hace falta rankear bien en más de una rama para
    ganar), valores bajos dejan que un solo primer puesto domine.

    Devuelve pares (id, score_fusionado) ordenados por score descendente;
    empates se desempatan por id ascendente para que el resultado sea
    determinístico.
    """
    if k <= 0:
        raise ValueError("k debe ser positivo")

    scores: dict[int, float] = {}
    for ranking in rankings:
        seen_in_this_ranking: set[int] = set()
        for position, item_id in enumerate(ranking):
            # Si un id aparece repetido dentro del mismo ranking, solo
            # cuenta su primera (mejor) aparición.
            if item_id in seen_in_this_ranking:
                continue
            seen_in_this_ranking.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position)

    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
