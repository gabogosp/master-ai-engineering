"""Wrapper de un cross-encoder para reordenar candidatos de retrieval.

Un bi-encoder (el embedder que ya usamos en `embedder.py`) codifica la
query y cada documento POR SEPARADO -- la "relevancia" es el coseno entre
dos vectores que nunca se vieron entre sí. Rápido, pero grueso. Un
cross-encoder recibe el par (query, documento) JUNTO en una sola pasada
del modelo y devuelve un score de relevancia que sí atiende a ambos
textos a la vez -- mucho más fino, mucho más lento para correr sobre todo
el corpus. De ahí el patrón recall-then-rerank: una recuperación amplia y
barata (vectorial, léxica o ambas) trae un pool de candidatos, y el
cross-encoder solo reordena ESE pool.

Este módulo es exclusivamente el wrapper (carga del modelo + scoring de
pares). La decisión de cuántos candidatos recuperar antes de rerankear y
cuántos quedarse al final vive en el endpoint/orquestador, no aquí.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import structlog

logger = structlog.get_logger(__name__)


class CrossEncoderReranker:
    """Cross-encoder cargado de forma perezosa que reordena candidatos.

    La carga está protegida con un lock: si dos requests concurrentes
    disparan el primer rerank a la vez, solo una descarga/carga el modelo
    y la otra espera en vez de duplicar el trabajo.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self) -> None:
        """Fuerza la carga del modelo ahora (usado por el script de
        verificación, para fallar rápido antes de levantar el servicio)."""
        self._ensure_loaded()

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            # Import diferido a propósito: mantiene el costo de importar
            # este módulo (tests, arranque de la app) libre de torch hasta
            # que alguien de verdad rerankea algo.
            from sentence_transformers import CrossEncoder

            started = time.perf_counter()
            self._model = CrossEncoder(self._model_name)
            logger.info(
                "reranker_model_loaded",
                model=self._model_name,
                load_time_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return self._model

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score de relevancia (mayor = más relevante) de `query` contra
        cada documento, en el mismo orden de entrada."""
        if not documents:
            return []
        model = self._ensure_loaded()
        pairs = [(query, document) for document in documents]
        started = time.perf_counter()
        scores = model.predict(pairs)
        logger.info(
            "reranker_scored_pairs",
            model=self._model_name,
            pairs=len(pairs),
            score_time_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return [float(score) for score in scores]

    def rerank(
        self,
        query: str,
        candidates: list,
        *,
        top_n: int,
        text_of: Callable = lambda candidate: candidate.content,
    ) -> list:
        """Reordena `candidates` por score del cross-encoder y devuelve
        los `top_n` mejores. `text_of` extrae el texto a scorear de cada
        candidato (default: atributo `.content`), para no atar este
        wrapper a una forma de dato concreta (fila de ORM, dict, etc.)."""
        if not candidates:
            return []
        scores = self.score(query, [text_of(candidate) for candidate in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [candidate for candidate, _score in ranked[:top_n]]
