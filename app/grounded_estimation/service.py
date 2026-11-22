import asyncio
import uuid
from dataclasses import dataclass
from typing import Literal

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.retrieval import RetrievedChunk, retrieve
from app.grounded_estimation.citations import verify_citations
from app.grounded_estimation.generator import generate_estimate
from app.grounded_estimation.schemas import (
    CitationReport,
    Estimate,
    LineStatus,
)
from app.services.openai_client import get_openai_client

logger = structlog.get_logger(__name__)


@dataclass
class GroundedEstimateResult:
    request_id: str
    estimate: Estimate
    citation_report: CitationReport
    chunks: list[RetrievedChunk]
    # Hours of lines whose citations were all verified. Lines with dangling
    # citations are excluded: their numbers cannot be trusted.
    verified_hours: float
    corrected_after_dangling: bool


def _correction_for(report: CitationReport, valid_ids: set[str]) -> str:
    dangling = sorted({cid for v in report.verdicts for cid in v.dangling_chunk_ids})
    return (
        f"Your previous answer cited chunk ids that do not exist in the context: {dangling}. "
        f"The only valid chunk ids are: {sorted(valid_ids)}. "
        "Answer again citing only valid ids; mark a line grounded=false if no valid chunk supports it."
    )


def _verified_hours(estimate: Estimate, report: CitationReport) -> float:
    return sum(
        line.hours
        for line, verdict in zip(estimate.lines, report.verdicts, strict=True)
        if verdict.status is LineStatus.GROUNDED
    )


async def build_grounded_estimate(
    session: AsyncSession,
    *,
    query: str,
    k: int = 5,
    search_mode: Literal["vector", "hybrid"] = "hybrid",
    rerank: bool = False,
    request_id: str | None = None,
) -> GroundedEstimateResult:
    request_id = request_id or uuid.uuid4().hex

    with structlog.contextvars.bound_contextvars(request_id=request_id):
        embedder = OpenAIEmbedder()
        query_vector = await asyncio.to_thread(embedder.embed_one, query)
        chunks, _metrics = await retrieve(
            session,
            query_text=query,
            query_vector=query_vector,
            search_mode=search_mode,
            rerank=rerank,
            k=k,
        )
        logger.info("grounded_estimate_retrieved", chunks=len(chunks), search_mode=search_mode)

        if not chunks:
            empty = Estimate(summary="Insufficient context: nothing was retrieved.", lines=[])
            return GroundedEstimateResult(
                request_id=request_id,
                estimate=empty,
                citation_report=verify_citations(empty, set()),
                chunks=[],
                verified_hours=0.0,
                corrected_after_dangling=False,
            )

        client = get_openai_client()
        if client is None:
            raise RuntimeError("OPENAI_API_KEY is required for grounded estimation")

        valid_ids = {str(chunk.id) for chunk in chunks}
        model = settings.grounded_generation_model

        estimate = await generate_estimate(client, query=query, chunks=chunks, model=model)
        report = verify_citations(estimate, valid_ids)

        corrected = False
        if report.has_dangling_citations:
            logger.warning("dangling_citations_retry", dangling=report.dangling_count)
            corrected = True
            estimate = await generate_estimate(
                client,
                query=query,
                chunks=chunks,
                model=model,
                correction=_correction_for(report, valid_ids),
            )
            report = verify_citations(estimate, valid_ids)

        return GroundedEstimateResult(
            request_id=request_id,
            estimate=estimate,
            citation_report=report,
            chunks=chunks,
            verified_hours=_verified_hours(estimate, report),
            corrected_after_dangling=corrected,
        )
