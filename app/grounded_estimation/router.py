import structlog
from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.grounded_estimation.generator import EstimateGenerationError
from app.grounded_estimation.schemas import (
    GroundedEstimateRequest,
    GroundedEstimateResponse,
    RetrievedChunkOut,
)
from app.grounded_estimation.service import build_grounded_estimate

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["grounded-estimation"])


@router.post("/estimate/grounded", response_model=GroundedEstimateResponse)
async def grounded_estimate(
    request: GroundedEstimateRequest, session: AsyncSession = Depends(get_session)
) -> GroundedEstimateResponse:
    try:
        result = await build_grounded_estimate(
            session,
            query=request.query,
            k=request.k,
            search_mode=request.search_mode,
            rerank=request.rerank,
        )
    except RuntimeError as exc:
        logger.error("grounded_estimation_misconfigured", error=str(exc))
        raise HTTPException(status_code=500, detail="Grounded estimation is misconfigured.") from exc
    except (OpenAIError, EstimateGenerationError) as exc:
        logger.error("grounded_estimation_failed", error_type=type(exc).__name__, exc_info=True)
        raise HTTPException(status_code=502, detail="Could not generate a valid estimate.") from exc

    return GroundedEstimateResponse(
        request_id=result.request_id,
        estimate=result.estimate,
        citation_report=result.citation_report,
        claimed_hours=result.estimate.grounded_hours,
        verified_hours=result.verified_hours,
        corrected_after_dangling=result.corrected_after_dangling,
        retrieved_chunks=[
            RetrievedChunkOut(
                chunk_id=str(chunk.id),
                document_id=chunk.metadata["budget_id"],
                content=chunk.content,
            )
            for chunk in result.chunks
        ],
    )
