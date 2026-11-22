import structlog
from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.loop import AgentError, run_estimation_agent
from app.agent.schemas import AgentEstimateRequest, AgentEstimateResponse
from app.agent.tools import AgentToolbox
from app.config import settings
from app.db import get_session
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.services.openai_client import get_openai_client

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["agent"])


@router.post("/estimate/agent", response_model=AgentEstimateResponse)
async def agent_estimate(
    request: AgentEstimateRequest, session: AsyncSession = Depends(get_session)
) -> AgentEstimateResponse:
    client = get_openai_client()
    if client is None:
        raise HTTPException(status_code=500, detail="The estimation agent is misconfigured.")
    try:
        toolbox = AgentToolbox(session, embed_one=OpenAIEmbedder().embed_one)
        result = await run_estimation_agent(
            client,
            toolbox,
            transcript=request.transcript,
            model=request.model or settings.agent_model,
            effort=request.effort or settings.agent_reasoning_effort,
            max_iterations=settings.agent_max_iterations,
        )
    except RuntimeError as exc:
        logger.error("agent_misconfigured", error=str(exc))
        raise HTTPException(status_code=500, detail="The estimation agent is misconfigured.") from exc
    except (OpenAIError, AgentError) as exc:
        logger.error("agent_failed", error_type=type(exc).__name__, exc_info=True)
        raise HTTPException(status_code=502, detail="The agent could not produce an estimate.") from exc

    return AgentEstimateResponse(result=result, rendered_trace=result.trace.render())
