"""Output schema for line-level grounded estimates.

No `Field(min_length=...)`/`ge=...` constraints on purpose: the schema is sent
to the Responses API in strict mode, which supports only a subset of JSON
Schema keywords. Integrity rules live in validators instead, which run
client-side after parsing and never reach the wire.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

INSUFFICIENT_DATA_PREFIX = "Insufficient data:"


class SourceReference(BaseModel):
    chunk_id: str
    document_id: str
    evidence: str


class EstimateLineItem(BaseModel):
    component: str
    hours: float
    rationale: str
    grounded: bool
    sources: list[SourceReference]

    @model_validator(mode="after")
    def enforce_grounding_integrity(self) -> "EstimateLineItem":
        if self.grounded:
            if not self.sources:
                raise ValueError(
                    f"line {self.component!r} is grounded=True but cites no source; "
                    "cite at least one chunk or mark it grounded=False"
                )
            if self.hours <= 0:
                raise ValueError(
                    f"line {self.component!r} is grounded=True but has hours <= 0"
                )
            return self

        if self.sources:
            raise ValueError(
                f"line {self.component!r} is grounded=False but lists sources; "
                "an ungrounded line must have sources=[]"
            )
        if self.hours != 0:
            raise ValueError(
                f"line {self.component!r} is grounded=False but has hours={self.hours}; "
                "hours must be 0 when there is no supporting data"
            )
        if not self.rationale.startswith(INSUFFICIENT_DATA_PREFIX):
            raise ValueError(
                f"line {self.component!r} is grounded=False; its rationale must start "
                f"with {INSUFFICIENT_DATA_PREFIX!r}"
            )
        return self


class Estimate(BaseModel):
    summary: str
    lines: list[EstimateLineItem]

    @property
    def grounded_hours(self) -> float:
        """Total is computed here, never asked from the model: LLMs are bad at
        summing, and ungrounded lines contribute 0 by construction."""
        return sum(line.hours for line in self.lines if line.grounded)

    @property
    def has_sufficient_context(self) -> bool:
        return any(line.grounded for line in self.lines)


class LineStatus(str, Enum):
    GROUNDED = "grounded"
    DANGLING_CITATION = "dangling_citation"
    INSUFFICIENT_DATA = "insufficient_data"


class LineVerdict(BaseModel):
    component: str
    status: LineStatus
    dangling_chunk_ids: list[str]


class CitationReport(BaseModel):
    verdicts: list[LineVerdict]

    def _count(self, status: LineStatus) -> int:
        return sum(1 for verdict in self.verdicts if verdict.status is status)

    @property
    def grounded_count(self) -> int:
        return self._count(LineStatus.GROUNDED)

    @property
    def dangling_count(self) -> int:
        return self._count(LineStatus.DANGLING_CITATION)

    @property
    def insufficient_count(self) -> int:
        return self._count(LineStatus.INSUFFICIENT_DATA)

    @property
    def has_dangling_citations(self) -> bool:
        return self.dangling_count > 0


class GroundedEstimateRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, gt=0)
    search_mode: Literal["vector", "hybrid"] = "hybrid"
    rerank: bool = False


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    document_id: str
    content: str


class GroundedEstimateResponse(BaseModel):
    request_id: str
    estimate: Estimate
    citation_report: CitationReport
    # What the model asserts (sum of its grounded lines) vs. what survives
    # verification (lines whose citations all exist in the retrieved context).
    claimed_hours: float
    verified_hours: float
    corrected_after_dangling: bool
    retrieved_chunks: list[RetrievedChunkOut]
